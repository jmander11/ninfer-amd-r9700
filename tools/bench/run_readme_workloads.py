#!/usr/bin/env python3
"""Measure workload-dependent full-batch decode throughput through the real HTTP server.

Seed each lane's prompt before submitting a concurrent generation wave. Only complete
full-batch decode-only throughput intervals count; there is no prefill-inclusive fallback.
Run under the shared GPU flock. Raw requests, responses, logs and reports go under --output.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import http.client
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.bench import run_serve_concurrency as concurrency_bench  # noqa: E402
from tools.bench import run_serve_corpus as corpus  # noqa: E402


def post(port: int, payload: dict) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    try:
        return corpus.post_json(connection, payload)
    finally:
        connection.close()


def prepare_workloads(manifest: Path, tokenizer_path: Path, concurrency: int) -> list[dict]:
    from tools.bench.make_bench_corpus import load_tokenizer

    definitions = json.loads(manifest.read_text())
    fixtures = corpus.load_fixtures()
    tokenizer = load_tokenizer(tokenizer_path)
    document = "\n\n".join((ROOT / path).read_text() for path in definitions["document_sources"])
    document_ids = tokenizer.encode(document, add_special_tokens=False)
    workloads = []
    for item in definitions["workloads"]:
        lanes = []
        for lane in range(concurrency):
            system = {
                "role": "system",
                "content": f"Benchmark lane {lane}. Follow the user request.",
            }
            if "fixture" in item:
                fixture = fixtures[item["fixture"]]
                messages = [system, *copy.deepcopy(fixture.messages)]
                thinking = fixture.thinking
            else:
                thinking = False
                messages = [
                    system,
                    {"role": "user", "content": definitions["document_instruction"]},
                ]
                # Count the actual rendered template; keep complete user/assistant boundaries.
                target = item["document_tokens"]
                count = target - len(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_dict=False,
                        add_generation_prompt=True,
                        enable_thinking=thinking,
                    )
                )
                for _ in range(8):
                    ids = (document_ids * (count // len(document_ids) + 1))[:count]
                    messages[1]["content"] = definitions["document_instruction"] + tokenizer.decode(
                        ids
                    )
                    actual = len(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            return_dict=False,
                            add_generation_prompt=True,
                            enable_thinking=thinking,
                        )
                    )
                    if abs(actual - target) <= 8:
                        break
                    count += target - actual
                if abs(actual - target) > 8:
                    raise ValueError(
                        f"could not construct a prompt near {target} tokens (got {actual})"
                    )
            if item.get("instruction"):
                messages[-1]["content"] += "\n\n" + item["instruction"]
            lanes.append({"messages": messages, "enable_thinking": thinking})
        workloads.append({"name": item["name"], "definition": item, "lanes": lanes})
    return workloads


def read_events(path: Path, offset: int) -> list[dict]:
    with path.open() as stream:
        stream.seek(offset)
        return [json.loads(line) for line in stream if line.strip()]


def run_wave(port: int, log: Path, workload: dict, output_tokens: int, seed: int) -> dict:
    payloads = [
        dict(
            model=corpus.PUBLIC_MODEL_ID,
            max_completion_tokens=output_tokens,
            seed=seed + lane,
            temperature=1.5,
            **prompt,
        )
        for lane, prompt in enumerate(workload["lanes"])
    ]
    # Per-lane prefixes differ near the start, so each request retains its own state.
    for payload in payloads:
        post(port, dict(payload, max_completion_tokens=1))
    time.sleep(0.2)  # let the periodic log close the final seed/prefill interval
    offset = log.stat().st_size
    barrier = threading.Barrier(len(payloads))

    def request(payload: dict) -> dict:
        barrier.wait()
        return post(port, payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        responses = list(pool.map(request, payloads))
    time.sleep(0.2)
    events = read_events(log, offset)
    done = [event for event in events if event.get("event") == "request_done"]
    if len(done) != len(payloads):
        raise RuntimeError(f"expected {len(payloads)} request completions, got {len(done)}")
    errors = []
    for lane, response in enumerate(responses):
        if workload["definition"].get("forbid_tool_calls") and "<tool_call>" in response["choices"][
            0
        ]["message"].get("content", ""):
            errors.append(f"lane {lane}: code workload requested a tool instead of writing code")
        if response["usage"]["prompt_tokens_details"]["cached_tokens"] <= 0:
            errors.append(f"lane {lane}: prompt seeding did not retain a reusable prefix")
    steady = concurrency_bench.steady_metrics(
        [event for event in events if event.get("event") == "throughput"], len(payloads)
    )
    drafted = sum(event["speculative"]["drafted_tokens"] for event in done)
    accepted = sum(event["speculative"]["accepted_tokens"] for event in done)
    return {
        "requests": payloads,
        "responses": responses,
        "request_done": done,
        "steady": steady,
        "acceptance": accepted / drafted if drafted else 0,
        "prompt_tokens": [response["usage"]["prompt_tokens"] for response in responses],
        "completion_tokens": [response["usage"]["completion_tokens"] for response in responses],
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--serve", type=Path, default=ROOT / "build-r9700/apps/ninfer-serve")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "tools/bench/fixtures/readme_workloads.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--spec", choices=["dflash", "mtp"], default="dflash")
    parser.add_argument("--draft-tokens", type=int, help="adaptive maximum (DFlash: 7; MTP: 3)")
    parser.add_argument(
        "--workload", action="append", help="workload name to run; repeat to select several"
    )
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()
    maximum_concurrency = 12 if args.spec == "dflash" else 16
    maximum_drafts = 7 if args.spec == "dflash" else 5
    if args.draft_tokens is None:
        args.draft_tokens = 7 if args.spec == "dflash" else 3
    if not 1 <= args.draft_tokens <= maximum_drafts:
        parser.error(f"{args.spec} requires --draft-tokens in [1,{maximum_drafts}]")
    if (
        any(not 1 <= c <= maximum_concurrency for c in args.concurrency)
        or args.tokens < 128
        or args.repetitions < 1
        or args.warmup < 0
    ):
        parser.error(
            f"require C=1..{maximum_concurrency}, at least 128 output tokens, "
            "positive repetitions and nonnegative warmup"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "command": sys.argv,
        "candidate": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "host": platform.platform(),
        "artifact": str(args.artifact.resolve()),
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("HIP_", "HSA_", "DEBUG_HIP_", "ROC_"))
        },
        "manifest": json.loads(args.manifest.read_text()),
        "settings": vars(args)
        | {"method": "full-batch decode-only periodic wall intervals; seeded prompts; p-less T1.5"},
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, default=str, indent=2) + "\n"
    )
    rows = []
    for c in args.concurrency:
        workloads = prepare_workloads(args.manifest, args.tokenizer, c)
        if args.workload:
            known = {workload["name"] for workload in workloads}
            if not set(args.workload) <= known:
                parser.error("unknown workload name")
            workloads = [workload for workload in workloads if workload["name"] in args.workload]
        log = args.output / f"server-c{c}.jsonl"
        command = [
            str(args.serve.resolve()),
            str(args.artifact.resolve()),
            "--port",
            str(args.port),
            "--max-context",
            "65536",
            "--max-concurrency",
            str(c),
            "--kv-capacity",
            "auto",
            "--spec",
            args.spec,
            "--draft-tokens",
            str(args.draft_tokens),
            "--adaptive-draft",
            "--lm-head-draft",
            "--prefill-chunk",
            "2048",
            "--log-stats-interval-ms",
            "100",
            "--request-log-jsonl",
            str(log.resolve()),
        ]
        (args.output / f"command-c{c}.json").write_text(json.dumps(command, indent=2) + "\n")
        with corpus.RunningServer(command, "127.0.0.1", args.port, log) as server:
            startup = server.wait_until_ready()
            (args.output / f"startup-c{c}.json").write_text(json.dumps(startup, indent=2) + "\n")
            for index, workload in enumerate(workloads):
                raw = []
                path = args.output / f"c{c}-w{index}.json"
                for rep in range(args.warmup + args.repetitions):
                    result = run_wave(
                        args.port, log, workload, args.tokens, 7632647173703958409 + rep * 100
                    )
                    result["warmup"] = rep < args.warmup
                    raw.append(result)
                    path.write_text(
                        json.dumps({"workload": workload, "runs": raw}, indent=2) + "\n"
                    )
                    if result["errors"]:
                        raise RuntimeError(f"{path}: {'; '.join(result['errors'])}")
                    print(
                        f"C{c} {workload['name']} rep{rep}: {result['steady']['decode_tokens_per_second']:.1f} tok/s",
                        flush=True,
                    )
                measured = raw[args.warmup :]
                rates = [run["steady"]["decode_tokens_per_second"] for run in measured]
                row = {
                    "workload": workload["name"],
                    "concurrency": c,
                    "mean_tok_s": statistics.mean(rates),
                    "stdev_tok_s": statistics.stdev(rates) if len(rates) > 1 else 0,
                    "prompt_tokens": sorted({n for run in measured for n in run["prompt_tokens"]}),
                    "completion_tokens": [n for run in measured for n in run["completion_tokens"]],
                    "acceptance": statistics.mean(run["acceptance"] for run in measured),
                }
                rows.append(row)
                (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    headers = "| Workload | Prompt tokens | C | Aggregate decode tokens/s |\n|---|---:|---:|---:|\n"
    table = (
        headers
        + "\n".join(
            f"| {row['workload']} | {min(row['prompt_tokens']):,} | {row['concurrency']} | {row['mean_tok_s']:.1f} |"
            for row in rows
        )
        + "\n"
    )
    (args.output / "summary.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
