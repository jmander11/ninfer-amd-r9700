#!/usr/bin/env python3
"""One isolated C8 Chat diagnostic cell; never starts a server or executes tools.

Exactly eight preselected private JSONL records are submitted together. Compare
separate processes/builds using identical inputs, seed, horizon and startup settings.
Schema-complete calls remain candidates for private context grading, not task success.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import socket
import threading
import time
import urllib.parse
from contextlib import suppress
from pathlib import Path

from reasoning_limit_http_qual import sse
from reasoning_limit_qual import intervention, private_path, score

ACTIVE = (
    "ninfer_scheduler_running_requests",
    "ninfer_scheduler_prefilling_requests",
    "ninfer_scheduler_decode_ready_requests",
    "ninfer_scheduler_waiting_requests",
    "ninfer_kv_growth_blocked_requests",
    "ninfer_kv_suspended_requests",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def gauges(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            fields = line.split()
            if len(fields) >= 2 and "{" not in fields[0]:
                result[fields[0]] = float(fields[1])
    require(
        all(name in result and math.isfinite(result[name]) for name in ACTIVE),
        "required scheduler gauges missing or nonfinite",
    )
    return result


def metrics(host: str, port: int, timeout: float) -> tuple[str, dict]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        require(response.status == 200, "metrics endpoint failed")
        text = response.read().decode()
        return text, gauges(text)
    finally:
        connection.close()


def run_cell(
    host: str,
    port: int,
    cases: list,
    payloads: list,
    output: Path,
    arm: str,
    kv_mode: str,
    horizon: float,
    drain_seconds: float,
) -> dict:
    """The CLI restricts the endpoint; host tests may supply an ephemeral mock receiver."""
    require(len(cases) == len(payloads) == 8, "C8 requires exactly eight original records")
    import jsonschema  # Preload independent grading before the release barrier.

    del jsonschema
    wires = [
        json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() for payload in payloads
    ]
    for index, wire in enumerate(wires):
        (output / f"{index:02d}.request.json").write_bytes(wire)
    before_text, before = metrics(host, port, min(2.0, drain_seconds))
    (output / "metrics-before.txt").write_text(before_text)
    require(all(before[name] == 0 for name in ACTIVE), "isolated server is not idle")
    require(before.get("ninfer_scheduler_max_concurrency") == 8, "server concurrency is not eight")
    window = before.get("ninfer_kv_execution_window_tokens")
    require(
        window == 0 if kv_mode == "full" else window is not None and window > 0,
        "resolved KV mode does not match the declared diagnostic cell",
    )

    records = [
        {
            "id": case["id"],
            "index": index,
            "arm": arm,
            "kv_mode": kv_mode,
            "quality": "unverified; private context grading required",
            "task_success": "unverified; proposed tools never executed",
        }
        for index, case in enumerate(cases)
    ]
    connections = [None] * 8
    sockets = [None] * 8
    registry = threading.Lock()
    stop = threading.Event()
    all_done = threading.Event()
    done_count = 0
    start = []
    barrier = threading.Barrier(9, action=lambda: start.append(time.monotonic()))
    strategy = "retry" if arm == "retry4k" else "close"
    budget = None if arm == "unlimited" else 4096

    def worker(index: int) -> None:
        nonlocal done_count
        connection = http.client.HTTPConnection(host, port, timeout=min(horizon, 10.0))
        events = []
        raw_lines = []
        record = records[index]
        try:
            with registry:
                connections[index] = connection
            barrier.wait(timeout=15)
            deadline = start[0] + horizon
            if stop.is_set():
                return
            connection.connect()
            with registry:
                sockets[index] = connection.sock
            if stop.is_set():
                return
            connection.sock.settimeout(max(0.01, deadline - time.monotonic()))
            record["dispatch_seconds"] = time.monotonic() - start[0]
            connection.request(
                "POST", "/v1/chat/completions", wires[index], {"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            record["http_status"] = response.status
            if response.status != 200:
                record["failed"] = True
                record["failure_observed_seconds"] = time.monotonic() - start[0]
                raw = response.read()
                (output / f"{index:02d}.error-response.bin").write_bytes(raw)
                record["failed"] = True
                record["response_complete_seconds"] = time.monotonic() - start[0]
                try:
                    error = json.loads(raw)
                    record["policy"] = intervention(
                        {"usage": {"ninfer": error.get("error", {}).get("ninfer", {})}},
                        strategy,
                        budget,
                    )
                except (ValueError, TypeError, AttributeError):
                    pass
                return
            message = {"content": "", "reasoning_content": "", "tool_calls": []}
            calls = {}
            finish = None
            usage = {}
            done_marker = False

            def wire_lines():
                for wire in response:
                    raw_lines.append(wire)
                    yield wire

            for event, data in sse(wire_lines()):
                now = time.monotonic()
                events.append({"event": event, "data": data, "seconds": now - start[0]})
                if data is None:
                    done_marker = True
                    break
                if "error" in data:
                    record["stream_error"] = data["error"]
                    record["policy"] = intervention(
                        {"usage": {"ninfer": data["error"].get("ninfer", {})}}, strategy, budget
                    )
                    record["failed"] = True
                    record["failure_observed_seconds"] = now - start[0]
                    break
                if data.get("usage"):
                    usage = data["usage"]
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {})
                    if (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("tool_calls")
                    ):
                        record.setdefault("first_output_delta_seconds", now - start[0])
                    for channel in ("content", "reasoning_content"):
                        message[channel] += delta.get(channel) or ""
                    for part in delta.get("tool_calls", []):
                        call = calls.setdefault(
                            part["index"],
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        call["id"] += part.get("id") or ""
                        for key in ("name", "arguments"):
                            call["function"][key] += part.get("function", {}).get(key) or ""
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
                        message["tool_calls"] = [calls[key] for key in sorted(calls)]
                        assembled = {"choices": [{"finish_reason": finish, "message": message}]}
                        grading = score(assembled, payloads[index])
                        record["grading"] = grading
                        record.setdefault("terminal_choice_seconds", now - start[0])
                        if grading["complete_action"]:
                            record.setdefault("schema_valid_tool_batch_seconds", now - start[0])
                if stop.is_set() or time.monotonic() >= deadline:
                    break
            record["done_marker"] = done_marker
            if done_marker:
                record["response_complete_seconds"] = time.monotonic() - start[0]
            assembled = {"choices": [{"finish_reason": finish, "message": message}], "usage": usage}
            (output / f"{index:02d}.assembled-response.json").write_text(
                json.dumps(assembled, ensure_ascii=False)
            )
            record["usage"] = usage
            record.setdefault("policy", intervention(assembled, strategy, budget))
            if not done_marker and not stop.is_set():
                record["failed"] = True
                record["error"] = "stream ended without DONE"
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            record["failed"] = bool(record.get("failed")) or (
                not stop.is_set() and (not start or time.monotonic() < start[0] + horizon)
            )
        finally:
            connection.close()
            record["worker_end_seconds"] = time.monotonic() - start[0] if start else None
            try:
                if raw_lines:
                    (output / f"{index:02d}.response.sse").write_bytes(b"".join(raw_lines))
                if events:
                    (output / f"{index:02d}.events.json").write_text(
                        json.dumps(events, ensure_ascii=False)
                    )
            finally:
                with registry:
                    done_count += 1
                    if done_count == 8:
                        all_done.set()

    workers = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(8)]
    for thread in workers:
        thread.start()
    try:
        barrier.wait(timeout=15)
        all_done.wait(timeout=max(0, start[0] + horizon - time.monotonic()))
    except threading.BrokenBarrierError:
        barrier.abort()
    stopped_at = time.monotonic()
    stop.set()
    with registry:
        for sock in sockets:
            if sock is not None:
                with suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
        for connection in connections:
            if connection is not None:
                connection.close()
    join_deadline = time.monotonic() + drain_seconds
    for thread in workers:
        thread.join(timeout=max(0, join_deadline - time.monotonic()))
    unfinished = sum(thread.is_alive() for thread in workers)

    drain_deadline = join_deadline
    zero_samples = 0
    drained = False
    after = {}
    drain_error = None
    while time.monotonic() < drain_deadline:
        try:
            text, after = metrics(host, port, min(1.0, drain_deadline - time.monotonic()))
            (output / "metrics-after.txt").write_text(text)
            zero_samples = zero_samples + 1 if all(after[name] == 0 for name in ACTIVE) else 0
            if zero_samples >= 2:
                drained = True
                break
        except Exception as error:
            drain_error = f"{type(error).__name__}: {error}"
            zero_samples = 0
        time.sleep(min(0.05, max(0, drain_deadline - time.monotonic())))
    elapsed = min(horizon, stopped_at - start[0]) if start else 0.0
    for record in records:
        terminal = record.get("terminal_choice_seconds", record.get("response_complete_seconds"))
        record["failed_within_horizon"] = bool(record.get("failed")) and (
            record.get("failure_observed_seconds", record.get("worker_end_seconds")) is not None
            and record.get("failure_observed_seconds", record.get("worker_end_seconds")) <= horizon
        )
        record["right_censored"] = (terminal is None or terminal > horizon) and not record[
            "failed_within_horizon"
        ]
        record["schema_valid_within_horizon"] = (
            record.get("schema_valid_tool_batch_seconds", math.inf) <= horizon
        )
    valid = sum(record["schema_valid_within_horizon"] for record in records)
    summary = {
        "requests": 8,
        "arm": arm,
        "kv_mode": kv_mode,
        "streaming": True,
        "seed": payloads[0].get("seed"),
        "horizon_seconds": horizon,
        "observed_seconds": elapsed,
        "schema_valid_tool_batches": valid,
        "schema_valid_batches_per_second": valid / elapsed if elapsed else 0,
        "context_appropriate_actions": "unverified; private grading required",
        "answer_quality": "unverified",
        "right_censored": sum(r["right_censored"] for r in records),
        "failed_within_horizon": sum(r["failed_within_horizon"] for r in records),
        "dispatch_order": sorted(
            range(8), key=lambda i: records[i].get("dispatch_seconds", math.inf)
        ),
        "response_completion_order": sorted(
            (i for i in range(8) if "response_complete_seconds" in records[i]),
            key=lambda i: records[i]["response_complete_seconds"],
        ),
        "terminal_choice_order": sorted(
            (i for i in range(8) if "terminal_choice_seconds" in records[i]),
            key=lambda i: records[i]["terminal_choice_seconds"],
        ),
        "submission_order": "concurrent barrier release; observed client dispatch order recorded; server FIFO arrival not inferred",
        "policy_acknowledged": sum(bool(r.get("policy", {}).get("acknowledged")) for r in records),
        "policy_interventions": sum(bool(r.get("policy", {}).get("intervened")) for r in records),
        "completed_stream_policy_unacknowledged": sum(
            bool(r.get("done_marker")) and not r.get("policy", {}).get("acknowledged", False)
            for r in records
        ),
        "policy_strategy_mismatches": sum(
            r.get("policy", {}).get("strategy_matches") is False for r in records
        ),
        "client_workers_unjoined": unfinished,
        "engine_drained": drained,
        "drain_seconds": time.monotonic() - stopped_at,
        "drain_error": drain_error,
        "counter_deltas_include_cancellation_drain": True,
        "counter_deltas": {
            name: after[name] - value
            for name, value in before.items()
            if name.endswith("_total") and name in after
        },
    }
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    require(not unfinished and drained, "cancellation/drain incomplete; inspect private evidence")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("unlimited", "close4k", "retry4k"), required=True)
    parser.add_argument("--kv-mode", choices=("full", "windowed"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--horizon-seconds", type=float, default=300)
    parser.add_argument("--drain-seconds", type=float, default=30)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    parsed = urllib.parse.urlparse(args.base_url)
    require(
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and parsed.port == 18080
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment,
        "only isolated localhost18080 is supported",
    )
    require(args.execute, "explicit --execute is required; no server is started here")
    require(
        all(
            math.isfinite(value) and value > 0
            for value in (args.horizon_seconds, args.drain_seconds)
        ),
        "horizon and drain must be finite positive durations",
    )
    source = private_path(args.input_jsonl)
    output = private_path(args.output_dir)
    cases = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    require(
        len(cases) == 8 and len({case["id"] for case in cases}) == 8,
        "select exactly eight distinct original records",
    )
    payloads = []
    for case in cases:
        payload = dict(case["request"])
        require(
            payload.get("max_completion_tokens", payload.get("max_tokens")) == 32768,
            "preserve each original32768 completion ceiling",
        )
        require(
            isinstance(payload.get("preserve_thinking"), bool),
            "set the controlled history policy explicitly",
        )
        payload.update(model=args.model, stream=True, seed=args.seed)
        payload["stream_options"] = {"include_usage": True}
        # The unlimited arm must opt out even when the server has a default cap.
        payload["thinking_token_budget"] = None if args.arm == "unlimited" else 4096
        payloads.append(payload)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    summary = run_cell(
        parsed.hostname,
        parsed.port,
        cases,
        payloads,
        output,
        args.arm,
        args.kv_mode,
        args.horizon_seconds,
        args.drain_seconds,
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "requests",
                    "arm",
                    "kv_mode",
                    "observed_seconds",
                    "schema_valid_tool_batches",
                    "right_censored",
                    "engine_drained",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
