#!/usr/bin/env python3
"""Prepare one terminal-winner C1/8K MTP3 shortlist-head execution trace."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
from pathlib import Path

from tools.bench.prepare_whole_profile import prepare as prepare_whole_profile
from tools.bench.produce_mtp_shortlist_head_evidence import (
    CODE_SYMBOL,
    HEAD_FORMAT,
    HEAD_SHAPE,
)
from tools.ppl.pareto import load_payload, validate_terminal_production_authority

REPO = Path(__file__).resolve().parents[2]
ROUTE_RESOLVER = REPO / "profiles/bench/post-terminal-focused-verification-20260905/resolve.py"
FINALIZER = REPO / "tools/bench/finalize_selected_mtp_shortlist_head.py"
PRODUCER = REPO / "tools/bench/produce_mtp_shortlist_head_evidence.py"
EXTRACTOR = REPO / "tools/bench/extract_embedded_code_object.py"
OUTPUT_STEM = "selected-mtp-shortlist-head"


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def resolve_route(selection: Path) -> dict:
    spec = importlib.util.spec_from_file_location("ninfer_terminal_route", ROUTE_RESOLVER)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load terminal route resolver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve(selection)


def prepare(selection: Path, output: Path) -> dict:
    selection = selection.resolve(strict=True)
    # Keep the caller-owned namespace lexical until the no-overwrite check.
    # Resolving a dangling symlink would instead create its missing target.
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise ValueError(f"output directory already exists: {output}")
    terminal, _ = validate_terminal_production_authority(
        load_payload(selection.read_text(encoding="utf-8"))
    )
    if (
        terminal.get("production_status")
        != "selected_route_pending_shortlist_head_trace_and_niah"
        or terminal.get("shortlist_head_precision_status")
        != "q4_round_gate_pass_pending_trace"
    ):
        raise ValueError("terminal selection does not authorize the Q4 shortlist-head trace")
    route = resolve_route(selection)
    matrix_manifest = Path(route["source_matrices"]["pareto-whole"]["path"]).resolve(strict=True)
    matrix_dir = matrix_manifest.parent
    artifact = route["artifact"]
    cache = route["cache_profile"]
    execution = route["execution_profile"]
    chunk = route["selected_prefill_chunk"]

    plan = prepare_whole_profile(
        matrix_dir, output, concurrency=1, prompt_tokens=8192, generated_tokens=256,
        kind="trace",
        question="attribute the executed optimized Q4 MTP shortlist head to its exact gfx1201 IU4 specialization",
        expected_weights_id=artifact["weights_id"],
        expected_kv_value_group=cache["value_group"],
        expected_xattention_profile=execution["xattention_profile"],
        expected_prefill_chunk=chunk,
    )
    if (
        plan["workload"].get("spec") != "mtp"
        or plan["workload"].get("draft_tokens") != 3
        or plan["workload"].get("dflash_verify_width") != 0
        or plan["benchmark_command"].count("--lm-head-draft") != 1
        or plan["benchmark_executable"] != route["benchmark"]
        or plan["artifact"].get("sha256") != artifact["sha256"]
    ):
        raise ValueError("selected whole matrix does not provide an exact optimized MTP3 row")

    plan_path = output / "plan.json"
    commands_path = output / "commands.sh"
    plan["terminal_selection"] = {"path": str(selection), "sha256": sha(selection)}
    plan["head_quantization"] = {
        "weight": {
            "object": "text/draft_head", "shape": list(HEAD_SHAPE),
            "format": HEAD_FORMAT, "codes": "packed signed int4",
            "group_size": 64, "scale": "FP16 per output-row/G64 group",
        },
        "activation": {
            "source": "BF16", "codec": "signed A8G64",
            "storage": "lossless packed unsigned-low and signed-high int4 planes plus FP16 per-token/G64 scale",
            "reconstruction": "a8 = low + 16 * high",
        },
        "native_path": {
            "code_symbol": CODE_SYMBOL,
            "opcode": "v_wmma_i32_16x16x32_iu4", "opcode_count": 4,
            "accumulation": "I32 low/high dots; FP32 scale/FMA accumulation; BF16 output",
        },
    }
    data_dir = Path(plan["profiler_command"][plan["profiler_command"].index("-d") + 1])
    plan["profiler_command"][plan["profiler_command"].index("-d") + 2:plan["profiler_command"].index("-d") + 2] = ["-o", OUTPUT_STEM]
    plan["expected_trace_database"] = str(data_dir / f"{OUTPUT_STEM}_results.db")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    original = commands_path.read_text(encoding="utf-8")
    benchmark_report = Path(
        plan["benchmark_command"][plan["benchmark_command"].index("--output-file") + 1]
    )
    for path in (
        Path(plan["required_power_profile"]["before_evidence"]),
        Path(plan["required_power_profile"]["after_evidence"]),
    ):
        old = f"test ! -e {shlex.quote(str(path))}"
        original = original.replace(old, old + f" && test ! -L {shlex.quote(str(path))}")
    namespace_guards = "\n".join(
        f"test ! -e {shlex.quote(str(path))} && test ! -L {shlex.quote(str(path))}"
        for path in (data_dir, benchmark_report)
    ) + "\n"
    original = original.replace("set +e\n", namespace_guards + "set +e\n", 1)
    profiler_lines = [line for line in original.splitlines() if line.startswith("/opt/rocm/")]
    if len(profiler_lines) != 1:
        raise ValueError("generated whole-profile command is ambiguous")
    replacement = shlex.join(plan["profiler_command"])
    original = original.replace(profiler_lines[0], replacement)
    postprocess = output / "postprocess.sh"
    postprocess.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        f"readonly root={shlex.quote(str(output))}\n"
        'sha256sum --check --strict "$root/prepared.sha256"\n'
        "python3 -m tools.bench.finalize_selected_mtp_shortlist_head --package \"$root\"\n",
        encoding="utf-8",
    )
    commands_path.write_text(
        original.replace("set -euo pipefail\n", "set -euo pipefail\n\nsha256sum --check --strict "
                         + shlex.quote(str(output / "prepared.sha256")) + "\n", 1)
        + f"bash {shlex.quote(str(postprocess))}\n",
        encoding="utf-8",
    )
    closure = [
        Path(__file__).resolve(), FINALIZER, PRODUCER, EXTRACTOR, ROUTE_RESOLVER,
        selection, matrix_manifest,
        Path(plan["source_matrix"]["report"]["path"]),
        Path(plan["artifact"]["path"]), Path(plan["benchmark_executable"]["path"]),
        plan_path, commands_path, postprocess,
    ]
    if route.get("hybrid_width_tool"):
        closure.append(Path(route["hybrid_width_tool"]["path"]))
    (output / "prepared.sha256").write_text(
        "".join(f"{sha(path)}  {path.relative_to(REPO)}\n" for path in closure),
        encoding="utf-8",
    )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        prepare(args.selection, args.out)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
