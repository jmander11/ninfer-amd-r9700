#!/usr/bin/env python3
"""Prepare, but never execute, the terminal winner's C1 P2048 hardware trace."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Sequence

from tools.bench.prepare_whole_profile import _write_plan, file_sha256
from tools.bench.verify_selected_hardware_use import selected_route
from tools.ppl.pareto import load_payload, validate_terminal_production_authority


HYBRID_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
CAPTURE_LIBRARY = (Path(__file__).resolve().parents[2]
                   / "build-r9700/src/libninfer_r9700_hip_code_object_capture.so")


def _attach_hybrid_capture(out_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    capture = {
        "path": str(CAPTURE_LIBRARY.resolve(strict=True)),
        "file_size_bytes": CAPTURE_LIBRARY.stat().st_size,
        "sha256": file_sha256(CAPTURE_LIBRARY),
    }
    capture_dir = out_dir.resolve() / "code-objects"
    old_command = plan.get("profiler_command")
    if not isinstance(old_command, list) or not all(isinstance(item, str) for item in old_command):
        raise ValueError("generated trace plan lacks its profiler command")
    new_command = ["/usr/bin/env", f"LD_PRELOAD={capture['path']}",
                   f"NINFER_CODE_OBJECT_CAPTURE_DIR={capture_dir}", *old_command]
    plan["profiler_command"] = new_command
    plan["code_object_capture"] = {
        "library": capture, "directory": str(capture_dir),
        "scope": "selected hybrid workload loaded code objects",
    }
    plan_path = out_dir / "plan.json"
    command_path = out_dir / "commands.sh"
    old_line, new_line = shlex.join(old_command), shlex.join(new_command)
    command_text = command_path.read_text(encoding="utf-8")
    if command_text.count(old_line) != 1:
        raise ValueError("generated command does not contain one exact profiler invocation")
    quoted_capture = shlex.quote(str(capture_dir))
    guard = f"test ! -e {quoted_capture} && test ! -L {quoted_capture}\n"
    header = "#!/usr/bin/env bash\nset -euo pipefail\n\n"
    if not command_text.startswith(header):
        raise ValueError("generated command lacks its exact fail-closed header")
    command_text = header + guard + command_text[len(header):]
    command_path.write_text(command_text.replace(old_line, new_line), encoding="utf-8")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def _snapshot_matches(value: object, path: Path, label: str) -> None:
    if not isinstance(value, dict) or Path(str(value.get("path", ""))).resolve() != path \
            or value.get("sha256") != file_sha256(path):
        raise ValueError(f"{label} differs from terminal selection provenance")


def prepare(selection_path: Path, out_dir: Path) -> dict[str, Any]:
    selection_path = selection_path.resolve(strict=True)
    selection_bytes = selection_path.read_bytes()
    selection = load_payload(selection_bytes.decode("utf-8"))
    terminal, _ = validate_terminal_production_authority(selection)
    route = selected_route(selection_path)
    sources = [row for row in selection.get("source_provenance", [])
               if isinstance(row, dict) and row.get("candidate") == route["winner"]]
    if len(sources) != 1:
        raise ValueError("terminal winner lacks unique source provenance")
    matrices = sources[0].get("matrices")
    whole = matrices.get("pareto-whole") if isinstance(matrices, dict) else None
    if not isinstance(whole, dict) or not isinstance(whole.get("path"), str):
        raise ValueError("terminal winner lacks a pareto-whole manifest authority")
    manifest_path = Path(whole["path"]).resolve(strict=True)
    _snapshot_matches(whole, manifest_path, "pareto-whole manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("pareto-whole manifest root is not an object")
    artifact, bench = manifest.get("artifact"), manifest.get("bench")
    if (not isinstance(artifact, dict) or not isinstance(bench, dict)
            or artifact.get("sha256") != route["artifact"]["sha256"]
            or artifact.get("weights_id") != route["weights_id"]
            or bench.get("sha256") != route["executable"]["sha256"]):
        raise ValueError("pareto-whole identities differ from terminal winner")
    if (manifest.get("expected_kv_value_group") != route["kv_value_group"]
            or manifest.get("expected_xattention_profile") != route["xattention_profile"]
            or manifest.get("selected_prefill_chunk") != route["prefill_chunk"]):
        raise ValueError("pareto-whole execution profile differs from terminal winner")
    concurrencies = manifest.get("concurrency")
    if concurrencies != [1, 2, 3, 4]:
        raise ValueError("pareto-whole manifest is not the exact C1..4 product matrix")
    corpus = Path(str(manifest.get("corpus", ""))).resolve(strict=True)
    if (manifest.get("corpus_sha256") != file_sha256(corpus)
            or manifest.get("corpus_tokens", 0) < 2048):
        raise ValueError("pareto-whole corpus is absent or too short")
    power = manifest.get("power_profile")
    power_name = power.get("sysfs_path") if isinstance(power, dict) else None
    if not isinstance(power_name, str) or not power_name or power.get("required") != "auto":
        raise ValueError("pareto-whole manifest lacks the exact power-profile path")
    power_path = Path(power_name)
    command = [
        route["executable"]["path"], "--weights", route["artifact"]["path"],
        "--corpus", str(corpus), "-p", "2048", "-r", "1", "--warmup", "1",
        "--prefill-chunk", str(route["prefill_chunk"]), "--concurrency", "1",
        "--spec", "mtp", "--draft-tokens", "0", "--kv-capacity", "workload",
        "--output", "json", "--output-file", str(out_dir.resolve() / "benchmark-report.json"),
        "--profile-measured",
    ]
    if file_sha256(selection_path) != route["terminal_selection"]["sha256"]:
        raise ValueError("terminal selection changed during preparation")
    plan = _write_plan(
        out_dir.resolve(), kind="trace",
        question="Which exact production symbols execute for the terminal C1 P2048 route?",
        benchmark_command=command, power_profile_path=power_path,
        source_matrix={"path": str(manifest_path), "sha256": file_sha256(manifest_path),
                       "preset": "pareto-whole"}, artifact=artifact, bench=bench,
        workload={"concurrency": 1, "prompt_tokens": 2048, "generated_tokens": 0,
                  "spec": "none", "draft_tokens": 0, "dflash_verify_width": 0,
                  "kv_value_group": route["kv_value_group"],
                  "xattention_profile": route["xattention_profile"],
                  "prefill_chunk": route["prefill_chunk"]}, kernel_include_regex=None,
        extra_provenance={"terminal_selection": {
            **route["terminal_selection"], "winner": terminal["winner"],
            "winner_artifact": terminal["winner_artifact"],
            "winner_cache_profile": terminal["winner_cache_profile"],
            "winner_execution_profile": terminal["winner_execution_profile"],
            "selected_prefill_chunk": selection["selected_prefill_chunk"],
        }, "status": "command_only_not_executed_selected_hardware_trace"},
    )
    if route["weights_id"] == HYBRID_ID:
        return _attach_hybrid_capture(out_dir, plan)
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        prepare(args.selection, args.out)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"prepared selected hardware trace at {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
