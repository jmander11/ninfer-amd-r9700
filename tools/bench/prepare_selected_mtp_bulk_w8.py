#!/usr/bin/env python3
"""Prepare the conditional C1 selected-chunk mixed MTP-bulk W8 gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESOLVER = REPO / "profiles/bench/post-terminal-focused-verification-20260905/resolve.py"
FINALIZER = REPO / "tools/bench/finalize_selected_mtp_bulk_w8.py"
EXTRACTOR = REPO / "tools/bench/extract_embedded_code_object.py"
TRACE_PARSER = REPO / "tools/bench/validate_profile_trace.py"
QUALIFIER_SOURCE = REPO / "tools/r9700/w8a8_wmma_linear_qual.hip"
KERNEL_SOURCE = REPO / "src/ops/r9700/linear/r9700_linear.hip"
PROFILE_SOURCE = REPO / "src/ops/r9700/linear/r9700_w8_activation_profile.h"
QUALIFIER = REPO / "tools/r9700/build/w8a8_wmma_linear_qual"
ASSEMBLY = REPO / "tools/r9700/build/w8a8_wmma_linear_qual.s"
POWER = Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
ROCPROF = Path("/opt/rocm/core-10.0/bin/rocprofv3")
OBJDUMP = Path("/opt/rocm/llvm/bin/llvm-objdump")
READELF = Path("/opt/rocm/llvm/bin/llvm-readelf")
MIXED_ID = "r9700-q4-w8-mse-n16k16-eval"


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def resolve(selection: Path) -> dict:
    spec = importlib.util.spec_from_file_location("ninfer_terminal_route", RESOLVER)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load terminal route resolver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve(selection)


def atomic_write(path: Path, contents: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(contents); output.flush(); os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare(selection: Path, operator_report: Path, output: Path) -> dict:
    selection = selection.resolve(strict=True)
    route = resolve(selection)
    if route["artifact"].get("weights_id") != MIXED_ID:
        raise ValueError("terminal winner is not mixed; MTP-bulk W8 proof is inapplicable")
    chunk = route["selected_prefill_chunk"]
    if chunk not in (1024, 2048, 4096, 8192):
        raise ValueError("selected prefill chunk is unsupported by the W8 qualifier")
    planner = (Path(route["build_directory"]) / "src/ninfer_r9700_runtime_planner_qual").resolve(strict=True)
    whole_manifest = Path(route["source_matrices"]["pareto-whole"]["path"])
    whole = json.loads(whole_manifest.read_text(encoding="utf-8"))
    corpus = Path(whole.get("corpus", "")).resolve(strict=True)
    if whole.get("corpus_sha256") != sha(corpus):
        raise ValueError("selected whole matrix corpus bytes changed")
    operator_report = operator_report.resolve(strict=True)
    operator_value = json.loads(operator_report.read_text(encoding="utf-8"))
    operator_executable = operator_value.get("executable", {})
    operator_sources = operator_value.get("sources")
    source_by_role = {row.get("role"): row for row in operator_sources
                      if isinstance(row, dict)} if isinstance(operator_sources, list) else {}
    if (operator_value.get("artifact_type") != "ninfer_r9700_mtp_bulk_w8_operator_qualification"
            or operator_value.get("schema_version") != 1
            or operator_value.get("status") != "passed"
            or operator_value.get("tokens") != chunk
            or operator_value.get("concurrency") != 1
            or Path(str(operator_executable.get("path", ""))).resolve() != QUALIFIER.resolve()
            or operator_executable.get("sha256") != sha(QUALIFIER)
            or not isinstance(operator_sources, list) or len(operator_sources) != 4
            or set(source_by_role) != {"qualifier", "contract_header", "kernel",
                                       "dispatch_profile"}
            or source_by_role["dispatch_profile"].get("sha256") == sha(PROFILE_SOURCE)):
        raise ValueError("operator report is not a retained pre-admission selected-chunk gate")
    required = [RESOLVER, FINALIZER, EXTRACTOR, TRACE_PARSER,
                QUALIFIER_SOURCE, KERNEL_SOURCE, PROFILE_SOURCE,
                QUALIFIER, operator_report, planner, selection,
                Path(route["artifact"]["path"]), Path(route["benchmark"]["path"]),
                whole_manifest, corpus, ROCPROF, OBJDUMP, READELF]
    for path in required:
        if not path.is_file():
            raise ValueError(f"required bound input is not a regular file: {path}")
    if os.path.lexists(output):
        raise ValueError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    raw = operator_report
    evidence = output / "evidence.json"
    trace_root = output / "trace"
    trace_db = trace_root / "selected-mtp-bulk-w8_results.db"
    benchmark_report = output / "benchmark-report.json"
    power_before = output / "power-before.txt"
    power_after = output / "power-after.txt"
    code_object = output / "selected-production-code-object.elf"
    production_assembly = output / "selected-production-code-object.s"
    production_metadata = output / "selected-production-code-object.notes"
    benchmark_command = [
        route["benchmark"]["path"], "--weights", route["artifact"]["path"],
        "--corpus", str(corpus), "--device", "0", "--concurrency", "1",
        "-p", str(chunk), "--prefill-chunk", str(chunk), "--spec", "mtp",
        "--draft-tokens", "3", "--lm-head-draft", "--output", "json",
        "--output-file", str(benchmark_report), "-r", "1", "--warmup", "1",
        "--profile-measured",
    ]
    plan = {
        "artifact_type": "ninfer_r9700_selected_mtp_bulk_w8_plan", "schema_version": 1,
        "status": "prepared_post_admission_trace", "conditional_recipe": MIXED_ID,
        "selected_route": route, "concurrency": 1, "selected_prefill_chunk": chunk,
        "shapes": [
            {"rows": 5120, "columns": 10240, "dispatches_per_full_chunk": 1},
            {"rows": 1024, "columns": 5120, "dispatches_per_full_chunk": 2},
        ],
        "gates": {"oracle_max_bf16_steps": 2,
                  "candidate_faster_than_one_wave_incumbent": True,
                  "native_opcode": "v_wmma_i32_16x16x16_iu8",
                  "native_signedness": "neg_lo:[1,1,0] at both sites",
                  "selected_engine_dispatches": 3, "power_profile": "auto"},
        "benchmark_command": benchmark_command,
        "corpus": {"path": str(corpus), "sha256": sha(corpus)},
        "outputs": {"operator": str(raw), "evidence": str(evidence),
                    "trace_database": str(trace_db),
                    "benchmark_report": str(benchmark_report),
                    "power_before": str(power_before), "power_after": str(power_after),
                    "production_code_object": str(code_object),
                    "production_assembly": str(production_assembly),
                    "production_metadata": str(production_metadata)},
        "note": ("This package is created only after a separately retained direct operator gate "
                 "passes, the two exact predicates are admitted, and affected whole rows plus "
                 "schema-v7 selection are regenerated against the rebuilt executable. It then "
                 "requires that exact selected Engine to execute one-plus-two CTA dispatches."),
    }
    plan_path = output / "plan.json"
    atomic_write(plan_path, json.dumps(plan, indent=2) + "\n")
    finalize = shlex.join(["python3", "-m", "tools.bench.finalize_selected_mtp_bulk_w8",
                           "--selection", str(selection),
                           "--plan", str(plan_path), "--raw", str(raw),
                           "--assembly", str(production_assembly),
                           "--metadata", str(production_metadata),
                           "--code-object", str(code_object),
                           "--benchmark-report", str(benchmark_report),
                           "--trace-database", str(trace_db),
                           "--power-before", str(power_before),
                           "--power-after", str(power_after),
                           "--qualifier", str(QUALIFIER), "--planner", str(planner),
                           "--out", str(evidence)])
    commands = output / "commands.sh"
    atomic_write(commands,
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        f"readonly package={shlex.quote(str(output.resolve()))}\n"
        f"readonly power={shlex.quote(str(POWER))}\n"
        'sha256sum --check --strict "$package/prepared.sha256"\n'
        'for path in "$package/evidence.json" "$package/trace" '
        '"$package/benchmark-report.json" "$package/power-before.txt" '
        '"$package/power-after.txt" "$package/selected-production-code-object.elf" '
        '"$package/selected-production-code-object.s" '
        '"$package/selected-production-code-object.notes"; do '
        'test ! -e "$path" && test ! -L "$path"; done\n'
        'test "$(<"$power")" = auto\n'
        'printf "%s\\n" auto >"$package/power-before.txt"\n'
        f"{shlex.join([str(ROCPROF), '--selected-regions', '-f', 'rocpd', '-d', str(trace_root), '-o', 'selected-mtp-bulk-w8', '--marker-trace', '--kernel-trace', '--', *benchmark_command])}\n"
        'test "$(<"$power")" = auto\n'
        'printf "%s\\n" auto >"$package/power-after.txt"\n'
        f"{shlex.join(['python3', str(EXTRACTOR), '--executable', route['benchmark']['path'], '--out', str(code_object), '--code-symbol', 'a8w8g32_linear_prefill_cta_kernel'])}\n"
        f"{shlex.join([str(OBJDUMP), '-d', '--mcpu=gfx1201', str(code_object)])} >{shlex.quote(str(production_assembly))}\n"
        f"{shlex.join([str(READELF), '--notes', str(code_object)])} >{shlex.quote(str(production_metadata))}\n"
        f"{finalize}\n")
    closure = [Path(__file__).resolve(), *required, plan_path, commands]
    atomic_write(output / "prepared.sha256", "".join(
        f"{sha(path)}  {path.resolve()}\n" for path in closure))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--operator-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        prepare(args.selection, args.operator_report, args.out)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
