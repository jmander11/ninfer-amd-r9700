#!/usr/bin/env python3
"""Prepare a selected-route C1..4 ordinary-decode cache/wait PMC package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path

from tools.bench.run_ninfer_bench_matrix import (
    FP8_QK_WMMA_PROFILE,
    FP8_QK_WMMA_T1_MIN_CONTEXT,
    FP8_QK_WMMA_T2_MIN_CONTEXT,
    MATRIX_SCHEMA_VERSION,
    PRODUCT_CONCURRENCIES,
    REPORT_SCHEMA_VERSION,
    inspect_artifact,
    inspect_executable,
    require_fp8_hybrid_artifact,
    validate_hybrid_shared_workspace_authority,
)
from tools.ppl.pareto import load_payload, validate_terminal_production_authority

REPO = Path(__file__).resolve().parents[2]
ROCPROF = Path("/opt/rocm/core-10.0/bin/rocprofv3")
ROCPROF_AVAIL = Path("/opt/rocm/bin/rocprofv3-avail")
POWER = Path("/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level")
MARKER_SOURCE = REPO / "src/targets/qwen3/impl/runtime/program_impl.h"
PROFILE_BUILD_DIR = REPO / "build-r9700-selected-ordinary-decode-profile"
PROFILE_BUILD_PRODUCER = REPO / "tools/bench/build_selected_decode_profile.py"
PROFILE_ROUTE_RESOLVER = (
    REPO / "profiles/bench/post-terminal-focused-verification-20260905/resolve.py"
)
COUNTERS = (
    "SQ_WAIT_ANY", "SQ_WAIT_INST_ANY", "SQ_WAVE_CYCLES", "SQ_WAVES",
    "GL2C_HIT", "GL2C_MISS", "GL2C_MC_WRREQ_STALL", "GL2C_EA_RDREQ",
    "TCP_REQ", "TCP_REQ_MISS", "GRBM_GUI_ACTIVE",
)

PROFILE_BUILD_RECEIPT_TYPE = (
    "ninfer_r9700_selected_decode_instrumentation_build_receipt"
)


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def replace_unique_option(command: list[str], option: str, value: str) -> None:
    if command.count(option) != 1:
        raise ValueError(f"source benchmark command lacks one {option}")
    index = command.index(option)
    if index + 1 >= len(command):
        raise ValueError(f"source benchmark command lacks a value for {option}")
    command[index + 1] = value


def read_cmake_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        if key:
            values[key] = value
    return values


def profile_build_commands(definitions: dict[str, str]) -> dict[str, list[str]]:
    return {
        "configure": [
            "/usr/bin/cmake", "-S", str(REPO), "-B", str(PROFILE_BUILD_DIR),
            "-G", "Ninja", *(f"-D{key}={value}" for key, value in definitions.items()),
        ],
        "build": [
            "/usr/bin/cmake", "--build", str(PROFILE_BUILD_DIR),
            "--target", "ninfer_bench", "-j4",
        ],
    }


def validate_profile_build_receipt(
    receipt_path: Path, *, selection_path: Path, selection_sha256: str,
    manifest_path: Path, manifest_sha256: str, artifact: dict,
    terminal_executable: dict, marker_source: dict, selected_route: dict,
    compiled_route: dict, hybrid_workspace_authority: object,
) -> tuple[dict, Path, Path]:
    receipt_path = receipt_path.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    profile_executable = receipt.get("instrumentation_executable", {})
    profile_path = Path(str(profile_executable.get("path", "")))
    build_dir = Path(str(receipt.get("build_directory", "")))
    cache = receipt.get("cmake_cache", {})
    cache_path = Path(str(cache.get("path", "")))
    compile_database = receipt.get("compile_database", {})
    compile_database_path = Path(str(compile_database.get("path", "")))
    expected_definitions = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_HIP_ARCHITECTURES": "gfx1201",
        "NINFER_BUILD_BENCHMARKS": "ON",
        "NINFER_R9700_KV_VALUE_GROUP": str(selected_route["cache_profile"]["value_group"]),
        "NINFER_R9700_Q4_ACTIVATION_BITS": str(compiled_route["q4_activation_bits"]),
        "NINFER_R9700_Q4_PREFILL_PINGPONG_QUALIFICATION": "OFF",
        "NINFER_R9700_W8_ACTIVATION_BITS": str(compiled_route["w8_activation_bits"]),
        "NINFER_R9700_FP8_QK_WMMA": "1" if compiled_route["fp8_qk_wmma_enabled"] else "0",
        "NINFER_R9700_XATTENTION_QUALIFICATION": (
            "OFF" if selected_route["execution_profile"]["xattention_profile"] == "dense"
            else "ON"
        ),
        "NINFER_R9700_XATTENTION_STRIDE": "16",
        "NINFER_R9700_XATTENTION_TAU_PERMILLE": "900",
    }
    exact = {
        "terminal_selection": {"path": str(selection_path), "sha256": selection_sha256},
        "source_matrix": {"path": str(manifest_path), "sha256": manifest_sha256},
        "artifact": artifact,
        "terminal_timing_executable": terminal_executable,
        "ordinary_round_marker_source": marker_source,
        "selected_route": selected_route,
        "compiled_route": compiled_route,
        "hybrid_shared_workspace_authority": hybrid_workspace_authority,
    }
    if (
        receipt.get("artifact_type") != PROFILE_BUILD_RECEIPT_TYPE
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "passed"
        or receipt.get("purpose") != "profiler_attribution_only"
        or receipt.get("profile_timing_admissible") is not False
        or any(receipt.get(key) != value for key, value in exact.items())
        or receipt.get("compile_definitions") != expected_definitions
        or not build_dir.is_absolute() or not build_dir.is_dir() or build_dir.is_symlink()
        or build_dir.resolve() != PROFILE_BUILD_DIR.resolve()
        or not cache_path.is_absolute() or not cache_path.is_file() or cache_path.is_symlink()
        or cache_path.resolve().parent != build_dir.resolve()
        or cache.get("file_size_bytes") != cache_path.stat().st_size
        or cache.get("sha256") != sha(cache_path)
        or not compile_database_path.is_absolute()
        or not compile_database_path.is_file() or compile_database_path.is_symlink()
        or compile_database_path.resolve() != (build_dir / "compile_commands.json").resolve()
        or compile_database.get("file_size_bytes") != compile_database_path.stat().st_size
        or compile_database.get("sha256") != sha(compile_database_path)
        or not profile_path.is_absolute() or not profile_path.is_file() or profile_path.is_symlink()
        or profile_path.resolve() != (build_dir / "bench/ninfer_bench").resolve()
        or inspect_executable(profile_path) != profile_executable
    ):
        raise ValueError("instrumentation build receipt differs from selected profiler route")
    terminal_path = Path(terminal_executable["path"]).resolve(strict=True)
    profile_resolved = profile_path.resolve(strict=True)
    if profile_resolved == terminal_path or profile_executable == terminal_executable:
        raise ValueError("instrumentation executable is not distinct from timing authority")
    terminal_stat, profile_stat = terminal_path.stat(), profile_resolved.stat()
    if (terminal_stat.st_dev, terminal_stat.st_ino) == (profile_stat.st_dev, profile_stat.st_ino):
        raise ValueError("instrumentation executable aliases timing authority")
    cache_values = read_cmake_cache(cache_path)
    if (
        any(cache_values.get(key) != value for key, value in expected_definitions.items())
        or cache_values.get("CMAKE_HOME_DIRECTORY") != str(REPO)
        or cache_values.get("CMAKE_PROJECT_NAME") != "ninfer"
        or cache_values.get("CMAKE_GENERATOR") != "Ninja"
        or cache_values.get("CMAKE_HIP_COMPILER") != "/opt/rocm/llvm/bin/clang++"
    ):
        raise ValueError("instrumentation CMake cache differs from selected compile profile")
    if receipt.get("build_commands") != profile_build_commands(expected_definitions):
        raise ValueError("instrumentation build receipt lacks canonical build commands")
    for name, authority in (
        ("producer", PROFILE_BUILD_PRODUCER), ("route_resolver", PROFILE_ROUTE_RESOLVER)
    ):
        expected = {"path": str(authority), "file_size_bytes": authority.stat().st_size,
                    "sha256": sha(authority)}
        if receipt.get(name) != expected:
            raise ValueError(f"instrumentation build receipt {name} bytes changed")
    return receipt, profile_resolved, cache_path


def prepare(selection_path: Path, receipt_path: Path, out: Path) -> dict:
    selection_path = selection_path.resolve(strict=True)
    selection_raw = selection_path.read_bytes()
    selection = load_payload(selection_raw.decode("utf-8"))
    terminal, selected = validate_terminal_production_authority(selection)
    winner = terminal["winner"]
    sources = [row for row in selection["source_provenance"] if row.get("candidate") == winner]
    if len(sources) != 1:
        raise ValueError("terminal winner lacks unique source provenance")
    source = sources[0]
    binding = source.get("matrices", {}).get("pareto-whole")
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise ValueError("terminal winner lacks a whole-matrix binding")
    manifest_path = Path(binding["path"]).resolve(strict=True)
    if sha(manifest_path) != binding.get("sha256"):
        raise ValueError("selected whole-matrix manifest bytes changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recipe = terminal["winner_artifact"]
    cache = terminal["winner_cache_profile"]
    execution = terminal["winner_execution_profile"]
    chunk = selection["selected_prefill_chunk"]
    corpus_path = Path(str(manifest.get("corpus", ""))).resolve(strict=True)
    corpus = {
        "path": str(corpus_path), "file_size_bytes": corpus_path.stat().st_size,
        "sha256": sha(corpus_path), "tokens": manifest.get("corpus_tokens"),
    }
    expected_execution = {
        "q4_activation_bits": 8, "w8_activation_bits": 8,
        "fp8_qk_wmma_enabled": True,
        "fp8_qk_wmma_profile": FP8_QK_WMMA_PROFILE,
        "fp8_qk_wmma_t1_min_context": FP8_QK_WMMA_T1_MIN_CONTEXT,
        "fp8_qk_wmma_t2_min_context": FP8_QK_WMMA_T2_MIN_CONTEXT,
        "q4_prefill_cta_profile": "m64n128-pingpong-production",
    }
    if (
        manifest.get("artifact_type") != "ninfer_bench_matrix_run"
        or manifest.get("schema_version") != MATRIX_SCHEMA_VERSION
        or manifest.get("preset") != "pareto-whole"
        or manifest.get("concurrency") != list(PRODUCT_CONCURRENCIES)
        or manifest.get("artifact") != source.get("artifact")
        or manifest.get("bench") != source.get("benchmark_executable")
        or manifest.get("expected_kv_value_group") != cache["value_group"]
        or manifest.get("expected_xattention_profile") != execution["xattention_profile"]
        or manifest.get("selected_prefill_chunk") != chunk
        or manifest.get("corpus_sha256") != corpus["sha256"]
        or manifest.get("corpus_tokens") != 65536
        or any(manifest.get(f"expected_{key}") != value
               for key, value in expected_execution.items())
        or any(execution.get(key) != expected_execution[key] for key in (
            "q4_activation_bits", "w8_activation_bits", "fp8_qk_wmma_profile"))
    ):
        raise ValueError("selected whole matrix differs from the terminal route")
    artifact_path = Path(manifest["artifact"]["path"]).resolve(strict=True)
    bench_path = Path(manifest["bench"]["path"]).resolve(strict=True)
    if (recipe.get("weights_id") != manifest["artifact"].get("weights_id")
            or recipe.get("sha256") != manifest["artifact"].get("sha256")):
        raise ValueError("terminal winner artifact differs from selected whole matrix")
    artifact = inspect_artifact(artifact_path)
    hybrid = recipe["weights_id"] == "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
    if hybrid:
        artifact = require_fp8_hybrid_artifact(artifact_path, artifact)
        if manifest.get("required_candidate_identity") != "fp8-hybrid-selection-authority":
            raise ValueError("selected hybrid matrix lacks its candidate authority")
        authority = validate_hybrid_shared_workspace_authority(
            manifest.get("hybrid_shared_workspace_authority"), [chunk]
        )
        planner_path = Path(authority["tool"]["path"]).resolve(strict=True)
        if inspect_executable(planner_path) != authority["tool"]:
            raise ValueError("selected hybrid planner bytes changed")
    elif (
        manifest.get("required_candidate_identity") is not None
        or manifest.get("hybrid_shared_workspace_authority") is not None
    ):
        raise ValueError("selected non-hybrid matrix carries hybrid authority")
    if artifact != manifest["artifact"]:
        raise ValueError("selected artifact bytes changed")
    if inspect_executable(bench_path) != manifest["bench"]:
        raise ValueError("selected benchmark bytes changed")
    marker_source = {
        "path": str(MARKER_SOURCE), "file_size_bytes": MARKER_SOURCE.stat().st_size,
        "sha256": sha(MARKER_SOURCE),
    }
    selected_route = {
        "winner": winner, "artifact": recipe, "cache_profile": cache,
        "execution_profile": execution, "prefill_chunk": chunk,
    }
    receipt, profile_bench_path, cache_path = validate_profile_build_receipt(
        receipt_path, selection_path=selection_path,
        selection_sha256=hashlib.sha256(selection_raw).hexdigest(),
        manifest_path=manifest_path, manifest_sha256=binding["sha256"],
        artifact=manifest["artifact"], terminal_executable=manifest["bench"],
        marker_source=marker_source, selected_route=selected_route,
        compiled_route=expected_execution,
        hybrid_workspace_authority=manifest.get("hybrid_shared_workspace_authority"),
    )
    records = {}
    for concurrency in PRODUCT_CONCURRENCIES:
        matches = [
            row for row in manifest.get("commands", [])
            if isinstance(row, dict) and row.get("concurrency") == concurrency
            and row.get("parity_role") == "ordinary"
        ]
        if len(matches) != 1:
            raise ValueError(f"selected whole matrix lacks one C{concurrency} ordinary control")
        report = Path(matches[0]["report"]).resolve(strict=True)
        reports = binding.get("reports")
        bound_reports = {
            Path(row["path"]).resolve(): row["sha256"]
            for row in reports
            if isinstance(row, dict) and isinstance(row.get("path"), str)
            and isinstance(row.get("sha256"), str)
        } if isinstance(reports, list) else {}
        if report not in bound_reports or sha(report) != bound_reports[report]:
            raise ValueError(f"C{concurrency} source report is not bound by terminal provenance")
        value = json.loads(report.read_text(encoding="utf-8"))
        config = value.get("config", {})
        tests = value.get("tests", [])
        selected_xattention = execution["xattention_profile"]
        xattention_matches = (
            config.get("xattention_qualification") is False
            and not any(key in config for key in (
                "xattention_profile", "xattention_find_block", "xattention_stride",
                "xattention_tau_permille",
            ))
        ) if selected_xattention == "dense" else (
            config.get("xattention_qualification") is True
            and config.get("xattention_profile") == selected_xattention
            and config.get("xattention_find_block") == 128
            and config.get("xattention_stride") == 16
            and config.get("xattention_tau_permille") == 900
        )
        if (
            value.get("artifact_type") != "ninfer_bench_report"
            or value.get("schema_version") != REPORT_SCHEMA_VERSION
            or value.get("command") != " ".join(matches[0]["command"])
            or config.get("concurrency") != concurrency or config.get("spec") != "none"
            or config.get("draft_tokens") != 0 or config.get("prefill_chunk") != chunk
            or config.get("kv_value_group") != cache["value_group"]
            or config.get("corpus_path") != str(corpus_path)
            or config.get("corpus_tokens") != corpus["tokens"]
            or any(config.get(key) != value for key, value in expected_execution.items())
            or not xattention_matches
            or not any(row.get("kind") == "whole" and row.get("n_prompt") == 8192
                       and row.get("n_gen") == 256 for row in tests if isinstance(row, dict))
        ):
            raise ValueError(f"C{concurrency} source is not ordinary 8K+256 decode")
        command = list(matches[0]["command"])
        if "--profile-measured" in command:
            raise ValueError(f"C{concurrency} source command is already profiler-specialized")
        for option, replacement in (
            ("--whole-pg", "8192,256"), ("-r", "1"), ("--warmup", "1")
        ):
            replace_unique_option(command, option, replacement)
        if not command or Path(command[0]).resolve() != bench_path.resolve():
            raise ValueError(f"C{concurrency} source command does not use timing executable")
        command[0] = str(profile_bench_path)
        output = out / f"benchmark-c{concurrency}.json"
        replace_unique_option(command, "--output-file", str(output))
        command.append("--profile-measured")
        records[concurrency] = {
            "source_report": {"path": str(report), "sha256": sha(report)},
            "benchmark_command": command,
            "output": str(output),
        }
    if sha(manifest_path) != binding["sha256"] or sha(selection_path) != hashlib.sha256(selection_raw).hexdigest():
        raise ValueError("selected authority changed while preparing decode profile")
    if os.path.lexists(out):
        raise ValueError(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    plan = {
        "artifact_type": "ninfer_r9700_selected_decode_memory_profile_plan",
        "schema_version": 1,
        "status": "command_only_not_executed",
        "workload": {"prompt_tokens": 8192, "generated_tokens": 256,
                     "concurrency": list(PRODUCT_CONCURRENCIES), "spec": "none",
                     "repetitions": 1, "warmup": 1},
        "selected_route": selected_route,
        "terminal_selection": {"path": str(selection_path), "sha256": sha(selection_path)},
        "source_matrix": {"path": str(manifest_path), "sha256": binding["sha256"]},
        "artifact": manifest["artifact"],
        "terminal_timing_executable": manifest["bench"],
        "profile_benchmark_executable": receipt["instrumentation_executable"],
        "profile_build_receipt": {
            "path": str(receipt_path.resolve()), "file_size_bytes": receipt_path.stat().st_size,
            "sha256": sha(receipt_path),
        },
        "profile_cmake_cache": receipt["cmake_cache"],
        "profile_compile_database": receipt["compile_database"],
        "hybrid_shared_workspace_authority": manifest.get(
            "hybrid_shared_workspace_authority"
        ),
        "corpus": corpus,
        "compiled_route": expected_execution,
        "ordinary_round_marker_source": marker_source,
        "profile_timing_admissible": False,
        "profiler": {"path": str(ROCPROF), "file_size_bytes": ROCPROF.stat().st_size,
                     "sha256": sha(ROCPROF), "sdk_version_required": "1.3.5"},
        "counter_catalog": {"path": str(ROCPROF_AVAIL),
                            "file_size_bytes": ROCPROF_AVAIL.stat().st_size,
                            "sha256": sha(ROCPROF_AVAIL)},
        "counters": list(COUNTERS), "runs": records,
        "measurement_scope": "all dispatches in the measured whole request; with spec=none, analysis requires exactly one outer production Device Graph replay/transaction range for each frontier 8192..8447 and retains only dispatches owned by those 256 ordinary rounds",
        "physical_memory_bandwidth_bytes_per_second": None,
        "physical_peak_fraction": None,
        "stall_freedom": None,
        "interpretation": "GL2/TCP hit ratios, relative GL2 read activity, broad overlapping SQ waits, occupancy, and GL2 write-stall instance-sum activity are proxies only. gfx1201 request-size buckets are known-zero, so physical GDDR6 bandwidth and complete stall freedom are unavailable.",
    }
    plan_path = out / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "", f'root={shlex.quote(str(out))}',
                f'power={shlex.quote(str(POWER))}', "current_after=", "",
                'sha256sum --check --strict "$root/prepared.sha256"',
                'test "$(cat "$power")" = auto',
                f'test "$(sha256sum {shlex.quote(str(ROCPROF))} | cut -d\' \' -f1)" = {sha(ROCPROF)}',
                f'test "$(sha256sum {shlex.quote(str(ROCPROF_AVAIL))} | cut -d\' \' -f1)" = {sha(ROCPROF_AVAIL)}',
                f'{shlex.quote(str(ROCPROF_AVAIL))} --device 0 pmc-check ' + " ".join(COUNTERS),
                "sudo -v", 'restore_auto() {',
                '  printf "%s\\n" auto | sudo tee "$power" >/dev/null',
                '  test "$(cat "$power")" = auto',
                '  if [[ -n "$current_after" ]]; then printf "%s\\n" auto > "$current_after"; fi',
                '}', 'trap restore_auto EXIT', ""]
    for concurrency, record in records.items():
        raw = out / f"raw-c{concurrency}"
        before = out / f"power-c{concurrency}-before.txt"
        after = out / f"power-c{concurrency}-after.txt"
        commands += [f'test ! -e {shlex.quote(str(raw))} && test ! -L {shlex.quote(str(raw))}',
                     f'test ! -e {shlex.quote(record["output"])} && test ! -L {shlex.quote(record["output"])}',
                     f'test ! -e {shlex.quote(str(before))} && test ! -L {shlex.quote(str(before))}',
                     f'test ! -e {shlex.quote(str(after))} && test ! -L {shlex.quote(str(after))}',
                     f'current_after={shlex.quote(str(after))}',
                     'printf "%s\\n" profile_standard | sudo tee "$power" >/dev/null',
                     'test "$(cat "$power")" = profile_standard',
                     f'printf "%s\\n" profile_standard > {shlex.quote(str(before))}',
                     shlex.join([str(ROCPROF), "--selected-regions", "-f", "csv", "rocpd",
                                 "-d", str(raw), "-o", f"selected-decode-c{concurrency}",
                                 "--marker-trace", "--kernel-trace", "--pmc", *COUNTERS,
                                 "--", *record["benchmark_command"]]),
                     "restore_auto", "current_after=", ""]
    commands += ['trap - EXIT', 'bash "$root/postprocess.sh"', ""]
    commands_path = out / "commands.sh"
    commands_path.write_text("\n".join(commands), encoding="utf-8")
    analyzer = REPO / "tools/bench/analyze_selected_decode_memory.py"
    evidence = out / "evidence.json"
    postprocess_path = out / "postprocess.sh"
    postprocess_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        f"root={shlex.quote(str(out))}\n"
        'sha256sum --check --strict "$root/prepared.sha256"\n'
        f"test ! -e {shlex.quote(str(evidence))} && test ! -L {shlex.quote(str(evidence))}\n"
        f"python3 -m tools.bench.analyze_selected_decode_memory --plan {shlex.quote(str(plan_path))} "
        f"--root {shlex.quote(str(out))} --out {shlex.quote(str(evidence))}\n",
        encoding="utf-8",
    )
    closure_paths = [
        REPO / "tools/bench/prepare_selected_decode_memory_profile.py", analyzer,
        PROFILE_BUILD_PRODUCER, PROFILE_ROUTE_RESOLVER,
        selection_path, manifest_path, artifact_path, bench_path, profile_bench_path,
        receipt_path.resolve(), cache_path, Path(receipt["compile_database"]["path"]), plan_path,
        commands_path, postprocess_path, corpus_path, MARKER_SOURCE,
        *([planner_path] if hybrid else []),
        *(Path(record["source_report"]["path"]) for record in records.values()),
    ]
    (out / "prepared.sha256").write_text(
        "".join(f"{sha(path)}  {path.relative_to(REPO)}\n" for path in closure_paths),
        encoding="utf-8",
    )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--profile-build-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        prepare(args.selection, args.profile_build_receipt, lexical_absolute(args.out))
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
