#!/usr/bin/env python3
"""Validate and evaluate the selected dense C1 low-context prefill ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.run_ninfer_bench_matrix import (
    FP8_QK_WMMA_PROFILE,
    FP8_QK_WMMA_T1_MIN_CONTEXT,
    FP8_QK_WMMA_T2_MIN_CONTEXT,
    LOW_CONTEXT_PREFILL_PROMPTS,
    MATRIX_SCHEMA_VERSION,
    R9700_KV_PLANE_LAYOUTS,
    R9700_POWER_PROFILE,
    add_repetition_args,
    build_cases,
    count_corpus_tokens,
    file_sha256,
    inspect_artifact,
    inspect_executable,
    load_bench_report,
    require_fp8_hybrid_artifact,
    validate_hybrid_shared_workspace_authority,
)
from tools.ppl.pareto import load_payload, validate_terminal_production_authority


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_power(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _positive_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _require_statistic(test: dict[str, Any], name: str, values: list[float]) -> None:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stddev = math.sqrt(variance)
    if not (
        type(test.get(f"{name}_mean")) in (int, float)
        and type(test.get(f"{name}_stddev")) in (int, float)
        and math.isfinite(test[f"{name}_mean"])
        and math.isfinite(test[f"{name}_stddev"])
        and math.isclose(test[f"{name}_mean"], mean, rel_tol=2e-6, abs_tol=1e-9)
        and math.isclose(test[f"{name}_stddev"], stddev, rel_tol=2e-6, abs_tol=1e-9)
    ):
        raise ValueError(f"low-context report {name} aggregate differs from repetitions")


def _load_terminal_selection(
    path: Path,
    artifact: dict[str, Any],
    value_group: int,
    prefill_chunk: int,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("terminal selection authority is not a file")
    raw = resolved.read_bytes()
    try:
        value = load_payload(raw.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("terminal selection authority is not UTF-8 JSON") from error
    if type(value.get("schema_version")) is not int:
        raise ValueError("terminal selection authority schema version is not an integer")
    terminal, selected = validate_terminal_production_authority(value)
    recipe = terminal.get("winner_artifact")
    cache = terminal.get("winner_cache_profile")
    if (
        not isinstance(recipe, dict)
        or recipe.get("weights_id") != artifact["weights_id"]
        or recipe.get("sha256") != artifact["sha256"]
    ):
        raise ValueError("terminal selection authority selects different artifact bytes")
    if (
        not isinstance(cache, dict)
        or cache.get("value_group") != value_group
        or selected.get("cache_profile") != cache
    ):
        raise ValueError("terminal selection authority selects a different value group")
    if value.get("selected_prefill_chunk") != prefill_chunk:
        raise ValueError("terminal selection authority selects a different prefill chunk")
    return {
        "path": str(resolved),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "winner": terminal["winner"],
        "winner_artifact": recipe,
        "winner_cache_profile": cache,
        "winner_execution_profile": terminal["winner_execution_profile"],
        "selected_prefill_chunk": value["selected_prefill_chunk"],
    }


def resolve_selected_dense_route(path: Path) -> dict[str, Any]:
    """Resolve the terminal winner's exact dense control without trusting path conventions."""

    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    value = load_payload(raw.decode("utf-8"))
    terminal, _selected = validate_terminal_production_authority(value)
    recipe = terminal["winner_artifact"]
    cache = terminal["winner_cache_profile"]
    chunk = value["selected_prefill_chunk"]
    candidates = [
        candidate for candidate in value["candidates"]
        if candidate.get("weight_recipe") == recipe
        and candidate.get("cache_profile", {}).get("value_group") == cache["value_group"]
        and candidate.get("execution_profile", {}).get("xattention_profile") == "dense"
    ]
    if len(candidates) != 1:
        raise ValueError("terminal selection lacks one exact dense control for its winner")
    candidate = candidates[0]
    sources = [
        source for source in value["source_provenance"]
        if source.get("candidate") == candidate["name"]
    ]
    if len(sources) != 1:
        raise ValueError("terminal selection lacks unique dense-control provenance")
    matrices = sources[0].get("matrices")
    if not isinstance(matrices, dict) or set(matrices) != {"pareto-capacity", "pareto-whole"}:
        raise ValueError("terminal dense control lacks exact capacity and whole matrices")
    manifests: dict[str, dict[str, Any]] = {}
    manifest_bindings: list[tuple[Path, str]] = []
    for preset in ("pareto-capacity", "pareto-whole"):
        binding = matrices[preset]
        if (
            not isinstance(binding, dict)
            or not isinstance(binding.get("path"), str)
            or not _sha256(binding.get("sha256"))
        ):
            raise ValueError("terminal dense-control matrix binding is malformed")
        manifest_path = Path(binding["path"]).resolve(strict=True)
        if file_sha256(manifest_path) != binding["sha256"]:
            raise ValueError("terminal dense-control matrix manifest bytes changed")
        manifest_bindings.append((manifest_path, binding["sha256"]))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("artifact_type") != "ninfer_bench_matrix_run"
            or manifest.get("schema_version") != MATRIX_SCHEMA_VERSION
            or manifest.get("preset") != preset
            or manifest.get("concurrency") != [1, 2, 3, 4]
            or manifest.get("selected_prefill_chunk") != chunk
            or manifest.get("expected_kv_value_group") != cache["value_group"]
            or manifest.get("expected_xattention_profile") != "dense"
            or manifest.get("artifact") != sources[0].get("artifact")
            or manifest.get("bench") != sources[0].get("benchmark_executable")
        ):
            raise ValueError("terminal dense-control matrix identity differs")
        manifests[preset] = manifest
    if manifests["pareto-capacity"]["artifact"] != manifests["pareto-whole"]["artifact"]:
        raise ValueError("terminal dense-control matrices bind different artifacts")
    if manifests["pareto-capacity"]["bench"] != manifests["pareto-whole"]["bench"]:
        raise ValueError("terminal dense-control matrices bind different executables")

    manifest = manifests["pareto-capacity"]
    artifact_path = Path(manifest["artifact"]["path"]).resolve(strict=True)
    executable_path = Path(manifest["bench"]["path"]).resolve(strict=True)
    artifact = inspect_artifact(artifact_path)
    hybrid = recipe["weights_id"] == "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
    if hybrid:
        artifact = require_fp8_hybrid_artifact(artifact_path, artifact)
    if artifact != manifest["artifact"] or inspect_executable(executable_path) != manifest["bench"]:
        raise ValueError("terminal dense-control artifact or executable bytes changed")
    planner_path = None
    if hybrid:
        for matrix in manifests.values():
            validate_hybrid_shared_workspace_authority(
                matrix.get("hybrid_shared_workspace_authority"), [chunk]
            )
        authorities = [
            matrix["hybrid_shared_workspace_authority"] for matrix in manifests.values()
        ]
        if authorities[0] != authorities[1]:
            raise ValueError("terminal dense-control matrices bind different hybrid planners")
        planner_path = Path(authorities[0]["tool"]["path"]).resolve(strict=True)
        if inspect_executable(planner_path) != authorities[0]["tool"]:
            raise ValueError("terminal dense-control hybrid planner bytes changed")
    elif any(matrix.get("hybrid_shared_workspace_authority") is not None for matrix in manifests.values()):
        raise ValueError("non-hybrid terminal dense control carries hybrid planner authority")
    if hashlib.sha256(resolved.read_bytes()).hexdigest() != hashlib.sha256(raw).hexdigest():
        raise ValueError("terminal selection authority changed while resolving dense control")
    if any(file_sha256(manifest_path) != digest for manifest_path, digest in manifest_bindings):
        raise ValueError("terminal dense-control matrix manifest changed while resolving")
    return {
        "terminal_selection": {
            "path": str(resolved), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "winner": terminal["winner"],
        "dense_control": candidate["name"],
        "weights_id": recipe["weights_id"],
        "value_group": cache["value_group"],
        "selected_prefill_chunk": chunk,
        "artifact": {**manifest["artifact"], "path": str(artifact_path)},
        "executable": {**manifest["bench"], "path": str(executable_path)},
        "hybrid_width_tool": str(planner_path) if planner_path is not None else None,
        "hybrid_width_tool_identity": (
            manifests["pareto-capacity"]["hybrid_shared_workspace_authority"]["tool"]
            if planner_path is not None else None
        ),
    }


def validate_ladder(
    manifest_path: Path,
    minimum_p2048_tok_s: float,
    expected_executable: Path,
    expected_artifact: Path,
    terminal_selection_path: Path,
    power_reader: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    if (
        type(minimum_p2048_tok_s) not in (int, float)
        or not math.isfinite(minimum_p2048_tok_s)
        or minimum_p2048_tok_s <= 0
    ):
        raise ValueError("P2048 throughput threshold must be finite and positive")
    manifest_sha256 = file_sha256(manifest_path)
    manifest_metadata = manifest_path.lstat()
    if not stat.S_ISREG(manifest_metadata.st_mode):
        raise ValueError("low-context ladder manifest must be a regular file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    power = manifest.get("power_profile") if isinstance(manifest, dict) else None
    artifact = manifest.get("artifact") if isinstance(manifest, dict) else None
    bench = manifest.get("bench") if isinstance(manifest, dict) else None
    chunk = manifest.get("selected_prefill_chunk") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("artifact_type") != "ninfer_bench_matrix_run"
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != MATRIX_SCHEMA_VERSION
        or manifest.get("preset") != "low-context-prefill"
        or manifest.get("dry_run") is not False
        or manifest.get("concurrency") != [1]
        or type(manifest["concurrency"][0]) is not int
        or type(manifest.get("case_count")) is not int
        or manifest["case_count"] != 5
        or type(manifest.get("point_count")) is not int
        or manifest["point_count"] != 5
        or manifest.get("expected_xattention_profile") != "dense"
        or type(manifest.get("expected_kv_value_group")) is not int
        or manifest["expected_kv_value_group"] not in (16, 32)
        or manifest.get("expected_kv_plane_layouts") != R9700_KV_PLANE_LAYOUTS
        or type(manifest.get("expected_q4_activation_bits")) is not int
        or manifest["expected_q4_activation_bits"] != 8
        or type(manifest.get("expected_w8_activation_bits")) is not int
        or manifest["expected_w8_activation_bits"] != 8
        or manifest.get("expected_fp8_qk_wmma_enabled") is not True
        or manifest.get("expected_fp8_qk_wmma_profile") != FP8_QK_WMMA_PROFILE
        or type(manifest.get("expected_fp8_qk_wmma_t1_min_context")) is not int
        or manifest["expected_fp8_qk_wmma_t1_min_context"] != FP8_QK_WMMA_T1_MIN_CONTEXT
        or type(manifest.get("expected_fp8_qk_wmma_t2_min_context")) is not int
        or manifest["expected_fp8_qk_wmma_t2_min_context"] != FP8_QK_WMMA_T2_MIN_CONTEXT
        or type(chunk) is not int
        or chunk not in (1024, 2048, 4096, 8192)
        or not isinstance(artifact, dict)
        or artifact.get("model_id") != "qwen3.8-27b"
        or not isinstance(artifact.get("weights_id"), str)
        or not artifact["weights_id"]
        or not _sha256(artifact.get("sha256"))
        or type(artifact.get("file_size_bytes")) is not int
        or artifact["file_size_bytes"] <= 0
        or not isinstance(bench, dict)
        or not _sha256(bench.get("sha256"))
        or type(bench.get("file_size_bytes")) is not int
        or bench["file_size_bytes"] <= 0
        or not _sha256(manifest.get("corpus_sha256"))
        or not isinstance(power, dict)
        or power.get("required") != "auto"
        or power.get("sysfs_path") != str(R9700_POWER_PROFILE)
        or power.get("observed") != "auto"
        or power.get("rechecked_after") != "auto"
        or os.path.lexists(manifest_path.parent / "failures.json")
    ):
        raise ValueError("manifest is not a complete physical low-context dense-prefill ladder")

    if power_reader is None:
        power_reader = _read_power
    try:
        live_power = power_reader(R9700_POWER_PROFILE).strip()
    except OSError as error:
        raise ValueError(f"cannot read live power profile: {error}") from error
    if live_power != "auto":
        raise ValueError("live power profile is not auto")
    executable = expected_executable.resolve(strict=True)
    selected_artifact = expected_artifact.resolve(strict=True)
    if bench != inspect_executable(executable):
        raise ValueError("benchmark executable identity differs")
    inspected_artifact = inspect_artifact(selected_artifact)
    hybrid = artifact["weights_id"] == "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
    if hybrid:
        inspected_artifact = require_fp8_hybrid_artifact(selected_artifact, inspected_artifact)
        if manifest.get("required_candidate_identity") != "fp8-hybrid-selection-authority":
            raise ValueError("hybrid low-context ladder lacks its candidate authority")
        validate_hybrid_shared_workspace_authority(
            manifest.get("hybrid_shared_workspace_authority"), [chunk]
        )
    elif (
        manifest.get("required_candidate_identity") is not None
        or manifest.get("hybrid_shared_workspace_authority") is not None
    ):
        raise ValueError("non-hybrid low-context ladder carries hybrid authority")
    if artifact != inspected_artifact:
        raise ValueError("artifact identity differs")
    selected_route = resolve_selected_dense_route(terminal_selection_path)
    if (
        selected_route.get("artifact") != artifact
        or selected_route.get("executable") != bench
        or selected_route.get("value_group") != manifest["expected_kv_value_group"]
        or selected_route.get("selected_prefill_chunk") != chunk
        or Path(selected_route.get("artifact", {}).get("path", "")).resolve()
        != selected_artifact
        or Path(selected_route.get("executable", {}).get("path", "")).resolve()
        != executable
    ):
        raise ValueError("low-context ladder does not use the terminal winner's dense control")
    selected_planner = selected_route.get("hybrid_width_tool_identity")
    if hybrid:
        if manifest["hybrid_shared_workspace_authority"]["tool"] != selected_planner:
            raise ValueError("low-context hybrid planner differs from terminal dense control")
    elif selected_planner is not None:
        raise ValueError("non-hybrid terminal dense control unexpectedly has a planner")
    terminal_selection = _load_terminal_selection(
        terminal_selection_path,
        artifact,
        manifest["expected_kv_value_group"],
        chunk,
    )
    corpus = Path(manifest.get("corpus", ""))
    if not (
        corpus.is_absolute()
        and corpus.is_file()
        and manifest.get("corpus_sha256") == file_sha256(corpus)
        and type(manifest.get("corpus_tokens")) is int
        and manifest["corpus_tokens"] == count_corpus_tokens(corpus)
    ):
        raise ValueError("corpus identity differs")

    cases = {
        case.name: case
        for case in build_cases(
            "low-context-prefill", production_prefill_chunk=chunk
        )
    }
    records = manifest.get("commands")
    expected = {("low_context_prefill", name, 1) for name in cases}
    actual = {
        (record.get("suite"), record.get("case"), record.get("concurrency"))
        for record in records
        if isinstance(record, dict)
    } if isinstance(records, list) else set()
    if not isinstance(records, list) or len(records) != len(expected) or actual != expected:
        raise ValueError("manifest does not retain the exact five-point low-context ladder")
    if len({record.get("report") for record in records}) != len(records):
        raise ValueError("low-context ladder does not retain one distinct report per point")

    ladder: list[dict[str, Any]] = []
    report_provenance: list[tuple[Path, Path, str]] = []
    root = manifest_path.parent.resolve(strict=True)
    for record in records:
        case = cases[record["case"]]
        report_path = Path(record.get("report", ""))
        expected_report = root / "json" / case.suite / "c1" / f"{case.name}.json"
        if report_path != expected_report:
            raise ValueError("low-context ladder report path differs from the fixed protocol")
        if report_path.is_symlink():
            raise ValueError("low-context ladder report must be a regular non-symlink file")
        try:
            resolved_report = report_path.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"low-context ladder report cannot be resolved: {error}") from error
        if not resolved_report.is_relative_to(root):
            raise ValueError("low-context ladder report resolves outside its campaign directory")
        expected_command = add_repetition_args([
            str(executable), "--weights", str(selected_artifact),
            "--corpus", str(corpus), "--device", "0", "--concurrency", "1",
            *case.args, "--output", "json", "--output-file", str(report_path),
        ], case, None, None)
        if record.get("command") != expected_command:
            raise ValueError("low-context ladder command differs from the fixed protocol")
        before = file_sha256(resolved_report)
        report = load_bench_report(
            resolved_report,
            manifest["expected_kv_value_group"],
            8,
            8,
            True,
            1,
            artifact,
            expected_command,
            case,
            "dense",
        )
        if (
            file_sha256(resolved_report) != before
            or report_path.resolve(strict=True) != resolved_report
        ):
            raise ValueError(f"benchmark report changed while validating: {report_path}")
        report_provenance.append((report_path, resolved_report, before))
        prompt = int(case.args[case.args.index("-p") + 1])
        tests = report["tests"]
        if len(tests) != 1 or tests[0].get("label") != f"pp{prompt}":
            raise ValueError(f"report does not retain exactly the P={prompt} prefill row")
        throughput = tests[0].get("prefill_tok_s_mean")
        if not _positive_number(throughput):
            raise ValueError(f"P={prompt} report has invalid prefill throughput")
        reps = tests[0].get("reps")
        prefill_seconds = [
            rep.get("timings", {}).get("prefill_seconds")
            for rep in reps
            if isinstance(rep, dict)
        ] if isinstance(reps, list) else []
        if len(prefill_seconds) != 3 or not all(
            _positive_number(value) for value in prefill_seconds
        ):
            raise ValueError(f"P={prompt} report has invalid raw prefill timings")
        _require_statistic(tests[0], "prefill_seconds", prefill_seconds)
        _require_statistic(
            tests[0], "prefill_tok_s", [prompt / value for value in prefill_seconds]
        )
        ladder.append({
            "prompt_tokens": prompt,
            "prefill_tok_s_mean": float(throughput),
            "report": {"path": str(report_path), "sha256": before},
        })
    ladder.sort(key=lambda row: row["prompt_tokens"])
    if [row["prompt_tokens"] for row in ladder] != list(LOW_CONTEXT_PREFILL_PROMPTS):
        raise ValueError("validated reports do not form the exact low-context ladder")
    p2048 = next(row["prefill_tok_s_mean"] for row in ladder if row["prompt_tokens"] == 2048)
    for report_path, resolved_report, digest in report_provenance:
        if (
            report_path.resolve(strict=True) != resolved_report
            or file_sha256(resolved_report) != digest
        ):
            raise ValueError(f"benchmark report changed while validating: {report_path}")
    if file_sha256(manifest_path) != manifest_sha256:
        raise ValueError("low-context ladder manifest changed while validating")
    if inspect_executable(executable) != bench:
        raise ValueError("benchmark executable changed while validating")
    final_artifact = inspect_artifact(selected_artifact)
    if hybrid:
        final_artifact = require_fp8_hybrid_artifact(selected_artifact, final_artifact)
    if final_artifact != artifact:
        raise ValueError("artifact changed while validating")
    if file_sha256(Path(terminal_selection["path"])) != terminal_selection["sha256"]:
        raise ValueError("terminal selection authority changed while validating")
    if selected_route["terminal_selection"]["sha256"] != terminal_selection["sha256"]:
        raise ValueError("terminal selection changed between dense-route resolution and validation")
    if file_sha256(corpus) != manifest["corpus_sha256"]:
        raise ValueError("corpus changed while validating")
    return {
        "artifact_type": "ninfer_r9700_low_context_prefill_evaluation",
        "schema_version": 1,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "artifact": artifact,
        "bench": bench,
        "terminal_selection": terminal_selection,
        "dense_control": {
            "candidate": selected_route["dense_control"],
            "winner": selected_route["winner"],
            "executable": selected_route["executable"],
            "hybrid_width_tool": selected_route["hybrid_width_tool_identity"],
        },
        "expected_kv_value_group": manifest["expected_kv_value_group"],
        "selected_prefill_chunk": chunk,
        "minimum_p2048_tok_s": float(minimum_p2048_tok_s),
        "observed_p2048_tok_s": p2048,
        "passes_p2048_gate": p2048 >= minimum_p2048_tok_s,
        "ladder": ladder,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--min-p2048-tok-s", required=True, type=float)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out is not None and os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    try:
        result = validate_ladder(
            args.manifest,
            args.min_p2048_tok_s,
            args.executable,
            args.artifact,
            args.selection,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as output:
            output.write(rendered)
    else:
        print(rendered, end="")
    return 0 if result["passes_p2048_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
