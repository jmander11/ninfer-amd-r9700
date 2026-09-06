#!/usr/bin/env python3
"""Select one production prefill chunk from all twelve matched candidates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.matrix_contract import (
    MATRIX_SCHEMA_VERSION,
    PRODUCTION_PREFILL_CHUNKS,
    R9700_KV_PLANE_LAYOUTS,
)
from tools.ppl.run import validate_n16_receipt_summary


def _runner():
    from tools.bench import run_ninfer_bench_matrix
    return run_ninfer_bench_matrix


def build_cases(*args, **kwargs):
    return _runner().build_cases(*args, **kwargs)


def file_sha256(path: Path) -> str:
    return _runner().file_sha256(path)


def load_bench_report(*args, **kwargs):
    return _runner().load_bench_report(*args, **kwargs)


def validate_hybrid_shared_workspace_authority(*args, **kwargs):
    return _runner().validate_hybrid_shared_workspace_authority(*args, **kwargs)

ARTIFACT_TYPE = "ninfer_r9700_prefill_chunk_selection"
SCREENING_ARTIFACT_TYPE = "ninfer_r9700_prefill_chunk_screening"
SCHEMA_VERSION = 2
REQUIRED_RECIPES = (
    "r9700-q4g64-n16k16-eval",
    "r9700-q4-w8-mse-n16k16-eval",
    "r9700-q4g64-f8e4m3-four-role-n16k16-eval",
)
REQUIRED_GROUPS = (16, 32)
REQUIRED_PROFILES = ("dense", "b128-s16-tau900")
RULE = "global_maximin_normalized_prefill_then_workspace_then_smaller_chunk_v2"
MEASUREMENT_SEMANTICS = {
    "requested_output_tokens": 1,
    "base_chunk_profile": "spec-none-ordinary",
    "configured_speculative_backend": "none",
    "configured_draft_window": 0,
    "proposal_head_executed": False,
    "decode_rounds": 0,
    "timed_scope": "ordinary_text_prefill_target_and_sampling",
}


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _identity(manifest: dict[str, Any]) -> tuple[str, int, str]:
    artifact = manifest.get("artifact")
    if (
        not isinstance(artifact, dict)
        or artifact.get("model_id") != "qwen3.8-27b"
        or artifact.get("weights_id") not in REQUIRED_RECIPES
    ):
        raise ValueError("prefill-chunk manifest has an unsupported artifact recipe")
    validate_n16_receipt_summary(
        artifact.get("conversion_receipt"), artifact["weights_id"])
    if (
        artifact.get("weights_id") == "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
        and manifest.get("required_candidate_identity")
        != "fp8-hybrid-selection-authority"
    ):
        raise ValueError("prefill-chunk hybrid manifest lacks its selection authority")
    if (
        not _valid_sha256(artifact.get("sha256"))
        or type(artifact.get("file_size_bytes")) is not int
        or artifact["file_size_bytes"] <= 0
    ):
        raise ValueError("prefill-chunk manifest has incomplete artifact provenance")
    group = manifest.get("expected_kv_value_group")
    profile = manifest.get("expected_xattention_profile")
    if group not in REQUIRED_GROUPS or profile not in REQUIRED_PROFILES:
        raise ValueError("prefill-chunk manifest has an unsupported static profile")
    return artifact["weights_id"], group, profile


def _validate_hybrid_authority(manifest: dict[str, Any], chunks: Sequence[int]) -> None:
    artifact = manifest["artifact"]
    authority = manifest.get("hybrid_shared_workspace_authority")
    if artifact["weights_id"] == "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
        validate_hybrid_shared_workspace_authority(authority, chunks)
    elif authority is not None or manifest.get("required_candidate_identity") is not None:
        raise ValueError("non-hybrid chunk candidate carries hybrid authority")


def _validated_prefill_measurement(
    test: dict[str, Any], prompt: int, report_path: Path
) -> tuple[float, int]:
    """Recompute the point estimate that drives selection from retained repetitions."""
    speculative = test.get("speculative")
    expected_speculative = {
        "enabled": False,
        "draft_window": 0,
        "rounds": 0,
        "drafted_tokens": 0,
        "accepted_tokens": 0,
        "fallback_steps": 0,
        "acceptance_rate": None,
        "acceptance_length": None,
        "accepted_per_position": [],
    }
    if not isinstance(speculative, dict) or any(
        speculative.get(key) != value for key, value in expected_speculative.items()
    ):
        raise ValueError(
            f"{report_path} is not the spec-none ordinary prefill protocol"
        )
    reps = test.get("reps")
    if not isinstance(reps, list) or len(reps) != 3:
        raise ValueError(f"{report_path} does not retain the exact three prefill repetitions")
    seconds: list[float] = []
    for rep in reps:
        timings = rep.get("timings") if isinstance(rep, dict) else None
        value = timings.get("prefill_seconds") if isinstance(timings, dict) else None
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{report_path} has an invalid retained prefill timing")
        seconds.append(float(value))
    throughput = [prompt / value for value in seconds]
    expected = {
        "prefill_seconds_mean": statistics.fmean(seconds),
        "prefill_seconds_stddev": statistics.stdev(seconds),
        "prefill_tok_s_mean": statistics.fmean(throughput),
        "prefill_tok_s_stddev": statistics.stdev(throughput),
    }
    for field, recomputed in expected.items():
        reported = test.get(field)
        if (
            type(reported) not in (int, float)
            or not math.isfinite(reported)
            or not math.isclose(float(reported), recomputed, rel_tol=2e-6, abs_tol=1e-9)
        ):
            raise ValueError(
                f"{report_path} {field} is not derived from its retained repetitions"
            )
    workspace = test.get("workspace_peak_bytes")
    if type(workspace) is not int or workspace <= 0:
        raise ValueError(f"{report_path} has invalid workspace peak")
    return float(test["prefill_tok_s_mean"]), workspace


def _owned_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"prefill-chunk manifest has an invalid {label} path")
    owner = root.expanduser().resolve()
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = owner / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(owner)
    except ValueError as error:
        raise ValueError(
            f"prefill-chunk manifest {label} path is outside its output directory: {value}"
        ) from error
    return resolved


def _manifest(root: Path, prompt: int, chunks: Sequence[int]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    root = root.expanduser().resolve()
    path = root / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    power_profile = value.get("power_profile")
    if (
        value.get("artifact_type") != "ninfer_bench_matrix_run"
        or value.get("schema_version") != MATRIX_SCHEMA_VERSION
        or value.get("preset") != "prefill-chunk"
        or value.get("base_chunk_profile") != "spec-none-ordinary"
        or value.get("dry_run") is not False
        or value.get("failures")
        or value.get("concurrency") != [1]
        or value.get("expected_kv_plane_layouts") != R9700_KV_PLANE_LAYOUTS
        or value.get("expected_q4_activation_bits") != 8
        or value.get("expected_w8_activation_bits") != 8
        or value.get("expected_fp8_qk_wmma_enabled") is not True
        or not isinstance(value.get("bench"), dict)
        or not _valid_sha256(value["bench"].get("sha256"))
        or type(value["bench"].get("file_size_bytes")) is not int
        or value["bench"]["file_size_bytes"] <= 0
        or not isinstance(power_profile, dict)
        or power_profile.get("required") != "auto"
        or power_profile.get("observed") != "auto"
        or power_profile.get("rechecked_after") != "auto"
        or not _valid_sha256(value.get("corpus_sha256"))
    ):
        raise ValueError(f"{path} is not a valid physical production prefill-chunk matrix")
    cases = {
        case.name: case
        for case in build_cases("prefill-chunk", prefill_chunks=chunks, prefill_prompt=prompt)
    }
    _validate_hybrid_authority(value, chunks)
    records = value.get("commands")
    expected = {("production_prefill_chunk", name, 1) for name in cases}
    actual = {
        (row.get("suite"), row.get("case"), row.get("concurrency"))
        for row in records if isinstance(row, dict)
    } if isinstance(records, list) else set()
    if not isinstance(records, list) or len(records) != len(expected) or actual != expected:
        raise ValueError(f"{root} does not retain the exact {prompt}-token chunk point set")
    reports: dict[int, dict[str, Any]] = {}
    for row in records:
        case = cases[row["case"]]
        chunk = int(case.args[case.args.index("--prefill-chunk") + 1])
        report_path = _owned_path(root, row.get("report"), "report")
        command = row.get("command")
        if not isinstance(command, list) or any(not isinstance(part, str) for part in command):
            raise ValueError("prefill-chunk manifest has an invalid command")
        output_indices = [index for index, part in enumerate(command) if part == "--output-file"]
        if len(output_indices) != 1 or output_indices[0] + 1 >= len(command):
            raise ValueError("prefill-chunk manifest command has invalid --output-file")
        if _owned_path(root, command[output_indices[0] + 1], "command output") != report_path:
            raise ValueError("prefill-chunk manifest command output differs from its report")
        report_sha256 = file_sha256(report_path)
        report = load_bench_report(
            report_path,
            value["expected_kv_value_group"], 8, 8, True, 1,
            value["artifact"], row["command"], case, value["expected_xattention_profile"],
        )
        if file_sha256(report_path) != report_sha256:
            raise ValueError(f"{report_path} bytes changed while validating the chunk matrix")
        tests = report.get("tests")
        matches = [test for test in tests if test.get("label") == f"pp{prompt}"] if isinstance(tests, list) else []
        if len(matches) != 1:
            raise ValueError(f"{report_path} lacks one pp{prompt} result")
        test = matches[0]
        throughput, workspace = _validated_prefill_measurement(test, prompt, report_path)
        reports[chunk] = {
            "prefill_tok_s_mean": float(throughput),
            "workspace_peak_bytes": workspace,
            "report": {"path": str(report_path), "sha256": report_sha256},
        }
    return value, reports


def _stable_manifest(
    root: Path, prompt: int, chunks: Sequence[int]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, str]]:
    root = root.expanduser().resolve()
    failures_path = root / "failures.json"
    if failures_path.exists() or failures_path.is_symlink():
        raise ValueError(f"{root} retains a failed matrix marker: {failures_path}")
    path = root / "manifest.json"
    before = file_sha256(path)
    manifest, reports = _manifest(root, prompt, chunks)
    if failures_path.exists() or failures_path.is_symlink():
        raise ValueError(f"{root} retains a failed matrix marker: {failures_path}")
    if file_sha256(path) != before:
        raise ValueError(f"{path} bytes changed while validating the chunk matrix")
    return manifest, reports, {"path": str(path), "sha256": before}


def _report_sources(rows: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"chunk": chunk, **rows[chunk]["report"]}
        for chunk in sorted(rows)
    ]


def _same_candidate(screen: dict[str, Any], final: dict[str, Any]) -> None:
    fields = (
        "artifact", "bench", "corpus_sha256", "base_chunk_profile", "expected_kv_value_group",
        "expected_kv_plane_layouts", "expected_q4_activation_bits",
        "expected_w8_activation_bits", "expected_fp8_qk_wmma_enabled",
        "expected_fp8_qk_wmma_profile", "expected_xattention_profile",
    )
    if any(screen.get(field) != final.get(field) for field in fields):
        raise ValueError("8K and 32K chunk matrices do not bind one candidate identity")
    if screen["artifact"]["weights_id"] == "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
        if (
            screen["hybrid_shared_workspace_authority"]["tool"]
            != final["hybrid_shared_workspace_authority"]["tool"]
        ):
            raise ValueError("8K and 32K hybrid matrices bind different planner bytes")


def _rank(
    chunks: Sequence[int],
    rows: dict[tuple[Any, ...], dict[int, dict[str, Any]]],
    normalizers: dict[tuple[Any, ...], float] | None = None,
) -> list[dict[str, Any]]:
    best = normalizers or {
        candidate: max(values[chunk]["prefill_tok_s_mean"] for chunk in chunks)
        for candidate, values in rows.items()
    }
    if set(best) != set(rows) or any(
        not math.isfinite(value) or value <= 0 for value in best.values()
    ):
        raise ValueError("prefill-chunk normalization inventory differs")
    ranked = []
    for chunk in chunks:
        normalized = {
            "/".join(map(str, candidate)): values[chunk]["prefill_tok_s_mean"] / best[candidate]
            for candidate, values in rows.items()
        }
        ranked.append({
            "chunk": chunk,
            "maximin_normalized_throughput": min(normalized.values()),
            "maximum_workspace_peak_bytes": max(values[chunk]["workspace_peak_bytes"] for values in rows.values()),
            "normalized_throughput": normalized,
        })
    return sorted(
        ranked,
        key=lambda row: (
            -row["maximin_normalized_throughput"],
            row["maximum_workspace_peak_bytes"],
            row["chunk"],
        ),
    )


def build_selection(candidate_roots: Sequence[tuple[Path, Path]]) -> dict[str, Any]:
    if len(candidate_roots) != 12:
        raise ValueError("production chunk selection requires exactly twelve candidates")
    screens: dict[tuple[str, int, str], dict[int, dict[str, Any]]] = {}
    finals: dict[tuple[str, int, str], dict[int, dict[str, Any]]] = {}
    sources = []
    sources_by_identity: dict[tuple[str, int, str], dict[str, Any]] = {}
    screen_manifests: dict[tuple[str, int, str], dict[str, Any]] = {}
    for screen_root, final_root in candidate_roots:
        screen, screen_rows, screen_manifest = _stable_manifest(
            screen_root, 8192, PRODUCTION_PREFILL_CHUNKS
        )
        identity = _identity(screen)
        if identity in screens:
            raise ValueError(f"duplicate prefill-chunk candidate {identity}")
        screens[identity] = screen_rows
        screen_manifests[identity] = screen
        source_record = {
            "weights_id": identity[0], "kv_value_group": identity[1],
            "xattention_profile": identity[2],
            "artifact": screen["artifact"],
            "benchmark_executable": screen["bench"],
            "screen_manifest": screen_manifest,
            "screen_reports": _report_sources(screen_rows),
        }
        sources.append(source_record)
        sources_by_identity[identity] = source_record
    required = {(r, g, p) for r in REQUIRED_RECIPES for g in REQUIRED_GROUPS for p in REQUIRED_PROFILES}
    if set(screens) != required:
        raise ValueError("production chunk selection lacks the exact recipe/group/profile Cartesian set")
    if len({manifest["corpus_sha256"] for manifest in screen_manifests.values()}) != 1:
        raise ValueError("production chunk screens do not share one corpus")
    for recipe in REQUIRED_RECIPES:
        artifacts = {
            json.dumps(manifest["artifact"], sort_keys=True, separators=(",", ":"))
            for identity, manifest in screen_manifests.items() if identity[0] == recipe
        }
        if len(artifacts) != 1:
            raise ValueError(f"production chunk screens bind multiple {recipe} artifacts")

    screen_ranking = _rank(PRODUCTION_PREFILL_CHUNKS, screens)
    finalists = tuple(row["chunk"] for row in screen_ranking[:2])
    for identity, source in zip(screens, candidate_roots, strict=True):
        final_root = source[1]
        final, final_rows, final_manifest = _stable_manifest(final_root, 32768, finalists)
        _same_candidate(screen_manifests[identity], final)
        finals[identity] = final_rows
        sources_by_identity[identity]["finalist_manifest"] = final_manifest
        sources_by_identity[identity]["finalist_reports"] = _report_sources(final_rows)
    combined = {
        (*identity, prompt): rows
        for prompt, collection in ((8192, screens), (32768, finals))
        for identity, rows in collection.items()
    }
    # Preserve each 8K objective's best observed value across the complete four-chunk screen.
    # Re-normalizing 8K over only the finalists would erase a real deficit against an eliminated
    # per-objective leader and can change the global maximin winner.
    final_normalizers = {
        (*identity, prompt): max(
            row["prefill_tok_s_mean"] for row in collection[identity].values()
        )
        for prompt, collection in ((8192, screens), (32768, finals))
        for identity in collection
    }
    final_ranking = _rank(finalists, combined, final_normalizers)
    sources.sort(key=lambda row: (row["weights_id"], row["kv_value_group"], row["xattention_profile"]))
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "selection_rule": RULE,
        "base_chunk_profile": "spec-none-ordinary",
        "measurement_semantics": MEASUREMENT_SEMANTICS,
        "candidate_count": 12,
        "screen_prompt_tokens": 8192,
        "screen_chunks": list(PRODUCTION_PREFILL_CHUNKS),
        "screen_ranking": screen_ranking,
        "final_prompt_tokens": 32768,
        "finalist_chunks": list(finalists),
        "final_ranking": final_ranking,
        "selected_prefill_chunk": final_ranking[0]["chunk"],
        "sources": sources,
    }


def validate_selection_record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("artifact_type") != ARTIFACT_TYPE or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported prefill-chunk selection record")
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != 12:
        raise ValueError("prefill-chunk selection record lacks twelve sources")
    roots = []
    for source in sources:
        screen = source.get("screen_manifest", {})
        final = source.get("finalist_manifest", {})
        screen_reports = source.get("screen_reports")
        finalist_reports = source.get("finalist_reports")
        if (
            not isinstance(screen_reports, list)
            or len(screen_reports) != len(PRODUCTION_PREFILL_CHUNKS)
            or not isinstance(finalist_reports, list)
            or len(finalist_reports) != 2
        ):
            raise ValueError("prefill-chunk selection record lacks bound raw reports")
        for item in (screen, final, *screen_reports, *finalist_reports):
            item_path = Path(item.get("path", ""))
            if not item_path.is_file() or file_sha256(item_path) != item.get("sha256"):
                raise ValueError("prefill-chunk source bytes changed")
        roots.append((Path(screen["path"]).parent, Path(final["path"]).parent))
    rebuilt = build_selection(roots)
    if rebuilt != value:
        raise ValueError("prefill-chunk selection record does not match its source evidence")
    return value


def build_screening(screen_roots: Sequence[Path]) -> dict[str, Any]:
    if len(screen_roots) != 12:
        raise ValueError("screening requires exactly twelve screen directories")
    loaded = [
        _stable_manifest(path, 8192, PRODUCTION_PREFILL_CHUNKS)
        for path in screen_roots
    ]
    rows = {_identity(manifest): reports for manifest, reports, _source in loaded}
    if len(rows) != 12:
        raise ValueError("screening contains duplicate candidate identities")
    required = {
        (recipe, group, profile)
        for recipe in REQUIRED_RECIPES
        for group in REQUIRED_GROUPS
        for profile in REQUIRED_PROFILES
    }
    if set(rows) != required:
        raise ValueError("screening lacks the exact recipe/group/profile Cartesian set")
    if len({manifest["corpus_sha256"] for manifest, _reports, _source in loaded}) != 1:
        raise ValueError("screening candidates do not share one corpus")
    for recipe in REQUIRED_RECIPES:
        artifacts = {
            json.dumps(manifest["artifact"], sort_keys=True, separators=(",", ":"))
            for manifest, _reports, _source in loaded
            if manifest["artifact"]["weights_id"] == recipe
        }
        if len(artifacts) != 1:
            raise ValueError(f"screening binds multiple {recipe} artifacts")
    ranking = _rank(PRODUCTION_PREFILL_CHUNKS, rows)
    return {
        "artifact_type": SCREENING_ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "selection_rule": RULE,
        "candidate_count": 12,
        "measurement_semantics": MEASUREMENT_SEMANTICS,
        "screen_prompt_tokens": 8192,
        "screen_chunks": list(PRODUCTION_PREFILL_CHUNKS),
        "screen_ranking": ranking,
        "finalist_chunks": [row["chunk"] for row in ranking[:2]],
        "sources": sorted(
            (
                {
                    "weights_id": identity[0],
                    "kv_value_group": identity[1],
                    "xattention_profile": identity[2],
                    "screen_manifest": source,
                    "screen_reports": _report_sources(reports),
                }
                for manifest, reports, source in loaded
                for identity in (_identity(manifest),)
            ),
            key=lambda row: (
                row["weights_id"], row["kv_value_group"], row["xattention_profile"]
            ),
        ),
    }


def validate_screening_record(
    path: Path, screen_roots: Sequence[Path]
) -> dict[str, Any]:
    before = file_sha256(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = build_screening(screen_roots)
    if file_sha256(path) != before:
        raise ValueError("prefill-chunk screening record changed while validating")
    if rebuilt != value:
        raise ValueError(
            "prefill-chunk screening record does not match its source evidence"
        )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", action="append", type=Path, required=True, metavar="SCREEN_8K_DIR")
    parser.add_argument("--finalist", action="append", type=Path, metavar="FINALIST_32K_DIR")
    parser.add_argument(
        "--verify-screening", type=Path, metavar="SCREENING_JSON",
        help="rebuild an existing screening record and print its two finalists",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--create-only", action="store_true",
        help="atomically create --out and reject an existing namespace",
    )
    args = parser.parse_args(argv)
    screens = [path.resolve() for path in args.screen]
    if args.verify_screening is not None:
        if args.finalist is not None or args.out is not None:
            parser.error("--verify-screening cannot be combined with --finalist or --out")
        value = validate_screening_record(args.verify_screening.resolve(), screens)
        print(*value["finalist_chunks"], sep="\n")
        return 0
    if args.out is None:
        parser.error("--out is required when creating a screening or selection record")
    if args.finalist is None:
        payload = build_screening(screens)
    else:
        finalists = [path.resolve() for path in args.finalist]
        if len(screens) != 12 or len(finalists) != 12:
            raise SystemExit("final selection requires exactly twelve --screen and --finalist directories")
        payload = build_selection(list(zip(screens, finalists, strict=True)))
    if args.create_only:
        from tools.bench.prefill_chunk_authority import durable_create_json
        durable_create_json(args.out, payload)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if "selected_prefill_chunk" in payload:
        print(f"selected prefill chunk: {payload['selected_prefill_chunk']}")
    else:
        print("global 32K finalists: " + ",".join(map(str, payload["finalist_chunks"])))
    print(f"record sha256: {file_sha256(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
