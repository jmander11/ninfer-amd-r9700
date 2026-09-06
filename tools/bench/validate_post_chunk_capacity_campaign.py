#!/usr/bin/env python3
"""Validate the exact twelve-matrix, 48-cell post-chunk capacity campaign."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
from typing import Any, Sequence

from tools.bench.run_ninfer_bench_matrix import (
    FP8_QK_WMMA_PROFILE,
    FP8_QK_WMMA_T1_MIN_CONTEXT,
    FP8_QK_WMMA_T2_MIN_CONTEXT,
    MATRIX_SCHEMA_VERSION,
    PRODUCT_CONCURRENCIES,
    R9700_KV_PLANE_LAYOUTS,
    build_cases,
    file_sha256,
    load_bench_report,
    manifest_owned_path,
    validate_automatic_feasibility,
    validate_manifest_output_ownership,
)
from tools.bench.prefill_chunk_authority import validate_prefill_chunk_authority
from tools.bench.select_prefill_chunk import REQUIRED_GROUPS, REQUIRED_PROFILES, REQUIRED_RECIPES
from tools.ppl.assemble_pareto import _missing_capacity_provenance
from tools.ppl.run import validate_n16_receipt_summary


EXPECTED_IDENTITIES = {
    (recipe, group, profile)
    for recipe in REQUIRED_RECIPES
    for group in REQUIRED_GROUPS
    for profile in REQUIRED_PROFILES
}


def _regular(path: Path, label: str) -> tuple[int, int]:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    return metadata.st_dev, metadata.st_ino


def _require_option(command: list[str], option: str, value: str) -> None:
    if command.count(option) != 1:
        raise ValueError(f"capacity command requires exactly one {option}")
    index = command.index(option)
    if index + 1 >= len(command) or command[index + 1] != value:
        raise ValueError(f"capacity command requires {option} {value}")


def _validate_capacity_outcomes(
    root: Path, manifest: dict[str, Any], records: list[dict[str, Any]],
) -> int:
    """Require each scheduled cell to be a valid report or one exact retained failure."""

    case = build_cases(
        "pareto-capacity", production_prefill_chunk=manifest["selected_prefill_chunk"]
    )[0]
    failures_path = root / "failures.json"
    failures = None
    if os.path.lexists(failures_path):
        _regular(failures_path, "capacity failures")
        failures = json.loads(failures_path.read_text(encoding="utf-8"))
        if not isinstance(failures, list) or not failures:
            raise ValueError("capacity failures.json must contain retained failures")
    matched: set[int] = set()
    failed = 0
    for record in records:
        report_value = Path(record["report"])
        if not report_value.is_absolute() or report_value.is_symlink():
            raise ValueError("capacity report path is not campaign-owned")
        report = manifest_owned_path(root, record["report"], "capacity report")
        if report.is_file():
            _regular(report, "capacity report")
            loaded = load_bench_report(
                report,
                manifest["expected_kv_value_group"],
                manifest["expected_q4_activation_bits"],
                manifest["expected_w8_activation_bits"],
                manifest["expected_fp8_qk_wmma_enabled"],
                record["concurrency"],
                manifest["artifact"],
                record["command"],
                case,
                manifest["expected_xattention_profile"],
            )
            classified = validate_automatic_feasibility(loaded)
            if classified.get("measurement_kind") != "resolved_effective_maximum":
                raise ValueError("capacity report is not an exact effective maximum")
            continue
        failed += 1
        if failures is None:
            raise ValueError("missing capacity report has no failures.json")
        stem = f"{record['suite']}.{record['case']}.c{record['concurrency']}"
        stderr = root / "logs" / f"{stem}.stderr.txt"
        stdout = root / "logs" / f"{stem}.stdout.txt"
        _regular(stderr, "capacity failure stderr")
        matches = [
            index for index, failure in enumerate(failures)
            if isinstance(failure, dict)
            and failure.get("suite") == record["suite"]
            and failure.get("case") == record["case"]
            and failure.get("concurrency") == record["concurrency"]
            and failure.get("command") == record["command"]
            and failure.get("stderr") == str(stderr)
            and failure.get("stdout") == str(stdout)
            and (
                type(failure.get("returncode")) is int and failure["returncode"] != 0
                or isinstance(failure.get("error"), str) and bool(failure["error"])
            )
        ]
        if len(matches) != 1:
            raise ValueError("missing capacity report has no unique retained failure")
        matched.add(matches[0])
    if failures is not None and matched != set(range(len(failures))):
        raise ValueError("capacity failures.json contains unrelated failures")
    if failed:
        retained = _missing_capacity_provenance(root, manifest)
        if len(retained) != failed:
            raise ValueError("capacity failure provenance differs from Pareto assembly")
    return failed


def validate_matrix(
    root: Path, authority: dict[str, Any], *, executed: bool,
) -> tuple[tuple[str, int, str], int]:
    root = root.expanduser().resolve(strict=True)
    path = root / "manifest.json"
    owner = _regular(path, "capacity manifest")
    before = file_sha256(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if _regular(path, "capacity manifest") != owner or file_sha256(path) != before:
        raise ValueError(f"capacity manifest changed while loading: {path}")
    if (
        not isinstance(manifest, dict)
        or manifest.get("artifact_type") != "ninfer_bench_matrix_run"
        or manifest.get("schema_version") != MATRIX_SCHEMA_VERSION
        or manifest.get("preset") != "pareto-capacity"
        or manifest.get("dry_run") is not False
        or manifest.get("prepare_only") is not (not executed)
        or manifest.get("post_chunk_capacity_gate") is not True
        or manifest.get("prefill_chunk_authority") != authority
        or manifest.get("selected_prefill_chunk") != authority["selected_prefill_chunk"]
        or manifest.get("concurrency") != list(PRODUCT_CONCURRENCIES)
        or manifest.get("case_count") != 1
        or manifest.get("point_count") != 4
        or manifest.get("expected_kv_plane_layouts") != R9700_KV_PLANE_LAYOUTS
        or manifest.get("expected_q4_activation_bits") != 8
        or manifest.get("expected_w8_activation_bits") != 8
        or manifest.get("expected_fp8_qk_wmma_enabled") is not True
        or manifest.get("expected_fp8_qk_wmma_profile") != FP8_QK_WMMA_PROFILE
        or manifest.get("expected_fp8_qk_wmma_t1_min_context")
        != FP8_QK_WMMA_T1_MIN_CONTEXT
        or manifest.get("expected_fp8_qk_wmma_t2_min_context")
        != FP8_QK_WMMA_T2_MIN_CONTEXT
    ):
        raise ValueError(f"{path} is not an exact schema-v14 post-chunk capacity matrix")
    validate_manifest_output_ownership(root, manifest)
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("weights_id") not in REQUIRED_RECIPES:
        raise ValueError("capacity manifest has an unsupported artifact recipe")
    validate_n16_receipt_summary(
        artifact.get("conversion_receipt"), artifact["weights_id"]
    )
    is_hybrid = artifact["weights_id"] == REQUIRED_RECIPES[2]
    if (
        is_hybrid
        != (manifest.get("required_candidate_identity") == "fp8-hybrid-selection-authority")
    ):
        raise ValueError("capacity manifest hybrid selection authority differs from its recipe")
    identity = (
        artifact["weights_id"],
        manifest.get("expected_kv_value_group"),
        manifest.get("expected_xattention_profile"),
    )
    if identity not in EXPECTED_IDENTITIES:
        raise ValueError(f"capacity manifest has an unsupported candidate identity: {identity!r}")
    records = manifest.get("commands")
    expected_points = {
        ("pareto_effective_capacity", "effective_capacity_mtp3", concurrency)
        for concurrency in PRODUCT_CONCURRENCIES
    }
    actual_points = {
        (record.get("suite"), record.get("case"), record.get("concurrency"))
        for record in records
        if isinstance(record, dict)
    } if isinstance(records, list) else set()
    if not isinstance(records, list) or len(records) != 4 or actual_points != expected_points:
        raise ValueError("capacity manifest does not contain exactly C=1,2,3,4")
    if len({record.get("report") for record in records}) != len(records):
        raise ValueError("capacity manifest reuses a report path across product cells")
    for record in records:
        command = record.get("command")
        if not isinstance(command, list) or any(not isinstance(part, str) for part in command):
            raise ValueError("capacity manifest has an invalid command")
        _require_option(command, "--spec", "mtp")
        _require_option(command, "--draft-tokens", "3")
        _require_option(command, "--kv-capacity", "auto")
        _require_option(command, "--max-ctx", "262144")
        _require_option(command, "--prefill-chunk", str(authority["selected_prefill_chunk"]))
        _require_option(command, "--concurrency", str(record["concurrency"]))
        if command.count("--lm-head-draft") != 1 or "--no-device-graph" in command:
            raise ValueError(
                "capacity command must materialize the optimized speculative weights and Device Graph"
            )
    failed = _validate_capacity_outcomes(root, manifest, records) if executed else 0
    if _regular(path, "capacity manifest") != owner or file_sha256(path) != before:
        raise ValueError(f"capacity manifest changed while validating: {path}")
    return identity, failed


def validate_campaign(
    authority_path: Path, roots: Sequence[Path], *, executed: bool,
) -> dict[str, Any]:
    if len(roots) != len(EXPECTED_IDENTITIES):
        raise ValueError("post-chunk capacity campaign requires exactly twelve matrix roots")
    authority_path = authority_path.expanduser()
    authority, source_record = validate_prefill_chunk_authority(authority_path)
    selected = authority["selected_prefill_chunk"]
    sources = source_record.get("sources")
    if not isinstance(sources, list) or len(sources) != 12:
        raise ValueError("prefill-chunk authority lacks twelve receipt-bound sources")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("prefill-chunk authority has an invalid source")
    validated = [validate_matrix(root, authority, executed=executed) for root in roots]
    identities = [item[0] for item in validated]
    failed_points = sum(item[1] for item in validated)
    if len(set(identities)) != len(identities) or set(identities) != EXPECTED_IDENTITIES:
        raise ValueError("post-chunk capacity matrices do not form the exact Cartesian set")
    result = {
        "artifact_type": "ninfer_r9700_post_chunk_capacity_validation",
        "schema_version": 1,
        "selected_prefill_chunk": selected,
        "matrix_count": len(identities),
        "point_count": len(identities) * len(PRODUCT_CONCURRENCIES),
        "validation_scope": "executed" if executed else "prepared",
        "identities": sorted(identities),
    }
    if executed:
        failed_by_identity = dict(validated)
        capacity_eligible = {
            identity for identity, failed in validated if failed == 0
        }
        asymmetric_pairs = [
            (recipe, group)
            for recipe in REQUIRED_RECIPES
            for group in REQUIRED_GROUPS
            if len({
                failed_by_identity[(recipe, group, profile)] == 0
                for profile in REQUIRED_PROFILES
            }) != 1
        ]
        if asymmetric_pairs:
            raise ValueError(
                "post-chunk capacity eligibility differs across matched dense/XAttention "
                f"profiles: {asymmetric_pairs!r}"
            )
        result.update({
            "successful_point_count": (
                len(identities) * len(PRODUCT_CONCURRENCIES) - failed_points
            ),
            "failed_point_count": failed_points,
            "capacity_eligible_matrix_count": len(capacity_eligible),
            "capacity_eligible_identities": sorted(capacity_eligible),
            "whole_eligible_matrix_count": len(capacity_eligible),
            "whole_eligible_identities": sorted(capacity_eligible),
        })
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--matrix", action="append", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--prepared", action="store_true")
    scope.add_argument("--executed", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = validate_campaign(args.authority, args.matrix, executed=args.executed)
    if args.out is None:
        print(json.dumps(result, indent=2))
    else:
        from tools.bench.run_ninfer_bench_matrix import durable_replace_json
        durable_replace_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
