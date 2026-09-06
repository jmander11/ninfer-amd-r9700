#!/usr/bin/env python3
"""Assemble provenance-bound quality/whole/capacity evidence for pareto.py.

Whole-inference rows retain the separately timed prefill and decode phases as well as
fresh-request makespan, so one physical matrix owns all three speed objectives.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.run_ninfer_bench_matrix import (
    MATRIX_SCHEMA_VERSION,
    PRODUCTION_PREFILL_CHUNKS,
    PRODUCT_CONCURRENCIES,
    R9700_POWER_PROFILE,
    R9700_KV_PLANE_LAYOUTS,
    build_cases,
    file_sha256,
    load_bench_report,
    validate_automatic_feasibility,
    validate_hybrid_shared_workspace_authority,
)
from tools.bench.select_prefill_chunk import (
    ARTIFACT_TYPE as PREFILL_CHUNK_ARTIFACT_TYPE,
    RULE as PREFILL_CHUNK_SELECTION_RULE,
    SCHEMA_VERSION as PREFILL_CHUNK_SCHEMA_VERSION,
)
from tools.bench.prefill_chunk_authority import validate_prefill_chunk_authority
from tools.ppl import run as ppl_run


HYBRID_WEIGHTS_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
TERMINAL_RECIPE_IDS = {
    "r9700-q4g64-n16k16-eval", "r9700-q4-w8-mse-n16k16-eval", HYBRID_WEIGHTS_ID,
}

_MINIMUM_RUNTIME_ADMISSION = re.compile(
    r"ninfer_bench: minimum Engine runtime reservation requires ([1-9][0-9]*) bytes "
    r"in addition to ([1-9][0-9]*) bytes of automatic headroom, but only "
    r"([1-9][0-9]*) bytes are available after weights"
)
_AUTOMATIC_HEADROOM_ADMISSION = re.compile(
    r"ninfer_bench: automatic KV headroom requires ([1-9][0-9]*) bytes, but only "
    r"([1-9][0-9]*) bytes are available after weights"
)


def _quality_migration_receipt(artifact: dict[str, Any], weights_id: str) -> dict | None:
    if weights_id not in TERMINAL_RECIPE_IDS:
        return artifact.get("conversion_receipt")
    retained = artifact.get("conversion_receipt")
    if retained is not None:
        return ppl_run.validate_n16_receipt_summary(retained, weights_id)
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("historical N16 quality artifact lacks a joinable path")
    inspected = ppl_run.inspect_candidate_artifact(Path(raw_path), digest=artifact.get("sha256"))
    if (inspected.get("weights_id") != weights_id
            or inspected.get("sha256") != artifact.get("sha256")
            or inspected.get("bytes") != artifact.get("bytes")):
        raise ValueError("historical N16 quality artifact differs from current migration authority")
    return ppl_run.validate_n16_receipt_summary(inspected.get("conversion_receipt"), weights_id)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifest(root: Path, preset: str) -> dict[str, Any]:
    path = root / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    failure_path = root / "failures.json"
    if (
        value.get("artifact_type") != "ninfer_bench_matrix_run"
        or value.get("schema_version") != MATRIX_SCHEMA_VERSION
        or value.get("preset") != preset
        or value.get("dry_run") is not False
        or value.get("failures")
        or value.get("prepare_only") is True
        or (preset == "pareto-whole" and (failure_path.exists() or failure_path.is_symlink()))
    ):
        raise ValueError(f"{path} is not a valid physical {preset} matrix")
    if preset == "pareto-whole" and value.get("power_profile") != {
        "required": "auto",
        "sysfs_path": str(R9700_POWER_PROFILE),
        "observed": "auto",
        "rechecked_after": "auto",
    }:
        raise ValueError(f"{path} does not bind stable auto power for terminal timing")
    return value


def _reports(
    root: Path, manifest: dict[str, Any], preset: str, prefill_chunk: int,
    *, allow_missing: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    campaign_root = root.resolve(strict=True)
    cases = {
        (case.suite, case.name): case
        for case in build_cases(preset, production_prefill_chunk=prefill_chunk)
    }
    expected = manifest["concurrency"]
    records = manifest.get("commands")
    if not isinstance(records, list) or len(records) != len(expected) * len(cases):
        raise ValueError(f"{root} does not retain every {preset} matrix point")
    expected_points = {
        (suite, case, concurrency)
        for suite, case in cases
        for concurrency in expected
    }
    actual_points = {
        (record.get("suite"), record.get("case"), record.get("concurrency"))
        for record in records
        if isinstance(record, dict)
    }
    if len(actual_points) != len(records) or actual_points != expected_points:
        raise ValueError(f"{root} does not retain the exact {preset} matrix point set")
    reports: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        key = (record.get("suite"), record.get("case"))
        if key not in cases:
            raise ValueError(f"{root} has an unknown {preset} case {key}")
        concurrency = record.get("concurrency")
        if concurrency not in expected:
            raise ValueError(f"{root} has unexpected concurrency {concurrency}")
        report_path = Path(record["report"])
        if not report_path.is_absolute() or report_path.is_symlink():
            raise ValueError(f"{root} report path is not campaign-owned: {report_path}")
        report_boundary = (
            report_path.resolve(strict=True)
            if report_path.is_file()
            else report_path.parent.resolve(strict=True) / report_path.name
        )
        try:
            report_boundary.relative_to(campaign_root)
        except ValueError as error:
            raise ValueError(
                f"{root} report path resolves outside its campaign: {report_path}"
            ) from error
        if allow_missing and not report_path.is_file():
            continue
        report = load_bench_report(
            report_path,
            manifest["expected_kv_value_group"],
            manifest["expected_q4_activation_bits"],
            manifest["expected_w8_activation_bits"],
            manifest["expected_fp8_qk_wmma_enabled"],
            concurrency,
            manifest["artifact"],
            record["command"],
            cases[key],
            manifest["expected_xattention_profile"],
        )
        reports.setdefault(concurrency, []).append(report)
    if not allow_missing and sorted(reports) != sorted(expected):
        raise ValueError(f"{root} has an incomplete concurrency set")
    return reports


def _manifest_prefill_chunk(manifest: dict[str, Any], preset: str) -> int:
    records = manifest.get("commands")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{preset} manifest has no commands")
    chunks: set[int] = set()
    for record in records:
        command = record.get("command") if isinstance(record, dict) else None
        if not isinstance(command, list) or command.count("--prefill-chunk") != 1:
            raise ValueError(f"{preset} command lacks one explicit selected prefill chunk")
        index = command.index("--prefill-chunk")
        if index + 1 >= len(command):
            raise ValueError(f"{preset} command has a malformed selected prefill chunk")
        try:
            chunk = int(command[index + 1])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{preset} command has a malformed selected prefill chunk"
            ) from error
        if chunk not in PRODUCTION_PREFILL_CHUNKS:
            raise ValueError(f"{preset} command uses an unsupported selected prefill chunk")
        chunks.add(chunk)
    if len(chunks) != 1:
        raise ValueError(f"{preset} commands do not bind one selected prefill chunk")
    chunk = chunks.pop()
    if type(manifest.get("selected_prefill_chunk")) is not int:
        raise ValueError(f"{preset} manifest lacks an exact selected prefill chunk")
    if manifest["selected_prefill_chunk"] != chunk:
        raise ValueError(f"{preset} manifest selected prefill chunk differs from its commands")
    return chunk


def _bind_prefill_chunk_authority(
    manifest: dict[str, Any], preset: str, authority: dict[str, Any],
) -> None:
    expected_keys = {
        "path", "sha256", "artifact_type", "schema_version", "base_chunk_profile",
        "selected_prefill_chunk",
    }
    if (
        not isinstance(authority, dict)
        or set(authority) != expected_keys
        or not isinstance(authority.get("path"), str)
        or not authority["path"]
        or not Path(authority["path"]).is_absolute()
        or not _valid_sha256(authority.get("sha256"))
        or authority.get("artifact_type") != PREFILL_CHUNK_ARTIFACT_TYPE
        or type(authority.get("schema_version")) is not int
        or authority.get("schema_version") != PREFILL_CHUNK_SCHEMA_VERSION
        or authority.get("base_chunk_profile") != "spec-none-ordinary"
        or type(authority.get("selected_prefill_chunk")) is not int
        or authority["selected_prefill_chunk"] not in PRODUCTION_PREFILL_CHUNKS
    ):
        raise ValueError("selected prefill-chunk authority is malformed")
    if manifest.get("prefill_chunk_authority") != authority:
        raise ValueError(f"{preset} does not bind the selected prefill-chunk authority")
    if preset == "pareto-capacity" and manifest.get("post_chunk_capacity_gate") is not True:
        raise ValueError("pareto-capacity is not a post-chunk capacity gate")


def _one_command_option(command: list[Any], option: str) -> str:
    if command.count(option) != 1:
        raise ValueError(f"capacity failure command lacks exactly one {option}")
    index = command.index(option)
    if index + 1 >= len(command) or not isinstance(command[index + 1], str):
        raise ValueError(f"capacity failure command has malformed {option}")
    return command[index + 1]


def _structured_memory_admission_failure(
    record: dict[str, Any], failure: dict[str, Any], stdout_text: str, stderr_text: str,
) -> dict[str, int | str]:
    if set(failure) != {
        "suite", "case", "concurrency", "returncode", "stdout", "stderr", "command",
    } or type(failure.get("returncode")) is not int or failure["returncode"] != 1:
        raise ValueError("missing capacity report is not an exact process failure record")
    if stdout_text:
        raise ValueError("memory-admission failure unexpectedly wrote stdout")
    command = record.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("capacity failure command is malformed")
    weights = _one_command_option(command, "--weights")
    max_context = _one_command_option(command, "--max-ctx")
    concurrency = _one_command_option(command, "--concurrency")
    kv_capacity = _one_command_option(command, "--kv-capacity")
    draft_tokens = _one_command_option(command, "--draft-tokens")
    if (
        max_context != "262144"
        or concurrency != str(record["concurrency"])
        or kv_capacity != "auto"
        or draft_tokens != "0"
        or "--spec" in command
        or "--lm-head-draft" in command
        or "--no-device-graph" in command
    ):
        raise ValueError("capacity failure command is not the exact automatic ordinary admission case")
    lines = stderr_text.splitlines()
    loading = (
        f"[ninfer_bench] loading {weights} (max_context=262144, "
        f"concurrency={record['concurrency']}, kv_format=fp8-k-int4-v)"
    )
    if len(lines) != 2 or lines[0] != loading or stderr_text != f"{loading}\n{lines[1]}\n":
        raise ValueError("missing capacity report stderr is not an exact startup admission failure")
    minimum = _MINIMUM_RUNTIME_ADMISSION.fullmatch(lines[1])
    if minimum is not None:
        runtime, headroom, available = map(int, minimum.groups())
        if headroom != 1 << 30 or runtime + headroom <= available:
            raise ValueError("reported minimum runtime reservation is not memory-inadmissible")
        return {
            "kind": "minimum_runtime_reservation",
            "minimum_runtime_reservation_bytes": runtime,
            "automatic_headroom_bytes": headroom,
            "available_after_weights_bytes": available,
            "shortfall_bytes": runtime + headroom - available,
        }
    headroom_only = _AUTOMATIC_HEADROOM_ADMISSION.fullmatch(lines[1])
    if headroom_only is not None:
        headroom, available = map(int, headroom_only.groups())
        if headroom != 1 << 30 or headroom <= available:
            raise ValueError("reported automatic headroom is not memory-inadmissible")
        return {
            "kind": "automatic_headroom",
            "automatic_headroom_bytes": headroom,
            "available_after_weights_bytes": available,
            "shortfall_bytes": headroom - available,
        }
    raise ValueError("missing capacity report lacks a recognized memory-admission failure")


def _validate_ordinary_whole_report(
    reports: list[dict[str, Any]], concurrency: int,
) -> dict[str, Any]:
    """Return the sole non-speculative report used by base selection."""

    if len(reports) != 1:
        raise ValueError(f"C{concurrency} pareto-whole requires one ordinary ranking row")
    report = reports[0]
    config = report.get("config", {})
    if (
        config.get("spec") != "none"
        or config.get("draft_tokens") != 0
        or config.get("speculative_execution") is not False
        or config.get("proposal_head") != "full"
    ):
        raise ValueError(f"C{concurrency} pareto-whole is not spec-none ordinary")
    tests = {test.get("label"): test for test in report.get("tests", [])}
    if set(tests) != {"whole-pp8192+tg256", "whole-pp32768+tg256"}:
        raise ValueError(f"C{concurrency} pareto-whole has wrong ranking geometry")
    for prompt in (8192, 32768):
        label = f"whole-pp{prompt}+tg256"
        test = tests[label]
        if (
            test.get("kind") != "whole"
            or test.get("n_prompt") != prompt
            or test.get("n_gen") != 256
            or test.get("requested_output_tokens") != 257
        ):
            raise ValueError(f"C{concurrency} {label} has wrong ordinary geometry")
        reps = test.get("reps")
        if not isinstance(reps, list) or len(reps) != 3:
            raise ValueError(f"C{concurrency} {label} lacks exactly three repetitions")
        timings: dict[str, list[float]] = {
            name: [] for name in ("prepare_seconds", "prefill_seconds", "decode_seconds", "total_seconds")
        }
        throughput = {
            "prefill_tok_s": [], "decode_output_tok_s": [],
            "decode_engine_tok_s": [], "whole_output_tok_s": [],
        }
        ordinary_speculative = {
            "enabled": False, "draft_window": 0, "rounds": 0,
            "drafted_tokens": 0, "accepted_tokens": 0, "fallback_steps": 0,
            "acceptance_rate": None, "acceptance_length": None,
            "accepted_per_position": [],
        }
        if test.get("speculative") != ordinary_speculative:
            raise ValueError(f"C{concurrency} {label} is not aggregate spec-none ordinary")
        for repetition, rep in enumerate(reps):
            timing = rep.get("timings") if isinstance(rep, dict) else None
            if (
                not isinstance(timing, dict)
                or rep.get("generated_output_tokens") != 257 * concurrency
                or rep.get("decode_output_tokens") != 256 * concurrency
                or rep.get("decode_engine_tokens") != 256 * concurrency
                or rep.get("speculative") != ordinary_speculative
                or timing.get("vision_seconds") != 0
                or any(
                    type(timing.get(name)) not in (int, float)
                    or not math.isfinite(timing[name]) or timing[name] <= 0
                    for name in timings
                )
                or timing["total_seconds"] < max(
                    timing["prepare_seconds"], timing["prefill_seconds"],
                    timing["decode_seconds"],
                )
            ):
                raise ValueError(f"C{concurrency} {label} repetition {repetition} has invalid timing")
            for name in timings:
                timings[name].append(float(timing[name]))
            throughput["prefill_tok_s"].append(prompt * concurrency / timing["prefill_seconds"])
            throughput["decode_output_tok_s"].append(256 * concurrency / timing["decode_seconds"])
            throughput["decode_engine_tok_s"].append(256 * concurrency / timing["decode_seconds"])
            throughput["whole_output_tok_s"].append(257 * concurrency / timing["total_seconds"])
        for name, values in {**timings, **throughput}.items():
            for suffix, expected in (
                ("mean", statistics.fmean(values)),
                ("stddev", statistics.stdev(values)),
            ):
                actual = test.get(f"{name}_{suffix}")
                if (
                    type(actual) not in (int, float) or not math.isfinite(actual)
                    or not math.isclose(float(actual), expected, rel_tol=2e-6, abs_tol=1e-9)
                ):
                    raise ValueError(
                        f"C{concurrency} {label} {name}_{suffix} is not derived from repetitions"
                    )
    return report


def _missing_capacity_provenance(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    missing = [record for record in manifest["commands"] if not Path(record["report"]).is_file()]
    failure_path = root / "failures.json"
    if not missing:
        if failure_path.is_file():
            raise ValueError(f"{root} has campaign failures despite complete capacity reports")
        return []
    if not failure_path.is_file():
        raise ValueError(f"{root} has missing capacity reports but no failures.json")
    campaign_failures = json.loads(failure_path.read_text(encoding="utf-8"))
    if not isinstance(campaign_failures, list) or not campaign_failures:
        raise ValueError(f"{failure_path} does not retain campaign failures")
    failure_identity = {"path": str(failure_path), "sha256": file_sha256(failure_path)}
    retained = []
    matched_failure_indexes: set[int] = set()
    for record in missing:
        report_path = Path(record["report"])
        stem = f"{record['suite']}.{record['case']}.c{record['concurrency']}"
        stderr_path = root / "logs" / f"{stem}.stderr.txt"
        stdout_path = root / "logs" / f"{stem}.stdout.txt"
        if (
            not stderr_path.is_file() or stderr_path.is_symlink()
            or not stdout_path.is_file() or stdout_path.is_symlink()
        ):
            raise ValueError(
                f"missing capacity report has no owned stdout/stderr: {report_path}"
            )
        matches = [
            (index, failure)
            for index, failure in enumerate(campaign_failures)
            if isinstance(failure, dict)
            and failure.get("suite") == record["suite"]
            and failure.get("case") == record["case"]
            and failure.get("concurrency") == record["concurrency"]
            and failure.get("command") == record["command"]
            and failure.get("stderr") == str(stderr_path)
            and failure.get("stdout") == str(stdout_path)
        ]
        if len(matches) != 1:
            raise ValueError(f"missing capacity report has no unique campaign failure: {report_path}")
        failure_index, campaign_failure = matches[0]
        matched_failure_indexes.add(failure_index)
        stderr_text = stderr_path.read_text(encoding="utf-8")
        stdout_text = stdout_path.read_text(encoding="utf-8")
        memory_admission = _structured_memory_admission_failure(
            record, campaign_failure, stdout_text, stderr_text
        )
        retained.append({
            "status": "memory_admission_ineligible",
            "concurrency": record["concurrency"],
            "command": record["command"],
            "missing_report": str(report_path),
            "memory_admission": memory_admission,
            "campaign_failure": campaign_failure,
            "failures_file": failure_identity,
            "stderr": {
                "path": str(stderr_path), "sha256": file_sha256(stderr_path),
                "text": stderr_text,
            },
            "stdout": {"path": str(stdout_path), "sha256": file_sha256(stdout_path)},
        })
    if matched_failure_indexes != set(range(len(campaign_failures))):
        raise ValueError(f"{root} has capacity campaign failures unrelated to missing reports")
    return retained


def _quality_candidate(
    quality: dict[str, Any], weights_id: str, group: int, prefill_chunk: int | None = None,
) -> tuple[dict, dict]:
    if (
        quality.get("artifact_type") == "ninfer_r9700_ppl_campaign"
        and quality.get("schema_version") == 6
    ):
        return _campaign_quality_candidate(quality, weights_id, group, prefill_chunk)
    representation = quality.get("representation", {})
    if (
        quality.get("schema") != "ninfer-r9700-q4-a8-final-quality-v2"
        or representation.get("q4_activation_bits") != 8
        or representation.get("w8_activation_bits") != 8
        or representation.get("fp8_qk_wmma_profile")
        != "t1-ge64-t2-ge320-t3plus-stream-v1"
        or representation.get("xattention_profile") not in ("dense", "b128-s16-tau900")
        or representation.get("kv_plane_layouts") != R9700_KV_PLANE_LAYOUTS
    ):
        raise ValueError("quality evidence has an unsupported identity or execution profile")
    artifacts = [item for item in quality.get("artifacts", []) if item.get("weights_id") == weights_id]
    if len(artifacts) != 1:
        raise ValueError(f"quality evidence has no unique artifact {weights_id}")
    artifact = artifacts[0]
    if (
        not _valid_sha256(artifact.get("sha256"))
        or type(artifact.get("bytes")) is not int
        or artifact["bytes"] <= 0
    ):
        raise ValueError(f"quality evidence has an invalid artifact identity for {weights_id}")
    conversion_receipt = _quality_migration_receipt(artifact, weights_id)
    cells: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for tokens, label in ((8192, "8k"), (32768, "32k")):
        matches = [
            row for row in quality.get("results", [])
            if row.get("artifact_weights_id") == weights_id
            and row.get("kv_value_group") == group and row.get("prompt_tokens") == tokens
        ]
        if len(matches) != 1:
            raise ValueError(f"quality evidence lacks unique {weights_id}/G{group}/{label}")
        row = matches[0]
        if row.get("kv_plane_layouts") != R9700_KV_PLANE_LAYOUTS:
            raise ValueError(
                f"quality {weights_id}/G{group}/{label} has the wrong plane layouts"
            )
        cell_path = Path(row["cell"])
        if not cell_path.is_absolute():
            cell_path = REPO_ROOT / cell_path
        for suffix in ("json", "nllf32", "argmaxi32"):
            source_path = cell_path if suffix == "json" else cell_path.with_suffix(f".{suffix}")
            if not source_path.is_file() or file_sha256(source_path) != row["sha256"].get(suffix):
                raise ValueError(f"quality {weights_id}/G{group}/{label} {suffix} bytes changed")
        gates = row.get("gates", {})
        tier = "accuracy" if gates.get("accuracy", {}).get("pass") is True else "capacity-speed"
        gate = gates.get(tier, {})
        cells[label] = {
            "eligible": gate.get("pass") is True,
            "tier": tier,
            "mean_nll_delta": row.get("paired_vs_bf16", {}).get("mean_nll_delta"),
            "complete_finite_aligned": row.get("complete_finite_aligned") is True,
            "scored_positions": row.get("scored_positions"),
            "new_severe_positions": row.get("paired_vs_bf16", {}).get("new_severe_positions"),
        }
        sources[label] = {"path": str(cell_path), "sha256": row["sha256"]}
    return cells, {
        "weights_id": weights_id,
        "sha256": artifact["sha256"],
        "file_size_bytes": artifact["bytes"],
        "conversion_receipt": conversion_receipt,
        "representation": representation,
        "cells": sources,
    }


def _campaign_quality_candidate(
    campaign: dict[str, Any], weights_id: str, group: int, prefill_chunk: int | None,
) -> tuple[dict, dict]:
    artifact = campaign.get("candidate_artifact")
    representation = {
        "q4_activation_bits": campaign.get("q4_activation_bits"),
        "w8_activation_bits": campaign.get("w8_activation_bits"),
        "fp8_qk_wmma_profile": campaign.get("fp8_qk_wmma_profile"),
        "xattention_profile": campaign.get("xattention_profile"),
        "kv_plane_layouts": campaign.get("candidate_kv_plane_layouts"),
    }
    hybrid = weights_id == HYBRID_WEIGHTS_ID
    if (
        campaign.get("pass") is not True
        or campaign.get("model_id") != "qwen3.8-27b"
        or campaign.get("schedules") != ["prefill"]
        or campaign.get("lengths") not in ([8192, 32768], [32768, 8192])
        or (prefill_chunk is not None and campaign.get("prefill_chunk") != prefill_chunk)
        or not isinstance(artifact, dict)
        or artifact.get("weights_id") != weights_id
        or not _valid_sha256(artifact.get("sha256"))
        or type(artifact.get("bytes")) is not int
        or artifact["bytes"] <= 0
        or representation["q4_activation_bits"] != 8
        or representation["w8_activation_bits"] != 8
        or representation["fp8_qk_wmma_profile"]
        != "t1-ge64-t2-ge320-t3plus-stream-v1"
        or representation["xattention_profile"] not in ("dense", "b128-s16-tau900")
        or representation["kv_plane_layouts"] != R9700_KV_PLANE_LAYOUTS
        or campaign.get("required_candidate_identity")
        != ("fp8-hybrid-selection-authority" if hybrid else None)
    ):
        raise ValueError("PPL campaign has an unsupported identity or execution profile")
    conversion_receipt = _quality_migration_receipt(artifact, weights_id)
    cells: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    campaign_cells = campaign.get("cells")
    if not isinstance(campaign_cells, list):
        raise ValueError("PPL campaign cells must be an array")
    for tokens, label in ((8192, "8k"), (32768, "32k")):
        matches = [
            cell for cell in campaign_cells
            if isinstance(cell, dict)
            and cell.get("scheme") == f"r9700-g{group}"
            and cell.get("schedule") == "prefill"
            and cell.get("prompt_tokens") == tokens
        ]
        if len(matches) != 1:
            raise ValueError(f"PPL campaign lacks unique {weights_id}/G{group}/{label}")
        cell = matches[0]
        expected_cell_identity = {
            "scheme": f"r9700-g{group}",
            "model_id": "qwen3.8-27b",
            "weights_id": weights_id,
            "kv_format": "fp8-k-int4-v",
            "kv_value_group": group,
            "schedule": "prefill",
            "prompt_tokens": tokens,
            **({"prefill_chunk": prefill_chunk} if prefill_chunk is not None else {}),
            "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
            "q4_activation_bits": 8,
            "w8_activation_bits": 8,
            "fp8_qk_wmma_enabled": True,
            "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
        }
        expected_cell_identity.update(
            {"xattention_qualification": False}
            if representation["xattention_profile"] == "dense"
            else {
                "xattention_qualification": True,
                "xattention_profile": "b128-s16-tau900",
                "xattention_find_block": 128,
                "xattention_stride": 16,
                "xattention_tau_permille": 900,
            }
        )
        if any(cell.get(key) != value for key, value in expected_cell_identity.items()):
            raise ValueError(
                f"PPL campaign {weights_id}/G{group}/{label} has the wrong compiled profile"
            )
        if representation["xattention_profile"] == "dense" and any(
            key in cell
            for key in (
                "xattention_profile", "xattention_find_block", "xattention_stride",
                "xattention_tau_permille",
            )
        ):
            raise ValueError(
                f"PPL campaign {weights_id}/G{group}/{label} dense cell retains sparse fields"
            )
        command = cell.get("command")
        if not isinstance(command, list) or command.count("--out-json") != 1:
            raise ValueError(f"PPL campaign {weights_id}/G{group}/{label} lacks its command")
        index = command.index("--out-json")
        if index + 1 >= len(command) or not isinstance(command[index + 1], str):
            raise ValueError(f"PPL campaign {weights_id}/G{group}/{label} has a malformed command")
        cell_path = Path(command[index + 1])
        if not cell_path.is_absolute():
            cell_path = REPO_ROOT / cell_path
        if not cell_path.is_file():
            raise ValueError(f"PPL campaign {weights_id}/G{group}/{label} json is missing")
        try:
            raw = json.loads(cell_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"PPL campaign {weights_id}/G{group}/{label} json is invalid"
            ) from error
        if (
            not isinstance(raw, dict)
            or any(raw.get(key) != value for key, value in expected_cell_identity.items())
            or any(cell.get(key) != value for key, value in raw.items())
        ):
            raise ValueError(
                f"PPL campaign {weights_id}/G{group}/{label} json differs from its campaign"
            )
        sources[label] = {"path": str(cell_path), "sha256": {}}
        for suffix, retained_key in (
            ("json", None), ("nllf32", "nll_sha256"), ("argmaxi32", "argmax_sha256")
        ):
            source_path = cell_path if suffix == "json" else cell_path.with_suffix(f".{suffix}")
            digest = file_sha256(source_path) if source_path.is_file() else None
            if digest is None or (retained_key is not None and digest != cell.get(retained_key)):
                raise ValueError(
                    f"PPL campaign {weights_id}/G{group}/{label} {suffix} bytes changed"
                )
            sources[label]["sha256"][suffix] = digest
        tier = cell.get("quality_tier")
        if (
            tier not in ("accuracy", "capacity-speed")
            or cell.get("pass") is not True
            or cell.get("quality_eligible") is not True
            or cell.get("complete_finite_aligned") is not True
            or type(cell.get("tokens_scored")) is not int
            or cell["tokens_scored"] <= 0
            or type(cell.get("new_severe_positions")) is not int
            or cell["new_severe_positions"] < 0
            or isinstance(cell.get("delta_mean_nll"), bool)
            or not isinstance(cell.get("delta_mean_nll"), (int, float))
            or not math.isfinite(float(cell["delta_mean_nll"]))
        ):
            raise ValueError(f"PPL campaign {weights_id}/G{group}/{label} failed quality")
        cells[label] = {
            "eligible": True,
            "tier": tier,
            "mean_nll_delta": cell["delta_mean_nll"],
            "complete_finite_aligned": True,
            "scored_positions": cell["tokens_scored"],
            "new_severe_positions": cell["new_severe_positions"],
        }
    return cells, {
        "weights_id": weights_id,
        "sha256": artifact["sha256"],
        "file_size_bytes": artifact["bytes"],
        "conversion_receipt": conversion_receipt,
        "representation": representation,
        "cells": sources,
        "campaign_identity": {
            "artifact_type": campaign.get("artifact_type"),
            "schema_version": campaign.get("schema_version"),
            "prefill_chunk": campaign.get("prefill_chunk"),
            "corpus": campaign.get("corpus"),
            "reference_weights_id": campaign.get("reference_weights_id"),
            "reference_source": campaign.get("reference_source"),
            "reference_execution": campaign.get("reference_execution"),
            "bf16_scorer": (
                campaign.get("scorers", {}).get("bf16-reference")
                if isinstance(campaign.get("scorers"), dict) else None
            ),
            "reused_bf16_campaign": campaign.get("reused_bf16_campaign"),
            "bf16_repeat_comparison": campaign.get("bf16_repeat_comparison"),
        },
    }


def assemble_candidate(
    name: str, weights_id: str, group: int, quality_path: Path,
    capacity_root: Path, whole_root: Path | None, prefill_chunk: int,
    prefill_chunk_authority: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if prefill_chunk not in PRODUCTION_PREFILL_CHUNKS:
        raise ValueError("selected prefill chunk is unsupported")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality_cells, quality_source = _quality_candidate(
        quality, weights_id, group, prefill_chunk
    )
    manifests = {"pareto-capacity": _manifest(capacity_root, "pareto-capacity")}
    capacity_failures = _missing_capacity_provenance(
        capacity_root, manifests["pareto-capacity"]
    )
    if not capacity_failures and whole_root is None:
        raise ValueError("capacity-eligible candidate requires a whole matrix")
    if whole_root is not None:
        manifests["pareto-whole"] = _manifest(whole_root, "pareto-whole")
    for preset, manifest in manifests.items():
        _bind_prefill_chunk_authority(manifest, preset, prefill_chunk_authority)
        if _manifest_prefill_chunk(manifest, preset) != prefill_chunk:
            raise ValueError(f"{preset} does not use selected prefill chunk {prefill_chunk}")
        profile_field = (
            "base_capacity_profile" if preset == "pareto-capacity"
            else "base_ranking_profile"
        )
        if manifest.get(profile_field) != "spec-none-ordinary":
            raise ValueError(f"{preset} is not bound to spec-none ordinary")
    if prefill_chunk_authority["selected_prefill_chunk"] != prefill_chunk:
        raise ValueError("selected prefill-chunk authority differs from assembled chunk")
    identity = manifests["pareto-capacity"]["artifact"]
    expected_identity = {
        "weights_id": weights_id,
        "sha256": quality_source["sha256"],
        "file_size_bytes": quality_source["file_size_bytes"],
    }
    for preset, manifest in manifests.items():
        if manifest.get("concurrency") != list(PRODUCT_CONCURRENCIES):
            raise ValueError(f"{preset} does not contain the required ordered C=1..4 sweep")
        actual = manifest["artifact"]
        if any(actual.get(key) != value for key, value in expected_identity.items()):
            raise ValueError(f"{preset} artifact does not match quality evidence")
        if manifest.get("expected_kv_value_group") != group:
            raise ValueError(f"{preset} cache group does not match candidate G{group}")
        if manifest.get("expected_kv_plane_layouts") != R9700_KV_PLANE_LAYOUTS:
            raise ValueError(f"{preset} cache plane layouts do not match the quality profile")
        if (
            manifest.get("expected_q4_activation_bits")
            != quality_source["representation"]["q4_activation_bits"]
            or manifest.get("expected_w8_activation_bits")
            != quality_source["representation"]["w8_activation_bits"]
            or manifest.get("expected_fp8_qk_wmma_profile")
            != quality_source["representation"]["fp8_qk_wmma_profile"]
            or manifest.get("expected_xattention_profile")
            != quality_source["representation"].get("xattention_profile")
        ):
            raise ValueError(f"{preset} execution profile does not match quality evidence")
        hybrid_identity = (
            "fp8-hybrid-selection-authority" if weights_id == HYBRID_WEIGHTS_ID else None
        )
        if manifest.get("required_candidate_identity") != hybrid_identity:
            raise ValueError(f"{preset} weight-recipe authority does not match quality evidence")
        if (weights_id in TERMINAL_RECIPE_IDS
                and actual.get("conversion_receipt")
                != quality_source.get("conversion_receipt")):
            raise ValueError(
                f"{preset} N16 migration receipt differs from quality evidence"
            )
        if weights_id == HYBRID_WEIGHTS_ID:
            validate_hybrid_shared_workspace_authority(
                manifest.get("hybrid_shared_workspace_authority"), [prefill_chunk]
            )
        elif manifest.get("hybrid_shared_workspace_authority") is not None:
            raise ValueError(f"{preset} non-hybrid candidate carries hybrid workspace authority")
    benches = {json.dumps(manifest["bench"], sort_keys=True) for manifest in manifests.values()}
    if len(benches) != 1:
        raise ValueError("capacity and whole matrices use different benchmark bytes")
    if weights_id == HYBRID_WEIGHTS_ID and len({
        json.dumps(manifest["hybrid_shared_workspace_authority"]["tool"], sort_keys=True)
        for manifest in manifests.values()
    }) != 1:
        raise ValueError("capacity and whole hybrid matrices use different planner bytes")

    capacity_reports = _reports(
        capacity_root, manifests["pareto-capacity"], "pareto-capacity", prefill_chunk,
        allow_missing=True,
    )
    whole_reports = (
        _reports(whole_root, manifests["pareto-whole"], "pareto-whole", prefill_chunk)
        if whole_root is not None else {}
    )
    capacity: dict[str, Any] = {}
    speeds: dict[str, float] = {}
    for concurrency in PRODUCT_CONCURRENCIES:
        if concurrency in capacity_reports:
            cap = validate_automatic_feasibility(capacity_reports[concurrency][0])
            if cap["measurement_kind"] != "resolved_effective_maximum":
                raise ValueError(f"C{concurrency} capacity is not an effective maximum")
            capacity[f"c{concurrency}"] = {
                "measurement_kind": cap["measurement_kind"],
                "binding_constraint": cap["binding_constraint"],
                "tokens": cap["resolved_effective_maximum_tokens"],
            }
        if whole_reports:
            ordinary_report = _validate_ordinary_whole_report(
                whole_reports[concurrency], concurrency
            )
            whole_tests = {test["label"]: test for test in ordinary_report["tests"]}
            for tokens in (8192, 32768):
                row = whole_tests[f"whole-pp{tokens}+tg256"]
                speeds[f"prefill_{tokens}_c{concurrency}"] = row["prefill_tok_s_mean"]
                speeds[f"decode_{tokens}_c{concurrency}"] = row["decode_output_tok_s_mean"]
                speeds[f"whole_{tokens}_c{concurrency}"] = row["whole_output_tok_s_mean"]
    candidate = {
        "name": name,
        "prefill_chunk": prefill_chunk,
        "cache_profile": {
            "value_group": group,
            "plane_layouts": R9700_KV_PLANE_LAYOUTS,
        },
        "execution_profile": {
            key: quality_source["representation"][key]
            for key in (
                "q4_activation_bits", "w8_activation_bits", "fp8_qk_wmma_profile",
                "xattention_profile",
            )
        },
        "quality_cells": quality_cells,
        "capacity_by_cell": capacity,
        "whole_inference_tokens_per_second": speeds,
        "whole_inference_profile": "spec-none-ordinary",
        "base_capacity_profile": "spec-none-ordinary",
    }
    provenance = {
        "candidate": name,
        "selected_prefill_chunk": prefill_chunk,
        "prefill_chunk_authority": prefill_chunk_authority,
        "cache_value_group": group,
        "artifact": identity,
        "benchmark_executable": manifests["pareto-capacity"]["bench"],
        "quality": {
            "path": str(quality_path),
            "sha256": file_sha256(quality_path),
            "artifact": {
                "weights_id": quality_source["weights_id"],
                "sha256": quality_source["sha256"],
                "file_size_bytes": quality_source["file_size_bytes"],
            },
            "representation": quality_source["representation"],
            "cells": quality_source["cells"],
            "campaign_identity": quality_source.get("campaign_identity"),
        },
        "matrices": {
            preset: {
                "path": str(root / "manifest.json"),
                "sha256": file_sha256(root / "manifest.json"),
                "reports": [
                    {"path": record["report"], "sha256": file_sha256(Path(record["report"]))}
                    for record in manifests[preset]["commands"]
                    if Path(record["report"]).is_file()
                ],
            }
            for preset, root in (
                ("pareto-capacity", capacity_root), ("pareto-whole", whole_root),
            )
            if root is not None
        },
        "capacity_failures": capacity_failures,
    }
    return candidate, provenance


def validate_xattention_dense_controls(
    candidates: list[dict[str, Any]], provenance: list[dict[str, Any]],
) -> None:
    """Require complete dense/sparse G16/G32 controls for every candidate recipe."""

    expected = {
        (16, "dense"),
        (32, "dense"),
        (16, "b128-s16-tau900"),
        (32, "b128-s16-tau900"),
    }
    if len(candidates) != len(provenance) or not candidates:
        raise ValueError("XAttention admission requires candidate provenance")
    candidates_by_artifact: dict[tuple[object, object, object], list[dict[str, Any]]] = {}
    for candidate, source in zip(candidates, provenance, strict=True):
        artifact = source.get("artifact", {})
        key = (
            artifact.get("weights_id"), artifact.get("sha256"),
            artifact.get("file_size_bytes"),
        )
        candidates_by_artifact.setdefault(key, []).append(candidate)
    for key, recipe_candidates in candidates_by_artifact.items():
        if (
            not isinstance(key[0], str) or not key[0]
            or not _valid_sha256(key[1])
            or type(key[2]) is not int or key[2] <= 0
        ):
            raise ValueError("XAttention candidate artifact identity is incomplete")
        actual = [
            (candidate.get("cache_profile", {}).get("value_group"),
             candidate.get("execution_profile", {}).get("xattention_profile"))
            for candidate in recipe_candidates
        ]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(
                f"XAttention admission requires dense and b128-s16-tau900 G16/G32 "
                f"controls for every candidate artifact; {key[0]} is incomplete"
            )

    campaign_by_artifact_profile: dict[
        tuple[object, object, object, str], set[tuple[str, str]]
    ] = {}
    capacity_eligibility: dict[
        tuple[object, object, object, int], dict[str, bool]
    ] = {}
    shared_bindings: set[str] = set()
    for candidate, source in zip(candidates, provenance, strict=True):
        profile = candidate["execution_profile"]["xattention_profile"]
        quality = source.get("quality", {})
        representation = quality.get("representation") if isinstance(quality, dict) else None
        matrices = source.get("matrices")
        if (
            source.get("candidate") != candidate.get("name")
            or not isinstance(representation, dict)
            or representation.get("xattention_profile") != profile
        ):
            raise ValueError("XAttention candidate and quality provenance identities disagree")
        capacity_failures = source.get("capacity_failures")
        if not isinstance(capacity_failures, list):
            raise ValueError("XAttention capacity failure provenance is malformed")
        capacity_eligible = not capacity_failures
        expected_matrices = (
            {"pareto-capacity", "pareto-whole"}
            if capacity_eligible else {"pareto-capacity"}
        )
        if not isinstance(matrices, dict) or set(matrices) != expected_matrices:
            raise ValueError(
                "XAttention admission requires matched capacity and whole matrices "
                "exactly for capacity-eligible profiles"
            )
        identity = quality.get("campaign_identity") if isinstance(quality, dict) else None
        if (
            not isinstance(identity, dict)
            or identity.get("artifact_type") != "ninfer_r9700_ppl_campaign"
            or identity.get("schema_version") != 6
            or identity.get("prefill_chunk") != candidate.get("prefill_chunk")
        ):
            raise ValueError("XAttention admission requires native schema-v6 PPL campaigns")
        corpus = identity.get("corpus")
        reference_source = identity.get("reference_source")
        reference_execution = identity.get("reference_execution")
        bf16_scorer = identity.get("bf16_scorer")
        reused_bf16 = identity.get("reused_bf16_campaign")
        repeat_comparison = identity.get("bf16_repeat_comparison")
        if (
            not isinstance(corpus, dict)
            or not _valid_sha256(corpus.get("ids_sha256"))
            or not _valid_sha256(corpus.get("manifest_sha256"))
            or identity.get("reference_weights_id") != "bf16-source"
            or not isinstance(reference_source, dict)
            or not _valid_sha256(reference_source.get("config_sha256"))
            or not _valid_sha256(reference_source.get("index_sha256"))
            or not isinstance(reference_execution, dict)
            or not isinstance(bf16_scorer, dict)
            or not _valid_sha256(bf16_scorer.get("sha256"))
            or type(bf16_scorer.get("bytes")) is not int
            or bf16_scorer["bytes"] <= 0
            or not isinstance(reused_bf16, dict)
            or set(reused_bf16) != {"path", "sha256"}
            or not isinstance(reused_bf16.get("path"), str)
            or not _valid_sha256(reused_bf16.get("sha256"))
            or not isinstance(repeat_comparison, dict)
            or set(repeat_comparison) != {
                "path", "sha256", "authority_input", "authority_campaign_sha256"
            }
            or repeat_comparison.get("authority_input") not in ("first", "second")
            or repeat_comparison.get("authority_campaign_sha256") != reused_bf16.get("sha256")
        ):
            raise ValueError("XAttention schema-v6 PPL authority bindings are incomplete")
        try:
            validated_repeat = ppl_run.validate_bf16_repeat_comparison(
                Path(repeat_comparison["path"]), Path(reused_bf16["path"])
            )
        except (OSError, SystemExit) as error:
            raise ValueError("XAttention BF16 repeat authority is invalid") from error
        if validated_repeat != repeat_comparison:
            raise ValueError("XAttention BF16 repeat authority binding differs")
        path = quality.get("path")
        digest = quality.get("sha256")
        if not isinstance(path, str) or not _valid_sha256(digest):
            raise ValueError("XAttention PPL campaign identity is incomplete")
        artifact = source["artifact"]
        group = candidate["cache_profile"]["value_group"]
        eligibility_key = (
            artifact["weights_id"], artifact["sha256"], artifact["file_size_bytes"], group,
        )
        capacity_eligibility.setdefault(eligibility_key, {})[profile] = capacity_eligible
        campaign_key = (
            artifact["weights_id"], artifact["sha256"], artifact["file_size_bytes"], profile,
        )
        campaign_by_artifact_profile.setdefault(campaign_key, set()).add((path, digest))
        shared_bindings.add(json.dumps({
            "corpus": identity.get("corpus"),
            "prefill_chunk": identity.get("prefill_chunk"),
            "reference_weights_id": identity.get("reference_weights_id"),
            "reference_source": identity.get("reference_source"),
            "reference_execution": identity.get("reference_execution"),
            "bf16_scorer": identity.get("bf16_scorer"),
            "reused_bf16_campaign": identity.get("reused_bf16_campaign"),
            "bf16_repeat_comparison": identity.get("bf16_repeat_comparison"),
        }, sort_keys=True, separators=(",", ":")))
    if any(
        set(profiles) != {"dense", "b128-s16-tau900"}
        or len(set(profiles.values())) != 1
        for profiles in capacity_eligibility.values()
    ):
        raise ValueError(
            "XAttention admission requires matched dense/sparse capacity eligibility "
            "for each recipe and cache group"
        )
    if any(len(campaigns) != 1 for campaigns in campaign_by_artifact_profile.values()):
        raise ValueError(
            "each artifact's XAttention profile must use one shared G16/G32 PPL campaign"
        )
    if len(shared_bindings) != 1:
        raise ValueError("dense and XAttention PPL campaigns do not share one BF16/corpus authority")
    required_recipes = TERMINAL_RECIPE_IDS
    present_recipes = {key[0] for key in candidates_by_artifact}
    if present_recipes != required_recipes:
        raise ValueError(
            "XAttention production admission requires complete all-Q4, mixed Q4/W8, "
            "and four-role FP8/Q4 artifact profile sets"
        )


def validate_chunk_candidate_bindings(
    chunk_selection: dict[str, Any], provenance: list[dict[str, Any]],
) -> None:
    sources = chunk_selection.get("sources")
    if not isinstance(sources, list) or len(sources) != len(provenance):
        raise ValueError("prefill-chunk selection does not cover every Pareto candidate")
    expected = {}
    for source in sources:
        key = (
            source.get("weights_id"), source.get("kv_value_group"),
            source.get("xattention_profile"),
        )
        if key in expected:
            raise ValueError("prefill-chunk selection has duplicate candidate bindings")
        expected[key] = source
    actual = set()
    for source in provenance:
        artifact = source.get("artifact", {})
        quality = source.get("quality", {})
        representation = quality.get("representation", {}) if isinstance(quality, dict) else {}
        key = (
            artifact.get("weights_id"), source.get("cache_value_group"),
            representation.get("xattention_profile"),
        )
        bound = expected.get(key)
        if (
            bound is None
            or bound.get("artifact") != artifact
            or bound.get("benchmark_executable") != source.get("benchmark_executable")
        ):
            raise ValueError("Pareto candidate does not match its prefill-chunk selection identity")
        actual.add(key)
    if actual != set(expected):
        raise ValueError("prefill-chunk selection and Pareto candidate sets differ")


def bind_post_chunk_capacity_validation(
    path: Path,
    chunk_selection_path: Path,
    capacity_roots: list[Path],
    candidates: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rebuild and bind the executed capacity campaign that controls whole eligibility."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("post-chunk capacity validation is not a regular file")
    resolved = path.resolve(strict=True)
    before = file_sha256(resolved)
    retained = json.loads(resolved.read_text(encoding="utf-8"))
    from tools.bench.validate_post_chunk_capacity_campaign import validate_campaign
    rebuilt = validate_campaign(chunk_selection_path, capacity_roots, executed=True)
    if retained != rebuilt or file_sha256(resolved) != before:
        raise ValueError("post-chunk capacity validation differs from its physical campaign")
    eligible = {
        tuple(identity) for identity in retained.get("capacity_eligible_identities", [])
        if isinstance(identity, list) and len(identity) == 3
    }
    actual = {
        (
            source["artifact"]["weights_id"],
            source["cache_value_group"],
            candidate["execution_profile"]["xattention_profile"],
        )
        for candidate, source in zip(candidates, provenance, strict=True)
        if not source["capacity_failures"]
    }
    if eligible != actual:
        raise ValueError("post-chunk capacity eligibility differs from assembled candidates")
    return {
        "path": str(resolved),
        "sha256": before,
        "artifact_type": retained["artifact_type"],
        "schema_version": retained["schema_version"],
        "selected_prefill_chunk": retained["selected_prefill_chunk"],
        "capacity_eligible_identities": retained["capacity_eligible_identities"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", nargs=6, required=True,
        metavar=(
            "NAME", "WEIGHTS_ID", "GROUP", "QUALITY_JSON", "CAPACITY_DIR", "WHOLE_DIR",
        ),
    )
    parser.add_argument(
        "--require-xattention-dense-controls", action="store_true",
        help="require matched dense and B128/S16/tau900 G16/G32 admission candidates",
    )
    parser.add_argument(
        "--prefill-chunk-selection", type=Path, required=True,
        help="validated schema-v2 global prefill-chunk selection record",
    )
    parser.add_argument(
        "--post-chunk-capacity-validation", type=Path,
        help="executed twelve-matrix capacity validation controlling whole eligibility",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    chunk_selection_path = args.prefill_chunk_selection.resolve()
    prefill_chunk_authority, chunk_selection = validate_prefill_chunk_authority(
        chunk_selection_path
    )
    chunk_selection_sha256 = prefill_chunk_authority["sha256"]
    prefill_chunk = chunk_selection["selected_prefill_chunk"]
    candidates, provenance = [], []
    for name, weights_id, group, quality, capacity, whole in args.candidate:
        whole_path = None if whole == "-" else Path(whole).resolve()
        candidate, source = assemble_candidate(
            name, weights_id, int(group), Path(quality).resolve(), Path(capacity).resolve(),
            whole_path, prefill_chunk, prefill_chunk_authority,
        )
        candidates.append(candidate)
        provenance.append(source)
    if args.require_xattention_dense_controls:
        validate_xattention_dense_controls(candidates, provenance)
        validate_chunk_candidate_bindings(chunk_selection, provenance)
        if args.post_chunk_capacity_validation is None:
            raise SystemExit(
                "static profile selection requires --post-chunk-capacity-validation"
            )
        capacity_binding = bind_post_chunk_capacity_validation(
            args.post_chunk_capacity_validation,
            chunk_selection_path,
            [Path(candidate[4]).resolve() for candidate in args.candidate],
            candidates,
            provenance,
        )
        for source in provenance:
            source["post_chunk_capacity_validation"] = capacity_binding
    elif args.post_chunk_capacity_validation is not None:
        raise SystemExit(
            "--post-chunk-capacity-validation requires --require-xattention-dense-controls"
        )
    if file_sha256(chunk_selection_path) != chunk_selection_sha256:
        raise SystemExit("prefill-chunk selection bytes changed during assembly")
    speed = sorted(
        f"{phase}_{tokens}_c{concurrency}"
        for concurrency in PRODUCT_CONCURRENCIES
        for tokens in (8192, 32768)
        for phase in ("prefill", "decode", "whole")
    )
    if any(
        candidate["whole_inference_tokens_per_second"]
        and sorted(candidate["whole_inference_tokens_per_second"]) != speed
        for candidate in candidates
    ):
        raise SystemExit("candidate speed workload sets differ")
    payload = {
        "artifact_type": "ninfer_r9700_pareto_input",
        "schema_version": 4,
        "required_quality_cells": ["8k", "32k"],
        "required_capacity_cells": [f"c{i}" for i in PRODUCT_CONCURRENCIES],
        "required_speed_workloads": speed,
        "base_ranking_profile": "spec-none-ordinary",
        "base_capacity_profile": "spec-none-ordinary",
        "selected_prefill_chunk": prefill_chunk,
        "prefill_chunk_selection": {
            "path": str(chunk_selection_path),
            "sha256": chunk_selection_sha256,
            "selection_rule": PREFILL_CHUNK_SELECTION_RULE,
            "selected_prefill_chunk": prefill_chunk,
        },
        "require_single_static_profile_selection": args.require_xattention_dense_controls,
        "candidates": candidates,
        "source_provenance": provenance,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
