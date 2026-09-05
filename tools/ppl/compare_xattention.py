#!/usr/bin/env python3
"""Assemble the matched dense-versus-XAttention real-model PPL comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN_TYPE = "ninfer_r9700_ppl_campaign"
CAMPAIGN_SCHEMA = 6
OUTPUT_TYPE = "ninfer_r9700_xattention_ppl_comparison"
OUTPUT_SCHEMA = 1
DENSE_PROFILE = "dense"
XATTENTION_PROFILE = "b128-s16-tau900"
CANDIDATES = ("r9700-g16", "r9700-g32")
EXPECTED_LENGTHS = (8192, 32768)
TOKEN_DOMAIN = 248077
QUALITY_TIER = "capacity-speed"
QUALITY_GATE = math.log(1.05)
EXPECTED_GATES = {candidate: QUALITY_GATE for candidate in CANDIDATES}

SHARED_FIELDS = (
    "model_id",
    "reference_weights_id",
    "reference_source",
    "reference_execution",
    "candidate_artifact",
    "corpus",
    "lengths",
    "skip",
    "prefill_chunk",
    "schedules",
    "spec",
    "draft_tokens",
    "q4_activation_bits",
    "w8_activation_bits",
    "candidate_kv_plane_layouts",
    "fp8_qk_wmma_enabled",
    "fp8_qk_wmma_profile",
    "fp8_qk_wmma_t1_min_context",
    "fp8_qk_wmma_t2_min_context",
    "baseline",
    "gates",
    "quality_tier",
    "quality_gate_contract",
    "schedule_parity_max_abs_nll",
    "execution_parity_max_abs_nll",
    "terrible_nll",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} root must be an object")
    return value


def _cell_path(cell: dict[str, Any]) -> Path:
    command = cell.get("command")
    if not isinstance(command, list) or command.count("--out-json") != 1:
        raise ValueError("campaign cell does not retain one --out-json command argument")
    index = command.index("--out-json")
    if index + 1 >= len(command) or not isinstance(command[index + 1], str):
        raise ValueError("campaign cell has a malformed --out-json command argument")
    path = Path(command[index + 1])
    return path if path.is_absolute() else REPO / path


def _read_sidecar(path: Path, code: str) -> list[float] | list[int]:
    data = path.read_bytes()
    if not data or len(data) % 4:
        raise ValueError(f"{path} is not a nonempty 32-bit sidecar")
    count = len(data) // 4
    return list(struct.unpack("<" + code * count, data))


def _validated_sidecars(cell: dict[str, Any]) -> tuple[Path, list[float], list[int], dict[str, str]]:
    cell_path = _cell_path(cell)
    if not cell_path.is_file():
        raise ValueError(f"missing retained cell report: {cell_path}")
    raw = _read_json(cell_path)
    for key, value in raw.items():
        if cell.get(key) != value:
            raise ValueError(f"retained cell report disagrees with campaign field {key}: {cell_path}")
    nll_path = cell_path.with_suffix(".nllf32")
    argmax_path = cell_path.with_suffix(".argmaxi32")
    if not nll_path.is_file() or not argmax_path.is_file():
        raise ValueError(f"missing retained sidecars for {cell_path}")
    hashes = {
        "json": file_sha256(cell_path),
        "nllf32": file_sha256(nll_path),
        "argmaxi32": file_sha256(argmax_path),
    }
    if hashes["nllf32"] != cell.get("nll_sha256"):
        raise ValueError(f"NLL sidecar hash changed: {nll_path}")
    if hashes["argmaxi32"] != cell.get("argmax_sha256"):
        raise ValueError(f"argmax sidecar hash changed: {argmax_path}")
    nll = _read_sidecar(nll_path, "f")
    argmax = _read_sidecar(argmax_path, "i")
    expected = cell.get("tokens_scored")
    prompt_tokens = cell.get("prompt_tokens")
    skip_tokens = cell.get("skip_tokens")
    if (
        type(expected) is not int
        or expected <= 0
        or type(prompt_tokens) is not int
        or type(skip_tokens) is not int
        or expected != prompt_tokens - skip_tokens - 1
        or cell.get("argmax_tokens") != expected
        or cell.get("non_finite") != 0
        or len(nll) != expected
        or len(argmax) != expected
        or cell.get("complete_finite_aligned") is not True
        or not all(math.isfinite(value) for value in nll)
        or not all(0 <= token < TOKEN_DOMAIN for token in argmax)
    ):
        raise ValueError(f"cell sidecars are not complete, finite, and aligned: {cell_path}")
    return cell_path, nll, argmax, hashes


def _validate_cell_contract(
    cell: dict[str, Any], campaign: dict[str, Any], *, reference: bool
) -> None:
    tokens = cell.get("prompt_tokens")
    expected_skip = tokens // 2 if type(tokens) is int else None
    common = {
        "model_id": campaign.get("model_id"),
        "schedule": "prefill",
        "spec": campaign.get("spec"),
        "draft_tokens": campaign.get("draft_tokens"),
        "prefill_chunk": campaign.get("prefill_chunk"),
        "skip_tokens": expected_skip,
        "terrible_nll": campaign.get("terrible_nll"),
    }
    for field, expected in common.items():
        if cell.get(field) != expected:
            raise ValueError(
                f"{cell.get('scheme')}/{tokens} disagrees with the campaign contract at {field}"
            )
    if reference:
        expected = {
            "weights_id": campaign.get("reference_weights_id"),
            "kv_format": "bf16-reference",
        }
        source = campaign.get("reference_source")
        if not isinstance(source, dict):
            raise ValueError("campaign reference_source must be an object")
        source_fields = {
            "source_config_sha256": "config_sha256",
            "source_index_sha256": "index_sha256",
            "source_shards_sha256": "shards_sha256",
            "source_tensor_count": "tensor_count",
            "source_text_tensor_count": "text_tensor_count",
            "source_shard_count": "shard_count",
        }
        expected.update({cell_field: source.get(top_field) for cell_field, top_field in source_fields.items()})
        expected["execution_provenance"] = campaign.get("reference_execution")
    else:
        artifact = campaign.get("candidate_artifact")
        if not isinstance(artifact, dict):
            raise ValueError("campaign candidate_artifact must be an object")
        expected = {
            "weights_id": artifact.get("weights_id"),
            "kv_format": "fp8-k-int4-v",
            "kv_plane_layouts": campaign.get("candidate_kv_plane_layouts"),
            "q4_activation_bits": campaign.get("q4_activation_bits"),
            "w8_activation_bits": campaign.get("w8_activation_bits"),
            "fp8_qk_wmma_enabled": campaign.get("fp8_qk_wmma_enabled"),
            "fp8_qk_wmma_profile": campaign.get("fp8_qk_wmma_profile"),
            "fp8_qk_wmma_t1_min_context": campaign.get("fp8_qk_wmma_t1_min_context"),
            "fp8_qk_wmma_t2_min_context": campaign.get("fp8_qk_wmma_t2_min_context"),
        }
    for field, expected_value in expected.items():
        if cell.get(field) != expected_value:
            raise ValueError(
                f"{cell.get('scheme')}/{tokens} disagrees with the campaign contract at {field}"
            )


def _load_campaign(path: Path, expected_profile: str) -> dict[str, Any]:
    campaign = _read_json(path)
    if (
        campaign.get("artifact_type") != CAMPAIGN_TYPE
        or campaign.get("schema_version") != CAMPAIGN_SCHEMA
        or campaign.get("xattention_profile") != expected_profile
    ):
        raise ValueError(f"{path} is not a schema-v6 {expected_profile} campaign")
    if (
        campaign.get("lengths") != list(EXPECTED_LENGTHS)
        or campaign.get("schedules") != ["prefill"]
        or campaign.get("spec") != "none"
    ):
        raise ValueError(f"{path} is not the exact 8K/32K prefill-only workload")
    contract = campaign.get("quality_gate_contract", {})
    if (
        campaign.get("quality_tier") != QUALITY_TIER
        or campaign.get("gates") != EXPECTED_GATES
        or contract.get("tier") != QUALITY_TIER
        or contract.get("tier_maximum_mean_nll_delta") != QUALITY_GATE
        or contract.get("maximum_mean_nll_delta_by_profile") != EXPECTED_GATES
    ):
        raise ValueError(f"{path} does not retain the all-Q4 capacity-speed gate")
    scorers = campaign.get("scorers")
    if not isinstance(scorers, dict) or set(scorers) != {"bf16-reference", *CANDIDATES}:
        raise ValueError(f"{path} does not bind the exact scorer set")
    cells = campaign.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"{path} cells must be an array")
    return campaign


def _candidate_key(cell: dict[str, Any]) -> tuple[str, int]:
    scheme = cell.get("scheme")
    prompt_tokens = cell.get("prompt_tokens")
    if scheme not in CANDIDATES or prompt_tokens not in EXPECTED_LENGTHS:
        raise ValueError(f"unexpected candidate cell identity: {scheme}/{prompt_tokens}")
    expected_group = 16 if scheme == "r9700-g16" else 32
    if cell.get("kv_value_group") != expected_group:
        raise ValueError(f"{scheme}/{prompt_tokens} reports the wrong cache group")
    return scheme, prompt_tokens


def _candidate_cells(campaign: dict[str, Any], profile: str) -> dict[tuple[str, int], dict[str, Any]]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in campaign["cells"]:
        if not isinstance(cell, dict) or cell.get("scheme") == "bf16-reference":
            continue
        key = _candidate_key(cell)
        _validate_cell_contract(cell, campaign, reference=False)
        if key in selected:
            raise ValueError(f"duplicate candidate workload {key}")
        expected_qualification = profile != DENSE_PROFILE
        if cell.get("xattention_qualification") is not expected_qualification:
            raise ValueError(f"{key} does not report the expected {profile} qualification")
        if expected_qualification:
            expected = {
                "xattention_profile": XATTENTION_PROFILE,
                "xattention_find_block": 128,
                "xattention_stride": 16,
                "xattention_tau_permille": 900,
            }
            if any(cell.get(name) != value for name, value in expected.items()):
                raise ValueError(f"{key} does not report the exact S16/tau900 profile")
        elif any(name in cell for name in (
            "xattention_profile", "xattention_find_block", "xattention_stride",
            "xattention_tau_permille",
        )):
            raise ValueError(f"{key} dense report retains sparse-profile fields")
        selected[key] = cell
    expected_keys = {(profile, tokens) for profile in CANDIDATES for tokens in EXPECTED_LENGTHS}
    if set(selected) != expected_keys:
        raise ValueError("campaign does not contain exactly G16/G32 at 8K/32K")
    return selected


def _reference_cells(campaign: dict[str, Any]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for cell in campaign["cells"]:
        if not isinstance(cell, dict) or cell.get("scheme") != "bf16-reference":
            continue
        tokens = cell.get("prompt_tokens")
        if tokens not in EXPECTED_LENGTHS or tokens in selected:
            raise ValueError("campaign has an unexpected or duplicate BF16 workload")
        _validate_cell_contract(cell, campaign, reference=True)
        selected[tokens] = cell
    if set(selected) != set(EXPECTED_LENGTHS):
        raise ValueError("campaign does not contain exactly the 8K/32K BF16 workloads")
    return selected


def _aligned_workload(dense: dict[str, Any], sparse: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "scheme", "model_id", "weights_id", "kv_format", "kv_value_group",
        "kv_plane_layouts", "q4_activation_bits", "w8_activation_bits",
        "fp8_qk_wmma_enabled", "fp8_qk_wmma_profile", "fp8_qk_wmma_t1_min_context",
        "fp8_qk_wmma_t2_min_context", "schedule", "spec", "draft_tokens",
        "device_graph", "prefill_chunk", "skip_tokens", "prompt_tokens", "tokens_scored",
        "argmax_tokens", "terrible_nll",
    )
    for key in keys:
        if dense.get(key) != sparse.get(key):
            raise ValueError(f"dense/XAttention workload differs at {key}")
    return {key: dense.get(key) for key in keys}


def _route_stats(
    dense_nll: list[float], sparse_nll: list[float], dense_argmax: list[int],
    sparse_argmax: list[int], threshold: float,
) -> dict[str, Any]:
    if not (len(dense_nll) == len(sparse_nll) == len(dense_argmax) == len(sparse_argmax)):
        raise ValueError("dense/XAttention sidecars lost position alignment")
    deltas = [sparse - dense for dense, sparse in zip(dense_nll, sparse_nll)]
    absolute = [abs(value) for value in deltas]
    dense_severe = {i for i, value in enumerate(dense_nll) if value >= threshold}
    sparse_severe = {i for i, value in enumerate(sparse_nll) if value >= threshold}
    flips = [i for i, (dense, sparse) in enumerate(zip(dense_argmax, sparse_argmax)) if dense != sparse]
    return {
        "comparison_kind": "xattention-minus-dense",
        "positions_compared": len(deltas),
        "mean_delta_nll": sum(deltas) / len(deltas),
        "mean_abs_delta_nll": sum(absolute) / len(absolute),
        "max_abs_delta_nll": max(absolute),
        "dense_severe_positions": len(dense_severe),
        "xattention_severe_positions": len(sparse_severe),
        "persistent_severe_positions": len(dense_severe & sparse_severe),
        "new_xattention_severe_positions": len(sparse_severe - dense_severe),
        "repaired_dense_severe_positions": len(dense_severe - sparse_severe),
        "new_xattention_severe_position_indices": sorted(sparse_severe - dense_severe),
        "repaired_dense_severe_position_indices": sorted(dense_severe - sparse_severe),
        "argmax_mismatches": len(flips),
        "argmax_flip_rate": len(flips) / len(deltas),
        "argmax_first_mismatch": flips[0] if flips else None,
        "argmax_identity_is_gate": False,
    }


def compare_campaigns(dense_path: Path, sparse_path: Path) -> dict[str, Any]:
    dense = _load_campaign(dense_path, DENSE_PROFILE)
    sparse = _load_campaign(sparse_path, XATTENTION_PROFILE)
    for field in SHARED_FIELDS:
        if dense.get(field) != sparse.get(field):
            raise ValueError(f"campaign provenance/workload differs at {field}")
    if dense.get("weights_inputs") != sparse.get("weights_inputs"):
        raise ValueError("campaign candidate weight paths differ")
    if dense["scorers"]["bf16-reference"] != sparse["scorers"]["bf16-reference"]:
        raise ValueError("campaign BF16 scorer identity differs")
    for candidate in CANDIDATES:
        if dense["scorers"][candidate].get("sha256") == sparse["scorers"][candidate].get("sha256"):
            raise ValueError(f"{candidate} dense and XAttention scorer hashes are identical")

    dense_references = _reference_cells(dense)
    sparse_references = _reference_cells(sparse)
    reference_sources: dict[str, Any] = {}
    for tokens in EXPECTED_LENGTHS:
        dense_path_cell, _, _, dense_hashes = _validated_sidecars(dense_references[tokens])
        sparse_path_cell, _, _, sparse_hashes = _validated_sidecars(sparse_references[tokens])
        if dense_hashes["nllf32"] != sparse_hashes["nllf32"] or dense_hashes["argmaxi32"] != sparse_hashes["argmaxi32"]:
            raise ValueError(f"BF16 sidecars differ between campaigns at {tokens} tokens")
        reference_sources[str(tokens)] = {
            "dense_cell": str(dense_path_cell),
            "xattention_cell": str(sparse_path_cell),
            "nllf32_sha256": dense_hashes["nllf32"],
            "argmaxi32_sha256": dense_hashes["argmaxi32"],
        }

    dense_cells = _candidate_cells(dense, DENSE_PROFILE)
    sparse_cells = _candidate_cells(sparse, XATTENTION_PROFILE)
    comparisons = []
    threshold = float(dense["terrible_nll"])
    for key in sorted(dense_cells, key=lambda item: (item[1], item[0])):
        dense_cell = dense_cells[key]
        sparse_cell = sparse_cells[key]
        workload = _aligned_workload(dense_cell, sparse_cell)
        dense_cell_path, dense_nll, dense_argmax, dense_hashes = _validated_sidecars(dense_cell)
        sparse_cell_path, sparse_nll, sparse_argmax, sparse_hashes = _validated_sidecars(sparse_cell)
        dense_seconds = float(dense_cell.get("score_seconds", float("nan")))
        sparse_seconds = float(sparse_cell.get("score_seconds", float("nan")))
        if not math.isfinite(dense_seconds) or not math.isfinite(sparse_seconds) or min(dense_seconds, sparse_seconds) <= 0:
            raise ValueError(f"{key} has invalid scorer timing")
        comparisons.append({
            "profile": key[0],
            "prompt_tokens": key[1],
            "workload": workload,
            "dense_source": {"path": str(dense_cell_path), "sha256": dense_hashes},
            "xattention_source": {"path": str(sparse_cell_path), "sha256": sparse_hashes},
            "route_quality": _route_stats(
                dense_nll, sparse_nll, dense_argmax, sparse_argmax, threshold
            ),
            "timing": {
                "dense_score_seconds": dense_seconds,
                "xattention_score_seconds": sparse_seconds,
                "xattention_minus_dense_seconds": sparse_seconds - dense_seconds,
                "dense_over_xattention_speedup": dense_seconds / sparse_seconds,
                "scope": "teacher-forced-ppl-scorer-diagnostic-not-production-throughput",
            },
        })

    return {
        "artifact_type": OUTPUT_TYPE,
        "schema_version": OUTPUT_SCHEMA,
        "source_campaigns": {
            "dense": {"path": str(dense_path), "sha256": file_sha256(dense_path)},
            "xattention": {"path": str(sparse_path), "sha256": file_sha256(sparse_path)},
        },
        "shared_provenance": {field: dense.get(field) for field in SHARED_FIELDS},
        "scorers": {"dense": dense["scorers"], "xattention": sparse["scorers"]},
        "bf16_reference_sidecars": reference_sources,
        "dense_profile": DENSE_PROFILE,
        "xattention_profile": XATTENTION_PROFILE,
        "route_comparison_is_diagnostic": True,
        "admission_authority": "each source campaign's independent BF16 quality gates",
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--xattention", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_campaigns(args.dense, args.xattention)
    except (OSError, ValueError, json.JSONDecodeError, struct.error) as error:
        raise SystemExit(str(error)) from error
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.out.parent, prefix=args.out.name + ".",
        suffix=".tmp", delete=False,
    ) as output:
        json.dump(result, output, indent=2)
        output.write("\n")
        temporary = Path(output.name)
    os.replace(temporary, args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
