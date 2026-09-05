#!/usr/bin/env python3
"""Run the Qwen3.8-27B R9700 FP8-K/INT4-V accuracy campaign.

The campaign compares separately built G16 and G32 executables against an
independent BF16 reference executable. Both candidate executables consume the
same explicit R9700 artifact; their JSON reports prove which value-group constant
was compiled through the runtime. The campaign never changes the product's fixed
cache format through a runtime flag. Device Graphs default on. Every candidate
must meet finite/aligned, mean-NLL, and new-severe-position quality guardrails.
Greedy-token differences from BF16 are retained as a diagnostic because the
fixed lossy cache is not expected to reproduce BF16 argmax identity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.ppl.schemes import BASELINE, ORDER, PROFILES
from tools.convert.qwen3_8_27b_r9700 import fp8_hybrid_decision, fp8_hybrid_inventory
from tools.reference.qwen3_8_27b_bf16.protocol import (
    ATTENTION_PV_EXECUTION as BF16_ATTENTION_PV_EXECUTION,
    DETERMINISTIC_ENVIRONMENT as BF16_DETERMINISTIC_ENVIRONMENT,
    DETERMINISTIC_EXECUTION_PROFILE as BF16_DETERMINISTIC_EXECUTION_PROFILE,
    EXECUTION_ENVIRONMENT_KEYS as BF16_EXECUTION_ENVIRONMENT_KEY_ORDER,
    FORBIDDEN_EXECUTION_ENVIRONMENT as BF16_FORBIDDEN_EXECUTION_ENVIRONMENT,
    GDN_RECURRENCE_EXECUTION as BF16_GDN_RECURRENCE_EXECUTION,
    MATMUL_REDUCTION_EXECUTION as BF16_MATMUL_REDUCTION_EXECUTION,
    TRITON_CODEGEN_EXECUTION as BF16_TRITON_CODEGEN_EXECUTION,
)
DEFAULT_TOKENS = 8192
LONG_TOKENS = 32768
PAGE = 64
MID_PAGE_SKIP = DEFAULT_TOKENS // 2 + PAGE // 2  # 4128: decode starts mid-page
SHORT_TOKENS = 257
SHORT_SKIP = 1
SCHEDULES = ("prefill", "decode")
TERRIBLE_NLL = 10.0
DEFAULT_DRAFT_TOKENS = 3
MTP_EXTRA_DRAFT_TOKENS = 4
MODEL_ID = "qwen3.8-27b"
BF16_WEIGHTS_ID = "bf16-source"


def prepare_output_directory(out_dir: Path) -> None:
    """Create/reopen a real campaign directory without following an output symlink."""
    if os.path.lexists(out_dir) and out_dir.is_symlink():
        raise SystemExit(f"refusing symlinked PPL output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.is_symlink() or not out_dir.is_dir():
        raise SystemExit(f"PPL output path is not a real directory: {out_dir}")
BF16_REPEAT_ARTIFACT_TYPE = "ninfer_qwen3_8_bf16_repeat_comparison"
BF16_REPEAT_SCHEMA_VERSION = 1
BF16_EXECUTION_ENVIRONMENT_KEYS = set(BF16_EXECUTION_ENVIRONMENT_KEY_ORDER)
CAMPAIGN_ARTIFACT_TYPE = "ninfer_r9700_ppl_campaign"
CAMPAIGN_SCHEMA_VERSION = 6
NINFER_MAGIC = b"NINFER\x00\x02"
NINFER_PREFIX = struct.Struct("<8sQ")
MAX_DIRECTORY_BYTES = 64 * 1024 * 1024
TOKEN_DOMAIN = 248077
FP8_QK_WMMA_PROFILE = "t1-ge64-t2-ge320-t3plus-stream-v1"
FP8_QK_WMMA_T1_MIN_CONTEXT = 64
FP8_QK_WMMA_T2_MIN_CONTEXT = 320
XATTENTION_PROFILES = ("dense", "b128-s16-tau900")
R9700_KV_PLANE_LAYOUTS = {
    "key": "token-fastest-head-major",
    "value": "feature-fastest-page-major",
    "value_scale": "feature-fastest-page-major",
}
SOURCE_TENSOR_COUNT = 1199
SOURCE_TEXT_TENSOR_COUNT = 851
SOURCE_SHARD_COUNT = 18
DEFAULT_MAX_NEW_SEVERE_RATE = 0.001
DEFAULT_MIN_NEW_SEVERE_BUDGET = 4
QUALITY_TIERS = {
    "accuracy": {
        "maximum_mean_nll_delta": 0.02,
        "maximum_new_severe_rate": 0.001,
        "minimum_new_severe_budget": 4,
    },
    "capacity-speed": {
        "maximum_mean_nll_delta": math.log(1.05),
        "maximum_new_severe_rate": 0.0025,
        "minimum_new_severe_budget": 0,
    },
}


def default_reference_ppl_bin() -> Path | None:
    value = os.environ.get("NINFER_BF16_REFERENCE_PPL")
    return Path(value) if value else None


def default_group_ppl_bin(group: int) -> Path | None:
    value = os.environ.get(f"NINFER_R9700_G{group}_PPL")
    return Path(value) if value else None


def scorer_command_prefix(ppl_bin: Path) -> list[str]:
    """Launch Python scorers with the runner's provenance-bound interpreter."""

    if ppl_bin.suffix.lower() == ".py":
        return [sys.executable, str(ppl_bin)]
    return [str(ppl_bin)]


def preflight_python_scorer(ppl_bin: Path) -> None:
    """Reject a non-ROCm Python environment before campaign outputs exist."""

    if ppl_bin.suffix.lower() != ".py":
        return
    probe = (
        "import torch; "
        "assert getattr(torch.version, 'hip', None), "
        "'PyTorch is not a ROCm build'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise SystemExit(
            f"Python scorer requires a ROCm PyTorch environment under {sys.executable}{suffix}"
        )


def parse_gates(values: list[str]) -> dict[str, float]:
    gates: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"invalid --gate {item!r}; expected scheme=delta_mean_nll")
        name, raw = item.split("=", 1)
        if name not in PROFILES or name == BASELINE:
            raise SystemExit(f"unknown candidate profile in --gate: {name}")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise SystemExit(f"--gate for {name} must be finite and nonnegative")
        gates[name] = value
    return gates


def select_profiles(names: list[str] | None) -> list[str]:
    if not names:
        return list(ORDER)
    selected: list[str] = []
    for raw_name in names:
        name = raw_name.strip()
        if name not in PROFILES:
            raise SystemExit(f"unknown profile {name!r}; known: {', '.join(ORDER)}")
        if name not in selected:
            selected.append(name)
    if BASELINE not in selected:
        selected.insert(0, BASELINE)
    else:
        selected = [BASELINE] + [name for name in selected if name != BASELINE]
    return selected


def select_schedules(raw: str | None) -> list[str]:
    if not raw:
        return list(SCHEDULES)
    selected: list[str] = []
    for name in raw.split(","):
        name = name.strip()
        if name not in SCHEDULES:
            raise SystemExit(f"unknown schedule {name!r}; known: {', '.join(SCHEDULES)}")
        if name not in selected:
            selected.append(name)
    return selected


def validate_weights_input(profile_name: str, path: Path) -> None:
    """Require source-directory authority for BF16 and one file for candidates."""
    if profile_name == BASELINE:
        if not path.is_dir():
            raise SystemExit(
                f"complete BF16 source directory not found for {profile_name}: {path}"
            )
        return
    if not path.is_file():
        raise SystemExit(f"candidate artifact not found for {profile_name}: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SystemExit(f"hybrid conversion receipt {label} is not SHA-256")
    return value


def validate_hybrid_conversion_receipt(path: Path, artifact: dict) -> dict | None:
    """Bind the decision-owned hybrid bytes to their adjacent conversion receipt."""

    decision = fp8_hybrid_decision.DECISION
    if artifact["weights_id"] != decision.weights_id:
        return None
    receipt_path = Path(str(path.resolve()) + ".conversion.json")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"hybrid conversion receipt is missing: {receipt_path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"hybrid conversion receipt is invalid: {receipt_path}: {error}") from error
    if not isinstance(receipt, dict):
        raise SystemExit("hybrid conversion receipt root must be an object")
    expected = {
        "identity": {"model_id": MODEL_ID, "weights_id": decision.weights_id},
        "target_key": fp8_hybrid_inventory.TARGET_KEY,
        "recipe_id": decision.recipe_id,
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise SystemExit(f"hybrid conversion receipt {name} differs from its authority")
    candidate = receipt.get("candidate")
    if (
        not isinstance(candidate, dict)
        or candidate.get("status") != "registered-evaluation-only"
        or candidate.get("weight_recipe_selected") is not False
        or candidate.get("selection_sha256") != decision.selection_sha256
        or candidate.get("format_counts") != fp8_hybrid_inventory.FORMAT_COUNTS
        or candidate.get("format_encoded_bytes")
        != fp8_hybrid_inventory.FORMAT_ENCODED_BYTES
        or candidate.get("tensor_encoded_bytes")
        != fp8_hybrid_inventory.TENSOR_ENCODED_BYTES
        or candidate.get("device_arena_bytes")
        != fp8_hybrid_inventory.DEVICE_ARENA_BYTES
    ):
        raise SystemExit("hybrid conversion receipt candidate record differs")
    object_plan_sha256 = _require_sha256(
        candidate.get("object_plan_sha256"), "object_plan_sha256"
    )
    artifact_record = receipt.get("artifact")
    if (
        not isinstance(artifact_record, dict)
        or Path(str(artifact_record.get("path"))).resolve() != path.resolve()
        or artifact_record.get("bytes") != artifact["bytes"]
        or artifact_record.get("sha256") != artifact["sha256"]
    ):
        raise SystemExit("hybrid conversion receipt artifact record differs")
    source = receipt.get("source")
    if not isinstance(source, dict):
        raise SystemExit("hybrid conversion receipt source record is missing")
    index_sha256 = _require_sha256(source.get("index_sha256"), "source index_sha256")
    ranking_sha256 = _require_sha256(source.get("ranking_sha256"), "source ranking_sha256")
    return {
        "path": str(receipt_path),
        "sha256": file_sha256(receipt_path),
        "recipe_id": decision.recipe_id,
        "selection_sha256": decision.selection_sha256,
        "object_plan_sha256": object_plan_sha256,
        "source_index_sha256": index_sha256,
        "source_ranking_sha256": ranking_sha256,
    }


def inspect_candidate_artifact(path: Path, *, digest: str | None = None) -> dict:
    """Read the v2 identity and bind it to the exact candidate bytes."""

    resolved = path.resolve()
    with resolved.open("rb") as source:
        prefix = source.read(NINFER_PREFIX.size)
        if len(prefix) != NINFER_PREFIX.size:
            raise SystemExit(f"candidate artifact has a truncated prefix: {path}")
        magic, directory_bytes = NINFER_PREFIX.unpack(prefix)
        if magic != NINFER_MAGIC:
            raise SystemExit(f"candidate artifact is not NInfer v2: {path}")
        if directory_bytes == 0 or directory_bytes > MAX_DIRECTORY_BYTES:
            raise SystemExit(
                f"candidate artifact has invalid directory size {directory_bytes}: {path}"
            )
        try:
            directory = json.loads(source.read(directory_bytes))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"candidate artifact directory is invalid: {path}: {error}") from error
    identity = directory.get("identity") if isinstance(directory, dict) else None
    if not isinstance(identity, dict):
        raise SystemExit(f"candidate artifact has no identity: {path}")
    model_id = identity.get("model_id")
    weights_id = identity.get("weights_id")
    if model_id != MODEL_ID or not isinstance(weights_id, str) or not weights_id:
        raise SystemExit(
            f"candidate artifact identity must be {MODEL_ID}/<weights-id>, got "
            f"{model_id!r}/{weights_id!r}: {path}"
        )
    result = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest or file_sha256(resolved),
        "model_id": model_id,
        "weights_id": weights_id,
    }
    receipt = validate_hybrid_conversion_receipt(resolved, result)
    if receipt is not None:
        result["conversion_receipt"] = receipt
    return result


def require_same_candidate_artifact(g16: Path, g32: Path) -> dict:
    """G16/G32 may use different paths, but never different artifact bytes."""

    first = inspect_candidate_artifact(g16)
    try:
        same_file = g16.resolve().samefile(g32.resolve())
    except OSError:
        same_file = False
    second = first if same_file else inspect_candidate_artifact(g32)
    if (
        first["sha256"] != second["sha256"]
        or first["bytes"] != second["bytes"]
        or first["model_id"] != second["model_id"]
        or first["weights_id"] != second["weights_id"]
    ):
        raise SystemExit(
            "G16 and G32 must consume byte-identical candidate artifacts with one identity"
        )
    return {
        **first,
        "g16_path": str(g16.resolve()),
        "g32_path": str(g32.resolve()),
    }


def require_fp8_hybrid_candidate(artifact: dict | None) -> None:
    decision = fp8_hybrid_decision.DECISION
    if (
        not isinstance(artifact, dict)
        or artifact.get("model_id") != MODEL_ID
        or artifact.get("weights_id") != decision.weights_id
        or artifact.get("conversion_receipt", {}).get("recipe_id") != decision.recipe_id
        or artifact.get("conversion_receipt", {}).get("selection_sha256")
        != decision.selection_sha256
    ):
        raise SystemExit(
            "--require-fp8-hybrid needs the exact authority-bound hybrid artifact and receipt"
        )


def validate_corpus(path: Path, required_tokens: int) -> dict:
    manifest_path = path.with_name(path.stem + ".manifest.json")
    try:
        payload = path.read_bytes()
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
    except FileNotFoundError as error:
        raise SystemExit(f"PPL corpus or sibling manifest is missing: {error.filename}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"PPL corpus manifest is invalid: {manifest_path}: {error}") from error
    expected_manifest = {
        "artifact_type": "ninfer_ppl_corpus",
        "schema_version": 2,
        "model_id": MODEL_ID,
        "add_special_tokens": False,
        "chat_template": False,
    }
    if not isinstance(manifest, dict):
        raise SystemExit(f"PPL corpus manifest root must be an object: {manifest_path}")
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise SystemExit(
                f"PPL corpus manifest {key} must be {expected!r}, got {manifest.get(key)!r}"
            )
    words = payload.split()
    ids: list[int] = []
    for index, word in enumerate(words):
        if not word or any(byte < 48 or byte > 57 for byte in word):
            raise SystemExit(f"PPL corpus token {index} is not an ASCII decimal ID")
        token = int(word)
        if token >= TOKEN_DOMAIN:
            raise SystemExit(
                f"PPL corpus token {index} is outside public domain 0..{TOKEN_DOMAIN - 1}"
            )
        ids.append(token)
    declared = manifest.get("tokens")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared != len(ids):
        raise SystemExit(
            f"PPL corpus manifest tokens={declared!r} does not match parsed {len(ids)}"
        )
    digest = hashlib.sha256(payload).hexdigest()
    if manifest.get("ids_sha256") != digest:
        raise SystemExit("PPL corpus SHA-256 does not match its sibling manifest")
    if len(ids) < required_tokens:
        raise SystemExit(
            f"PPL corpus has {len(ids)} tokens, fewer than required {required_tokens}"
        )
    return {
        "path": str(path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "tokens": len(ids),
        "ids_sha256": digest,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "source": manifest.get("source"),
    }


def corpus_prefix_ids_sha256(path: Path, tokens: int) -> str:
    """Match the BF16 scorer's SHA-256 over selected little-endian I32 IDs."""

    words = path.read_bytes().split()
    if len(words) < tokens:
        raise SystemExit(f"PPL corpus has fewer than requested {tokens} tokens")
    digest = hashlib.sha256()
    for word in words[:tokens]:
        digest.update(struct.pack("<i", int(word)))
    return digest.hexdigest()


def corpus_token_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    return len(text.split())


def ensure_corpus(corpus: Path, tokens: int, weights: Path, ppl_bin: Path) -> None:
    if corpus.is_file() and corpus_token_count(corpus) >= tokens:
        return
    baker = Path(__file__).resolve().parent / "bake_corpus.py"
    cmd = [
        sys.executable,
        str(baker),
        "--weights",
        str(weights),
        "--tokens",
        str(tokens),
        "--out-dir",
        str(corpus.parent),
        "--ppl-bin",
        str(ppl_bin),
    ]
    print(f"baking corpus: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    if corpus_token_count(corpus) < tokens:
        raise SystemExit(f"baked corpus {corpus} has fewer than {tokens} tokens")


def load_nlls(cell_path: Path) -> list[float]:
    nll_path = cell_path.with_suffix(".nllf32")
    if not nll_path.is_file():
        return []
    data = nll_path.read_bytes()
    if len(data) % 4 != 0:
        raise SystemExit(f"malformed NLL sidecar (not f32-aligned): {nll_path}")
    count = len(data) // 4
    return list(struct.unpack("<" + "f" * count, data))


def load_argmax(cell_path: Path) -> list[int]:
    argmax_path = cell_path.with_suffix(".argmaxi32")
    if not argmax_path.is_file():
        return []
    data = argmax_path.read_bytes()
    if len(data) % 4 != 0:
        raise SystemExit(f"malformed argmax sidecar (not i32-aligned): {argmax_path}")
    count = len(data) // 4
    return list(struct.unpack("<" + "i" * count, data))


def exact_argmax_stats(candidate: list[int], reference: list[int]) -> dict:
    compared = min(len(candidate), len(reference))
    mismatch_positions = [i for i in range(compared) if candidate[i] != reference[i]]
    length_mismatch = len(candidate) != len(reference)
    first = mismatch_positions[0] if mismatch_positions else (compared if length_mismatch else None)
    mismatches = len(mismatch_positions) + abs(len(candidate) - len(reference))
    return {
        "argmax_exact": not mismatch_positions and not length_mismatch and compared > 0,
        "argmax_compared": compared,
        "argmax_mismatches": mismatches,
        "argmax_first_mismatch": first,
        "argmax_flip_rate": mismatches / max(len(candidate), len(reference), 1),
    }


def severe_position_stats(
    candidate: list[float],
    reference: list[float],
    *,
    threshold: float,
    maximum_new_rate: float,
    minimum_budget: int,
) -> dict:
    """Compare severe-NLL membership at aligned positions.

    The allowed count scales with the evaluated sample but has a small fixed
    floor so short execution-control cells do not define the long-corpus rate.
    Aggregate terrible-token deltas are insufficient because a candidate may
    repair one reference position while creating a different severe position.
    """

    if len(candidate) != len(reference):
        raise ValueError(
            f"paired NLL sidecars lost alignment: candidate={len(candidate)}, "
            f"baseline={len(reference)}"
        )
    if not candidate:
        raise ValueError("severe-position comparison requires nonempty NLL vectors")
    if not math.isfinite(threshold):
        raise ValueError("severe-position threshold must be finite")
    if not math.isfinite(maximum_new_rate) or maximum_new_rate < 0.0:
        raise ValueError("maximum new-severe rate must be finite and nonnegative")
    if type(minimum_budget) is not int or minimum_budget < 0:
        raise ValueError("minimum new-severe budget must be a nonnegative integer")

    candidate_positions = {i for i, value in enumerate(candidate) if value >= threshold}
    reference_positions = {i for i, value in enumerate(reference) if value >= threshold}
    new_positions = sorted(candidate_positions - reference_positions)
    repaired_positions = sorted(reference_positions - candidate_positions)
    persistent_positions = candidate_positions & reference_positions
    budget = max(minimum_budget, math.ceil(maximum_new_rate * len(candidate)))
    return {
        "severe_threshold_nll": threshold,
        "severe_positions_compared": len(candidate),
        "persistent_severe_positions": len(persistent_positions),
        "new_severe_positions": len(new_positions),
        "repaired_severe_positions": len(repaired_positions),
        "new_severe_position_rate": len(new_positions) / len(candidate),
        "maximum_new_severe_rate": maximum_new_rate,
        "minimum_new_severe_budget": minimum_budget,
        "new_severe_position_budget": budget,
        "new_severe_positions_pass": len(new_positions) <= budget,
        "new_severe_position_indices": new_positions,
        "repaired_severe_position_indices": repaired_positions,
    }


def paired_nll_stats(prefill: list[float], decode: list[float]) -> dict | None:
    if len(prefill) != len(decode):
        raise ValueError(
            f"paired NLL sidecars lost alignment: prefill={len(prefill)}, decode={len(decode)}"
        )
    n = len(prefill)
    if n == 0:
        return None
    if not all(math.isfinite(value) for value in (*prefill, *decode)):
        raise ValueError("paired NLL sidecars contain non-finite values")
    delta = [decode[i] - prefill[i] for i in range(n)]
    abs_delta = [abs(value) for value in delta]
    return {
        "tokens": n,
        "mean_delta_nll": sum(delta) / n,
        "mean_abs_delta_nll": sum(abs_delta) / n,
        "max_abs_delta_nll": max(abs_delta),
    }


def sidecar_parity(
    first_path: Path,
    second_path: Path,
    *,
    max_abs_nll: float,
) -> dict:
    if not math.isfinite(max_abs_nll) or max_abs_nll < 0.0:
        raise ValueError("sidecar parity threshold must be finite and nonnegative")
    first_nlls = load_nlls(first_path)
    second_nlls = load_nlls(second_path)
    nll = paired_nll_stats(first_nlls, second_nlls)
    if nll is None:
        raise ValueError("sidecar parity requires nonempty NLL vectors")
    first_argmax = load_argmax(first_path)
    second_argmax = load_argmax(second_path)
    if len(first_argmax) != nll["tokens"] or len(second_argmax) != nll["tokens"]:
        raise ValueError("sidecar parity sidecars are not position-aligned")
    if not all(0 <= token < TOKEN_DOMAIN for token in (*first_argmax, *second_argmax)):
        raise ValueError("sidecar parity argmax sidecar contains an invalid token")
    argmax = exact_argmax_stats(first_argmax, second_argmax)
    result = {
        **nll,
        **argmax,
        "comparison_kind": "same-route-execution",
        "complete_finite_aligned": True,
        "max_abs_nll_gate": max_abs_nll,
        "argmax_identity_is_gate": True,
    }
    result["pass"] = (
        result["argmax_exact"] and result["max_abs_delta_nll"] <= max_abs_nll
    )
    return result


def schedule_sidecar_comparison(
    prefill_path: Path,
    decode_path: Path,
    *,
    max_abs_nll: float,
) -> dict:
    """Compare two valid schedules without treating private-precision argmax flips as errors."""

    if not math.isfinite(max_abs_nll) or max_abs_nll < 0.0:
        raise ValueError("schedule comparison threshold must be finite and nonnegative")
    prefill_nlls = load_nlls(prefill_path)
    decode_nlls = load_nlls(decode_path)
    nll = paired_nll_stats(prefill_nlls, decode_nlls)
    if nll is None:
        raise ValueError("schedule comparison requires nonempty NLL vectors")
    prefill_argmax = load_argmax(prefill_path)
    decode_argmax = load_argmax(decode_path)
    if len(prefill_argmax) != nll["tokens"] or len(decode_argmax) != nll["tokens"]:
        raise ValueError("schedule comparison sidecars are not position-aligned")
    if not all(0 <= token < TOKEN_DOMAIN for token in (*prefill_argmax, *decode_argmax)):
        raise ValueError("schedule comparison argmax sidecar contains an invalid token")
    result = {
        **nll,
        **exact_argmax_stats(prefill_argmax, decode_argmax),
        "comparison_kind": "prefill-decode-schedule",
        "complete_finite_aligned": True,
        "max_abs_nll_gate": max_abs_nll,
        "argmax_identity_is_gate": False,
    }
    result["pass"] = result["max_abs_delta_nll"] <= max_abs_nll
    return result


def nll_std_se(nlls: list[float]) -> dict | None:
    """Per-token NLL spread: sample std and its SE (std/sqrt(n))."""
    n = len(nlls)
    if n < 2:
        return None
    mean = sum(nlls) / n
    var = sum((x - mean) ** 2 for x in nlls) / (n - 1)
    std = math.sqrt(var)
    return {"nll_std": std, "nll_se": std / math.sqrt(n), "nll_tokens": n}


def paired_delta_se(cell_nlls: list[float], base_nlls: list[float]) -> float | None:
    """SE of the per-token paired delta (cell_i - base_i) vs the group baseline.

    Both cells score the same positions of the same corpus, so index-aligned
    per-token deltas are the paired estimator: their mean equals
    cell.mean_nll - baseline.mean_nll exactly, and std/sqrt(n) is the delta's
    1-sigma. Much tighter than the independent-mean SEs of the two cells.
    """
    if len(cell_nlls) != len(base_nlls):
        raise ValueError(
            f"paired NLL sidecars lost alignment: candidate={len(cell_nlls)}, "
            f"baseline={len(base_nlls)}"
        )
    n = len(cell_nlls)
    if n < 2:
        return None
    deltas = [cell_nlls[i] - base_nlls[i] for i in range(n)]
    mean = sum(deltas) / n
    var = sum((x - mean) ** 2 for x in deltas) / (n - 1)
    return math.sqrt(var) / math.sqrt(n)


def paired_delta_summary(cell_nlls: list[float], base_nlls: list[float]) -> dict | None:
    """Absolute paired-delta percentiles and largest-magnitude signed token deltas."""
    if len(cell_nlls) != len(base_nlls):
        raise ValueError(
            f"paired NLL sidecars lost alignment: candidate={len(cell_nlls)}, "
            f"baseline={len(base_nlls)}"
        )
    if not cell_nlls:
        return None
    if not all(math.isfinite(value) for value in (*cell_nlls, *base_nlls)):
        raise ValueError("paired NLL sidecars contain non-finite values")
    signed = [cell_nlls[i] - base_nlls[i] for i in range(len(cell_nlls))]
    ordered = sorted(abs(value) for value in signed)

    def percentile(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    worst = sorted(range(len(signed)), key=lambda i: (-abs(signed[i]), i))[:10]
    return {
        "delta_abs_p50": percentile(0.50),
        "delta_abs_p95": percentile(0.95),
        "delta_abs_p99": percentile(0.99),
        "delta_abs_max": ordered[-1],
        "worst_delta_tokens": [
            # Position is the zero-based index in the aligned scored-position sidecars. The
            # enclosing cell's skip_tokens maps it back to the original prompt coordinate.
            {"position": i, "delta_nll": signed[i], "abs_delta_nll": abs(signed[i])}
            for i in worst
        ],
    }


def attach_nll_stats(cell: dict, nlls: list[float]) -> None:
    stats = nll_std_se(nlls)
    if stats:
        cell.update(stats)


def _extra_value(extra: list[str], flag: str) -> str | None:
    try:
        index = extra.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(extra):
        raise SystemExit(f"campaign generated {flag} without a value")
    return extra[index + 1]


def validate_cell_report(
    cell: dict,
    *,
    profile_name: str,
    weights: Path,
    ids: Path,
    expected_weights_id: str,
    schedule: str,
    skip: str,
    tokens: int,
    prefill_chunk: int,
    extra: list[str],
    expected_q4_activation_bits: int,
    expected_w8_activation_bits: int,
    expected_fp8_qk_wmma: bool,
    expected_xattention_profile: str = "dense",
) -> None:
    profile = PROFILES[profile_name]
    expected_format = "bf16-reference" if profile.reference else "fp8-k-int4-v"
    expected_graph = False if profile.reference else "--no-device-graph" not in extra
    expected_spec = _extra_value(extra, "--spec") or "none"
    draft_text = _extra_value(extra, "--draft-tokens")
    expected_draft = int(draft_text) if draft_text is not None else 0
    requested_skip = tokens // 2 if skip == "half" else int(skip)
    expected_skip = max(requested_skip, 1) if schedule == "decode" else requested_skip
    expected = {
        "scheme": profile.name,
        "model_id": MODEL_ID,
        "weights_id": expected_weights_id,
        "kv_format": expected_format,
        "kv_value_group": profile.value_group,
        "schedule": schedule,
        "spec": expected_spec,
        "draft_tokens": expected_draft,
        "device_graph": expected_graph,
        "prefill_chunk": prefill_chunk,
        "prompt_tokens": tokens,
        "skip_tokens": expected_skip,
        "terrible_nll": TERRIBLE_NLL,
    }
    for key, value in expected.items():
        if cell.get(key) != value:
            raise SystemExit(
                f"{profile_name} report {key}={cell.get(key)!r}; expected {value!r}"
            )
    if not profile.reference and cell.get("kv_plane_layouts") != R9700_KV_PLANE_LAYOUTS:
        raise SystemExit(
            f"{profile_name} report kv_plane_layouts={cell.get('kv_plane_layouts')!r}; "
            f"expected compiled {R9700_KV_PLANE_LAYOUTS!r}"
        )
    if not profile.reference and cell.get("q4_activation_bits") != expected_q4_activation_bits:
        raise SystemExit(
            f"{profile_name} report q4_activation_bits={cell.get('q4_activation_bits')!r}; "
            f"expected compiled A{expected_q4_activation_bits}"
        )
    if not profile.reference and cell.get("w8_activation_bits") != expected_w8_activation_bits:
        raise SystemExit(
            f"{profile_name} report w8_activation_bits={cell.get('w8_activation_bits')!r}; "
            f"expected compiled A{expected_w8_activation_bits}"
        )
    if not profile.reference and cell.get("fp8_qk_wmma_enabled") is not expected_fp8_qk_wmma:
        raise SystemExit(
            f"{profile_name} report fp8_qk_wmma_enabled="
            f"{cell.get('fp8_qk_wmma_enabled')!r}; expected "
            f"{expected_fp8_qk_wmma!r}"
        )
    if not profile.reference:
        expected_attention_profile = {
            "fp8_qk_wmma_profile": FP8_QK_WMMA_PROFILE,
            "fp8_qk_wmma_t1_min_context": FP8_QK_WMMA_T1_MIN_CONTEXT,
            "fp8_qk_wmma_t2_min_context": FP8_QK_WMMA_T2_MIN_CONTEXT,
        }
        for key, value in expected_attention_profile.items():
            if cell.get(key) != value:
                raise SystemExit(
                    f"{profile_name} report {key}={cell.get(key)!r}; expected {value!r}"
                )
        expected_xattention = (
            {"xattention_qualification": False}
            if expected_xattention_profile == "dense"
            else {
                "xattention_qualification": True,
                "xattention_profile": "b128-s16-tau900",
                "xattention_find_block": 128,
                "xattention_stride": 16,
                "xattention_tau_permille": 900,
            }
        )
        for key, value in expected_xattention.items():
            if cell.get(key) != value:
                raise SystemExit(
                    f"{profile_name} report {key}={cell.get(key)!r}; expected {value!r}"
                )
        if expected_xattention_profile == "dense":
            stale_keys = {
                "xattention_profile",
                "xattention_find_block",
                "xattention_stride",
                "xattention_tau_permille",
            }.intersection(cell)
            if stale_keys:
                raise SystemExit(
                    f"{profile_name} dense report retains XAttention fields: "
                    + ", ".join(sorted(stale_keys))
                )
    raw_weights = cell.get("weights")
    if not isinstance(raw_weights, str) or Path(raw_weights).resolve() != weights.resolve():
        raise SystemExit(
            f"{profile_name} report weights={raw_weights!r}; expected {str(weights)!r}"
        )
    required_integers = (
        "skip_tokens",
        "tokens_scored",
        "argmax_tokens",
        "non_finite",
        "terrible_tokens",
    )
    for key in required_integers:
        if type(cell.get(key)) is not int or cell[key] < 0:
            raise SystemExit(f"{profile_name} report {key} must be a nonnegative integer")
    required_numbers = ("sum_nll", "mean_nll", "max_nll", "ppl", "score_seconds")
    for key in required_numbers:
        value = cell.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise SystemExit(f"{profile_name} report {key} must be finite and numeric")
    if profile.reference:
        expected_reference = {
            "corpus_ids_sha256": corpus_prefix_ids_sha256(ids, tokens),
            "source_tensor_count": SOURCE_TENSOR_COUNT,
            "source_text_tensor_count": SOURCE_TEXT_TENSOR_COUNT,
            "source_shard_count": SOURCE_SHARD_COUNT,
        }
        for key, value in expected_reference.items():
            if cell.get(key) != value:
                raise SystemExit(
                    f"{profile_name} report {key}={cell.get(key)!r}; expected {value!r}"
                )
        for key in ("source_config_sha256", "source_index_sha256"):
            value = cell.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SystemExit(f"{profile_name} report {key} must be lowercase SHA-256")
        shard_hashes = cell.get("source_shards_sha256")
        expected_shards = {
            f"model-{part:05d}-of-{SOURCE_SHARD_COUNT:05d}.safetensors"
            for part in range(1, SOURCE_SHARD_COUNT + 1)
        }
        if not isinstance(shard_hashes, dict) or set(shard_hashes) != expected_shards:
            raise SystemExit(
                f"{profile_name} report source_shards_sha256 must name all source shards"
            )
        for name, value in shard_hashes.items():
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SystemExit(
                    f"{profile_name} report source shard {name} must be lowercase SHA-256"
                )
        validate_bf16_execution_provenance(cell.get("execution_provenance"))


def run_cell(
    ppl_bin: Path,
    weights: Path,
    ids: Path,
    profile_name: str,
    schedule: str,
    skip: str,
    tokens: int,
    prefill_chunk: int,
    device: int,
    cell_path: Path,
    extra: list[str],
    expected_weights_id: str,
    expected_q4_activation_bits: int = 8,
    expected_w8_activation_bits: int = 8,
    expected_fp8_qk_wmma: bool = True,
    expected_xattention_profile: str = "dense",
) -> dict:
    profile = PROFILES[profile_name]
    cmd = [
        *scorer_command_prefix(ppl_bin),
        "--weights",
        str(weights),
        "--ids",
        str(ids),
        "--scheme",
        profile.name,
        "--schedule",
        schedule,
        "--skip",
        skip,
        "--tokens",
        str(tokens),
        "--prefill-chunk",
        str(prefill_chunk),
        "--device",
        str(device),
        "--out-json",
        str(cell_path),
        *extra,
    ]
    print(f"running {cell_path.name}: {' '.join(cmd)}", file=sys.stderr)
    for stale in (
        cell_path,
        cell_path.with_suffix(".nllf32"),
        cell_path.with_suffix(".argmaxi32"),
    ):
        stale.unlink(missing_ok=True)
    subprocess.run(cmd, check=True)
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    if not isinstance(cell, dict):
        raise SystemExit(f"scorer report root must be an object: {cell_path}")
    validate_cell_report(
        cell,
        profile_name=profile_name,
        weights=weights,
        ids=ids,
        expected_weights_id=expected_weights_id,
        schedule=schedule,
        skip=skip,
        tokens=tokens,
        prefill_chunk=prefill_chunk,
        extra=extra,
        expected_q4_activation_bits=expected_q4_activation_bits,
        expected_w8_activation_bits=expected_w8_activation_bits,
        expected_fp8_qk_wmma=expected_fp8_qk_wmma,
        expected_xattention_profile=expected_xattention_profile,
    )
    nll_path = cell_path.with_suffix(".nllf32")
    argmax_path = cell_path.with_suffix(".argmaxi32")
    if not nll_path.is_file() or not argmax_path.is_file():
        raise SystemExit(f"scorer did not publish both sidecars for {cell_path}")
    cell["command"] = cmd
    cell["nll_sha256"] = file_sha256(nll_path)
    cell["argmax_sha256"] = file_sha256(argmax_path)
    return cell


def _command_output_path(cell: dict, *, evidence: str = "BF16") -> Path:
    command = cell.get("command")
    if not isinstance(command, list) or command.count("--out-json") != 1:
        raise SystemExit(
            f"reused {evidence} cell must retain one --out-json command argument"
        )
    index = command.index("--out-json")
    if index + 1 >= len(command) or not isinstance(command[index + 1], str):
        raise SystemExit(
            f"reused {evidence} cell has a malformed --out-json command argument"
        )
    path = Path(command[index + 1])
    return path if path.is_absolute() else REPO / path


BF16_SCORER_REPORT_FIELDS = (
    "scheme", "model_id", "weights_id", "weight_format", "formula_profile",
    "source_config_sha256", "source_index_sha256", "source_shards_sha256",
    "corpus_ids_sha256", "source_tensor_count", "source_text_tensor_count",
    "source_shard_count", "execution_provenance", "kv_format", "kv_value_group",
    "cache_only_diagnostic", "cache_diagnostic_scope", "cache_key_codec",
    "cache_value_codec", "cache_append_boundary", "cache_use_boundary", "schedule",
    "spec", "draft_tokens", "speculative_execution", "device_graph", "prefill_chunk",
    "prompt_tokens", "skip_tokens", "tokens_scored", "argmax_tokens", "non_finite",
    "terrible_tokens", "terrible_nll", "sum_nll", "mean_nll", "max_nll", "ppl",
)
CANDIDATE_SCORER_REPORT_FIELDS = (
    "scheme", "model_id", "weights_id", "weights", "kv_format", "kv_value_group",
    "kv_plane_layouts", "schedule", "spec", "draft_tokens", "device_graph",
    "prefill_chunk", "prompt_tokens", "skip_tokens", "terrible_nll", "tokens_scored",
    "argmax_tokens", "non_finite", "terrible_tokens", "sum_nll", "mean_nll",
    "max_nll", "ppl", "score_seconds", "q4_activation_bits", "w8_activation_bits",
    "fp8_qk_wmma_enabled", "fp8_qk_wmma_profile", "fp8_qk_wmma_t1_min_context",
    "fp8_qk_wmma_t2_min_context", "xattention_qualification",
)


def validate_bf16_execution_provenance(value: object) -> str:
    """Validate and canonically serialize the deterministic reference execution identity."""
    if not isinstance(value, dict):
        raise SystemExit("BF16 report execution_provenance must be an object")
    expected = {
        "profile": BF16_DETERMINISTIC_EXECUTION_PROFILE,
        "preferred_blas_library": "hipblas",
        "deterministic_algorithms": {"enabled": True, "warn_only": False},
        "stage_trace_enabled": False,
        "attention_pv": BF16_ATTENTION_PV_EXECUTION,
        "gdn_recurrence": BF16_GDN_RECURRENCE_EXECUTION,
        "tunable_op": {"enabled": False},
        "triton_codegen": BF16_TRITON_CODEGEN_EXECUTION,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise SystemExit(
                f"BF16 report execution_provenance {key}={value.get(key)!r}; expected {wanted!r}"
            )
    environment = value.get("environment")
    if not isinstance(environment, dict) or set(environment) != BF16_EXECUTION_ENVIRONMENT_KEYS:
        raise SystemExit("BF16 report execution_provenance environment inventory differs")
    if any(environment.get(key) != wanted
           for key, wanted in BF16_DETERMINISTIC_ENVIRONMENT.items()):
        raise SystemExit("BF16 report does not bind the required deterministic execution environment")
    if any(environment.get(key) is not None
           for key in BF16_FORBIDDEN_EXECUTION_ENVIRONMENT):
        raise SystemExit("BF16 report contains a forbidden execution override")
    required_values = (
        "python", "python_executable", "platform", "torch", "torch_git", "hip",
        "device_name", "device_arch", "matmul_precision",
    )
    if any(not isinstance(value.get(key), str) or not value[key] for key in required_values):
        raise SystemExit("BF16 report execution_provenance lacks a runtime/device identity")
    if not value["device_arch"].startswith("gfx1201"):
        raise SystemExit("BF16 report execution_provenance is not for gfx1201")
    if type(value.get("device_index")) is not int or value["device_index"] < 0:
        raise SystemExit("BF16 report execution_provenance has an invalid device index")
    if value.get("matmul_precision") != "highest":
        raise SystemExit("BF16 report execution_provenance requires highest matmul precision")
    for key in (
        "python_executable_sha256", "scorer_python_tree_sha256", "fla_python_tree_sha256"
    ):
        digest = value.get(key)
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SystemExit(f"BF16 report execution_provenance {key} is not SHA-256")
    reductions = value.get("matmul_reduction")
    if reductions != BF16_MATMUL_REDUCTION_EXECUTION:
        raise SystemExit("BF16 report execution_provenance has noncanonical matmul reductions")
    distributions = value.get("distributions")
    if not isinstance(distributions, dict) or set(distributions) != {
        "torch", "triton", "flash-linear-attention", "safetensors"
    }:
        raise SystemExit("BF16 report execution_provenance lacks exact package records")
    for name, record in distributions.items():
        if not isinstance(record, dict) or set(record) != {
            "version", "record_sha256", "direct_url_sha256"
        }:
            raise SystemExit(f"BF16 report package record {name!r} is malformed")
        if record["version"] is not None and not isinstance(record["version"], str):
            raise SystemExit(f"BF16 report package record {name!r} has an invalid version")
        for hash_key in ("record_sha256", "direct_url_sha256"):
            digest = record[hash_key]
            if digest is not None and (
                not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise SystemExit(
                    f"BF16 report package record {name!r} has an invalid {hash_key}"
                )
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_bf16_repeat_comparison(
    comparison_path: Path, authority_campaign_path: Path,
) -> dict[str, object]:
    """Bind one exact A/B comparison proving the reused BF16 campaign is an authority input."""

    comparison_path = comparison_path.resolve(strict=True)
    authority_campaign_path = authority_campaign_path.resolve(strict=True)
    try:
        value = json.loads(comparison_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("BF16 repeat comparison is not valid JSON") from error
    rows = value.get("rows") if isinstance(value, dict) else None
    inputs = value.get("inputs") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("artifact_type") != BF16_REPEAT_ARTIFACT_TYPE
        or value.get("schema_version") != BF16_REPEAT_SCHEMA_VERSION
        or value.get("quality_evidence") is not False
        or value.get("exact") is not True
        or not isinstance(inputs, dict)
        or set(inputs) != {"first", "second"}
        or not isinstance(rows, list)
        or len(rows) != 2
    ):
        raise SystemExit("BF16 repeat comparison is not an exact schema-v1 authority")
    matched_input = None
    input_paths = set()
    resolved_inputs = {}
    for label in ("first", "second"):
        binding = inputs[label]
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding.get("path"), str)
            or not binding["path"]
        ):
            raise SystemExit("BF16 repeat comparison has a malformed campaign input")
        path = Path(binding["path"]).resolve(strict=True)
        digest = binding.get("sha256")
        if digest != file_sha256(path):
            raise SystemExit("BF16 repeat comparison campaign input bytes changed")
        input_paths.add(path)
        resolved_inputs[label] = path
        if path == authority_campaign_path:
            matched_input = label
    if len(input_paths) != 2:
        raise SystemExit("BF16 repeat comparison must bind two distinct fresh campaign paths")
    if matched_input is None:
        raise SystemExit("reused BF16 campaign is not an input to the exact repeat comparison")
    expected_tokens = {DEFAULT_TOKENS, LONG_TOKENS}
    actual_tokens = set()
    for row in rows:
        hashes_valid = True
        if isinstance(row, dict):
            for key in ("nll_sha256", "argmax_sha256"):
                hashes = row.get(key)
                hashes_valid = hashes_valid and isinstance(hashes, dict) and set(hashes) == {
                    "first", "second"
                } and hashes["first"] == hashes["second"] and isinstance(
                    hashes["first"], str
                ) and len(hashes["first"]) == 64 and all(
                    character in "0123456789abcdef" for character in hashes["first"]
                )
        if (
            not isinstance(row, dict)
            or type(row.get("prompt_tokens")) is not int
            or row.get("exact") is not True
            or row.get("semantic_fields_exact") is not True
            or not isinstance(row.get("semantic_fields"), dict)
            or not hashes_valid
        ):
            raise SystemExit("BF16 repeat comparison has a non-exact or malformed row")
        actual_tokens.add(row["prompt_tokens"])
    if actual_tokens != expected_tokens:
        raise SystemExit("BF16 repeat comparison lacks the exact 8K/32K row inventory")
    # Do not trust retained pass booleans: reopen both schema-v6 campaigns and recompute the
    # semantic/sidecar comparison through the same strict owner that created this report.
    from tools.ppl import compare_bf16_repeats as repeat_comparator
    try:
        first_payload, first_cells = repeat_comparator.load_campaign(resolved_inputs["first"])
        second_payload, second_cells = repeat_comparator.load_campaign(resolved_inputs["second"])
        recomputed = repeat_comparator.compare(
            first_payload, first_cells, second_payload, second_cells
        )
    except (OSError, ValueError) as error:
        raise SystemExit("BF16 repeat comparison inputs do not revalidate") from error
    if recomputed.get("exact") is not True or value.get("rows") != recomputed.get("rows"):
        raise SystemExit("BF16 repeat comparison does not recompute exactly from its inputs")
    return {
        "path": str(comparison_path),
        "sha256": file_sha256(comparison_path),
        "authority_input": matched_input,
        "authority_campaign_sha256": file_sha256(authority_campaign_path),
    }


def load_reused_bf16_cells(
    campaign_path: Path,
    *,
    bf16_weights: Path,
    bf16_scorer: Path,
    scorer_identity: dict,
    ids: Path,
    corpus_provenance: dict,
    lengths: list[int],
    skip: str,
    prefill_chunk: int,
    device: int,
) -> dict[int, tuple[dict, Path]]:
    """Validate and import only provenance-matched BF16 prefill cells."""

    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if not isinstance(campaign, dict):
        raise SystemExit("reused BF16 campaign root must be an object")
    expected_top = {
        "artifact_type": CAMPAIGN_ARTIFACT_TYPE,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "reference_weights_id": BF16_WEIGHTS_ID,
        "corpus": corpus_provenance,
        "skip": skip,
        "prefill_chunk": prefill_chunk,
        "schedules": ["prefill"],
        "spec": "none",
        "draft_tokens": 0,
        "terrible_nll": TERRIBLE_NLL,
    }
    for key, expected in expected_top.items():
        if campaign.get(key) != expected:
            raise SystemExit(
                f"reused BF16 campaign {key}={campaign.get(key)!r}; expected {expected!r}"
            )
    campaign_lengths = campaign.get("lengths")
    if (
        not isinstance(campaign_lengths, list)
        or any(type(value) is not int for value in campaign_lengths)
        or len(campaign_lengths) != len(set(campaign_lengths))
        or not set(lengths) <= set(campaign_lengths)
    ):
        raise SystemExit("reused BF16 campaign does not contain every requested length")
    source_weights = campaign.get("weights_inputs", {}).get(BASELINE)
    if not isinstance(source_weights, str) or Path(source_weights).resolve() != bf16_weights.resolve():
        raise SystemExit("reused BF16 campaign source path differs from the requested source")
    scorers = campaign.get("scorers")
    if not isinstance(scorers, dict) or scorers.get(BASELINE) != scorer_identity:
        raise SystemExit("reused BF16 campaign scorer identity differs")

    cells = campaign.get("cells")
    if not isinstance(cells, list):
        raise SystemExit("reused BF16 campaign cells must be an array")
    selected: dict[int, tuple[dict, Path]] = {}
    source_identities = set()
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("scheme") != BASELINE:
            continue
        tokens = cell.get("prompt_tokens")
        if tokens not in lengths:
            continue
        if tokens in selected:
            raise SystemExit("reused BF16 campaign has a duplicate requested reference cell")
        validate_cell_report(
            cell,
            profile_name=BASELINE,
            weights=bf16_weights,
            ids=ids,
            expected_weights_id=BF16_WEIGHTS_ID,
            schedule="prefill",
            skip=skip,
            tokens=tokens,
            prefill_chunk=prefill_chunk,
            extra=[],
            expected_q4_activation_bits=8,
            expected_w8_activation_bits=8,
            expected_fp8_qk_wmma=True,
            expected_xattention_profile="dense",
        )
        execution_json = validate_bf16_execution_provenance(cell.get("execution_provenance"))
        if cell["execution_provenance"]["device_index"] != device:
            raise SystemExit("reused BF16 execution device differs from the requested device")
        cell_path = _command_output_path(cell)
        command = cell.get("command")
        expected_arguments = [
            "--weights", str(bf16_weights), "--ids", str(ids),
            "--scheme", BASELINE, "--schedule", "prefill", "--skip", skip,
            "--tokens", str(tokens), "--prefill-chunk", str(prefill_chunk),
            "--device", str(device), "--out-json", str(cell_path),
        ]
        if not isinstance(command, list) or any(not isinstance(value, str) for value in command):
            raise SystemExit(f"reused BF16 cell command differs from the exact gate: {cell_path}")
        if command and Path(command[0]).resolve() == bf16_scorer.resolve():
            # Historical schema-v6 campaigns used the scorer's env-python shebang.
            arguments = command[1:]
        elif (
            len(command) >= 2
            and Path(command[1]).resolve() == bf16_scorer.resolve()
            and Path(command[0]).name.startswith("python")
            and isinstance(cell.get("execution_provenance"), dict)
            and isinstance(cell["execution_provenance"].get("python_executable"), str)
            and Path(command[0]).resolve()
            == Path(cell["execution_provenance"]["python_executable"]).resolve()
        ):
            arguments = command[2:]
        else:
            raise SystemExit(f"reused BF16 cell command differs from the exact gate: {cell_path}")
        if len(arguments) != len(expected_arguments):
            raise SystemExit(f"reused BF16 cell command differs from the exact gate: {cell_path}")
        path_value_indices = {1, 3, len(arguments) - 1}
        for index, (actual, expected) in enumerate(zip(arguments, expected_arguments)):
            equal = (
                Path(actual).resolve() == Path(expected).resolve()
                if index in path_value_indices
                else actual == expected
            )
            if not equal:
                raise SystemExit(
                    f"reused BF16 cell command differs from the exact gate: {cell_path}"
                )
        if not cell_path.is_file():
            raise SystemExit(f"reused BF16 cell report is missing: {cell_path}")
        raw = json.loads(cell_path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or any(key not in raw for key in BF16_SCORER_REPORT_FIELDS)
            or any(raw[key] != cell.get(key) for key in BF16_SCORER_REPORT_FIELDS)
        ):
            raise SystemExit(
                f"reused BF16 raw scorer fields differ from its campaign: {cell_path}"
            )
        nll_path = cell_path.with_suffix(".nllf32")
        argmax_path = cell_path.with_suffix(".argmaxi32")
        if (
            not nll_path.is_file()
            or not argmax_path.is_file()
            or file_sha256(nll_path) != cell.get("nll_sha256")
            or file_sha256(argmax_path) != cell.get("argmax_sha256")
        ):
            raise SystemExit(f"reused BF16 sidecar hashes do not match: {cell_path}")
        nlls = load_nlls(cell_path)
        argmax = load_argmax(cell_path)
        if not cell_ok(cell, nlls, argmax):
            raise SystemExit(f"reused BF16 cell is not complete, finite, and aligned: {cell_path}")
        source_identities.add((
            cell.get("source_config_sha256"),
            cell.get("source_index_sha256"),
            json.dumps(cell.get("source_shards_sha256"), sort_keys=True),
            cell.get("source_tensor_count"),
            cell.get("source_text_tensor_count"),
            cell.get("source_shard_count"),
            execution_json,
        ))
        reused = copy.deepcopy(cell)
        reused["reused_bf16_campaign"] = {
            "path": str(campaign_path), "sha256": file_sha256(campaign_path)
        }
        selected[tokens] = reused, cell_path
    if set(selected) != set(lengths):
        raise SystemExit("reused BF16 campaign lacks the exact requested reference workloads")
    if len(source_identities) != 1:
        raise SystemExit("reused BF16 source identity changes across cells")
    source = next(iter(source_identities))
    expected_source = campaign.get("reference_source", {})
    actual_source = {
        "config_sha256": source[0],
        "index_sha256": source[1],
        "shards_sha256": json.loads(source[2]),
        "tensor_count": source[3],
        "text_tensor_count": source[4],
        "shard_count": source[5],
    }
    if expected_source != actual_source:
        raise SystemExit("reused BF16 source hashes/counts disagree with its campaign cells")
    if campaign.get("reference_execution") != json.loads(source[6]):
        raise SystemExit("reused BF16 execution provenance disagrees with its campaign cells")
    return selected


def load_reused_candidate_cells(
    campaign_path: Path,
    *,
    selected_candidates: list[str],
    candidate_artifact: dict,
    profile_weights: dict[str, Path],
    profile_bins: dict[str, Path],
    scorer_provenance: dict[str, dict],
    ids: Path,
    corpus_provenance: dict,
    lengths: list[int],
    skip: str,
    prefill_chunk: int,
    device: int,
    expected_q4_activation_bits: int,
    expected_w8_activation_bits: int,
    expected_fp8_qk_wmma: bool,
    expected_xattention_profile: str,
    allow_partial: bool = False,
) -> dict[tuple[str, int], tuple[dict, Path]]:
    """Import unchanged candidate sidecars while recomputing BF16-relative gates.

    This deliberately accepts only primary prefill cells. Execution variants and sparse/dense
    profile changes require fresh product execution rather than an offline re-label.
    """

    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if not isinstance(campaign, dict):
        raise SystemExit("reused candidate campaign root must be an object")
    source_lengths = campaign.get("lengths")
    if allow_partial:
        if (
            not isinstance(source_lengths, list)
            or not source_lengths
            or any(type(tokens) is not int or tokens not in lengths for tokens in source_lengths)
            or len(set(source_lengths)) != len(source_lengths)
        ):
            raise SystemExit(
                "partial reused candidate campaign lengths are not a unique requested subset"
            )
    else:
        source_lengths = lengths
    expected_top = {
        "artifact_type": CAMPAIGN_ARTIFACT_TYPE,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "corpus": corpus_provenance,
        "lengths": source_lengths,
        "skip": skip,
        "prefill_chunk": prefill_chunk,
        "schedules": ["prefill"],
        "spec": "none",
        "draft_tokens": 0,
        "terrible_nll": TERRIBLE_NLL,
        "q4_activation_bits": expected_q4_activation_bits,
        "w8_activation_bits": expected_w8_activation_bits,
        "candidate_kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
        "fp8_qk_wmma_enabled": expected_fp8_qk_wmma,
        "fp8_qk_wmma_profile": FP8_QK_WMMA_PROFILE,
        "fp8_qk_wmma_t1_min_context": FP8_QK_WMMA_T1_MIN_CONTEXT,
        "fp8_qk_wmma_t2_min_context": FP8_QK_WMMA_T2_MIN_CONTEXT,
        "xattention_profile": expected_xattention_profile,
    }
    for key, expected in expected_top.items():
        if campaign.get(key) != expected:
            raise SystemExit(
                f"reused candidate campaign {key}={campaign.get(key)!r}; expected {expected!r}"
            )
    source_artifact = campaign.get("candidate_artifact")
    if allow_partial:
        def portable_artifact(value: object) -> object:
            if not isinstance(value, dict):
                return value
            return {
                key: item for key, item in value.items()
                if key not in {"g16_path", "g32_path"}
            }

        if portable_artifact(source_artifact) != portable_artifact(candidate_artifact):
            raise SystemExit("partial reused candidate campaign artifact identity differs")
    elif source_artifact != candidate_artifact:
        raise SystemExit("reused candidate campaign candidate_artifact identity differs")

    weights_inputs = campaign.get("weights_inputs")
    scorers = campaign.get("scorers")
    if not isinstance(weights_inputs, dict) or not isinstance(scorers, dict):
        raise SystemExit("reused candidate campaign lacks weights/scorer identities")
    source_candidates = [
        name for name in selected_candidates
        if name in weights_inputs or name in scorers
    ]
    if not source_candidates:
        raise SystemExit("reused candidate campaign has no requested candidate profile")
    if not allow_partial and source_candidates != selected_candidates:
        raise SystemExit("reused candidate campaign lacks the complete requested profile matrix")
    unexpected_candidates = (
        (set(weights_inputs) | set(scorers)) & {"r9700-g16", "r9700-g32"}
    ) - set(selected_candidates)
    if unexpected_candidates:
        raise SystemExit("reused candidate campaign contains an unrequested candidate profile")
    for name in source_candidates:
        source_weights = weights_inputs.get(name)
        if (
            not isinstance(source_weights, str)
            or Path(source_weights).resolve() != profile_weights[name].resolve()
        ):
            raise SystemExit(f"reused candidate campaign {name} artifact path differs")
        if scorers.get(name) != scorer_provenance[name]:
            raise SystemExit(f"reused candidate campaign {name} scorer identity differs")

    cells = campaign.get("cells")
    if not isinstance(cells, list):
        raise SystemExit("reused candidate campaign cells must be an array")
    selected: dict[tuple[str, int], tuple[dict, Path]] = {}
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("scheme") not in source_candidates:
            continue
        name = cell["scheme"]
        tokens = cell.get("prompt_tokens")
        key = (name, tokens)
        if tokens not in source_lengths or key in selected:
            raise SystemExit("reused candidate campaign has an unexpected or duplicate cell")
        validate_cell_report(
            cell,
            profile_name=name,
            weights=profile_weights[name],
            ids=ids,
            expected_weights_id=candidate_artifact["weights_id"],
            schedule="prefill",
            skip=skip,
            tokens=tokens,
            prefill_chunk=prefill_chunk,
            extra=[],
            expected_q4_activation_bits=expected_q4_activation_bits,
            expected_w8_activation_bits=expected_w8_activation_bits,
            expected_fp8_qk_wmma=expected_fp8_qk_wmma,
            expected_xattention_profile=expected_xattention_profile,
        )
        cell_path = _command_output_path(cell, evidence="candidate")
        expected_command = [
            str(profile_bins[name]), "--weights", str(profile_weights[name]),
            "--ids", str(ids), "--scheme", name, "--schedule", "prefill",
            "--skip", skip, "--tokens", str(tokens), "--prefill-chunk",
            str(prefill_chunk), "--device", str(device), "--out-json",
            cell["command"][cell["command"].index("--out-json") + 1],
        ]
        command = cell.get("command")
        if command != expected_command:
            raise SystemExit(f"reused candidate cell command differs from the exact gate: {cell_path}")
        if not cell_path.is_file():
            raise SystemExit(f"reused candidate cell report is missing: {cell_path}")
        raw = json.loads(cell_path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or any(field not in raw for field in CANDIDATE_SCORER_REPORT_FIELDS)
            or any(cell.get(field) != value for field, value in raw.items())
        ):
            raise SystemExit(
                f"reused candidate raw scorer fields differ from its campaign: {cell_path}"
            )
        nll_path = cell_path.with_suffix(".nllf32")
        argmax_path = cell_path.with_suffix(".argmaxi32")
        if (
            not nll_path.is_file()
            or not argmax_path.is_file()
            or file_sha256(nll_path) != cell.get("nll_sha256")
            or file_sha256(argmax_path) != cell.get("argmax_sha256")
        ):
            raise SystemExit(f"reused candidate sidecar hashes do not match: {cell_path}")
        nlls = load_nlls(cell_path)
        argmax = load_argmax(cell_path)
        if not cell_ok(cell, nlls, argmax):
            raise SystemExit(
                f"reused candidate cell is not complete, finite, and aligned: {cell_path}"
            )
        reused = copy.deepcopy(cell)
        # Remove every BF16-derived field. The main assembly path recomputes them against the
        # newly selected deterministic authority from the retained raw candidate sidecars.
        for field in (
            "argmax_compared", "argmax_exact", "argmax_first_mismatch",
            "argmax_flip_rate", "argmax_identity_is_gate", "argmax_mismatches",
            "delta_abs_max", "delta_abs_p50", "delta_abs_p95", "delta_abs_p99",
            "delta_mean_nll", "delta_nll_se", "gate", "in_noise",
            "maximum_new_severe_rate", "minimum_new_severe_budget",
            "new_severe_position_budget", "new_severe_position_indices",
            "new_severe_position_rate", "new_severe_positions",
            "new_severe_positions_pass", "pass", "persistent_severe_positions",
            "quality_eligible", "quality_tier", "repaired_severe_position_indices",
            "repaired_severe_positions", "severe_positions_compared",
            "severe_threshold_nll", "terrible_baseline", "terrible_delta",
            "worst_delta_tokens",
        ):
            reused.pop(field, None)
        reused["reused_candidate_campaign"] = {
            "path": str(campaign_path), "sha256": file_sha256(campaign_path)
        }
        selected[key] = reused, cell_path
    expected_keys = {
        (name, tokens) for name in source_candidates for tokens in source_lengths
    }
    if set(selected) != expected_keys:
        raise SystemExit("reused candidate campaign lacks the complete requested profile matrix")
    return selected


def expected_tokens_scored(cell: dict) -> int:
    """Prefill: n - skip - 1. Decode rewrites skip_tokens to prefix, so the same formula."""
    n = int(cell.get("prompt_tokens", 0))
    skip = int(cell.get("skip_tokens", 0))
    return n - skip - 1


def cell_ok(cell: dict, cell_nlls: list[float], cell_argmax: list[int]) -> bool:
    expected = expected_tokens_scored(cell)
    return (
        cell.get("non_finite", 1) == 0
        and math.isfinite(float(cell.get("mean_nll", float("nan"))))
        and expected > 0
        and int(cell.get("tokens_scored", 0)) == expected
        and int(cell.get("argmax_tokens", 0)) == expected
        and len(cell_nlls) == expected
        and len(cell_argmax) == expected
        and all(math.isfinite(value) for value in cell_nlls)
        and all(0 <= token < TOKEN_DOMAIN for token in cell_argmax)
    )


def apply_baseline(cell: dict, cell_nlls: list[float], baseline_nll: float | None,
                   gates: dict[str, float], name: str,
                   base_nlls: list[float] | None, cell_argmax: list[int],
                   base_argmax: list[int] | None,
                   baseline_terrible_tokens: int | None,
                   maximum_new_severe_rate: float = DEFAULT_MAX_NEW_SEVERE_RATE,
                   minimum_new_severe_budget: int = DEFAULT_MIN_NEW_SEVERE_BUDGET,
                   ) -> tuple[float | None, bool]:
    failed = False
    if name == BASELINE and baseline_nll is None:
        baseline_nll = cell["mean_nll"]
        cell["delta_mean_nll"] = 0.0
        cell["gate"] = None
        cell.update(exact_argmax_stats(cell_argmax, cell_argmax))
        cell["argmax_identity_is_gate"] = False
        cell["terrible_baseline"] = int(cell.get("terrible_tokens", 0))
        cell["terrible_delta"] = 0
        cell["complete_finite_aligned"] = cell_ok(cell, cell_nlls, cell_argmax)
        cell["quality_eligible"] = False
        cell["pass"] = cell["complete_finite_aligned"]
        failed = not cell["pass"]
        return baseline_nll, failed
    if baseline_nll is None:
        raise SystemExit("baseline scheme did not produce mean_nll")
    if base_argmax is None:
        raise SystemExit("baseline scheme did not produce argmax tokens")
    if baseline_terrible_tokens is None:
        raise SystemExit("baseline scheme did not produce a terrible-token count")
    cell.update(exact_argmax_stats(cell_argmax, base_argmax))
    cell["argmax_identity_is_gate"] = False
    cell["delta_mean_nll"] = cell["mean_nll"] - baseline_nll
    cell["terrible_baseline"] = baseline_terrible_tokens
    cell["terrible_delta"] = int(cell.get("terrible_tokens", 0)) - baseline_terrible_tokens
    if base_nlls is not None:
        delta_se = paired_delta_se(cell_nlls, base_nlls)
        cell["delta_nll_se"] = delta_se
        delta_summary = paired_delta_summary(cell_nlls, base_nlls)
        if delta_summary:
            cell.update(delta_summary)
        cell["in_noise"] = (
            delta_se is not None and abs(cell["delta_mean_nll"]) <= 2.0 * delta_se
        )
        cell.update(
            severe_position_stats(
                cell_nlls,
                base_nlls,
                threshold=TERRIBLE_NLL,
                maximum_new_rate=maximum_new_severe_rate,
                minimum_budget=minimum_new_severe_budget,
            )
        )
    else:
        raise SystemExit("candidate comparison requires aligned BF16 NLL sidecars")
    cell["complete_finite_aligned"] = cell_ok(cell, cell_nlls, cell_argmax)
    if name in gates:
        cell["gate"] = gates[name]
        cell["pass"] = (
            cell["complete_finite_aligned"]
            and cell["new_severe_positions_pass"]
            and cell["delta_mean_nll"] <= gates[name]
        )
        cell["quality_eligible"] = cell["pass"]
    else:
        cell["gate"] = None
        cell["pass"] = (
            cell["complete_finite_aligned"]
            and cell["new_severe_positions_pass"]
        )
        cell["quality_eligible"] = False
    failed = not cell["pass"]
    return baseline_nll, failed


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        f"# Perplexity: {payload['model_id']}",
        "",
        f"- weight inputs: {payload['weights_inputs']}",
        f"- candidate artifact: {payload.get('candidate_artifact')}",
        f"- lengths: {payload['lengths']}",
        f"- skip default: {payload['skip']}",
        f"- device graphs: on unless a cell sets device_graph=false",
        f"- terrible token: nll >= {TERRIBLE_NLL}",
        f"- independent baseline: `{payload['baseline']}` per (length, schedule, spec)",
        f"- decode spec: {payload.get('spec', '-')} (draft {payload.get('draft_tokens', '-')}) "
        f"unless a cell sets spec=none; prefill lane is spec-free",
        "",
        "| length | schedule | scheme | spec | graph | skip | scored | greedy exact (diagnostic) | flips | flip rate | mean_nll | max_nll | severe | severe Δ | new / budget | ppl | Δ mean_nll | Δabs p50 | p95 | p99 | max | Δ 1σ | σ nll | noise | quality gate |",
        "|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for cell in payload["cells"]:
        delta = cell.get("delta_mean_nll")
        delta_text = "-" if delta is None else f"{delta:.6f}"
        delta_se = cell.get("delta_nll_se")
        se_text = "-" if delta_se is None else f"{delta_se:.6f}"
        percentile_text = [
            "-" if cell.get(field) is None else f"{cell[field]:.6f}"
            for field in ("delta_abs_p50", "delta_abs_p95", "delta_abs_p99", "delta_abs_max")
        ]
        nll_std = cell.get("nll_std")
        std_text = "-" if nll_std is None else f"{nll_std:.3f}"
        noise_text = "yes" if cell.get("in_noise") else ""
        gate = cell.get("gate")
        if cell.get("scheme") == BASELINE and gate is None:
            gate_text = "reference" if cell.get("pass") else "FAIL"
        elif gate is None:
            gate_text = "PASS" if cell.get("pass") else "FAIL"
        else:
            gate_text = "PASS" if cell.get("pass") else "FAIL"
        graph = "on" if cell.get("device_graph", True) else "off"
        lines.append(
            f"| {cell.get('prompt_tokens', '')} | {cell.get('schedule', '')} | `{cell['scheme']}` | "
            f"{cell.get('spec', 'none')} | {graph} | {cell.get('skip_tokens', '')} | "
            f"{cell['tokens_scored']} | {'yes' if cell.get('argmax_exact') else 'NO'} | "
            f"{cell.get('argmax_mismatches', '-')} | {cell.get('argmax_flip_rate', 0.0):.6f} | "
            f"{cell['mean_nll']:.6f} | {cell.get('max_nll', 0):.4f} | "
            f"{cell.get('terrible_tokens', 0)} | {cell.get('terrible_delta', '-')} | "
            f"{cell.get('new_severe_positions', '-')} / {cell.get('new_severe_position_budget', '-')} | "
            f"{cell['ppl']:.4f} | {delta_text} | {' | '.join(percentile_text)} | {se_text} | "
            f"{std_text} | {noise_text} | {gate_text} |"
        )
    if payload.get("parity"):
        lines.extend(["", "## Per-profile prefill vs decode schedule comparison", ""])
        for row in payload["parity"]:
            lines.append(
                f"- {row['profile']}, {row['prompt_tokens']} tokens: "
                f"mean Δ={row['mean_delta_nll']:.6f}, "
                f"mean |Δ|={row['mean_abs_delta_nll']:.6f}, "
                f"max |Δ|={row['max_abs_delta_nll']:.6f}, greedy exact="
                f"{'yes' if row.get('argmax_exact') else 'NO'} "
                f"({row.get('argmax_mismatches', 0)} flips), gate="
                f"{row.get('max_abs_nll_gate', '-')}, "
                f"{'PASS' if row.get('pass') else 'FAIL'}"
            )
    lines.extend(
        [
            "",
            "_Greedy exactness and flip rate against BF16 are diagnostics, not quality gates. "
            "Prefill/decode schedule flips are also diagnostic because the selected attention "
            "routes may use different qualified private precision; their finite aligned NLL "
            "sidecars must meet the explicitly supplied bound. Graph/eager, MTP/ordinary, and "
            "draft-window execution variants remain exact-token comparisons._",
            "",
            "_Noise columns: `σ nll` is the per-token NLL std for the cell; `Δ 1σ` is the SE of "
            "the per-token paired Δnll vs the independent BF16 reference (index-aligned tokens, from "
            "the .nllf32 sidecars). `noise` marks |Δ| ≤ 2·Δ1σ — the delta is not resolved above "
            "the per-token noise floor._",
        ]
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-reference-weights", type=Path, required=True,
                        help="complete original BF16 source-checkpoint directory")
    parser.add_argument("--g16-weights", type=Path,
                        help="explicit FP8-K/INT4-V candidate artifact for the G16 build")
    parser.add_argument("--g32-weights", type=Path,
                        help="explicit FP8-K/INT4-V candidate artifact for the G32 build")
    parser.add_argument("--bf16-reference-ppl-bin", type=Path,
                        default=default_reference_ppl_bin(),
                        help="independent BF16 reference scorer (or NINFER_BF16_REFERENCE_PPL)")
    parser.add_argument("--g16-ppl-bin", type=Path, default=default_group_ppl_bin(16),
                        help="G16 candidate scorer (or NINFER_R9700_G16_PPL)")
    parser.add_argument("--g32-ppl-bin", type=Path, default=default_group_ppl_bin(32),
                        help="G32 candidate scorer (or NINFER_R9700_G32_PPL)")
    parser.add_argument("--ids", type=Path, default=Path(__file__).resolve().parent / "corpus.ids")
    parser.add_argument("--tokens", type=int, default=None, help="single length (default: 8k then 32k)")
    parser.add_argument("--long", action="store_true", help=f"only {LONG_TOKENS} tokens")
    parser.add_argument("--profiles", default=None,
                        help="comma-separated profiles; default: bf16-reference,r9700-g16,r9700-g32")
    parser.add_argument(
        "--schedule",
        default="prefill,decode",
        help="prefill, decode, or comma-separated list",
    )
    parser.add_argument("--skip", default="half", help="warmup tokens not scored: half (default) or an integer")
    parser.add_argument("--gate", action="append", default=[],
                        help="candidate-profile=max_delta_mean_nll")
    parser.add_argument(
        "--quality-tier", choices=tuple(QUALITY_TIERS), required=True,
        help="explicit quality guardrail profile; recorded in every candidate result",
    )
    parser.add_argument(
        "--allow-ungated",
        action="store_true",
        help="bring-up only: permit selected candidates without a mean-NLL gate",
    )
    parser.add_argument(
        "--schedule-parity-max-abs-nll",
        type=float,
        default=None,
        help="required with both prefill and decode: measured per-token max |delta NLL| limit",
    )
    parser.add_argument(
        "--execution-parity-max-abs-nll",
        type=float,
        default=0.0,
        help="graph/eager and speculative/ordinary max |delta NLL| gate (default: exact)",
    )
    parser.add_argument("--prefill-chunk", type=int, default=4096)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--expected-q4-activation-bits", type=int, choices=(4, 8), default=8,
        help="reject candidate scorers not compiled for this Q4 activation width (default: 8)",
    )
    parser.add_argument(
        "--expected-w8-activation-bits", type=int, choices=(8, 16), default=8,
        help="reject candidate scorers not compiled for this W8 activation width (default: 8)",
    )
    parser.add_argument(
        "--expected-fp8-qk-wmma", type=int, choices=(0, 1), default=1,
        help="require the selected T1/T2 FP8-Q/K attention profile (default: 1)",
    )
    parser.add_argument(
        "--expected-xattention-profile", choices=XATTENTION_PROFILES, default="dense",
        help="require dense Text prefill or the private compile-bound B128/S16/tau=.9 route",
    )
    parser.add_argument(
        "--reuse-bf16-campaign", type=Path,
        help="import provenance-matched BF16 cells from a schema-v6 prefill campaign",
    )
    parser.add_argument(
        "--bf16-repeat-comparison", type=Path,
        help=("exact schema-v1 fresh-process A/B proof for --reuse-bf16-campaign; "
              "required when candidate quality is derived from reused BF16 cells"),
    )
    parser.add_argument(
        "--reuse-candidate-campaign", type=Path, action="append", default=[],
        help=("import provenance-matched primary candidate cells from a schema-v6 "
              "prefill campaign and recompute their BF16-relative metrics; repeat to "
              "combine disjoint profile/length subsets"),
    )
    parser.add_argument(
        "--require-fp8-hybrid", action="store_true",
        help="require the target-authority hybrid identity and validated conversion receipt",
    )
    parser.add_argument("--no-extras", action="store_true", help="skip mid-page, short-context, graphs-off, mtp")
    parser.add_argument(
        "--no-position-extras",
        action="store_true",
        help="skip only the paired mid-page and short-context probes",
    )
    parser.add_argument("--spec", default="mtp", choices=("mtp", "none"),
                        help="speculative backend for decode-lane cells (default: mtp, "
                             "matching production serve; prefill-lane cells are spec-free)")
    parser.add_argument("--draft-tokens", type=int, default=DEFAULT_DRAFT_TOKENS,
                        help=f"MTP draft tokens for decode-lane cells (default: "
                             f"{DEFAULT_DRAFT_TOKENS})")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.long:
        lengths = [LONG_TOKENS]
    elif args.tokens is not None:
        lengths = [args.tokens]
    else:
        lengths = [DEFAULT_TOKENS, LONG_TOKENS]
    if any(tokens < 3 for tokens in lengths):
        raise SystemExit("campaign lengths must be at least three tokens")
    if args.prefill_chunk < 1:
        raise SystemExit("--prefill-chunk must be positive")
    if args.device < 0:
        raise SystemExit("--device must be nonnegative")
    if args.skip != "half":
        try:
            requested_skip = int(args.skip)
        except ValueError as error:
            raise SystemExit("--skip must be half or a nonnegative integer") from error
        if requested_skip < 0 or any(requested_skip > tokens - 2 for tokens in lengths):
            raise SystemExit("--skip leaves no teacher-forced target")
    profiles = select_profiles(None if args.profiles is None else args.profiles.split(","))
    schedules = select_schedules(args.schedule)
    gates = parse_gates(args.gate)
    if set(schedules) == {"prefill", "decode"} and args.schedule_parity_max_abs_nll is None:
        raise SystemExit(
            "--schedule-parity-max-abs-nll is required when both prefill and decode are selected"
        )
    for threshold_name, threshold in (
        ("--schedule-parity-max-abs-nll", args.schedule_parity_max_abs_nll),
        ("--execution-parity-max-abs-nll", args.execution_parity_max_abs_nll),
    ):
        if threshold is None:
            continue
        if not math.isfinite(threshold) or threshold < 0.0:
            raise SystemExit(f"{threshold_name} must be finite and nonnegative")
    quality_tier = QUALITY_TIERS[args.quality_tier]
    tier_mean_limit = quality_tier["maximum_mean_nll_delta"]
    mismatched_gates = [
        name for name, value in gates.items()
        if not math.isclose(value, tier_mean_limit, rel_tol=0.0, abs_tol=1e-9)
    ]
    if mismatched_gates:
        raise SystemExit(
            f"--gate must equal the {args.quality_tier} tier mean-NLL limit "
            f"{tier_mean_limit}: {', '.join(mismatched_gates)}"
        )
    gates = {name: tier_mean_limit for name in gates}
    spec = args.spec
    if spec == "mtp" and not 1 <= args.draft_tokens <= 5:
        raise SystemExit("MTP --draft-tokens must be in 1..5")
    draft_tokens = args.draft_tokens if spec == "mtp" else 0
    if args.bf16_reference_ppl_bin is None or not args.bf16_reference_ppl_bin.is_file():
        raise SystemExit("independent BF16 scorer not found; pass --bf16-reference-ppl-bin")
    profile_weights = {
        BASELINE: args.bf16_reference_weights,
        "r9700-g16": args.g16_weights,
        "r9700-g32": args.g32_weights,
    }
    profile_bins = {
        BASELINE: args.bf16_reference_ppl_bin,
        "r9700-g16": args.g16_ppl_bin,
        "r9700-g32": args.g32_ppl_bin,
    }
    for name in profiles:
        weights_input = profile_weights[name]
        if weights_input is None:
            option = "--g16-weights" if name == "r9700-g16" else "--g32-weights"
            raise SystemExit(f"{name} selected without {option}")
        # A hash/provenance-validated retained BF16 authority carries its own complete source
        # identity and sidecars; importing it must not pretend the 18-shard checkpoint is needed.
        # Fresh BF16 execution still validates the source input normally.
        if name != BASELINE or args.reuse_bf16_campaign is None:
            validate_weights_input(name, weights_input)
        scorer = profile_bins[name]
        if scorer is None or not scorer.is_file():
            option = "--bf16-reference-ppl-bin" if name == BASELINE else (
                "--g16-ppl-bin" if name == "r9700-g16" else "--g32-ppl-bin"
            )
            raise SystemExit(f"scorer not found for {name}; pass {option}")
        if PROFILES[name].reference:
            preflight_python_scorer(scorer)
    scorer_provenance = {
        name: {
            "path": str(profile_bins[name].resolve()),
            "bytes": profile_bins[name].stat().st_size,
            "sha256": file_sha256(profile_bins[name]),
        }
        for name in profiles
    }
    selected_candidates = [name for name in profiles if name != BASELINE]
    missing_gates = [name for name in selected_candidates if name not in gates]
    if missing_gates and not args.allow_ungated:
        raise SystemExit(
            "selected candidates require explicit --gate values: "
            + ", ".join(missing_gates)
            + "; use --allow-ungated only for bring-up"
        )

    candidate_artifact = None
    if selected_candidates:
        if set(selected_candidates) == {"r9700-g16", "r9700-g32"}:
            candidate_artifact = require_same_candidate_artifact(
                profile_weights["r9700-g16"], profile_weights["r9700-g32"]
            )
        else:
            only = selected_candidates[0]
            candidate_artifact = inspect_candidate_artifact(profile_weights[only])
    if args.require_fp8_hybrid:
        require_fp8_hybrid_candidate(candidate_artifact)
    expected_weights_ids = {
        BASELINE: BF16_WEIGHTS_ID,
        **(
            {name: candidate_artifact["weights_id"] for name in selected_candidates}
            if candidate_artifact is not None
            else {}
        ),
    }

    corpus_profile = next((name for name in profiles if name != BASELINE), BASELINE)
    ensure_corpus(args.ids, max(lengths), profile_weights[corpus_profile],
                  profile_bins[corpus_profile])
    corpus_provenance = validate_corpus(args.ids, max(lengths))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or (REPO / "profiles" / "ppl" / stamp)

    reused_bf16: dict[int, tuple[dict, Path]] = {}
    bf16_repeat_comparison = None
    if args.reuse_bf16_campaign is not None or args.reuse_candidate_campaign:
        if (
            schedules != ["prefill"]
            or spec != "none"
        ):
            raise SystemExit(
                "campaign reuse requires a prefill-only non-speculative campaign"
            )
    if (args.reuse_bf16_campaign is None) != (args.bf16_repeat_comparison is None):
        raise SystemExit(
            "--reuse-bf16-campaign and --bf16-repeat-comparison are required together"
        )
    if args.reuse_bf16_campaign is not None:
        bf16_repeat_comparison = validate_bf16_repeat_comparison(
            args.bf16_repeat_comparison, args.reuse_bf16_campaign
        )
        reused_bf16 = load_reused_bf16_cells(
            args.reuse_bf16_campaign,
            bf16_weights=profile_weights[BASELINE],
            bf16_scorer=profile_bins[BASELINE],
            scorer_identity=scorer_provenance[BASELINE],
            ids=args.ids,
            corpus_provenance=corpus_provenance,
            lengths=lengths,
            skip=args.skip,
            prefill_chunk=args.prefill_chunk,
            device=args.device,
        )

    reused_candidates: dict[tuple[str, int], tuple[dict, Path]] = {}
    if args.reuse_candidate_campaign:
        if not selected_candidates or candidate_artifact is None:
            raise SystemExit("--reuse-candidate-campaign requires a selected candidate")
        allow_partial = len(args.reuse_candidate_campaign) > 1
        for campaign_path in args.reuse_candidate_campaign:
            imported = load_reused_candidate_cells(
                campaign_path,
                selected_candidates=selected_candidates,
                candidate_artifact=candidate_artifact,
                profile_weights=profile_weights,
                profile_bins=profile_bins,
                scorer_provenance=scorer_provenance,
                ids=args.ids,
                corpus_provenance=corpus_provenance,
                lengths=lengths,
                skip=args.skip,
                prefill_chunk=args.prefill_chunk,
                device=args.device,
                expected_q4_activation_bits=args.expected_q4_activation_bits,
                expected_w8_activation_bits=args.expected_w8_activation_bits,
                expected_fp8_qk_wmma=bool(args.expected_fp8_qk_wmma),
                expected_xattention_profile=args.expected_xattention_profile,
                allow_partial=allow_partial,
            )
            overlap = set(reused_candidates) & set(imported)
            if overlap:
                raise SystemExit(
                    "reused candidate campaigns overlap at "
                    + ", ".join(f"{name}/{tokens}" for name, tokens in sorted(overlap))
                )
            reused_candidates.update(imported)

    # A durable campaign may intentionally reuse a real directory for imported cells, but must
    # never follow a symlinked output namespace (including a dangling symlink).
    prepare_output_directory(out_dir)
    cells: list[dict] = []
    failed = False

    def score_matrix(tokens: int) -> None:
        nonlocal failed
        for schedule in schedules:
            baseline_nll = None
            base_nlls: list[float] = []
            base_argmax: list[int] = []
            base_terrible_tokens: int | None = None
            # The decode lane scores under the production MTP spec; the prefill lane is
            # spec-free (MTP is not consulted by chunked prompt scoring).
            spec_extra = (
                ["--spec", "mtp", "--draft-tokens", str(draft_tokens)]
                if spec == "mtp" and schedule == "decode" else []
            )
            for name in profiles:
                cell_path = out_dir / f"{tokens}.{schedule}.{name}.json"
                evidence_path = cell_path
                if name == BASELINE and tokens in reused_bf16:
                    cell, evidence_path = reused_bf16[tokens]
                elif (name, tokens) in reused_candidates:
                    cell, evidence_path = reused_candidates[(name, tokens)]
                else:
                    cell = run_cell(
                        profile_bins[name], profile_weights[name], args.ids, name, schedule,
                        args.skip, tokens, args.prefill_chunk, args.device, cell_path,
                        list(spec_extra), expected_weights_ids[name],
                        args.expected_q4_activation_bits, args.expected_w8_activation_bits,
                        bool(args.expected_fp8_qk_wmma), args.expected_xattention_profile,
                    )
                cell["profile_value_group"] = PROFILES[name].value_group
                cell["quality_tier"] = args.quality_tier
                nlls = load_nlls(evidence_path)
                argmax = load_argmax(evidence_path)
                attach_nll_stats(cell, nlls)
                if name == BASELINE:
                    base_nlls = nlls
                    base_argmax = argmax
                    base_terrible_tokens = int(cell.get("terrible_tokens", 0))
                baseline_nll, cell_failed = apply_baseline(
                    cell, nlls, baseline_nll, gates, name,
                    base_nlls if name != BASELINE else None, argmax,
                    base_argmax if name != BASELINE else None,
                    base_terrible_tokens, quality_tier["maximum_new_severe_rate"],
                    quality_tier["minimum_new_severe_budget"])
                failed = failed or cell_failed
                cells.append(cell)

    def score_paired_extra(label: str, skip: str, tokens: int, extra: list[str]) -> None:
        nonlocal failed
        baseline_nll: float | None = None
        base_nlls: list[float] = []
        base_argmax: list[int] = []
        base_terrible_tokens: int | None = None
        for name in profiles:
            cell_path = out_dir / f"{tokens}.decode.{name}.{label}.json"
            cell = run_cell(
                profile_bins[name], profile_weights[name], args.ids, name, "decode", skip,
                tokens, args.prefill_chunk, args.device, cell_path, list(extra),
                expected_weights_ids[name],
                args.expected_q4_activation_bits,
                args.expected_w8_activation_bits,
                bool(args.expected_fp8_qk_wmma),
                args.expected_xattention_profile,
            )
            cell["profile_value_group"] = PROFILES[name].value_group
            cell["quality_tier"] = args.quality_tier
            nlls = load_nlls(cell_path)
            argmax = load_argmax(cell_path)
            attach_nll_stats(cell, nlls)
            if name == BASELINE:
                base_nlls = nlls
                base_argmax = argmax
                base_terrible_tokens = int(cell.get("terrible_tokens", 0))
            baseline_nll, cell_failed = apply_baseline(
                cell, nlls, baseline_nll, gates, name,
                base_nlls if name != BASELINE else None, argmax,
                base_argmax if name != BASELINE else None,
                base_terrible_tokens, quality_tier["maximum_new_severe_rate"],
                quality_tier["minimum_new_severe_budget"])
            failed = failed or cell_failed
            cells.append(cell)

    def score_candidate_execution_extra(
        label: str,
        tokens: int,
        extra: list[str],
        parity_key: str,
    ) -> None:
        """Pair a candidate execution variant without rerunning invariant BF16 math."""

        nonlocal failed
        reference_path = out_dir / f"{tokens}.decode.{BASELINE}.json"
        reference_cell = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_nlls = load_nlls(reference_path)
        reference_argmax = load_argmax(reference_path)
        reference_terrible = int(reference_cell.get("terrible_tokens", 0))
        for name in selected_candidates:
            cell_path = out_dir / f"{tokens}.decode.{name}.{label}.json"
            cell = run_cell(
                profile_bins[name], profile_weights[name], args.ids, name, "decode", args.skip,
                tokens, args.prefill_chunk, args.device, cell_path, list(extra),
                expected_weights_ids[name],
                args.expected_q4_activation_bits,
                args.expected_w8_activation_bits,
                bool(args.expected_fp8_qk_wmma),
                args.expected_xattention_profile,
            )
            cell["profile_value_group"] = PROFILES[name].value_group
            cell["quality_tier"] = args.quality_tier
            cell["reference_cell"] = str(reference_path)
            nlls = load_nlls(cell_path)
            argmax = load_argmax(cell_path)
            attach_nll_stats(cell, nlls)
            _, cell_failed = apply_baseline(
                cell, nlls, float(reference_cell["mean_nll"]), gates, name,
                reference_nlls, argmax, reference_argmax, reference_terrible,
                quality_tier["maximum_new_severe_rate"],
                quality_tier["minimum_new_severe_budget"],
            )
            primary_path = out_dir / f"{tokens}.decode.{name}.json"
            execution = sidecar_parity(
                primary_path,
                cell_path,
                max_abs_nll=args.execution_parity_max_abs_nll,
            )
            cell[parity_key] = execution
            cell_failed = cell_failed or not execution["pass"]
            cell["pass"] = cell["pass"] and execution["pass"]
            cell["quality_eligible"] = cell["quality_eligible"] and execution["pass"]
            failed = failed or cell_failed
            cells.append(cell)

    def score_execution_extras(tokens: int) -> None:
        decode_extra = (
            ["--spec", "mtp", "--draft-tokens", str(draft_tokens)] if spec == "mtp" else []
        )
        score_candidate_execution_extra(
            "eager", tokens, [*decode_extra, "--no-device-graph"], "device_graph_parity"
        )
        if spec != "mtp":
            return
        score_candidate_execution_extra("ordinary", tokens, [], "spec_parity")
        # The alternate draft-length probe is a focused 8K control. MTP/ordinary parity
        # already covers speculative execution at both required campaign lengths.
        if tokens == DEFAULT_TOKENS and draft_tokens != MTP_EXTRA_DRAFT_TOKENS:
            score_candidate_execution_extra(
                "mtp",
                tokens,
                ["--spec", "mtp", "--draft-tokens", str(MTP_EXTRA_DRAFT_TOKENS)],
                "draft_window_parity",
            )

    def score_8k_position_extras() -> None:
        decode_extra = (
            ["--spec", "mtp", "--draft-tokens", str(draft_tokens)] if spec == "mtp" else []
        )
        score_paired_extra("midpage", str(MID_PAGE_SKIP), DEFAULT_TOKENS, decode_extra)
        score_paired_extra("short", str(SHORT_SKIP), SHORT_TOKENS, decode_extra)

    for tokens in lengths:
        score_matrix(tokens)
        if not args.no_extras and "decode" in schedules:
            score_execution_extras(tokens)
            if tokens == DEFAULT_TOKENS and not args.no_position_extras:
                score_8k_position_extras()

    parity = []
    if "prefill" in schedules and "decode" in schedules:
        for tokens in lengths:
            for name in profiles:
                prefill_path = out_dir / f"{tokens}.prefill.{name}.json"
                decode_path = out_dir / f"{tokens}.decode.{name}.json"
                stats = schedule_sidecar_comparison(
                    prefill_path,
                    decode_path,
                    max_abs_nll=args.schedule_parity_max_abs_nll,
                )
                stats["prompt_tokens"] = tokens
                stats["profile"] = name
                parity.append(stats)
                failed = failed or not stats["pass"]

    reference_sources = {
        (
            cell.get("source_config_sha256"),
            cell.get("source_index_sha256"),
            json.dumps(cell.get("source_shards_sha256"), sort_keys=True),
            cell.get("source_tensor_count"),
            cell.get("source_text_tensor_count"),
            cell.get("source_shard_count"),
            validate_bf16_execution_provenance(cell.get("execution_provenance")),
        )
        for cell in cells
        if cell.get("scheme") == BASELINE
    }
    if len(reference_sources) != 1:
        raise SystemExit("BF16 source provenance changed across campaign cells")
    (
        source_config_sha256,
        source_index_sha256,
        source_shards_json,
        source_count,
        text_count,
        shard_count,
        execution_json,
    ) = next(iter(reference_sources))

    payload = {
        "artifact_type": CAMPAIGN_ARTIFACT_TYPE,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights_inputs": {name: str(profile_weights[name]) for name in profiles},
        "scorers": scorer_provenance,
        "model_id": MODEL_ID,
        "reference_weights_id": BF16_WEIGHTS_ID,
        "reference_source": {
            "config_sha256": source_config_sha256,
            "index_sha256": source_index_sha256,
            "shards_sha256": json.loads(source_shards_json),
            "tensor_count": source_count,
            "text_tensor_count": text_count,
            "shard_count": shard_count,
        },
        "reference_execution": json.loads(execution_json),
        "candidate_artifact": candidate_artifact,
        "required_candidate_identity": (
            "fp8-hybrid-selection-authority" if args.require_fp8_hybrid else None
        ),
        "corpus": corpus_provenance,
        "lengths": lengths,
        "skip": args.skip,
        "prefill_chunk": args.prefill_chunk,
        "schedules": schedules,
        "spec": spec,
        "draft_tokens": draft_tokens,
        "q4_activation_bits": args.expected_q4_activation_bits,
        "w8_activation_bits": args.expected_w8_activation_bits,
        "candidate_kv_plane_layouts": (
            R9700_KV_PLANE_LAYOUTS if selected_candidates else None
        ),
        "fp8_qk_wmma_enabled": bool(args.expected_fp8_qk_wmma),
        "fp8_qk_wmma_profile": FP8_QK_WMMA_PROFILE,
        "fp8_qk_wmma_t1_min_context": FP8_QK_WMMA_T1_MIN_CONTEXT,
        "fp8_qk_wmma_t2_min_context": FP8_QK_WMMA_T2_MIN_CONTEXT,
        "xattention_profile": args.expected_xattention_profile,
        "reused_bf16_campaign": (
            {
                "path": str(args.reuse_bf16_campaign),
                "sha256": file_sha256(args.reuse_bf16_campaign),
            }
            if args.reuse_bf16_campaign is not None else None
        ),
        "bf16_repeat_comparison": bf16_repeat_comparison,
        "reused_candidate_campaign": (
            {
                "path": str(args.reuse_candidate_campaign[0]),
                "sha256": file_sha256(args.reuse_candidate_campaign[0]),
            }
            if len(args.reuse_candidate_campaign) == 1 else None
        ),
        "reused_candidate_campaigns": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in args.reuse_candidate_campaign
        ],
        "baseline": BASELINE,
        "gates": gates,
        "quality_tier": args.quality_tier,
        "quality_gate_contract": {
            "complete_finite_aligned_sidecars": True,
            "maximum_mean_nll_delta_by_profile": gates,
            "severe_nll_threshold": TERRIBLE_NLL,
            "tier": args.quality_tier,
            "tier_maximum_mean_nll_delta": tier_mean_limit,
            "maximum_new_severe_rate": quality_tier["maximum_new_severe_rate"],
            "minimum_new_severe_budget": quality_tier["minimum_new_severe_budget"],
            "new_severe_budget_formula": "max(minimum_budget, ceil(rate * scored_positions))",
            "bf16_argmax_identity": "diagnostic_only",
            "prefill_decode_schedule": {
                "complete_finite_aligned_sidecars": True,
                "maximum_absolute_nll_delta": args.schedule_parity_max_abs_nll,
                "argmax_identity": "diagnostic_only",
            },
            "same_route_execution_variants": {
                "maximum_absolute_nll_delta": args.execution_parity_max_abs_nll,
                "argmax_identity": "required",
            },
            "pareto_selection": "requires separate whole-inference speed and capacity evidence",
        },
        "schedule_parity_max_abs_nll": args.schedule_parity_max_abs_nll,
        "execution_parity_max_abs_nll": args.execution_parity_max_abs_nll,
        "position_extras_enabled": not args.no_extras and not args.no_position_extras,
        "terrible_nll": TERRIBLE_NLL,
        "parity": parity,
        "cells": cells,
        "pass": not failed,
    }
    json_path = out_dir / "results.json"
    md_path = out_dir / "results.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"ppl results: {out_dir}")
    print(f"  {json_path}")
    print(f"  {md_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
