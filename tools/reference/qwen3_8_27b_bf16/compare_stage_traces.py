#!/usr/bin/env python3
"""Fail-closed comparison of two diagnostic BF16 stage traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.reference.qwen3_8_27b_bf16.protocol import (  # noqa: E402
    ATTENTION_PV_EXECUTION, DETERMINISTIC_ENVIRONMENT,
    DETERMINISTIC_EXECUTION_PROFILE, FORBIDDEN_EXECUTION_ENVIRONMENT,
    FULL_ATTENTION_LAYERS, GDN_RECURRENCE_EXECUTION, HIDDEN_SIZE, LAYERS,
    MATMUL_REDUCTION_EXECUTION, TOKEN_DOMAIN, TRITON_CODEGEN_EXECUTION,
)
from tools.reference.qwen3_8_27b_bf16.stage_trace import (  # noqa: E402
    ATTENTION_DETAIL_KV_HEADS, ATTENTION_DETAIL_LAYER,
    ATTENTION_DETAIL_Q_PER_KV, ATTENTION_DETAIL_HEAD_DIM,
    ATTENTION_DETAIL_STAGE_SHAPES, ATTENTION_KERNEL_DETAIL_STAGES,
    LOGIT_ROWS, TRACE_ENVIRONMENT_KEYS, TRACE_ROWS, TYPE, VERSION,
)


COMPARISON_TYPE = "ninfer_qwen3_8_bf16_stage_trace_comparison"
COMPARISON_VERSION = 8
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require_sha256(value, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"invalid stage trace {field}")


def _expected_checkpoints(prompt_tokens: int, skip_tokens: int) -> list[tuple]:
    hidden_rows = prompt_tokens - 1
    width = min(TRACE_ROWS, hidden_rows)
    result = []

    def hidden(name: str) -> None:
        result.extend((
            (name + ".first", 0, width, [width, HIDDEN_SIZE]),
            (name + ".last", hidden_rows - width, hidden_rows, [width, HIDDEN_SIZE]),
        ))

    hidden("embedding")
    for layer in range(LAYERS):
        route = "attention" if layer in FULL_ATTENTION_LAYERS else "gdn"
        if layer == ATTENTION_DETAIL_LAYER:
            for stage, tail_shape in ATTENTION_DETAIL_STAGE_SHAPES:
                result.append((
                    f"layer-{layer:02d}.attention-detail.{stage}.last",
                    hidden_rows - width,
                    hidden_rows,
                    [width, *tail_shape],
                ))
                if stage == "rope-k":
                    for kv_head in range(ATTENTION_DETAIL_KV_HEADS):
                        for kernel_stage in ATTENTION_KERNEL_DETAIL_STAGES:
                            tail = (
                                [ATTENTION_DETAIL_Q_PER_KV, hidden_rows]
                                if kernel_stage != "pv-fp32"
                                else [ATTENTION_DETAIL_Q_PER_KV,
                                      ATTENTION_DETAIL_HEAD_DIM]
                            )
                            result.append((
                                f"layer-{layer:02d}.attention-detail.kv-{kv_head:02d}."
                                f"{kernel_stage}.last",
                                hidden_rows - width,
                                hidden_rows,
                                [width, *tail],
                            ))
        hidden(f"layer-{layer:02d}.{route}")
        hidden(f"layer-{layer:02d}.mlp")
    hidden("final-norm")
    first_end = min(hidden_rows, skip_tokens + LOGIT_ROWS)
    result.append(("logits.first", skip_tokens, first_end,
                   [first_end - skip_tokens, TOKEN_DOMAIN]))
    if first_end < hidden_rows:
        last_begin = skip_tokens + ((hidden_rows - skip_tokens - 1) // LOGIT_ROWS) * LOGIT_ROWS
        result.append(("logits.last", last_begin, hidden_rows,
                       [hidden_rows - last_begin, TOKEN_DOMAIN]))
    return result


def validate_trace(trace: dict) -> None:
    if not isinstance(trace, dict):
        raise ValueError("invalid stage trace root")
    if (trace.get("artifact_type") != TYPE or trace.get("schema_version") != VERSION
            or trace.get("quality_evidence") is not False):
        raise ValueError("invalid stage trace header")
    prompt_tokens = trace.get("prompt_tokens")
    skip_tokens = trace.get("skip_tokens")
    prefill_chunk = trace.get("prefill_chunk")
    if (not isinstance(prompt_tokens, int) or prompt_tokens < 2
            or not isinstance(skip_tokens, int) or not 0 <= skip_tokens < prompt_tokens - 1
            or not isinstance(prefill_chunk, int) or prefill_chunk < 1
            or trace.get("trace_rows") != TRACE_ROWS
            or trace.get("logit_rows") != LOGIT_ROWS):
        raise ValueError("invalid stage trace workload")

    run = trace.get("run")
    if (not isinstance(run, dict) or set(run) != {"id", "process_id"}
            or not isinstance(run["id"], str) or not run["id"]
            or not isinstance(run["process_id"], int) or run["process_id"] < 1):
        raise ValueError("invalid stage trace run identity")

    source = trace.get("source")
    if not isinstance(source, dict) or set(source) != {
        "config_sha256", "index_sha256", "shards_sha256", "corpus_ids_sha256",
    }:
        raise ValueError("invalid stage trace source")
    for key in ("config_sha256", "index_sha256", "corpus_ids_sha256"):
        _require_sha256(source[key], f"source.{key}")
    shards = source["shards_sha256"]
    if not isinstance(shards, dict) or not shards:
        raise ValueError("invalid stage trace source.shards_sha256")
    for name, digest in shards.items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid stage trace source shard name")
        _require_sha256(digest, f"source.shards_sha256.{name}")

    provenance = trace.get("provenance")
    required_provenance = {
        "execution_profile", "attention_pv", "gdn_recurrence", "tunable_op",
        "triton_codegen",
        "python", "python_executable", "python_executable_sha256", "platform",
        "torch", "torch_git", "hip", "device_index", "device_name", "device_arch",
        "matmul_precision", "preferred_blas_library", "deterministic_algorithms",
        "matmul_reduction", "environment",
        "scorer_python_tree_sha256", "fla_python_tree_sha256", "distributions",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise ValueError("invalid stage trace provenance")
    if provenance["execution_profile"] != DETERMINISTIC_EXECUTION_PROFILE:
        raise ValueError("invalid stage trace provenance.execution_profile")
    if provenance["attention_pv"] != ATTENTION_PV_EXECUTION:
        raise ValueError("invalid stage trace provenance.attention_pv")
    if provenance["gdn_recurrence"] != GDN_RECURRENCE_EXECUTION:
        raise ValueError("invalid stage trace provenance.gdn_recurrence")
    if provenance["tunable_op"] != {"enabled": False}:
        raise ValueError("invalid stage trace provenance.tunable_op")
    if provenance["triton_codegen"] != TRITON_CODEGEN_EXECUTION:
        raise ValueError("invalid stage trace provenance.triton_codegen")
    if provenance["matmul_precision"] != "highest":
        raise ValueError("invalid stage trace provenance.matmul_precision")
    for key in ("python_executable_sha256", "scorer_python_tree_sha256",
                "fla_python_tree_sha256"):
        _require_sha256(provenance[key], f"provenance.{key}")
    if not isinstance(provenance["device_index"], int) or provenance["device_index"] < 0:
        raise ValueError("invalid stage trace provenance.device_index")
    if (not isinstance(provenance["device_arch"], str)
            or provenance["device_arch"].split(":", 1)[0] != "gfx1201"):
        raise ValueError("invalid stage trace provenance.device_arch")
    for key in ("python", "python_executable", "platform", "torch", "device_name"):
        if not isinstance(provenance[key], str) or not provenance[key]:
            raise ValueError(f"invalid stage trace provenance.{key}")
    if provenance["preferred_blas_library"] != "hipblas":
        raise ValueError("invalid stage trace provenance.preferred_blas_library")
    if provenance["matmul_reduction"] != MATMUL_REDUCTION_EXECUTION:
        raise ValueError("invalid stage trace provenance.matmul_reduction")
    for key in ("matmul_reduction", "environment", "distributions"):
        if not isinstance(provenance[key], dict) or not provenance[key]:
            raise ValueError(f"invalid stage trace provenance.{key}")
    deterministic = provenance["deterministic_algorithms"]
    if (not isinstance(deterministic, dict)
            or set(deterministic) != {"enabled", "warn_only"}
            or deterministic != {"enabled": True, "warn_only": False}):
        raise ValueError("invalid stage trace provenance.deterministic_algorithms")
    if set(provenance["environment"]) != set(TRACE_ENVIRONMENT_KEYS):
        raise ValueError("invalid stage trace provenance.environment")
    environment = provenance["environment"]
    for key, expected_value in DETERMINISTIC_ENVIRONMENT.items():
        if environment[key] != expected_value:
            raise ValueError(f"invalid stage trace provenance.environment.{key}")
    for key in FORBIDDEN_EXECUTION_ENVIRONMENT:
        if environment[key] is not None:
            raise ValueError(f"invalid stage trace provenance.environment.{key}")

    expected = _expected_checkpoints(prompt_tokens, skip_tokens)
    checkpoints = trace.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(expected):
        raise ValueError("invalid stage trace checkpoint inventory")
    keys = {
        "name", "begin", "end", "dtype", "shape", "sha256", "finite",
        "nan", "positive_infinity", "negative_infinity",
    }
    for index, (checkpoint, expected_row) in enumerate(zip(checkpoints, expected, strict=True)):
        name, begin, end, shape = expected_row
        if not isinstance(checkpoint, dict) or set(checkpoint) != keys:
            raise ValueError(f"invalid stage trace checkpoint {index}")
        fp32 = any(token in name for token in ATTENTION_KERNEL_DETAIL_STAGES)
        masked = "masked-qk-fp32" in name
        expected_negative_infinity = masked and shape[0] > 1
        if (checkpoint["name"] != name or checkpoint["begin"] != begin
                or checkpoint["end"] != end
                or checkpoint["dtype"] != ("torch.float32" if fp32 else "torch.bfloat16")
                or checkpoint["shape"] != shape
                or checkpoint["finite"] is not (not expected_negative_infinity)
                or checkpoint["nan"] is not False
                or checkpoint["positive_infinity"] is not False
                or checkpoint["negative_infinity"] is not expected_negative_infinity):
            raise ValueError(f"invalid stage trace checkpoint {name}")
        _require_sha256(checkpoint["sha256"], f"checkpoint {name} sha256")

    scores = trace.get("scores")
    if not isinstance(scores, dict) or set(scores) != {"nll_sha256", "argmax_sha256"}:
        raise ValueError("invalid stage trace scores")
    _require_sha256(scores["nll_sha256"], "scores.nll_sha256")
    _require_sha256(scores["argmax_sha256"], "scores.argmax_sha256")


def compare(first: dict, second: dict) -> dict:
    validate_trace(first)
    validate_trace(second)
    if first["run"]["id"] == second["run"]["id"]:
        raise ValueError("stage traces must come from distinct runs")
    for key in ("prompt_tokens", "skip_tokens", "prefill_chunk", "trace_rows",
                "logit_rows", "source", "provenance"):
        if first[key] != second[key]:
            raise ValueError(f"trace provenance differs at {key}")
    mismatches = [
        left["name"]
        for left, right in zip(first["checkpoints"], second["checkpoints"], strict=True)
        if left != right
    ]
    if first["scores"] != second["scores"]:
        mismatches.append("scores")
    return {
        "artifact_type": COMPARISON_TYPE,
        "schema_version": COMPARISON_VERSION,
        "quality_evidence": False,
        "exact": not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "mismatches": mismatches,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    try:
        first = json.loads(options.first.read_text())
        second = json.loads(options.second.read_text())
        result = compare(first, second)
        result["inputs"] = {
            "first": {"path": str(options.first.resolve()),
                      "sha256": _file_sha256(options.first)},
            "second": {"path": str(options.second.resolve()),
                       "sha256": _file_sha256(options.second)},
        }
        _atomic_json(options.out, result)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"compare-stage-traces: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
