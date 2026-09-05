#!/usr/bin/env python3
"""Source-only 8K PPL gate for selective MSE-Q4G128 Text weights.

This qualification-only checkpoint view applies one selected codec to the 160
P2048-dominant Text matrices (MLP down and GDN value-Z/output) and leaves every
other Text weight BF16.  The paired modes are canonical Q4G64 and the screened
eight-step source-MSE Q4G128.  It never writes a product artifact or registers
a runtime numeric format.  Activations and model arithmetic remain on the
independent BF16 source scorer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys
from typing import Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.convert.qwen3_8_27b_r9700 import mse_quantize
from tools.convert.qwen3_8_27b_r9700 import source_inventory
from tools.convert.qwen3_8_27b_r9700.screen_selective_q4g128_mse_quality import (
    MAX_CANDIDATE_OVER_CONTROL_RELATIVE_L2,
    Q4G128_DIAGNOSTIC,
    _mse_q4g128_decode,
    _validate_provenance as validate_screen_provenance,
    selective_specs,
    validate_report as validate_screen_report,
)
from tools.ppl.compare_q4_group_source import _against, _load_bf16, _source_key
from tools.ppl.fp8_hybrid_source_diagnostic import _source_rows
from tools.ppl.q4_group_source_diagnostic import (
    MAXIMUM_MEAN_NLL_DELTA, MAXIMUM_NEW_SEVERE_RATE, ROW_CHUNK, TERRIBLE_NLL,
    _source_identity, _write_score, quantize_decode_q4, sha256_file,
    validate_source_metadata,
)
from tools.reference.qwen3_8_27b_bf16 import protocol


ARTIFACT_TYPE = "ninfer_qwen3_8_selective_q4g128_source_diagnostic"
SCHEMA_VERSION = 1
TOKENS = 8192
CODECS = ("q4g64-absmax", "q4g128-mse")
SELECTED_SPECS = selective_specs()
SELECTED_SOURCE_ROWS = _source_rows(spec.name for spec in SELECTED_SPECS)
SELECTED_RAW_SOURCE_TENSORS = 208
SELECTED_RAW_SOURCE_ROWS = 1_163_264


def _expected_source_rows() -> dict[str, tuple[tuple[int, int], ...]]:
    rows: dict[str, tuple[tuple[int, int], ...]] = {}
    for layer in range(64):
        prefix = f"model.language_model.layers.{layer}."
        rows[prefix + "mlp.down_proj.weight"] = ((0, 5120),)
        if layer not in source_inventory.FULL_ATTENTION_LAYERS:
            rows[prefix + "linear_attn.in_proj_qkv.weight"] = ((4096, 10240),)
            rows[prefix + "linear_attn.in_proj_z.weight"] = ((0, 6144),)
            rows[prefix + "linear_attn.out_proj.weight"] = ((0, 5120),)
    return rows

if (len(SELECTED_SOURCE_ROWS) != SELECTED_RAW_SOURCE_TENSORS
        or sum(end - begin for spans in SELECTED_SOURCE_ROWS.values()
               for begin, end in spans) != SELECTED_RAW_SOURCE_ROWS
        or SELECTED_SOURCE_ROWS != _expected_source_rows()):
    raise ValueError("selective Q4G128 raw-source row inventory differs")


def _mse_q4g128_bf16(weight, *, row_chunk: int = mse_quantize.ROW_CHUNK):
    """Decode exact screened Q4G128 codes/FP16 scales at the BF16 boundary."""

    import torch

    if weight.device.type != "cpu":
        raise ValueError("selective Q4G128 fitting requires represented CPU source rows")
    if type(row_chunk) is not int or row_chunk <= 0:
        raise ValueError("selective Q4G128 row chunk must be a positive integer")
    result = torch.empty_like(weight)
    for begin in range(0, weight.shape[0], row_chunk):
        end = min(weight.shape[0], begin + row_chunk)
        result[begin:end] = _mse_q4g128_decode(weight[begin:end]).to(torch.bfloat16)
    return result


class SelectiveQ4Checkpoint:
    """Exact selective-codec view over the unchanged BF16 checkpoint loader."""

    def __init__(self, checkpoint, codec: str):
        if codec not in CODECS:
            raise ValueError("selective source diagnostic codec differs")
        self.checkpoint = checkpoint
        self.codec = codec

    def validate_metadata(self) -> None:
        self.checkpoint.validate_metadata()

    def _represented(self, name, tensor):
        if tensor.ndim != 2:
            return tensor
        spans = SELECTED_SOURCE_ROWS.get(name, ())
        if not spans:
            return tensor
        if max(end for _, end in spans) > tensor.shape[0]:
            raise ValueError(f"{name}: selected Q4G128 row is outside the source tensor")
        def encode(rows):
            return (quantize_decode_q4(rows, 64) if self.codec == "q4g64-absmax"
                    else _mse_q4g128_bf16(rows))
        if spans == ((0, tensor.shape[0]),):
            return encode(tensor)
        result = tensor.clone()
        for begin, end in spans:
            result[begin:end] = encode(tensor[begin:end])
        return result

    def load(self, name, device):
        import torch

        source = self.checkpoint.load(name, torch.device("cpu"))
        return self._represented(name, source).to(device=device, dtype=torch.bfloat16)

    def load_many(self, names: Iterable[str], device):
        import torch

        source = self.checkpoint.load_many(names, torch.device("cpu"))
        return {
            name: self._represented(name, tensor).to(device=device, dtype=torch.bfloat16)
            for name, tensor in source.items()
        }

    def embedding_rows(self, ids: list[int], device):
        import torch

        source = self.checkpoint.embedding_rows(ids, torch.device("cpu"))
        return source.to(device=device, dtype=torch.bfloat16)


def _scope() -> dict[str, object]:
    roles: dict[str, int] = {}
    for spec in SELECTED_SPECS:
        role = spec.name.rsplit("/", 1)[-1]
        if spec.name.endswith("/mlp/down"):
            role = "mlp_down"
        elif spec.name.endswith("/gdn/value_z"):
            role = "gdn_value_z"
        elif spec.name.endswith("/gdn/output"):
            role = "gdn_output"
        roles[role] = roles.get(role, 0) + 1
    return {
        "selected_logical_matrix_count": len(SELECTED_SPECS),
        "selected_roles": dict(sorted(roles.items())),
        "selected_raw_source_tensor_count": len(SELECTED_SOURCE_ROWS),
        "selected_raw_source_row_count": SELECTED_RAW_SOURCE_ROWS,
        "remaining_text_weights": "unchanged-bfloat16-source",
        "raw_and_logical_equivalence": (
            "both codecs are per-output-row and commute with target recipe row slices, "
            "head reshapes, and row concatenations"
        ),
        "excludes_mtp": True,
    }


def _implementation() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPO / "tools/convert/qwen3_8_27b_r9700/mse_quantize.py",
        REPO / "tools/convert/qwen3_8_27b_r9700/screen_selective_q4g128_mse_quality.py",
        REPO / "tools/convert/qwen3_8_27b_r9700/source_inventory.py",
        REPO / "tools/convert/qwen3_8_27b_r9700/source_recipe.py",
        REPO / "tools/ppl/fp8_hybrid_source_diagnostic.py",
        REPO / "tools/ppl/q4_group_source_diagnostic.py",
        REPO / "tools/reference/qwen3_8_27b_bf16/backend.py",
        REPO / "tools/reference/qwen3_8_27b_bf16/protocol.py",
    )
    return {str(path.relative_to(REPO)): sha256_file(path) for path in paths}


def _quantization(codec: str) -> dict[str, object]:
    if codec not in CODECS:
        raise ValueError("selective source diagnostic codec differs")
    return {
        "selected": (
            "signed-q4g64-rne-clamp-minus8-plus7-fp16-absmax-scale-decoded-to-bfloat16"
            if codec == "q4g64-absmax" else
            "signed-q4g128-eight-step-source-mse-rne-clamp-minus8-plus7-"
            "fp16-scale-decoded-to-bfloat16"
        ),
        "remaining_text_weights": "unchanged-bfloat16-source",
        "scale_selection": (
            "canonical FP16(absmax/7)"
            if codec == "q4g64-absmax" else
            "earliest minimum decoded-weight SSE over canonical plus eight ALS steps"
        ),
        "activation_dtype": "bfloat16-not-product-a8",
        "matrix_arithmetic": "unchanged-bf16-reference",
        "candidate_registered": False,
        "canonical_q4g64_row_chunk": ROW_CHUNK,
        "mse_q4g128_row_chunk": mse_quantize.ROW_CHUNK,
    }


def _load_screen(path: Path, weights: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    validate_screen_report(report)
    validate_screen_provenance(report, weights)
    decision = report["decision"]
    if (decision.get("sampled_source_gate_pass") is not True
            or decision.get("maximum_candidate_over_control_relative_l2")
            != MAX_CANDIDATE_OVER_CONTROL_RELATIVE_L2):
        raise ValueError("selective Q4G128 sampled source gate did not pass exactly")
    return report


def _load_bf16_authority(args: argparse.Namespace, source: Mapping[str, object]):
    bf16, bf16_nll, bf16_argmax = _load_bf16(args.bf16)
    bf16_key = _source_key({"source": {
        "config_sha256": bf16["source_config_sha256"],
        "index_sha256": bf16["source_index_sha256"],
        "shards_sha256": bf16["source_shards_sha256"],
        "corpus_ids_sha256": bf16["corpus_ids_sha256"],
    }})
    # Runtime source identity retains shard pairs as tuples; normalize through
    # its exact JSON representation before using the report-oriented join.
    normalized_source = json.loads(json.dumps(dict(source), allow_nan=False))
    if bf16_key != _source_key({"source": normalized_source}):
        raise ValueError("BF16 authority source/corpus differs")
    if (bf16.get("prompt_tokens") != TOKENS or bf16.get("skip_tokens") != TOKENS // 2
            or bf16.get("tokens_scored") != TOKENS // 2 - 1):
        raise ValueError("BF16 authority is not the exact 8K/half cell")
    return bf16, bf16_nll, bf16_argmax


def _gate(score_nll, score_argmax, bf16_nll, bf16_argmax) -> dict[str, object]:
    against = _against(bf16_nll, score_nll, bf16_argmax, score_argmax)
    passed = against["mean_nll_delta_pass"] and against["new_severe_positions_pass"]
    return {
        "quality_tier": "capacity-speed",
        "maximum_mean_nll_delta": MAXIMUM_MEAN_NLL_DELTA,
        "maximum_new_severe_rate": MAXIMUM_NEW_SEVERE_RATE,
        "against_bf16": against,
        "pass": passed,
        "next_if_pass": (
            "format/kernel feasibility only; no artifact conversion or product admission"
        ),
    }


def _float32(values) -> tuple[float, ...]:
    """Return the exact values retained by the raw float32 score sidecar."""

    return tuple(struct.unpack("<f", struct.pack("<f", float(value)))[0] for value in values)


def preflight_payload(args: argparse.Namespace, weight_map: dict[str, str],
                      source: Mapping[str, object]) -> dict[str, object]:
    validate_source_metadata(args.weights, weight_map)
    screen = _load_screen(args.source_screen, args.weights)
    _load_bf16_authority(args, source)
    return {
        "artifact_type": ARTIFACT_TYPE + "_preflight",
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_source_only_8k_gpu_diagnostic",
        "source": dict(source),
        "sampled_source_gate": {
            "path": str(args.source_screen.resolve()),
            "sha256": sha256_file(args.source_screen),
            "aggregate_relative_l2_ratio": screen["decision"][
                "aggregate_candidate_over_control_relative_l2"
            ],
        },
        "bf16_authority": {
            "path": str(args.bf16.resolve()), "sha256": sha256_file(args.bf16),
        },
        "workload": {"tokens": TOKENS, "skip": "half", "prefill_chunk": 4096,
                     "schedule": "prefill", "device": args.device},
        "matrix_scope": _scope(),
        "implementation_sha256": _implementation(),
        "limitations": [
            "No product artifact, registered format, or runtime kernel is created.",
            "Weights are decoded at BF16; activations and model arithmetic remain BF16.",
            "This does not create, execute, or validate an A8G128 activation profile.",
            "The gate is limited to the fixed 8192-token prefill/half corpus cell.",
        ],
    }


def run(args: argparse.Namespace) -> int:
    protocol.establish_deterministic_environment()
    ids = protocol.read_ids(args.ids, TOKENS)
    if len(ids) != TOKENS:
        raise ValueError("selective Q4G128 source diagnostic requires exactly 8192 tokens")
    weight_map = protocol.validate_checkpoint_files(args.weights)
    source = _source_identity(args.weights, weight_map, ids)
    _load_screen(args.source_screen, args.weights)
    bf16, bf16_nll, bf16_argmax = _load_bf16_authority(args, source)
    if args.preflight_only:
        from tools.ppl.q4_group_source_diagnostic import _atomic_new
        payload = preflight_payload(args, weight_map, source)
        _atomic_new(args.out, (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())
        return 0

    torch = __import__("torch")
    protocol.enable_strict_torch_determinism(torch)
    from tools.reference.qwen3_8_27b_bf16.backend import LayerMajorTextScorer, SourceCheckpoint

    checkpoint = SelectiveQ4Checkpoint(SourceCheckpoint(args.weights, weight_map), args.codec)
    checkpoint.validate_metadata()
    scorer = LayerMajorTextScorer(
        checkpoint, device_index=args.device, prefill_chunk=4096, schedule="prefill",
        skip_text="half", kv_value_group=None,
    )
    vectors = scorer.score(ids)
    retained_nlls = _float32(vectors.nlls)
    retained_argmax = tuple(int(value) for value in vectors.argmax)
    finite = [value for value in retained_nlls if math.isfinite(value)]
    if len(finite) != len(vectors.nlls):
        raise ValueError("selective Q4G128 source score contains nonfinite NLL")
    gate = _gate(retained_nlls, retained_argmax, bf16_nll, bf16_argmax)
    report = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_weight_codec_only_not_product_ppl",
        "codec_profile": args.codec,
        "source": source,
        "quantization": _quantization(args.codec),
        "matrix_scope": _scope(),
        "sampled_source_gate": {
            "path": str(args.source_screen.resolve()),
            "sha256": sha256_file(args.source_screen),
        },
        "bf16_authority": {
            "path": str(args.bf16.resolve()), "sha256": sha256_file(args.bf16),
        },
        "execution": protocol.execution_provenance(torch, args.device, stage_trace_enabled=False),
        "implementation_sha256": _implementation(),
        "workload": {"tokens": TOKENS, "skip": "half", "prefill_chunk": 4096,
                     "schedule": "prefill", "device": args.device},
        "result": {
            "tokens_scored": len(retained_nlls), "argmax_tokens": len(retained_argmax),
            "non_finite": 0,
            "terrible_tokens": sum(value >= TERRIBLE_NLL for value in finite),
            "sum_nll": sum(finite), "mean_nll": sum(finite) / len(finite),
            "max_nll": max(finite), "ppl": math.exp(sum(finite) / len(finite)),
            "score_seconds": vectors.score_seconds,
        },
        "quality_gate": gate,
        "limitations": [
            "This is a weight-grouping/codec diagnostic, not product A8Q4 execution.",
            "Activations remain BF16; no A8G128 activation profile is exercised or validated.",
            "Pass permits no artifact conversion or runtime-format admission.",
        ],
        "command": [str(Path(sys.argv[0])), *sys.argv[1:]],
    }
    _write_score(args.out, report, retained_nlls, retained_argmax)
    # The paired comparator owns the terminal gate.  Both arms must be retained
    # even when one arm is outside the quality tier.
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--codec", choices=CODECS, required=True)
    parser.add_argument("--source-screen", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--device", type=int, choices=(0,), default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"selective-q4g128-source-diagnostic: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
