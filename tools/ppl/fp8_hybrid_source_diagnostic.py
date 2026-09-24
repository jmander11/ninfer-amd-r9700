#!/usr/bin/env python3
"""Source-weight-codec PPL diagnostic for the canonical FP8/Q4 hybrid.

The target-owned selection record and source recipes determine which raw BF16
rows become row-scaled E4M3. The inventory's Q4 rows become canonical Q4G64,
while its direct BF16 matrices remain unchanged. The represented weights are
then evaluated by the unchanged BF16 source scorer, so this isolates the
persistent weight codec and is not product A8 execution evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.convert.qwen3.common.recipe import (
    Concat, Expression, Reshape, Slice, SourceTensor, expression_shape,
    expression_sources,
)
from tools.convert.qwen3_8_27b_r9700 import (
    fp8_hybrid_decision, fp8_hybrid_inventory, source_inventory, source_recipe,
)
from tools.ppl.q4_group_source_diagnostic import (
    ROW_CHUNK, TERRIBLE_NLL, _atomic_new, _source_identity, _write_score,
    quantize_decode_q4, sha256_file, validate_source_metadata,
)
from tools.reference.qwen3_8_27b_bf16 import protocol


ARTIFACT_TYPE = "ninfer_qwen3_8_fp8_q4_hybrid_source_diagnostic"
COMPARISON_TYPE = "ninfer_qwen3_8_fp8_q4_hybrid_source_comparison"
SCHEMA_VERSION = 1
TOKENS = 8192
E4M3FN_MAX = 448.0
QUANTIZATION = {
    "selected": "row-scaled-e4m3fn-rne-fp32-scale-k128",
    "remaining": "signed-q4g64-rne-fp16-absmax-scale-k128",
    "decoded_weight_dtype": "bfloat16",
    "activation_dtype": "bfloat16-not-product-a8",
    "matrix_arithmetic": "unchanged-bf16-reference",
    "row_chunk": ROW_CHUNK,
}


@dataclass(frozen=True, slots=True)
class _RowSpan:
    source: str
    begin: int
    end: int


@dataclass(frozen=True, slots=True)
class _Lineage:
    shape: tuple[int, ...]
    spans: tuple[_RowSpan, ...]

    @property
    def row_count(self) -> int:
        return sum(span.end - span.begin for span in self.spans)


def _sublineage(source: _Lineage, begin: int, end: int) -> tuple[_RowSpan, ...]:
    result: list[_RowSpan] = []
    cursor = 0
    for span in source.spans:
        length = span.end - span.begin
        overlap_begin = max(begin, cursor)
        overlap_end = min(end, cursor + length)
        if overlap_begin < overlap_end:
            offset = overlap_begin - cursor
            result.append(_RowSpan(
                span.source, span.begin + offset,
                span.begin + offset + overlap_end - overlap_begin,
            ))
        cursor += length
    if sum(span.end - span.begin for span in result) != end - begin:
        raise ValueError("selected hybrid recipe lineage interval is incomplete")
    return tuple(result)


def _slice_lineage(source: _Lineage, axis: int, begin: int, end: int) -> _Lineage:
    if not 0 <= axis < len(source.shape):
        raise ValueError("selected hybrid recipe slice is outside the row dimensions")
    stride = math.prod(source.shape[axis + 1 :])
    period = source.shape[axis] * stride
    outer = math.prod(source.shape[:axis])
    selected = tuple(
        span
        for index in range(outer)
        for span in _sublineage(
            source, index * period + begin * stride,
            index * period + end * stride,
        )
    )
    shape = list(source.shape)
    shape[axis] = end - begin
    if sum(span.end - span.begin for span in selected) != math.prod(shape):
        raise ValueError("selected hybrid recipe slice lineage is incomplete")
    return _Lineage(tuple(shape), selected)


def _row_lineage(expression: Expression) -> _Lineage:
    """Map each logical output row to its unique raw source row."""

    if isinstance(expression, SourceTensor):
        if len(expression.shape) != 2:
            raise ValueError("selected hybrid object must derive from rank-two sources")
        return _Lineage(
            (expression.shape[0],),
            (_RowSpan(expression.name, 0, expression.shape[0]),),
        )
    if isinstance(expression, Reshape):
        source = _row_lineage(expression.source)
        source_shape = expression_shape(expression.source)
        if source_shape[-1] != expression.shape[-1]:
            raise ValueError("selected hybrid reshape does not preserve complete K rows")
        shape = expression.shape[:-1]
        if math.prod(shape) != source.row_count:
            raise ValueError("selected hybrid reshape changes the row count")
        return _Lineage(shape, source.spans)
    if isinstance(expression, Slice):
        source_shape = expression_shape(expression.source)
        if expression.axis == len(source_shape) - 1:
            raise ValueError("selected hybrid recipe slices the K dimension")
        return _slice_lineage(
            _row_lineage(expression.source), expression.axis,
            expression.begin, expression.end,
        )
    if isinstance(expression, Concat):
        if expression.axis != 0:
            raise ValueError("selected hybrid concat must join complete output rows")
        parts = tuple(_row_lineage(part) for part in expression.sources)
        if any(len(part.shape) != 1 for part in parts):
            raise ValueError("selected hybrid concat has unresolved row dimensions")
        return _Lineage(
            (sum(part.shape[0] for part in parts),),
            tuple(span for part in parts for span in part.spans),
        )
    raise ValueError(f"selected hybrid recipe uses unsupported transform {type(expression).__name__}")


def _source_rows(object_names: Iterable[str]) -> dict[str, tuple[tuple[int, int], ...]]:
    """Derive raw-source row intervals for the given logical artifact objects."""

    rows: dict[str, list[tuple[int, int]]] = {}
    for object_name in object_names:
        try:
            recipe = source_recipe.RECIPES_BY_NAME[object_name]
        except KeyError as error:
            raise ValueError(f"hybrid selection has no source recipe: {object_name}") from error
        lineage = _row_lineage(recipe.expression)
        if (len(lineage.shape) != 1
                or lineage.row_count != expression_shape(recipe.expression)[0]):
            raise ValueError(f"{object_name}: source-row lineage differs from logical rows")
        for span in lineage.spans:
            rows.setdefault(span.source, []).append((span.begin, span.end))
    result: dict[str, tuple[tuple[int, int], ...]] = {}
    for name, spans in rows.items():
        merged: list[tuple[int, int]] = []
        for begin, end in sorted(spans):
            if merged and begin < merged[-1][1]:
                raise ValueError(f"hybrid selection overlaps raw rows in {name}")
            if merged and begin == merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((begin, end))
        result[name] = tuple(merged)
    return result


TEXT_SPECS = fp8_hybrid_inventory.TENSOR_SPECS[:len(
    source_inventory.TEXT_CORE_TENSOR_SPECS
)]
SELECTED_SOURCE_ROWS = _source_rows(fp8_hybrid_decision.DECISION.matrix_names)
Q4_SOURCE_ROWS = _source_rows(
    spec.name for spec in TEXT_SPECS if spec.format == fp8_hybrid_inventory.Q4
)
_selected_row_count = sum(
    end - begin for spans in SELECTED_SOURCE_ROWS.values() for begin, end in spans
)
if len(SELECTED_SOURCE_ROWS) != 224 or _selected_row_count != 2_654_208:
    raise ValueError("hybrid selection raw-source row inventory differs")
for _name in SELECTED_SOURCE_ROWS.keys() & Q4_SOURCE_ROWS.keys():
    for _fp8_begin, _fp8_end in SELECTED_SOURCE_ROWS[_name]:
        if any(
            max(_fp8_begin, q4_begin) < min(_fp8_end, q4_end)
            for q4_begin, q4_end in Q4_SOURCE_ROWS[_name]
        ):
            raise ValueError(f"hybrid E4M3 and Q4 source rows overlap in {_name}")
DIRECT_SOURCE_NAMES = frozenset(
    source.name
    for spec in TEXT_SPECS
    if spec.format == "BF16"
    for source in expression_sources(source_recipe.RECIPES_BY_NAME[spec.name].expression)
)
if DIRECT_SOURCE_NAMES & (SELECTED_SOURCE_ROWS.keys() | Q4_SOURCE_ROWS.keys()):
    raise ValueError("hybrid lossy-codec rows overlap direct-BF16 source tensors")
_requirements = {item.name: item.shape for item in protocol.expected_text_tensors()}
for _name in SELECTED_SOURCE_ROWS.keys() | Q4_SOURCE_ROWS.keys():
    _spans = sorted((*SELECTED_SOURCE_ROWS.get(_name, ()), *Q4_SOURCE_ROWS.get(_name, ())))
    _cursor = 0
    for _begin, _end in _spans:
        if _begin != _cursor:
            raise ValueError(f"hybrid codec source rows are not complete in {_name}")
        _cursor = _end
    if _cursor != _requirements[_name][0]:
        raise ValueError(f"hybrid codec source rows do not cover {_name}")
_raw_rank_two_names = {
    name for name, shape in _requirements.items() if len(shape) == 2
}
_direct_rank_two_source_names = {
    name for name in DIRECT_SOURCE_NAMES if len(_requirements[name]) == 2
}
if _raw_rank_two_names != (
    SELECTED_SOURCE_ROWS.keys() | Q4_SOURCE_ROWS.keys() | _direct_rank_two_source_names
):
    raise ValueError("hybrid source routing does not classify every raw rank-two Text tensor")


def quantize_decode_e4m3(weight, *, row_chunk: int = ROW_CHUNK):
    """Return canonical row-scaled-E4M3 represented weights as BF16."""

    import torch

    if type(row_chunk) is not int or row_chunk <= 0:
        raise ValueError("E4M3 diagnostic row chunk must be a positive integer")
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise TypeError("E4M3 diagnostic input must be a rank-two BF16 tensor")
    if weight.device.type != "cpu":
        raise ValueError("exact source E4M3 coding requires a CPU BF16 tensor")
    rows, columns = weight.shape
    result = torch.empty_like(weight)
    for begin in range(0, rows, row_chunk):
        end = min(rows, begin + row_chunk)
        values = weight[begin:end].float().contiguous()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("E4M3 diagnostic source contains NaN or infinity")
        maximum = values.abs().amax(dim=1)
        scales = (maximum.to(torch.float64) / E4M3FN_MAX).to(torch.float32)
        if bool(((maximum > 0) & (~torch.isfinite(scales) | (scales <= 0))).any()):
            raise ValueError("E4M3 diagnostic scale is not finite and positive")
        normalized = torch.zeros_like(values)
        positive = scales > 0
        normalized[positive] = values[positive] / scales[positive].unsqueeze(1)
        normalized.clamp_(-E4M3FN_MAX, E4M3FN_MAX)
        decoded = normalized.to(torch.float8_e4m3fn).to(torch.float32)
        result[begin:end] = (decoded * scales.unsqueeze(1)).to(torch.bfloat16)
    return result


class Fp8HybridCheckpoint:
    """Exact codec view over the unchanged source checkpoint loader."""

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def validate_metadata(self) -> None:
        self.checkpoint.validate_metadata()

    @staticmethod
    def _represented(name, tensor):
        import torch

        if tensor.ndim != 2:
            return tensor
        fp8_spans = SELECTED_SOURCE_ROWS.get(name, ())
        q4_spans = Q4_SOURCE_ROWS.get(name, ())
        if not fp8_spans and not q4_spans:
            return tensor
        if max((end for _, end in (*fp8_spans, *q4_spans)), default=0) > tensor.shape[0]:
            raise ValueError(f"{name}: selected codec row is outside the source tensor")
        if fp8_spans == ((0, tensor.shape[0]),) and not q4_spans:
            return quantize_decode_e4m3(tensor)
        if q4_spans == ((0, tensor.shape[0]),) and not fp8_spans:
            return quantize_decode_q4(tensor, 64)
        result = tensor.clone()
        if fp8_spans:
            indices = torch.cat(tuple(
                torch.arange(begin, end, dtype=torch.long) for begin, end in fp8_spans
            ))
            result[indices] = quantize_decode_e4m3(tensor[indices])
        if q4_spans:
            indices = torch.cat(tuple(
                torch.arange(begin, end, dtype=torch.long) for begin, end in q4_spans
            ))
            result[indices] = quantize_decode_q4(tensor[indices], 64)
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
        return quantize_decode_q4(source, 64).to(device=device, dtype=torch.bfloat16)


def _matrix_scope() -> dict[str, object]:
    fp8 = fp8_hybrid_inventory.F8E4M3_ROW_F32S
    q4 = fp8_hybrid_inventory.Q4
    return {
        "raw_text_matrix_count": sum(
            len(item.shape) == 2 for item in protocol.expected_text_tensors()
        ),
        "logical_text_matrix_count": sum(len(spec.shape) == 2 for spec in TEXT_SPECS),
        "logical_e4m3_matrix_count": sum(spec.format == fp8 for spec in TEXT_SPECS),
        "logical_q4g64_matrix_count": sum(spec.format == q4 for spec in TEXT_SPECS),
        "logical_direct_bf16_matrix_count": sum(
            len(spec.shape) == 2 and spec.format == "BF16" for spec in TEXT_SPECS
        ),
        "selected_raw_source_tensor_count": len(SELECTED_SOURCE_ROWS),
        "selected_raw_source_row_count": sum(
            end - begin for spans in SELECTED_SOURCE_ROWS.values() for begin, end in spans
        ),
        "q4g64_raw_source_tensor_count": len(Q4_SOURCE_ROWS),
        "q4g64_raw_source_row_count": sum(
            end - begin for spans in Q4_SOURCE_ROWS.values() for begin, end in spans
        ),
        "raw_and_logical_equivalence": (
            "both lossy codecs are per-output-row and commute with the selected row "
            "slices, head reshapes, and row concatenations; direct BF16 rows are unchanged"
        ),
        "bounded_row_chunk": ROW_CHUNK,
    }


def preflight_payload(args: argparse.Namespace, weight_map: dict[str, str], ids: list[int]) -> dict:
    validate_source_metadata(args.weights, weight_map)
    decision = fp8_hybrid_decision.DECISION
    return {
        "artifact_type": ARTIFACT_TYPE + "_preflight",
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_source_weight_codec_diagnostic",
        "identity": {
            "weights_id": decision.weights_id,
            "recipe_id": decision.recipe_id,
            "selection_sha256": decision.selection_sha256,
        },
        "format_counts": fp8_hybrid_inventory.FORMAT_COUNTS,
        "source": _source_identity(args.weights, weight_map, ids),
        "workload": {"tokens": len(ids), "skip": "half", "prefill_chunk": 4096,
                     "schedule": "prefill", "device": args.device},
        "matrix_scope": _matrix_scope(),
        "implementation": {
            "diagnostic": sha256_file(Path(__file__).resolve()),
            "selection_authority": sha256_file(fp8_hybrid_decision._authority_path()),
            "bf16_backend": sha256_file(REPO / "tools/reference/qwen3_8_27b_bf16/backend.py"),
            "bf16_protocol": sha256_file(REPO / "tools/reference/qwen3_8_27b_bf16/protocol.py"),
        },
        "limitations": [
            "This proves source completeness and exact codec/recipe geometry without scoring.",
            "The later score uses BF16 activations and matmul, not product A8 execution.",
        ],
    }


def run_score(args: argparse.Namespace) -> int:
    protocol.establish_deterministic_environment()
    ids = protocol.read_ids(args.ids, TOKENS)
    if len(ids) != TOKENS:
        raise ValueError("hybrid source diagnostic requires exactly 8192 tokens")
    weight_map = protocol.validate_checkpoint_files(args.weights)
    if args.preflight_only:
        _atomic_new(
            args.out,
            (json.dumps(preflight_payload(args, weight_map, ids), indent=2,
                        allow_nan=False) + "\n").encode(),
        )
        return 0
    source = _source_identity(args.weights, weight_map, ids)
    torch = __import__("torch")
    protocol.enable_strict_torch_determinism(torch)
    from tools.reference.qwen3_8_27b_bf16.backend import LayerMajorTextScorer, SourceCheckpoint

    checkpoint = Fp8HybridCheckpoint(SourceCheckpoint(args.weights, weight_map))
    checkpoint.validate_metadata()
    scorer = LayerMajorTextScorer(
        checkpoint, device_index=args.device, prefill_chunk=4096, schedule="prefill",
        skip_text="half", kv_value_group=None,
    )
    vectors = scorer.score(ids)
    finite = [value for value in vectors.nlls if math.isfinite(value)]
    decision = fp8_hybrid_decision.DECISION
    report = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_weight_codec_only_not_product_ppl",
        "identity": {
            "weights_id": decision.weights_id,
            "recipe_id": decision.recipe_id,
            "selection_sha256": decision.selection_sha256,
        },
        "format_counts": fp8_hybrid_inventory.FORMAT_COUNTS,
        "quantization": QUANTIZATION,
        "matrix_scope": _matrix_scope(),
        "source": source,
        "execution": protocol.execution_provenance(torch, args.device, stage_trace_enabled=False),
        "implementation": {
            "diagnostic": sha256_file(Path(__file__).resolve()),
            "selection_authority": sha256_file(fp8_hybrid_decision._authority_path()),
            "bf16_backend": sha256_file(REPO / "tools/reference/qwen3_8_27b_bf16/backend.py"),
            "bf16_protocol": sha256_file(REPO / "tools/reference/qwen3_8_27b_bf16/protocol.py"),
        },
        "workload": {"tokens": TOKENS, "skip": "half", "prefill_chunk": 4096,
                     "schedule": "prefill", "device": args.device},
        "result": {
            "tokens_scored": len(vectors.nlls), "argmax_tokens": len(vectors.argmax),
            "non_finite": len(vectors.nlls) - len(finite),
            "terrible_tokens": sum(value >= TERRIBLE_NLL for value in finite),
            "sum_nll": sum(finite), "mean_nll": sum(finite) / len(finite),
            "max_nll": max(finite), "ppl": math.exp(sum(finite) / len(finite)),
            "score_seconds": vectors.score_seconds,
        },
        "command": [str(Path(sys.argv[0])), *sys.argv[1:]],
    }
    _write_score(args.out, report, vectors.nlls, vectors.argmax)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return run_score(build_parser().parse_args(argv))
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"fp8-hybrid-source-diagnostic: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
