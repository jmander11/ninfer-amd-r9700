#!/usr/bin/env python3
"""Source-only Q4 group-size PPL diagnostic for Qwen3.8-27B.

This is deliberately not a product scorer.  It replaces each rank-two Text
weight with the BF16 decode of canonical signed-Q4/FP16-scale bytes, while
leaving activations and matrix arithmetic on the independent BF16 path.  The
paired G64/G128 result is therefore an implementation-free weight-codec gate,
not evidence for the product A8Q4 execution profile.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.reference.qwen3_8_27b_bf16 import protocol


ARTIFACT_TYPE = "ninfer_qwen3_8_q4_group_source_diagnostic"
COMPARISON_TYPE = "ninfer_qwen3_8_q4_group_source_comparison"
SCHEMA_VERSION = 1
GROUPS = (64, 128)
ROW_CHUNK = 4096
TERRIBLE_NLL = 10.0
MAXIMUM_MEAN_NLL_DELTA = math.log(1.05)
MAXIMUM_NEW_SEVERE_RATE = 0.0025


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantize_decode_q4(weight, group: int, *, row_chunk: int = ROW_CHUNK):
    """Return canonical absmax Q4/FP16-scale represented weights as BF16.

    Rows are independent, so chunking changes neither codes nor scales.  The
    implementation uses torch operations on the input device and round-to-even
    through torch.round, matching the existing source codec.
    """

    import torch

    if type(group) is not int or group not in GROUPS:
        raise ValueError("Q4 diagnostic group must be exactly 64 or 128")
    if type(row_chunk) is not int or row_chunk <= 0:
        raise ValueError("row chunk must be a positive integer")
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise TypeError("Q4 diagnostic input must be a rank-two BF16 tensor")
    rows, columns = weight.shape
    if columns % group or columns % 128:
        raise ValueError("Q4 diagnostic matrix K must be divisible by group and 128")
    result = torch.empty_like(weight)
    smallest = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float16, device=weight.device),
        torch.tensor(1.0, dtype=torch.float16, device=weight.device),
    )
    for begin in range(0, rows, row_chunk):
        end = min(rows, begin + row_chunk)
        values = weight[begin:end].float().reshape(end - begin, columns // group, group)
        maximum = values.abs().amax(dim=2, keepdim=True)
        scale16 = (maximum / 7.0).to(torch.float16)
        scale16 = torch.where((maximum != 0) & (scale16 == 0), smallest, scale16)
        if not bool(torch.isfinite(scale16).all()):
            raise ValueError("Q4 diagnostic scale is not finite")
        scale = scale16.float()
        codes = torch.where(
            scale == 0,
            torch.zeros_like(values),
            torch.round(values / scale).clamp(-8, 7),
        )
        result[begin:end] = (codes * scale).reshape(end - begin, columns).to(torch.bfloat16)
    return result


class GroupQuantizedCheckpoint:
    """Weight-only view over the unchanged BF16 checkpoint loader."""

    def __init__(self, checkpoint, group: int):
        self.checkpoint = checkpoint
        self.group = group

    def validate_metadata(self) -> None:
        self.checkpoint.validate_metadata()

    def _represented(self, tensor):
        return quantize_decode_q4(tensor, self.group) if tensor.ndim == 2 else tensor

    def load(self, name, device):
        return self._represented(self.checkpoint.load(name, device))

    def load_many(self, names: Iterable[str], device):
        return {
            name: self._represented(tensor)
            for name, tensor in self.checkpoint.load_many(names, device).items()
        }

    def embedding_rows(self, ids: list[int], device):
        # Q4 scales are per output row, so selecting token rows before coding is exact.
        return self._represented(self.checkpoint.embedding_rows(ids, device))


def _atomic_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _sidecar(path: Path, suffix: str) -> Path:
    return path.with_suffix(suffix)


def _write_score(path: Path, report: dict, nlls: tuple[float, ...], argmax: tuple[int, ...]) -> None:
    targets = (path, _sidecar(path, ".nllf32"), _sidecar(path, ".argmaxi32"))
    if any(target.exists() or target.is_symlink() for target in targets):
        raise FileExistsError("refusing to overwrite Q4 group diagnostic output")
    nll_bytes = struct.pack("<" + "f" * len(nlls), *nlls)
    argmax_bytes = struct.pack("<" + "i" * len(argmax), *argmax)
    report["sidecars"] = {
        "nll": {"path": targets[1].name, "sha256": hashlib.sha256(nll_bytes).hexdigest()},
        "argmax": {"path": targets[2].name, "sha256": hashlib.sha256(argmax_bytes).hexdigest()},
    }
    created: list[Path] = []
    try:
        _atomic_new(targets[1], nll_bytes); created.append(targets[1])
        _atomic_new(targets[2], argmax_bytes); created.append(targets[2])
        _atomic_new(path, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    except BaseException:
        for target in created:
            target.unlink(missing_ok=True)
        raise


def _source_identity(weights: Path, weight_map: dict[str, str], ids: list[int]) -> dict:
    return {
        "root": str(weights.resolve()),
        "config_sha256": sha256_file(weights / "config.json"),
        "index_sha256": sha256_file(weights / "model.safetensors.index.json"),
        "shards_sha256": protocol.source_shard_sha256(weights, weight_map),
        "tensor_count": len(weight_map),
        "text_tensor_count": len(protocol.expected_text_tensors()),
        "shard_count": len(set(weight_map.values())),
        "corpus_ids_sha256": protocol.ids_sha256(ids),
    }


def validate_source_metadata(weights: Path, weight_map: dict[str, str]) -> None:
    from safetensors import safe_open
    by_shard: dict[str, list] = defaultdict(list)
    for requirement in protocol.expected_text_tensors():
        by_shard[weight_map[requirement.name]].append(requirement)
    for shard, requirements in by_shard.items():
        with safe_open(str(weights / shard), framework="numpy") as handle:
            for requirement in requirements:
                tensor = handle.get_slice(requirement.name)
                if (tuple(tensor.get_shape()) != requirement.shape
                        or str(tensor.get_dtype()) != requirement.dtype):
                    raise ValueError(f"{requirement.name} source metadata differs")


def preflight_payload(args: argparse.Namespace, weight_map: dict[str, str], ids: list[int]) -> dict:
    matrices = [item for item in protocol.expected_text_tensors() if len(item.shape) == 2]
    if len(matrices) != 498:
        raise ValueError(f"expected 498 raw Text matrices, got {len(matrices)}")
    if any(item.shape[1] % args.group or item.shape[1] % 128 for item in matrices):
        raise ValueError("a source Text matrix is incompatible with the selected Q4 group")
    validate_source_metadata(args.weights, weight_map)
    elements = sum(item.shape[0] * item.shape[1] for item in matrices)
    groups = sum(item.shape[0] * (item.shape[1] // args.group) for item in matrices)
    largest = max(item.shape[0] * item.shape[1] for item in matrices)
    largest_chunk = max(min(ROW_CHUNK, item.shape[0]) * item.shape[1] for item in matrices)
    return {
        "artifact_type": ARTIFACT_TYPE + "_preflight",
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_source_only_gpu_diagnostic",
        "group_size": args.group,
        "source": _source_identity(args.weights, weight_map, ids),
        "workload": {"tokens": len(ids), "skip": "half", "prefill_chunk": 4096,
                     "schedule": "prefill", "device": args.device},
        "matrix_scope": {
            "raw_text_matrix_count": len(matrices),
            "logical_product_q4_matrix_count": 322,
            "raw_and_logical_equivalence": (
                "per-output-row quantization commutes with product row slicing/concatenation"
            ),
            "elements": elements,
            "represented_code_bytes": (elements + 1) // 2,
            "represented_scale_bytes": groups * 2,
            "largest_source_bf16_matrix_bytes": largest * 2,
            "largest_quantization_chunk_elements": largest_chunk,
            "bounded_row_chunk": ROW_CHUNK,
        },
        "implementation": {
            "diagnostic": sha256_file(Path(__file__).resolve()),
            "bf16_backend": sha256_file(REPO / "tools/reference/qwen3_8_27b_bf16/backend.py"),
            "bf16_protocol": sha256_file(REPO / "tools/reference/qwen3_8_27b_bf16/protocol.py"),
        },
        "limitations": [
            "This proves source completeness and codec geometry without importing an accelerator runtime.",
            "The later score is a weight-codec diagnostic with BF16 activations, not product A8Q4 PPL.",
        ],
    }


def run_score(args: argparse.Namespace) -> int:
    protocol.establish_deterministic_environment()
    ids = protocol.read_ids(args.ids, args.tokens)
    if len(ids) not in (8192, 32768):
        raise ValueError("source Q4 group diagnostic is bounded to 8192 or 32768 tokens")
    weight_map = protocol.validate_checkpoint_files(args.weights)
    if args.preflight_only:
        _atomic_new(
            args.out,
            (json.dumps(preflight_payload(args, weight_map, ids), indent=2, allow_nan=False)
             + "\n").encode(),
        )
        return 0
    source = _source_identity(args.weights, weight_map, ids)
    torch = __import__("torch")
    protocol.enable_strict_torch_determinism(torch)
    from tools.reference.qwen3_8_27b_bf16.backend import LayerMajorTextScorer, SourceCheckpoint

    checkpoint = GroupQuantizedCheckpoint(SourceCheckpoint(args.weights, weight_map), args.group)
    checkpoint.validate_metadata()
    scorer = LayerMajorTextScorer(
        checkpoint, device_index=args.device, prefill_chunk=4096, schedule="prefill",
        skip_text="half", kv_value_group=None,
    )
    vectors = scorer.score(ids)
    finite = [value for value in vectors.nlls if math.isfinite(value)]
    report = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_weight_codec_only_not_product_ppl",
        "group_size": args.group,
        "quantization": {
            "codes": "signed-q4-rne-clamp-minus8-plus7",
            "scale": "fp16(absmax/7)-per-output-row-group",
            "decoded_weight_dtype": "bfloat16",
            "activation_dtype": "bfloat16-not-product-a8",
            "matrix_arithmetic": "unchanged-bf16-reference",
            "row_chunk": ROW_CHUNK,
        },
        "source": source,
        "execution": protocol.execution_provenance(torch, args.device, stage_trace_enabled=False),
        "implementation": {
            "diagnostic": sha256_file(Path(__file__).resolve()),
            "bf16_backend": sha256_file(REPO / "tools/reference/qwen3_8_27b_bf16/backend.py"),
            "bf16_protocol": sha256_file(REPO / "tools/reference/qwen3_8_27b_bf16/protocol.py"),
        },
        "workload": {"tokens": len(ids), "skip": "half", "prefill_chunk": 4096,
                     "schedule": "prefill", "device": args.device},
        "result": {
            "tokens_scored": len(vectors.nlls), "argmax_tokens": len(vectors.argmax),
            "non_finite": len(vectors.nlls) - len(finite),
            "terrible_tokens": sum(value >= TERRIBLE_NLL for value in finite),
            "sum_nll": sum(finite),
            "mean_nll": sum(finite) / len(finite),
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
    parser.add_argument("--group", type=int, choices=GROUPS, required=True)
    parser.add_argument("--tokens", type=int, choices=(8192, 32768), default=8192)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="validate/hash the complete source and matrix geometry without importing torch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return run_score(build_parser().parse_args(argv))
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"q4-group-source-diagnostic: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
