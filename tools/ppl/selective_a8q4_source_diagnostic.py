#!/usr/bin/env python3
"""Activation-inclusive source/reference gate for the selective G128 profile.

Only the exact 160 selected Text projection roles use the represented grouped
formula.  The control is A8G64 by canonical Q4G64; the candidate is A8G128 by
eight-step source-MSE Q4G128.  Every group forms an exact integer dot, applies
represented FP16 activation and weight scales in FP32, and accumulates in group
order before one BF16 linear-output cast.  Every nonselected weight and Op stays
on the unchanged BF16 source scorer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import sys
from typing import Iterable, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.artifact.numeric import QuantFormat, get_format
from tools.convert.qwen3_8_27b_r9700 import mse_quantize
from tools.convert.qwen3_8_27b_r9700.screen_selective_q4g128_mse_quality import (
    Q4G128_DIAGNOSTIC,
)
from tools.ppl.compare_q4_group_source import _against
from tools.ppl.q4_group_source_diagnostic import (
    MAXIMUM_MEAN_NLL_DELTA, MAXIMUM_NEW_SEVERE_RATE, TERRIBLE_NLL,
    _source_identity, _write_score, sha256_file, validate_source_metadata,
)
from tools.ppl.selective_q4g128_source_diagnostic import (
    SELECTED_SOURCE_ROWS, TOKENS, _load_bf16_authority, _load_screen, _scope,
)
from tools.reference.qwen3_8_27b_bf16 import protocol


ARTIFACT_TYPE = "ninfer_qwen3_8_selective_a8q4_source_diagnostic"
SCHEMA_VERSION = 1
PROFILES = ("a8g64-q4g64-control", "a8g128-q4g128-mse")
WEIGHT_ROW_CHUNK = 128


@dataclass(frozen=True)
class QuantizedRowSpan:
    begin: int
    end: int
    codes: object
    scales: object


def _format(profile: str) -> tuple[int, QuantFormat, bool]:
    if profile == "a8g64-q4g64-control":
        spec = get_format("Q4G64_F16S")
        if not isinstance(spec, QuantFormat):
            raise RuntimeError("registered Q4G64 format is unavailable")
        return 64, spec, False
    if profile == "a8g128-q4g128-mse":
        return 128, Q4G128_DIAGNOSTIC, True
    raise ValueError("activation-inclusive profile differs")


def quantize_weight_rows(weight, profile: str, *, row_chunk: int = WEIGHT_ROW_CHUNK):
    """Return exact signed-Q4 codes and represented FP16 scales on the host."""

    import torch

    group, spec, refined = _format(profile)
    if (weight.device.type != "cpu" or weight.dtype != torch.bfloat16
            or weight.ndim != 2 or weight.shape[1] % group):
        raise ValueError("selected weight must be CPU BF16 rank two with exact group geometry")
    if type(row_chunk) is not int or row_chunk <= 0:
        raise ValueError("weight row chunk must be a positive integer")
    rows, columns = map(int, weight.shape)
    codes = torch.empty((rows, columns // group, group), dtype=torch.int8)
    scales = torch.empty((rows, columns // group), dtype=torch.float16)
    for begin in range(0, rows, row_chunk):
        end = min(rows, begin + row_chunk)
        grouped = weight[begin:end].float().numpy().reshape(end - begin, columns // group, group)
        if refined:
            scale = mse_quantize._optimize_group_scales(grouped, spec)
        else:
            maximum = np.max(np.abs(grouped), axis=2).astype(np.float32)
            scale = mse_quantize._canonical_fp16(
                maximum.astype(np.float64) / 7.0, maximum > 0, spec.name
            )
        code = mse_quantize._codes(grouped, scale, -8, 7)
        codes[begin:end].copy_(torch.from_numpy(code))
        scales[begin:end].copy_(torch.from_numpy(scale))
    return codes, scales


def quantize_activation(activation, group: int):
    """Return exact signed-A8 codes and represented FP16 absmax/127 scales."""

    import torch

    if (activation.dtype != torch.bfloat16 or activation.ndim < 2
            or activation.shape[-1] % group or group not in (64, 128)):
        raise ValueError("selected activation must be BF16 with exact G64/G128 geometry")
    matrix = activation.reshape(-1, activation.shape[-1])
    values = matrix.float().reshape(matrix.shape[0], matrix.shape[1] // group, group)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("selected activation contains NaN or infinity")
    maximum = values.abs().amax(dim=2)
    scales = (maximum / 127.0).to(torch.float16)
    smallest = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float16, device=activation.device),
        torch.tensor(1.0, dtype=torch.float16, device=activation.device),
    )
    scales = torch.where((maximum != 0) & (scales == 0), smallest, scales)
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("selected activation scale is nonfinite")
    scale32 = scales.float()
    codes = torch.where(
        scale32[..., None] == 0, torch.zeros_like(values),
        torch.round(values / scale32[..., None]).clamp(-127, 127),
    ).to(torch.int8)
    return codes, scales


def represented_linear(activation, span: QuantizedRowSpan, group: int):
    """Evaluate the complete selected grouped formula with one BF16 boundary."""

    import torch

    activation_shape = activation.shape
    matrix = activation.reshape(-1, activation_shape[-1])
    activation_codes, activation_scales = quantize_activation(matrix, group)
    weight_codes = span.codes
    weight_scales = span.scales
    if (weight_codes.device != activation.device or weight_scales.device != activation.device
            or weight_codes.shape[1:] != activation_codes.shape[1:]
            or weight_scales.shape != weight_codes.shape[:2]):
        raise ValueError("selected represented operands differ")
    rows = weight_codes.shape[0]
    total = torch.zeros((matrix.shape[0], rows), dtype=torch.float32,
                        device=activation.device)
    for index in range(weight_codes.shape[1]):
        # The integer magnitude is below 2^24 for both supported groups, so the
        # FP32 matmul is an exact realization of the signed integer dot.
        dot = activation_codes[:, index].float() @ weight_codes[:, index].float().t()
        factor = activation_scales[:, index].float()[:, None] * \
            weight_scales[:, index].float()[None, :]
        total = torch.addcmul(total, dot, factor)
    return total.to(torch.bfloat16).reshape(*activation_shape[:-1], rows)


class ActivationInclusiveCheckpoint:
    """BF16 checkpoint loader plus ephemeral selected-row represented operands."""

    def __init__(self, checkpoint, profile: str):
        self.checkpoint = checkpoint
        self.profile = profile
        self.group, _, _ = _format(profile)
        self.bindings: dict[int, tuple[object, tuple[QuantizedRowSpan, ...]]] = {}

    def validate_metadata(self) -> None:
        self.checkpoint.validate_metadata()

    def load(self, name, device):
        self.bindings = {}
        return self.checkpoint.load(name, device)

    def load_many(self, names: Iterable[str], device):
        import torch

        source = self.checkpoint.load_many(names, torch.device("cpu"))
        result = {}
        self.bindings = {}
        for name, tensor in source.items():
            target = tensor.to(device=device, dtype=torch.bfloat16)
            result[name] = target
            spans = []
            for begin, end in SELECTED_SOURCE_ROWS.get(name, ()):
                codes, scales = quantize_weight_rows(tensor[begin:end], self.profile)
                spans.append(QuantizedRowSpan(
                    begin, end, codes.to(device=device), scales.to(device=device)
                ))
            if spans:
                self.bindings[id(target)] = (target, tuple(spans))
        return result

    def embedding_rows(self, ids: list[int], device):
        self.bindings = {}
        return self.checkpoint.embedding_rows(ids, device)


class ScopedLinearDispatcher:
    """Install the selected formula only around one scorer invocation."""

    def __init__(self, backend, checkpoint: ActivationInclusiveCheckpoint):
        self.backend = backend
        self.checkpoint = checkpoint
        self.original = None

    def __enter__(self):
        self.original = self.backend._linear

        def dispatch(activation, weight):
            binding = self.checkpoint.bindings.get(id(weight))
            if binding is None or binding[0] is not weight:
                return self.original(activation, weight)
            parts = []
            cursor = 0
            for span in binding[1]:
                if cursor < span.begin:
                    parts.append(self.original(activation, weight[cursor:span.begin]))
                parts.append(represented_linear(activation, span, self.checkpoint.group))
                cursor = span.end
            if cursor < weight.shape[0]:
                parts.append(self.original(activation, weight[cursor:]))
            return parts[0] if len(parts) == 1 else __import__("torch").cat(parts, dim=-1)

        self.backend._linear = dispatch
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.backend._linear = self.original
        return False


def _implementation() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPO / "tools/ppl/selective_q4g128_source_diagnostic.py",
        REPO / "tools/convert/qwen3_8_27b_r9700/mse_quantize.py",
        REPO / "tools/reference/qwen3_8_27b_bf16/backend.py",
        REPO / "tools/reference/qwen3_8_27b_bf16/protocol.py",
    )
    return {str(path.relative_to(REPO)): sha256_file(path) for path in paths}


def _profile_contract(profile: str) -> dict[str, object]:
    group, _, refined = _format(profile)
    return {
        "activation": f"signed-A8G{group}-rne-clamp-minus127-plus127-fp16-absmax-scale",
        "weight": (
            "signed-Q4G64-rne-clamp-minus8-plus7-fp16-absmax-scale" if not refined else
            "signed-Q4G128-eight-step-source-mse-rne-clamp-minus8-plus7-fp16-scale"
        ),
        "integer_dot": f"exact signed int8-by-int4 K{group}",
        "group_epilogue": "FP32 dot times FP32(activation FP16 scale times weight FP16 scale)",
        "accumulation": "ascending-group FP32 addcmul",
        "output_boundary": "one BF16 cast after complete selected linear",
        "nonselected": "unchanged BF16 source linear and operations",
        "weight_row_chunk": WEIGHT_ROW_CHUNK,
    }


def _float32(values) -> tuple[float, ...]:
    return tuple(struct.unpack("<f", struct.pack("<f", float(value)))[0] for value in values)


def preflight_payload(args, weight_map, source) -> dict[str, object]:
    validate_source_metadata(args.weights, weight_map)
    screen = _load_screen(args.source_screen, args.weights)
    _load_bf16_authority(args, source)
    return {
        "artifact_type": ARTIFACT_TYPE + "_preflight",
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_activation_inclusive_8k_source_gate",
        "source": dict(source),
        "sampled_source_gate": {
            "path": str(args.source_screen.resolve()), "sha256": sha256_file(args.source_screen),
            "passed": screen["decision"]["sampled_source_gate_pass"],
        },
        "bf16_authority": {
            "path": str(args.bf16.resolve()), "sha256": sha256_file(args.bf16),
        },
        "profiles": {profile: _profile_contract(profile) for profile in PROFILES},
        "matrix_scope": _scope(),
        "workload": {"tokens": TOKENS, "skip": "half", "prefill_chunk": 4096,
                     "schedule": "prefill", "device": args.device,
                     "scored_positions": 4095},
        "acceptance": {"maximum_mean_nll_delta": MAXIMUM_MEAN_NLL_DELTA,
                       "maximum_new_severe_rate": MAXIMUM_NEW_SEVERE_RATE,
                       "maximum_new_severe_positions": 11,
                       "severe_threshold_nll": TERRIBLE_NLL},
        "implementation_sha256": _implementation(),
        "limitations": [
            "This executes a source/reference represented formula, not a product kernel.",
            "No artifact or registered Q4G128/A8G128 runtime format is created.",
            "No 32K score is authorized by this preparation.",
        ],
    }


def run(args) -> int:
    protocol.establish_deterministic_environment()
    ids = protocol.read_ids(args.ids, TOKENS)
    if len(ids) != TOKENS:
        raise ValueError("activation-inclusive source gate requires exactly 8192 tokens")
    weight_map = protocol.validate_checkpoint_files(args.weights)
    source = _source_identity(args.weights, weight_map, ids)
    _load_screen(args.source_screen, args.weights)
    _, bf16_nll, bf16_argmax = _load_bf16_authority(args, source)
    if args.preflight_only:
        from tools.ppl.q4_group_source_diagnostic import _atomic_new
        payload = preflight_payload(args, weight_map, source)
        _atomic_new(args.out, (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())
        return 0

    torch = __import__("torch")
    protocol.enable_strict_torch_determinism(torch)
    from tools.reference.qwen3_8_27b_bf16 import backend

    checkpoint = ActivationInclusiveCheckpoint(
        backend.SourceCheckpoint(args.weights, weight_map), args.profile
    )
    checkpoint.validate_metadata()
    scorer = backend.LayerMajorTextScorer(
        checkpoint, device_index=args.device, prefill_chunk=4096, schedule="prefill",
        skip_text="half", kv_value_group=None,
    )
    with ScopedLinearDispatcher(backend, checkpoint):
        vectors = scorer.score(ids)
    nlls = _float32(vectors.nlls)
    argmax = tuple(int(value) for value in vectors.argmax)
    if len(nlls) != 4095 or not all(math.isfinite(value) for value in nlls):
        raise ValueError("activation-inclusive score output differs")
    gate = _against(bf16_nll, nlls, bf16_argmax, argmax)
    report = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_represented_formula_not_product_ppl",
        "profile": args.profile,
        "formula": _profile_contract(args.profile),
        "source": source,
        "sampled_source_gate": {
            "path": str(args.source_screen.resolve()), "sha256": sha256_file(args.source_screen),
        },
        "bf16_authority": {
            "path": str(args.bf16.resolve()), "sha256": sha256_file(args.bf16),
        },
        "matrix_scope": _scope(),
        "workload": {"tokens": TOKENS, "skip": "half", "prefill_chunk": 4096,
                     "schedule": "prefill", "device": args.device,
                     "scored_positions": 4095},
        "execution": protocol.execution_provenance(torch, args.device, stage_trace_enabled=False),
        "implementation_sha256": _implementation(),
        "result": {"tokens_scored": len(nlls), "argmax_tokens": len(argmax),
                   "non_finite": 0,
                   "terrible_tokens": sum(value >= TERRIBLE_NLL for value in nlls),
                   "sum_nll": sum(nlls), "mean_nll": sum(nlls) / len(nlls),
                   "max_nll": max(nlls), "ppl": math.exp(sum(nlls) / len(nlls)),
                   "score_seconds": vectors.score_seconds},
        "against_bf16": gate,
        "command": [str(Path(sys.argv[0])), *sys.argv[1:]],
        "limitations": [
            "This is source/reference arithmetic, not a timed product kernel.",
            "Pass permits only a disconnected kernel qualification.",
        ],
    }
    _write_score(args.out, report, nlls, argmax)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
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
        print(f"selective-a8q4-source-diagnostic: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
