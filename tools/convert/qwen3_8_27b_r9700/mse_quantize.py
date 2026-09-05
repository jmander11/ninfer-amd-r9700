"""Deterministic source-only grouped scale refinement for evaluation.

The stored formats are unchanged: signed codes and one FP16 scale per registered
group. Starting from the canonical absmax scale, eight alternating least-squares
steps produce candidate FP16 scales. The selected scale is the earliest candidate
with minimum decoded-weight squared error, so the absmax result is retained on a
tie and no group can regress under the stated objective.

Scale fitting always runs on represented source values on the host. It does not
consume calibration activations, text, token labels, PPL output, or GPU results.
"""

from __future__ import annotations

import numpy as np
import torch

from tools.artifact.layouts import (encode_q4_n16k16, encode_row_split,
                                    q4_n16k16_geometry, row_split_geometry)
from tools.artifact.numeric import QuantFormat, get_format
from tools.convert.common.quantize import QuantizedMatrix, pick_device


REFINEMENT_STEPS = 8
ROW_CHUNK = 128
_FP16_MIN_SUBNORMAL = np.float16(2.0**-24)


def _canonical_fp16(
    raw_scale: np.ndarray, nonzero: np.ndarray, format_name: str
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        scale = raw_scale.astype(np.float32).astype(np.float16)
    underflow = (scale == 0) & nonzero
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = _FP16_MIN_SUBNORMAL
    if np.any(nonzero & (~np.isfinite(scale) | (scale <= 0))):
        raise ValueError(f"{format_name} MSE scale is not finite and positive")
    return scale


def _reciprocal(scale: np.ndarray) -> np.ndarray:
    reciprocal = np.zeros(scale.shape, dtype=np.float32)
    positive = scale > 0
    reciprocal[positive] = (
        1.0 / scale[positive].astype(np.float64)
    ).astype(np.float32)
    return reciprocal


def _codes(
    grouped: np.ndarray, scale: np.ndarray, qmin: int, qmax: int
) -> np.ndarray:
    reciprocal = _reciprocal(scale)
    return np.clip(
        np.rint(grouped * reciprocal[..., None]), qmin, qmax
    ).astype(np.int8)


def _decoded_sse(
    grouped: np.ndarray, codes: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    error = grouped.astype(np.float64) - (
        codes.astype(np.float64) * scale.astype(np.float64)[..., None]
    )
    return np.sum(error * error, axis=2, dtype=np.float64)


def _optimize_group_scales(
    grouped: np.ndarray, spec: QuantFormat
) -> np.ndarray:
    if (
        grouped.dtype != np.float32
        or grouped.ndim != 3
        or grouped.shape[2] != spec.group_size
    ):
        raise ValueError(
            f"{spec.name} MSE fitting requires float32 groups of {spec.group_size}"
        )
    if not np.isfinite(grouped).all():
        raise ValueError("grouped quantization source contains NaN or infinity")

    maximum = np.max(np.abs(grouped), axis=2).astype(np.float32)
    nonzero = maximum > 0
    current = _canonical_fp16(
        maximum.astype(np.float64) / float(spec.qmax), nonzero, spec.name
    )
    baseline_codes = _codes(grouped, current, spec.qmin, spec.qmax)
    best = current.copy()
    best_sse = _decoded_sse(grouped, baseline_codes, current)

    for _ in range(REFINEMENT_STEPS):
        current_codes = _codes(grouped, current, spec.qmin, spec.qmax)
        numerator = np.sum(
            grouped.astype(np.float64) * current_codes.astype(np.float64),
            axis=2,
            dtype=np.float64,
        )
        denominator = np.sum(
            current_codes.astype(np.int64) * current_codes.astype(np.int64),
            axis=2,
            dtype=np.int64,
        )
        raw = np.divide(
            numerator,
            denominator,
            out=np.zeros(numerator.shape, dtype=np.float64),
            where=denominator != 0,
        )
        current = _canonical_fp16(raw, nonzero, spec.name)
        candidate_codes = _codes(grouped, current, spec.qmin, spec.qmax)
        candidate_sse = _decoded_sse(grouped, candidate_codes, current)
        improved = candidate_sse < best_sse
        best[improved] = current[improved]
        best_sse[improved] = candidate_sse[improved]

    return best


def _optimize_scales(
    weight: torch.Tensor,
    spec: QuantFormat,
    *,
    row_chunk: int = ROW_CHUNK,
) -> torch.Tensor:
    """Return host FP16 scales minimizing the fixed format trajectory."""

    if weight.dim() != 2:
        raise ValueError(f"grouped quantization requires rank 2, got {tuple(weight.shape)}")
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")
    if row_chunk <= 0:
        raise ValueError(f"{spec.name} MSE row chunk must be positive")
    if spec.name == "Q4G64_F16S":
        n, k = map(int, weight.shape)
        k_pad = ((k + 127) // 128) * 128
        groups_per_row = k_pad // spec.group_size
    else:
        geometry = row_split_geometry(spec, weight.shape)
        n, k, k_pad = geometry.n, geometry.k, geometry.k_pad
        groups_per_row = geometry.groups_per_row

    scales = np.empty((n, groups_per_row), dtype=np.float16)
    for row_begin in range(0, n, row_chunk):
        row_end = min(row_begin + row_chunk, n)
        logical = (
            weight[row_begin:row_end]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
        if k_pad != k:
            physical = np.zeros((row_end - row_begin, k_pad), dtype=np.float32)
            physical[:, :k] = logical
            logical = physical
        grouped = logical.reshape(
            row_end - row_begin, groups_per_row, spec.group_size
        )
        scales[row_begin:row_end] = _optimize_group_scales(grouped, spec)
    return torch.from_numpy(scales)


def optimize_w8g32_scales(
    weight: torch.Tensor,
    *,
    row_chunk: int = ROW_CHUNK,
) -> torch.Tensor:
    """Return host FP16 W8G32 scales from the fixed refinement trajectory."""

    spec = get_format("W8G32_F16S")
    if not isinstance(spec, QuantFormat):
        raise RuntimeError("W8G32 is not a grouped quantized format")
    return _optimize_scales(weight, spec, row_chunk=row_chunk)


def optimize_q4g64_scales(
    weight: torch.Tensor,
    *,
    row_chunk: int = ROW_CHUNK,
) -> torch.Tensor:
    """Return host FP16 Q4G64 scales from the fixed refinement trajectory."""

    spec = get_format("Q4G64_F16S")
    if not isinstance(spec, QuantFormat):
        raise RuntimeError("Q4G64 is not a grouped quantized format")
    return _optimize_scales(weight, spec, row_chunk=row_chunk)


def _quantize_matrix_mse(
    weight: torch.Tensor,
    spec: QuantFormat,
    *,
    device: str | torch.device,
) -> QuantizedMatrix:
    """Quantize one registered format with source-only refined group scales."""

    if weight.dim() != 2:
        raise ValueError(f"grouped quantization requires rank 2, got {tuple(weight.shape)}")
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")
    if spec.name == "Q4G64_F16S":
        n, k = map(int, weight.shape)
        k_pad = ((k + 127) // 128) * 128
        groups_per_row = k_pad // spec.group_size
    else:
        geometry = row_split_geometry(spec, weight.shape)
        n, k, k_pad = geometry.n, geometry.k, geometry.k_pad
        groups_per_row = geometry.groups_per_row
    target = pick_device(device)
    host_scales = _optimize_scales(weight, spec)
    host_reciprocal = torch.from_numpy(_reciprocal(host_scales.numpy()))

    logical = weight.detach().to(device=target, dtype=torch.float32)
    if k_pad != k:
        physical = torch.zeros(
            (n, k_pad), dtype=torch.float32, device=target
        )
        physical[:, :k].copy_(logical)
        logical = physical
    grouped = logical.reshape(
        n, groups_per_row, spec.group_size
    )
    reciprocal = host_reciprocal.to(target)
    codes = torch.clamp(
        torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    ).to(torch.int8)
    return QuantizedMatrix(codes=codes, scales=host_scales.to(target))


def quantize_matrix_mse(
    weight: torch.Tensor,
    *,
    device: str | torch.device,
) -> QuantizedMatrix:
    """Quantize W8G32 with source-only MSE-refined FP16 group scales."""

    spec = get_format("W8G32_F16S")
    if not isinstance(spec, QuantFormat):
        raise RuntimeError("W8G32 is not a grouped quantized format")
    return _quantize_matrix_mse(weight, spec, device=device)


def quantize_q4g64_matrix_mse(
    weight: torch.Tensor,
    *,
    device: str | torch.device,
) -> QuantizedMatrix:
    """Quantize Q4G64 with source-only MSE-refined FP16 group scales."""

    spec = get_format("Q4G64_F16S")
    if not isinstance(spec, QuantFormat):
        raise RuntimeError("Q4G64 is not a grouped quantized format")
    return _quantize_matrix_mse(weight, spec, device=device)


def quantize_and_encode_mse(
    weight: torch.Tensor,
    *,
    device: str | torch.device,
) -> bytes:
    """Encode one logical matrix as unchanged row-split W8G32 bytes."""

    spec = get_format("W8G32_F16S")
    if not isinstance(spec, QuantFormat):
        raise RuntimeError("W8G32 is not a grouped quantized format")
    quantized = quantize_matrix_mse(weight, device=device)
    return encode_row_split(quantized.codes, quantized.scales, spec, weight.shape)


def quantize_and_encode_q4g64_mse(
    weight: torch.Tensor,
    *,
    device: str | torch.device,
) -> bytes:
    """Encode one logical matrix as production N16/K16 Q4G64 bytes."""

    spec = get_format("Q4G64_F16S")
    if not isinstance(spec, QuantFormat):
        raise RuntimeError("Q4G64 is not a grouped quantized format")
    quantized = quantize_q4g64_matrix_mse(weight, device=device)
    return encode_q4_n16k16(quantized.codes, quantized.scales, weight.shape)


__all__ = [
    "REFINEMENT_STEPS",
    "optimize_q4g64_scales",
    "optimize_w8g32_scales",
    "quantize_and_encode_mse",
    "quantize_and_encode_q4g64_mse",
    "quantize_q4g64_matrix_mse",
    "quantize_matrix_mse",
]
