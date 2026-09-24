"""CPU-only represented-source comparison for Q4G64 and row-scaled signed Q4."""

from __future__ import annotations

import math

import torch


QMIN = -8
QMAX = 7
GROUP_SIZE = 64


def _represented_scale(maximum: torch.Tensor) -> torch.Tensor:
    """Return the canonical FP16 absmax/QMAX scale represented as FP32."""

    scale = (maximum.to(torch.float32) / float(QMAX)).to(torch.float16)
    # Match the artifact codec's nonzero-underflow rule.
    smallest = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float16),
        torch.tensor(1.0, dtype=torch.float16),
    )
    scale = torch.where((maximum != 0) & (scale == 0), smallest, scale)
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("Q4 scale is not finite")
    return scale.to(torch.float32)


def quantize_dequantize(source: torch.Tensor, *, group_size: int | None) -> torch.Tensor:
    """Quantize BF16 rank-two rows to signed Q4 and return represented FP32 values.

    ``group_size=None`` selects one scale for the complete logical row.  A positive
    group size selects independent contiguous groups and requires exact divisibility.
    """

    if source.dtype != torch.bfloat16 or source.ndim != 2:
        raise TypeError("source must be a rank-two BF16 CPU tensor")
    if source.device.type != "cpu" or not bool(torch.isfinite(source).all()):
        raise ValueError("source must contain finite CPU values")
    rows, columns = source.shape
    width = columns if group_size is None else group_size
    if width <= 0:
        raise ValueError("Q4 group width must be positive")
    values = source.to(torch.float32)
    padded_columns = columns
    if group_size is not None:
        # Q4G64 artifact rows use canonical K128 physical padding.  Padding is
        # represented zero input and cannot affect an absmax scale.
        alignment = math.lcm(128, width)
        padded_columns = ((columns + alignment - 1) // alignment) * alignment
        if padded_columns != columns:
            values = torch.cat(
                (values, torch.zeros((rows, padded_columns - columns), dtype=torch.float32)),
                dim=1,
            )
    grouped = values.reshape(rows, padded_columns // width, width)
    scale = _represented_scale(grouped.abs().amax(dim=2, keepdim=True))
    codes = torch.where(
        scale == 0,
        torch.zeros_like(grouped),
        torch.round(grouped / scale).clamp(QMIN, QMAX),
    )
    return (codes * scale).reshape(rows, padded_columns)[:, :columns]


def error_metrics(source: torch.Tensor, decoded: torch.Tensor) -> dict[str, float]:
    if source.shape != decoded.shape:
        raise ValueError("source and decoded shapes differ")
    reference = source.to(torch.float64)
    difference = decoded.to(torch.float64) - reference
    squared_error = float(difference.square().sum().item())
    squared_reference = float(reference.square().sum().item())
    return {
        "squared_error": squared_error,
        "squared_reference": squared_reference,
        "relative_l2": math.sqrt(squared_error / squared_reference)
        if squared_reference else 0.0,
        "max_abs": float(difference.abs().max().item()) if difference.numel() else 0.0,
    }


def compare_sample(source: torch.Tensor) -> dict[str, dict[str, float] | float | None]:
    grouped = error_metrics(
        source, quantize_dequantize(source, group_size=GROUP_SIZE)
    )
    row = error_metrics(source, quantize_dequantize(source, group_size=None))
    return {
        "q4g64": grouped,
        "q4_row_scaled": row,
        "ratios": {
            "relative_l2": row["relative_l2"] / grouped["relative_l2"]
            if grouped["relative_l2"] else None,
            "max_abs": row["max_abs"] / grouped["max_abs"]
            if grouped["max_abs"] else None,
        },
    }


__all__ = ["GROUP_SIZE", "compare_sample", "error_metrics", "quantize_dequantize"]
