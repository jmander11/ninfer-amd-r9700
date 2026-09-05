"""CPU encoder/decoder for the isolated rowwise-E4M3 weight candidate."""

from __future__ import annotations

import math
import struct
from typing import Iterator

import numpy as np
import torch


K_ALIGNMENT = 128
PLANE_ALIGNMENT = 256
E4M3FN_MAX = 448.0


def _geometry(rows: int, columns: int) -> tuple[int, int]:
    if type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0:
        raise ValueError("E4M3 rowwise shape requires positive integer rows and columns")
    padded_columns = (columns + K_ALIGNMENT - 1) // K_ALIGNMENT * K_ALIGNMENT
    code_bytes = rows * padded_columns
    scale_offset = (code_bytes + PLANE_ALIGNMENT - 1) // PLANE_ALIGNMENT * PLANE_ALIGNMENT
    return padded_columns, scale_offset + rows * 4


def _validate_weight(weight: torch.Tensor) -> tuple[int, int, int]:
    if weight.dim() != 2 or any(dimension <= 0 for dimension in weight.shape):
        raise ValueError(
            f"E4M3 rowwise encoding requires a nonempty rank-2 matrix, got {tuple(weight.shape)}"
        )
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"E4M3 rowwise encoding requires represented BF16, got {weight.dtype}")
    if weight.device.type != "cpu":
        raise ValueError("E4M3 rowwise CPU encoder requires a CPU source tensor")
    rows, columns = weight.shape
    padded_columns, _ = _geometry(rows, columns)
    return rows, columns, padded_columns


def encode_e4m3_rowwise_chunks(
    weight: torch.Tensor, *, rows_per_chunk: int = 256
) -> Iterator[bytes]:
    """Yield codes then FP32 scales while bounding temporary encoding storage."""

    rows, columns, padded_columns = _validate_weight(weight)
    if type(rows_per_chunk) is not int or rows_per_chunk <= 0:
        raise ValueError("E4M3 rowwise rows_per_chunk must be a positive integer")
    scale_payload = bytearray()
    for row_begin in range(0, rows, rows_per_chunk):
        row_end = min(row_begin + rows_per_chunk, rows)
        logical = (
            weight[row_begin:row_end]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        if not bool(torch.isfinite(logical).all()):
            raise ValueError("E4M3 rowwise source contains NaN or infinity")

        physical = torch.zeros((row_end - row_begin, padded_columns), dtype=torch.float32)
        physical[:, :columns].copy_(logical)
        host_max = logical.abs().amax(dim=1).numpy().astype(np.float32, copy=False)
        scales_array = (host_max.astype(np.float64) / E4M3FN_MAX).astype(np.float32)
        if np.any((host_max > 0) & (~np.isfinite(scales_array) | (scales_array <= 0))):
            raise ValueError("E4M3 rowwise scale is not finite and positive")
        scales = torch.from_numpy(scales_array)
        normalized = torch.zeros_like(physical)
        positive = scales > 0
        normalized[positive] = physical[positive] / scales[positive].unsqueeze(1)
        normalized.clamp_(-E4M3FN_MAX, E4M3FN_MAX)
        codes = normalized.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        yield codes.numpy().tobytes()
        scale_payload.extend(scales_array.astype("<f4", copy=False).tobytes())

    code_bytes = rows * padded_columns
    scale_offset = (code_bytes + PLANE_ALIGNMENT - 1) // PLANE_ALIGNMENT * PLANE_ALIGNMENT
    if scale_offset != code_bytes:
        yield bytes(scale_offset - code_bytes)
    yield bytes(scale_payload)


def encode_e4m3_rowwise(weight: torch.Tensor) -> bytes:
    """Encode represented BF16 ``[N,K]`` weights without device execution."""

    return b"".join(encode_e4m3_rowwise_chunks(weight))


def decode_e4m3_rowwise(
    payload: bytes, rows: int, columns: int
) -> torch.Tensor:
    """Decode an E4M3 rowwise payload to its logical CPU FP32 matrix."""

    padded_columns, expected = _geometry(rows, columns)
    if len(payload) != expected:
        raise ValueError(
            f"E4M3 rowwise payload has {len(payload)} bytes, expected {expected}"
        )
    code_bytes = rows * padded_columns
    code_array = np.frombuffer(payload, dtype=np.uint8, count=code_bytes).copy()
    if np.any((code_array & 0x7F) == 0x7F):
        raise ValueError("E4M3 rowwise payload contains an E4M3FN NaN word")
    scale_offset = (code_bytes + PLANE_ALIGNMENT - 1) // PLANE_ALIGNMENT * PLANE_ALIGNMENT
    if any(payload[code_bytes:scale_offset]):
        raise ValueError("E4M3 rowwise scale-plane alignment padding is nonzero")
    scales_array = np.frombuffer(payload, dtype="<f4", count=rows, offset=scale_offset).copy()
    if np.any(
        ~np.isfinite(scales_array)
        | (scales_array < 0)
        | ((scales_array == 0) & np.signbit(scales_array))
    ):
        raise ValueError("E4M3 rowwise scale is not finite and nonnegative")
    codes = torch.from_numpy(code_array).reshape(rows, padded_columns)
    if columns != padded_columns and bool((codes[:, columns:] != 0).any()):
        raise ValueError("E4M3 rowwise K128 padding is nonzero")
    scales = torch.from_numpy(scales_array)
    zero_rows = scales == 0
    if bool(zero_rows.any()) and bool((codes[zero_rows] != 0).any()):
        raise ValueError("E4M3 rowwise zero-scale row has nonzero codes")
    decoded = codes.view(torch.float8_e4m3fn).to(torch.float32)
    decoded.mul_(scales.unsqueeze(1))
    return decoded[:, :columns].contiguous()


def error_metrics(source: torch.Tensor, decoded: torch.Tensor) -> dict[str, float]:
    """Return source-represented absolute, RMSE, and relative-L2 errors."""

    if source.shape != decoded.shape:
        raise ValueError("E4M3 rowwise error operands have different shapes")
    reference = source.detach().to(device="cpu", dtype=torch.float64)
    candidate = decoded.detach().to(device="cpu", dtype=torch.float64)
    difference = candidate - reference
    squared_error = float(torch.sum(difference * difference))
    squared_reference = float(torch.sum(reference * reference))
    return {
        "max_abs": float(torch.max(torch.abs(difference))),
        "rmse": math.sqrt(squared_error / reference.numel()),
        "relative_l2": math.sqrt(squared_error / squared_reference) if squared_reference else 0.0,
    }


__all__ = [
    "decode_e4m3_rowwise",
    "encode_e4m3_rowwise",
    "encode_e4m3_rowwise_chunks",
    "error_metrics",
]
