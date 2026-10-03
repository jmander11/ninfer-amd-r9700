"""Source quantizer for the FP8LUT4 codebook weight format.

Each row gets one FP32 multiplier R = max|w| / (7.5 * 16), so the largest group's natural
codebook exponent is 4. Every 32-wide group then chooses the group code (exponent E and mantissa
step m) among E in {e0 - 1, e0, e0 + 1} (e0 = floor(log2(max|group| / (7.5 R)))) and all eight m,
minimizing the decoded squared error; each element takes the nearest codebook magnitude and its
own sign (zero magnitudes keep a clear sign bit). Ties keep the earliest candidate in (E, m) order
and the smaller magnitude. With a `Calibration` (the projection's input second moments), the same
row multiplier and group-code search are applied column-sequentially with GPTQ error compensation
instead of independent rounding. The selected words are exactly those
`tools.artifact.decode_fp8lut4` reconstructs.
"""

from __future__ import annotations

from typing import Iterator

import torch

from tools.artifact.layouts import (
    fp8lut4_geometry,
    fp8lut4_magnitude_table,
    fp8lut4_tile_codes,
    fp8lut4_tile_groups,
)

GROUP = 32
_E_MIN, _E_MAX = -26, 5


def _magnitudes(device: torch.device) -> torch.Tensor:
    words = torch.tensor(fp8lut4_magnitude_table(), dtype=torch.uint8)
    return words.view(torch.float8_e4m3fn).to(torch.float32).to(device)  # [256, 8]


def _search_groups(
    grouped: torch.Tensor, table: torch.Tensor, importance: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """grouped FP32 [r, G, 32] in row-multiplier units -> (magnitude indices u8 [r, G, 32], group
    codes u8 [r, G]) minimizing the (importance-weighted) decoded squared error."""

    magnitude = grouped.abs()
    amax = magnitude.amax(-1)
    e0 = torch.floor(torch.log2((amax / 7.5).clamp(min=2.0 ** (_E_MIN - 1))))
    best_error = torch.full_like(amax, float("inf"))
    best_index = torch.zeros(grouped.shape, dtype=torch.uint8, device=grouped.device)
    best_code = torch.zeros(amax.shape, dtype=torch.uint8, device=grouped.device)
    for delta in (-1.0, 0.0, 1.0):
        exponent = (e0 + delta).clamp(_E_MIN, _E_MAX)
        for step in range(8):
            code = ((exponent - _E_MIN) * 8 + step).to(torch.long)  # [r, G]
            book = table[code]  # [r, G, 8]
            middle = 0.5 * (book[..., 1:] + book[..., :-1])  # [r, G, 7]
            index = (magnitude[..., None] > middle[..., None, :]).sum(-1)  # [r, G, 32]
            squared = (magnitude - torch.gather(book, -1, index)) ** 2
            error = (squared if importance is None else squared * importance).sum(-1)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_index = torch.where(better[..., None], index.to(torch.uint8), best_index)
            best_code = torch.where(better, code.to(torch.uint8), best_code)
    return best_index, best_code


def _signed(index: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return index | (((values < 0) & (index > 0)).to(torch.uint8) << 3)


class Calibration:
    """Second-moment matrix H = sum_t x_t x_t^T of one projection's represented inputs, prepared
    once for every row chunk of that projection: `damping` x mean-diagonal damping, never-active inputs
    pinned to zero weight, and the upper Cholesky factor of H^-1."""

    def __init__(self, hessian: torch.Tensor, damping: float):
        if hessian.dim() != 2 or hessian.shape[0] != hessian.shape[1] or hessian.shape[0] % GROUP:
            raise ValueError("FP8LUT4 calibration requires a square [K, K] matrix with K % 32 == 0")
        h = hessian.to(torch.float32).clone()
        diagonal = torch.diagonal(h)
        self.dead = diagonal <= 0
        h[self.dead, :] = 0.0
        h[:, self.dead] = 0.0
        diagonal[self.dead] = 1.0
        diagonal += damping * diagonal[~self.dead].mean() if bool((~self.dead).any()) else 0.0
        lower_inverse = torch.linalg.solve_triangular(
            torch.linalg.cholesky(h),
            torch.eye(h.shape[0], dtype=h.dtype, device=h.device),
            upper=False,
        )
        inverse = lower_inverse.t() @ lower_inverse
        self.factor = torch.linalg.cholesky(inverse, upper=True)


def _gptq(
    scaled: torch.Tensor, calibration: Calibration, table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Column-sequential error-compensated rounding onto the FP8LUT4 grid (GPTQ): each 32-wide
    group code is searched on the error-updated weights at the group's first column, weighted by
    1 / U_jj^2, and every column's rounding error is propagated through U to the later columns."""

    rows, columns = scaled.shape
    factor = calibration.factor.to(scaled.device)
    weight = scaled.clone()
    weight[:, calibration.dead.to(scaled.device)] = 0.0
    codes = torch.empty(rows, columns, dtype=torch.uint8, device=scaled.device)
    groups = torch.empty(rows, columns // GROUP, dtype=torch.uint8, device=scaled.device)
    importance = torch.diagonal(factor).reciprocal().square()
    block = 128
    for begin in range(0, columns, block):
        end = min(columns, begin + block)
        current = weight[:, begin:end].clone()
        errors = torch.empty_like(current)
        local = factor[begin:end, begin:end]
        for j in range(end - begin):
            column = begin + j
            if column % GROUP == 0:
                _, code = _search_groups(
                    current[:, None, j : j + GROUP], table, importance[column : column + GROUP]
                )
                groups[:, column // GROUP] = code[:, 0]
                book = table[code[:, 0].long()]  # [r, 8]
                middle = 0.5 * (book[:, 1:] + book[:, :-1])  # [r, 7]
            value = current[:, j]
            index = (value.abs()[:, None] > middle).sum(-1)
            quantized = torch.gather(book, 1, index[:, None])[:, 0].copysign(value)
            codes[:, column] = _signed(index.to(torch.uint8), value)
            error = (value - quantized) / local[j, j]
            current[:, j:] -= error[:, None] * local[j, j:][None, :]
            errors[:, j] = error
        weight[:, end:] -= errors @ factor[begin:end, end:]
    return codes, groups


def quantize_rows(
    weight: torch.Tensor, calibration: Calibration | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """weight FP32 [r, K_pad] (K_pad % 32 == 0) -> (codes u8 [r, K_pad], groups u8 [r, K_pad/32],
    row multipliers FP32 [r]). Without calibration every group is rounded independently; with
    one, the rounding is error-compensated against that projection's input statistics."""

    if weight.dtype != torch.float32 or weight.dim() != 2 or weight.shape[1] % GROUP:
        raise ValueError("FP8LUT4 quantization requires FP32 [rows, K] with K % 32 == 0")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("FP8LUT4 quantization source contains NaN or infinity")
    rows, columns = weight.shape
    table = _magnitudes(weight.device)
    rowmax = weight.abs().amax(1)
    scale = rowmax / (7.5 * 16.0)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    scaled = weight / safe[:, None]
    if calibration is None:
        index, best_code = _search_groups(scaled.reshape(rows, columns // GROUP, GROUP), table)
        codes = _signed(index, scaled.reshape(index.shape)).reshape(rows, columns)
    else:
        if calibration.factor.shape[0] != columns:
            raise ValueError("FP8LUT4 calibration width differs from the weight")
        codes, best_code = _gptq(scaled, calibration, table)
    zero = scale == 0
    codes[zero] = 0
    best_code[zero] = 0
    return codes, best_code, scale


def interleave_gate_up(weight: torch.Tensor) -> torch.Tensor:
    """[gate; up] rows [2F, K] -> the FP8LUT4 MLP storage order: stored row 16 b + i is gate feature
    8 b + i (i < 8) or up feature 8 b + i - 8, so one 16-row tile holds eight gate features and the
    matching eight up features (the SiLU-pair epilogue's operand pairs)."""

    rows, columns = weight.shape
    if rows % 16:
        raise ValueError("gate/up interleave requires 16-row tiles")
    return weight.reshape(2, rows // 16, 8, columns).permute(1, 0, 2, 3).reshape(rows, columns)


def encode_chunks(
    weight: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    rows_per_chunk: int = 2048,
    calibration: Calibration | None = None,
) -> Iterator[bytes]:
    """Yield one r9700-fp8lut4-n16k64-v1 payload for represented BF16 [N, K] in plane order."""

    if weight.dim() != 2 or weight.dtype != torch.bfloat16:
        raise TypeError("FP8LUT4 encoding requires a represented BF16 rank-2 matrix")
    rows, columns = weight.shape
    geometry = fp8lut4_geometry((rows, columns))
    if rows_per_chunk % 16:
        raise ValueError("FP8LUT4 encoding chunks must hold whole 16-row tiles")
    codes_out, groups_out, scales_out = [], [], []
    for begin in range(0, rows, rows_per_chunk):
        block = weight[begin : begin + rows_per_chunk].to(device=device, dtype=torch.float32)
        if geometry.k_pad != columns:
            block = torch.nn.functional.pad(block, (0, geometry.k_pad - columns))
        codes, groups, scales = quantize_rows(block, calibration)
        codes = codes.cpu()
        codes_out.append(
            fp8lut4_tile_codes(codes[:, 0::2] | (codes[:, 1::2] << 4)).numpy().tobytes()
        )
        groups_out.append(fp8lut4_tile_groups(groups.cpu()).numpy().tobytes())
        scales_out.append(scales.cpu().numpy().astype("<f4").tobytes())
    yield from codes_out
    yield bytes(geometry.group_offset - geometry.code_bytes)
    yield from groups_out
    yield bytes(geometry.scale_offset - geometry.group_offset - geometry.group_bytes)
    yield from scales_out


__all__ = ["Calibration", "encode_chunks", "interleave_gate_up", "quantize_rows"]
