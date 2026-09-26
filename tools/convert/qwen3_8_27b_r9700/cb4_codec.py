"""Source quantizer for the CB4G32 codebook weight format.

Each row gets one FP32 multiplier R = max|w| / (7.5 * 16), so the largest group's natural
codebook exponent is 4. Every 32-wide group then chooses the group code (exponent E and mantissa
step m) among E in {e0 - 1, e0, e0 + 1} (e0 = floor(log2(max|group| / (7.5 R)))) and all eight m,
minimizing the decoded squared error; each element takes the nearest codebook magnitude and its
own sign (zero magnitudes keep a clear sign bit). Ties keep the earliest candidate in (E, m) order and the smaller magnitude. The selected
words are exactly those `tools.artifact.decode_cb4` reconstructs.
"""

from __future__ import annotations

from typing import Iterator

import torch

from tools.artifact.layouts import cb4_geometry, cb4_magnitude_table, cb4_tile_codes, cb4_tile_groups

GROUP = 32
_E_MIN, _E_MAX = -26, 5


def _magnitudes(device: torch.device) -> torch.Tensor:
    words = torch.tensor(cb4_magnitude_table(), dtype=torch.uint8)
    return words.view(torch.float8_e4m3fn).to(torch.float32).to(device)  # [256, 8]


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """weight FP32 [r, K_pad] (K_pad % 32 == 0) -> (codes u8 [r, K_pad], groups u8 [r, K_pad/32],
    row multipliers FP32 [r])."""

    if weight.dtype != torch.float32 or weight.dim() != 2 or weight.shape[1] % GROUP:
        raise ValueError("CB4 quantization requires FP32 [rows, K] with K % 32 == 0")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("CB4 quantization source contains NaN or infinity")
    rows, columns = weight.shape
    table = _magnitudes(weight.device)
    rowmax = weight.abs().amax(1)
    scale = rowmax / (7.5 * 16.0)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    grouped = (weight / safe[:, None]).reshape(rows, columns // GROUP, GROUP)
    magnitude = grouped.abs()
    amax = magnitude.amax(-1)
    e0 = torch.floor(torch.log2((amax / 7.5).clamp(min=2.0 ** (_E_MIN - 1))))
    best_error = torch.full_like(amax, float("inf"))
    best_index = torch.zeros(grouped.shape, dtype=torch.uint8, device=weight.device)
    best_code = torch.zeros(amax.shape, dtype=torch.uint8, device=weight.device)
    for delta in (-1.0, 0.0, 1.0):
        exponent = (e0 + delta).clamp(_E_MIN, _E_MAX)
        for step in range(8):
            code = ((exponent - _E_MIN) * 8 + step).to(torch.long)              # [r, G]
            book = table[code]                                                  # [r, G, 8]
            middle = 0.5 * (book[..., 1:] + book[..., :-1])                     # [r, G, 7]
            index = (magnitude[..., None] > middle[..., None, :]).sum(-1)       # [r, G, 32]
            decoded = torch.gather(book, -1, index)
            error = ((magnitude - decoded) ** 2).sum(-1)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_index = torch.where(better[..., None], index.to(torch.uint8), best_index)
            best_code = torch.where(better, code.to(torch.uint8), best_code)
    sign = ((grouped < 0) & (best_index > 0)).to(torch.uint8) << 3
    codes = (best_index | sign).reshape(rows, columns)
    zero = scale == 0
    codes[zero] = 0
    best_code[zero] = 0
    return codes, best_code, scale


def interleave_gate_up(weight: torch.Tensor) -> torch.Tensor:
    """[gate; up] rows [2F, K] -> the CB4 MLP storage order: stored row 16 b + i is gate feature
    8 b + i (i < 8) or up feature 8 b + i - 8, so one 16-row tile holds eight gate features and the
    matching eight up features (the SiLU-pair epilogue's operand pairs)."""

    rows, columns = weight.shape
    if rows % 16:
        raise ValueError("gate/up interleave requires 16-row tiles")
    return weight.reshape(2, rows // 16, 8, columns).permute(1, 0, 2, 3).reshape(rows, columns)


def encode_chunks(weight: torch.Tensor, *, device: str | torch.device = "cpu",
                  rows_per_chunk: int = 2048) -> Iterator[bytes]:
    """Yield one r9700-cb4g32-n16k64-v1 payload for represented BF16 [N, K] in plane order."""

    if weight.dim() != 2 or weight.dtype != torch.bfloat16:
        raise TypeError("CB4 encoding requires a represented BF16 rank-2 matrix")
    rows, columns = weight.shape
    geometry = cb4_geometry((rows, columns))
    if rows_per_chunk % 16:
        raise ValueError("CB4 encoding chunks must hold whole 16-row tiles")
    codes_out, groups_out, scales_out = [], [], []
    for begin in range(0, rows, rows_per_chunk):
        block = weight[begin:begin + rows_per_chunk].to(device=device, dtype=torch.float32)
        if geometry.k_pad != columns:
            block = torch.nn.functional.pad(block, (0, geometry.k_pad - columns))
        codes, groups, scales = quantize_rows(block)
        codes = codes.cpu()
        codes_out.append(cb4_tile_codes(codes[:, 0::2] | (codes[:, 1::2] << 4)).numpy().tobytes())
        groups_out.append(cb4_tile_groups(groups.cpu()).numpy().tobytes())
        scales_out.append(scales.cpu().numpy().astype("<f4").tobytes())
    yield from codes_out
    yield bytes(geometry.group_offset - geometry.code_bytes)
    yield from groups_out
    yield bytes(geometry.scale_offset - geometry.group_offset - geometry.group_bytes)
    yield from scales_out


__all__ = ["encode_chunks", "interleave_gate_up", "quantize_rows"]
