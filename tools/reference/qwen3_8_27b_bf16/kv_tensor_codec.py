"""Vectorized cache planes for the checkpoint-direct cache-only diagnostic."""

from __future__ import annotations

import torch


E4M3FN_MAX = 448.0
VALUE_CODE_MAX = 7
VALUE_GROUPS = (16, 32)


def quantize_keys(keys: torch.Tensor) -> torch.Tensor:
    """Store finite represented BF16 K as saturating OCP E4M3FN codes."""
    if not bool(torch.isfinite(keys).all()):
        raise ValueError("KV keys must be finite")
    return keys.float().clamp(-E4M3FN_MAX, E4M3FN_MAX).to(torch.float8_e4m3fn)


def dequantize_keys(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.float8_e4m3fn:
        raise ValueError("KV key plane must contain E4M3FN codes")
    return codes.float()


def quantize_values(
    values: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Store finite represented BF16 V as low-lane-first INT4 plus FP16 scales."""
    if group_size not in VALUE_GROUPS or values.shape[-1] % group_size:
        raise ValueError("KV values require a final dimension divisible by G16 or G32")
    if values.shape[-1] % 2:
        raise ValueError("KV value dimension must be even")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("KV values must be finite")

    groups = values.float().reshape(
        *values.shape[:-1], values.shape[-1] // group_size, group_size
    )
    scales = (groups.abs().amax(dim=-1) / float(VALUE_CODE_MAX)).to(torch.float16)
    if not bool(torch.isfinite(scales).all()):
        raise OverflowError("KV value scale is not representable as finite FP16")
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales).float()
    codes = torch.round(groups / safe_scales.unsqueeze(-1)).clamp(
        -VALUE_CODE_MAX, VALUE_CODE_MAX
    ).to(torch.int8)
    codes = torch.where((scales == 0).unsqueeze(-1), 0, codes).to(torch.int8)
    low = codes[..., 0::2].to(torch.int16) & 0x0F
    high = codes[..., 1::2].to(torch.int16) & 0x0F
    packed = (low | (high << 4)).to(torch.uint8)
    return packed.reshape(*values.shape[:-1], values.shape[-1] // 2), scales


def dequantize_values(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if group_size not in VALUE_GROUPS:
        raise ValueError("KV value group must be G16 or G32")
    if packed.dtype != torch.uint8 or scales.dtype != torch.float16:
        raise ValueError("KV value planes must be packed UINT8 codes and FP16 scales")
    dimension = packed.shape[-1] * 2
    if dimension % group_size or scales.shape != (
        *packed.shape[:-1],
        dimension // group_size,
    ):
        raise ValueError("KV value code and scale planes have incompatible shapes")
    if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
        raise ValueError("KV value scales must be finite and nonnegative")

    packed_i16 = packed.to(torch.int16)
    low = packed_i16 & 0x0F
    high = (packed_i16 >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], dimension)
    codes = torch.where(codes >= 8, codes - 16, codes)
    if bool((codes == -8).any()):
        raise ValueError("reserved symmetric INT4 code -8")
    groups = codes.float().reshape(
        *codes.shape[:-1], dimension // group_size, group_size
    )
    return (groups * scales.float().unsqueeze(-1)).reshape(*codes.shape)


def validate_codec_device(device: torch.device, group_size: int) -> None:
    """Fail before scoring if this PyTorch device violates the fixed storage profile."""
    if group_size not in VALUE_GROUPS:
        raise ValueError("KV value group must be G16 or G32")
    key_source = torch.tensor(
        (0.0, -0.0, 1.0, -1.0, 448.0, -448.0, 1.0625, 1.1875, 1000.0),
        device=device,
        dtype=torch.float32,
    )
    expected_keys = (0x00, 0x80, 0x38, 0xB8, 0x7E, 0xFE, 0x38, 0x3A, 0x7E)
    actual_keys = tuple(quantize_keys(key_source).view(torch.uint8).cpu().tolist())
    if actual_keys != expected_keys:
        raise RuntimeError("PyTorch device does not implement the required E4M3FN codec")

    values = (
        0.5,
        1.5,
        2.5,
        3.5,
        -0.5,
        -1.5,
        -2.5,
        -3.5,
        7.0,
        -7.0,
        0.0,
        1.0,
        -1.0,
        6.0,
        -6.0,
        0.0,
    ) + (0.0,) * (group_size - 16)
    value_source = torch.tensor(
        values, device=device, dtype=torch.bfloat16
    ).reshape(1, 1, group_size)
    packed, scales = quantize_values(value_source, group_size)
    expected_packed = bytes((0x20, 0x42, 0xE0, 0xCE, 0x97, 0x10, 0x6F, 0x0A)) + bytes(
        (group_size - 16) // 2
    )
    actual_packed = bytes(packed.flatten().cpu().tolist())
    actual_scale = int(scales.view(torch.int16).cpu().item()) & 0xFFFF
    if actual_packed != expected_packed or actual_scale != 0x3C00:
        raise RuntimeError("PyTorch device does not implement the required INT4/FP16 codec")

    tiny = torch.full(
        (1, 1, group_size),
        2.0**-24,
        device=device,
        dtype=torch.bfloat16,
    )
    tiny_packed, tiny_scale = quantize_values(tiny, group_size)
    if (
        int(tiny_scale.view(torch.int16).cpu().item()) & 0xFFFF
    ) != 0 or any(tiny_packed.flatten().cpu().tolist()):
        raise RuntimeError("PyTorch device does not canonicalize FP16 scale underflow")


class QuantizedFullAttentionCache:
    """Append-owned cache planes with decode only at the attention-use boundary."""

    def __init__(
        self,
        capacity: int,
        heads: int,
        head_dim: int,
        group_size: int,
        device: torch.device,
    ) -> None:
        if capacity <= 0 or heads <= 0 or head_dim <= 0:
            raise ValueError("KV cache dimensions must be positive")
        if group_size not in VALUE_GROUPS or head_dim % group_size or head_dim % 2:
            raise ValueError("KV head dimension is incompatible with G16/G32 INT4 values")
        self.capacity = capacity
        self.heads = heads
        self.head_dim = head_dim
        self.group_size = group_size
        self.length = 0
        self.key_codes = torch.empty(
            (capacity, heads, head_dim), device=device, dtype=torch.float8_e4m3fn
        )
        self.value_codes = torch.empty(
            (capacity, heads, head_dim // 2), device=device, dtype=torch.uint8
        )
        self.value_scales = torch.empty(
            (capacity, heads, head_dim // group_size), device=device, dtype=torch.float16
        )

    def append(self, begin: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        expected_tail = (self.heads, self.head_dim)
        if begin != self.length or keys.shape != values.shape:
            raise ValueError("KV cache append must be contiguous with matching K/V shapes")
        if keys.ndim != 3 or tuple(keys.shape[1:]) != expected_tail:
            raise ValueError("KV cache append has the wrong head geometry")
        end = begin + keys.shape[0]
        if end > self.capacity:
            raise ValueError("KV cache append exceeds capacity")
        key_codes = quantize_keys(keys)
        value_codes, value_scales = quantize_values(values, self.group_size)
        self.key_codes[begin:end].copy_(key_codes)
        self.value_codes[begin:end].copy_(value_codes)
        self.value_scales[begin:end].copy_(value_scales)
        self.length = end

    def read(self, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= end <= self.length:
            raise ValueError("KV cache read exceeds the represented prefix")
        return (
            dequantize_keys(self.key_codes[:end]),
            dequantize_values(
                self.value_codes[:end], self.value_scales[:end], self.group_size
            ),
        )


__all__ = [
    "QuantizedFullAttentionCache",
    "dequantize_keys",
    "dequantize_values",
    "quantize_keys",
    "quantize_values",
    "validate_codec_device",
]
