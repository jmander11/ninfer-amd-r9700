"""Dependency-free exact codec authority for the cache-only BF16 diagnostic.

This module describes storage semantics only.  The model-sized scorer uses a
separate vectorized PyTorch implementation and is checked against this code by
CPU tests; neither implementation imports the product cache or its kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Sequence


E4M3FN_MAX = 448.0
VALUE_CODE_MAX = 7
VALUE_GROUPS = (16, 32)


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def decode_e4m3fn(word: int) -> float:
    if not 0 <= word <= 0xFF:
        raise ValueError("E4M3FN word must be one byte")
    negative = bool(word & 0x80)
    exponent = (word >> 3) & 0x0F
    fraction = word & 0x07
    if exponent == 0:
        magnitude = fraction * math.ldexp(1.0, -9)
    else:
        if exponent == 15 and fraction == 7:
            raise ValueError("E4M3FN NaN word")
        magnitude = (1.0 + fraction / 8.0) * math.ldexp(1.0, exponent - 7)
    return -magnitude if negative else magnitude


def encode_e4m3fn(source: float) -> int:
    source = _float32(source)
    if not math.isfinite(source):
        raise ValueError("E4M3FN source must be finite")
    negative = math.copysign(1.0, source) < 0.0
    magnitude = abs(source)
    if magnitude == 0.0:
        return 0x80 if negative else 0x00
    if magnitude >= E4M3FN_MAX:
        return (0x80 if negative else 0x00) | 0x7E

    best = 0
    best_distance = magnitude
    for candidate in range(1, 0x7F):
        distance = abs(magnitude - decode_e4m3fn(candidate))
        if distance < best_distance or (
            distance == best_distance and candidate % 2 == 0 and best % 2 != 0
        ):
            best = candidate
            best_distance = distance
    return (0x80 if negative else 0x00) | best


def _round_shift_rne(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    mask = (1 << shift) - 1
    half = 1 << (shift - 1)
    base = value >> shift
    remainder = value & mask
    return base + int(remainder > half or (remainder == half and base % 2 != 0))


def fp32_to_fp16_rne(source: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", source))[0]
    sign = (bits >> 16) & 0x8000
    absolute = bits & 0x7FFFFFFF
    if absolute >= 0x7F800000:
        mantissa = absolute & 0x007FFFFF
        return sign | 0x7C00 | (0x0200 if mantissa else 0)

    exponent = ((absolute >> 23) & 0xFF) - 127 + 15
    mantissa = absolute & 0x007FFFFF
    if exponent <= 0:
        if exponent < -10:
            return sign
        mantissa |= 0x00800000
        return sign | _round_shift_rne(mantissa, 14 - exponent)
    if exponent >= 31:
        return sign | 0x7C00

    half_mantissa = _round_shift_rne(mantissa, 13)
    if half_mantissa == 0x0400:
        half_mantissa = 0
        exponent += 1
        if exponent >= 31:
            return sign | 0x7C00
    return sign | (exponent << 10) | half_mantissa


def fp16_to_fp32(word: int) -> float:
    if not 0 <= word <= 0xFFFF:
        raise ValueError("FP16 word must be two bytes")
    sign = (word & 0x8000) << 16
    exponent = (word >> 10) & 0x1F
    mantissa = word & 0x03FF
    if exponent == 0:
        if mantissa == 0:
            return struct.unpack("<f", struct.pack("<I", sign))[0]
        unbiased = -14
        while not mantissa & 0x0400:
            mantissa <<= 1
            unbiased -= 1
        mantissa &= 0x03FF
        bits = sign | ((unbiased + 127) << 23) | (mantissa << 13)
    elif exponent == 31:
        bits = sign | 0x7F800000 | (mantissa << 13)
    else:
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13)
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def round_nearest_even(value: float) -> int:
    lower = math.floor(value)
    fraction = value - lower
    if fraction < 0.5:
        return lower
    if fraction > 0.5:
        return lower + 1
    return lower if lower % 2 == 0 else lower + 1


def pack_signed_int4(codes: Sequence[int]) -> bytes:
    if len(codes) % 2:
        raise ValueError("INT4 code count must be even")
    packed = bytearray(len(codes) // 2)
    for lane, code in enumerate(codes):
        if not -7 <= code <= 7:
            raise ValueError("symmetric INT4 code must be in [-7,7]")
        nibble = code & 0x0F
        if lane % 2 == 0:
            packed[lane // 2] = nibble
        else:
            packed[lane // 2] |= nibble << 4
    return bytes(packed)


def unpack_signed_int4(packed: bytes, lanes: int) -> tuple[int, ...]:
    if lanes < 0 or lanes % 2 or len(packed) != lanes // 2:
        raise ValueError("packed INT4 byte count does not match lanes")
    result = []
    for lane in range(lanes):
        byte = packed[lane // 2]
        nibble = (byte & 0x0F) if lane % 2 == 0 else (byte >> 4)
        code = nibble if nibble < 8 else nibble - 16
        if code == -8:
            raise ValueError("reserved symmetric INT4 code -8")
        result.append(code)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class EncodedValueGroup:
    group_size: int
    scale_bits: int
    codes: tuple[int, ...]
    packed_codes: bytes

    def decoded(self) -> tuple[float, ...]:
        scale = fp16_to_fp32(self.scale_bits)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("INT4 value scale must be finite and nonnegative")
        return tuple(code * scale for code in self.codes)


def encode_value_group(source: Sequence[float], group_size: int) -> EncodedValueGroup:
    if group_size not in VALUE_GROUPS or len(source) != group_size:
        raise ValueError("INT4 values require one complete G16 or G32 group")
    values = tuple(_float32(value) for value in source)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("INT4 value source must be finite")
    maximum = max(abs(value) for value in values)
    if maximum == 0.0:
        codes = (0,) * group_size
        return EncodedValueGroup(group_size, 0, codes, pack_signed_int4(codes))

    scale_bits = fp32_to_fp16_rne(_float32(maximum / 7.0))
    scale = fp16_to_fp32(scale_bits)
    if not math.isfinite(scale):
        raise OverflowError("INT4 value scale is not representable as finite FP16")
    if scale == 0.0:
        codes = (0,) * group_size
        return EncodedValueGroup(group_size, 0, codes, pack_signed_int4(codes))

    codes = tuple(
        max(-7, min(7, round_nearest_even(_float32(value / scale))))
        for value in values
    )
    return EncodedValueGroup(group_size, scale_bits, codes, pack_signed_int4(codes))


__all__ = [
    "E4M3FN_MAX",
    "EncodedValueGroup",
    "VALUE_CODE_MAX",
    "VALUE_GROUPS",
    "decode_e4m3fn",
    "encode_e4m3fn",
    "encode_value_group",
    "fp16_to_fp32",
    "fp32_to_fp16_rne",
    "pack_signed_int4",
    "round_nearest_even",
    "unpack_signed_int4",
]
