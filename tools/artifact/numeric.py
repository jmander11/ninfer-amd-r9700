"""Closed registry of persistent NInfer tensor numeric formats."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class DirectFormat:
    """One fixed-width word per logical tensor element."""

    name: str
    word_bytes: int


@dataclass(frozen=True, slots=True)
class QuantFormat:
    """Signed grouped codes with one binary16 multiplier per group."""

    name: str
    bits: int
    group_size: int
    qmin: int
    qmax: int


@dataclass(frozen=True, slots=True)
class RowScaledFormat:
    """Eight-bit row-scaled codes with one FP32 multiplier per matrix row."""

    name: str


@dataclass(frozen=True, slots=True)
class CodebookFormat:
    """Four-bit sign-magnitude codes read through a per-group E4M3 magnitude codebook, with one
    FP32 multiplier per matrix row."""

    name: str
    group_size: int


NumericFormat: TypeAlias = DirectFormat | QuantFormat | RowScaledFormat | CodebookFormat


BF16 = DirectFormat("BF16", 2)
FP32 = DirectFormat("FP32", 4)
I32 = DirectFormat("I32", 4)

Q4G64_F16S = QuantFormat("Q4G64_F16S", 4, 64, -8, 7)
Q5G64_F16S = QuantFormat("Q5G64_F16S", 5, 64, -16, 15)
Q6G64_F16S = QuantFormat("Q6G64_F16S", 6, 64, -32, 31)
W8G32_F16S = QuantFormat("W8G32_F16S", 8, 32, -127, 127)
F8E4M3_ROW_F32S = RowScaledFormat("F8E4M3_ROW_F32S")
CB4G32_F32S = CodebookFormat("CB4G32_F32S", 32)


DIRECT_FORMATS = MappingProxyType(
    {item.name: item for item in (BF16, FP32, I32)}
)
QUANT_FORMATS = MappingProxyType(
    {
        item.name: item
        for item in (Q4G64_F16S, Q5G64_F16S, Q6G64_F16S, W8G32_F16S)
    }
)
ROW_SCALED_FORMATS = MappingProxyType(
    {item.name: item for item in (F8E4M3_ROW_F32S,)}
)
CODEBOOK_FORMATS = MappingProxyType({item.name: item for item in (CB4G32_F32S,)})
NUMERIC_FORMATS = MappingProxyType(
    {**DIRECT_FORMATS, **QUANT_FORMATS, **ROW_SCALED_FORMATS, **CODEBOOK_FORMATS}
)


def decode_e4m3fn_word(word: int) -> float:
    """Decode one exact eight-bit E4M3FN word."""

    if type(word) is not int or not 0 <= word <= 0xFF:
        raise ValueError("E4M3FN word must be an integer in [0, 255]")
    sign = -1.0 if word & 0x80 else 1.0
    exponent = (word >> 3) & 0xF
    fraction = word & 0x7
    if exponent == 0:
        if fraction == 0:
            return math.copysign(0.0, sign)
        return sign * fraction * (2.0**-9)
    if exponent == 0xF and fraction == 0x7:
        return math.copysign(math.nan, sign)
    return sign * (1.0 + fraction / 8.0) * (2.0 ** (exponent - 7))


def round_e4m3fn_magnitude(numerator: int, exponent: int) -> int:
    """Return the E4M3FN word (sign clear) nearest numerator * 2**exponent, ties to even,
    saturating at 448. *numerator* is a nonnegative integer."""

    if type(numerator) is not int or numerator < 0 or type(exponent) is not int:
        raise ValueError("E4M3FN rounding requires a nonnegative integer numerator")
    if numerator == 0:
        return 0
    # Exact value v = numerator * 2**exponent. Unit in the last place is 2**(e - 3) for a
    # normal binade [2**e, 2**(e + 1)) with e >= -6, and 2**-9 for subnormals.
    e = numerator.bit_length() - 1 + exponent
    ulp_exponent = max(e, -6) - 3
    shift = ulp_exponent - exponent
    if shift <= 0:
        units = numerator << -shift
    else:
        units, remainder = divmod(numerator, 1 << shift)
        half = 1 << (shift - 1)
        if remainder > half or (remainder == half and units & 1):
            units += 1
    # units * 2**ulp_exponent; renormalize a carry into the next binade.
    if ulp_exponent == -9 and units < 8:
        return units  # subnormal (or rounded up to the smallest normal 8 -> handled below)
    while units >= 16:
        units >>= 1
        ulp_exponent += 1
    biased = ulp_exponent + 3 + 7
    if biased > 15 or (biased == 15 and units - 8 > 6):
        return 0x7E  # 448, saturated
    return (biased << 3) | (units - 8)


CB4_BASE_SIXTEENTHS = (0, 13, 27, 41, 56, 74, 94, 120)


def cb4_group_magnitudes(group_code: int) -> tuple[int, ...]:
    """The eight E4M3FN magnitude words of one CB4G32 group code.

    Code byte b holds m = b & 7 and E = (b >> 3) - 26; magnitude j (0..7) is the E4M3FN rounding
    of (n_j / 16) * (1 + m / 8) * 2**E = n_j * (8 + m) * 2**(E - 7) with the fixed base
    n = CB4_BASE_SIXTEENTHS (a zero level plus a Lloyd fit of group-normalized weights).
    """

    if type(group_code) is not int or not 0 <= group_code <= 0xFF:
        raise ValueError("CB4 group code must be an integer in [0, 255]")
    m, e = group_code & 7, (group_code >> 3) - 26
    return tuple(round_e4m3fn_magnitude(n * (8 + m), e - 7) for n in CB4_BASE_SIXTEENTHS)


def get_format(name: str) -> NumericFormat:
    """Return the registered format named *name*."""

    try:
        return NUMERIC_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown numeric format: {name!r}") from None


__all__ = [
    "BF16",
    "CB4G32_F32S",
    "CB4_BASE_SIXTEENTHS",
    "CODEBOOK_FORMATS",
    "CodebookFormat",
    "cb4_group_magnitudes",
    "round_e4m3fn_magnitude",
    "DIRECT_FORMATS",
    "DirectFormat",
    "FP32",
    "F8E4M3_ROW_F32S",
    "I32",
    "NUMERIC_FORMATS",
    "NumericFormat",
    "Q4G64_F16S",
    "Q5G64_F16S",
    "Q6G64_F16S",
    "QUANT_FORMATS",
    "QuantFormat",
    "ROW_SCALED_FORMATS",
    "RowScaledFormat",
    "W8G32_F16S",
    "decode_e4m3fn_word",
    "get_format",
]
