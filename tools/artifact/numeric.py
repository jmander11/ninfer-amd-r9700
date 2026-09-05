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


NumericFormat: TypeAlias = DirectFormat | QuantFormat | RowScaledFormat


BF16 = DirectFormat("BF16", 2)
FP32 = DirectFormat("FP32", 4)
I32 = DirectFormat("I32", 4)

Q4G64_F16S = QuantFormat("Q4G64_F16S", 4, 64, -8, 7)
Q5G64_F16S = QuantFormat("Q5G64_F16S", 5, 64, -16, 15)
Q6G64_F16S = QuantFormat("Q6G64_F16S", 6, 64, -32, 31)
W8G32_F16S = QuantFormat("W8G32_F16S", 8, 32, -127, 127)
F8E4M3_ROW_F32S = RowScaledFormat("F8E4M3_ROW_F32S")


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
NUMERIC_FORMATS = MappingProxyType(
    {**DIRECT_FORMATS, **QUANT_FORMATS, **ROW_SCALED_FORMATS}
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


def get_format(name: str) -> NumericFormat:
    """Return the registered format named *name*."""

    try:
        return NUMERIC_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown numeric format: {name!r}") from None


__all__ = [
    "BF16",
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
