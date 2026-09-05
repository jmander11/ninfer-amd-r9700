"""Independent scalar oracles for the R9700 grouped integer candidates.

This module intentionally depends only on the Python standard library. It
defines the finite represented-value, FP16-scale, ties-to-even, K=128-padded,
row-split byte contract used by the vectorized converter path. Tests can
therefore exercise codec semantics without Torch, a GPU, or source weights.
"""

from __future__ import annotations

import math
import struct
from typing import Sequence


GROUP_SIZE = 32
K_ALIGNMENT = 128
PLANE_ALIGNMENT = 256
QMIN = -127
QMAX = 127
FP16_MIN_SUBNORMAL_BITS = 0x0001
MSE_REFINEMENT_STEPS = 8
Q4_GROUP_SIZE = 64
Q4_QMIN = -8
Q4_QMAX = 7
E4M3FN_MAX = 448.0


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _fp16_bits(value: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", value))[0]
    except OverflowError as error:
        raise ValueError("W8G32 scale overflows FP16") from error


def _fp16_value(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def _round_even(value: float) -> int:
    # Python's integral float rounding is IEEE ties-to-even. The finite check
    # keeps bad source state from becoming a platform-dependent integer cast.
    if not math.isfinite(value):
        raise ValueError("W8G32 code input is non-finite")
    return int(round(value))


def _validate_shape(rows: int, columns: int, values: Sequence[float]) -> None:
    if type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0:
        raise ValueError("W8G32 shape requires positive integer rows and columns")
    if len(values) != rows * columns:
        raise ValueError("W8G32 source value count does not match shape")


def encode_w8g32_reference(values: Sequence[float], rows: int, columns: int) -> bytes:
    """Encode row-major represented values into canonical W8G32 bytes.

    Input values must already be the public BF16-represented values. The
    converter's Torch route supplies exactly those values; this scalar oracle
    independently specifies the quantization and packing boundary.
    """

    _validate_shape(rows, columns, values)
    padded_columns = _align_up(columns, K_ALIGNMENT)
    groups_per_row = padded_columns // GROUP_SIZE
    base_bytes = rows * padded_columns
    scale_offset = _align_up(base_bytes, PLANE_ALIGNMENT)
    payload = bytearray(scale_offset + rows * groups_per_row * 2)

    for row in range(rows):
        source = [float(item) for item in values[row * columns : (row + 1) * columns]]
        if not all(math.isfinite(item) for item in source):
            raise ValueError("W8G32 source contains NaN or infinity")
        padded = source + [0.0] * (padded_columns - columns)
        for group in range(groups_per_row):
            begin = group * GROUP_SIZE
            logical = padded[begin : begin + GROUP_SIZE]
            maximum = max(abs(item) for item in logical)
            scale_bits = 0
            reciprocal = 0.0
            if maximum != 0.0:
                raw_scale = _f32(maximum / float(QMAX))
                scale_bits = _fp16_bits(raw_scale)
                if scale_bits == 0:
                    scale_bits = FP16_MIN_SUBNORMAL_BITS
                scale = _fp16_value(scale_bits)
                if not math.isfinite(scale) or scale <= 0.0:
                    raise ValueError("W8G32 scale is not finite and positive")
                reciprocal = _f32(1.0 / scale)

            base = row * padded_columns + begin
            for offset, item in enumerate(logical):
                code = 0 if scale_bits == 0 else _round_even(_f32(item * reciprocal))
                payload[base + offset] = max(QMIN, min(QMAX, code)) & 0xFF
            scale_index = (row * groups_per_row + group) * 2
            struct.pack_into("<H", payload, scale_offset + scale_index, scale_bits)
    return bytes(payload)


def _scale_bits_from_raw(raw_scale: float, nonzero: bool) -> int:
    scale_bits = _fp16_bits(_f32(raw_scale))
    if scale_bits == 0 and nonzero:
        scale_bits = FP16_MIN_SUBNORMAL_BITS
    scale = _fp16_value(scale_bits)
    if nonzero and (not math.isfinite(scale) or scale <= 0.0):
        raise ValueError("W8G32 scale is not finite and positive")
    return scale_bits


def _group_codes(values: Sequence[float], scale_bits: int) -> list[int]:
    if scale_bits == 0:
        return [0] * len(values)
    reciprocal = _f32(1.0 / _fp16_value(scale_bits))
    return [
        max(QMIN, min(QMAX, _round_even(_f32(float(item) * reciprocal))))
        for item in values
    ]


def _group_sse(values: Sequence[float], codes: Sequence[int], scale_bits: int) -> float:
    scale = _fp16_value(scale_bits)
    return math.fsum(
        (float(item) - float(code) * scale) ** 2
        for item, code in zip(values, codes, strict=True)
    )


def _mse_scale_bits(values: Sequence[float]) -> int:
    maximum = max(abs(float(item)) for item in values)
    best_bits = _scale_bits_from_raw(maximum / float(QMAX), maximum != 0.0)
    current_bits = best_bits
    best_codes = _group_codes(values, best_bits)
    best_sse = _group_sse(values, best_codes, best_bits)

    for _ in range(MSE_REFINEMENT_STEPS):
        current_codes = _group_codes(values, current_bits)
        numerator = math.fsum(
            float(item) * float(code)
            for item, code in zip(values, current_codes, strict=True)
        )
        denominator = sum(code * code for code in current_codes)
        raw_scale = numerator / float(denominator) if denominator else 0.0
        current_bits = _scale_bits_from_raw(raw_scale, maximum != 0.0)
        candidate_codes = _group_codes(values, current_bits)
        candidate_sse = _group_sse(values, candidate_codes, current_bits)
        if candidate_sse < best_sse:
            best_bits = current_bits
            best_sse = candidate_sse
    return best_bits


def encode_w8g32_mse_reference(values: Sequence[float], rows: int, columns: int) -> bytes:
    """Encode the source-only MSE-refined W8G32 evaluation recipe.

    This independent scalar implementation fixes the eight-step candidate
    trajectory, exact stored FP16-scale boundary, ties-to-even code selection,
    and earliest-candidate tie break used by the vectorized converter.
    """

    _validate_shape(rows, columns, values)
    padded_columns = _align_up(columns, K_ALIGNMENT)
    groups_per_row = padded_columns // GROUP_SIZE
    base_bytes = rows * padded_columns
    scale_offset = _align_up(base_bytes, PLANE_ALIGNMENT)
    payload = bytearray(scale_offset + rows * groups_per_row * 2)

    for row in range(rows):
        source = [float(item) for item in values[row * columns : (row + 1) * columns]]
        if not all(math.isfinite(item) for item in source):
            raise ValueError("W8G32 source contains NaN or infinity")
        padded = source + [0.0] * (padded_columns - columns)
        for group in range(groups_per_row):
            begin = group * GROUP_SIZE
            logical = padded[begin : begin + GROUP_SIZE]
            scale_bits = _mse_scale_bits(logical)
            codes = _group_codes(logical, scale_bits)
            base = row * padded_columns + begin
            for offset, code in enumerate(codes):
                payload[base + offset] = code & 0xFF
            scale_index = (row * groups_per_row + group) * 2
            struct.pack_into("<H", payload, scale_offset + scale_index, scale_bits)
    return bytes(payload)


def decode_w8g32_reference(payload: bytes, rows: int, columns: int) -> tuple[list[int], list[int]]:
    """Return canonical signed codes and FP16 scale words for test inspection."""

    _validate_shape(rows, columns, [0.0] * (rows * columns))
    padded_columns = _align_up(columns, K_ALIGNMENT)
    groups_per_row = padded_columns // GROUP_SIZE
    scale_offset = _align_up(rows * padded_columns, PLANE_ALIGNMENT)
    expected = scale_offset + rows * groups_per_row * 2
    if len(payload) != expected:
        raise ValueError(f"W8G32 payload has {len(payload)} bytes, expected {expected}")
    codes = [byte - 256 if byte >= 128 else byte for byte in payload[: rows * padded_columns]]
    scales = [
        struct.unpack_from("<H", payload, scale_offset + index * 2)[0]
        for index in range(rows * groups_per_row)
    ]
    return codes, scales


def _validate_q4_shape(rows: int, columns: int, values: Sequence[float]) -> None:
    if type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0:
        raise ValueError("Q4G64 shape requires positive integer rows and columns")
    if len(values) != rows * columns:
        raise ValueError("Q4G64 source value count does not match shape")


def _q4_scale_bits_from_raw(raw_scale: float, nonzero: bool) -> int:
    try:
        scale_bits = _fp16_bits(_f32(raw_scale))
    except ValueError as error:
        raise ValueError("Q4G64 scale overflows FP16") from error
    if scale_bits == 0 and nonzero:
        scale_bits = FP16_MIN_SUBNORMAL_BITS
    scale = _fp16_value(scale_bits)
    if nonzero and (not math.isfinite(scale) or scale <= 0.0):
        raise ValueError("Q4G64 scale is not finite and positive")
    return scale_bits


def _q4_group_codes(values: Sequence[float], scale_bits: int) -> list[int]:
    if scale_bits == 0:
        return [0] * len(values)
    reciprocal = _f32(1.0 / _fp16_value(scale_bits))
    return [
        max(Q4_QMIN, min(Q4_QMAX, _round_even(_f32(float(item) * reciprocal))))
        for item in values
    ]


def _q4_group_sse(
    values: Sequence[float], codes: Sequence[int], scale_bits: int
) -> float:
    scale = _fp16_value(scale_bits)
    return math.fsum(
        (float(item) - float(code) * scale) ** 2
        for item, code in zip(values, codes, strict=True)
    )


def _q4_mse_scale_bits(values: Sequence[float]) -> int:
    maximum = max(abs(float(item)) for item in values)
    best_bits = _q4_scale_bits_from_raw(
        maximum / float(Q4_QMAX), maximum != 0.0
    )
    current_bits = best_bits
    best_codes = _q4_group_codes(values, best_bits)
    best_sse = _q4_group_sse(values, best_codes, best_bits)

    for _ in range(MSE_REFINEMENT_STEPS):
        current_codes = _q4_group_codes(values, current_bits)
        numerator = math.fsum(
            float(item) * float(code)
            for item, code in zip(values, current_codes, strict=True)
        )
        denominator = sum(code * code for code in current_codes)
        raw_scale = numerator / float(denominator) if denominator else 0.0
        current_bits = _q4_scale_bits_from_raw(raw_scale, maximum != 0.0)
        candidate_codes = _q4_group_codes(values, current_bits)
        candidate_sse = _q4_group_sse(values, candidate_codes, current_bits)
        if candidate_sse < best_sse:
            best_bits = current_bits
            best_sse = candidate_sse
    return best_bits


def _encode_q4g64(
    values: Sequence[float], rows: int, columns: int, *, refined: bool
) -> bytes:
    _validate_q4_shape(rows, columns, values)
    padded_columns = _align_up(columns, K_ALIGNMENT)
    groups_per_row = padded_columns // Q4_GROUP_SIZE
    base_bytes = rows * padded_columns // 2
    scale_offset = _align_up(base_bytes, PLANE_ALIGNMENT)
    payload = bytearray(scale_offset + rows * groups_per_row * 2)

    for row in range(rows):
        source = [float(item) for item in values[row * columns : (row + 1) * columns]]
        if not all(math.isfinite(item) for item in source):
            raise ValueError("Q4G64 source contains NaN or infinity")
        padded = source + [0.0] * (padded_columns - columns)
        for group in range(groups_per_row):
            begin = group * Q4_GROUP_SIZE
            logical = padded[begin : begin + Q4_GROUP_SIZE]
            if refined:
                scale_bits = _q4_mse_scale_bits(logical)
            else:
                maximum = max(abs(item) for item in logical)
                scale_bits = _q4_scale_bits_from_raw(
                    maximum / float(Q4_QMAX), maximum != 0.0
                )
            codes = _q4_group_codes(logical, scale_bits)
            for offset in range(0, Q4_GROUP_SIZE, 2):
                base = row * (padded_columns // 2) + group * 32 + offset // 2
                payload[base] = (
                    (codes[offset] & 0x0F) | ((codes[offset + 1] & 0x0F) << 4)
                )
            scale_index = (row * groups_per_row + group) * 2
            struct.pack_into("<H", payload, scale_offset + scale_index, scale_bits)
    return bytes(payload)


def encode_q4g64_reference(values: Sequence[float], rows: int, columns: int) -> bytes:
    """Encode represented values with canonical Q4G64 absmax scales."""

    return _encode_q4g64(values, rows, columns, refined=False)


def encode_q4g64_mse_reference(
    values: Sequence[float], rows: int, columns: int
) -> bytes:
    """Encode represented values with the source-only Q4G64 MSE trajectory."""

    return _encode_q4g64(values, rows, columns, refined=True)


def encode_q4g64_n16k16_reference(
    values: Sequence[float], rows: int, columns: int, *, refined: bool = False
) -> bytes:
    """Independently transpose the logical scalar Q4 encoding to N16/K16 storage."""
    if rows % 16:
        raise ValueError("N16/K16 Q4G64 requires rows divisible by 16")
    logical = _encode_q4g64(values, rows, columns, refined=refined)
    padded_columns = _align_up(columns, K_ALIGNMENT)
    groups = padded_columns // Q4_GROUP_SIZE
    code_bytes = rows * padded_columns // 2
    scale_offset = _align_up(code_bytes, PLANE_ALIGNMENT)
    result = bytearray(scale_offset + rows * groups * 2)
    for tile in range(rows // 16):
        for group in range(groups):
            for pair in range(4):
                for lane in range(16):
                    source = (tile * 16 + lane) * (padded_columns // 2) + group * 32 + pair * 8
                    target = (((tile * groups + group) * 4 + pair) * 16 + lane) * 8
                    result[target:target + 8] = logical[source:source + 8]
            for lane in range(16):
                source = scale_offset + ((tile * 16 + lane) * groups + group) * 2
                target = scale_offset + ((tile * groups + group) * 16 + lane) * 2
                result[target:target + 2] = logical[source:source + 2]
    return bytes(result)


def decode_q4g64_reference(
    payload: bytes, rows: int, columns: int
) -> tuple[list[int], list[int]]:
    """Return canonical signed Q4 codes and FP16 scale words for inspection."""

    _validate_q4_shape(rows, columns, [0.0] * (rows * columns))
    padded_columns = _align_up(columns, K_ALIGNMENT)
    groups_per_row = padded_columns // Q4_GROUP_SIZE
    base_bytes = rows * padded_columns // 2
    scale_offset = _align_up(base_bytes, PLANE_ALIGNMENT)
    expected = scale_offset + rows * groups_per_row * 2
    if len(payload) != expected:
        raise ValueError(f"Q4G64 payload has {len(payload)} bytes, expected {expected}")
    codes: list[int] = []
    scales: list[int] = []
    for row in range(rows):
        for group in range(groups_per_row):
            base = row * (padded_columns // 2) + group * 32
            for byte in payload[base:base + 32]:
                low, high = byte & 0x0F, byte >> 4
                codes.extend((low - 16 if low >= 8 else low,
                              high - 16 if high >= 8 else high))
            scale_index = (row * groups_per_row + group) * 2
            scales.append(struct.unpack_from("<H", payload, scale_offset + scale_index)[0])
    return codes, scales


def encode_e4m3fn_reference(source: float) -> int:
    """Encode one finite value as OCP E4M3FN with RNE and saturation."""

    value = float(source)
    if not math.isfinite(value):
        raise ValueError("E4M3 rowwise source contains NaN or infinity")
    negative = math.copysign(1.0, value) < 0.0
    magnitude = abs(value)
    if magnitude == 0.0:
        return 0x80 if negative else 0x00
    if magnitude >= E4M3FN_MAX:
        return (0x80 if negative else 0x00) | 0x7E

    best = 0
    best_distance = magnitude
    for candidate in range(1, 0x7F):
        decoded = _decode_e4m3fn_positive(candidate)
        distance = abs(magnitude - decoded)
        if distance < best_distance or (
            distance == best_distance and candidate % 2 == 0 and best % 2 != 0
        ):
            best = candidate
            best_distance = distance
    return (0x80 if negative else 0x00) | best


def _decode_e4m3fn_positive(word: int) -> float:
    exponent = (word >> 3) & 0x0F
    fraction = word & 0x07
    if exponent == 0:
        return fraction * (2.0**-9)
    if exponent == 0x0F and fraction == 0x07:
        return math.nan
    return (1.0 + fraction / 8.0) * (2.0 ** (exponent - 7))


def decode_e4m3fn_reference(word: int) -> float:
    """Decode one finite OCP E4M3FN word independently of Torch."""

    if type(word) is not int or not 0 <= word <= 0xFF:
        raise ValueError("E4M3FN word must be an integer in [0, 255]")
    if word & 0x7F == 0x7F:
        raise ValueError("E4M3 rowwise payload contains an E4M3FN NaN word")
    magnitude = _decode_e4m3fn_positive(word & 0x7F)
    return -magnitude if word & 0x80 else magnitude


def _validate_e4m3_rowwise_shape(
    rows: int, columns: int, value_count: int | None = None
) -> None:
    if type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0:
        raise ValueError("E4M3 rowwise shape requires positive integer rows and columns")
    if value_count is not None and value_count != rows * columns:
        raise ValueError("E4M3 rowwise source value count does not match shape")


def e4m3_rowwise_payload_size(rows: int, columns: int) -> int:
    """Return bytes for row-major K128-padded codes followed by row FP32 scales."""

    _validate_e4m3_rowwise_shape(rows, columns)
    code_bytes = rows * _align_up(columns, K_ALIGNMENT)
    return _align_up(code_bytes, PLANE_ALIGNMENT) + rows * 4


def encode_e4m3_rowwise_reference(
    values: Sequence[float], rows: int, columns: int
) -> bytes:
    """Scalar oracle for BF16-source rowwise E4M3FN weight encoding.

    The represented values are scaled per output row by an exactly stored
    little-endian FP32 ``max(abs(row)) / 448`` value. Codes are direct
    row-major E4M3FN bytes with zero-filled K128 padding; the FP32 scale plane
    begins at the next 256-byte boundary after the code plane.
    """

    _validate_e4m3_rowwise_shape(rows, columns, len(values))
    padded_columns = _align_up(columns, K_ALIGNMENT)
    code_bytes = rows * padded_columns
    scale_offset = _align_up(code_bytes, PLANE_ALIGNMENT)
    payload = bytearray(scale_offset + rows * 4)
    for row in range(rows):
        logical = [float(item) for item in values[row * columns : (row + 1) * columns]]
        if not all(math.isfinite(item) for item in logical):
            raise ValueError("E4M3 rowwise source contains NaN or infinity")
        maximum = max(abs(item) for item in logical)
        try:
            scale = _f32(maximum / E4M3FN_MAX) if maximum != 0.0 else 0.0
        except OverflowError as error:
            raise ValueError("E4M3 rowwise scale overflows FP32") from error
        if maximum != 0.0 and (not math.isfinite(scale) or scale <= 0.0):
            raise ValueError("E4M3 rowwise scale is not finite and positive")
        base = row * padded_columns
        if scale != 0.0:
            for column, item in enumerate(logical):
                payload[base + column] = encode_e4m3fn_reference(_f32(item / scale))
        struct.pack_into("<f", payload, scale_offset + row * 4, scale)
    return bytes(payload)


def decode_e4m3_rowwise_reference(
    payload: bytes, rows: int, columns: int
) -> tuple[list[float], list[int], list[float]]:
    """Decode logical FP32 values, physical code words, and row scales."""

    _validate_e4m3_rowwise_shape(rows, columns)
    padded_columns = _align_up(columns, K_ALIGNMENT)
    code_bytes = rows * padded_columns
    scale_offset = _align_up(code_bytes, PLANE_ALIGNMENT)
    expected = scale_offset + rows * 4
    if len(payload) != expected:
        raise ValueError(
            f"E4M3 rowwise payload has {len(payload)} bytes, expected {expected}"
        )
    codes = list(payload[:code_bytes])
    if any(word & 0x7F == 0x7F for word in codes):
        raise ValueError("E4M3 rowwise payload contains an E4M3FN NaN word")
    if any(payload[code_bytes:scale_offset]):
        raise ValueError("E4M3 rowwise scale-plane alignment padding is nonzero")
    scales = [struct.unpack_from("<f", payload, scale_offset + row * 4)[0] for row in range(rows)]
    decoded: list[float] = []
    for row, scale in enumerate(scales):
        if (
            not math.isfinite(scale)
            or scale < 0.0
            or (scale == 0.0 and math.copysign(1.0, scale) < 0.0)
        ):
            raise ValueError("E4M3 rowwise scale is not finite and nonnegative")
        physical = codes[row * padded_columns : (row + 1) * padded_columns]
        if any(physical[columns:]):
            raise ValueError("E4M3 rowwise K128 padding is nonzero")
        if scale == 0.0 and any(word != 0 for word in physical):
            raise ValueError("E4M3 rowwise zero-scale row has nonzero codes")
        for word in physical[:columns]:
            decoded.append(_f32(decode_e4m3fn_reference(word) * scale))
    return decoded, codes, scales
