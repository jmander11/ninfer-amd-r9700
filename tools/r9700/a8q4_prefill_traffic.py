#!/usr/bin/env python3
"""Host-only traffic/roofline models for gfx1201 A8Q4G64 and A8W8G32 prefill."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


GROUP = 64
OUTPUT_BYTES = 2
W8_GROUP = 32
W8_K_ALIGNMENT = 128


@dataclass(frozen=True)
class Tile:
    name: str
    m: int
    n: int
    waves: int
    stage_unique_operands: bool


OLD_WAVE16 = Tile("wave16x16-global", 16, 16, 1, False)
M32_N64 = Tile("cta-m32-n64-lds", 32, 64, 8, True)
M64_N64 = Tile("cta-m64-n64-lds", 64, 64, 16, True)
M64_N128 = Tile("cta-m64-n128-lds", 64, 128, 16, True)
M64_N256 = Tile("cta-m64-n256-lds", 64, 256, 16, True)
TILES = (OLD_WAVE16, M32_N64, M64_N64, M64_N128, M64_N256)
W8_OLD_WAVE16 = Tile("w8-wave16x16-global", 16, 16, 1, False)
W8_M64_N64 = Tile("w8-cta-m64-n64-lds", 64, 64, 16, True)
W8_TILES = (W8_OLD_WAVE16, W8_M64_N64)


@dataclass(frozen=True)
class Shape:
    tokens: int
    rows: int
    columns: int

    def validate(self) -> None:
        if self.tokens <= 0 or self.rows <= 0 or self.columns <= 0:
            raise ValueError("shape dimensions must be positive")
        if self.columns % GROUP:
            raise ValueError("A8Q4G64 padded columns must be a multiple of 64")


@dataclass(frozen=True)
class Traffic:
    tile: str
    blocks: int
    waves: int
    activation_code_bytes: int
    weight_code_bytes: int
    activation_scale_bytes: int
    weight_scale_bytes: int
    status_bytes: int
    output_bytes: int
    total_requested_bytes: int
    unique_tensor_bytes: int
    request_over_unique: float
    logical_ops: int
    issued_iu4_ops: int
    logical_ops_per_requested_byte: float
    issued_iu4_ops_per_requested_byte: float
    static_lds_bytes: int


@dataclass(frozen=True)
class W8Traffic:
    tile: str
    padded_columns: int
    blocks: int
    waves: int
    activation_code_bytes: int
    weight_code_bytes: int
    activation_scale_bytes: int
    weight_scale_bytes: int
    status_bytes: int
    output_bytes: int
    total_requested_bytes: int
    unique_tensor_bytes: int
    request_over_unique: float
    logical_ops: int
    issued_iu8_ops: int
    logical_ops_per_requested_byte: float
    issued_iu8_ops_per_requested_byte: float
    static_lds_bytes: int


def _tile_extent(origin: int, tile: int, extent: int) -> int:
    return max(0, min(tile, extent - origin))


def requested_traffic(shape: Shape, tile: Tile) -> Traffic:
    """Count source-level memory requests and logical/issued matrix operations.

    The old wave kernel reloads A scales once per output and W scales twice per row
    (one load from each lane half). The CTA designs stage each represented code and
    scale once per CTA/G64 tile. IU4 work is twice the logical A8xQ4 work because A8
    is exactly decomposed into unsigned-low and signed-high nibble planes.
    """
    shape.validate()
    output_tiles = (tile.m // 16) * (tile.n // 16)
    if (tile.m % 16 or tile.n % 16 or tile.waves <= 0 or
            output_tiles % tile.waves):
        raise ValueError("tile must assign an integral number of 16x16 WMMA tiles per wave")

    groups = shape.columns // GROUP
    a_codes = w_codes = a_scales = w_scales = status = output = 0
    blocks = 0
    for m0 in range(0, shape.tokens, tile.m):
        valid_m = _tile_extent(m0, tile.m, shape.tokens)
        for n0 in range(0, shape.rows, tile.n):
            valid_n = _tile_extent(n0, tile.n, shape.rows)
            blocks += 1
            output += valid_m * valid_n * OUTPUT_BYTES
            if tile.stage_unique_operands:
                # The uniform scalar status load is issued independently by every wave.
                status += 4 * tile.waves
                a_codes += groups * valid_m * GROUP       # two Q4 planes == one byte/K
                w_codes += groups * valid_n * GROUP // 2  # one Q4 plane
                a_scales += groups * valid_m * 2
                w_scales += groups * valid_n * 2
            else:
                # The uniform status pointer compiles to one scalar dword load per wave.
                status += 4
                a_codes += groups * valid_m * GROUP
                w_codes += groups * valid_n * GROUP // 2
                # Current fragment ownership reloads the token scale for every output row.
                a_scales += groups * valid_m * 16 * 2
                # Each row scale is loaded by both 16-lane halves.
                w_scales += groups * valid_n * 2 * 2

    total = a_codes + w_codes + a_scales + w_scales + status + output
    unique = (
        shape.tokens * shape.columns
        + shape.rows * shape.columns // 2
        + shape.tokens * groups * 2
        + shape.rows * groups * 2
        + shape.tokens * shape.rows * OUTPUT_BYTES
        + 4
    )
    logical_ops = 2 * shape.tokens * shape.rows * shape.columns
    rounded_m = math.ceil(shape.tokens / tile.m) * tile.m
    rounded_n = math.ceil(shape.rows / tile.n) * tile.n
    issued_iu4_ops = 4 * rounded_m * rounded_n * shape.columns
    lds = 0
    if tile.stage_unique_operands:
        lds = tile.m * GROUP + tile.n * GROUP // 2 + (tile.m + tile.n) * 2
    return Traffic(
        tile=tile.name,
        blocks=blocks,
        waves=blocks * tile.waves,
        activation_code_bytes=a_codes,
        weight_code_bytes=w_codes,
        activation_scale_bytes=a_scales,
        weight_scale_bytes=w_scales,
        status_bytes=status,
        output_bytes=output,
        total_requested_bytes=total,
        unique_tensor_bytes=unique,
        request_over_unique=total / unique,
        logical_ops=logical_ops,
        issued_iu4_ops=issued_iu4_ops,
        logical_ops_per_requested_byte=logical_ops / total,
        issued_iu4_ops_per_requested_byte=issued_iu4_ops / total,
        static_lds_bytes=lds,
    )


def requested_w8_traffic(shape: Shape, tile: Tile) -> W8Traffic:
    """Count the explicit signed-A8G32/W8G32 consumer contract.

    Unlike Q4, one IU8 path represents the logical dot directly. Both kernels
    traverse canonical K128 padding in complete G32 groups. The old wave reloads
    token scales for every output row and row scales from both lane halves; the
    CTA stages each represented code and scale once per 64x64 CTA/G32 group.
    """
    if shape.tokens <= 0 or shape.rows <= 0 or shape.columns <= 0:
        raise ValueError("shape dimensions must be positive")
    if tile not in W8_TILES:
        raise ValueError("W8 traffic requires an explicit W8 tile")
    if tile.m % 16 or tile.n % 16 or tile.waves != (tile.m // 16) * (tile.n // 16):
        raise ValueError("tile must be an exact collection of 16x16 wave WMMA tiles")

    padded = math.ceil(shape.columns / W8_K_ALIGNMENT) * W8_K_ALIGNMENT
    groups = padded // W8_GROUP
    a_codes = w_codes = a_scales = w_scales = status = output = 0
    blocks = 0
    for m0 in range(0, shape.tokens, tile.m):
        valid_m = _tile_extent(m0, tile.m, shape.tokens)
        for n0 in range(0, shape.rows, tile.n):
            valid_n = _tile_extent(n0, tile.n, shape.rows)
            blocks += 1
            output += valid_m * valid_n * OUTPUT_BYTES
            if tile.stage_unique_operands:
                status += 4 * tile.waves
                a_codes += groups * valid_m * W8_GROUP
                w_codes += groups * valid_n * W8_GROUP
                a_scales += groups * valid_m * 2
                w_scales += groups * valid_n * 2
            else:
                status += 4
                a_codes += groups * valid_m * W8_GROUP
                w_codes += groups * valid_n * W8_GROUP
                a_scales += groups * valid_m * 16 * 2
                w_scales += groups * valid_n * 2 * 2

    total = a_codes + w_codes + a_scales + w_scales + status + output
    unique = (
        shape.tokens * padded
        + shape.rows * padded
        + shape.tokens * groups * 2
        + shape.rows * groups * 2
        + shape.tokens * shape.rows * OUTPUT_BYTES
        + 4
    )
    logical_ops = 2 * shape.tokens * shape.rows * shape.columns
    rounded_m = math.ceil(shape.tokens / tile.m) * tile.m
    rounded_n = math.ceil(shape.rows / tile.n) * tile.n
    issued_iu8_ops = 2 * rounded_m * rounded_n * padded
    lds = 0
    if tile.stage_unique_operands:
        lds = (tile.m + tile.n) * W8_GROUP + (tile.m + tile.n) * 2
    return W8Traffic(
        tile=tile.name,
        padded_columns=padded,
        blocks=blocks,
        waves=blocks * tile.waves,
        activation_code_bytes=a_codes,
        weight_code_bytes=w_codes,
        activation_scale_bytes=a_scales,
        weight_scale_bytes=w_scales,
        status_bytes=status,
        output_bytes=output,
        total_requested_bytes=total,
        unique_tensor_bytes=unique,
        request_over_unique=total / unique,
        logical_ops=logical_ops,
        issued_iu8_ops=issued_iu8_ops,
        logical_ops_per_requested_byte=logical_ops / total,
        issued_iu8_ops_per_requested_byte=issued_iu8_ops / total,
        static_lds_bytes=lds,
    )
def with_roofline(traffic: Traffic, bandwidth_gbps: float | None,
                  iu4_tops: float | None) -> dict[str, object]:
    result: dict[str, object] = asdict(traffic)
    bounds: dict[str, float] = {}
    if bandwidth_gbps is not None:
        if bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be positive")
        bounds["requested_bandwidth_service_time_ms"] = (
            traffic.total_requested_bytes / (bandwidth_gbps * 1e9) * 1e3
        )
        bounds["bandwidth_logical_tops"] = (
            traffic.logical_ops_per_requested_byte * bandwidth_gbps / 1000.0
        )
    if iu4_tops is not None:
        if iu4_tops <= 0:
            raise ValueError("iu4_tops must be positive")
        bounds["issued_iu4_service_time_ms"] = traffic.issued_iu4_ops / (iu4_tops * 1e12) * 1e3
        bounds["compute_logical_tops"] = iu4_tops / 2.0
    if bounds:
        bounds["combined_service_roof_ms"] = max(
            bounds.get("requested_bandwidth_service_time_ms", 0.0),
            bounds.get("issued_iu4_service_time_ms", 0.0),
        )
        result["roofline"] = bounds
    return result


def with_w8_roofline(traffic: W8Traffic, bandwidth_gbps: float | None,
                     iu8_tops: float | None) -> dict[str, object]:
    result: dict[str, object] = asdict(traffic)
    bounds: dict[str, float] = {}
    if bandwidth_gbps is not None:
        if bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be positive")
        bounds["requested_bandwidth_service_time_ms"] = (
            traffic.total_requested_bytes / (bandwidth_gbps * 1e9) * 1e3
        )
        bounds["bandwidth_logical_tops"] = (
            traffic.logical_ops_per_requested_byte * bandwidth_gbps / 1000.0
        )
    if iu8_tops is not None:
        if iu8_tops <= 0:
            raise ValueError("iu8_tops must be positive")
        bounds["issued_iu8_service_time_ms"] = traffic.issued_iu8_ops / (iu8_tops * 1e12) * 1e3
        bounds["compute_logical_tops"] = iu8_tops * traffic.logical_ops / traffic.issued_iu8_ops
    if bounds:
        bounds["combined_service_roof_ms"] = max(
            bounds.get("requested_bandwidth_service_time_ms", 0.0),
            bounds.get("issued_iu8_service_time_ms", 0.0),
        )
        result["roofline"] = bounds
    return result


def parse_shape(text: str) -> Shape:
    try:
        tokens, rows, columns = (int(item) for item in text.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("shape must be T,N,K") from error
    shape = Shape(tokens, rows, columns)
    try:
        shape.validate()
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return shape


def parse_w8_shape(text: str) -> Shape:
    try:
        tokens, rows, columns = (int(item) for item in text.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("W8 shape must be T,N,logical-K") from error
    if tokens <= 0 or rows <= 0 or columns <= 0:
        raise argparse.ArgumentTypeError("W8 shape dimensions must be positive")
    return Shape(tokens, rows, columns)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Model source-requested traffic for A8Q4G64 and A8W8G32 prefill tiles.")
    parser.add_argument(
        "--shape", action="append", type=parse_shape,
        help="T,N,K; repeatable (defaults to the three dominant traced prefill shapes)")
    parser.add_argument(
        "--w8-shape", action="append", type=parse_w8_shape,
        help="T,N,logical-K; repeatable (defaults to four dominant mixed Text shapes)")
    parser.add_argument("--bandwidth-gbps", type=float)
    parser.add_argument("--iu4-tops", type=float)
    parser.add_argument("--iu8-tops", type=float)
    parser.add_argument("--out-json")
    args = parser.parse_args()
    shapes = args.shape or [
        Shape(4096, 34816, 5120),
        Shape(4096, 5120, 17408),
        Shape(4096, 12288, 5120),
    ]
    w8_shapes = args.w8_shape or [
        Shape(4096, 12288, 5120),
        Shape(4096, 5120, 17408),
        Shape(4096, 5120, 6144),
        Shape(4096, 7168, 5120),
    ]
    report = {
        "schema": "ninfer_r9700_quantized_prefill_traffic",
        "schema_version": 2,
        "scope": "source/ISA-requested bytes; cache coalescing, residency, and HBM traffic excluded",
        "roofline_scope": "scenario over a caller-supplied requested-byte service rate, not an HBM bound",
        "a8_decomposition": "unsigned-low plus signed-high IU4; issued IU4 ops are 2x logical",
        "a8q4g64": {"issued_math": "two IU4 paths for unsigned-low/signed-high A8", "shapes": [
            {
                "tokens": shape.tokens,
                "rows": shape.rows,
                "columns": shape.columns,
                "tiles": [
                    with_roofline(requested_traffic(shape, tile),
                                  args.bandwidth_gbps, args.iu4_tops)
                    for tile in TILES
                ],
            }
            for shape in shapes
        ]},
        "a8w8g32": {
            "issued_math": "two signed IU8 K16 instructions per exact G32 group",
            "canonical_k_alignment": W8_K_ALIGNMENT,
            "shapes": [
                {
                    "tokens": shape.tokens,
                    "rows": shape.rows,
                    "columns": shape.columns,
                    "tiles": [
                        with_w8_roofline(requested_w8_traffic(shape, tile),
                                         args.bandwidth_gbps, args.iu8_tops)
                        for tile in W8_TILES
                    ],
                }
                for shape in w8_shapes
            ],
        },
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out_json:
        with open(args.out_json, "x", encoding="utf-8") as output:
            output.write(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
