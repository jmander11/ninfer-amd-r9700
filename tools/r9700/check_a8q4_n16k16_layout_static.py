#!/usr/bin/env python3
"""Exact offset bijection and fail-closed gfx1201 gate for N16/K16 Q4."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PREFILL = "a8q4_n16k16_prefill_p2048_m64n128"
DECODE = "a8q4_n16k16_decode_t1_wmma32"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def interval(text: str, symbol: str) -> str:
    start = re.search(rf"(?m)^\s*{symbol}:.*$", text)
    require(start is not None, f"missing exact kernel interval: {symbol}")
    end = re.search(r"(?m)^.*; -- Begin function ", text[start.end():])
    stop = len(text) if end is None else start.end() + end.start()
    return text[start.start():stop]


def metadata(text: str, symbol: str, field: str) -> int:
    start = text.find(f".amdhsa_kernel {symbol}")
    require(start >= 0, f"missing metadata: {symbol}")
    end = text.find(".end_amdhsa_kernel", start)
    body = text[start:end]
    hit = re.search(rf"(?m)^\s*\.amdhsa_{field}\s+(\d+)\s*$", body)
    require(hit is not None, f"missing {field}: {symbol}")
    return int(hit.group(1))


def resource_comment(body: str, field: str) -> int:
    hit = re.search(rf"(?m)^; {field}:\s*(\d+)\s*$", body)
    require(hit is not None, f"missing resource comment {field}")
    return int(hit.group(1))


def yaml_kernel_field(text: str, symbol: str, field: str) -> int:
    metadata_start = text.find(".amdgpu_metadata")
    metadata_end = text.find(".end_amdgpu_metadata", metadata_start)
    require(metadata_start >= 0 and metadata_end >= 0, "missing YAML metadata envelope")
    starts = [m.start() for m in re.finditer(r"(?m)^  - \.args:\s*$",
                                             text[metadata_start:metadata_end])]
    starts = [metadata_start + offset for offset in starts]
    selected = None
    for index, item_start in enumerate(starts):
        item_end = starts[index + 1] if index + 1 < len(starts) else metadata_end
        item = text[item_start:item_end]
        if re.search(rf"(?m)^\s*\.name:\s+{re.escape(symbol)}\s*$", item):
            require(selected is None, f"duplicate YAML kernel metadata: {symbol}")
            selected = item
    require(selected is not None, f"missing YAML kernel metadata: {symbol}")
    body = selected
    hit = re.search(rf"(?m)^\s*\.{re.escape(field)}:\s*(\d+)\s*$", body)
    require(hit is not None, f"missing YAML {field}: {symbol}")
    return int(hit.group(1))


def oracle() -> None:
    for rows, groups in ((16, 1), (128, 80), (5120, 272), (34816, 80), (248320, 80)):
        require(rows % 16 == 0, "N16 layout requires a complete row tile")
        # The mixed-radix mapping is dense by construction. Exhaustively test
        # one complete CTA tile, then check endpoints/inversion at the larger
        # production extents without allocating tens of millions of Python ints.
        if rows > 128:
            count = rows * groups * 4
            probes = ((0, 0, 0), (15, groups - 1, 3),
                      (rows - 16, 0, 0), (rows - 1, groups - 1, 3))
            for row, group, pair in probes:
                index = (((row >> 4) * groups + group) * 4 + pair) * 16 + (row & 15)
                decoded_row = ((index // (groups * 64)) << 4) | (index & 15)
                rem = (index // 16) % (groups * 4)
                require((decoded_row, rem // 4, rem % 4) == (row, group, pair),
                        "large-shape code offset inverse failed")
            require((((rows - 1) >> 4) * groups * 64 +
                     (groups - 1) * 64 + 3 * 16 + 15) == count - 1,
                    "large-shape code plane endpoint failed")
            require((((rows - 1) >> 4) * groups * 16 +
                     (groups - 1) * 16 + 15) == rows * groups - 1,
                    "large-shape scale plane endpoint failed")
            continue
        seen = set()
        for row in range(rows):
            for group in range(groups):
                for pair in range(4):
                    index = (((row >> 4) * groups + group) * 4 + pair) * 16 + (row & 15)
                    require(index not in seen, "code offset collision")
                    seen.add(index)
                    decoded_row = ((index // (groups * 64)) << 4) | (index & 15)
                    rem = (index // 16) % (groups * 4)
                    require((decoded_row, rem // 4, rem % 4) == (row, group, pair),
                            "code offset inverse failed")
        require(len(seen) == rows * groups * 4 and max(seen) + 1 == len(seen),
                "code plane is not a byte-preserving bijection")
        scales = {((row >> 4) * groups + group) * 16 + (row & 15)
                  for row in range(rows) for group in range(groups)}
        require(len(scales) == rows * groups and max(scales) + 1 == len(scales),
                "scale plane is not a byte-preserving bijection")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args()
    oracle()
    text = args.assembly.read_text()
    prefill = interval(text, PREFILL)
    decode = interval(text, DECODE)
    for symbol, body, vgpr_ceiling, lds_expected, wmma_expected in (
        (PREFILL, prefill, 96, 17152, 8),
        (DECODE, decode, 64, 0, 4),
    ):
        vgpr = metadata(text, symbol, "next_free_vgpr")
        lds = metadata(text, symbol, "group_segment_fixed_size")
        private = metadata(text, symbol, "private_segment_fixed_size")
        require(vgpr <= vgpr_ceiling, f"{symbol}: VGPR {vgpr}>{vgpr_ceiling}")
        require(lds == lds_expected, f"{symbol}: LDS {lds}!={lds_expected}")
        require(private == 0, f"{symbol}: private bytes {private}")
        require(resource_comment(body, "ScratchSize") == 0, f"{symbol}: scratch")
        require(yaml_kernel_field(text, symbol, "sgpr_spill_count") == 0,
                f"{symbol}: SGPR spills")
        require(yaml_kernel_field(text, symbol, "vgpr_spill_count") == 0,
                f"{symbol}: VGPR spills")
        require(resource_comment(body, "Occupancy") == 16, f"{symbol}: occupancy !=16")
        wmma = re.findall(r"(?m)^\s*v_wmma_i32_16x16x32_iu4\b.*$", body)
        require(len(wmma) == wmma_expected,
                f"{symbol}: IU4 sites {len(wmma)}!={wmma_expected}")
        require(sum("neg_lo:[0,1,0]" in line for line in wmma) == wmma_expected // 2,
                f"{symbol}: low signedness")
        require(sum("neg_lo:[1,1,0]" in line for line in wmma) == wmma_expected // 2,
                f"{symbol}: high signedness")
    first_wmma = prefill.find("v_wmma_i32_16x16x32_iu4")
    before = prefill[:first_wmma]
    require(len(re.findall(r"(?m)^\s*global_load_b64\b", before)) >= 1,
            "prefill: no b64 N16 weight load before WMMA")
    # Three successor payload loads: A-low b32, A-high b32, and W b64. The
    # prologue can contribute equivalent sites, so enforce the complete loop
    # shape by exact static VMEM widths and absence of a second successor W b32.
    require(len(re.findall(r"(?m)^\s*global_load_b64\b", prefill)) == 2,
            "prefill: expected one prologue plus one successor b64 weight site")
    require(len(re.findall(r"(?m)^\s*s_barrier_signal\s+-1", prefill)) == 2 and
            len(re.findall(r"(?m)^\s*s_barrier_wait\s+-1", prefill)) == 2,
            "prefill: barrier topology changed")
    print("PASS offset_bijection=true prefill_vgpr<=96 prefill_lds=17152 "
          "occupancy=16 spills=0 iu4=8 successor_weight_b64_sites=1 decode_vgpr<=64 "
          "decode_lds=0 decode_iu4=4")


if __name__ == "__main__":
    main()
