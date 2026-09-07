#!/usr/bin/env python3
"""Fail-closed gfx1201 resource/ISA check for the attention parity kernels."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def section(text: str, fragment: str) -> str:
    match = re.search(
        rf"(?ms)^\s*\.section\s+\.text\.[^\n]*{re.escape(fragment)}[^\n]*\n"
        rf".*?(?=^\s*\.section\s+\.text\.|\Z)", text)
    if match is None:
        raise RuntimeError(f"missing assembly section: {fragment}")
    return match.group(0)


def resources(text: str, fragment: str) -> tuple[int, int, int, int, int, int]:
    name = re.search(rf"(?m)^\s*\.name:\s+.*{re.escape(fragment)}.*$", text)
    if name is None:
        raise RuntimeError(f"missing metadata: {fragment}")
    before = text[:name.start()]
    group = list(re.finditer(r"(?m)^\s*\.group_segment_fixed_size:\s+(\d+)\s*$", before))
    require(bool(group), f"missing LDS metadata: {fragment}")
    block = text[name.end():]
    next_name = re.search(r"(?m)^\s*\.name:", block)
    if next_name is not None:
        block = block[:next_name.start()]
    values = [int(group[-1].group(1))]
    for field in ("private_segment_fixed_size", "sgpr_count", "vgpr_count",
                  "sgpr_spill_count", "vgpr_spill_count", "wavefront_size",
                  "max_flat_workgroup_size"):
        found = re.search(rf"(?m)^\s*\.{field}:\s+(\d+)\s*$", block)
        require(found is not None, f"missing {field}: {fragment}")
        values.append(int(found.group(1)))
    require(values[4] == 0 and values[5] == 0 and values[6] == 32,
            f"spill/wave contract failed for {fragment}: {values}")
    return values[0], values[1], values[2], values[3], values[6], values[7]


def occupancy(text: str, fragment: str) -> int:
    match = re.search(
        rf"(?s)\.set [^\n]*{re.escape(fragment)}[^\n]*\.num_vgpr[^\n]*\n"
        rf".*?; Occupancy:\s*(\d+)", text)
    if match is None:
        raise RuntimeError(f"missing occupancy: {fragment}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args()
    text = args.assembly.read_text(encoding="utf-8")
    expected = {
        "qk_wmma_batched_w5_kernelILb1EE": (0, 0, 32, 24, 32, 1024),
        "softmax_wmma_scores_batched_w5_in_place_kernel": (76, 0, 20, 23, 32, 1024),
        "pv_vector_batched_w5_kernelILj16ELb0ELb0EE": (0, 0, 42, 15, 32, 1024),
    }
    for symbol, wanted in expected.items():
        actual = resources(text, symbol)
        require(actual == wanted, f"{symbol} resources {actual} != {wanted}")
        require(occupancy(text, symbol) == 16, f"{symbol} occupancy is not 16")
    require("qk_wmma_batched_w5_kernelILb0EE" not in text,
            "unqualified feature-fastest K specialization was emitted")
    for symbol in ("qk_wmma_batched_w5_kernelILb1EE",):
        body = section(text, symbol)
        require(body.count("v_wmma_f32_16x16x16_fp8_fp8") == 1,
                f"{symbol} does not use native gfx1201 FP8 WMMA")
    print("attention parity gfx1201 static checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
