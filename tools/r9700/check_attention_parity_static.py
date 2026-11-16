#!/usr/bin/env python3
"""Fail-closed gfx1201 resource/ISA check for the DFlash split-context verify kernels."""

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
        rf".*?(?=^\s*\.section\s+\.text\.|\Z)",
        text,
    )
    if match is None:
        raise RuntimeError(f"missing assembly section: {fragment}")
    return match.group(0)


def resources(text: str, fragment: str) -> tuple[int, int, int, int, int, int, int]:
    name = re.search(rf"(?m)^\s*\.name:\s+.*{re.escape(fragment)}.*$", text)
    if name is None:
        raise RuntimeError(f"missing metadata: {fragment}")
    # Metadata keys are sorted: fields before `.name` belong to the text preceding it (the last
    # match there), fields after it end at the next kernel's `.name`.
    before = text[: name.start()]
    block = text[name.end() :]
    next_name = re.search(r"(?m)^\s*\.name:", block)
    if next_name is not None:
        block = block[: next_name.start()]

    def field_value(field: str, preceding: bool) -> int:
        pattern = rf"(?m)^\s*\.{field}:\s+(\d+)\s*$"
        found = list(re.finditer(pattern, before)) if preceding else [re.search(pattern, block)]
        require(bool(found) and found[-1] is not None, f"missing {field}: {fragment}")
        return int(found[-1].group(1))

    values = [field_value("group_segment_fixed_size", True)]
    for field in (
        "private_segment_fixed_size",
        "sgpr_count",
        "vgpr_count",
        "sgpr_spill_count",
        "vgpr_spill_count",
        "wavefront_size",
    ):
        values.append(field_value(field, False))
    values.append(field_value("max_flat_workgroup_size", True))
    # No scratch and no VGPR spills; SGPR spills (to VGPR lanes, no memory traffic) are pinned.
    require(
        values[1] == 0 and values[5] == 0 and values[6] == 32,
        f"spill/wave contract failed for {fragment}: {values}",
    )
    return values[0], values[1], values[2], values[3], values[4], values[6], values[7]


def occupancy(text: str, fragment: str) -> int:
    match = re.search(
        rf"(?s)\.set [^\n]*{re.escape(fragment)}[^\n]*\.num_vgpr[^\n]*\n"
        rf".*?; Occupancy:\s*(\d+)",
        text,
    )
    if match is None:
        raise RuntimeError(f"missing occupancy: {fragment}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args()
    text = args.assembly.read_text(encoding="utf-8")

    # DFlash split-context verification: the packed, warp-specialized, double-buffered dense verify
    # kernel (three compute waves plus loader waves, 16-key blocks) for each cache codec and both
    # partition plans (LivePlan false/true), and its merge. NVFP4 and TQ select five or six loader
    # waves; FP8/INT4 uses three.
    def verify(codec: str, loaders: int, live: int) -> str:
        return f"dense_verify_kernelINS2_{codec}ELj16ELj{loaders}ELb{live}EE"

    expected: dict[str, tuple[tuple[int, ...], int]] = {}
    for live in (0, 1):
        for loaders, threads in ((5, 256), (6, 288)):
            expected[verify("15Nvfp4DenseCodec", loaders, live)] = (
                (41664, 0, 38, 234, 0, 32, threads),
                6,
            )
            expected[verify("12TqDenseCodec", loaders, live)] = (
                (41664, 0, 40, 234, 0, 32, threads),
                6,
            )
    # FP8/INT4 pins its small SGPR-to-VGPR-lane spill count.
    expected[verify("17Fp8Int4DenseCodec", 3, 0)] = ((41664, 0, 107, 235, 4, 32, 192), 5)
    expected[verify("17Fp8Int4DenseCodec", 3, 1)] = ((41664, 0, 107, 235, 3, 32, 192), 5)
    expected["dense_verify_merge_kernelILb0EE"] = ((768, 0, 20, 31, 0, 32, 256), 16)
    expected["dense_verify_merge_kernelILb1EE"] = ((768, 0, 18, 34, 0, 32, 256), 16)
    for symbol, (wanted, wanted_occupancy) in expected.items():
        actual = resources(text, symbol)
        require(actual == wanted, f"{symbol} resources {actual} != {wanted}")
        require(
            occupancy(text, symbol) == wanted_occupancy,
            f"{symbol} occupancy is not {wanted_occupancy}",
        )
    # QK/PV WMMAs per verify kernel: FP8/INT4 stages QK in BF16 and PV in FP16; NVFP4 and TQ stage
    # both in FP16.
    for symbol in expected:
        if "merge" in symbol:
            continue
        body = section(text, symbol)
        bf16 = body.count("v_wmma_f32_16x16x16_bf16")
        f16 = body.count("v_wmma_f32_16x16x16_f16")
        wanted_wmma = (16, 16) if "Fp8Int4" in symbol else (0, 32)
        require(
            (bf16, f16) == wanted_wmma,
            f"{symbol} WMMAs (bf16, f16) {(bf16, f16)} != {wanted_wmma}",
        )
    print("attention parity gfx1201 static checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
