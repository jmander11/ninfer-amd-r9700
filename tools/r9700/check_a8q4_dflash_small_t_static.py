#!/usr/bin/env python3
"""Fail-closed gfx1201 ISA/resource gate for the DFlash small-T Q4 challenger."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

WIDTHS = (4, 5, 6, 8, 10, 12, 15, 16, 18, 20, 24)
EXACT_VGPR = {4: 26, 5: 29, 6: 41, 8: 51, 10: 60, 12: 54,
              15: 64, 16: 67, 18: 74, 20: 84, 24: 99}
EXACT_OCCUPANCY = {tokens: (12 if tokens == 24 else 16) for tokens in WIDTHS}


def _symbol(assembly: str, tokens: int) -> tuple[str, str]:
    marker = f"a8q4g64_linear_dflash_small_t_kernelILj{tokens}EE"
    matches = list(re.finditer(r"; -- Begin function ([^\n]*" + re.escape(marker) + r"[^\n]*)", assembly))
    if len(matches) != 1:
        raise ValueError(f"T{tokens}: expected exactly one kernel symbol, found {len(matches)}")
    begin = matches[0].start()
    metadata_end = assembly.find(".end_amdhsa_kernel", begin)
    if metadata_end < 0:
        raise ValueError(f"T{tokens}: missing kernel metadata")
    next_begin = assembly.find("; -- Begin function", metadata_end)
    if next_begin < 0:
        next_begin = len(assembly)
    return assembly[begin:metadata_end], assembly[metadata_end:next_begin]


def _field(block: str, name: str, tokens: int) -> int:
    values = re.findall(rf"\.{re.escape(name)}\s+(\d+)", block)
    if len(values) != 1:
        raise ValueError(f"T{tokens}: expected one {name} field, found {len(values)}")
    return int(values[0])


def check(assembly: str) -> dict[int, dict[str, int]]:
    symbols = re.findall(
        r"; -- Begin function [^\n]*a8q4g64_linear_dflash_small_t_kernelILj(\d+)EE[^\n]*",
        assembly,
    )
    if sorted(map(int, symbols)) != sorted(WIDTHS):
        raise ValueError(f"expected exact flattened width kernel set {WIDTHS}, found {symbols}")
    result: dict[int, dict[str, int]] = {}
    for tokens in WIDTHS:
        block, resource_comment = _symbol(assembly, tokens)
        dot8 = block.count("v_dot8_i32_iu4")
        expected_dot8 = 16 * tokens
        if dot8 != expected_dot8:
            raise ValueError(f"T{tokens}: expected {expected_dot8} packed dot8 sites, found {dot8}")
        forbidden = tuple(op for op in ("v_wmma_", "scratch_load", "scratch_store") if op in block)
        if forbidden:
            raise ValueError(f"T{tokens}: forbidden instructions: {', '.join(forbidden)}")
        weight_loads = block.count("global_load_b64")
        scale_loads = block.count("global_load_d16_b16")
        if weight_loads != 4 or scale_loads != tokens + 1:
            raise ValueError(
                f"T{tokens}: expected four packed-W4 pair loads and {tokens + 1} scale loads "
                f"per K64 loop, found {weight_loads} and {scale_loads}"
            )
        vgpr = _field(block, "amdhsa_next_free_vgpr", tokens)
        lds = _field(block, "amdhsa_group_segment_fixed_size", tokens)
        private = _field(block, "amdhsa_private_segment_fixed_size", tokens)
        wave32 = _field(block, "amdhsa_wavefront_size32", tokens)
        occupancies = re.findall(r"Occupancy:\s*(\d+)", resource_comment)
        scratch = re.findall(r"ScratchSize:\s*(\d+)", resource_comment)
        marker = f"a8q4g64_linear_dflash_small_t_kernelILj{tokens}EE"
        yaml = re.findall(
            r"\.max_flat_workgroup_size:\s*(\d+)\s*\n\s*\.name:\s*[^\n]*"
            + re.escape(marker) + r"[^\n]*\n[\s\S]*?\.vgpr_count:\s*(\d+)"
            r"\s*\n\s*\.vgpr_spill_count:\s*(\d+)\s*\n\s*\.wavefront_size:\s*(\d+)",
            assembly,
        )
        if len(occupancies) != 1 or len(scratch) != 1 or len(yaml) != 1:
            raise ValueError(f"T{tokens}: incomplete compiler resource report")
        maximum_workgroup, yaml_vgpr, vgpr_spills, yaml_wave = map(int, yaml[0])
        occupancy = int(occupancies[0])
        scratch_bytes = int(scratch[0])
        if (vgpr != EXACT_VGPR[tokens] or int(yaml_vgpr) != vgpr or lds != 0 or
                private != 0 or scratch_bytes != 0 or int(vgpr_spills) != 0 or
                wave32 != 1 or int(yaml_wave) != 32 or maximum_workgroup != 256 or
                occupancy != EXACT_OCCUPANCY[tokens]):
            raise ValueError(
                f"T{tokens}: resource identity mismatch: vgpr={vgpr}/{yaml_vgpr} "
                f"lds={lds} private={private} scratch={scratch_bytes} spills={vgpr_spills} "
                f"wave={wave32}/{yaml_wave} max_workgroup={maximum_workgroup} "
                f"occupancy={occupancy}"
            )
        result[tokens] = {"dot8": dot8, "weight_loads": weight_loads,
                          "scale_loads": scale_loads, "vgpr": vgpr,
                          "lds": lds, "private": private,
                          "max_workgroup": maximum_workgroup, "occupancy": occupancy}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args()
    result = check(args.assembly.read_text())
    summary = " ".join(
        f"T{t}:dot8={v['dot8']},wloads={v['weight_loads']},vgpr={v['vgpr']}"
        for t, v in result.items()
    )
    print(f"a8q4_dflash_small_t_static=passed {summary} lds=0 private=0 wave32=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
