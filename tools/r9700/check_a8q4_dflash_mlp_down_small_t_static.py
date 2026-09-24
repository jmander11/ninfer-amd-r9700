#!/usr/bin/env python3
"""Fail-closed gfx1201 ISA/resource gate for the DFlash MLP-down T5 screen."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MARKER = "a8q4g64_linear_dflash_mlp_down_t5_kernel"


def _one(pattern: str, text: str, label: str) -> int:
    values = re.findall(pattern, text)
    if len(values) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(values)}")
    return int(values[0])


def check(assembly: str) -> dict[str, int]:
    symbols = re.findall(
        r"; -- Begin function ([^\n]*a8q4g64_linear_dflash_mlp_down_t(\d+)_kernel[^\n]*)",
        assembly,
    )
    if len(symbols) != 1 or symbols[0][1] != "5":
        raise ValueError(f"expected only the T5 MLP-down candidate kernel, found {symbols}")
    begin = assembly.find("; -- Begin function " + symbols[0][0])
    metadata_end = assembly.find(".end_amdhsa_kernel", begin)
    if begin < 0 or metadata_end < 0:
        raise ValueError("T5 candidate kernel or metadata terminator is missing")
    block = assembly[begin:metadata_end]
    next_function = assembly.find("; -- Begin function", metadata_end)
    resource_comment = assembly[metadata_end:next_function if next_function >= 0 else len(assembly)]

    dot8 = block.count("v_dot8_i32_iu4")
    weight_loads = block.count("global_load_b64")
    scale_loads = block.count("global_load_d16_b16")
    forbidden = tuple(
        opcode for opcode in ("v_wmma_", "scratch_load", "scratch_store") if opcode in block
    )
    if dot8 != 80 or weight_loads != 4 or scale_loads != 6 or forbidden:
        raise ValueError(
            "T5 instruction identity mismatch: "
            f"dot8={dot8} weight_loads={weight_loads} scale_loads={scale_loads} "
            f"forbidden={forbidden}"
        )

    vgpr = _one(r"\.amdhsa_next_free_vgpr\s+(\d+)", block, "next-free VGPR")
    sgpr = _one(r"\.amdhsa_next_free_sgpr\s+(\d+)", block, "next-free SGPR")
    lds = _one(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", block, "LDS size")
    private = _one(r"\.amdhsa_private_segment_fixed_size\s+(\d+)", block, "private size")
    wave32 = _one(r"\.amdhsa_wavefront_size32\s+(\d+)", block, "wave32 flag")
    scratch = _one(r"ScratchSize:\s*(\d+)", resource_comment, "scratch size")
    occupancy = _one(r"Occupancy:\s*(\d+)", resource_comment, "occupancy")

    yaml = re.findall(
        r"\.max_flat_workgroup_size:\s*(\d+)\s*\n"
        r"\s*\.name:\s*[^\n]*" + MARKER + r"[^\n]*\n[\s\S]*?"
        r"\.sgpr_count:\s*(\d+)\s*\n"
        r"\s*\.sgpr_spill_count:\s*(\d+)\s*\n[\s\S]*?"
        r"\.vgpr_count:\s*(\d+)\s*\n"
        r"\s*\.vgpr_spill_count:\s*(\d+)\s*\n"
        r"\s*\.wavefront_size:\s*(\d+)",
        assembly,
    )
    if len(yaml) != 1:
        raise ValueError(f"expected one T5 YAML resource record, found {len(yaml)}")
    maximum_workgroup, yaml_sgpr, sgpr_spills, yaml_vgpr, vgpr_spills, yaml_wave = map(
        int, yaml[0]
    )
    expected = (33, 100, 0, 0, 1, 0, 16, 256, 102, 0, 33, 0, 32)
    actual = (
        vgpr, sgpr, lds, private, wave32, scratch, occupancy, maximum_workgroup,
        yaml_sgpr, sgpr_spills, yaml_vgpr, vgpr_spills, yaml_wave,
    )
    if actual != expected:
        raise ValueError(f"T5 resource identity mismatch: expected {expected}, found {actual}")
    return {
        "dot8": dot8,
        "weight_loads": weight_loads,
        "scale_loads": scale_loads,
        "vgpr": vgpr,
        "sgpr": yaml_sgpr,
        "occupancy": occupancy,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args()
    result = check(args.assembly.read_text())
    print(
        "a8q4_dflash_mlp_down_small_t_static=passed "
        f"T5:dot8={result['dot8']},wloads={result['weight_loads']},"
        f"scale_loads={result['scale_loads']},vgpr={result['vgpr']},"
        f"sgpr={result['sgpr']},occupancy={result['occupancy']} "
        "lds=0 private=0 scratch=0 sgpr_spills=0 vgpr_spills=0 wave32=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
