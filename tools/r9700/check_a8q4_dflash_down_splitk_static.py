#!/usr/bin/env python3
"""Fail-closed gfx1201 ISA/resource gate for DFlash down split-K."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def one(body: str, pattern: str, label: str) -> int:
    values = re.findall(pattern, body, re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"expected one {label}, found {len(values)}")
    return int(values[0])


def kernel_blocks(assembly: str, fragment: str) -> dict[int, str]:
    matches = list(re.finditer(
        rf"; -- Begin function ([^\n]*{fragment}ILj(2|4|8)EE[^\n]*)", assembly
    ))
    if len(matches) != 3:
        raise ValueError(f"expected three {fragment} instantiations, found {len(matches)}")
    result: dict[int, str] = {}
    for match in matches:
        end = assembly.find("; -- Begin function", match.end())
        result[int(match.group(2))] = assembly[match.start(): end if end >= 0 else len(assembly)]
    return result


def resources(body: str, *, vgpr_max: int) -> dict[str, int]:
    values = {
        "vgpr": one(body, r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)", "VGPR"),
        "lds": one(body, r"^\s*\.amdhsa_group_segment_fixed_size\s+(\d+)", "LDS"),
        "private": one(body, r"^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)", "private"),
        "wave32": one(body, r"^\s*\.amdhsa_wavefront_size32\s+(\d+)", "wave32"),
        "scratch": one(body, r"^;\s*ScratchSize:\s*(\d+)", "scratch"),
        "occupancy": one(body, r"^;\s*Occupancy:\s*(\d+)", "occupancy"),
    }
    if (values["vgpr"] > vgpr_max or values["lds"] != 0 or values["private"] != 0 or
            values["wave32"] != 1 or values["scratch"] != 0 or
            values["occupancy"] != 16):
        raise ValueError(f"resource gate failed: {values}, vgpr_max={vgpr_max}")
    if "scratch_load" in body or "scratch_store" in body:
        raise ValueError("scratch instruction present")
    return values


def require_yaml(assembly: str, fragment: str, factor: int, vgpr: int) -> None:
    records = re.findall(
        rf"(?s)\.max_flat_workgroup_size:\s*(\d+)\s*\n"
        rf"\s*\.name:\s*[^\n]*{fragment}ILj{factor}EE[^\n]*\n"
        rf".*?\.private_segment_fixed_size:\s*(\d+)\s*\n"
        rf"\s*\.sgpr_count:\s*(\d+)\s*\n"
        rf"\s*\.sgpr_spill_count:\s*(\d+)\s*\n"
        rf".*?\.vgpr_count:\s*(\d+)\s*\n"
        rf"\s*\.vgpr_spill_count:\s*(\d+)\s*\n"
        rf"\s*\.wavefront_size:\s*(\d+)",
        assembly,
    )
    if len(records) != 1:
        raise ValueError(f"expected one YAML record for {fragment} S={factor}")
    workgroup, private, _sgpr, sgpr_spills, yaml_vgpr, vgpr_spills, wave = map(
        int, records[0]
    )
    if (workgroup, private, sgpr_spills, yaml_vgpr, vgpr_spills, wave) != (
            32, 0, 0, vgpr, 0, 32):
        raise ValueError(f"YAML resource gate failed for {fragment} S={factor}: {records[0]}")


def check(assembly: str) -> dict[str, dict[int, dict[str, int]]]:
    split = kernel_blocks(assembly, "a8q4g64_linear_down_splitk_kernel")
    reduce = kernel_blocks(assembly, "a8q4g64_linear_down_splitk_reduce_kernel")
    result = {"split": {}, "reduce": {}}
    for factor, body in split.items():
        if body.count("v_wmma_i32_16x16x32_iu4") != 4:
            raise ValueError(f"S={factor} split kernel does not use the expected IU4 WMMA body")
        result["split"][factor] = resources(body, vgpr_max=64)
        require_yaml(assembly, "a8q4g64_linear_down_splitk_kernel", factor,
                     result["split"][factor]["vgpr"])
    for factor, body in reduce.items():
        if "v_wmma_" in body:
            raise ValueError(f"S={factor} reduction kernel unexpectedly contains WMMA")
        result["reduce"][factor] = resources(body, vgpr_max=32)
        require_yaml(assembly, "a8q4g64_linear_down_splitk_reduce_kernel", factor,
                     result["reduce"][factor]["vgpr"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args()
    result = check(args.assembly.read_text())
    fields = []
    for kind in ("split", "reduce"):
        for factor in (2, 4, 8):
            values = result[kind][factor]
            fields.append(
                f"{kind}_S{factor}:vgpr={values['vgpr']},occupancy={values['occupancy']}"
            )
    print("a8q4_dflash_down_splitk_static=passed " + " ".join(fields) +
          " lds=0 private=0 scratch=0 spills=0 wave32=1 iu4=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
