#!/usr/bin/env python3
"""Fail-closed gfx1201 static gate for the K5120/token8 RMSNorm challenger."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence


def one(body: str, pattern: str, label: str) -> int:
    matches = re.findall(pattern, body, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"{label}: expected one value, found {len(matches)}")
    return int(matches[0])


def check(path: Path) -> dict[str, int | str]:
    text = path.read_text(encoding="utf-8")
    starts = list(re.finditer(r"^.*; -- Begin function (\S+)\s*$", text, re.MULTILINE))
    functions: list[tuple[str, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        functions.append((start.group(1), text[start.start():end]))
    marker = "rmsnorm_k5120_token8_kernel"
    selected = [(symbol, body) for symbol, body in functions if marker in symbol]
    if len(selected) != 1:
        raise ValueError(f"expected one exact qualification symbol, found {len(selected)}")
    symbol, body = selected[0]
    records = [record for record in re.split(r"(?=^  - \.args:)", text, flags=re.MULTILINE)
               if re.search(rf"^    \.name:\s+{re.escape(symbol)}$", record,
                            flags=re.MULTILINE)]
    if len(records) != 1:
        raise ValueError(f"expected one exact metadata record, found {len(records)}")
    maximum_workgroup = one(records[0], r"^\s*\.max_flat_workgroup_size:\s*(\d+)",
                            "maximum workgroup")
    lds = one(body, r"^\s*\.amdhsa_group_segment_fixed_size\s+(\d+)", "LDS")
    private = one(body, r"^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)", "private")
    vgprs = one(body, r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)", "VGPR")
    scratch = one(body, r"^;\s*ScratchSize:\s*(\d+)", "scratch")
    occupancy = one(body, r"^;\s*Occupancy:\s*(\d+)", "occupancy")
    wave32 = one(body, r"^\s*\.amdhsa_wavefront_size32\s+(\d+)", "wave32")
    wgp = one(body, r"^\s*\.amdhsa_workgroup_processor_mode\s+(\d+)", "WGP")
    vector_load_pattern = r"^\s*(?:global_load_dwordx4|global_load_b128)(?:\s|$)"
    if len(re.findall(vector_load_pattern, body, re.MULTILINE)) < 1:
        raise ValueError("qualification kernel lacks a 128-bit vector input load")
    forbidden = re.findall(r"^\s*(?:global|buffer)_atomic_\S+", body, re.MULTILINE)
    if forbidden:
        raise ValueError(f"qualification kernel contains atomics: {forbidden}")
    if lds != 0 or private != 0 or scratch != 0:
        raise ValueError(f"LDS/private/scratch must be zero, got {lds}/{private}/{scratch}")
    if vgprs > 120 or occupancy < 12:
        raise ValueError(f"resources fail vgprs={vgprs}/120 occupancy={occupancy}/12")
    if wave32 != 1 or wgp != 1 or maximum_workgroup != 256:
        raise ValueError(
            f"execution geometry fails wave32={wave32} WGP={wgp} maxWG={maximum_workgroup}")
    return {"symbol": symbol, "vgprs": vgprs, "occupancy": occupancy,
            "maximum_workgroup": maximum_workgroup, "vector_loads": len(re.findall(
                vector_load_pattern, body, re.MULTILINE))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args(argv)
    try:
        result = check(args.assembly)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    print("production=true selected_route=k5120_token8 rows_minimum=128")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
