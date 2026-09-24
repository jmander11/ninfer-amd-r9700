#!/usr/bin/env python3
"""Fail-closed gfx1201 static gate for the production K6144/token8 gated RMSNorm."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence


def _one(body: str, pattern: str, label: str) -> int:
    matches = re.findall(pattern, body, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"{label}: expected one value, found {len(matches)}")
    return int(matches[0])


def check(path: Path) -> dict[str, int | str | bool]:
    text = path.read_text(encoding="utf-8")
    starts = list(re.finditer(r"^.*; -- Begin function (\S+)\s*$", text, re.MULTILINE))
    functions: list[tuple[str, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        functions.append((start.group(1), text[start.start():end]))
    marker = "gated_rmsnorm_k6144_token8_kernel"
    selected = [(symbol, body) for symbol, body in functions if marker in symbol]
    if len(selected) != 1:
        raise ValueError(f"expected one exact qualification symbol, found {len(selected)}")
    symbol, body = selected[0]
    records = [record for record in re.split(r"(?=^  - \.args:)", text, flags=re.MULTILINE)
               if re.search(rf"^    \.name:\s+{re.escape(symbol)}$", record,
                            flags=re.MULTILINE)]
    if len(records) != 1:
        raise ValueError(f"expected one exact metadata record, found {len(records)}")
    maximum_workgroup = _one(records[0], r"^\s*\.max_flat_workgroup_size:\s*(\d+)",
                             "maximum workgroup")
    lds = _one(body, r"^\s*\.amdhsa_group_segment_fixed_size\s+(\d+)", "LDS")
    private = _one(body, r"^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)", "private")
    vgprs = _one(body, r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)", "VGPR")
    scratch = _one(body, r"^;\s*ScratchSize:\s*(\d+)", "scratch")
    occupancy = _one(body, r"^;\s*Occupancy:\s*(\d+)", "occupancy")
    wave32 = _one(body, r"^\s*\.amdhsa_wavefront_size32\s+(\d+)", "wave32")
    wgp = _one(body, r"^\s*\.amdhsa_workgroup_processor_mode\s+(\d+)", "WGP")
    vector_load_pattern = r"^\s*(?:global_load_dwordx4|global_load_b128)(?:\s|$)"
    vector_loads = len(re.findall(vector_load_pattern, body, re.MULTILINE))
    if vector_loads < 1:
        raise ValueError("qualification kernel lacks a 128-bit vector reduction load")
    if re.findall(r"^\s*(?:global|buffer)_atomic_\S+", body, re.MULTILINE):
        raise ValueError("qualification kernel must not contain atomics")
    if re.search(r"^\s*s_barrier(?:\s|$)", body, re.MULTILINE):
        raise ValueError("independent wave/token kernel must not contain a workgroup barrier")
    if lds != 0 or private != 0 or scratch != 0:
        raise ValueError(f"LDS/private/scratch must be zero, got {lds}/{private}/{scratch}")
    if vgprs > 120 or occupancy < 12:
        raise ValueError(f"resources fail vgprs={vgprs}/120 occupancy={occupancy}/12")
    if wave32 != 1 or wgp != 1 or maximum_workgroup != 256:
        raise ValueError(
            f"execution geometry fails wave32={wave32} WGP={wgp} maxWG={maximum_workgroup}")
    return {"symbol": symbol, "vgprs": vgprs, "occupancy": occupancy,
            "maximum_workgroup": maximum_workgroup, "vector_loads": vector_loads,
            "production": True, "rows_minimum": 64}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args(argv)
    try:
        result = check(args.assembly)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    print("production=true selected_route=k6144_token8 rows_minimum=64")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
