#!/usr/bin/env python3
"""Fail-closed gfx1201 static gate for production K128/rows8 gated RMSNorm."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence


def _one(body: str, pattern: str, label: str) -> int:
    values = re.findall(pattern, body, re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"{label}: expected one value, found {len(values)}")
    return int(values[0])


def check(path: Path) -> dict[str, int | str]:
    text = path.read_text(encoding="utf-8")
    starts = list(re.finditer(r"^.*; -- Begin function (\S+)\s*$", text, re.MULTILINE))
    functions = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        functions.append((start.group(1), text[start.start():end]))
    marker = "gated_rmsnorm_k128_rows8_kernel"
    selected = [(symbol, body) for symbol, body in functions if marker in symbol]
    if len(selected) != 1:
        raise ValueError(f"expected one exact production symbol, found {len(selected)}")
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
    vector_loads = len(re.findall(
        r"^\s*(?:global_load_dwordx4|global_load_b128)(?:\s|$)", body, re.MULTILINE))
    if vector_loads < 1:
        raise ValueError("production kernel lacks a 128-bit reduction load")
    if not re.search(r"^\s*v_(?:s_)?exp_f32(?:_e32|_e64)?(?:\s|$)", body, re.MULTILINE):
        raise ValueError("production kernel lacks native FP32 exponential")
    if re.search(r"^\s*(?:s_barrier|v_wmma_|global_atomic_|buffer_atomic_|scratch_load|scratch_store)", body,
                 re.MULTILINE):
        raise ValueError("production kernel contains forbidden barrier/WMMA/atomic opcode")
    if lds != 0 or private != 0 or scratch != 0:
        raise ValueError(f"LDS/private/scratch must be zero, got {lds}/{private}/{scratch}")
    if vgprs > 24 or occupancy < 16:
        raise ValueError(f"resources fail vgprs={vgprs}/24 occupancy={occupancy}/16")
    if wave32 != 1 or wgp != 1 or maximum_workgroup != 256:
        raise ValueError(
            f"execution geometry fails wave32={wave32} WGP={wgp} maxWG={maximum_workgroup}")
    return {"symbol": symbol, "vgprs": vgprs, "occupancy": occupancy,
            "maximum_workgroup": maximum_workgroup, "vector_loads": vector_loads}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args(argv)
    try:
        result = check(args.assembly)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    print("production=true features=128 rows_per_cta=8 rows=49152,98304,196608,393216")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
