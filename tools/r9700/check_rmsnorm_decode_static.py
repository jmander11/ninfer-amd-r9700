#!/usr/bin/env python3
"""Fail-closed gfx1201 static gate for the K5120 rows1..4 RMSNorm candidate."""

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
    marker = "rmsnorm_k5120_rows4_cta_kernel"
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
    global_loads = len(re.findall(r"^\s*(?:global|buffer)_load_(?:u16|ushort|dword)",
                                  body, re.MULTILINE))
    shuffle_ops = len(re.findall(r"^\s*(?:ds_bpermute_b32|v_permlane\S*|v_mov_b32_dpp)",
                                 body, re.MULTILINE))
    instructions = [line.strip() for line in body.splitlines()
                    if line.strip() and not line.lstrip().startswith((";", "."))]
    positions = lambda prefix: [index for index, line in enumerate(instructions)
                                if line.startswith(prefix)]
    stores = positions("ds_store_b32")
    loads = positions("ds_load_b32")
    signals = positions("s_barrier_signal -1")
    waits = positions("s_barrier_wait -1")
    invalidations = positions("global_inv scope:SCOPE_SE")
    if global_loads < 1:
        raise ValueError("candidate lacks a distributed vector-addressed global input load")
    if shuffle_ops < 1:
        raise ValueError("candidate lacks a wave32 shuffle reduction")
    forbidden = re.findall(
        r"^\s*(?:s_load_u16|s_fmac_f32|s_barrier(?:\s|$)|v_wmma_\S*|"
        r"(?:global|buffer|flat)_atomic_\S*|scratch_(?:load|store)\S*)",
        body, re.MULTILINE)
    if forbidden:
        raise ValueError(f"candidate contains forbidden instructions: {forbidden}")
    if lds != 32 or private != 0 or scratch != 0:
        raise ValueError(f"LDS/private/scratch fail {lds}/32 {private}/0 {scratch}/0")
    if vgprs > 48 or occupancy != 16:
        raise ValueError(f"resources fail vgprs={vgprs}/48 occupancy={occupancy}/16")
    if wave32 != 1 or wgp != 1 or maximum_workgroup != 256:
        raise ValueError(
            f"execution geometry fails wave32={wave32} WGP={wgp} maxWG={maximum_workgroup}")
    if not (len(stores) == len(loads) == len(signals) == len(waits) ==
            len(invalidations) == 2):
        raise ValueError(
            "barrier/LDS protocol requires exactly two store/signal/wait/global_inv/load groups")
    if not (stores[0] < signals[0] < waits[0] < invalidations[0] < loads[0] <
            stores[1] < signals[1] < waits[1] < invalidations[1] < loads[1]):
        raise ValueError("barrier/LDS protocol ordering drift")
    return {"symbol": symbol, "vgprs": vgprs, "lds": lds, "occupancy": occupancy,
            "maximum_workgroup": maximum_workgroup, "global_loads": global_loads,
            "shuffle_ops": shuffle_ops, "barrier_pairs": len(signals),
            "global_invalidations": len(invalidations)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args(argv)
    try:
        result = check(args.assembly)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    print("qualification_only=true features=5120 rows=1..4 production_cutover=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
