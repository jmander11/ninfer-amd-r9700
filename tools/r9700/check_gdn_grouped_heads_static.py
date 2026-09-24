#!/usr/bin/env python3
"""Fail-closed ISA/resource gate for the grouped-head GDN challenger."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SYMBOL = "_ZN6ninfer3ops5r970014gdn_recurrence12_GLOBAL__N_131grouped_heads_normalized_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_if"


def one(text: str, pattern: str, label: str) -> int:
    values = re.findall(pattern, text, re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"selected symbol must contain one {label}")
    return int(values[0])


def check(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    begins = list(re.finditer(r"-- Begin function\s+(\S+)", text))
    selected = [i for i, match in enumerate(begins) if match.group(1) == SYMBOL]
    if len(selected) != 1:
        raise ValueError(f"exact grouped-head symbol count is {len(selected)}, expected one")
    index = selected[0]
    end = begins[index + 1].start() if index + 1 < len(begins) else len(text)
    body = text[begins[index].start():end]
    lds = one(body, r"^\s*\.amdhsa_group_segment_fixed_size\s+(\d+)", "LDS size")
    private = one(body, r"^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)", "private size")
    vgpr = one(body, r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)", "VGPR count")
    scratch = one(body, r"^;\s*ScratchSize:\s*(\d+)", "scratch size")
    occupancy = one(body, r"^;\s*Occupancy:\s*(\d+)", "occupancy")
    maximum = one(body, r"^\s*\.max_flat_workgroup_size:\s*(\d+)", "workgroup size")
    signals = len(re.findall(r"^\s*s_barrier_signal\s+-1", body, re.MULTILINE))
    waits = len(re.findall(r"^\s*s_barrier_wait\s+-1", body, re.MULTILINE))
    if (lds, private, scratch, maximum) != (1056, 0, 0, 256):
        raise ValueError("grouped-head LDS/private/scratch/workgroup resources differ")
    if vgpr > 128 or occupancy < 8:
        raise ValueError("grouped-head VGPR/occupancy gate failed")
    if signals != 4 or waits != 4 or re.search(r"^\s*s_barrier(?:\s|$)", body, re.MULTILINE):
        raise ValueError("grouped-head kernel must retain four workgroup barrier pairs")
    if re.search(r"\b(?:v_wmma|v_mfma|global_inv)\b", body):
        raise ValueError("grouped-head symbol contains a forbidden matrix/cache instruction")
    return {"lds": lds, "vgpr": vgpr, "occupancy": occupancy,
            "scratch": scratch, "barriers": signals}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_gdn_grouped_heads_static.py ASSEMBLY")
    try:
        print(check(Path(sys.argv[1])))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
