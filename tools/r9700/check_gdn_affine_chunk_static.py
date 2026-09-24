#!/usr/bin/env python3
"""Fail-closed gfx1201 static gate for the three affine-chunk GDN stages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

STAGES = ("build", "boundaries", "replay")


def one(body: str, pattern: str, label: str) -> int:
    values = re.findall(pattern, body, re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"{label}: expected one value, found {len(values)}")
    return int(values[0])


def check(path: Path) -> dict[str, dict[str, int | str]]:
    text = path.read_text(encoding="utf-8")
    starts = list(re.finditer(r"^.*; -- Begin function (\S+)\s*$", text, re.MULTILINE))
    functions = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        functions.append((start.group(1), text[start.start():end]))
    records = re.split(r"(?=^  - \.args:)", text, flags=re.MULTILINE)
    result = {}
    for stage in STAGES:
        marker = f"affine_chunk_{stage}_qualification_kernel"
        selected = [(symbol, body) for symbol, body in functions if marker in symbol]
        if len(selected) != 1:
            raise ValueError(f"{stage}: expected one exact stage symbol, found {len(selected)}")
        symbol, body = selected[0]
        metadata = [record for record in records if re.search(
            rf"^    \.name:\s+{re.escape(symbol)}$", record, re.MULTILINE)]
        if len(metadata) != 1:
            raise ValueError(f"{stage}: expected one exact metadata record, found {len(metadata)}")
        lds = one(body, r"^\s*\.amdhsa_group_segment_fixed_size\s+(\d+)", "LDS")
        private = one(body, r"^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)", "private")
        vgprs = one(body, r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)", "VGPR")
        scratch = one(body, r"^;\s*ScratchSize:\s*(\d+)", "scratch")
        occupancy = one(body, r"^;\s*Occupancy:\s*(\d+)", "occupancy")
        wave32 = one(body, r"^\s*\.amdhsa_wavefront_size32\s+(\d+)", "wave32")
        wgp = one(body, r"^\s*\.amdhsa_workgroup_processor_mode\s+(\d+)", "WGP")
        maximum_workgroup = one(metadata[0], r"^\s*\.max_flat_workgroup_size:\s*(\d+)",
                                "maximum workgroup")
        forbidden = re.findall(r"^\s*(v_wmma_\S+)", body, re.MULTILINE)
        if forbidden:
            raise ValueError(f"{stage}: FP32 profile contains matrix opcodes: {forbidden}")
        if private != 0 or scratch != 0:
            raise ValueError(f"{stage}: private/scratch must be zero, got {private}/{scratch}")
        if vgprs > 240 or occupancy < 6 or lds > 43520:
            raise ValueError(
                f"{stage}: resources fail vgprs={vgprs}/240 occupancy={occupancy}/6 lds={lds}/43520")
        if wave32 != 1 or wgp != 1 or maximum_workgroup != 256:
            raise ValueError(
                f"{stage}: geometry fails wave32={wave32} WGP={wgp} maxWG={maximum_workgroup}")
        result[stage] = {"symbol": symbol, "vgprs": vgprs, "lds": lds,
                         "occupancy": occupancy, "scratch": scratch}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args()
    try:
        result = check(args.assembly)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for stage, values in result.items():
        print(f"stage={stage} vgprs={values['vgprs']} lds={values['lds']} "
              f"occupancy={values['occupancy']} scratch=0")
    print("qualification_only=true production_dispatch=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
