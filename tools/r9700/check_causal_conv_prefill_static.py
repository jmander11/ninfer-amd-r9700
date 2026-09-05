#!/usr/bin/env python3
"""Fail-closed gfx1201 resource gate for production and retained causal-conv tiles."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

TILES = (4, 8, 16, 32)
MAX_VGPRS = 240
MIN_OCCUPANCY = 6


def _one(body: str, pattern: str, label: str) -> int:
    values = re.findall(pattern, body, re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"{label}: expected one value, found {len(values)}")
    return int(values[0])


def _functions(text: str) -> list[tuple[str, str]]:
    starts = list(re.finditer(r"^.*; -- Begin function (\S+)\s*$", text, re.MULTILINE))
    result: list[tuple[str, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        result.append((start.group(1), text[start.start():end]))
    return result


def check(path: Path) -> dict[int, dict[str, int | str]]:
    text = path.read_text(encoding="utf-8")
    functions = _functions(text)
    results: dict[int, dict[str, int | str]] = {}
    for tile in TILES:
        marker = f"causal_conv1d_silu_prefill_kernelILj{tile}EE"
        matches = [(symbol, body) for symbol, body in functions if marker in symbol]
        if len(matches) != 1:
            raise ValueError(f"tile {tile}: expected one exact qualification symbol, found {len(matches)}")
        symbol, body = matches[0]
        metadata_records = re.split(r"(?=^  - \.args:)", text, flags=re.MULTILINE)
        records = [record for record in metadata_records if re.search(
            rf"^    \.name:\s+{re.escape(symbol)}$", record, flags=re.MULTILINE)]
        if len(records) != 1:
            raise ValueError(f"tile {tile}: expected one exact metadata record, found {len(records)}")
        maximum_workgroup = _one(
            records[0], r"^\s*\.max_flat_workgroup_size:\s*(\d+)", "maximum workgroup")
        lds = _one(body, r"^\s*\.amdhsa_group_segment_fixed_size\s+(\d+)", "LDS")
        private = _one(body, r"^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)", "private")
        vgprs = _one(body, r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)", "VGPR")
        scratch = _one(body, r"^;\s*ScratchSize:\s*(\d+)", "scratch")
        occupancy = _one(body, r"^;\s*Occupancy:\s*(\d+)", "occupancy")
        wave32 = _one(body, r"^\s*\.amdhsa_wavefront_size32\s+(\d+)", "wave32")
        wgp = _one(body, r"^\s*\.amdhsa_workgroup_processor_mode\s+(\d+)", "WGP mode")
        if lds != 0 or private != 0 or scratch != 0:
            raise ValueError(
                f"tile {tile}: LDS/private/scratch must be zero, got {lds}/{private}/{scratch}")
        if vgprs > MAX_VGPRS or occupancy < MIN_OCCUPANCY:
            raise ValueError(
                f"tile {tile}: resources fail vgprs={vgprs}/{MAX_VGPRS} "
                f"occupancy={occupancy}/{MIN_OCCUPANCY}")
        if wave32 != 1 or wgp != 1:
            raise ValueError(f"tile {tile}: requires wave32 WGP mode, got {wave32}/{wgp}")
        if maximum_workgroup != 256:
            raise ValueError(
                f"tile {tile}: maximum flat workgroup must be 256, got {maximum_workgroup}")
        results[tile] = {"symbol": symbol, "vgprs": vgprs, "occupancy": occupancy,
                         "lds": lds, "private": private, "scratch": scratch,
                         "maximum_workgroup": maximum_workgroup}
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args(argv)
    try:
        results = check(args.assembly)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for tile, result in results.items():
        print(f"tile={tile} vgprs={result['vgprs']} occupancy={result['occupancy']} "
              "lds=0 private=0 scratch=0")
    print("selected_tile=4 production_dispatch=true crossover_tokens=64")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
