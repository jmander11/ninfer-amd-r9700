#!/usr/bin/env python3
"""Static admission checks for the compile-selected GDN verify normalization association."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def fail(message: str) -> None:
    raise RuntimeError(message)


def section(text: str, symbol_fragment: str) -> str:
    pattern = re.compile(
        rf"(?ms)^\s*\.section\s+\.text\.[^\n]*{re.escape(symbol_fragment)}[^\n]*\n"
        rf".*?(?=^\s*\.section\s+\.text\.|\Z)"
    )
    match = pattern.search(text)
    if match is None:
        fail(f"missing assembly section: {symbol_fragment}")
    return match.group(0)


def metadata(text: str, symbol_fragment: str) -> tuple[int, int, int, int]:
    name = re.search(rf"(?m)^\s*\.name:\s+.*{re.escape(symbol_fragment)}.*$", text)
    if name is None:
        fail(f"missing metadata name: {symbol_fragment}")
    groups = list(re.finditer(r"(?m)^\s*\.group_segment_fixed_size:\s+(\d+)\s*$",
                              text[:name.start()]))
    if not groups:
        fail(f"missing group_segment_fixed_size: {symbol_fragment}")
    values = [int(groups[-1].group(1))]
    block = text[name.end():]
    next_name = re.search(r"(?m)^\s*\.name:", block)
    if next_name is not None:
        block = block[:next_name.start()]
    for field in ("private_segment_fixed_size", "sgpr_count", "vgpr_count"):
        match = re.search(rf"(?m)^\s*\.{field}:\s+(\d+)\s*$", block)
        if match is None:
            fail(f"missing {field}: {symbol_fragment}")
        values.append(int(match.group(1)))
    return tuple(values)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("off", type=Path)
    parser.add_argument("on", type=Path)
    args = parser.parse_args()
    off = args.off.read_text(encoding="utf-8")
    on = args.on.read_text(encoding="utf-8")

    ordinary_true = "ordinary_kernelILb1EE"
    ordinary_false = "ordinary_kernelILb0EE"
    snapshot_false = "snapshot_kernelILb0EE"
    for symbol in (ordinary_true, ordinary_false, snapshot_false):
        if section(off, symbol) != section(on, symbol):
            fail(f"selector changed an excluded kernel: {symbol}")

    expected = {
        ("off", "snapshot_kernelILb1EE"): (1536, 0, 44, 58),
        ("on", "snapshot_kernelILb1EE"): (1032, 0, 48, 60),
        ("off", "record_kernelILb1EE"): (1536, 0, 56, 66),
        ("on", "record_kernelILb1EE"): (1032, 0, 52, 67),
        ("off", "record_kernelILb0EE"): (1536, 0, 52, 54),
        ("on", "record_kernelILb0EE"): (1032, 0, 52, 60),
    }
    for (arm, symbol), wanted in expected.items():
        actual = metadata(off if arm == "off" else on, symbol)
        if actual != wanted:
            fail(f"{arm} {symbol} resources {actual} != {wanted}")
    if on.count("ds_bpermute_b32") <= off.count("ds_bpermute_b32"):
        fail("candidate does not add the canonical wave32 permutation reduction")
    if on.count("s_barrier") >= off.count("s_barrier"):
        fail("candidate does not remove shared-tree synchronization")
    print("gdn verify wave-Q/K static checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
