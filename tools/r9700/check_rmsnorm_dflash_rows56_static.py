#!/usr/bin/env python3
"""Exact gfx1201 static gate for the reused K5120 rows5/6 CTA challenger."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .check_rmsnorm_decode_static import check
except ImportError:
    from check_rmsnorm_decode_static import check

EXPECTED = {
    "vgprs": 17,
    "lds": 32,
    "occupancy": 16,
    "maximum_workgroup": 256,
    "global_loads": 15,
    "shuffle_ops": 8,
    "barrier_pairs": 2,
    "global_invalidations": 2,
}


def exact(path: Path) -> dict:
    result = check(path)
    for key, expected in EXPECTED.items():
        if result.get(key) != expected:
            raise ValueError(f"exact {key} differs: {result.get(key)} != {expected}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args()
    try:
        result = exact(args.assembly)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    print("qualification_only=true features=5120 rows=5,6 production_selector_unchanged=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
