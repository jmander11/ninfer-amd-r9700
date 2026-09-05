#!/usr/bin/env python3
"""Fail-closed ISA/resource check for the disconnected FP8 gate/up challenger."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SYMBOL_FRAGMENT = "fp8_gate_up_m128n128_k32_kernel"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def function_interval(assembly: str) -> str:
    begins = [
        match.start()
        for match in re.finditer(r"^.*-- Begin function .*$", assembly, re.MULTILINE)
        if SYMBOL_FRAGMENT in match.group(0)
    ]
    require(len(begins) == 1, "expected exactly one challenger function")
    end = assembly.find("\n\t.size", begins[0])
    require(end >= 0, "challenger function has no .size boundary")
    return assembly[begins[0] : end]


def integer_metadata(interval: str, directive: str) -> int:
    match = re.search(rf"{re.escape(directive)}\s+(\d+)", interval)
    require(match is not None, f"missing {directive}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=Path)
    arguments = parser.parse_args()
    assembly = arguments.assembly.read_text(encoding="utf-8")
    interval = function_interval(assembly)

    vgpr = integer_metadata(assembly, ".amdhsa_next_free_vgpr")
    lds = integer_metadata(assembly, ".amdhsa_group_segment_fixed_size")
    private = integer_metadata(assembly, ".amdhsa_private_segment_fixed_size")
    require(vgpr <= 88, f"VGPR gate failed: {vgpr} > 88")
    require(lds == 16384, f"LDS gate failed: {lds} != 16384")
    require(private == 0, f"private segment is nonzero: {private}")
    require("; ScratchSize: 0" in assembly, "scratch metadata is not zero")
    require("; Occupancy: 16" in assembly, "occupancy is not 16")

    wmma = "v_wmma_f32_16x16x16_fp8_fp8"
    require(interval.count(wmma) == 8, "expected exactly eight native FP8 WMMA sites")
    require(not re.search(r"v_(?:s?wmma|mfma).*?(?:iu4|iu8|bf16)", interval),
            "unexpected non-FP8 matrix instruction")
    require(interval.count("global_load_b128") == 2,
            "expected initial and successor 128-bit global-load sites")
    require(interval.count("ds_store_b128") == 2,
            "expected one 128-bit LDS publication site per pipeline form")
    require(interval.count("ds_load_2addr_b64") == 4,
            "expected four paired 64-bit LDS operand-load sites")
    require(interval.count("global_load_b32") == 2 and
            interval.count("ds_store_b32") == 2,
            "outer scales are not loaded and published once by role")
    require(interval.count("s_barrier_signal -1") == 3 and
            interval.count("s_barrier_wait -1") == 3,
            "expected initial, loop, and scale-publication barrier generations")
    require("flat_scratch" not in interval and "scratch_store" not in interval,
            "hidden scratch/spill path is present")
    require("global_atomic" not in interval and "buffer_atomic" not in interval,
            "unexpected atomic path")

    loads = [match.start() for match in re.finditer("global_load_b128", interval)]
    wmmas = [match.start() for match in re.finditer(wmma, interval)]
    stores = [match.start() for match in re.finditer("ds_store_b128", interval)]
    waits = [match.start() for match in re.finditer(r"s_wait_loadcnt\s+0x0", interval)]
    require(loads[1] < wmmas[0] < wmmas[-1] < stores[1],
            "successor global load does not span the current WMMA body")
    require(any(wmmas[-1] < wait < stores[1] for wait in waits),
            "successor load is not drained after current WMMAs and before publication")

    outputs: set[tuple[int, int]] = set()
    for wave in range(16):
        token_tile, row_tile = wave // 4, wave % 4
        for lane in range(32):
            axis, lane_group = lane & 15, lane >> 4
            for token_half in range(2):
                for row_half in range(2):
                    for element in range(8):
                        token = token_half * 64 + token_tile * 16 + lane_group * 8 + element
                        row = row_half * 64 + row_tile * 16 + axis
                        require((token, row) not in outputs, "duplicate output-lane mapping")
                        outputs.add((token, row))
    require(len(outputs) == 128 * 128 and
            min(outputs) == (0, 0) and max(outputs) == (127, 127),
            "wave/lane mapping does not cover the exact M128xN128 tile")

    print(
        "fp8 gate/up M128xN128 static PASS: "
        f"vgpr={vgpr} lds={lds} private={private} occupancy=16 "
        "wmma_sites=8 global_b128_sites=2 barriers=3 overlap=present"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
