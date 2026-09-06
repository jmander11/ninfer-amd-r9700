#!/usr/bin/env python3
"""Static qualification gate for the disconnected T5/T6 target gate-up challenger."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SYMBOL = "fp8_gate_up_n34816_k5120_t5t6_wmma_kernel"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def analyze(text: str) -> dict[str, int]:
    starts = [m.start() for m in re.finditer(r"^.*-- Begin function .*$", text, re.MULTILINE)
              if SYMBOL in m.group(0)]
    require(len(starts) == 1, "expected exactly one T5/T6 challenger kernel")
    end = text.find("\n\t.size", starts[0])
    require(end >= 0, "challenger kernel has no size boundary")
    body = text[starts[0]:end]
    next_function = text.find("-- Begin function", end)
    record = text[starts[0]:next_function if next_function >= 0 else len(text)]

    def metadata(name: str) -> int:
        match = re.search(rf"{re.escape(name)}\s+(\d+)", record)
        require(match is not None, f"missing {name}")
        return int(match.group(1))

    vgpr = metadata(".amdhsa_next_free_vgpr")
    sgpr = metadata(".amdhsa_next_free_sgpr")
    lds = metadata(".amdhsa_group_segment_fixed_size")
    private = metadata(".amdhsa_private_segment_fixed_size")
    require(vgpr == 54, f"logical VGPR identity changed: {vgpr} != 54")
    require(sgpr == 19, f"logical SGPR identity changed: {sgpr} != 19")
    require(lds == 0 and private == 0, "candidate must use zero LDS/private storage")
    require("\t\t.amdhsa_wavefront_size32 1" in record,
            "candidate must retain wave32 execution")
    require("; ScratchSize: 0" in record and "; Occupancy: 16" in record and
            "; NumVgprs: 54" in record and "; TotalNumSgprs: 21" in record,
            "candidate emitted resource profile changed")
    require(body.count("v_wmma_f32_16x16x16_fp8_fp8") == 4,
            "candidate must have four independent native FP8 WMMA sites")
    require(not re.search(r"v_(?:s?wmma|mfma).*?(?:iu4|iu8|bf16)", body),
            "candidate contains an unexpected non-FP8 matrix instruction")
    require("flat_scratch" not in body and "scratch_store" not in body,
            "candidate contains a hidden scratch path")
    require("s_barrier" not in body and "ds_read" not in body and "ds_write" not in body,
            "candidate unexpectedly uses LDS/barriers")
    require(body.count("global_load_b64") == 5,
            "candidate must retain exactly five FP8 vector-load sites")

    metadata_name = re.search(rf"^    \.name:\s+.*{SYMBOL}.*$", text, re.MULTILINE)
    require(metadata_name is not None, "candidate code-object metadata entry is missing")
    metadata_begin = text.rfind("  - .args:", 0, metadata_name.start())
    metadata_end = text.find("\n  - .args:", metadata_name.end())
    require(metadata_begin >= 0 and metadata_end >= 0,
            "candidate code-object metadata boundaries are missing")
    code_object = text[metadata_begin:metadata_end]
    for exact in ("    .max_flat_workgroup_size: 128",
                  "    .sgpr_count:     21",
                  "    .sgpr_spill_count: 0",
                  "    .vgpr_count:     54",
                  "    .vgpr_spill_count: 0",
                  "    .wavefront_size: 32"):
        require(exact in code_object, f"candidate metadata changed: missing {exact.strip()}")
    for tokens in (5, 6):
        outputs: set[tuple[int, int]] = set()
        for block in range(34816 // 256):
            for wave in range(4):
                for lane in range(32):
                    axis, lane_group = lane & 15, lane >> 4
                    for fragment in range(4):
                        row = block * 256 + wave * 64 + fragment * 16 + axis
                        for element in range(8):
                            token = lane_group * 8 + element
                            if token < tokens:
                                require((token, row) not in outputs,
                                        "duplicate output-lane mapping")
                                outputs.add((token, row))
        require(len(outputs) == tokens * 34816 and min(outputs) == (0, 0) and
                max(outputs) == (tokens - 1, 34815),
                f"T{tokens} output-lane mapping is incomplete")
    return {"logical_vgpr": vgpr, "logical_sgpr": sgpr, "lds": lds,
            "private": private, "wmma_sites": 4, "wavefront": 32,
            "max_workgroup": 128, "vgpr_spills": 0, "sgpr_spills": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=Path)
    result = analyze(parser.parse_args().assembly.read_text(encoding="utf-8"))
    print("FP8 target gate-up T5/T6 static PASS: " +
          " ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
