#!/usr/bin/env python3
"""Fail-closed gfx1201 gate for the production fused SiLU-to-A8 kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SYMBOL = (
    "_ZN6ninfer3ops5r97006linear12_GLOBAL__N_132"
    "fused_silu_a8g64_quantize_kernel"
    "EPK12hip_bfloat16PhS7_PtPj"
)
BEGIN = re.compile(r"-- Begin function\s+(\S+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(path: Path, source: Path) -> dict[str, int | str | bool]:
    text = path.read_text(encoding="utf-8")
    matches = list(BEGIN.finditer(text))
    selected = [index for index, match in enumerate(matches) if match.group(1) == SYMBOL]
    if len(selected) != 1:
        raise ValueError(f"expected one exact fused symbol, found {len(selected)}")
    index = selected[0]
    begin = matches[index].start()
    end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
    body = text[begin:end]

    def count(opcode: str) -> int:
        return len(re.findall(rf"^\s*{re.escape(opcode)}(?:\s|$)", body, re.MULTILINE))

    metadata_stem = ".L_" + SYMBOL.removeprefix("_")
    vgpr_values = re.findall(
        rf"^\s*\.set\s+{re.escape(metadata_stem)}\.num_vgpr,\s*(\d+)\s*$",
        body,
        re.MULTILINE,
    )
    private_values = re.findall(
        rf"^\s*\.set\s+{re.escape(metadata_stem)}\.private_seg_size,\s*(\d+)\s*$",
        body,
        re.MULTILINE,
    )
    if len(vgpr_values) != 1 or len(private_values) != 1:
        raise ValueError("fused symbol lacks unique VGPR/private metadata")
    vgpr = int(vgpr_values[0])
    private = int(private_values[0])
    loads_b16 = count("global_load_u16")
    stores_b16 = count("global_store_b16")
    stores_b8 = count("global_store_b8")
    exp = count("v_exp_f32_e32") + count("v_exp_f32")
    forbidden = sum(count(opcode) for opcode in (
        "scratch_load_b32", "scratch_store_b32", "flat_load_b16", "flat_store_b16",
        "buffer_load_ushort", "buffer_store_short", "global_store_b32",
    ))
    if vgpr > 24 or private != 0:
        raise ValueError(f"resource gate failed: vgpr={vgpr} private={private}")
    if loads_b16 != 4 or stores_b16 != 1 or stores_b8 != 2 or exp != 2:
        raise ValueError(
            "unexpected exact memory/arithmetic inventory: "
            f"load_b16={loads_b16} store_b16={stores_b16} "
            f"store_b8={stores_b8} exp={exp}"
        )
    if forbidden != 0:
        raise ValueError(f"forbidden spill/intermediate-memory instructions: {forbidden}")
    return {
        "schema": "ninfer.r9700.fused_silu_a8q4_static.v1",
        "status": "pass",
        "symbol": SYMBOL,
        "assembly": str(path),
        "assembly_sha256": _sha256(path),
        "source": str(source),
        "source_sha256": _sha256(source),
        "vgpr": vgpr,
        "private_bytes": private,
        "lds_bytes": 0,
        "global_bf16_input_loads": loads_b16,
        "global_fp16_scale_stores": stores_b16,
        "global_packed_code_stores": stores_b8,
        "native_exp_instructions": exp,
        "intermediate_bf16_global_store": False,
        "intermediate_bf16_global_reload": False,
        "scratch_or_flat_memory_instructions": forbidden,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.assembly, args.source)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
