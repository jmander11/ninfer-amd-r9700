#!/usr/bin/env python3
"""Fail-closed ISA/resource gate for the fused production dense-prefill attention kernel."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.r9700.check_prefill_cta_static import _function, _one_integer

# One loop body per 32-key block: S^T = K Q^T as 2 key tiles x 16 feature tiles of BF16 WMMA,
# then O += P V as 16 feature tiles x 2 key tiles of FP16 WMMA.
PROFILE = {"bf16_wmma": 32, "f16_wmma": 32, "lds": 37536, "vgpr": 256, "occupancy": 5,
           "workgroup": 384, "barrier_pairs": 3}


def _kernel_metadata_record(text: str, symbol: str) -> str:
    records = re.split(r"(?=^  - \.args:)", text, flags=re.MULTILINE)
    selected = [record for record in records if re.search(
        rf"^    \.name:\s+{re.escape(symbol)}$", record, flags=re.MULTILINE)]
    if len(selected) != 1:
        raise ValueError(
            "metadata: exact selected symbol must have one AMDGPU metadata record, "
            f"found {len(selected)}")
    return selected[0]


def _count(body: str, opcode: str) -> int:
    return len(re.findall(rf"^\s*{opcode}(?:_e32|_e64)?(?:\s|$)", body, re.MULTILINE))


def check(*, assembly: Path, metadata: Path, symbol: str, value_group: int) -> dict[str, int | str]:
    if value_group not in (16, 32):
        raise ValueError("dense attention requires G16 or G32")
    if f"20dense_prefill_kernelILj{value_group}EE" not in symbol:
        raise ValueError("selected symbol is not the exact production fused dense-prefill "
                         f"G{value_group} specialization")
    assembly_body = _function(assembly.read_text(encoding="utf-8"), symbol, "assembly")
    metadata_text = metadata.read_text(encoding="utf-8")
    metadata_body = _function(metadata_text, symbol, "metadata")
    kernel_record = _kernel_metadata_record(metadata_text, symbol)
    bf16 = _count(assembly_body, "v_wmma_f32_16x16x16_bf16")
    f16 = _count(assembly_body, "v_wmma_f32_16x16x16_f16")
    if bf16 != PROFILE["bf16_wmma"] or f16 != PROFILE["f16_wmma"]:
        raise ValueError(f"WMMA counts bf16={bf16} f16={f16}, expected "
                         f"{PROFILE['bf16_wmma']}/{PROFILE['f16_wmma']}")
    unexpected = [op for op in re.findall(r"^\s*(v_wmma_\S+)", assembly_body, re.MULTILINE)
                  if op not in ("v_wmma_f32_16x16x16_bf16", "v_wmma_f32_16x16x16_f16")]
    if unexpected:
        raise ValueError(f"fused dense prefill contains unexpected matrix opcodes: {unexpected}")
    if _count(assembly_body, "v_exp_f32") == 0:
        raise ValueError("fused dense prefill is missing native FP32 exponential")
    barrier_pairs = len(re.findall(r"^\s*s_barrier_signal\s+-1(?:\s|$)", assembly_body,
                                   re.MULTILINE))
    if (barrier_pairs != PROFILE["barrier_pairs"] or
            len(re.findall(r"^\s*s_barrier_wait\s+-1(?:\s|$)", assembly_body,
                           re.MULTILINE)) != barrier_pairs or
            re.search(r"^\s*s_barrier(?:\s|$)", assembly_body, re.MULTILINE)):
        raise ValueError(f"expected {PROFILE['barrier_pairs']} signal/wait barrier pairs")
    lds = _one_integer(metadata_body,
                       r"^\s*\.amdhsa_group_segment_fixed_size\s+(\d+)", "LDS size")
    private = _one_integer(metadata_body,
                           r"^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)",
                           "private segment size")
    vgpr = _one_integer(metadata_body, r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)",
                        "next-free VGPR")
    flat_scratch = _one_integer(
        metadata_body, r"^\s*\.set\s+\S+\.uses_flat_scratch,\s*(\d+)", "flat-scratch use")
    scratch = _one_integer(metadata_body, r"^;\s*ScratchSize:\s*(\d+)", "scratch size")
    occupancy = _one_integer(metadata_body, r"^;\s*Occupancy:\s*(\d+)", "occupancy")
    wave32 = _one_integer(metadata_body, r"^\s*\.amdhsa_wavefront_size32\s+(\d+)",
                          "wave32 mode")
    wgp_mode = _one_integer(
        metadata_body, r"^\s*\.amdhsa_workgroup_processor_mode\s+(\d+)", "WGP mode")
    maximum_workgroup = _one_integer(
        kernel_record, r"^\s*\.max_flat_workgroup_size:\s*(\d+)",
        "maximum flat workgroup size")
    wavefront_size = _one_integer(
        kernel_record, r"^\s*\.wavefront_size:\s*(\d+)", "wavefront size")
    vgpr_spills = _one_integer(
        kernel_record, r"^\s*\.vgpr_spill_count:\s*(\d+)", "VGPR spill count")
    sgpr_spills = _one_integer(
        kernel_record, r"^\s*\.sgpr_spill_count:\s*(\d+)", "SGPR spill count")
    if lds != PROFILE["lds"]:
        raise ValueError(f"LDS size {lds}, expected exactly {PROFILE['lds']}")
    if vgpr > PROFILE["vgpr"] or occupancy < PROFILE["occupancy"]:
        raise ValueError(f"resources fail: next_free_vgpr={vgpr}/{PROFILE['vgpr']} "
                         f"occupancy={occupancy}/{PROFILE['occupancy']}")
    if (wave32 != 1 or wavefront_size != 32 or wgp_mode != 1 or
            maximum_workgroup != PROFILE["workgroup"]):
        raise ValueError(
            f"execution mode fails: wave32={wave32} wavefront={wavefront_size} "
            f"wgp={wgp_mode} max_workgroup={maximum_workgroup}/{PROFILE['workgroup']}")
    if (private != 0 or scratch != 0 or flat_scratch != 0 or
            vgpr_spills != 0 or sgpr_spills != 0):
        raise ValueError(
            f"private={private} scratch={scratch} flat_scratch={flat_scratch} "
            f"vgpr_spills={vgpr_spills} sgpr_spills={sgpr_spills}; all must be zero")
    return {"symbol": symbol, "value_group": value_group, "bf16_wmma": bf16, "f16_wmma": f16,
            "barrier_pairs": barrier_pairs, "lds_bytes": lds, "next_free_vgpr": vgpr,
            "occupancy": occupancy, "private_bytes": private, "scratch_bytes": scratch,
            "flat_scratch": flat_scratch, "maximum_workgroup_size": maximum_workgroup,
            "wavefront_size": wavefront_size, "workgroup_processor_mode": wgp_mode,
            "vgpr_spills": vgpr_spills, "sgpr_spills": sgpr_spills}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--value-group", required=True, type=int, choices=(16, 32))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = check(assembly=args.assembly, metadata=args.metadata, symbol=args.symbol,
                       value_group=args.value_group)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    print("production=true selected_route=fused_dense_prefill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
