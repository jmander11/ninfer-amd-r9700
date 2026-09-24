#!/usr/bin/env python3
"""Fail-closed ISA/resource gate for the selected production full-score stages."""

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


FULL_SCORE_PROFILES = {
    "qk_bk16": {"lds": 8296, "wmma": 16, "vgpr": 64, "occupancy": 8,
                "workgroup": 192, "rounded_lds": 12288},
    "qk_bk32": {"lds": 16488, "wmma": 32, "vgpr": 128, "occupancy": 8,
                "workgroup": 192, "rounded_lds": 16896},
    "maximum": {"lds": 64, "wmma": 0, "vgpr": 64, "occupancy": 8,
                "workgroup": 256, "rounded_lds": 512},
    "pv_g16": {"lds": 9208, "wmma": 0, "vgpr": 128, "occupancy": 8,
               "workgroup": 256, "rounded_lds": 12288},
    "pv_g32": {"lds": 8952, "wmma": 0, "vgpr": 128, "occupancy": 8,
               "workgroup": 256, "rounded_lds": 12288},
}
LDS_ALLOCATION_GRANULE = 512


def _kernel_metadata_record(text: str, symbol: str) -> str:
    records = re.split(r"(?=^  - \.args:)", text, flags=re.MULTILINE)
    selected = [record for record in records if re.search(
        rf"^    \.name:\s+{re.escape(symbol)}$", record, flags=re.MULTILINE)]
    if len(selected) != 1:
        raise ValueError(
            "metadata: exact selected symbol must have one AMDGPU metadata record, "
            f"found {len(selected)}")
    return selected[0]


def check(*, assembly: Path, metadata: Path, symbol: str,
          value_group: int, full_score_stage: str) -> dict[str, int | str]:
    if value_group not in (16, 32):
        raise ValueError("dense attention requires G16 or G32")
    if full_score_stage == "qk_bk16":
        specialization = "26dense_full_score_qk_kernelILb0EE"
    elif full_score_stage == "qk_bk32":
        specialization = "31dense_full_score_qk_bk32_kernelILb0EE"
    elif full_score_stage == "maximum":
        specialization = "31dense_full_score_maximum_kernelILb0EE"
    elif full_score_stage == "pv":
        specialization = f"26dense_full_score_pv_kernelILj{value_group}ELb0EE"
    else:
        raise ValueError("full-score stage must be qk_bk16, qk_bk32, maximum, or pv")
    if specialization not in symbol:
        raise ValueError(
            "selected symbol is not the exact production "
            f"full-score {full_score_stage} non-active specialization")
    assembly_body = _function(assembly.read_text(encoding="utf-8"), symbol, "assembly")
    metadata_text = metadata.read_text(encoding="utf-8")
    metadata_body = _function(metadata_text, symbol, "metadata")
    kernel_record = _kernel_metadata_record(metadata_text, symbol)
    opcode = "v_wmma_f32_16x16x16_bf16"
    count = len(re.findall(rf"^\s*{opcode}(?:\s|$)", assembly_body, re.MULTILINE))
    profile_name = f"pv_g{value_group}" if full_score_stage == "pv" else full_score_stage
    profile = FULL_SCORE_PROFILES[profile_name]
    expected_wmma = profile["wmma"]
    if count != expected_wmma:
        raise ValueError(
            f"BF16 WMMA count {count}, expected {expected_wmma} for selected stage")
    forbidden = re.findall(r"^\s*(v_wmma_\S*(?:fp8|iu[48])\S*)", assembly_body,
                           re.MULTILINE | re.IGNORECASE)
    if forbidden:
        raise ValueError(f"selected BF16-Q profile contains forbidden matrix opcodes: {forbidden}")
    if full_score_stage in ("maximum", "pv") and re.search(
            r"^\s*v_wmma_", assembly_body, re.MULTILINE):
        raise ValueError("maximum/PV stage must not contain any WMMA instruction")
    if full_score_stage == "pv":
        if not re.search(r"^\s*v_exp_f32(?:_e(?:32|64))?(?:\s|$)", assembly_body,
                         re.MULTILINE):
            raise ValueError("PV stage is missing native FP32 exponential")
        if not re.search(r"^\s*v_fma(?:c)?_f32(?:_e(?:32|64))?(?:\s|$)", assembly_body,
                         re.MULTILINE):
            raise ValueError("PV stage is missing FP32 probability-value accumulation")
    lds = _one_integer(metadata_body,
                       r"^\s*\.amdhsa_group_segment_fixed_size\s+(\d+)", "LDS size")
    private = _one_integer(metadata_body,
                           r"^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)",
                           "private segment size")
    vgpr = _one_integer(metadata_body, r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)",
                        "VGPR count")
    flat_scratch = _one_integer(
        metadata_body, r"^\s*\.set\s+\S+\.uses_flat_scratch,\s*(\d+)",
        "flat-scratch use")
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
    metadata_wgp_mode = _one_integer(
        kernel_record, r"^\s*\.workgroup_processor_mode:\s*(\d+)",
        "metadata WGP mode")
    vgpr_spills = _one_integer(
        kernel_record, r"^\s*\.vgpr_spill_count:\s*(\d+)", "VGPR spill count")
    sgpr_spills = _one_integer(
        kernel_record, r"^\s*\.sgpr_spill_count:\s*(\d+)", "SGPR spill count")
    rounded_lds = ((lds + LDS_ALLOCATION_GRANULE - 1) // LDS_ALLOCATION_GRANULE *
                   LDS_ALLOCATION_GRANULE)
    expected_lds = profile["lds"]
    if lds != expected_lds:
        raise ValueError(f"Bq16 LDS size {lds}, expected exactly {expected_lds}")
    maximum_vgpr = profile["vgpr"]
    maximum_rounded_lds = profile["rounded_lds"]
    minimum_occupancy = profile["occupancy"]
    if (vgpr > maximum_vgpr or rounded_lds > maximum_rounded_lds or
            occupancy < minimum_occupancy):
        raise ValueError(
            f"resources fail: vgpr={vgpr}/{maximum_vgpr} "
            f"rounded_lds={rounded_lds}/{maximum_rounded_lds} "
            f"occupancy={occupancy}/{minimum_occupancy}")
    expected_workgroup = profile["workgroup"]
    if (wave32 != 1 or wavefront_size != 32 or wgp_mode != 1 or metadata_wgp_mode != 1 or
            maximum_workgroup != expected_workgroup):
        raise ValueError(
            f"execution mode fails: wave32={wave32} wavefront={wavefront_size} "
            f"wgp={wgp_mode}/{metadata_wgp_mode} "
            f"max_workgroup={maximum_workgroup}/{expected_workgroup}")
    if (private != 0 or scratch != 0 or flat_scratch != 0 or
            vgpr_spills != 0 or sgpr_spills != 0):
        raise ValueError(
            f"private={private} scratch={scratch} flat_scratch={flat_scratch} "
            f"vgpr_spills={vgpr_spills} sgpr_spills={sgpr_spills}; all must be zero")
    return {"symbol": symbol, "value_group": value_group, "query_tile": 16,
            "full_score_stage": full_score_stage,
            "bf16_wmma_count": count, "lds_bytes": lds, "rounded_lds_bytes": rounded_lds,
            "vgpr_count": vgpr,
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
    parser.add_argument("--full-score-stage", required=True,
                        choices=("qk_bk16", "qk_bk32", "maximum", "pv"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = check(assembly=args.assembly, metadata=args.metadata, symbol=args.symbol,
                       value_group=args.value_group, full_score_stage=args.full_score_stage)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    print("production=true selected_route=full_score_" + args.full_score_stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
