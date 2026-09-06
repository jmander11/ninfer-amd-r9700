#!/usr/bin/env python3
"""Fail-closed gfx1201 static gate for the Q4/W8 prefill CTA memory pipeline."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

BEGIN = re.compile(r"-- Begin function\s+(\S+)")

@dataclass(frozen=True)
class Profile:
    incumbent_symbol: str
    production_symbol: str
    opcode: str
    opcode_count: int
    lds_ceiling: int
    vgpr_ceiling: int
    incumbent_global_inv_count: int
    incumbent_opcode_count: int
    incumbent_lds_ceiling: int
    incumbent_vgpr_ceiling: int
    maximum_workgroup_size: int
    occupancy: int = 0

PROFILES = {
    "q4": Profile(
        "_ZN6ninfer3ops5r97006linear12_GLOBAL__N_152a8q4g64_linear_prefill_cta_m64n128_regression_kernelEPKhS5_PKtPKjS5_S7_P12hip_bfloat16jjjj",
        "_ZN6ninfer3ops5r97006linear12_GLOBAL__N_133a8q4g64_linear_prefill_cta_kernelEPKhS5_PKtPKjS5_S7_P12hip_bfloat16jjjj",
        "v_wmma_i32_16x16x32_iu4", 8, 17152, 96, 0, 8, 8576, 96, 512, 16,
    ),
    "w8": Profile(
        "_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158a8w8g32_linear_prefill_cta_global_inv_qualification_kernelEPKaPKtPKjS5_S7_P12hip_bfloat16jjjj",
        "_ZN6ninfer3ops5r97006linear12_GLOBAL__N_133a8w8g32_linear_prefill_cta_kernelEPKaPKtPKjS5_S7_P12hip_bfloat16jjjj",
        "v_wmma_i32_16x16x16_iu8", 2, 4352, 50, 2, 2, 4352, 50, 512,
    ),
    "q4-m128n128": Profile(
        "",
        "_ZN6ninfer3ops5r97006linear12_GLOBAL__N_156a8q4g64_linear_prefill_cta_m128n128_qualification_kernelEPKhS5_PKtPKjS5_S7_P12hip_bfloat16jjjj",
        "v_wmma_i32_16x16x32_iu4", 8, 12800, 96, 0, 0, 0, 0, 1024,
    ),
    "q4-a4-m64n128": Profile(
        "",
        "_ZN6ninfer3ops5r97006linear12_GLOBAL__N_153q4g64_linear_prefill_cta_m64n128_qualification_kernelEPKhPKtPKjS5_S7_P12hip_bfloat16jjjj",
        "v_wmma_i32_16x16x32_iu4", 4, 6528, 84, 0, 0, 0, 0, 512, 16,
    ),
}

def _function(text: str, symbol: str, label: str) -> str:
    matches = list(BEGIN.finditer(text))
    selected = [i for i, match in enumerate(matches) if match.group(1) == symbol]
    if len(selected) != 1:
        raise ValueError(
            f"{label}: exact selected symbol must have one function body, found {len(selected)}"
        )
    index = selected[0]
    end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
    return text[matches[index].start():end]

def _one_integer(body: str, pattern: str, field: str) -> int:
    values = re.findall(pattern, body, flags=re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"selected metadata must contain one {field}, found {len(values)}")
    return int(values[0])

def _maximum_workgroup(text: str, symbol: str) -> int:
    values = re.findall(
        rf"^\s*\.max_flat_workgroup_size:\s*(\d+)\s*$\n"
        rf"^\s*\.name:\s*{re.escape(symbol)}\s*$",
        text, flags=re.MULTILINE,
    )
    if len(values) != 1:
        raise ValueError(
            "selected metadata must contain one symbol-bound maximum flat workgroup size, "
            f"found {len(values)}")
    return int(values[0])

def _kernel_metadata_record(text: str, symbol: str) -> str:
    names = list(re.finditer(rf"^\s*\.name:\s*{re.escape(symbol)}\s*$", text,
                             flags=re.MULTILINE))
    if len(names) != 1:
        raise ValueError(
            "selected metadata must contain one symbol-bound kernel record, "
            f"found {len(names)}")
    starts = [match.start() for match in re.finditer(r"^\s{2}- \.args:\s*$", text,
                                                      flags=re.MULTILINE)
              if match.start() < names[0].start()]
    if not starts:
        raise ValueError("selected metadata kernel record has no argument-list boundary")
    end = text.find("\n  - .args:", names[0].end())
    return text[starts[-1]:len(text) if end < 0 else end]

def check(recipe: str, mode: str, assembly: Path, metadata: Path) -> dict[str, int | str]:
    if mode not in ("lds-scope", "m128n128", "a4-m64n128",
                    "incumbent-diagnostic"):
        raise ValueError(f"unsupported static-gate mode: {mode}")
    if (recipe == "q4-m128n128") != (mode == "m128n128"):
        raise ValueError("q4-m128n128 recipe and mode must be selected together")
    if (recipe == "q4-a4-m64n128") != (mode == "a4-m64n128"):
        raise ValueError("q4-a4-m64n128 recipe and mode must be selected together")
    profile = PROFILES[recipe]
    symbol = (profile.incumbent_symbol if mode == "incumbent-diagnostic"
              else profile.production_symbol)
    assembly_body = _function(assembly.read_text(encoding="utf-8"), symbol, "assembly")
    metadata_body = _function(metadata.read_text(encoding="utf-8"), symbol, "metadata")
    expected_opcode_count = (profile.opcode_count if mode != "incumbent-diagnostic"
                             else profile.incumbent_opcode_count)
    opcode_count = len(re.findall(
        rf"^\s*{re.escape(profile.opcode)}(?:\s|$)", assembly_body, flags=re.MULTILINE
    ))
    if opcode_count != expected_opcode_count:
        raise ValueError(
            f"{recipe}: {profile.opcode} count {opcode_count}, expected {expected_opcode_count}"
        )
    if mode == "a4-m64n128":
        opcode_lines = re.findall(
            rf"^\s*{re.escape(profile.opcode)}[^\n]*$", assembly_body, flags=re.MULTILINE)
        if any("neg_lo:[1,1,0]" not in line for line in opcode_lines):
            raise ValueError(
                "q4-a4-m64n128: every IU4 WMMA must use signed A, signed W, unclamped I32")
    signal_operands = re.findall(r"^\s*s_barrier_signal\s+([^\s;]+)", assembly_body,
                                 flags=re.MULTILINE)
    wait_operands = re.findall(r"^\s*s_barrier_wait\s+([^\s;]+)", assembly_body,
                               flags=re.MULTILINE)
    signal_count = len(signal_operands)
    wait_count = len(wait_operands)
    if signal_count != 2 or wait_count != 2:
        raise ValueError(
            f"{recipe}: expected exactly two signal+wait pairs, got "
            f"signal={signal_count} wait={wait_count}"
        )
    if signal_operands != ["-1", "-1"] or wait_operands != ["-1", "-1"]:
        raise ValueError(
            f"{recipe}: every barrier must select the workgroup barrier id -1; "
            f"signal={signal_operands} wait={wait_operands}"
        )
    barrier_sequence = re.findall(
        r"^\s*(s_barrier_(?:signal|wait))(?:\s|$)", assembly_body, flags=re.MULTILINE
    )
    if barrier_sequence != [
        "s_barrier_signal", "s_barrier_wait", "s_barrier_signal", "s_barrier_wait"
    ]:
        raise ValueError(f"{recipe}: signal/wait operations are not two ordered barrier pairs")
    if re.search(r"^\s*s_barrier(?:\s|$)", assembly_body, flags=re.MULTILINE):
        raise ValueError(f"{recipe}: monolithic s_barrier is forbidden")
    barrier_mnemonics = re.findall(
        r"^\s*(s_barrier(?:_[A-Za-z0-9_]+)?)(?:\s|$)", assembly_body,
        flags=re.MULTILINE,
    )
    if barrier_mnemonics != barrier_sequence:
        raise ValueError(
            f"{recipe}: unexpected barrier-family instruction in selected symbol: "
            f"{barrier_mnemonics}"
        )
    global_inv_count = len(re.findall(r"^\s*global_inv(?:\s|$)", assembly_body,
                                      flags=re.MULTILINE))
    expected_global_inv = (0 if mode in ("lds-scope", "a4-m64n128")
                           else profile.incumbent_global_inv_count)
    if global_inv_count != expected_global_inv:
        raise ValueError(
            f"{recipe}: global_inv count {global_inv_count}, expected {expected_global_inv} "
            f"for {mode}"
        )
    lds = _one_integer(metadata_body, r"^\s*\.amdhsa_group_segment_fixed_size\s+(\d+)",
                       "LDS size")
    private = _one_integer(metadata_body, r"^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)",
                           "private segment size")
    vgpr = _one_integer(metadata_body, r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)", "VGPR count")
    flat_scratch = _one_integer(
        metadata_body, r"^\s*\.set\s+\S+\.uses_flat_scratch,\s*(\d+)", "flat-scratch use"
    )
    scratch = _one_integer(metadata_body, r"^;\s*ScratchSize:\s*(\d+)", "scratch size")
    occupancy = _one_integer(metadata_body, r"^;\s*Occupancy:\s*(\d+)", "occupancy")
    metadata_text = metadata.read_text(encoding="utf-8")
    maximum_workgroup = _maximum_workgroup(metadata_text, symbol)
    kernel_record = _kernel_metadata_record(metadata_text, symbol)
    sgpr_spills = _one_integer(kernel_record, r"^\s*\.sgpr_spill_count:\s*(\d+)",
                               "SGPR spill count")
    vgpr_spills = _one_integer(kernel_record, r"^\s*\.vgpr_spill_count:\s*(\d+)",
                               "VGPR spill count")
    if maximum_workgroup != profile.maximum_workgroup_size:
        raise ValueError(
            f"{recipe}: maximum flat workgroup {maximum_workgroup}, "
            f"expected {profile.maximum_workgroup_size}")
    lds_ceiling = (profile.lds_ceiling if mode != "incumbent-diagnostic"
                   else profile.incumbent_lds_ceiling)
    vgpr_ceiling = (profile.vgpr_ceiling if mode != "incumbent-diagnostic"
                    else profile.incumbent_vgpr_ceiling)
    if lds > lds_ceiling or vgpr > vgpr_ceiling:
        raise ValueError(
            f"{recipe}: resources lds={lds}/{lds_ceiling} "
            f"vgpr={vgpr}/{vgpr_ceiling} exceed the selected ceilings"
        )
    pipelined = recipe == "q4" and mode == "lds-scope"
    if pipelined and lds != lds_ceiling:
        raise ValueError(
            f"{recipe}: LDS size {lds}, expected exactly {lds_ceiling}")
    if private != 0 or scratch != 0 or flat_scratch != 0 or sgpr_spills != 0 or vgpr_spills != 0:
        raise ValueError(
            f"{recipe}: private={private} scratch={scratch} flat_scratch={flat_scratch} "
            f"sgpr_spills={sgpr_spills} vgpr_spills={vgpr_spills}; "
            "all must be zero"
        )
    if profile.occupancy and occupancy != profile.occupancy:
        raise ValueError(
            f"{recipe}: occupancy {occupancy}, expected exactly {profile.occupancy}")
    n16_weight_b64_sites = 0
    scalar_base_load_sites = 0
    if pipelined:
        opcode_lines = re.findall(
            rf"^\s*{re.escape(profile.opcode)}[^\n]*$", assembly_body,
            flags=re.MULTILINE,
        )
        if (sum("neg_lo:[0,1,0]" in line for line in opcode_lines) != 4 or
                sum("neg_lo:[1,1,0]" in line for line in opcode_lines) != 4):
            raise ValueError(
                "q4: expected four unsigned-low/signed-W and four signed-high/signed-W IU4 sites")
        first_wmma = assembly_body.find(profile.opcode)
        loads = list(re.finditer(r"^\s*global_load_b(?:32|64)(?:\s|$)",
                                 assembly_body[:first_wmma], flags=re.MULTILINE))
        if (sum("global_load_b32" in match.group(0) for match in loads) != 4 or
                sum("global_load_b64" in match.group(0) for match in loads) != 2):
            raise ValueError(
                "q4: expected prologue+successor pairs of two activation b32 and one N16/K16 weight b64 load")
        n16_weight_b64_sites = 2
        global_loads = re.findall(
            r"^\s*(global_load_(?:b32|b64|d16_b16))\s+"
            r"v(?:\[\d+:\d+\]|\d+),\s+v\d+,\s+(s\[\d+:\d+\])\s*$",
            assembly_body, flags=re.MULTILINE,
        )
        expected_loads = [
            ("global_load_b32", "s[4:5]"),
            ("global_load_b32", "s[6:7]"),
            ("global_load_b64", "s[16:17]"),
            ("global_load_d16_b16", "s[8:9]"),
            ("global_load_d16_b16", "s[18:19]"),
        ] * 2
        if global_loads != expected_loads:
            raise ValueError(
                "q4: production loads must retain exact scalar-base role/order "
                "with one VGPR U32 offset and default cache policy")
        scalar_base_load_sites = len(global_loads)
        overlap_window = assembly_body[loads[-3].start():first_wmma]
        overlap_widths = re.findall(r"^\s*global_load_b(32|64)(?:\s|$)",
                                    overlap_window, flags=re.MULTILINE)
        if overlap_widths != ["32", "32", "64"] or re.search(
                r"^\s*s_wait_loadcnt\s+0x0", overlap_window, flags=re.MULTILINE):
            raise ValueError(
                "q4: next-group activation b32+b32 and N16/K16 weight b64 must remain outstanding across current WMMA")
        last_wmma = assembly_body.rfind(profile.opcode)
        publish_window = assembly_body[last_wmma:]
        if len(re.findall(r"^\s*ds_store(?:_\S+)?", publish_window,
                          flags=re.MULTILINE)) < 3 or "s_wait_loadcnt 0x0" not in publish_window:
            raise ValueError(
                f"{recipe}: prefetched payload must be waited before LDS publication")
    return {
        "recipe": recipe, "mode": mode, "opcode_count": opcode_count,
        "barrier_pair_count": signal_count, "global_inv_count": global_inv_count,
        "lds_bytes": lds, "vgpr_count": vgpr, "private_bytes": private,
        "scratch_bytes": scratch, "flat_scratch": flat_scratch,
        "sgpr_spill_count": sgpr_spills, "vgpr_spill_count": vgpr_spills,
        "maximum_workgroup_size": maximum_workgroup,
        "occupancy": occupancy,
        "n16_weight_b64_sites": n16_weight_b64_sites,
        "scalar_base_load_sites": scalar_base_load_sites,
    }

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, choices=tuple(PROFILES))
    parser.add_argument(
        "--mode", required=True,
        choices=("lds-scope", "m128n128", "a4-m64n128",
                 "incumbent-diagnostic"),
        help="LDS-scope mode selects the promoted production symbol; incumbent mode is diagnostic.",
    )
    parser.add_argument("--assembly", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = check(args.recipe, args.mode, args.assembly, args.metadata)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    print("diagnostic_only=true admission=false" if args.mode == "incumbent-diagnostic"
          else ("a4_m64n128_challenger_static_gate=passed" if args.mode == "a4-m64n128"
                else "m128n128_challenger_static_gate=passed" if args.mode == "m128n128"
                else "lds_scope_production_static_gate=passed"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
