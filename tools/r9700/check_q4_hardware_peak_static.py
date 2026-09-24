#!/usr/bin/env python3
"""Fail-closed ISA/resource gate for the qualification-only Q4 peak probe."""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

BEGIN = re.compile(r"-- Begin function\s+(\S+)")


def function(text: str, needle: str) -> tuple[str, str]:
    matches = list(BEGIN.finditer(text))
    selected = [index for index, match in enumerate(matches) if needle in match.group(1)]
    if len(selected) != 1:
        raise ValueError(f"{needle}: expected one exact function body, found {len(selected)}")
    index = selected[0]
    end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
    return matches[index].group(1), text[matches[index].start():end]


def integer(body: str, pattern: str, name: str) -> int:
    values = re.findall(pattern, body, flags=re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"{name}: expected one value, found {len(values)}")
    return int(values[0])


def resources(body: str, *, vgpr_ceiling: int, occupancy_floor: int,
              lds_ceiling: int = 0) -> dict[str, int]:
    private = integer(body, r"^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)", "private")
    scratch = integer(body, r"^;\s*ScratchSize:\s*(\d+)", "scratch")
    flat = integer(body, r"^\s*\.set\s+\S+\.uses_flat_scratch,\s*(\d+)", "flat scratch")
    vgpr = integer(body, r"^;\s*NumVgprs:\s*(\d+)", "VGPR")
    occupancy = integer(body, r"^;\s*Occupancy:\s*(\d+)", "occupancy")
    lds = integer(body, r"^;\s*LDSByteSize:\s*(\d+)", "LDS")
    if private or scratch or flat:
        raise ValueError(f"scratch/private storage forbidden: private={private} scratch={scratch} flat={flat}")
    if vgpr > vgpr_ceiling:
        raise ValueError(f"VGPR count {vgpr} exceeds ceiling {vgpr_ceiling}")
    if occupancy < occupancy_floor:
        raise ValueError(f"occupancy {occupancy} is below floor {occupancy_floor}")
    if lds > lds_ceiling:
        raise ValueError(f"LDS bytes {lds} exceeds ceiling {lds_ceiling}")
    return {"vgpr": vgpr, "occupancy": occupancy, "scratch": scratch, "lds": lds}


def wmma_lines(body: str, opcode: str, label: str) -> list[str]:
    lines = re.findall(rf"^\s*{re.escape(opcode)}[^\n]*$", body, re.MULTILINE)
    if len(lines) != 8:
        raise ValueError(f"{label} must contain exactly eight static instructions, got {len(lines)}")
    if any("neg_lo:[1,1,0]" not in line for line in lines):
        raise ValueError(f"every {label} instruction must be signed/signed and unclamped")
    if re.search(rf"^\s*v_wmma_(?!{re.escape(opcode.removeprefix('v_wmma_'))})", body,
                 re.MULTILINE):
        raise ValueError(f"{label} contains another WMMA opcode")
    return lines


def wmma_register_topology(lines: list[str], expected_destinations: int,
                           expected_reuses: int, label: str) -> dict[str, object]:
    parsed = []
    for line in lines:
        match = re.search(r"(v\[\d+:\d+\]),\s*(v\[\d+:\d+\]),\s*"
                          r"(v\[\d+:\d+\]),\s*(v\[\d+:\d+\])", line)
        if match is None or match.group(1) != match.group(4):
            raise ValueError(f"{label} must update each accumulator in place")
        parsed.append(match.groups())
    destination_counts = collections.Counter(item[0] for item in parsed)
    if len(destination_counts) != expected_destinations or set(destination_counts.values()) != {
            expected_reuses}:
        raise ValueError(f"{label} dependency topology mismatch: {dict(destination_counts)}")
    operand_pairs = {(item[1], item[2]) for item in parsed}
    if len(operand_pairs) != 1:
        raise ValueError(f"{label} must reuse one register-resident operand pair")
    return {"accumulator_chains": len(destination_counts),
            "instructions_per_chain": expected_reuses,
            "operand_register_pairs": len(operand_pairs)}


def check(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    iu4_symbol, iu4 = function(text, "iu4_issue_peak_kernel")
    iu8_topology_symbol, iu8_topology = function(text, "iu8_topology_peak_kernel")
    iu8_saturation_symbol, iu8_saturation = function(text, "iu8_saturation_peak_kernel")
    stream_symbol, stream = function(text, "q4_code_scale_stream_kernel")
    iu4_lines = wmma_lines(iu4, "v_wmma_i32_16x16x32_iu4", "IU4 saturation peak")
    iu8_topology_lines = wmma_lines(
        iu8_topology, "v_wmma_i32_16x16x16_iu8", "IU8 topology peak")
    iu8_saturation_lines = wmma_lines(
        iu8_saturation, "v_wmma_i32_16x16x16_iu8", "IU8 saturation peak")
    if len(re.findall(r"^\s*global_load_b128\b", stream, re.MULTILINE)) != 2:
        raise ValueError("Q4 stream must contain exactly code and scale b128 load sites")
    if len(re.findall(r"^\s*global_store_b128\b", stream, re.MULTILINE)) != 1:
        raise ValueError("Q4 stream must contain exactly one observable checksum store")
    if not re.search(r"^\s*v_and_b32_e32\s+\S+,\s*15,", stream, re.MULTILINE):
        raise ValueError("Q4 stream lacks the exact one-scale-vector-per-16-code-vectors selector")
    if not re.search(r"^\s*v_lshlrev_b64_e32\s+\S+,\s*4,", stream, re.MULTILINE):
        raise ValueError("Q4 stream lacks the 16-byte packed-code vector address scale")
    return {
        "iu4_symbol": iu4_symbol,
        "iu8_topology_symbol": iu8_topology_symbol,
        "iu8_saturation_symbol": iu8_saturation_symbol,
        "stream_symbol": stream_symbol,
        "iu4_opcode_count": len(iu4_lines),
        "iu8_topology_opcode_count": len(iu8_topology_lines),
        "iu8_saturation_opcode_count": len(iu8_saturation_lines),
        "iu8_topology_dependency": wmma_register_topology(
            iu8_topology_lines, 2, 4, "IU8 topology peak"),
        "iu8_saturation_dependency": wmma_register_topology(
            iu8_saturation_lines, 8, 1, "IU8 saturation peak"),
        "iu4": resources(iu4, vgpr_ceiling=128, occupancy_floor=10),
        "iu8_topology": resources(iu8_topology, vgpr_ceiling=128, occupancy_floor=10),
        "iu8_saturation": resources(iu8_saturation, vgpr_ceiling=128, occupancy_floor=10),
        "stream": resources(stream, vgpr_ceiling=32, occupancy_floor=16),
        "stream_static_access_ratio": "16 code bytes per represented scale byte",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args()
    result = check(args.assembly)
    print("Q4 hardware peak static gate: PASS", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
