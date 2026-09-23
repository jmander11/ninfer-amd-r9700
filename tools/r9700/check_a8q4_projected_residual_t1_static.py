#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_kernel(text: str, name: str, dots: int, vgpr_limit: int,
                 sgpr_limit: int, staged: bool = False) -> tuple[str, int, int]:
    symbols = sorted(set(re.findall(
        rf"^\s*\.globl\s+(\S*{re.escape(name)}\S*)\s+; -- Begin function",
        text, re.MULTILINE)))
    require(len(symbols) == 1,
            f"expected one exact projected-residual kernel, found {len(symbols)}")
    symbol = symbols[0]
    require(symbol.startswith("_ZN6ninfer3ops5r97006linear"),
            "projected-residual kernel must be owned by the linked R9700 linear implementation")
    body_match = re.search(
        rf"^\s*\.globl\s+{re.escape(symbol)}.*?^\s*\.size\s+{re.escape(symbol)},",
        text, re.MULTILINE | re.DOTALL)
    require(body_match is not None, "projected-residual kernel body is missing")
    body = body_match.group(0)
    require(len(re.findall(r"\bv_dot8_i32_iu4\b", body)) == dots,
            f"{name} does not contain {dots} native IU4 dot8 instructions")
    require("scratch" not in body.lower(),
            "projected-residual kernel body contains scratch traffic")
    if staged:
        before_dot = body.split("v_dot8_i32_iu4", 1)[0]
        require(len(re.findall(r"\bglobal_load_b64\b", before_dot)) >= 8,
                "down stage2 must issue both groups' eight packed-weight loads before compute")
        require("sched_barrier" in before_dot,
                "down stage2 load/compute scheduling boundary is missing")
    descriptor_match = re.search(
        rf"\.amdhsa_kernel\s+{re.escape(symbol)}(.*?)\.end_amdhsa_kernel",
        text, re.DOTALL)
    require(descriptor_match is not None, "projected-residual descriptor is missing")
    descriptor = descriptor_match.group(1)
    for key, value in {
        ".amdhsa_wavefront_size32": "1",
        ".amdhsa_group_segment_fixed_size": "0",
        ".amdhsa_private_segment_fixed_size": "0",
    }.items():
        require(re.search(rf"{re.escape(key)}\s+{value}\b", descriptor) is not None,
                f"projected-residual descriptor differs for {key}")
    vgpr = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", descriptor)
    sgpr = re.search(r"\.amdhsa_next_free_sgpr\s+(\d+)", descriptor)
    require(vgpr is not None and int(vgpr.group(1)) <= vgpr_limit,
            f"{name} exceeds {vgpr_limit} VGPR")
    require(sgpr is not None and int(sgpr.group(1)) <= sgpr_limit,
            f"{name} exceeds {sgpr_limit} SGPR")
    after_descriptor = text[descriptor_match.end():]
    occupancy = re.search(r"^; Occupancy:\s+(\d+)",
                          after_descriptor.split("; -- Begin function", 1)[0], re.MULTILINE)
    require(occupancy is not None and int(occupancy.group(1)) >= 16,
            f"{name} does not retain occupancy 16")
    metadata = [block for block in re.findall(
        r"^  - \.args:.*?(?=^  - \.args:|^\.\.\.)", text, re.MULTILINE | re.DOTALL)
        if re.search(rf"\.name:\s+{re.escape(symbol)}\s*$", block, re.MULTILINE)]
    require(len(metadata) == 1, "projected-residual metadata missing or duplicated")
    for key, value in {
        ".group_segment_fixed_size": "0", ".private_segment_fixed_size": "0",
        ".sgpr_spill_count": "0", ".vgpr_spill_count": "0", ".wavefront_size": "32",
    }.items():
        require(re.search(rf"{re.escape(key)}:\s+{value}\b", metadata[0]) is not None,
                f"projected-residual metadata differs for {key}")
    # Metadata includes the reserved SGPRs omitted by next_free_sgpr.
    total_sgpr = re.search(r"\.sgpr_count:\s+(\d+)", metadata[0])
    require(total_sgpr is not None and int(total_sgpr.group(1)) <= sgpr_limit,
            f"{name} exceeds {sgpr_limit} total SGPR")
    return symbol, int(vgpr.group(1)), int(total_sgpr.group(1))


def check(text: str) -> str:
    symbol, vgpr, sgpr = check_kernel(
        text, "a8q4g64_projected_residual_t1_kernel", 16, 32, 64)
    _, down_vgpr, down_sgpr = check_kernel(
        text, "a8q4g64_projected_residual_down_stage2_kernel", 32, 58, 18, True)
    return (f"a8q4_projected_residual_t1_static: PASS symbol={symbol} "
            f"vgpr={vgpr} sgpr={sgpr} wave32=true lds=0 scratch=0 spills=0 "
            f"down_stage2_dot8=32 down_stage2_vgpr={down_vgpr} "
            f"down_stage2_sgpr={down_sgpr} down_stage2_occupancy=16")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=Path)
    parser.add_argument("--expect-receipt", type=Path)
    args = parser.parse_args()
    receipt = check(args.assembly.read_text(encoding="utf-8"))
    if args.expect_receipt is not None:
        require(args.expect_receipt.is_file() and not args.expect_receipt.is_symlink(),
                "bound static receipt is missing or unsafe")
        require(args.expect_receipt.read_text(encoding="utf-8").strip() == receipt,
                "bound static receipt does not describe this assembly")
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
