#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=Path)
    parser.add_argument("--expect-receipt", type=Path)
    args = parser.parse_args()
    text = args.assembly.read_text(encoding="utf-8")
    symbols = sorted(set(re.findall(
        r"^\s*\.globl\s+(\S*paired_projection_q4_c2c4_kernel\S*)\s+; -- Begin function",
        text, re.MULTILINE)))
    require(len(symbols) == 1, f"expected one production paired-WMMA kernel, found {len(symbols)}")
    symbol = symbols[0]
    body_match = re.search(
        rf"^\s*\.globl\s+{re.escape(symbol)}.*?^\s*\.size\s+{re.escape(symbol)},",
        text, re.MULTILINE | re.DOTALL)
    require(body_match is not None, "production paired-WMMA body is missing")
    body = body_match.group(0)
    require("v_wmma_i32_16x16x32_iu4" in body,
            "production paired-WMMA kernel lacks native gfx1201 IU4 WMMA")
    require("scratch" not in body.lower(), "production paired-WMMA body contains scratch traffic")
    descriptor_match = re.search(
        rf"\.amdhsa_kernel\s+{re.escape(symbol)}(.*?)\.end_amdhsa_kernel", text, re.DOTALL)
    require(descriptor_match is not None, "production paired-WMMA descriptor is missing")
    descriptor = descriptor_match.group(1)
    for key, value in {
        ".amdhsa_wavefront_size32": "1",
        ".amdhsa_group_segment_fixed_size": "0",
        ".amdhsa_private_segment_fixed_size": "0",
    }.items():
        require(re.search(rf"{re.escape(key)}\s+{value}\b", descriptor) is not None,
                f"production paired-WMMA descriptor differs for {key}")
    vgpr = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", descriptor)
    sgpr = re.search(r"\.amdhsa_next_free_sgpr\s+(\d+)", descriptor)
    require(vgpr is not None and int(vgpr.group(1)) <= 64, "production kernel exceeds 64 VGPR")
    require(sgpr is not None and int(sgpr.group(1)) <= 64, "production kernel exceeds 64 SGPR")
    metadata = [block for block in re.findall(
        r"^  - \.args:.*?(?=^  - \.args:|^\.\.\.)", text, re.MULTILINE | re.DOTALL)
        if re.search(rf"\.name:\s+{re.escape(symbol)}\s*$", block, re.MULTILINE)]
    require(len(metadata) == 1, "production paired-WMMA metadata missing or duplicated")
    for key, value in {
        ".group_segment_fixed_size": "0", ".private_segment_fixed_size": "0",
        ".sgpr_spill_count": "0", ".vgpr_spill_count": "0", ".wavefront_size": "32",
    }.items():
        require(re.search(rf"{re.escape(key)}:\s+{value}\b", metadata[0]) is not None,
                f"production paired-WMMA metadata differs for {key}")
    receipt = (f"paired_projection_c2c4_op_static: PASS symbol={symbol} "
               f"vgpr={vgpr.group(1)} sgpr={sgpr.group(1)} "
               "wave32=true lds=0 scratch=0 spills=0 native_iu4_wmma=true")
    if args.expect_receipt is not None:
        require(args.expect_receipt.is_file() and not args.expect_receipt.is_symlink(),
                "bound production static receipt is missing or unsafe")
        require(args.expect_receipt.read_text(encoding="utf-8").strip() == receipt,
                "bound production static receipt differs")
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
