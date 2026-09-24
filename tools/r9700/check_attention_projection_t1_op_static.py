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
        r"^\s*\.globl\s+(\S*attention_projection_q4_t1_kernel\S*)\s+; -- Begin function",
        text, re.MULTILINE)))
    require(len(symbols) == 1,
            f"expected one production attention-projection kernel, found {len(symbols)}")
    symbol = symbols[0]
    body_match = re.search(
        rf"^\s*\.globl\s+{re.escape(symbol)}.*?^\s*\.size\s+{re.escape(symbol)},",
        text, re.MULTILINE | re.DOTALL)
    require(body_match is not None, "production attention-projection body missing")
    body = body_match.group(0)
    require("v_dot8_i32_iu4" in body and "neg_lo:[0,1,0]" in body and
            "neg_lo:[1,1,0]" in body,
            "production attention projection lacks native mixed-sign IU4 dot8 operations")
    require("scratch" not in body.lower(), "production attention projection uses scratch")
    descriptor_match = re.search(
        rf"\.amdhsa_kernel\s+{re.escape(symbol)}(.*?)\.end_amdhsa_kernel",
        text, re.DOTALL)
    require(descriptor_match is not None, "production attention-projection descriptor missing")
    descriptor = descriptor_match.group(1)
    for key, value in {
        ".amdhsa_wavefront_size32": "1",
        ".amdhsa_group_segment_fixed_size": "0",
        ".amdhsa_private_segment_fixed_size": "0",
    }.items():
        require(re.search(rf"{re.escape(key)}\s+{value}\b", descriptor) is not None,
                f"production attention-projection descriptor differs for {key}")
    vgpr = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", descriptor)
    sgpr = re.search(r"\.amdhsa_next_free_sgpr\s+(\d+)", descriptor)
    require(vgpr is not None and int(vgpr.group(1)) <= 32,
            "production attention projection exceeds 32 VGPR")
    require(sgpr is not None and int(sgpr.group(1)) <= 64,
            "production attention projection exceeds 64 SGPR")
    metadata = [block for block in re.findall(
        r"^  - \.args:.*?(?=^  - \.args:|^\.\.\.)", text, re.MULTILINE | re.DOTALL)
        if re.search(rf"\.name:\s+{re.escape(symbol)}\s*$", block, re.MULTILINE)]
    require(len(metadata) == 1, "production attention-projection metadata missing or duplicated")
    for key, value in {
        ".group_segment_fixed_size": "0", ".private_segment_fixed_size": "0",
        ".sgpr_spill_count": "0", ".vgpr_spill_count": "0", ".wavefront_size": "32",
    }.items():
        require(re.search(rf"{re.escape(key)}:\s+{value}\b", metadata[0]) is not None,
                f"production attention-projection metadata differs for {key}")
    receipt = (f"attention_projection_t1_op_static: PASS symbol={symbol} "
               f"vgpr={vgpr.group(1)} sgpr={sgpr.group(1)} wave32=true "
               "lds=0 scratch=0 spills=0 native_mixed_iu4_dot8=true")
    if args.expect_receipt is not None:
        require(args.expect_receipt.is_file() and not args.expect_receipt.is_symlink(),
                "bound static receipt is missing or unsafe")
        require(args.expect_receipt.read_text(encoding="utf-8").strip() == receipt,
                "bound static receipt does not describe this assembly")
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
