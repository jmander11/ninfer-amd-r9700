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
    args = parser.parse_args()
    text = args.assembly.read_text(encoding="utf-8")
    symbols = sorted(set(re.findall(
        r"^\s*\.globl\s+(\S*a8q4g64_gdn_pair_t1_kernel\S*)\s+; -- Begin function",
        text, re.MULTILINE)))
    require(len(symbols) == 1, f"expected one exact paired kernel symbol, found {len(symbols)}")
    symbol = symbols[0]
    body_match = re.search(
        rf"^\s*\.globl\s+{re.escape(symbol)}.*?^\s*\.size\s+{re.escape(symbol)},",
        text, re.MULTILINE | re.DOTALL)
    require(body_match is not None, "paired kernel body is missing")
    body = body_match.group(0)
    require("v_dot8_i32_iu4" in body, "paired kernel lacks native packed IU4 dot8")
    require("scratch" not in body.lower(), "paired kernel body contains scratch traffic")

    descriptor_match = re.search(
        rf"\.amdhsa_kernel\s+{re.escape(symbol)}(.*?)\.end_amdhsa_kernel",
        text, re.DOTALL)
    require(descriptor_match is not None, "paired kernel descriptor is missing")
    descriptor = descriptor_match.group(1)
    require(".amdhsa_wavefront_size32 1" in descriptor, "paired kernel is not wave32")
    require(".amdhsa_group_segment_fixed_size 0" in descriptor, "paired kernel uses LDS")
    require(".amdhsa_private_segment_fixed_size 0" in descriptor, "paired kernel uses scratch")
    vgpr = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", descriptor)
    sgpr = re.search(r"\.amdhsa_next_free_sgpr\s+(\d+)", descriptor)
    require(vgpr is not None and int(vgpr.group(1)) <= 32,
            "paired kernel exceeds the 32-VGPR envelope")
    require(sgpr is not None and int(sgpr.group(1)) <= 64,
            "paired kernel exceeds the 64-SGPR envelope")

    metadata_blocks = re.findall(r"^  - \.args:.*?(?=^  - \.args:|^\.\.\.)",
                                 text, re.MULTILINE | re.DOTALL)
    matching_metadata = [block for block in metadata_blocks
                         if re.search(rf"\.name:\s+{re.escape(symbol)}\s*$",
                                      block, re.MULTILINE)]
    require(len(matching_metadata) == 1, "paired kernel metadata is missing or duplicated")
    metadata = matching_metadata[0]
    expected = {
        ".group_segment_fixed_size": "0",
        ".private_segment_fixed_size": "0",
        ".sgpr_spill_count": "0",
        ".vgpr_spill_count": "0",
        ".wavefront_size": "32",
    }
    for key, value in expected.items():
        require(re.search(rf"{re.escape(key)}:\s+{value}\b", metadata) is not None,
                f"paired kernel metadata differs for {key}")
    print(f"a8q4_gdn_pair_t1_static: PASS symbol={symbol} "
          f"vgpr={vgpr.group(1)} sgpr={sgpr.group(1)} wave32=true lds=0 scratch=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
