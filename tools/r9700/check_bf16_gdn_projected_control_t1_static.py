#!/usr/bin/env python3
import re
import sys
from pathlib import Path


KERNEL = "bf16_projected_control_t1_kernel"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ASSEMBLY")
    text = Path(sys.argv[1]).read_text()
    symbols = [symbol for symbol in re.findall(r"(?m)^\s*\.amdhsa_kernel\s+(\S+)\s*$", text)
               if KERNEL in symbol]
    if len(symbols) != 1:
        raise SystemExit(f"{KERNEL}: expected one AMDGPU kernel, found {len(symbols)}")
    symbol = symbols[0]
    start_match = re.search(rf"(?m)^\s*\.type\s+{re.escape(symbol)},@function\s*$", text)
    if start_match is None:
        raise SystemExit(f"{KERNEL}: device body was not found")
    next_body = re.search(r"(?m)^\s*\.type\s+[^,]+,@function\s*$", text[start_match.end():])
    end = len(text) if next_body is None else start_match.end() + next_body.start()
    body = text[start_match.start():end]
    required = {
        "wave32": ".amdhsa_wavefront_size32 1",
        "fp32_fma": "v_fmac_f32",
        "exponential": "_exp_f32",
        "reciprocal": "_rcp_f32",
        "no_scratch": ".amdhsa_private_segment_fixed_size 0",
        "two_bf16_reductions": ".amdhsa_group_segment_fixed_size 2048",
    }
    for label, needle in required.items():
        if needle not in body:
            raise SystemExit(f"{KERNEL}: missing {label}")
    if "scratch_load" in body or "scratch_store" in body:
        raise SystemExit(f"{KERNEL}: unexpected scratch traffic")
    stores32 = len(re.findall(r"(?m)^\s*global_store_b32\b", body))
    stores16 = len(re.findall(r"(?m)^\s*global_store(?:_d16_hi)?_b16\b", body))
    if stores32 != 2 or stores16 != 0:
        raise SystemExit(
            f"{KERNEL}: expected exactly two FP32 stores and no BF16 materialization, "
            f"found b32={stores32} b16={stores16}")
    fields = {}
    for field in ("next_free_vgpr", "next_free_sgpr", "group_segment_fixed_size",
                  "private_segment_fixed_size"):
        match = re.search(rf"\.amdhsa_{field}\s+(\d+)", body)
        if match is None:
            raise SystemExit(f"{KERNEL}: missing {field}")
        fields[field] = int(match.group(1))
    print("PASS kernel=bf16_projected_control_t1_kernel "
          f"vgpr={fields['next_free_vgpr']} sgpr={fields['next_free_sgpr']} "
          f"lds={fields['group_segment_fixed_size']} "
          f"scratch={fields['private_segment_fixed_size']} stores_fp32=2 stores_bf16=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
