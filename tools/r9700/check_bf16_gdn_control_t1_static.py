#!/usr/bin/env python3
import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ASSEMBLY")
    text = Path(sys.argv[1]).read_text()
    kernel_names = (
        "incumbent_projection_kernel",
        "combined_projection_kernel",
        "incumbent_control_kernel",
        "fused_projection_control_kernel",
    )
    # Select only symbols declared as AMDGPU kernels. A normal HIP -S output may also contain
    # host __device_stub__ functions with the readable kernel name; those are launch wrappers,
    # not device evidence, and must never overwrite the actual code object body.
    amdgpu_symbols = re.findall(r"(?m)^\s*\.amdhsa_kernel\s+(\S+)\s*$", text)
    kernels = {}
    starts = list(re.finditer(r"(?m)^\s*\.type\s+([^,]+),@function", text))
    for key in kernel_names:
        matches = [symbol for symbol in amdgpu_symbols if key in symbol]
        if len(matches) != 1:
            raise SystemExit(f"{key}: expected one .amdhsa_kernel symbol, found {len(matches)}")
        symbol = matches[0]
        selected = [index for index, match in enumerate(starts) if match.group(1) == symbol]
        if len(selected) != 1:
            raise SystemExit(f"{key}: expected one exact AMDGPU function body, found {len(selected)}")
        index = selected[0]
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        body = text[starts[index].start():end]
        if f".amdhsa_kernel {symbol}" not in body:
            raise SystemExit(f"{key}: selected body lacks its AMDGPU kernel descriptor")
        kernels[key] = body

    projection_names = (
        "incumbent_projection_kernel",
        "combined_projection_kernel",
        "fused_projection_control_kernel",
    )
    for name, body in kernels.items():
        if "scratch_load" in body or "scratch_store" in body:
            raise SystemExit(f"{name}: unexpected scratch traffic")
        if ".amdhsa_wavefront_size32 1" not in body:
            raise SystemExit(f"{name}: not emitted as wave32")
    for name in projection_names:
        body = kernels[name]
        if "v_fma_f32" not in body and "v_fmac_f32" not in body:
            raise SystemExit(f"{name}: missing FP32 FMA")
    for name in ("incumbent_control_kernel", "fused_projection_control_kernel"):
        body = kernels[name]
        if "_exp_f32" not in body or "_rcp_f32" not in body or "global_store_b32" not in body:
            raise SystemExit(f"{name}: missing exponential/reciprocal/control-output evidence")
    metadata = {}
    for name, body in kernels.items():
        fields = []
        for field in ("next_free_vgpr", "next_free_sgpr", "group_segment_fixed_size",
                      "private_segment_fixed_size"):
            match = re.search(rf"\.amdhsa_{field}\s+(\d+)", body)
            if match is None:
                raise SystemExit(f"{name}: missing {field} resource metadata")
            fields.append(int(match.group(1)))
        if len(fields) != 4:
            raise SystemExit(f"{name}: missing resource metadata")
        metadata[name] = tuple(fields)
    print("PASS " + " ".join(f"{name}=vgpr{v}/sgpr{s}/lds{lds}/scratch{scratch}"
                              for name, (v, s, lds, scratch) in metadata.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
