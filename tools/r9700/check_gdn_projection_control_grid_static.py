#!/usr/bin/env python3
"""CPU-only, gfx1201 emitted-code feasibility check; no numerical/perf claim."""
import argparse
import json
import re
from pathlib import Path


KERNEL = "gdn_projection_control_grid_qualification_kernel"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def kernel(text: str, fragment: str) -> tuple[str, str, dict]:
    matches = re.findall(r"(?m)^\s*\.amdhsa_kernel\s+(\S+)\s*$", text)
    symbols = [name for name in matches if fragment in name]
    require(len(symbols) == 1, f"expected exactly one {fragment} kernel")
    symbol = symbols[0]
    body_match = re.search(
        rf"(?m)^{re.escape(symbol)}:.*?^\s*\.size\s+{re.escape(symbol)},",
        text, re.DOTALL)
    require(body_match is not None, f"missing body: {symbol}")
    body = body_match.group(0)
    desc_match = re.search(
        rf"\.amdhsa_kernel\s+{re.escape(symbol)}\n(.*?)\.end_amdhsa_kernel",
        text, re.DOTALL)
    require(desc_match is not None, f"missing descriptor: {symbol}")
    desc = desc_match.group(1)
    fields = {}
    for field in ("next_free_vgpr", "next_free_sgpr", "group_segment_fixed_size",
                  "private_segment_fixed_size", "wavefront_size32"):
        match = re.search(rf"\.amdhsa_{field}\s+(\d+)\b", desc)
        require(match is not None, f"missing {field}")
        fields[field] = int(match.group(1))
    after = text[desc_match.end():]
    occupancy = re.search(r"; Occupancy:\s*(\d+)", after)
    require(occupancy is not None, "missing compiler occupancy")
    fields["compiler_occupancy"] = int(occupancy.group(1))
    blocks = re.findall(r"^  - \.args:.*?(?=^  - \.args:|^\.\.\.)",
                        text, re.MULTILINE | re.DOTALL)
    metadata = [b for b in blocks if re.search(
        rf"(?m)^\s*\.name:\s+{re.escape(symbol)}\s*$", b)]
    require(len(metadata) == 1, "missing unique metadata")
    for key in ("sgpr_spill_count", "vgpr_spill_count"):
        match = re.search(rf"\.{key}:\s+(\d+)\b", metadata[0])
        require(match is not None, f"missing {key}")
        fields[key] = int(match.group(1))
    require(fields["wavefront_size32"] == 1, "wave32 required")
    require(fields["private_segment_fixed_size"] == 0, "unexpected scratch allocation")
    require(fields["sgpr_spill_count"] == fields["vgpr_spill_count"] == 0,
            "unexpected spills")
    require("scratch_" not in body, "unexpected scratch instructions")
    return symbol, body, fields


def reduction_signature(body: str) -> list[str]:
    """Actual LDS accesses/offsets and full-CTA barrier sequence, ignoring registers."""
    last_barrier = body.rfind("s_barrier_wait")
    require(last_barrier >= 0, "no control barriers")
    prefix = body[:last_barrier + len("s_barrier_wait")]
    result = []
    for line in prefix.splitlines():
        instruction = re.match(r"\s*(ds_\w+|s_barrier_\w+)\b(.*)", line)
        if instruction:
            offsets = re.findall(r"offset\w*:\d+", instruction.group(2))
            result.append(" ".join([instruction.group(1), *offsets]))
    return result


def check(candidate: str, q4_text: str, control_text: str) -> dict:
    require('"amdgcn-amd-amdhsa--gfx1201"' in candidate, "wrong ISA target")
    symbol, body, resources = kernel(candidate, KERNEL)
    _, q4_baseline, q4_resources = kernel(q4_text, "a8q4g64_gdn_pair_t1_kernel")
    _, control_baseline, control_resources = kernel(control_text, "bf16_projected_control_t1_kernel")
    # Installed compiler emits control first and jumps around it for blocks 0..63.
    # Fail closed when that layout changes: a reviewer must re-establish the CFG.
    split = re.search(
        r"s_cmp_gt_u32\s+(ttmp\d+), 63\n\s*s_mov_b32\s+s\d+, -1\n"
        r"\s*s_cbranch_scc0\s+(\.LBB\d+_\d+)", body)
    require(split is not None, "missing scalar block-uniform 0..63 Q4 split")
    q4_label = re.search(rf"(?m)^{re.escape(split.group(2))}:.*$", body)
    require(q4_label is not None, "missing Q4 branch target")
    control = body[split.end():q4_label.start()]
    q4 = body[q4_label.end():]
    require(re.search(rf"s_cmp_gt_u32\s+{split.group(1)}, 0x6f\n"
                      r"\s*s_cbranch_scc1", control) is not None,
            "missing uniform upper bound 111")
    require(re.search(rf"s_sub_co_i32\s+s\d+, {split.group(1)}, 64", control) is not None,
            "control head must be block minus 64")
    require("ds_" not in q4 and "s_barrier" not in q4,
            "Q4 branch unexpectedly contains LDS/barriers")
    require("v_dot8_i32_iu4" not in control, "Q4 instructions leaked into control")
    require(len(re.findall(r"\bv_dot8_i32_iu4\b", q4)) == 16,
            "Q4 group must retain eight low/high native IU4 pairs")
    require(len(re.findall(r"neg_lo:\[0,1,0\]", q4)) == 8 and
            len(re.findall(r"neg_lo:\[1,1,0\]", q4)) == 8,
            "wrong low-unsigned/high-signed IU4 operands")
    require(reduction_signature(control) == reduction_signature(control_baseline),
            "control LDS reduction/barrier sequence differs from production")
    require(control.count("s_barrier_signal") == control.count("s_barrier_wait") == 9,
            "control needs initial plus eight reduction barriers")
    require(len(re.findall(r"s_add_co_i32[^\n]*0x7fff", control)) == 2,
            "missing two explicit BF16 RNE seams before gating")
    require(len(re.findall(r"(?m)^\s*global_store_b32\b", control)) == 2 and
            not re.search(r"(?m)^\s*global_store[^\n]*b16\b", control),
            "control must publish only FP32 g/beta")
    require(resources["group_segment_fixed_size"] == 2048,
            "merged LDS must remain exactly two 256-float reductions")
    require(resources["compiler_occupancy"] >= q4_resources["compiler_occupancy"],
            "combined resource envelope reduces Q4 compiler occupancy")
    require(resources["next_free_vgpr"] <= 40 and resources["next_free_sgpr"] <= 64,
            "unexpectedly expanded merged register envelope")
    return {
        "status": "static_feasible_requires_full_op_qualification",
        "kernel": symbol, "grid": 112, "block": 256,
        "q4_blocks": [0, 63], "control_blocks": [64, 111],
        "resources": resources,
        "production_q4_resources": q4_resources,
        "production_control_resources": control_resources,
        "q4_native_iu4_static_instructions": 16,
        "control_lds_reduction_matches_production": True,
        "control_bf16_rne_seams": 2,
        "q4_activation_scalar_b256_loads": len(re.findall(r"\bs_load_b256\b", q4)),
        "production_q4_activation_scalar_b256_loads": len(re.findall(r"\bs_load_b256\b", q4_baseline)),
        "q4_activation_vector_b128_loads": len(re.findall(r"\bglobal_load_b128\b", q4)),
        "limitations": [
            "Compiler occupancy is a resource ceiling, not measured active occupancy.",
            "Merged control flow replaces two scalar b256 activation loads with four vector b128 loads; issue-cost effect is unmeasured.",
            "No GPU run, independent numerical oracle, complete semantic launcher, or performance claim.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("q4_production", type=Path)
    parser.add_argument("control_production", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.candidate.read_text(), args.q4_production.read_text(),
                           args.control_production.read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
