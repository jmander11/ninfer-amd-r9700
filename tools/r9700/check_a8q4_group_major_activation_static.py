#!/usr/bin/env python3
"""Static gate for the disconnected group-major A8G64 qualifier."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PREFILL = "a8q4_group_major_prefill"
WAVE = "a8q4_group_major_wave"
ORDINARY = "a8g64_group_major_quantize"
FUSED = "fused_silu_a8g64_group_major_quantize"


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def interval(text: str, symbol: str) -> str:
    start = re.search(rf"(?m)^\s*\.protected\s+{symbol}\b.*$", text)
    need(start is not None, f"missing symbol {symbol}")
    tail = text[start.end():]
    end = re.search(r"(?m)^\s*\.protected\s+\w+\b", tail)
    return text[start.start():start.end() + (end.start() if end else len(tail))]


def directive(body: str, name: str) -> int:
    match = re.search(rf"(?m)^\s*\.amdhsa_{name}\s+(\d+)\s*$", body)
    need(match is not None, f"missing {name}")
    return int(match.group(1))


def comment(body: str, name: str) -> int:
    match = re.search(rf"(?m)^; {name}:\s*(\d+)\s*$", body)
    need(match is not None, f"missing {name}")
    return int(match.group(1))


def yaml(text: str, symbol: str, name: str) -> int:
    blocks = text.split("  - .args:")
    hits = [b for b in blocks if re.search(rf"(?m)^\s*\.name:\s+{symbol}\s*$", b)]
    need(len(hits) == 1, f"metadata identity {symbol}")
    match = re.search(rf"(?m)^\s*\.{name}:\s*(\d+)\s*$", hits[0])
    need(match is not None, f"missing metadata {name}")
    return int(match.group(1))


def source_gate(source: str) -> None:
    need("static_assert(sizeof(Banks)==17152)" in source, "exact LDS source size")
    need("(static_cast<std::size_t>(g)*tokens+token)*32U+byte" in source,
         "code plane is not group-major [G,T,32]")
    need("static_cast<std::size_t>(g)*tokens+token" in source,
         "scale plane is not group-major [G,T]")
    need(source.count("gm_byte(g,token,word*4U,tokens)") == 1,
         "prefill activation address must be one common group-major word")
    need("gm_scale(g,m0+th,tokens)" in source, "prefill scale address")
    need("gm_scale(g,m,tokens)" in source, "wave scale address")
    need("quantize<false>" in source and "quantize<true>" in source,
         "ordinary and fused direct publishers")
    need("q0&15" in source and "(q0-l0)/16" in source,
         "exact signed-A8 low/high decomposition")
    need("__float2int_rn(a/s)" in source and "max(-127,min(127" in source,
         "RNE/clamp represented codec")
    need("float(hip_bfloat16((a/(1.0F+expf(-a)))*b))" in source,
         "fused SiLU BF16 represented boundary")
    need("float(low[nh][e]+16*high[nh][e])" in source,
         "exact low+16*high reconstruction")
    need("a.tokens==2048" in source and "a8q4_group_major_wave" in source,
         "prefill and T1/small-T dispatch")


def isa_gate(assembly: str, symbol: str, lds: int, maximum_vgpr: int,
             iu4_sites: int) -> tuple[int, str]:
    body = interval(assembly, symbol)
    vgpr = directive(body, "next_free_vgpr")
    need(vgpr <= maximum_vgpr, f"{symbol} VGPR {vgpr}>{maximum_vgpr}")
    need(directive(body, "group_segment_fixed_size") == lds, f"{symbol} LDS")
    need(directive(body, "private_segment_fixed_size") == 0, f"{symbol} private")
    need(comment(body, "ScratchSize") == 0, f"{symbol} scratch")
    need(comment(body, "Occupancy") == 16, f"{symbol} occupancy")
    need(yaml(assembly, symbol, "sgpr_spill_count") == 0, f"{symbol} SGPR spill")
    need(yaml(assembly, symbol, "vgpr_spill_count") == 0, f"{symbol} VGPR spill")
    ops = re.findall(r"(?m)^\s*v_wmma_i32_16x16x32_iu4\b.*$", body)
    need(len(ops) == iu4_sites, f"{symbol} IU4 site count")
    need(sum("neg_lo:[0,1,0]" in x for x in ops) == iu4_sites // 2,
         f"{symbol} low signedness")
    need(sum("neg_lo:[1,1,0]" in x for x in ops) == iu4_sites // 2,
         f"{symbol} high signedness")
    need("global_inv" not in body, f"{symbol} global invalidate")
    return vgpr, body


def check(source: str, assembly: str) -> tuple[int, int, int, int]:
    source_gate(source)
    pvgpr, prefill = isa_gate(assembly, PREFILL, 17152, 96, 8)
    wvgpr, wave = isa_gate(assembly, WAVE, 0, 96, 4)
    ovgpr, ordinary = isa_gate(assembly, ORDINARY, 0, 32, 0)
    fvgpr, fused = isa_gate(assembly, FUSED, 0, 32, 0)
    need(len(re.findall(r"(?m)^\s*s_barrier_signal\s+-1", prefill)) == 2 and
         len(re.findall(r"(?m)^\s*s_barrier_wait\s+-1", prefill)) == 2,
         "unchanged two-bank barrier topology")
    first_wmma = prefill.index("v_wmma_i32_16x16x32_iu4")
    loads = re.findall(r"(?m)^\s*global_load_b32\b.*$", prefill[:first_wmma])
    need(len(loads) >= 2, "two activation b32 sites were not emitted")
    need("global_load_b64" in prefill[:first_wmma], "unchanged packed-W b64 load")
    last_load = max(prefill.rfind("global_load_b32", 0, first_wmma),
                    prefill.rfind("global_load_b64", 0, first_wmma))
    need(prefill.find("s_wait_loadcnt 0x0", last_load, first_wmma) < 0,
         "successor payload drained before current IU4 work")
    need(prefill.find("s_wait_loadcnt", first_wmma) > first_wmma,
         "successor payload has no post-IU4 retirement")
    need(len(re.findall(r"(?m)^\s*global_load_b32\b", wave)) >= 4,
         "small-T low/high loads missing")
    return pvgpr,wvgpr,ovgpr,fvgpr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assembly", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    assembly = args.assembly.read_text()
    pvgpr,wvgpr,ovgpr,fvgpr=check(source,assembly)
    print(f"PASS layout=group-major-g64 code=[G,T,32] scales=[G,T] "
          f"activation_b32_sites=2 iu4_sites=8 lds=17152 prefill_vgpr={pvgpr} "
          f"wave_vgpr={wvgpr} ordinary_quantizer_vgpr={ovgpr} "
          f"fused_quantizer_vgpr={fvgpr} occupancy=16 scratch=0 spills=0")


if __name__ == "__main__":
    main()
