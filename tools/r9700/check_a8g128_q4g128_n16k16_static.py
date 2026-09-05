#!/usr/bin/env python3
"""Fail-closed static gate for the disconnected exact-G128 qualifier."""
from __future__ import annotations
import argparse
import re
from pathlib import Path

PREFILL = "a8g128_q4g128_n16k16_prefill"
WAVE = "a8g128_q4g128_n16k16_wave"

def need(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(why)

def interval(text: str, symbol: str) -> str:
    hit = re.search(rf"(?m)^\s*\.protected\s+{symbol}\b.*$", text)
    need(hit is not None, f"missing symbol {symbol}")
    end = re.search(r"(?m)^\s*\.protected\s+\w+\b", text[hit.end():])
    return text[hit.start():hit.end()+(end.start() if end else len(text)-hit.end())]

def yaml(text: str, symbol: str, field: str) -> int:
    blocks = text.split("  - .args:")
    match = [b for b in blocks if re.search(rf"(?m)^\s*\.name:\s+{symbol}\s*$", b)]
    need(len(match) == 1, f"metadata identity {symbol}")
    v = re.search(rf"(?m)^\s*\.{field}:\s*(\d+)\s*$", match[0])
    need(v is not None, f"missing {field} for {symbol}")
    return int(v.group(1))

def comment(body: str, name: str) -> int:
    v = re.search(rf"(?m)^; {name}:\s*(\d+)\s*$", body)
    need(v is not None, f"missing resource comment {name}")
    return int(v.group(1))

def directive(body: str, name: str) -> int:
    v = re.search(rf"(?m)^\s*\.amdhsa_{name}\s+(\d+)\s*$", body)
    need(v is not None, f"missing amdhsa {name}")
    return int(v.group(1))

def source_gate(src: str) -> None:
    need("static_assert(sizeof(PingPong) == 17152)" in src, "LDS source contract")
    need("(slab&1U)==0U && th<kM" in src and
         "(slab&1U)==0U && th<kN" in src, "one even-bank scale publication")
    need("Tile& scale_dst=tiles.bank[(slab/2U)&1U]" in src,
         "scale publication must use logical-G128 parity")
    scale = src.index("Tile& scale_bank=tiles.bank[(slab/2U)&1U]")
    consume = src.index("hfloat(scale_bank.activation_scales", scale)
    publish = src.index("if(has_next){publish_payload", consume)
    need(scale < consume < publish, "G128 scales must be consumed before successor publication")
    need("const unsigned slabs=groups*2U" in src, "two K64 slabs per G128")
    need("if((slab&1U)==0U)" in src and "if((slab&1U)!=0U)" in src,
         "pair accumulator/reset topology")
    need("(slab&1U)*4U+lp" in src, "eight K16 weight pairs per G128")

def isa_gate(text: str, symbol: str, lds: int, vgpr: int) -> None:
    body = interval(text, symbol)
    need(directive(body, "next_free_vgpr") <= vgpr, f"{symbol} VGPR")
    need(directive(body, "group_segment_fixed_size") == lds, f"{symbol} LDS")
    need(directive(body, "private_segment_fixed_size") == 0, f"{symbol} private")
    need(yaml(text, symbol, "sgpr_spill_count") == 0, f"{symbol} SGPR spills")
    need(yaml(text, symbol, "vgpr_spill_count") == 0, f"{symbol} VGPR spills")
    need(comment(body, "ScratchSize") == 0, f"{symbol} scratch")
    need(comment(body, "Occupancy") == 16, f"{symbol} occupancy")
    ops = re.findall(r"(?m)^\s*v_wmma_i32_16x16x32_iu4\b.*$", body)
    need(len(ops) == 8, f"{symbol}: expected 8 static/16 dynamic-per-G128 IU4 sites")
    need(sum("neg_lo:[0,1,0]" in x for x in ops) == 4, f"{symbol} low signedness")
    need(sum("neg_lo:[1,1,0]" in x for x in ops) == 4, f"{symbol} high signedness")
    need("global_inv" not in body, f"{symbol} global invalidate")

def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,required=True);ap.add_argument("--assembly",type=Path,required=True);a=ap.parse_args()
    src=a.source.read_text();asm=a.assembly.read_text();source_gate(src)
    isa_gate(asm,PREFILL,17152,96);isa_gate(asm,WAVE,0,96)
    pbody=interval(asm,PREFILL);wbody=interval(asm,WAVE)
    pre=interval(asm,PREFILL); first=pre.index("v_wmma_i32_16x16x32_iu4")
    need(re.search(r"global_load_b(32|64)",pre[:first]) is not None,"prefill prologue VMEM")
    # Loads for the successor slab are issued before the current slab's WMMAs;
    # a complete drain before the first WMMA would destroy the intended overlap.
    last_load=max(pre.rfind("global_load_b32",0,first),pre.rfind("global_load_b64",0,first))
    drained=pre.find("s_waitcnt vmcnt(0)",last_load,first)
    need(last_load>=0 and drained<0,"successor payload is drained before current WMMA")
    need(len(re.findall(r"(?m)^\s*s_barrier_signal\s+-1",pre)) == 2 and
         len(re.findall(r"(?m)^\s*s_barrier_wait\s+-1",pre)) == 2,
         "rolled two-bank publication barrier topology")
    print("PASS g128_pair=true k64_banks=2 scale_loads_per_g128=1 "
          "logical_group_scale_parity=true scale_read_before_opposite_parity_reuse=true prefill_vgpr<=96 lds=17152 "
          "wave_vgpr<=96 occupancy=16 scratch=0 spills=0 iu4_static=8 iu4_dynamic_per_g128=16 "
          f"prefill_vgpr={directive(pbody,'next_free_vgpr')} "
          f"prefill_lds={directive(pbody,'group_segment_fixed_size')} "
          f"wave_vgpr={directive(wbody,'next_free_vgpr')} wave_lds=0")

if __name__ == "__main__": main()
