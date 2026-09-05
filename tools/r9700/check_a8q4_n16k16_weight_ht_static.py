#!/usr/bin/env python3
"""Prove production HSACO changes only at two W-load policy fields."""
from __future__ import annotations
import argparse,re,struct
from pathlib import Path
from tools.r9700.patch_a8q4_n16k16_weight_ht_assembly import POLICY,SYMBOL,executable_offset
def need(x:bool,s:str)->None:
 if not x:raise RuntimeError(s)
def block(text:str)->str:
 m=re.search(rf"(?ms)^[0-9a-f]+ <{re.escape(SYMBOL)}>:\n(.*?)(?=^[0-9a-f]+ <[^>]+>:\n|\Z)",text);need(m is not None,"symbol");return m.group(1)
def loads(text:str)->list[tuple[int,int,int,int,str]]:
 out=[]
 for line in block(text).splitlines():
  if "global_load_b64" in line:
   m=re.search(r"// ([0-9A-F]+): ([0-9A-F]{8}) ([0-9A-F]{8}) ([0-9A-F]{8})",line);need(m is not None,"encoding")
   out.append((*[int(x,16) for x in m.groups()],line))
 return out
def meta(text:str,name:str)->int:
 m=re.search(rf"(?m)^\s*\.{name}:\s+(\d+)\s*$",text);need(m is not None,name);return int(m.group(1))
def check(base:bytes,cand:bytes,bd:str,cd:str,notes:str)->dict[str,int]:
 need(len(base)==len(cand),"HSACO size")
 bl,cl=loads(bd),loads(cd);need(len(bl)==len(cl)==2,"exact W b64 inventory")
 expected=[]
 for b,c in zip(bl,cl):
  ba,b0,b1,b2,_=b;ca,c0,c1,c2,line=c;need(ba==ca and (c0,c1,c2)==(b0,b1|POLICY,b2),"modifier encoding only")
  need("th:TH_LOAD_HT" in line and "scope:SCOPE_DEV" in line,"modifier spelling")
  off=executable_offset(base,ba);need(base[off:off+12]==struct.pack("<III",b0,b1,b2),"baseline bytes");need(cand[off:off+12]==struct.pack("<III",c0,c1,c2),"candidate bytes")
  expected.extend(i for i in range(off,off+12) if base[i]!=cand[i])
 need([i for i,(a,b) in enumerate(zip(base,cand)) if a!=b]==expected,"complete HSACO equality beyond policy bytes")
 cb=block(cd);need(sum("v_wmma_i32_16x16x32_iu4" in x for x in cb.splitlines())==8,"IU4")
 need(sum("global_load_b32" in x and (" th:" in x or " scope:" in x) for x in cb.splitlines())==0,"A modifier")
 hits=[x for x in notes.split("  - .args:") if re.search(rf"(?m)^\s*\.name:\s+{re.escape(SYMBOL)}\s*$",x)];need(len(hits)==1,"metadata symbol");scope=hits[0]
 vgpr=meta(scope,"vgpr_count");need(88<=vgpr<=96,"VGPR")
 need(meta(scope,"group_segment_fixed_size")==17152 and meta(scope,"private_segment_fixed_size")==0,"storage")
 need(meta(scope,"sgpr_spill_count")==0 and meta(scope,"vgpr_spill_count")==0,"spills")
 return {"vgpr":vgpr,"architectural_vgpr":((vgpr+7)//8)*8,"lds":17152,"iu4":8,"weight_ht_b64":2}
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--baseline",type=Path,required=True);p.add_argument("--candidate",type=Path,required=True);p.add_argument("--baseline-disassembly",type=Path,required=True);p.add_argument("--candidate-disassembly",type=Path,required=True);p.add_argument("--notes",type=Path,required=True);a=p.parse_args()
 r=check(a.baseline.read_bytes(),a.candidate.read_bytes(),a.baseline_disassembly.read_text(),a.candidate_disassembly.read_text(),a.notes.read_text())
 print("PASS "+" ".join(f"{k}={v}" for k,v in r.items())+" occupancy=16 scratch=0 spills=0 exact_hsaco_equality=true A_policy=default scale_policy=default")
if __name__=="__main__":main()
