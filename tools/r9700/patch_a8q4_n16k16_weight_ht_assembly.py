#!/usr/bin/env python3
"""Extract exact production HSACO and patch only two W-load policy fields."""
from __future__ import annotations
import argparse,re,struct
from pathlib import Path
SYMBOL="_ZN6ninfer3ops5r97006linear12_GLOBAL__N_133a8q4g64_linear_prefill_cta_kernelEPKhS5_PKtPKjS5_S7_P12hip_bfloat16jjjj"
POLICY=0x00280000
def executable_offset(elf:bytes,address:int)->int:
 if elf[:4]!=b"\x7fELF" or elf[4:6]!=b"\x02\x01":raise RuntimeError("ELF64 little-endian HSACO")
 phoff=struct.unpack_from("<Q",elf,32)[0];entsz,num=struct.unpack_from("<HH",elf,54)
 for i in range(num):
  p=phoff+i*entsz;typ,flags,off,vaddr,_,filesz,_=struct.unpack_from("<IIQQQQQ",elf,p)
  if typ==1 and flags&1 and vaddr<=address<vaddr+filesz:return off+address-vaddr
 raise RuntimeError("instruction address not in executable LOAD segment")
def extract(fatbin:bytes)->bytes:
 start=fatbin.find(b"\x7fELF")
 if start<0 or fatbin.find(b"\x7fELF",start+4)>=0:raise RuntimeError("exactly one embedded HSACO")
 return fatbin[start:]
def patch(fatbin:bytes,disasm:str)->tuple[bytes,bytes,list[int]]:
 base=extract(fatbin)
 m=re.search(rf"(?ms)^([0-9a-f]+) <{re.escape(SYMBOL)}>:\n(.*?)(?=^[0-9a-f]+ <[^>]+>:\n|\Z)",disasm)
 if not m:raise RuntimeError("exact production symbol")
 loads=[]
 for line in m.group(2).splitlines():
  if "global_load_b64" in line:
   q=re.search(r"// ([0-9A-F]+): ([0-9A-F]{8}) ([0-9A-F]{8}) ([0-9A-F]{8})",line)
   if not q:raise RuntimeError("W load encoding")
   loads.append(tuple(int(x,16) for x in q.groups()))
 if len(loads)!=2:raise RuntimeError("exact two production W b64 loads")
 out=bytearray(base);offsets=[]
 for address,w0,w1,w2 in loads:
  if w1&POLICY:raise RuntimeError("baseline W load already has policy bits")
  off=executable_offset(base,address);expected=struct.pack("<III",w0,w1,w2)
  if base[off:off+12]!=expected:raise RuntimeError("disassembly/ELF encoding mismatch")
  struct.pack_into("<I",out,off+4,w1|POLICY);offsets.append(off+4)
 changed=[i for i,(a,b) in enumerate(zip(base,out)) if a!=b]
 expected_changed=sorted(i for o in offsets for i in range(o,o+4) if base[i]!=out[i])
 if changed!=expected_changed:raise RuntimeError("unexpected byte delta")
 return base,bytes(out),offsets
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--fatbin",type=Path,required=True);p.add_argument("--disassembly",type=Path);p.add_argument("--baseline",type=Path,required=True);p.add_argument("--candidate",type=Path);a=p.parse_args()
 if a.candidate is None:
  if a.baseline.exists() or a.baseline.is_symlink():raise RuntimeError("baseline output must be absent")
  a.baseline.write_bytes(extract(a.fatbin.read_bytes()));return
 if a.disassembly is None or a.candidate.exists() or a.candidate.is_symlink():raise RuntimeError("candidate/disassembly contract")
 base,cand,_=patch(a.fatbin.read_bytes(),a.disassembly.read_text())
 if a.baseline.read_bytes()!=base:raise RuntimeError("baseline changed")
 a.candidate.write_bytes(cand)
if __name__=="__main__":main()
