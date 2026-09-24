"""Extract a named gfx1201 body safely; static instruction sites are not issue counters."""
import argparse,re,subprocess
from pathlib import Path
from tools.bench.extract_embedded_code_object import extract
from tools.ppl.compare_nvfp4 import write_json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('binary',type=Path);p.add_argument('symbol');p.add_argument('label')
a=p.parse_args();here=Path(__file__).resolve().parent
obj=here/(a.label+'.hsaco');receipt=extract(a.binary,obj,code_symbol=a.symbol)
notes=subprocess.check_output(['/opt/rocm/llvm/bin/llvm-readelf','--notes',obj],text=True)
isa=subprocess.check_output(['/opt/rocm/llvm/bin/llvm-objdump','-d',obj],text=True)
rows=[]
for note in notes.split('  - .args:')[1:]:
    symbol=re.search(r'\.name:\s+(\S+)',note)[1]
    if a.symbol not in symbol:continue
    body=isa.split('<'+symbol+'>:',1)[1].split('\n\n',1)[0]
    instructions=[line.split('//')[0].strip() for line in body.splitlines()
                  if line.strip() and line.strip()!='...']
    fields={k:int(re.search(r'\.'+k+r':\s+(\d+)',note)[1]) for k in
        ['vgpr_count','sgpr_count','group_segment_fixed_size','private_segment_fixed_size','wavefront_size']}
    rows.append(dict(symbol=symbol,**fields,instruction_sites={
        s:sum(s in line for line in instructions) for s in
        ['v_wmma','s_barrier','v_add_f32','v_mul_f32','v_fma_f32','s_wait_loadcnt']},
        instructions=instructions))
assert rows
write_json(here/(a.label+'.json'),dict(extraction=receipt,kernels=rows))
for row in rows:print({k:v for k,v in row.items() if k!='instructions'})
