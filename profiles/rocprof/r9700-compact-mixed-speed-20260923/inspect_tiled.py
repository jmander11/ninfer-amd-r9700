"""Check tiled resources and preservation of admitted small-width instructions."""
import re
import subprocess
from pathlib import Path
from tools.ppl.compare_nvfp4 import write_json
HERE=Path(__file__).resolve().parent
def load(name):
    obj=HERE/name
    notes=subprocess.check_output(['/opt/rocm/llvm/bin/llvm-readelf','--notes',obj],text=True)
    isa=subprocess.check_output(['/opt/rocm/llvm/bin/llvm-objdump','-d',obj],text=True)
    rows={}
    for note in notes.split('  - .args:')[1:]:
        symbol=re.search(r'\.name:\s+(\S+)',note)[1]
        key=tuple(map(int,re.search(r'kernelILj(\d+)ELj(\d+)ELj(\d+)EEE',symbol).groups()))
        body=isa.split('<'+symbol+'>:',1)[1].split('\n\n',1)[0]
        instructions=[line.split('//')[0].strip() for line in body.splitlines() if line.strip()]
        fields={k:int(re.search(r'\.'+k+r':\s+(\d+)',note)[1]) for k in
                ['vgpr_count','sgpr_count','group_segment_fixed_size','private_segment_fixed_size','wavefront_size']}
        rows[key]=(fields,instructions)
    return rows
old,new=load('mlp-candidate.hsaco'),load('tiled-candidate.hsaco')
retained=[]
for key,(fields,instructions) in new.items():
    if key[2]>8:continue
    assert fields==old[key][0], (key,'resources changed')
    assert instructions==old[key][1], (key,'instruction stream changed')
    retained.append(key)
rows=[]
for key,(fields,instructions) in new.items():
    if key[2]<=8:continue
    sites=sum('v_wmma_i32_16x16x32_iu4' in s for s in instructions)
    assert sites>=4 and fields['private_segment_fixed_size']==0 and fields['wavefront_size']==32
    rows.append(dict(shape=key,**fields,iu4_sites=sites))
assert len(rows)==6
write_json(HERE/'tiled-isa.json',dict(rows=rows,retained_instruction_exact=retained))
print(rows)
print('instruction-identical retained routes:',retained)
