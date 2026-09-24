"""Inspect new output projections and preserve all previously qualified instructions."""
import re
import subprocess
from pathlib import Path
from tools.ppl.compare_nvfp4 import write_json

HERE = Path(__file__).resolve().parent

def load(filename):
    obj = HERE / filename
    notes = subprocess.check_output(['/opt/rocm/llvm/bin/llvm-readelf', '--notes', obj], text=True)
    isa = subprocess.check_output(['/opt/rocm/llvm/bin/llvm-objdump', '-d', obj], text=True)
    result = {}
    for note in notes.split('  - .args:')[1:]:
        name = re.search(r'\.name:\s+(\S+)', note)[1]
        shape = tuple(map(int, re.search(r'kernelILj(\d+)ELj(\d+)ELj(\d+)EEE', name).groups()))
        body = isa.split('<' + name + '>:', 1)[1].split('\n\n', 1)[0]
        instructions = [line.split('//')[0].strip() for line in body.splitlines() if line.strip()]
        fields = {key: int(re.search(r'\.' + key + r':\s+(\d+)', note)[1]) for key in
                  ['vgpr_count', 'sgpr_count', 'group_segment_fixed_size',
                   'private_segment_fixed_size', 'wavefront_size']}
        result[shape] = (fields, instructions)
    return result

old = load(HERE.parents[2] / 'profiles/rocprof/r9700-compact-mixed-speed-20260923/final-candidate.hsaco')
current = load('k4-candidate.hsaco')
for shape, original in old.items():
    assert current[shape] == original, (shape, 'retained instructions/resources changed')
rows = []
for n,k in [(34816,5120),(5120,17408)]:
    for t in [10,15,20]:
        shape=(n,k,t)
        fields,instructions=current[shape]
        sites=sum('v_wmma_i32_16x16x32_iu4' in line for line in instructions)
        assert sites >= 4 and fields['wavefront_size'] == 32
        assert fields['group_segment_fixed_size'] == fields['private_segment_fixed_size'] == 0
        rows.append(dict(shape=shape,**fields,iu4_sites=sites))
write_json(HERE/'k4-isa.json',dict(rows=rows,retained_instruction_exact=list(old)))
print(rows)
print('Unchanged retained routes:',len(old))
