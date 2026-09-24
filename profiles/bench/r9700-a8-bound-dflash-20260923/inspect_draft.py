"""Check the four new draft kernels and exact retained instruction streams."""
import re
import subprocess
from pathlib import Path
from tools.bench.extract_embedded_code_object import extract
from tools.ppl.compare_nvfp4 import write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def inspect(binary, label):
    obj = HERE/f'{label}.hsaco'
    receipt = extract(binary, obj, code_symbol='small_batch_projection_kernel')
    notes = subprocess.check_output(['/opt/rocm/llvm/bin/llvm-readelf', '--notes', obj], text=True)
    isa = subprocess.check_output(['/opt/rocm/llvm/bin/llvm-objdump', '-d', obj], text=True)
    result = {}
    for note in notes.split('  - .args:')[1:]:
        name = re.search(r'\.name:\s+(\S+)', note)[1]
        shape = tuple(map(int, re.search(r'kernelILj(\d+)ELj(\d+)ELj(\d+)EEE', name).groups()))
        body = isa.split('<'+name+'>:', 1)[1].split('\n\n', 1)[0]
        instructions = [line.split('//')[0].strip() for line in body.splitlines() if line.strip()]
        fields = {key:int(re.search(r'\.'+key+r':\s+(\d+)', note)[1]) for key in
                  ['vgpr_count','sgpr_count','group_segment_fixed_size',
                   'private_segment_fixed_size','wavefront_size']}
        result[shape] = (fields, instructions)
    return receipt, result

old_receipt, old = inspect(HERE/'baseline-bin/bench', 'draft-baseline')
new_receipt, new = inspect(ROOT/'build-r9700/bench/ninfer_bench', 'draft-candidate')
for shape, original in old.items():
    assert new[shape] == original, (shape, 'retained instructions/resources changed')
added = {(5120,k,t) for k in [17408,25600] for t in [5,6]}
assert set(new)-set(old) == added
rows = []
for shape in sorted(added):
    fields, instructions = new[shape]
    sites = sum('v_wmma_i32_16x16x32_iu4' in line for line in instructions)
    assert sites >= 4 and fields['wavefront_size'] == 32
    assert fields['group_segment_fixed_size'] == fields['private_segment_fixed_size'] == 0
    rows.append(dict(shape=shape, **fields, iu4_sites=sites))
write_json(HERE/'draft-isa.json', dict(rows=rows, retained_instruction_exact=list(old),
           baseline=old_receipt, candidate=new_receipt))
print(rows)
print('Retained instruction/resource streams exact:', len(old))
