"""Verify renamed dynamic-row attention retains its qualified device arithmetic."""
import re
import subprocess
from pathlib import Path
from tools.bench.extract_embedded_code_object import extract
from tools.ppl.compare_nvfp4 import write_json

here = Path(__file__).resolve().parent
root = here.parents[2]
pairs = [
    ('qk_wmma_batched_w5w6_kernel', 'qk_wmma_batched_dflash_verify_kernel'),
    ('softmax_wmma_scores_batched_w5w6_in_place_kernel', 'softmax_wmma_scores_batched_dflash_verify_in_place_kernel'),
    ('pv_vector_batched_w5w6_kernel', 'pv_vector_batched_dflash_verify_kernel'),
]
def inspect(binary, label, names):
    obj = here/f'{label}.hsaco'
    receipt = extract(binary, obj, code_symbol=names[0])
    notes = subprocess.check_output(['/opt/rocm/llvm/bin/llvm-readelf','--notes',obj],text=True)
    isa = subprocess.check_output(['/opt/rocm/llvm/bin/llvm-objdump','-d',obj],text=True)
    result = {}
    for note in notes.split('  - .args:')[1:]:
        name = re.search(r'\.name:\s+(\S+)',note)[1]
        selected = next((i for i,n in enumerate(names) if n in name),None)
        if selected is None: continue
        body = isa.split('<'+name+'>:',1)[1].split('\n\n',1)[0]
        instructions = [l.split('//')[0].strip() for l in body.splitlines() if l.strip() and l.strip()!='...']
        fields = {k:int(re.search(r'\.'+k+r':\s+(\d+)',note)[1]) for k in
                  ['vgpr_count','sgpr_count','group_segment_fixed_size','private_segment_fixed_size','wavefront_size']}
        assert selected not in result
        result[selected] = (fields,instructions)
    assert len(result)==3
    return receipt,result
old_receipt,old = inspect(root/'profiles/bench/r9700-a8-bound-dflash-20260923/final-bin/bench',
                          'attention-control',[p[0] for p in pairs])
new_receipt,new = inspect(root/'build-r9700/bench/ninfer_bench','attention-w4',[p[1] for p in pairs])
rows=[]
for i,(old_name,new_name) in enumerate(pairs):
    assert old[i]==new[i], (new_name,'device arithmetic/resources changed')
    fields,inst = new[i]
    assert fields['wavefront_size']==32 and fields['private_segment_fixed_size']==0
    sites = sum('v_wmma_f32_16x16x16_fp8_fp8' in l for l in inst)
    if i==0: assert sites>=1
    rows.append(dict(kernel=new_name,**fields,instructions=len(inst),fp8_wmma_sites=sites,old_instruction_exact=True))
write_json(here/'attention-isa.json',dict(rows=rows,control=old_receipt,candidate=new_receipt))
print('PASS unchanged dynamic-row attention ISA/resources',rows)
