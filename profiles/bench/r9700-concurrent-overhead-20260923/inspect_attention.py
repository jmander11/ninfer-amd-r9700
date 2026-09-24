"""Check paired QK's native FP8 WMMA and unchanged softmax/PV device bodies."""
import re
import subprocess
from pathlib import Path
from tools.bench.extract_embedded_code_object import extract
from tools.ppl.compare_nvfp4 import write_json
here=Path(__file__).resolve().parent
root=here.parents[2]
names=['qk_wmma_batched_dflash_verify_kernel',
       'softmax_wmma_scores_batched_dflash_verify_in_place_kernel',
       'pv_vector_batched_dflash_verify_kernel']
def inspect(binary,label):
    obj=here/f'{label}.hsaco'
    receipt=extract(binary,obj,code_symbol=names[0])
    notes=subprocess.check_output(['/opt/rocm/llvm/bin/llvm-readelf','--notes',obj],text=True)
    isa=subprocess.check_output(['/opt/rocm/llvm/bin/llvm-objdump','-d',obj],text=True)
    result={}
    for note in notes.split('  - .args:')[1:]:
        symbol=re.search(r'\.name:\s+(\S+)',note)[1]
        name=next((n for n in names if n in symbol),None)
        if name is None:continue
        body=isa.split('<'+symbol+'>:',1)[1].split('\n\n',1)[0]
        instructions=[l.split('//')[0].strip() for l in body.splitlines() if l.strip() and l.strip()!='...']
        fields={k:int(re.search(r'\.'+k+r':\s+(\d+)',note)[1]) for k in
                ['vgpr_count','sgpr_count','group_segment_fixed_size','private_segment_fixed_size','wavefront_size']}
        assert name not in result
        result[name]=(fields,instructions)
    assert len(result)==3
    return receipt,result
before,old=inspect(here/'projection-bin/bench','attention-baseline')
after,new=inspect(root/'build-r9700/bench/ninfer_bench','attention-candidate')
for n in names[1:]:assert old[n]==new[n],(n,'unchanged stage changed')
fields,inst=new[names[0]]
assert fields['wavefront_size']==32 and fields['private_segment_fixed_size']==0
assert fields['group_segment_fixed_size']==0
sites=sum('v_wmma_f32_16x16x16_fp8_fp8' in l for l in inst)
assert sites>0
write_json(here/'attention-isa.json',dict(baseline=before,candidate=after,
    qk_before=old[names[0]][0],qk_after=fields,fp8_wmma_sites=sites,
    unchanged_instruction_exact=names[1:]))
print(fields,'native FP8 sites',sites,'softmax/PV unchanged')
