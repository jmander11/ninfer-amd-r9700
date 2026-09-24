"""Rerun the exact failing cases after W4 qualification; no timing admission."""
import json
import os
from pathlib import Path
from tools.ppl.compare_nvfp4 import run, write_json

here=Path(__file__).resolve().parent
root=here.parents[2]
artifact='/ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-q4-fp8-selective-cap/qwen3.8-27b-r9700-q4-fp8-selective-cap-n16k16-dflash2-q4-eval.ninfer'
trace=here/'tail-w4-decisions.json'
results=[]
for mode,stride in [('fixed3',1),('graph',1),('graph',17),('eager',1)]:
    name=f'tail-w4-{mode}-s{stride}'
    if mode=='eager': os.environ['NINFER_DFLASH_DECISION_TRACE_OUT']=str(trace)
    run([here/'tail-check-w4',artifact,root/'profiles/bench/r9700-compact-mixed-delivery-20260923/code.ids',
         mode,here/(name+'.json'),str(stride)],here/(name+'-run'))
    actual=json.loads((here/(name+'.json')).read_text())['cases']
    prior='tail-aligned-transitions' if stride==1 else 'tail-transitions'
    ordinary=json.loads((here/prior/'tail-ordinary.json').read_text())['cases']
    assert actual==ordinary, f'{name} differs from ordinary'
    results.append(dict(mode=mode,stride=stride,cases=len(actual),exact_ordinary=True))
    print(name,'PASS exact ordinary tokens and limits',flush=True)
rows=[e for e in json.loads(trace.read_text())['events'] if e['kind']=='dflash']
padded=[e for e in rows if e['physical_verify_width']-1>e['proposal_extent']]
pending=[e for e in rows if e['physical_verify_width']==5 and e['base_frontier']-e['context_frontier']==6]
zero=[e for e in rows if e['proposal_extent']==0]
for e in rows:
    assert 1<=len(e['licensed_tokens'])<=e['proposal_extent']+1
    assert e['next_frontier']==e['base_frontier']+len(e['licensed_tokens'])
assert padded and zero
write_json(here/'tail-w4-summary.json',dict(results=results,eager_round_rows=len(rows),
    padded_rows=len(padded),zero_extent_rows=len(zero),pending_six_at_k4_rows=len(pending),
    timing_eligible=False))
print('transition counts',len(padded),len(zero),len(pending),flush=True)
