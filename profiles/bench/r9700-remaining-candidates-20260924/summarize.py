"""Join qualified Ops and completed whole/cold tests; never launch a GPU job."""
import json,statistics
from pathlib import Path
from tools.ppl.compare_nvfp4 import write_json
here=Path(__file__).resolve().parent
prior=here.parent/'r9700-concurrent-overhead-20260923'
def read(path):return json.loads(path.read_text())
def paired_cells(old,new,fields):
    result=[]
    for a,b in zip(read(old)['cells'],read(new)['cells']):
        identity={k:b[k] for k in fields};assert all(a[k]==v for k,v in identity.items())
        result.append(dict(**identity,**{key:dict(baseline=statistics.median(a[key]),
            selected=statistics.median(b[key]),saving_fraction=1-statistics.median(b[key])/statistics.median(a[key]))
            for key in ['warm_ms','scrub_cold_ms']}))
    return result
whole=[]
for c,k,adaptive in [(1,5,False),(2,4,False),(2,5,False),(3,4,False),(3,5,False),
                      (4,4,False),(4,5,False),(4,5,True)]:
    prior_name=f'final-c{c}-'+('adaptive' if adaptive else f'k{k}')
    name=prior_name if c==1 else prior_name.replace('final-','selected-')
    s=read(here/name/'summary.json')
    assert s['timing_eligible'] and s['all_repetitions_exact_ordinary']
    assert read(here/name/'receipt.json')['exit_code']==0
    fresh=here/f'control-c{c}-k{k}'
    old=fresh if (fresh/'summary.json').exists() and not adaptive else prior/prior_name
    row=dict(concurrency=c,draft=k,adaptive=adaptive,aggregate_tok_s=s['aggregate_tok_s'],
             per_request_tok_s=s['aggregate_tok_s']/c,repetition_rates=s['repetition_rates'])
    if (old/'summary.json').exists():
        base=read(old/'summary.json');row['baseline_tok_s']=base['aggregate_tok_s']
        row['gain_fraction']=s['aggregate_tok_s']/base['aggregate_tok_s']-1
        row['baseline_kind']='fresh' if old==fresh else 'retained_previous_pass'
        if not adaptive:
            a=read(old/'report.json')['tests'][0]['reps'][0]
            for b in read(here/name/'report.json')['tests'][0]['reps']:
                assert a['generated_token_ids_by_lane']==b['generated_token_ids_by_lane']
                assert a['speculative']['accepted_tokens']==b['speculative']['accepted_tokens']
                assert a['speculative']['rounds']==b['speculative']['rounds']
    whole.append(row)
gdn_whole=[]
for c,k in [(1,5),(4,4)]:
    a=read(here/f'gdn-control-c{c}-k{k}/summary.json')
    b=read(here/f'final-c{c}-k{k}/summary.json')
    assert max(a['repetition_rates'])<min(b['repetition_rates'])
    gdn_whole.append(dict(concurrency=c,draft=k,baseline_tok_s=a['aggregate_tok_s'],
        selected_tok_s=b['aggregate_tok_s'],gain_fraction=b['aggregate_tok_s']/a['aggregate_tok_s']-1))
tails=read(here/'tail-selected-summary.json');assert sum(x['cases'] for x in tails['results'] if x['exact_ordinary'])==44
gateup=[]
for a,b in zip(read(here/'gateup-timing-baseline.json')['cells'],read(here/'gateup-timing-candidate.json')['cells']):
    assert a['tokens']==b['tokens']
    old=statistics.median(x['selected_ms'] for x in a['samples'])
    new=statistics.median(x['selected_ms'] for x in b['samples'])
    assert new<old
    gateup.append(dict(tokens=b['tokens'],cold_baseline_ms=old,cold_selected_ms=new,saving_fraction=1-new/old))
assert read(here/'gateup-production-isa.json')['kernels']==read(here/'gateup-final-isa.json')['kernels']
q=read(here/'gateup-qualification.json')['cells'];assert [c['tokens'] for c in q]==[15,18,20,24]
report=dict(status='complete',workload='R9700 gfx1201 auto P4096/G128 chunk2048; warmup1/reps3',
    weights_precision_unchanged=True,ordinary_and_prefill_not_remeasured=True,
    whole=whole,gdn_isolated_whole=gdn_whole,gateup=gateup,
    gateup_canonical_cells=4,gateup_output_corruptions_rejected=sum(c['public_bf16_oracle']['output_corruptions_rejected'] for c in q),
    pv=paired_cells(here/'pv-boundary-baseline.json',here/'pv-boundary-candidate.json',['rows','context']),
    gdn=paired_cells(here/'gdn-baseline.json',here/'gdn-final.json',['concurrency','width']),
    gdn_batch_policy='exact-tree actual batch1/4; incumbent actual batch2/3',
    gdn_all_batch_qualification_cells=len(read(here/'gdn-batch-selected.json')['cells']),
    gdn_instruction_parity=read(here/'gdn-batch-isa-parity.json'),
    c1_measurement='retained unchanged qualified instruction body; concurrent modes remeasured after narrowing',
    final_exact_ordinary_repetitions=24,cold_exact_ordinary_cases=44,
    pv_rejected='broad W4/W5 pairing; retained only W6 G16 feature-fast at4096<=context<8192',
    six_pending_at_k4_observed=bool(tails['pending_six_at_k4_rows']))
write_json(here/'final-summary.json',report)
for row in whole:print(row)
