"""Join completed, exact-token admission reports without rerunning measurements."""
import json
import statistics
from pathlib import Path
from tools.ppl.compare_nvfp4 import write_json
here=Path(__file__).resolve().parent
def read(path):return json.loads((here/path).read_text())
rows=[]
for c,k,adaptive in [(1,5,False),(2,4,False),(3,4,False),(4,4,False),(4,5,False),(4,5,True)]:
    label=f'final-c{c}-'+('adaptive' if adaptive else f'k{k}')
    summary=read(label+'/summary.json')
    assert summary['timing_eligible'] and summary['all_repetitions_exact_ordinary']
    assert read(label+'/receipt.json')['exit_code']==0
    result=dict(concurrency=c,draft=k,adaptive=adaptive,
        aggregate_tok_s=summary['aggregate_tok_s'],per_request_tok_s=summary['aggregate_tok_s']/c,
        repetition_rates=summary['repetition_rates'])
    if not adaptive:
        projection=read(f'projection-c{c}-k{k}/summary.json')['aggregate_tok_s']
        result.update(projection_only_tok_s=projection,paired_qk_gain_fraction=summary['aggregate_tok_s']/projection-1)
        before_label=f'control-c{c}-k{k}' if c>1 else f'projection-c{c}-k{k}'
        before=read(before_label+'/summary.json')['aggregate_tok_s']
        result.update(control_tok_s=before,total_gain_fraction=summary['aggregate_tok_s']/before-1)
        reference=read(before_label+'/report.json')['tests'][0]['reps'][0]
        reps=read(label+'/report.json')['tests'][0]['reps']
        for r in reps:
            assert r['generated_token_ids_by_lane']==reference['generated_token_ids_by_lane']
            assert r['speculative']['rounds']==reference['speculative']['rounds']
            assert r['speculative']['accepted_tokens']==reference['speculative']['accepted_tokens']
    rows.append(result)
old=read('attention-baseline.json')['cells'];new=read('attention-candidate.json')['cells']
attention=[]
for a,b in zip(old,new):
    assert (a['rows'],a['context'])==(b['rows'],b['context'])
    assert b['serial_exact'] and b['graph_exact'] and b['guards']
    attention.append(dict(rows=b['rows'],context=b['context'],**{
        name:dict(before=statistics.median(a[name]),after=statistics.median(b[name]),
                  saving_fraction=1-statistics.median(b[name])/statistics.median(a[name]))
        for name in ['warm_ms','scrub_cold_ms']}))
tails=read('tail-paired-summary.json')
assert sum(r['cases'] for r in tails['results'] if r['exact_ordinary'])==44
q=read('projection-qualification.json')
assert len(q['cells'])==48 and q['status']=='qualified'
write_json(here/'final-summary.json',dict(status='complete',
    workload='R9700 gfx1201 auto P4096/G128 chunk2048; warmup1 repetitions3',
    weights_precision_cache_unchanged=True,whole=rows,attention=attention,
    projection_cells_pass=48,projection_new_routes=45,
    rejected_output_corruptions=sum(c['public_bf16_oracle']['output_corruptions_rejected'] for c in q['cells']),
    cold_engine_cases_exact_ordinary=44,all_18_final_repetitions_exact_ordinary=True,
    fixed_draft_rounds_and_acceptance_unchanged=True,
    six_pending_at_k4_observed=bool(tails['pending_six_at_k4_rows']),
    focused_ctest='4 passed',static_unittest='2 passed',prefill='paused; not changed or remeasured'))
for r in rows:print(r)
