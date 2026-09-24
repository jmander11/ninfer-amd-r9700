"""Compare retained C3/C4 attribution; these are not performance-admission timings."""
import json
from pathlib import Path
from tools.ppl.compare_nvfp4 import write_json

here = Path(__file__).resolve().parent
families = {
    'gate_up': 'projection_kernel<34816u',
    'down': 'projection_kernel<5120u, 17408',
    'attention_pv': 'pv_vector_batched',
    'gdn_record': 'record_kernel<false>',
}
result = dict(scope='attribution_only', production_changed=False, concurrency={})
for concurrency, name in [(3, 'investigate-c3-attribution-20260924.json'),
                          (4, 'final-c4-attribution.json')]:
    trace = json.loads((here / name).read_text())
    graph = max(trace['families'], key=lambda g: g['rounds'])
    rows = {}
    for label, symbol in families.items():
        matches = [f for f in graph['families'] if symbol in f['name']]
        assert len(matches) == 1
        f = matches[0]
        rows[label] = dict(calls_per_round=f['calls'] / graph['rounds'],
                           us_per_call=1000 * f['kernel_ms'] / f['calls'],
                           ms_per_round=f['kernel_ms'] / graph['rounds'],
                           kernel_service_fraction=f['fraction'])
    report = json.loads((here / f'final-c{concurrency}-k4/report.json').read_text())
    spec = report['tests'][0]['reps'][0]['speculative']
    result['concurrency'][str(concurrency)] = dict(
        graph=graph['graph'], steady_rounds=graph['rounds'], families=rows,
        ms_kernel_service_per_round=graph['kernel_ms'] / graph['rounds'],
        unprofiled_acceptance_length=spec['acceptance_length'],
        unprofiled_acceptance_rate=spec['acceptance_rate'])
result['limitations'] = [
    'Distinct captures; profiler interception perturbs durations.',
    'C3 and C4 differ in request count and acceptance; not a pure tile A/B.',
    'Repeated weight instructions do not establish physical DRAM byte counts.',
    'No new candidate timing or stall/cache-counter capture in this investigation.',
]
write_json(here / 'remaining-investigation-20260924.json', result)
print(json.dumps(result, indent=2))
