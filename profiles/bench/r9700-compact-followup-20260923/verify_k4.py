"""Check all measured K4 outputs against same-C ordinary decode and summarize."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parents[2] / 'profiles/rocprof/r9700-compact-mixed-speed-20260923'
rows = []
for c in (2, 3, 4):
    reference = json.loads((PRIOR / f'final-c{c}-k0/report.json').read_text())
    tokens = reference['tests'][0]['reps'][0]['generated_token_ids_by_lane']
    rates = []
    for label in ('k4-baseline', 'k4-candidate'):
        cell = HERE / f'{label}-c{c}-k4'
        reps = json.loads((cell / 'report.json').read_text())['tests'][0]['reps']
        assert len(reps) == 3
        assert all(r['generated_token_ids_by_lane'] == tokens for r in reps)
        rates.append(json.loads((cell / 'summary.json').read_text())['aggregate_decode_tok_s'])
    rows.append(dict(concurrency=c, baseline_tok_s=rates[0], selected_tok_s=rates[1],
                     per_request_tok_s=rates[1] / c, gain_percent=100*(rates[1]/rates[0]-1),
                     all_repetitions_exact_ordinary_tokens=True))
print(json.dumps(dict(rows=rows), indent=2))
