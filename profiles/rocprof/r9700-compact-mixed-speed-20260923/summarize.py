"""Recompute concurrency rates from raw reports (decode_output_tokens is per lane)."""
import argparse
import json
from pathlib import Path
import statistics
from tools.ppl.compare_nvfp4 import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label', required=True)
parser.add_argument('--c1-label', help='reuse an explicitly unchanged C1 checkpoint')
parser.add_argument('--reference-label', help='also compare C2-C4 tokens with this prior checkpoint')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]/'profiles/bench/r9700-compact-mixed-delivery-20260923'
rows = []
for c in range(1,5):
    ordinary = None
    for k in [0,5]:
        label = args.c1_label if c == 1 and args.c1_label else args.label
        cell = HERE/f'{label}-c{c}-k{k}'
        if c == 1 and args.label == 'baseline': cell = SOURCE/f'whole-k{k}'
        report = json.loads((cell/'report.json').read_text())
        reps = report['tests'][0]['reps']
        if k == 0: ordinary = reps[0]['generated_token_ids_by_lane']
        assert all(r['generated_token_ids_by_lane'] == ordinary for r in reps)
        prior_exact = None
        if c > 1 and args.reference_label:
            prior = json.loads((HERE/f'{args.reference_label}-c{c}-k0/report.json').read_text())
            expected = prior['tests'][0]['reps'][0]['generated_token_ids_by_lane']
            prior_exact = all(r['generated_token_ids_by_lane'] == expected for r in reps)
            assert prior_exact, 'changed output versus prior same-C ordinary baseline'
        per_request = statistics.median(r['decode_output_tokens']/r['timings']['decode_seconds'] for r in reps)
        rows.append(dict(concurrency=c, draft_tokens=k, aggregate_decode_tok_s=c*per_request,
            per_request_decode_tok_s=per_request,
            prefill_tok_s=statistics.median(c*4096/r['timings']['prefill_seconds'] for r in reps),
            exact_same_c_ordinary=True, exact_prior_ordinary=prior_exact, source=str(cell)))
write_json(args.output, dict(rows=rows, units='output tokens per second'))
print(json.dumps(rows, indent=2))
