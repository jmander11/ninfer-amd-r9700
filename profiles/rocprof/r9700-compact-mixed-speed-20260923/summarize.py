"""Recompute concurrency rates from raw reports (decode_output_tokens is per lane)."""
import argparse
import json
from pathlib import Path
import statistics
from tools.ppl.compare_nvfp4 import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]/'profiles/bench/r9700-compact-mixed-delivery-20260923'
rows = []
for c in range(1,5):
    ordinary = None
    for k in [0,5]:
        cell = HERE/f'{args.label}-c{c}-k{k}'
        if c == 1 and args.label == 'baseline': cell = SOURCE/f'whole-k{k}'
        report = json.loads((cell/'report.json').read_text())
        reps = report['tests'][0]['reps']
        if k == 0: ordinary = reps[0]['generated_token_ids_by_lane']
        assert all(r['generated_token_ids_by_lane'] == ordinary for r in reps)
        per_request = statistics.median(r['decode_output_tokens']/r['timings']['decode_seconds'] for r in reps)
        rows.append(dict(concurrency=c, draft_tokens=k, aggregate_decode_tok_s=c*per_request,
            per_request_decode_tok_s=per_request,
            prefill_tok_s=statistics.median(c*4096/r['timings']['prefill_seconds'] for r in reps),
            exact_same_c_ordinary=True, source=str(cell)))
write_json(args.output, dict(rows=rows, units='output tokens per second'))
print(json.dumps(rows, indent=2))
