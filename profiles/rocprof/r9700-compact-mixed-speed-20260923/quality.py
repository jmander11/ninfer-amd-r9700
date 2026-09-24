"""Check changed composition against six retained mixed-profile NLL sidecars."""
import argparse
import json
from pathlib import Path
from tools.ppl.compare_nvfp4 import run, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--label', required=True)
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]/'profiles/bench/r9700-compact-mixed-delivery-20260923'
rows = []
for prefix in ['quality','decode-quality']:
    for sample in ['wiki','technical','code']:
        prior = SOURCE/f'{prefix}-selected-compact-mixed_a8-{sample}'
        cell = HERE/f'{args.label}-{prefix}-{sample}'
        command = json.loads((prior/'command.json').read_text())
        command[0] = str(args.binary.resolve())
        command[command.index('--out-json')+1] = str(cell/'report.json')
        run(command, cell)
        exact = (cell/'report.nllf32').read_bytes() == (prior/'report.nllf32').read_bytes()
        report = json.loads((cell/'report.json').read_text())
        rows.append(dict(schedule=prefix, sample=sample, ppl=report['ppl'], exact=exact))
        print(rows[-1], flush=True)
        assert exact
write_json(HERE/f'{args.label}-quality.json', dict(status='PASS', rows=rows))
