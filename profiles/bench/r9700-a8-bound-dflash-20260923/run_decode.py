"""One serial, fixed-model decode cell with actual draft-K telemetry."""
import argparse
import json
import statistics
from pathlib import Path
from tools.ppl.compare_nvfp4 import run, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--concurrency', type=int, choices=[1, 2, 3, 4], required=True)
parser.add_argument('--draft', type=int, choices=[4, 5], required=True)
parser.add_argument('--adaptive', action='store_true')
parser.add_argument('--summarize-existing', action='store_true',
                    help='finish analysis of a successful retained run without retiming')
args = parser.parse_args()
if args.adaptive and args.draft != 5:
    parser.error('adaptive measurement uses maxK5')
pci = Path('/sys/bus/pci/devices/0000:13:00.0')
assert (pci/'vendor').read_text().strip() == '0x1002'
assert (pci/'device').read_text().strip() == '0x7551'
assert (pci/'power_dpm_force_performance_level').read_text().strip() == 'auto'
prior = ROOT/'profiles/rocprof/r9700-compact-mixed-speed-20260923'
command = json.loads((prior/'final-c4-k5/command.json').read_text())
cell = HERE/args.label
command[0] = str(args.binary.resolve())
for flag, value in [('--concurrency', args.concurrency), ('--draft-tokens', args.draft),
                    ('--output-file', cell/'report.json')]:
    command[command.index(flag)+1] = str(value)
if args.adaptive:
    command.append('--adaptive-draft')
if args.summarize_existing:
    assert json.loads((cell/'receipt.json').read_text())['exit_code'] == 0
    assert json.loads((cell/'command.json').read_text()) == command
else:
    run(command, cell)
report = json.loads((cell/'report.json').read_text())
reps = report['tests'][0]['reps']
ordinary_path = (ROOT/'profiles/bench/r9700-compact-followup-20260923/baseline-c1-k0/report.json'
                 if args.concurrency == 1 else prior/f'final-c{args.concurrency}-k0/report.json')
ordinary = json.loads(ordinary_path.read_text())
tokens = ordinary['tests'][0]['reps'][0]['generated_token_ids_by_lane']
assert all(r['generated_token_ids_by_lane'] == tokens for r in reps)
histograms = [r['speculative']['rounds_per_draft'] for r in reps]
assert all(sum(h) == r['speculative']['rounds'] for h, r in zip(histograms, reps))
rates = [args.concurrency*r['decode_output_tokens']/r['timings']['decode_seconds'] for r in reps]
summary = dict(concurrency=args.concurrency, max_draft=args.draft, adaptive=args.adaptive,
    aggregate_decode_tok_s=statistics.median(rates), repetition_rates=rates,
    per_request_tok_s=statistics.median(rates)/args.concurrency,
    request_lane_rounds_per_draft=histograms, all_repetitions_exact_ordinary=True)
assert (pci/'power_dpm_force_performance_level').read_text().strip() == 'auto'
if not args.summarize_existing:
    group = Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
    summary['resources'] = {key:(group/key).read_text() for key in ['memory.events','memory.peak','cpu.stat']}
    events = dict(line.split() for line in summary['resources']['memory.events'].splitlines())
    assert all(int(events[key]) == 0 for key in ['high','max','oom','oom_kill'])
    assert int(dict(line.split() for line in summary['resources']['cpu.stat'].splitlines()).get('nr_throttled',0)) == 0
else:
    summary['resources'] = 'scope counters not retained; completed measurement recovered without retiming'
write_json(cell/'summary.json', summary)
print(json.dumps(summary), flush=True)
