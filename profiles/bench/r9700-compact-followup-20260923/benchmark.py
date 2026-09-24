"""Serial fixed-model follow-up cells; explicit binary, C<=4, fresh outputs."""
import argparse
import json
import statistics
from pathlib import Path
from tools.ppl.compare_nvfp4 import run, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = ROOT / 'profiles/rocprof/r9700-compact-mixed-speed-20260923'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--concurrency', type=int, choices=range(1, 5), nargs='+', required=True)
parser.add_argument('--draft-tokens', type=int, choices=[0, 4, 5], nargs='+', default=[0, 5])
parser.add_argument('--chunk', type=int, choices=[1024, 2048, 4096], default=2048)
parser.add_argument('--prompt', type=int, choices=[4096, 8192], default=4096)
parser.add_argument('--corpus', type=Path, help='explicit alternate corpus for a matched prompt comparison')
parser.add_argument('--adaptive', action='store_true')
args = parser.parse_args()
if args.adaptive and args.draft_tokens != [5]:
    parser.error('adaptive requires --draft-tokens 5')
power = Path('/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level')
rows = []
for c in args.concurrency:
    for k in args.draft_tokens:
        assert power.read_text().strip() == 'auto'
        cell = HERE / f'{args.label}-c{c}-k{k}'
        command = json.loads((PRIOR / f'final-c4-k{5 if k else 0}' / 'command.json').read_text())
        command[0] = str(args.binary.resolve())
        for flag, value in [('--concurrency', c), ('--draft-tokens', k),
                            ('--whole-pg', f'{args.prompt},128'), ('--max-ctx', args.prompt + 144),
                            ('--prefill-chunk', args.chunk), ('--output-file', cell / 'report.json')]:
            command[command.index(flag) + 1] = str(value)
        if args.adaptive:
            command.append('--adaptive-draft')
        if args.corpus:
            command[command.index('--corpus') + 1] = str(args.corpus.resolve())
        run(command, cell)
        report = json.loads((cell / 'report.json').read_text())
        reps = report['tests'][0]['reps']
        reference = reps[0]['generated_token_ids_by_lane']
        assert all(r['generated_token_ids_by_lane'] == reference for r in reps)
        row = dict(concurrency=c, draft_tokens=k, adaptive=args.adaptive, chunk=args.chunk, prompt=args.prompt,
            prefill_tok_s=statistics.median(c * args.prompt / r['timings']['prefill_seconds'] for r in reps),
            aggregate_decode_tok_s=statistics.median(c * r['decode_output_tokens'] /
                r['timings']['decode_seconds'] for r in reps), exact_repeated_tokens=True)
        write_json(cell / 'summary.json', row)
        rows.append(row)
        print(json.dumps(row), flush=True)
        assert power.read_text().strip() == 'auto'
group = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
resources = {name: (group / name).read_text() for name in ['memory.events', 'memory.peak', 'cpu.stat']}
write_json(HERE / f'{args.label}-summary.json', dict(rows=rows, resources=resources))
events = dict(line.split() for line in resources['memory.events'].splitlines())
assert all(int(events[name]) == 0 for name in ['high', 'max', 'oom', 'oom_kill'])
cpu = dict(line.split() for line in resources['cpu.stat'].splitlines())
assert int(cpu.get('nr_throttled', 0)) == 0
