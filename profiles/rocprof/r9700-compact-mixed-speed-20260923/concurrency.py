"""Serial, matched compact-profile C1-C4 timing with retained greedy tokens."""
import argparse
import json
from pathlib import Path
import statistics
from tools.ppl.compare_nvfp4 import run, write_json

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'profiles/bench/r9700-compact-mixed-delivery-20260923'
HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1,2,3,4])
    args = parser.parse_args()
    assert all(1 <= c <= 4 for c in args.concurrency)
    power = Path('/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level')
    group = Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
    rows = []
    for c in args.concurrency:
        ordinary = None
        for k in [0,5]:
            assert power.read_text().strip() == 'auto'
            cell = HERE/f'{args.label}-c{c}-k{k}'
            command = json.loads((SOURCE/f'whole-k{k}/command.json').read_text())
            command[0] = str(args.binary.resolve())
            command[command.index('--concurrency')+1] = str(c)
            command[command.index('--output-file')+1] = str(cell/'report.json')
            run(command, cell)
            report = json.loads((cell/'report.json').read_text())
            reps = report['tests'][0]['reps']
            if k == 0: ordinary = reps[0]['generated_token_ids_by_lane']
            exact = all(rep['generated_token_ids_by_lane'] == ordinary for rep in reps)
            row = dict(concurrency=c, draft_tokens=k, exact_ordinary_tokens=exact,
                aggregate_decode_tok_s=statistics.median(c*rep['decode_output_tokens']/rep['timings']['decode_seconds'] for rep in reps),
                per_request_decode_tok_s=statistics.median(rep['decode_output_tokens']/rep['timings']['decode_seconds'] for rep in reps),
                prefill_tok_s=statistics.median(c*4096/rep['timings']['prefill_seconds'] for rep in reps))
            rows.append(row)
            write_json(cell/'summary.json', row)
            print(json.dumps(row), flush=True)
            assert exact, 'concurrent speculative/repeated tokens differ from concurrent ordinary'
            assert power.read_text().strip() == 'auto'
    write_json(HERE/f'{args.label}-complete-summary.json', dict(rows=rows))
    resources = {n:(group/n).read_text() for n in ['memory.events','memory.peak','cpu.stat']}
    write_json(HERE/f'{args.label}-resources.json', resources)
    events = dict(line.split() for line in resources['memory.events'].splitlines())
    cpu = dict(line.split() for line in resources['cpu.stat'].splitlines())
    assert all(int(events[n]) == 0 for n in ['high','max','oom','oom_kill'])
    assert int(cpu.get('nr_throttled',0)) == 0

if __name__ == '__main__': main()
