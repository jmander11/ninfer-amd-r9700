"""Serial fixed-model continuation measurements; profiler results are attribution only."""
import argparse
import json
import statistics
from pathlib import Path
from tools.ppl.compare_nvfp4 import run, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, default=ROOT/'build-r9700/bench/ninfer_bench')
p.add_argument('--label', required=True)
p.add_argument('--concurrency', type=int, choices=[1,2,3,4], required=True)
p.add_argument('--draft', type=int, choices=[4,5], required=True)
p.add_argument('--trace', action='store_true')
p.add_argument('--adaptive', action='store_true')
a = p.parse_args()
if a.adaptive and a.draft != 5:
    p.error('adaptive measurements use maximum draft5')
pci = Path('/sys/bus/pci/devices/0000:13:00.0')
assert (pci/'vendor').read_text().strip() == '0x1002'
assert (pci/'device').read_text().strip() == '0x7551'
assert (pci/'power_dpm_force_performance_level').read_text().strip() == 'auto'
prior = ROOT/'profiles/bench/r9700-dflash-projections-tail-20260923'
cmd = json.loads((prior/'final-c4-k4/command.json').read_text())
cmd[0] = str(a.binary.resolve())
cell = HERE/a.label
for flag,value in [('--concurrency',a.concurrency),('--draft-tokens',a.draft),
                   ('--output-file',cell/'report.json'),('-r',1 if a.trace else 3)]:
    cmd[cmd.index(flag)+1] = str(value)
if a.adaptive:
    cmd.append('--adaptive-draft')
if a.trace:
    cmd += ['--profile-measured']
    cmd = ['/opt/rocm/bin/rocprofv3','--selected-regions','-f','rocpd',
           '-d',str(cell/'raw'),'--marker-trace','--kernel-trace',
           '--memory-copy-trace','--hip-graph-trace','--hip-runtime-trace','--'] + cmd
run(cmd,cell)
report = json.loads((cell/'report.json').read_text())
reps = report['tests'][0]['reps']
ordinary_path = (ROOT/'profiles/bench/r9700-compact-followup-20260923/baseline-c1-k0/report.json'
    if a.concurrency == 1 else ROOT/f'profiles/rocprof/r9700-compact-mixed-speed-20260923/final-c{a.concurrency}-k0/report.json')
ordinary = json.loads(ordinary_path.read_text())['tests'][0]['reps'][0]['generated_token_ids_by_lane']
assert all(r['generated_token_ids_by_lane'] == ordinary for r in reps)
assert (pci/'power_dpm_force_performance_level').read_text().strip() == 'auto'
summary = dict(concurrency=a.concurrency,draft=a.draft,adaptive=a.adaptive,timing_eligible=not a.trace,
               all_repetitions_exact_ordinary=True)
if not a.trace:
    rates = [a.concurrency*r['decode_output_tokens']/r['timings']['decode_seconds'] for r in reps]
    summary.update(aggregate_tok_s=statistics.median(rates),repetition_rates=rates)
group = Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
summary['resources'] = {key:(group/key).read_text() for key in ['memory.events','memory.peak','cpu.stat']}
assert all(int(v)==0 for k,v in (line.split() for line in summary['resources']['memory.events'].splitlines()) if k in ['high','max','oom','oom_kill'])
write_json(cell/'summary.json',summary)
print(json.dumps(summary),flush=True)
