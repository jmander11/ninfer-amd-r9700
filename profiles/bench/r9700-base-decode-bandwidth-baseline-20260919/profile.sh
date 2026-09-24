#!/usr/bin/env bash
set -euo pipefail
root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$root/profiles/bench/r9700-base-decode-bandwidth-baseline-20260919"
result="$package/results-profile"
power=/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level
python3 "$package/preflight.py" --mode profile
plan="$package/bound-plan.json"
read_plan() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["path"])' "$plan" "$1"; }
bench=$(read_plan benchmark)
artifact=$(read_plan artifact)
corpus=$(read_plan corpus)
profiler=$(read_plan profiler)
avail=$(read_plan counter_preflight)
mkdir "$result"
current_after=
restore_auto() {
  printf '%s\n' auto | sudo tee "$power" >/dev/null
  test "$(cat "$power")" = auto
  if [[ -n "$current_after" ]]; then printf '%s\n' auto >"$current_after"; fi
}
trap restore_auto EXIT
sudo -v
test "$(cat "$power")" = auto
printf '%s\n' profile_standard | sudo tee "$power" >/dev/null
test "$(cat "$power")" = profile_standard
"$avail" --device 0 pmc-check SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_WAVE_CYCLES SQ_WAVES >"$result/counter-preflight.stdout" 2>"$result/counter-preflight.stderr"
printf '%s\n' auto | sudo tee "$power" >/dev/null
test "$(cat "$power")" = auto
current_after="$result/power-after.txt"
printf '%s\n' profile_standard | sudo tee "$power" >/dev/null
test "$(cat "$power")" = profile_standard
printf '%s\n' profile_standard >"$result/power-before.txt"
"$profiler" --selected-regions -f csv rocpd -d "$result/raw" -o base-decode-dot8-sq \
  --marker-trace --kernel-trace --kernel-include-regex 'a8q4g64_linear_decode_dot8_t1_kernel' \
  --pmc SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_WAVE_CYCLES SQ_WAVES -- \
  "$bench" --weights "$artifact" --corpus "$corpus" --device 0 --concurrency 1 \
  --whole-pg 8192,1 --prefill-chunk 4096 --kv-capacity workload --spec mtp --draft-tokens 0 \
  --retain-token-ids --output json --output-file "$result/benchmark.json" -r 1 --warmup 1 --profile-measured \
  >"$result/benchmark.stdout" 2>"$result/benchmark.stderr"
restore_auto
current_after=
trap - EXIT
python3 - "$result" <<'PY'
import csv, hashlib, json, math, pathlib, sys
r=pathlib.Path(sys.argv[1])
report=json.load(open(r/'benchmark.json'))
assert report['environment']['gpu_name']=='AMD Radeon AI PRO R9700'
assert report['environment']['architecture_name']=='gfx1201'
assert report['config']['spec']=='none' and report['config']['draft_tokens']==0
assert report['tests'][0]['label']=='whole-pp8192+tg1'
databases=list((r/'raw').rglob('*_results.db'))
counters=list((r/'raw').rglob('*_counter_collection.csv'))
assert len(databases)==1 and databases[0].stat().st_size>0
assert len(counters)==1 and counters[0].stat().st_size>0
required={'SQ_WAIT_ANY','SQ_WAIT_INST_ANY','SQ_WAVE_CYCLES','SQ_WAVES'}
seen={name:0 for name in required}
with counters[0].open(newline='') as stream:
    rows=csv.DictReader(stream)
    expected_columns={'Kernel_Name','Counter_Name','Counter_Value'}
    assert expected_columns.issubset(rows.fieldnames or [])
    for row in rows:
        if 'a8q4g64_linear_decode_dot8_t1_kernel' not in row['Kernel_Name']:
            continue
        name=row['Counter_Name']
        if name in seen:
            value=float(row['Counter_Value'])
            assert math.isfinite(value) and value>=0
            seen[name]+=1
expected_dispatches=321
assert seen=={name:expected_dispatches for name in required}
def identity(path):
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path.resolve()),'bytes':path.stat().st_size,
            'sha256':digest}
summary={'schema':'ninfer.r9700.base-decode-dot8-sq-profile-summary.v1','status':'captured',
         'timing_admissible':False,'physical_hbm_bandwidth_gbps':None,'stall_freedom':None,
         'counters':['SQ_WAIT_ANY','SQ_WAIT_INST_ANY','SQ_WAVE_CYCLES','SQ_WAVES'],
         'expected_dot8_dispatches':expected_dispatches,
         'counter_rows_by_name':seen,
         'database':identity(databases[0]),'counter_csv':identity(counters[0]),
         'scope':'selected measured ordinary P8192 G1 dot8 dispatches only'}
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
files=sorted(p for p in r.rglob('*') if p.is_file() and p.name!='result.sha256')
records=[]
for p in files:
    with p.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    records.append(f'{digest}  {p.relative_to(r)}\n')
(r/'result.sha256').write_text(''.join(records))
PY
