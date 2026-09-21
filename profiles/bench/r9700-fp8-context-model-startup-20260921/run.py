#!/usr/bin/env python3
"""Single real-model startup accounting check; not a timing admission."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tools.ppl import fp8_context_recovery as recovery
from tools.bench import run_ninfer_bench_matrix as bench
from tools.bench.prefill_chunk_authority import durable_create_json

spec = importlib.util.spec_from_file_location('recovery_campaign', REPO / 'profiles/bench/r9700-terminal-base-fp8-context-recovery-20260921/campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)

def main():
    output = ROOT / 'physical'
    if output.exists():
        raise ValueError('startup output already exists; do not overwrite or rerun')
    case = next(c for c in campaign.candidates() if c['weights_id'] == recovery.HYBRID and c['group'] == 16 and c['attention'] == 'dense')
    campaign.ready_build(case, ['ninfer_bench', 'ninfer_r9700_runtime_planner_qual'])
    report = output / 'model-c1.json'
    command = campaign.startup_command(case, {'authority': {'selected_prefill_chunk': 2048}}, report)
    artifact = bench.inspect_artifact(case['artifact'])
    inputs = {
        'command': command, 'artifact': artifact,
        'files': [recovery.identity(path) for path in (
            case['build'] / 'bench/ninfer_bench',
            case['build'] / 'src/ninfer_r9700_runtime_planner_qual',
            campaign.CORPUS, campaign.SELECTION,
            campaign.QUALIFICATION / 'receipt.json',
            campaign.PACKAGE / 'source-review.json',
            Path(__file__),
        )],
    }
    bench.require_auto_power_profile()
    bench.require_hip_pci_device(0)
    output.mkdir()
    durable_create_json(output / 'inputs.json', inputs)
    with (output / 'stdout.txt').open('x') as stdout, (output / 'stderr.txt').open('x') as stderr:
        code = subprocess.run(command, cwd=REPO, stdout=stdout, stderr=stderr).returncode
    durable_create_json(output / 'exit.json', {'returncode': code})
    if code:
        raise ValueError(f'startup process failed: {code}')
    for record in inputs['files']:
        recovery.checked_file(record)
    bench.require_auto_power_profile()
    value = json.loads(report.read_text())
    bench.validate_automatic_feasibility(value)
    memory = value['memory']
    if memory['available_after_startup_bytes'] < memory['planned_slack_bytes']:
        raise ValueError('actual startup free memory is below planned slack; capacity remains blocked')
    durable_create_json(output / 'result.json', {
        'artifact_type': 'ninfer_r9700_fp8_context_model_startup', 'schema_version': 1,
        'pass': True, 'report': recovery.identity(report),
        'inputs': recovery.identity(output / 'inputs.json'), 'memory': memory,
        'scope': 'first real hybrid C1 startup accounting, not a capacity matrix or speed result',
    })
    print(json.dumps(memory, indent=2))

if __name__ == '__main__':
    main()
