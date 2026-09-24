#!/usr/bin/env python3
"""Current base capacity/whole continuation; no artifact materialization or cutover."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tools.bench.prefill_chunk_authority import durable_create_json, validate_prefill_chunk_authority
from tools.bench.validate_post_chunk_capacity_campaign import validate_campaign
from tools.ppl import pareto, run
from tools.ppl.quality_recovery_io import EXPECTED_AUTHORITIES, validate_authority_map
from tools.ppl.terminal_selection_io import publish, require_absent

PYTHON = Path('/home/battlefront/.local/bin/python3.11')
SELECTION = REPO / 'profiles/bench/prefill-chunk-selection-panel-attention-20260921.json'
FROZEN = REPO / 'profiles/bench/r9700-chunk-selection-panel-attention-20260921/inputs.json'
QUALITY = REPO / 'profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/quality-authorities-receipt-bound-n16k16.json'
CORPUS = REPO / 'bench/fixtures/bench_corpus.ids'
RUNNER = REPO / 'tools/bench/run_ninfer_bench_matrix.py'


def candidates():
    for quality_name, (weights_id, attention) in EXPECTED_AUTHORITIES.items():
        route = 'dense' if attention == 'dense' else 'xattention'
        for group in (16, 32):
            yield {
                'name': f'{quality_name.removesuffix("_QUALITY").lower()}-g{group}',
                'quality_name': quality_name, 'weights_id': weights_id,
                'attention': attention, 'group': group,
                'build': REPO / f'build-r9700-selection-panel-{route}-g{group}-20260921',
                'artifact': REPO / f'out/qwen3.8-27b-{weights_id}.ninfer',
            }


def identity(candidate):
    return [candidate['weights_id'], candidate['group'], candidate['attention']]


def matrix(candidate, stage):
    return PACKAGE / stage / candidate['name']


def preflight():
    # Missing quality publication is a read-only rejection, not a request to run quality.
    quality = validate_authority_map(QUALITY)
    authority, record = validate_prefill_chunk_authority(SELECTION)
    if quality['selected_prefill_chunk_authority'] != {
        key: authority[key] for key in ('path', 'sha256')
    } or quality['selected_prefill_chunk'] != authority['selected_prefill_chunk']:
        raise ValueError('quality publication differs from selected-chunk authority')
    frozen = json.loads(FROZEN.read_text())
    if (frozen.get('artifact_type') != 'ninfer_r9700_chunk_campaign_inputs'
            or frozen.get('schema_version') != 1):
        raise ValueError('unsupported frozen panel inputs')
    artifacts = {item['weights_id']: item for item in frozen['artifacts']}
    for quality_name, (weights_id, _attention) in EXPECTED_AUTHORITIES.items():
        expected = artifacts[weights_id]
        if quality['authorities'][quality_name]['artifact'] != {
            'weights_id': weights_id, 'sha256': expected['sha256'],
            'file_size_bytes': expected['bytes'], 'conversion_receipt': expected['conversion_receipt'],
        }:
            raise ValueError('quality artifact differs from frozen panel inputs')
    cases = list(candidates())
    for artifact in {case['artifact'] for case in cases}:
        actual = run.inspect_candidate_artifact(artifact)
        if actual != artifacts.get(actual['weights_id']):
            raise ValueError(f'artifact differs from frozen panel inputs: {artifact}')
    files = {CORPUS}
    for case in cases:
        files.update(case['build'] / name for name in (
            'bench/ninfer_bench', 'src/ninfer_r9700_runtime_planner_qual'))
    for path in files:
        if (not path.is_file() or (path != CORPUS and not os.access(path, os.X_OK))
                or run.file_sha256(path) != frozen['files'].get(str(path.relative_to(REPO)))):
            raise ValueError(f'file differs from frozen panel inputs: {path}')
    for source in record['sources']:
        expected = dict(artifacts[source['weights_id']])
        expected['file_size_bytes'] = expected.pop('bytes')
        case = next(case for case in cases if identity(case) == [
            source['weights_id'], source['kv_value_group'], source['xattention_profile']])
        bench = case['build'] / 'bench/ninfer_bench'
        if (source['artifact'] != expected
                or source['benchmark_executable']['path'] != str(bench)
                or source['benchmark_executable']['sha256'] != frozen['files'][str(bench.relative_to(REPO))]):
            raise ValueError('selection evidence differs from frozen panel inputs')
    return {
        'authority': authority, 'quality': quality,
        'bindings': {str(path): run.file_sha256(path) for path in (SELECTION, FROZEN, QUALITY)},
    }


def unchanged(state):
    if any(run.file_sha256(Path(path)) != digest for path, digest in state['bindings'].items()):
        raise ValueError('continuation input authority changed during stage')


def command(case, stage, state):
    args = [str(PYTHON), str(RUNNER), '--preset', f'pareto-{stage}', '--no-build',
            '--bench', str(case['build'] / 'bench/ninfer_bench'),
            '--weights', str(case['artifact']), '--corpus', str(CORPUS), '--device', '0',
            '--prefill-chunk', str(state['authority']['selected_prefill_chunk']),
            '--prefill-chunk-authority', str(SELECTION),
            '--expected-kv-value-group', str(case['group']),
            '--expected-q4-activation-bits', '8', '--expected-w8-activation-bits', '8',
            '--expected-fp8-qk-wmma', '1', '--expected-xattention-profile', case['attention'],
            '--output-dir', str(matrix(case, stage))]
    for concurrency in (1, 2, 3, 4):
        args += ['--concurrency', str(concurrency)]
    if stage == 'capacity':
        args += ['--require-post-chunk-capacity']
    if 'four-role' in case['weights_id']:
        args += ['--require-fp8-hybrid', '--hybrid-width-tool',
                 str(case['build'] / 'src/ninfer_r9700_runtime_planner_qual')]
    return args


def create_stage(stage, state):
    root = PACKAGE / stage
    require_absent([root])
    root.mkdir()
    durable_create_json(root / 'inputs.json', state['bindings'])
    return root


def require_stage_bindings(stage, state):
    if json.loads((PACKAGE / stage / 'inputs.json').read_text()) != state['bindings']:
        raise ValueError(f'{stage} input bindings differ from current authorities')


def execute(args, root, name):
    with (root / f'{name}.stdout.txt').open('x') as stdout, \
            (root / f'{name}.stderr.txt').open('x') as stderr:
        return subprocess.run(args, cwd=REPO, stdout=stdout, stderr=stderr).returncode


def validated_capacity():
    result = validate_campaign(SELECTION, [matrix(case, 'capacity') for case in candidates()], executed=True)
    # The validator returns tuple identities; its persisted JSON authority uses lists.
    # Normalize once here so publication, replay comparison and membership agree.
    return json.loads(json.dumps(result))


def capacity(state):
    root = create_stage('capacity', state)
    outcomes = []
    for case in candidates():
        unchanged(state)
        code = execute(command(case, 'capacity', state), root, case['name'])
        outcomes.append({'identity': identity(case), 'returncode': code})
    durable_create_json(root / 'runner-outcomes.json', outcomes)
    unchanged(state)
    # Valid retained capacity exclusions are evidence, not reasons to skip other profiles.
    result = validated_capacity()
    durable_create_json(root / 'validation.json', result)


def capacity_evidence(state):
    require_stage_bindings('capacity', state)
    recorded = json.loads((PACKAGE / 'capacity/validation.json').read_text())
    actual = validated_capacity()
    if actual != recorded:
        raise ValueError('capacity outcomes changed after validation')
    return actual


def whole_eligible(state):
    capacity_ids = capacity_evidence(state)['whole_eligible_identities']
    return [identity(case) for case in candidates()
            if identity(case) in capacity_ids
            and state['quality']['authorities'][case['quality_name']]
                ['eligibility_by_group'][str(case['group'])]]


def whole(state):
    eligible = whole_eligible(state)
    root = create_stage('whole', state)
    for case in candidates():
        if identity(case) not in eligible:
            continue
        unchanged(state)
        code = execute(command(case, 'whole', state), root, case['name'])
        if code:
            raise ValueError(f'whole matrix failed; retained outputs: {case["name"]} (exit {code})')
    unchanged(state)


def select(state):
    import controls
    controls.validate_completed()
    eligible = whole_eligible(state)
    require_stage_bindings('whole', state)
    root = create_stage('select', state)
    pending_input, final_input = root / 'input.pending.json', root / 'pareto-input.json'
    pending_result, final_result = root / 'result.pending.json', root / 'result.json'
    args = [str(PYTHON), str(REPO / 'tools/ppl/assemble_pareto.py'),
            '--require-xattention-dense-controls', '--prefill-chunk-selection', str(SELECTION),
            '--post-chunk-capacity-validation', str(PACKAGE / 'capacity/validation.json'),
            '--out', str(pending_input)]
    for case in candidates():
        quality = state['quality']['authorities'][case['quality_name']]['path']
        args += ['--candidate', case['name'], case['weights_id'], str(case['group']),
                 quality, str(matrix(case, 'capacity')),
                 str(matrix(case, 'whole')) if identity(case) in eligible else '-']
    code = execute(args, root, 'assemble')
    if code:
        raise ValueError(f'Pareto assembly failed; retained logs (exit {code})')
    result = pareto.classify(pareto.load_payload(pending_input.read_text()))
    result['pareto_input'] = {'path': str(final_input.resolve()), 'sha256': run.file_sha256(pending_input)}
    durable_create_json(pending_result, result)
    unchanged(state)
    publish(pending_input, final_input, pending_result, final_result,
            guard_path=SELECTION, guard_sha256=state['authority']['sha256'])
    print(f'terminal base selection: {final_result}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'capacity', 'whole', 'select'))
    args = parser.parse_args()
    state = preflight()
    if args.stage == 'preflight':
        print(f'read-only preflight passed; selected chunk {state["authority"]["selected_prefill_chunk"]}')
    else:
        {'capacity': capacity, 'whole': whole, 'select': select}[args.stage](state)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'terminal-base-campaign: {error}', file=sys.stderr)
        raise SystemExit(1)
