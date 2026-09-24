#!/usr/bin/env python3
"""Create-only shared FP8-context recovery; retain valid nonhybrid capacity evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tools.bench.prefill_chunk_authority import durable_create_json, validate_prefill_chunk_authority
from tools.bench.validate_post_chunk_capacity_campaign import validate_campaign, validate_matrix
from tools.ppl import pareto, run
from tools.ppl import fp8_context_recovery as recovery
from tools.ppl.quality_recovery_io import EXPECTED_AUTHORITIES, validate_authority_map
from tools.ppl.terminal_selection_io import publish, require_absent

PYTHON = Path('/home/battlefront/.local/bin/python3.11')
SELECTION = REPO / 'profiles/bench/prefill-chunk-selection-panel-attention-20260921.json'
FROZEN = REPO / 'profiles/bench/r9700-chunk-selection-panel-attention-20260921/inputs.json'
QUALITY = REPO / 'profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/quality-authorities-receipt-bound-n16k16.json'
CORPUS = REPO / 'bench/fixtures/bench_corpus.ids'
RUNNER = REPO / 'tools/bench/run_ninfer_bench_matrix.py'
OLD = REPO / 'profiles/bench/r9700-terminal-base-panel-attention-20260921'
BRIDGE = PACKAGE / 'resource-recovery.json'
QUALIFICATION = PACKAGE / 'qualification-accounted'
STARTUP = PACKAGE / 'startup-accounted'
CLOSURE = OLD / 'capacity/closure.json'


def candidates():
    for quality_name, (weights_id, attention) in EXPECTED_AUTHORITIES.items():
        route = 'dense' if attention == 'dense' else 'xattention'
        for group in (16, 32):
            old_build = REPO / f'build-r9700-selection-panel-{route}-g{group}-20260921'
            recovery_route = '' if attention == 'dense' else '-xattention'
            build = (REPO / f'build-r9700-fp8-accounted{recovery_route}-g{group}-20260921'
                     if weights_id == recovery.HYBRID else old_build)
            yield {
                'name': f'{quality_name.removesuffix("_QUALITY").lower()}-g{group}',
                'quality_name': quality_name, 'weights_id': weights_id,
                'attention': attention, 'group': group,
                'build': build, 'old_build': old_build,
                'artifact': REPO / f'out/qwen3.8-27b-{weights_id}.ninfer',
            }


def identity(candidate):
    return [candidate['weights_id'], candidate['group'], candidate['attention']]


def matrix(candidate, stage):
    root = OLD if stage == 'capacity' and candidate['weights_id'] != recovery.HYBRID else PACKAGE
    return root / stage / candidate['name']


def ready_build(case, targets):
    old = case['old_build'] / 'CMakeCache.txt'
    new = case['build'] / 'CMakeCache.txt'
    if recovery.cache_profile(new) != recovery.cache_profile(old):
        raise ValueError('recovery build changed compiled arithmetic/profile options')
    pending = subprocess.run(['cmake', '--build', str(case['build']), '--target', *targets,
                              '--', '-n'], check=True, capture_output=True, text=True)
    if pending.stdout.strip() != 'ninja: no work to do.':
        raise ValueError('recovery build is stale; complete explicit compile-only targets')
    for target in targets:
        path = case['build'] / ('bench' if target == 'ninfer_bench' else 'src') / target
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f'recovery executable is absent: {path}')


def qualify():
    """Root-invoked GPU stage; freeze exact launch inputs before the only process."""
    from tools.bench import run_ninfer_bench_matrix as bench
    case = next(case for case in candidates() if case['weights_id'] == recovery.HYBRID
                and case['group'] == 16 and case['attention'] == 'dense')
    ready_build(case, ['ninfer_r9700_fp8_gate_up_qual'])
    require_absent([QUALIFICATION])
    executable = case['build'] / 'src/ninfer_r9700_fp8_gate_up_qual'
    report = QUALIFICATION / 'shared-context.json'
    command_line = [str(executable), '--shared-context', '--output', str(report)]
    inputs = {
        'executable': recovery.identity(executable),
        'command': command_line,
        'sources': [recovery.identity(REPO / path) for path in recovery.SOURCE_FILES],
        'qualifier_source': recovery.identity(REPO / 'tools/r9700/fp8_gate_up_qual.hip'),
        'build_cache': recovery.identity(case['build'] / 'CMakeCache.txt'),
    }
    deadline = time.monotonic() + 10.
    while True:
        bench.require_auto_power_profile()
        device = bench.R9700_POWER_PROFILE.parent
        if (int((device / 'mem_info_vram_used').read_text()) < 1024**3
                and int((device / 'gpu_busy_percent').read_text()) == 0):
            break
        if time.monotonic() >= deadline:
            raise ValueError('R9700 must be idle for shared-context qualification')
        time.sleep(.1)
    bench.require_hip_pci_device(0)
    QUALIFICATION.mkdir()
    durable_create_json(QUALIFICATION / 'inputs.json', inputs)
    code = execute(command_line, QUALIFICATION, 'shared-context')
    if code:
        durable_create_json(QUALIFICATION / 'failure.json', {'returncode': code})
        raise ValueError('shared-context qualification failed; outputs preserved, no bridge')
    for item in [inputs['executable'], inputs['qualifier_source'], inputs['build_cache'],
                 *inputs['sources']]:
        recovery.checked_file(item)
    bench.require_auto_power_profile()
    recovery.check_physical(json.loads(report.read_text()))
    durable_create_json(QUALIFICATION / 'receipt.json', {
        'artifact_type': 'ninfer_r9700_fp8_context_physical_receipt', 'schema_version': 1,
        'pass': True, **inputs, 'report': recovery.identity(report),
        'prelaunch_inputs': recovery.identity(QUALIFICATION / 'inputs.json'),
    })


def startup(review_path):
    """First real C1 model proof before bridge freeze; never reuse failed outputs."""
    from tools.bench import run_ninfer_bench_matrix as bench
    require_absent([STARTUP])
    case = next(case for case in candidates() if case['weights_id'] == recovery.HYBRID
                and case['group'] == 16 and case['attention'] == 'dense')
    ready_build(case, ['ninfer_bench', 'ninfer_r9700_runtime_planner_qual'])
    review = json.loads(review_path.read_text())
    recovery.check_review(review)
    qualification = recovery.validate_qualification(QUALIFICATION / 'receipt.json', review)
    if qualification['build_cache'] != recovery.identity(case['build'] / 'CMakeCache.txt'):
        raise ValueError('first model build differs from public-Op qualification')
    authority, selection = validate_prefill_chunk_authority(SELECTION)
    source, = [source for source in selection['sources'] if [
        source['weights_id'], source['kv_value_group'], source['xattention_profile']] == identity(case)]
    artifact = bench.inspect_artifact(case['artifact'])
    if any(artifact.get(key) != source['artifact'][key]
           for key in ('path', 'weights_id', 'sha256', 'file_size_bytes')):
        raise ValueError('first model artifact differs from selected hybrid source')
    report = STARTUP / 'model-c1.json'
    command_line = startup_command(case, {'authority': authority}, report)
    paths = (case['build'] / 'bench/ninfer_bench',
             case['build'] / 'src/ninfer_r9700_runtime_planner_qual', CORPUS, SELECTION,
             QUALIFICATION / 'receipt.json', review_path, Path(__file__))
    inputs = {'command': command_line, 'artifact': artifact,
              'files': [recovery.identity(path) for path in paths]}
    bench.require_auto_power_profile()
    bench.require_hip_pci_device(0)
    STARTUP.mkdir()
    durable_create_json(STARTUP / 'inputs.json', inputs)
    code = execute(command_line, STARTUP, 'model-c1')
    durable_create_json(STARTUP / 'exit.json', {'returncode': code})
    if code:
        raise ValueError('first model startup failed; retained output, no capacity admission')
    for item in inputs['files']:
        recovery.checked_file(item)
    bench.require_auto_power_profile()
    value = {
        'artifact_type': 'ninfer_r9700_fp8_context_model_startup', 'schema_version': 1,
        'pass': True, 'report': recovery.identity(report),
        'inputs': recovery.identity(STARTUP / 'inputs.json'),
        'memory': json.loads(report.read_text())['memory'],
        'scope': 'first real hybrid C1 startup accounting, not a capacity matrix or speed result',
    }
    bridge = {'hybrid_builds': [{
        'kv_value_group': 16, 'xattention_profile': 'dense', 'artifact': source['artifact'],
        'new_benchmark': recovery.executable_identity(case['build'] / 'bench/ninfer_bench'),
        'new_planner': recovery.executable_identity(case['build'] / 'src/ninfer_r9700_runtime_planner_qual')}],
        'chunk_selection': recovery.identity(SELECTION),
        'qualification': recovery.identity(QUALIFICATION / 'receipt.json'),
        'source_review': recovery.identity(review_path)}
    recovery.validate_model_startup_value(value, bridge)
    durable_create_json(STARTUP / 'result.json', value)


def original_inputs():
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
        files.update(case['old_build'] / name for name in (
            'bench/ninfer_bench', 'src/ninfer_r9700_runtime_planner_qual'))
        files.add(case['old_build'] / 'CMakeCache.txt')
    for path in files:
        if (not path.is_file() or (path != CORPUS and path.name != 'CMakeCache.txt'
                                   and not os.access(path, os.X_OK))
                or run.file_sha256(path) != frozen['files'].get(str(path.relative_to(REPO)))):
            raise ValueError(f'file differs from frozen panel inputs: {path}')
    for source in record['sources']:
        expected = dict(artifacts[source['weights_id']])
        expected['file_size_bytes'] = expected.pop('bytes')
        case = next(case for case in cases if identity(case) == [
            source['weights_id'], source['kv_value_group'], source['xattention_profile']])
        bench = case['old_build'] / 'bench/ninfer_bench'
        if (source['artifact'] != expected
                or source['benchmark_executable']['path'] != str(bench)
                or source['benchmark_executable']['sha256'] != frozen['files'][str(bench.relative_to(REPO))]):
            raise ValueError('selection evidence differs from frozen panel inputs')
    return {
        'authority': authority, 'quality': quality, 'selection': record,
        'bindings': {str(path): run.file_sha256(path) for path in (SELECTION, FROZEN, QUALITY, CLOSURE)},
    }


def retained_capacity(state):
    closure = json.loads(CLOSURE.read_text())
    expected = [case['name'] for case in candidates() if case['weights_id'] != recovery.HYBRID]
    if (closure.get('status') != 'stopped_unaccounted_fp8_library_allocation'
            or closure.get('reusable_completed_profiles') != expected
            or closure.get('completed_concurrency_per_reusable_profile') != [1, 2, 3, 4]):
        raise ValueError('stopped capacity closure does not identify all eight reusable matrices')
    records = {}
    for case in candidates():
        if case['weights_id'] == recovery.HYBRID:
            continue
        root = matrix(case, 'capacity')
        result = validate_matrix(root, state['authority'], executed=True)
        # validate_matrix validates all four physical reports or structured failures;
        # reuse is narrower: these eight matrices must retain four actual successes.
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        reports = [Path(row['report']) for row in manifest['commands']]
        if len(reports) != 4 or any(not path.is_file() for path in reports):
            raise ValueError('retained nonhybrid capacity matrix no longer has four successes')
        source = next(source for source in state['selection']['sources'] if [
            source['weights_id'], source['kv_value_group'], source['xattention_profile']]
            == identity(case))
        if manifest['bench'] != source['benchmark_executable'] or manifest['artifact'] != source['artifact']:
            raise ValueError('retained capacity differs from original frozen candidate')
        for path in [manifest_path, *reports]:
            records[str(path)] = run.file_sha256(path)
    return records


def freeze(review_path, model_startup):
    require_absent([BRIDGE])
    state = original_inputs()
    reviewed = json.loads(review_path.read_text())
    recovery.check_review(reviewed)
    mappings = []
    for case in candidates():
        if case['weights_id'] != recovery.HYBRID:
            continue
        ready_build(case, ['ninfer_bench', 'ninfer_r9700_runtime_planner_qual'])
        source = next(source for source in state['selection']['sources'] if [
            source['weights_id'], source['kv_value_group'], source['xattention_profile']]
            == identity(case))
        mappings.append({
            'weights_id': case['weights_id'], 'kv_value_group': case['group'],
            'xattention_profile': case['attention'], 'artifact': source['artifact'],
            'old_benchmark': source['benchmark_executable'],
            'new_benchmark': recovery.executable_identity(case['build'] / 'bench/ninfer_bench'),
            'old_planner': recovery.executable_identity(case['old_build'] / 'src/ninfer_r9700_runtime_planner_qual'),
            'new_planner': recovery.executable_identity(case['build'] / 'src/ninfer_r9700_runtime_planner_qual'),
            'old_build_cache': recovery.identity(case['old_build'] / 'CMakeCache.txt'),
            'build_cache': recovery.identity(case['build'] / 'CMakeCache.txt'),
        })
    value = {
        'artifact_type': 'ninfer_r9700_fp8_context_resource_recovery', 'schema_version': 1,
        'scope': recovery.SCOPE, 'preserves': ['numerical_quality', 'selected_prefill_chunk'],
        'capacity_or_speed_reuse': False, 'selected_prefill_chunk': 2048,
        'chunk_selection': recovery.identity(SELECTION), 'quality_authorities': recovery.identity(QUALITY),
        'frozen_panel_inputs': recovery.identity(FROZEN), 'stopped_capacity': recovery.identity(CLOSURE),
        'source_review': recovery.identity(review_path),
        'qualification': recovery.identity(QUALIFICATION / 'receipt.json'),
        'model_startup': recovery.identity(model_startup),
        'hybrid_builds': mappings, 'retained_nonhybrid_capacity': retained_capacity(state),
    }
    # Validate the complete prospective authority before exclusive publication.
    recovery.validate_bridge_value(value, state['selection'])
    unchanged(state)
    durable_create_json(BRIDGE, value)


def preflight():
    state = original_inputs()
    state['recovery'] = recovery.validate_bridge(BRIDGE, state['selection'])
    if state['recovery']['retained_nonhybrid_capacity'] != retained_capacity(state):
        raise ValueError('retained nonhybrid capacity bytes changed after recovery freeze')
    state['bindings'].update(state['recovery']['retained_nonhybrid_capacity'])
    state['bindings'][str(BRIDGE)] = run.file_sha256(BRIDGE)
    for mapping in state['recovery']['hybrid_builds']:
        for key in ('new_benchmark', 'new_planner', 'build_cache'):
            item = mapping[key]
            state['bindings'][item['path']] = item['sha256']
    return state


def startup_command(case, state, report):
    from tools.bench import run_ninfer_bench_matrix as bench
    workload, = bench.build_cases('pareto-capacity',
                                 production_prefill_chunk=state['authority']['selected_prefill_chunk'])
    return bench.add_repetition_args([
        str(case['build'] / 'bench/ninfer_bench'), '--weights', str(case['artifact']),
        '--corpus', str(CORPUS), '--device', '0', '--concurrency', '1', *workload.args,
        '--output', 'json', '--output-file', str(report)], workload, None, None)




def validate_recovered_headroom():
    for case in candidates():
        if case['weights_id'] != recovery.HYBRID:
            continue
        manifest = json.loads((matrix(case, 'capacity') / 'manifest.json').read_text())
        for record in manifest['commands']:
            path = Path(record['report'])
            if path.is_file():
                recovery.check_startup_headroom(json.loads(path.read_text()))


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
    recovery.validate_model_startup(recovery.checked_file(state['recovery']['model_startup']), state['recovery'])
    root = create_stage('capacity', state)
    outcomes = []
    for case in candidates():
        if case['weights_id'] != recovery.HYBRID:
            continue
        unchanged(state)
        code = execute(command(case, 'capacity', state), root, case['name'])
        outcomes.append({'identity': identity(case), 'returncode': code})
    durable_create_json(root / 'runner-outcomes.json', outcomes)
    unchanged(state)
    # Valid retained capacity exclusions are evidence, not reasons to skip other profiles.
    result = validated_capacity()
    validate_recovered_headroom()
    durable_create_json(root / 'validation.json', result)


def capacity_evidence(state):
    require_stage_bindings('capacity', state)
    recorded = json.loads((PACKAGE / 'capacity/validation.json').read_text())
    actual = validated_capacity()
    validate_recovered_headroom()
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
            '--fp8-context-recovery', str(BRIDGE),
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
    parser.add_argument('stage', choices=('qualify', 'startup', 'freeze', 'preflight', 'capacity', 'whole', 'select'))
    parser.add_argument('--review', type=Path, help='reviewed unchanged-arithmetic source record (freeze only)')
    parser.add_argument('--model-startup', type=Path, help='successful bound real-model startup receipt (freeze only)')
    args = parser.parse_args()
    if args.stage == 'qualify':
        qualify()
        return 0
    if args.stage == 'startup':
        if args.review is None:
            raise ValueError('startup requires --review for accounted allocation sources')
        startup(args.review)
        return 0
    if args.stage == 'freeze':
        if args.review is None or args.model_startup is None:
            raise ValueError('freeze requires --review and --model-startup; neither proof can be inferred')
        freeze(args.review, args.model_startup)
        return 0
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
