#!/usr/bin/env python3
"""Fresh schema21 whole/control evidence; reuse validated capacity and numerical proofs."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tools.bench.prefill_chunk_authority import durable_create_json
from tools.ppl import benchmark_reporting_recovery as reporting
from tools.ppl import fp8_context_recovery as recovery, pareto, run
from tools.ppl.terminal_selection_io import publish, require_absent

OLD_PACKAGE = REPO / 'profiles/bench/r9700-terminal-base-fp8-context-recovery-20260921'
spec = importlib.util.spec_from_file_location('retained_resource_campaign', OLD_PACKAGE / 'campaign.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
PYTHON, SELECTION, FROZEN, QUALITY, CORPUS, RUNNER = (
    old.PYTHON, old.SELECTION, old.FROZEN, old.QUALITY, old.CORPUS, old.RUNNER)
BRIDGE = PACKAGE / 'reporting-recovery.json'
CAPACITY_VALIDATION = OLD_PACKAGE / 'capacity/validation.json'


def host_checks():
    root = PACKAGE / 'host-checks'
    require_absent([root])
    root.mkdir()
    for case in candidates():
        if case['weights_id'] != recovery.HYBRID:
            continue
        profile = ('dense' if case['attention'] == 'dense' else 'xattention') + f'-g{case["group"]}'
        executable = case['build'] / 'src/ninfer_r9700_runtime_planner_qual'
        before = recovery.executable_identity(executable)
        command_line = [str(executable), '--host-split512-routing']
        code = execute(command_line, root, profile)
        if recovery.executable_identity(executable) != before:
            raise ValueError('host planner changed during execution')
        durable_create_json(root / f'{profile}.json', {
            'command': command_line, 'exit_code': code,
            'stdout': recovery.identity(root / f'{profile}.stdout.txt'),
            'stderr': recovery.identity(root / f'{profile}.stderr.txt')})
        if code:
            raise ValueError(f'host planner failed; preserve outputs: {profile}')


def candidates():
    for case in old.candidates():
        suffix = '' if case['attention'] == 'dense' else '-xattention'
        yield {**case, 'build': REPO / f'build-r9700-phase-sum{suffix}-g{case["group"]}-20260921'}


identity = old.identity


def matrix(case, stage):
    return old.matrix(case, stage) if stage == 'capacity' else PACKAGE / stage / case['name']


def original_inputs():
    state = old.preflight()
    state['capacity'] = old.capacity_evidence(state)
    paths = [CAPACITY_VALIDATION, CORPUS]
    for case in candidates():
        manifest_path = matrix(case, 'capacity') / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        paths += [manifest_path, *(Path(row['report']) for row in manifest['commands'])]
    # This continuation retains the actual completed48 startup reports, not a new capacity inference.
    state['bindings'].update({str(path): run.file_sha256(path) for path in paths})
    return state


def freeze(review_path, host_checks):
    require_absent([BRIDGE])
    state = original_inputs()
    mappings = []
    for case in candidates():
        if case['weights_id'] != recovery.HYBRID:
            continue
        source = next(item for item in state['selection']['sources']
                      if item['weights_id'] != recovery.HYBRID and
                      (item['kv_value_group'], item['xattention_profile']) == (case['group'], case['attention']))
        hybrid = next(item for item in state['recovery']['hybrid_builds']
                      if (item['kv_value_group'], item['xattention_profile']) == (case['group'], case['attention']))
        original_build = Path(source['benchmark_executable']['path']).parent.parent
        profile = ('dense' if case['attention'] == 'dense' else 'xattention') + f'-g{case["group"]}'
        mappings.append({
            'kv_value_group': case['group'], 'xattention_profile': case['attention'],
            'old_nonhybrid_benchmark': source['benchmark_executable'],
            'old_nonhybrid_planner': recovery.executable_identity(original_build / 'src/ninfer_r9700_runtime_planner_qual'),
            'old_nonhybrid_build_cache': recovery.identity(original_build / 'CMakeCache.txt'),
            'old_hybrid_benchmark': hybrid['new_benchmark'],
            'old_hybrid_planner': hybrid['new_planner'], 'old_hybrid_build_cache': hybrid['build_cache'],
            'new_benchmark': recovery.executable_identity(case['build'] / 'bench/ninfer_bench'),
            'new_planner': recovery.executable_identity(case['build'] / 'src/ninfer_r9700_runtime_planner_qual'),
            'build_cache': recovery.identity(case['build'] / 'CMakeCache.txt'),
            'host_planner_check': recovery.identity(host_checks / f'{profile}.json'),
        })
    value = {'artifact_type': 'ninfer_r9700_benchmark_reporting_recovery', 'schema_version': 1,
             'scope': reporting.SCOPE,
             'preserves': ['numerical_quality', 'selected_prefill_chunk', 'capacity'],
             'reuses_concurrent_prefill_metrics': False,
             'fp8_context_resource_recovery': recovery.identity(old.BRIDGE),
             'source_review': recovery.identity(review_path), 'profiles': mappings}
    reporting.validate_bridge_value(value, state['selection'])
    unchanged(state)
    durable_create_json(BRIDGE, value)


def preflight():
    state = original_inputs()
    state['reporting'] = reporting.validate_bridge(BRIDGE, state['selection'])
    state['bindings'][str(BRIDGE)] = run.file_sha256(BRIDGE)
    review = state['reporting']['source_review']
    state['bindings'][review['path']] = review['sha256']
    for row in state['reporting']['profiles']:
        for key in ('new_benchmark', 'new_planner', 'build_cache', 'host_planner_check'):
            item = row[key]
            state['bindings'][item['path']] = item['sha256']
    for source in reporting.SOURCE_FILES:
        path = REPO / source
        state['bindings'][str(path)] = run.file_sha256(path)
    for path in (Path(__file__), PACKAGE / 'controls.py', RUNNER):
        state['bindings'][str(path)] = run.file_sha256(path)
    return state


def whole_eligible(state):
    capacity_ids = state['capacity']['whole_eligible_identities']
    return [identity(case) for case in candidates() if identity(case) in capacity_ids
            and state['quality']['authorities'][case['quality_name']]['eligibility_by_group'][str(case['group'])]]


def expected_benchmark(case, state):
    return reporting.mapping(state['reporting'], case['group'], case['attention'])['new_benchmark']


def unchanged(state):
    if any(run.file_sha256(Path(path)) != digest for path, digest in state['bindings'].items()):
        raise ValueError('phase-sum campaign input changed')


def command(case, state):
    args = [str(PYTHON), str(RUNNER), '--preset', 'pareto-whole', '--no-build',
            '--bench', str(case['build'] / 'bench/ninfer_bench'), '--weights', str(case['artifact']),
            '--corpus', str(CORPUS), '--device', '0',
            '--prefill-chunk', str(state['authority']['selected_prefill_chunk']),
            '--prefill-chunk-authority', str(SELECTION), '--expected-kv-value-group', str(case['group']),
            '--expected-q4-activation-bits', '8', '--expected-w8-activation-bits', '8',
            '--expected-fp8-qk-wmma', '1', '--expected-xattention-profile', case['attention'],
            '--output-dir', str(matrix(case, 'whole'))]
    for concurrency in (1, 2, 3, 4):
        args += ['--concurrency', str(concurrency)]
    if case['weights_id'] == recovery.HYBRID:
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
        raise ValueError(f'{stage} inputs differ from current bridge')


def execute(args, root, name):
    with (root / f'{name}.stdout.txt').open('x') as stdout, (root / f'{name}.stderr.txt').open('x') as stderr:
        return subprocess.run(args, cwd=REPO, stdout=stdout, stderr=stderr).returncode


def whole(state):
    eligible = whole_eligible(state)
    root = create_stage('whole', state)
    for case in candidates():
        if identity(case) not in eligible:
            continue
        unchanged(state)
        code = execute(command(case, state), root, case['name'])
        if code:
            raise ValueError(f'whole failed; preserve namespace: {case["name"]} exit{code}')
        validate_whole(case, state)
    unchanged(state)


def validate_whole(case, state):
    from tools.ppl import assemble_pareto
    from tools.bench import run_ninfer_bench_matrix as bench
    root = matrix(case, 'whole')
    manifest = assemble_pareto._manifest(root, 'pareto-whole')
    if manifest['bench'] != expected_benchmark(case, state):
        raise ValueError('whole matrix uses another reporter executable')
    reports = assemble_pareto._reports(root, manifest, 'pareto-whole', state['authority']['selected_prefill_chunk'])
    for concurrency, rows in reports.items():
        report, = rows
        if report['schema_version'] != 21 or not bench.prefill_timing_eligible(report, concurrency):
            raise ValueError('fresh whole evidence requires corrected schema21 phase rates')
    return reports


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
            '--post-chunk-capacity-validation', str(CAPACITY_VALIDATION),
            '--fp8-context-recovery', str(old.BRIDGE), '--benchmark-reporting-recovery', str(BRIDGE),
            '--out', str(pending_input)]
    for case in candidates():
        args += ['--candidate', case['name'], case['weights_id'], str(case['group']),
                 state['quality']['authorities'][case['quality_name']]['path'],
                 str(matrix(case, 'capacity')), str(matrix(case, 'whole')) if identity(case) in eligible else '-']
    if execute(args, root, 'assemble'):
        raise ValueError('selection assembly failed; outputs preserved')
    result = pareto.classify(pareto.load_payload(pending_input.read_text()))
    result['pareto_input'] = {'path': str(final_input.resolve()), 'sha256': run.file_sha256(pending_input)}
    durable_create_json(pending_result, result)
    unchanged(state)
    publish(pending_input, final_input, pending_result, final_result,
            guard_path=SELECTION, guard_sha256=state['authority']['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('host-checks', 'freeze', 'preflight', 'whole', 'controls', 'select'))
    parser.add_argument('--review', type=Path)
    parser.add_argument('--host-checks', type=Path, default=PACKAGE / 'host-checks')
    args = parser.parse_args()
    if args.stage == 'host-checks':
        host_checks()
    elif args.stage == 'freeze':
        if args.review is None or args.host_checks is None:
            raise ValueError('freeze requires --review and --host-checks')
        freeze(args.review, args.host_checks)
    elif args.stage == 'controls':
        import controls
        controls.execute()
    else:
        state = preflight()
        if args.stage == 'preflight':
            print(f'read-only preflight: {len(whole_eligible(state))} eligible whole candidates')
        else:
            {'whole': whole, 'select': select}[args.stage](state)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
