#!/usr/bin/env python3
"""Compare every corrected graph repetition with one exact public eager run."""
from __future__ import annotations
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import campaign
from tools.bench import run_ninfer_bench_matrix as bench
from tools.bench.prefill_chunk_authority import durable_create_json
from tools.ppl import assemble_pareto

# Reuse only the closed command transformation and exact-token comparison, not old orchestration.
spec = importlib.util.spec_from_file_location('retained_control_helpers', campaign.OLD_PACKAGE / 'controls.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
eager_command, exact_tokens = helpers.eager_command, helpers.exact_tokens
OUTPUT = campaign.PACKAGE / 'controls'


def preflight():
    state = campaign.preflight()
    eligible = campaign.whole_eligible(state)
    campaign.require_stage_bindings('whole', state)
    chunk = state['authority']['selected_prefill_chunk']
    graph_case, = bench.build_cases('pareto-whole', production_prefill_chunk=chunk)
    if not graph_case.retain_token_ids:
        raise ValueError('whole preset must retain all public tokens')
    entries, bindings = [], dict(state['bindings'])
    for case in campaign.candidates():
        if campaign.identity(case) not in eligible:
            continue
        root = campaign.matrix(case, 'whole')
        manifest = assemble_pareto._manifest(root, 'pareto-whole')
        source = next(source for source in state['selection']['sources'] if
                      [source['weights_id'], source['kv_value_group'], source['xattention_profile']]
                      == campaign.identity(case))
        if (manifest.get('artifact') != source['artifact']
                or manifest.get('bench') != campaign.expected_benchmark(case, state)
                or manifest.get('corpus') != str(campaign.CORPUS)
                or manifest.get('corpus_sha256') != bindings[str(campaign.CORPUS)]
                or manifest.get('prefill_chunk_authority') != state['authority']
                or manifest.get('selected_prefill_chunk') != chunk
                or manifest.get('concurrency') != [1, 2, 3, 4]):
            raise ValueError('corrected graph controls differ from frozen inputs')
        reports = assemble_pareto._reports(root, manifest, 'pareto-whole', chunk)
        bindings[str(root / 'manifest.json')] = campaign.run.file_sha256(root / 'manifest.json')
        for record in manifest['commands']:
            concurrency = record['concurrency']
            report, = reports[concurrency]
            if report['schema_version'] != 21:
                raise ValueError('new controls require corrected schema21 reports')
            command = record['command']
            if (Path(command[0]) != case['build'] / 'bench/ninfer_bench'
                    or manifest['expected_kv_value_group'] != case['group']
                    or manifest['expected_xattention_profile'] != case['attention']):
                raise ValueError('graph report uses another compiled profile')
            for option, value in (('--weights', str(case['artifact'])),
                                  ('--corpus', str(campaign.CORPUS)), ('--device', '0')):
                if command.count(option) != 1 or command[command.index(option) + 1] != value:
                    raise ValueError(f'graph command differs from {option}')
            name = f'{case["name"]}-c{concurrency}'
            path, output = Path(record['report']), OUTPUT / f'{name}.json'
            bindings[str(path)] = campaign.run.file_sha256(path)
            entries.append({'name': name, 'candidate': case, 'manifest': manifest, 'graph': report,
                            'graph_path': path, 'output': output, 'concurrency': concurrency,
                            'command': eager_command(command, output)})
    if len(entries) != len(eligible) * 4 or not entries:
        raise ValueError('controls must cover every eligible candidate and C1..4')
    return state, entries, bindings, replace(graph_case, args=(*graph_case.args, '--no-device-graph'),
                                            repetitions=1, warmup=0)


def check_bindings(bindings):
    campaign.unchanged({'bindings': bindings})


def comparison(entry, bindings, eager_case):
    manifest = entry['manifest']
    eager = bench.load_bench_report(entry['output'], manifest['expected_kv_value_group'],
        manifest['expected_q4_activation_bits'], manifest['expected_w8_activation_bits'],
        manifest['expected_fp8_qk_wmma_enabled'], entry['concurrency'], manifest['artifact'],
        entry['command'], eager_case, manifest['expected_xattention_profile'])
    if eager['schema_version'] != 21:
        raise ValueError('new eager controls require schema21')
    return {'candidate': campaign.identity(entry['candidate']), 'concurrency': entry['concurrency'],
            'tests': exact_tokens(entry['graph'], eager),
            'graph': {'path': str(entry['graph_path']), 'sha256': bindings[str(entry['graph_path'])]},
            'eager': {'path': str(entry['output']), 'sha256': campaign.run.file_sha256(entry['output'])}}


def validate_completed():
    _, entries, bindings, eager_case = preflight()
    result = json.loads((OUTPUT / 'result.json').read_text())
    if (result.get('artifact_type') != 'ninfer_r9700_terminal_graph_eager_controls'
            or result.get('schema_version') != 1 or result.get('pass') is not True
            or result.get('timing_eligible') is not False or result.get('inputs') != bindings
            or json.loads((OUTPUT / 'inputs.json').read_text()) != bindings):
        raise ValueError('control publication differs from current inputs')
    comparisons = []
    for entry in entries:
        if json.loads((OUTPUT / f'{entry["name"]}.command.json').read_text()) != entry['command']:
            raise ValueError('retained eager command changed')
        replay = comparison(entry, bindings, eager_case)
        if (not all(test['exact'] for test in replay['tests']) or
                json.loads((OUTPUT / f'{entry["name"]}.comparison.json').read_text()) != replay):
            raise ValueError('public graph/eager token comparison failed')
        comparisons.append(replay)
    if result.get('comparisons') != comparisons:
        raise ValueError('control publication omits an eligible cell')
    check_bindings(bindings)
    return result


def execute():
    state, entries, bindings, eager_case = preflight()
    campaign.require_absent([OUTPUT])
    bench.require_auto_power_profile()
    bench.require_hip_pci_device(0)
    OUTPUT.mkdir()
    durable_create_json(OUTPUT / 'inputs.json', bindings)
    results = []
    for entry in entries:
        check_bindings(bindings)
        bench.require_auto_power_profile()
        durable_create_json(OUTPUT / f'{entry["name"]}.command.json', entry['command'])
        code = campaign.execute(entry['command'], OUTPUT, entry['name'])
        if code:
            raise ValueError(f'eager execution failed; preserve {entry["name"]} exit{code}')
        result = comparison(entry, bindings, eager_case)
        durable_create_json(OUTPUT / f'{entry["name"]}.comparison.json', result)
        results.append(result)
        bench.require_auto_power_profile()
        check_bindings(bindings)
        if not all(test['exact'] for test in result['tests']):
            raise ValueError(f'exact public parity failed: {entry["name"]}')
    campaign.unchanged(state)
    durable_create_json(OUTPUT / 'result.json', {
        'artifact_type': 'ninfer_r9700_terminal_graph_eager_controls', 'schema_version': 1,
        'pass': True, 'timing_eligible': False, 'inputs': bindings, 'comparisons': results,
        'criterion': 'all public tokens exact across corrected graph repetitions and one eager run'})
