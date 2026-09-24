#!/usr/bin/env python3
"""Exact ordinary graph/eager controls reusing the completed whole campaign."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import campaign
from tools.bench import run_ninfer_bench_matrix as bench
from tools.bench.prefill_chunk_authority import durable_create_json
from tools.ppl import assemble_pareto

OUTPUT = campaign.PACKAGE / 'controls'


def eager_command(command, output):
    result = list(command)
    if '--no-device-graph' in result or result.count('--retain-token-ids') != 1:
        raise ValueError('graph control must retain tokens and enable Device Graph')
    for option, value in (('-r', '1'), ('--warmup', '0'), ('--output-file', str(output))):
        if result.count(option) != 1 or result.index(option) + 1 >= len(result):
            raise ValueError(f'graph command lacks unique {option}')
        result[result.index(option) + 1] = value
    return [*result, '--no-device-graph']


def exact_tokens(graph, eager):
    """Reports have already passed native shape/profile validation."""
    first = {test['label']: test for test in graph['tests']}
    second = {test['label']: test for test in eager['tests']}
    if (len(first) != len(graph['tests']) or len(second) != len(eager['tests'])
            or set(first) != set(second) or not first):
        raise ValueError('graph/eager test identities differ')
    results = []
    for label, test in first.items():
        repetitions = test['reps']
        counterpart = second[label]['reps']
        if not repetitions or len(counterpart) != 1:
            raise ValueError('graph/eager repetition set is incomplete')
        expected = repetitions[0].get('generated_token_ids_by_lane')
        if not isinstance(expected, list) or not expected or any(not lane for lane in expected):
            raise ValueError('graph control lacks public token IDs')
        matches = all(rep.get('generated_token_ids_by_lane') == expected
                      for rep in [*repetitions, *counterpart])
        results.append({'label': label, 'exact': matches,
                        'graph_repetitions': len(repetitions),
                        'lanes': len(expected), 'tokens_per_lane': [len(lane) for lane in expected]})
    return results


def validate_graph_binding(manifest, candidate, state, source, corpus_sha):
    if (manifest.get('artifact') != source['artifact']
            or manifest.get('bench') != campaign.recovery.resolved_benchmark(source, state['recovery'])
            or manifest.get('corpus') != str(campaign.CORPUS)
            or manifest.get('corpus_sha256') != corpus_sha
            or manifest.get('prefill_chunk_authority') != state['authority']
            or manifest.get('selected_prefill_chunk') != state['authority']['selected_prefill_chunk']):
        raise ValueError('graph manifest differs from current artifact/build/corpus/chunk inputs')
    for record in manifest['commands']:
        command = record['command']
        for option, value in (('--weights', str(candidate['artifact'])),
                              ('--corpus', str(campaign.CORPUS)), ('--device', '0')):
            if (command.count(option) != 1 or command.index(option) + 1 >= len(command)
                    or command[command.index(option) + 1] != value):
                raise ValueError(f'graph command differs from current {option}')


def preflight():
    state = campaign.preflight()
    eligible = campaign.whole_eligible(state)
    campaign.require_stage_bindings('whole', state)
    chunk = state['authority']['selected_prefill_chunk']
    graph_case, = bench.build_cases('pareto-whole', production_prefill_chunk=chunk)
    if not graph_case.retain_token_ids:
        raise ValueError('whole preset must retain public token IDs')
    entries = []
    bindings = dict(state['bindings'])
    selection = json.loads(campaign.SELECTION.read_text())
    frozen = json.loads(campaign.FROZEN.read_text())
    corpus_sha = frozen['files'][str(campaign.CORPUS.relative_to(campaign.REPO))]
    for candidate in campaign.candidates():
        if campaign.identity(candidate) not in eligible:
            continue
        root = campaign.matrix(candidate, 'whole')
        manifest = assemble_pareto._manifest(root, 'pareto-whole')
        source = next(source for source in selection['sources'] if
                      [source['weights_id'], source['kv_value_group'], source['xattention_profile']]
                      == campaign.identity(candidate))
        validate_graph_binding(manifest, candidate, state, source, corpus_sha)
        reports = assemble_pareto._reports(root, manifest, 'pareto-whole', chunk)
        manifest_path = root / 'manifest.json'
        bindings[str(manifest_path)] = campaign.run.file_sha256(manifest_path)
        if manifest['concurrency'] != [1, 2, 3, 4]:
            raise ValueError('graph control lacks C1..4')
        for record in manifest['commands']:
            concurrency = record['concurrency']
            report, = reports[concurrency]
            command = record['command']
            if (Path(command[0]) != candidate['build'] / 'bench/ninfer_bench'
                    or manifest['expected_kv_value_group'] != candidate['group']
                    or manifest['expected_xattention_profile'] != candidate['attention']
                    or manifest['artifact']['weights_id'] != candidate['weights_id']):
                raise ValueError('graph control differs from eligible candidate')
            name = f"{candidate['name']}-c{concurrency}"
            graph_path = Path(record['report'])
            bindings[str(graph_path)] = campaign.run.file_sha256(graph_path)
            output = OUTPUT / f'{name}.json'
            entries.append({'name': name, 'candidate': candidate, 'manifest': manifest,
                            'graph': report, 'graph_path': graph_path, 'output': output,
                            'concurrency': concurrency,
                            'command': eager_command(command, output)})
    if not entries:
        raise ValueError('no eligible graph controls')
    eager_case = replace(graph_case, args=(*graph_case.args, '--no-device-graph'),
                         repetitions=1, warmup=0)
    return state, entries, bindings, eager_case


def check_bindings(bindings):
    if any(campaign.run.file_sha256(Path(path)) != digest for path, digest in bindings.items()):
        raise ValueError('graph/eager control input changed')


def comparison(entry, bindings, eager_case):
    manifest = entry['manifest']
    eager = bench.load_bench_report(
        entry['output'], manifest['expected_kv_value_group'],
        manifest['expected_q4_activation_bits'], manifest['expected_w8_activation_bits'],
        manifest['expected_fp8_qk_wmma_enabled'], entry['concurrency'],
        manifest['artifact'], entry['command'], eager_case, manifest['expected_xattention_profile'])
    return {'candidate': campaign.identity(entry['candidate']),
            'concurrency': entry['concurrency'], 'tests': exact_tokens(entry['graph'], eager),
            'graph': {'path': str(entry['graph_path']),
                      'sha256': bindings[str(entry['graph_path'])]},
            'eager': {'path': str(entry['output']),
                      'sha256': campaign.run.file_sha256(entry['output'])}}


def validate_completed():
    _, entries, bindings, eager_case = preflight()
    result = json.loads((OUTPUT / 'result.json').read_text())
    if (result.get('artifact_type') != 'ninfer_r9700_terminal_graph_eager_controls'
            or result.get('schema_version') != 1 or result.get('pass') is not True
            or result.get('timing_eligible') is not False or result.get('inputs') != bindings
            or json.loads((OUTPUT / 'inputs.json').read_text()) != bindings):
        raise ValueError('graph/eager control authority differs from current inputs')
    comparisons = []
    for entry in entries:
        if json.loads((OUTPUT / f"{entry['name']}.command.json").read_text()) != entry['command']:
            raise ValueError('retained eager command differs from graph control')
        replay = comparison(entry, bindings, eager_case)
        if (not all(test['exact'] for test in replay['tests'])
                or json.loads((OUTPUT / f"{entry['name']}.comparison.json").read_text()) != replay):
            raise ValueError('retained graph/eager comparison is not exact')
        comparisons.append(replay)
    if result.get('comparisons') != comparisons:
        raise ValueError('graph/eager authority does not retain every current eligible control')
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
        command = entry['command']
        durable_create_json(OUTPUT / f"{entry['name']}.command.json", command)
        code = campaign.execute(command, OUTPUT, entry['name'])
        if code:
            durable_create_json(OUTPUT / 'failure.json', {'name': entry['name'], 'exit_code': code})
            raise ValueError(f"eager execution failed; preserved {entry['name']}")
        result = comparison(entry, bindings, eager_case)
        durable_create_json(OUTPUT / f"{entry['name']}.comparison.json", result)
        results.append(result)
        bench.require_auto_power_profile()
        check_bindings(bindings)
        if not all(test['exact'] for test in result['tests']):
            raise ValueError(f"exact public graph/eager parity failed: {entry['name']}")
    campaign.unchanged(state)
    durable_create_json(OUTPUT / 'result.json', {
        'artifact_type': 'ninfer_r9700_terminal_graph_eager_controls', 'schema_version': 1,
        'pass': True, 'timing_eligible': False, 'inputs': bindings, 'comparisons': results,
        'criterion': 'all public tokens exact across retained graph repetitions and one eager run',
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'run'))
    args = parser.parse_args()
    if args.stage == 'preflight':
        _, entries, _, _ = preflight()
        print(f'read-only preflight passed: {len(entries)} matched eager controls')
    else:
        execute()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
