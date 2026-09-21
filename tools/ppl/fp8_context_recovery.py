"""Narrow resource-only bridge for the R9700 shared FP8-context recovery.

This preserves selected chunk and numerical evidence, never capacity or speed.
The source review and independently qualified physical run are both prerequisites.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[2]
HYBRID = 'r9700-q4g64-f8e4m3-four-role-n16k16-eval'
SCOPE = 'shared_fp8_library_context_before_capacity_snapshot'
SOURCE_FILES = (
    'src/ops/r9700/linear/linear_execution.h',
    'src/ops/r9700/linear/linear_execution.hip',
    'src/targets/qwen3_8_27b/impl/load/bindings.h',
    'src/targets/qwen3_8_27b/impl/load/bindings.cpp',
    'src/targets/qwen3_8_27b/impl/variant.h',
    'src/targets/qwen3_8_27b/impl/variant.cpp',
    'src/targets/qwen3_8_27b/impl/package.cpp',
    'src/targets/qwen3/impl/runtime/layouts_impl.h',
    'src/targets/qwen3/impl/runtime/instance.h',
    'src/targets/qwen3/export/ninfer/targets/qwen3/startup_features.h',
)


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    with path.open('rb') as stream:
        return {'path': str(path), 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def checked_file(record: dict) -> Path:
    if (not isinstance(record, dict) or set(record) not in (
            {'path', 'sha256'}, {'path', 'sha256', 'file_size_bytes'})
            or not isinstance(record.get('path'), str)
            or not Path(record['path']).is_absolute()
            or not isinstance(record.get('sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', record['sha256'])):
        raise ValueError('FP8 recovery has malformed file identity')
    path = Path(record['path'])
    if (not path.is_file() or identity(path) != {key: record[key] for key in ('path', 'sha256')}
            or 'file_size_bytes' in record and record['file_size_bytes'] != path.stat().st_size):
        raise ValueError(f'FP8 recovery bound file changed: {path}')
    return path


def executable_identity(path: Path) -> dict:
    return {**identity(path), 'file_size_bytes': path.stat().st_size}


def check_review(value: dict) -> None:
    if (value.get('artifact_type') != 'ninfer_r9700_fp8_context_source_review'
            or value.get('schema_version') != 1 or value.get('status') != 'unchanged_arithmetic'
            or value.get('scope') != SCOPE
            or value.get('arithmetic_changed') is not False
            or value.get('kernel_or_recipe_changed') is not False):
        raise ValueError('FP8 recovery requires reviewed unchanged arithmetic/kernel/recipe')
    records = value.get('sources')
    if not isinstance(records, list) or len(records) != len(SOURCE_FILES):
        raise ValueError('FP8 recovery source review is incomplete')
    if {str(checked_file(item)) for item in records} != {
            str(REPO / path) for path in SOURCE_FILES}:
        raise ValueError('FP8 recovery source review does not cover exact allocation changes')


def check_physical(value: dict) -> None:
    expected = {
        'artifact_type': 'ninfer_r9700_fp8_shared_context_qualification', 'schema_version': 1,
        'rows': [34816, 7168, 4096], 'columns': 5120, 'prepared_widths': [1, 2, 3, 4, 2048],
        'oracle_widths': [4, 2048], 'graph_replays_per_width': 2,
        'independent_context_fingerprints_match': True, 'pass': True,
        'oracle': 'fp64_public_bf16_times_exact_e4m3_weight_and_stored_fp32_scale',
        'graph_outputs_poisoned_before_replay': True,
        'prepared_before_binding': True, 'unbound_run_rejected': True,
    }
    if any((value.get(key) is not expected_value if type(expected_value) is bool
            else value.get(key) != expected_value) for key, expected_value in expected.items()):
        raise ValueError('FP8 recovery lacks complete passing shared-context qualification')
    if (str(value.get('pci_bus_id', '')).lower() != '0000:13:00.0'
            or type(value.get('maximum_bf16_steps')) is not int
            or not 0 <= value['maximum_bf16_steps'] <= 1):
        raise ValueError('FP8 recovery target or independent FP64 oracle failed')
    for field in ('hip_runtime_version', 'hipblaslt_version', 'free_before_context',
                  'free_after_context', 'free_before_prepare', 'free_after_prepare',
                  'free_after_binding', 'free_after_execution'):
        if type(value.get(field)) is not int or value[field] <= 0:
            raise ValueError(f'FP8 recovery lacks physical memory/toolchain evidence: {field}')
    if value['free_after_binding'] != value['free_after_prepare']:
        raise ValueError('FP8 workspace binding still changes physical memory after preparation')
    profiles = value.get('profiles')
    if not isinstance(profiles, list) or len(profiles) != 15:
        raise ValueError('FP8 recovery lacks all fifteen startup algorithm profiles')
    observed = set()
    for row in profiles:
        if (not isinstance(row, dict)
                or not isinstance(row.get('fingerprint'), str)
                or not re.fullmatch('[0-9a-f]{32}', row['fingerprint'])
                or type(row.get('matmul_workspace_bytes')) is not int
                or row['matmul_workspace_bytes'] != 0):
            raise ValueError('FP8 recovery changed selected algorithm/workspace profile')
        observed.add((row.get('rows'), row.get('tokens')))
    if observed != {(n, t) for n in (34816, 7168, 4096) for t in (1, 2, 3, 4, 2048)}:
        raise ValueError('FP8 recovery algorithm geometry differs from production startup')


def validate_bridge(path: Path, chunk_selection: dict | None = None) -> dict:
    """Reopen every small authority; never hash large weight artifacts here."""
    value = json.loads(path.read_text())
    return validate_bridge_value(value, chunk_selection)


def validate_bridge_value(value: dict, chunk_selection: dict | None = None) -> dict:
    if (value.get('artifact_type') != 'ninfer_r9700_fp8_context_resource_recovery'
            or value.get('schema_version') != 1 or value.get('scope') != SCOPE
            or value.get('preserves') != ['numerical_quality', 'selected_prefill_chunk']
            or value.get('capacity_or_speed_reuse') is not False
            or value.get('selected_prefill_chunk') != 2048):
        raise ValueError('unsupported FP8 resource-only recovery authority')
    selection_path = checked_file(value.get('chunk_selection'))
    selection = json.loads(selection_path.read_text())
    if chunk_selection is not None and selection != chunk_selection:
        raise ValueError('FP8 recovery is bound to another chunk selection')
    if selection.get('selected_prefill_chunk') != 2048:
        raise ValueError('FP8 recovery changed selected chunk')
    quality_path = checked_file(value.get('quality_authorities'))
    quality = json.loads(quality_path.read_text())
    if (quality.get('selected_prefill_chunk_authority') != value['chunk_selection']
            or quality.get('selected_prefill_chunk') != 2048):
        raise ValueError('FP8 recovery quality/chunk authorities differ')
    closure = json.loads(checked_file(value.get('stopped_capacity')).read_text())
    if closure.get('status') != 'stopped_unaccounted_fp8_library_allocation':
        raise ValueError('FP8 recovery lacks the retained stopped capacity attempt')
    review = json.loads(checked_file(value.get('source_review')).read_text())
    check_review(review)
    receipt = validate_qualification(checked_file(value.get('qualification')), review)
    executable = Path(receipt['executable']['path'])
    frozen = json.loads(checked_file(value.get('frozen_panel_inputs')).read_text())
    mappings = value.get('hybrid_builds')
    if not isinstance(mappings, list) or len(mappings) != 4:
        raise ValueError('FP8 recovery must replace exactly four hybrid builds')
    keys = set()
    for mapping in mappings:
        key = (mapping.get('weights_id'), mapping.get('kv_value_group'),
               mapping.get('xattention_profile'))
        if key in keys:
            raise ValueError('FP8 recovery has duplicate build mapping')
        keys.add(key)
        old = next((source for source in selection['sources'] if (
            source['weights_id'], source['kv_value_group'], source['xattention_profile']) == key), None)
        if (old is None or mapping.get('artifact') != old.get('artifact')
                or mapping.get('old_benchmark') != old.get('benchmark_executable')):
            raise ValueError('FP8 recovery changed original artifact/build/profile binding')
        for field in ('old_benchmark', 'new_benchmark', 'old_planner', 'new_planner',
                      'old_build_cache', 'build_cache'):
            checked_file(mapping.get(field))
        old_build = Path(mapping['old_benchmark']['path']).parent.parent
        if (Path(mapping['old_planner']['path']) != old_build / 'src/ninfer_r9700_runtime_planner_qual'
                or Path(mapping['old_build_cache']['path']) != old_build / 'CMakeCache.txt'):
            raise ValueError('FP8 recovery old planner/cache differ from frozen build')
        for field in ('old_benchmark', 'old_planner', 'old_build_cache'):
            item = mapping[field]
            if frozen['files'].get(str(Path(item['path']).relative_to(REPO))) != item['sha256']:
                raise ValueError('FP8 recovery old build differs from original frozen inputs')
        if cache_profile(Path(mapping['build_cache']['path'])) != cache_profile(
                Path(mapping['old_build_cache']['path'])):
            raise ValueError('FP8 recovery changed compiled arithmetic/profile configuration')
        if mapping['old_benchmark'] == mapping['new_benchmark']:
            raise ValueError('FP8 recovery must bind a fresh benchmark')
        build = Path(mapping['new_benchmark']['path']).parent.parent
        if (Path(mapping['new_planner']['path']) != build / 'src/ninfer_r9700_runtime_planner_qual'
                or Path(mapping['build_cache']['path']) != build / 'CMakeCache.txt'):
            raise ValueError('FP8 recovery benchmark/planner build differs')
    if keys != {(HYBRID, group, attention) for group in (16, 32)
                for attention in ('dense', 'b128-s16-tau900')}:
        raise ValueError('FP8 recovery may replace only the complete hybrid profile quartet')
    dense16 = next(item for item in mappings if item['kv_value_group'] == 16
                   and item['xattention_profile'] == 'dense')
    if executable != Path(dense16['new_benchmark']['path']).parent.parent / 'src/ninfer_r9700_fp8_gate_up_qual':
        raise ValueError('FP8 recovery physical executable differs from new build')
    if receipt['build_cache'] != dense16['build_cache']:
        raise ValueError('FP8 recovery physical qualification differs from frozen build cache')
    retained = value.get('retained_nonhybrid_capacity')
    if not isinstance(retained, dict) or len(retained) != 40:
        raise ValueError('FP8 recovery requires eight retained nonhybrid manifests and 32 reports')
    for path, digest in retained.items():
        checked_file({'path': path, 'sha256': digest})
    validate_model_startup(checked_file(value.get('model_startup')), value)
    return value


def validate_qualification(path: Path, review: dict) -> dict:
    receipt = json.loads(path.read_text())
    if (receipt.get('artifact_type') != 'ninfer_r9700_fp8_context_physical_receipt'
            or receipt.get('schema_version') != 1 or receipt.get('pass') is not True):
        raise ValueError('FP8 recovery lacks a bound physical qualification receipt')
    report = checked_file(receipt.get('report'))
    check_physical(json.loads(report.read_text()))
    executable = checked_file(receipt.get('executable'))
    if receipt.get('command') != [str(executable), '--shared-context', '--output', str(report)]:
        raise ValueError('FP8 recovery qualification command differs from retained report')
    sources = receipt.get('sources')
    if not isinstance(sources, list) or sources != review['sources']:
        raise ValueError('FP8 recovery physical run differs from reviewed allocation sources')
    qualifier_source = checked_file(receipt.get('qualifier_source'))
    if qualifier_source != REPO / 'tools/r9700/fp8_gate_up_qual.hip':
        raise ValueError('FP8 recovery lacks the actual shared-context oracle source')
    for item in sources:
        checked_file(item)
    prelaunch = json.loads(checked_file(receipt.get('prelaunch_inputs')).read_text())
    if prelaunch != {key: receipt[key] for key in (
            'executable', 'command', 'sources', 'qualifier_source', 'build_cache')}:
        raise ValueError('FP8 recovery qualification differs from prelaunch frozen inputs')
    checked_file(receipt.get('build_cache'))
    return receipt


def cache_profile(path: Path) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line:
            name, value = line.split('=', 1)
            name = name.split(':', 1)[0]
            if (name.startswith('NINFER_') and not name.endswith('-STRINGS')) or name in (
                    'CMAKE_BUILD_TYPE', 'CMAKE_HIP_ARCHITECTURES',
                    'CMAKE_HIP_FLAGS', 'CMAKE_HIP_FLAGS_RELEASE',
                    'CMAKE_CXX_FLAGS', 'CMAKE_CXX_FLAGS_RELEASE'):
                result[name] = value
    return result


def check_startup_headroom(report: dict) -> dict:
    memory = report.get('memory', {})
    observed = memory.get('available_after_startup_bytes')
    planned = memory.get('planned_slack_bytes')
    headroom = memory.get('kv_capacity_headroom_bytes')
    if (type(observed) is not int or type(planned) is not int
            or observed < 0 or planned < 0 or headroom != 1024**3
            or observed < planned or observed < headroom):
        raise ValueError('hybrid startup consumed unaccounted planned slack; capacity blocked')
    return {key: memory[key] for key in (
        'available_after_startup_bytes', 'planned_slack_bytes', 'kv_capacity_headroom_bytes')}


def validate_recovered_capacity(source: dict, bridge: dict) -> None:
    if source.get('artifact', {}).get('weights_id') != HYBRID:
        return
    key = (source['cache_value_group'], source['quality']['representation']['xattention_profile'])
    mapping, = [item for item in bridge['hybrid_builds']
                if (item['kv_value_group'], item['xattention_profile']) == key]
    for preset, evidence in source['matrices'].items():
        manifest = json.loads(checked_file({key: evidence[key] for key in ('path', 'sha256')}).read_text())
        if (manifest['bench'] != mapping['new_benchmark']
                or manifest.get('hybrid_shared_workspace_authority', {}).get('tool') != mapping['new_planner']):
            raise ValueError('recovered capacity/whole differs from exact fresh hybrid build/planner')
        if preset == 'pareto-capacity':
            for record in evidence['reports']:
                check_startup_headroom(json.loads(checked_file(record).read_text()))


def validate_model_startup(path: Path, bridge: dict) -> dict:
    value = json.loads(path.read_text())
    return validate_model_startup_value(value, bridge)


def validate_model_startup_value(value: dict, bridge: dict) -> dict:
    from tools.bench import run_ninfer_bench_matrix as bench
    if (value.get('artifact_type') != 'ninfer_r9700_fp8_context_model_startup'
            or value.get('schema_version') != 1 or value.get('pass') is not True):
        raise ValueError('FP8 recovery requires the real-model startup accounting proof')
    inputs = json.loads(checked_file(value.get('inputs')).read_text())
    mapping, = [item for item in bridge['hybrid_builds']
                if item['kv_value_group'] == 16 and item['xattention_profile'] == 'dense']
    files = inputs.get('files')
    if not isinstance(files, list):
        raise ValueError('FP8 model startup lacks prelaunch identities')
    for item in files:
        checked_file(item)
    required = [{key: mapping[field][key] for key in ('path', 'sha256')}
                for field in ('new_benchmark', 'new_planner')]
    required += [bridge['chunk_selection'],
                bridge['qualification'], bridge['source_review']]
    if any(item not in files for item in required):
        raise ValueError('FP8 model startup differs from recovered benchmark/planner/proof')
    artifact = mapping['artifact']
    if any(inputs.get('artifact', {}).get(key) != artifact[key] for key in (
            'path', 'weights_id', 'sha256', 'file_size_bytes')):
        raise ValueError('FP8 model startup artifact differs from recovered hybrid recipe')
    report = checked_file(value.get('report'))
    workload, = bench.build_cases('pareto-capacity', production_prefill_chunk=2048)
    corpus = REPO / 'bench/fixtures/bench_corpus.ids'
    if identity(corpus) not in files:
        raise ValueError('FP8 model startup corpus changed')
    expected = bench.add_repetition_args([
        mapping['new_benchmark']['path'], '--weights', artifact['path'], '--corpus', str(corpus),
        '--device', '0', '--concurrency', '1', *workload.args,
        '--output', 'json', '--output-file', str(report)], workload, None, None)
    if inputs.get('command') != expected:
        raise ValueError('FP8 model startup command differs from exact C1 capacity geometry')
    loaded = bench.load_bench_report(report, 16, 8, 8, True, 1, artifact, expected, workload, 'dense')
    bench.validate_automatic_feasibility(loaded)
    check_startup_headroom(loaded)
    if value.get('memory') != loaded['memory']:
        raise ValueError('FP8 model startup memory summary differs from raw report')
    return value


def resolved_benchmark(source: dict, bridge: dict | None) -> dict:
    key = (source['weights_id'], source['kv_value_group'], source['xattention_profile'])
    if bridge is None or key[0] != HYBRID:
        return source['benchmark_executable']
    matches = [item for item in bridge['hybrid_builds'] if (
        item['weights_id'], item['kv_value_group'], item['xattention_profile']) == key]
    if len(matches) != 1 or matches[0]['old_benchmark'] != source['benchmark_executable']:
        raise ValueError('FP8 recovery does not bind requested hybrid profile')
    return matches[0]['new_benchmark']
