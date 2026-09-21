"""CPU checks for narrowly reusing numerical evidence after resource recovery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.ppl import fp8_context_recovery as recovery
from tools.ppl.assemble_pareto import validate_chunk_candidate_bindings


def physical():
    return {
        'artifact_type': 'ninfer_r9700_fp8_shared_context_qualification', 'schema_version': 1,
        'rows': [34816, 7168, 4096], 'columns': 5120, 'prepared_widths': [1, 2, 3, 4, 2048],
        'oracle_widths': [4, 2048], 'graph_replays_per_width': 2,
        'independent_context_fingerprints_match': True, 'pass': True,
        'oracle': 'fp64_public_bf16_times_exact_e4m3_weight_and_stored_fp32_scale',
        'graph_outputs_poisoned_before_replay': True,
        'prepared_before_binding': True, 'unbound_run_rejected': True,
        'maximum_bf16_steps': 0, 'pci_bus_id': '0000:13:00.0',
        'hip_runtime_version': 10000000, 'hipblaslt_version': 100000,
        **{key: 20 * 1024**3 for key in ('free_before_context', 'free_after_context',
            'free_before_prepare', 'free_after_prepare', 'free_after_binding', 'free_after_execution')},
        'profiles': [{'rows': n, 'tokens': t, 'fingerprint': 'a' * 32,
                      'matmul_workspace_bytes': 0}
                     for n in (34816, 7168, 4096) for t in (1, 2, 3, 4, 2048)],
    }


def startup_report():
    return {
        'config': {'max_context': 262144, 'concurrency': 1, 'kv_cache_format': 'fp8-k-int4-v'},
        'memory': {'max_context': 262144, 'kv_cache_format': 'fp8-k-int4-v',
            'kv_capacity': 262144, 'kv_capacity_page_groups': 4096, 'kv_capacity_max_page_groups': 4096,
            'minimum_runtime_reservation_bytes': 1000, 'kv_capacity_increment_bytes': 0,
            'runtime_reservation_bytes': 1000, 'available_after_weights_bytes': 1000 + 2 * 1024**3,
            'planned_slack_bytes': 2 * 1024**3, 'available_after_startup_bytes': 2 * 1024**3,
            'kv_capacity_headroom_bytes': 1024**3, 'device_graph_allowance_bytes': 4096,
            'device_graph_observed_bytes': 3072},
    }


class RecoveryTest(unittest.TestCase):
    def test_complete_public_oracle_graph_and_algorithm_evidence_required(self):
        value = physical()
        recovery.check_physical(value)
        for key, changed in (
            ('oracle', 'private_fp8_activation_staging'),
            ('graph_outputs_poisoned_before_replay', False), ('maximum_bf16_steps', 2),
            ('independent_context_fingerprints_match', False), ('graph_replays_per_width', 1),
            ('prepared_widths', [1, 4, 2048]), ('free_before_prepare', None),
            ('pci_bus_id', '0000:7c:00.0'), ('pass', False),
            ('profiles', value['profiles'][:-1]),
            ('prepared_before_binding', False), ('unbound_run_rejected', False),
            ('free_after_binding', value['free_after_prepare'] - 1),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                recovery.check_physical({**value, key: changed})

    def test_actual_startup_free_must_cover_planned_slack_not_just_one_gib(self):
        memory = {'available_after_startup_bytes': 5213519872,
                  'planned_slack_bytes': 5214743808, 'kv_capacity_headroom_bytes': 1024**3}
        with self.assertRaisesRegex(ValueError, 'unaccounted planned slack'):
            recovery.check_startup_headroom({'memory': memory})
        memory['available_after_startup_bytes'] = memory['planned_slack_bytes']
        self.assertEqual(recovery.check_startup_headroom({'memory': memory}), memory)
        memory['available_after_startup_bytes'] += 32 * 1024**2
        recovery.check_startup_headroom({'memory': memory})

    def test_real_executable_identity_shape_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bench'
            path.write_bytes(b'benchmark')
            record = recovery.executable_identity(path)
            self.assertEqual(set(record), {'path', 'sha256', 'file_size_bytes'})
            self.assertEqual(recovery.checked_file(record), path)
            with self.assertRaises(ValueError):
                recovery.checked_file({**record, 'file_size_bytes': 1})

    def fixture(self, root):
        def write(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value) if not isinstance(value, str) else value)
            return recovery.identity(path)
        sources = [write(name, 'reviewed source') for name in recovery.SOURCE_FILES]
        review = write('review.json', {
            'artifact_type': 'ninfer_r9700_fp8_context_source_review', 'schema_version': 1,
            'status': 'unchanged_arithmetic', 'scope': recovery.SCOPE,
            'arithmetic_changed': False, 'kernel_or_recipe_changed': False, 'sources': sources})
        mappings, selected, frozen = [], [], {}
        for recipe in ('q4', 'mixed', recovery.HYBRID):
            for group in (16, 32):
                for attention in ('dense', 'b128-s16-tau900'):
                    old = root / f'old-{attention}-{group}'
                    new = root / f'new-{attention}-{group}'
                    for build in (old, new):
                        write(str(build.relative_to(root) / 'bench/ninfer_bench'), 'bench')
                        write(str(build.relative_to(root) / 'src/ninfer_r9700_runtime_planner_qual'), 'planner')
                        write(str(build.relative_to(root) / 'CMakeCache.txt'),
                              f'CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_HIP_ARCHITECTURES:STRING=gfx1201\n'
                              f'NINFER_R9700_KV_VALUE_GROUP:STRING={group}\n')
                    old_bench = recovery.executable_identity(old / 'bench/ninfer_bench')
                    artifact = {'path': str(root / f'{recipe}.ninfer'), 'weights_id': recipe,
                                'sha256': '0' * 64, 'file_size_bytes': 32, 'conversion_receipt': {}}
                    selected.append({'weights_id': recipe, 'kv_value_group': group,
                        'xattention_profile': attention, 'artifact': artifact,
                        'benchmark_executable': old_bench})
                    if recipe != recovery.HYBRID:
                        continue
                    mapping = {'weights_id': recipe, 'kv_value_group': group,
                        'xattention_profile': attention, 'artifact': artifact,
                        'old_benchmark': old_bench,
                        'new_benchmark': recovery.executable_identity(new / 'bench/ninfer_bench'),
                        'old_planner': recovery.executable_identity(old / 'src/ninfer_r9700_runtime_planner_qual'),
                        'new_planner': recovery.executable_identity(new / 'src/ninfer_r9700_runtime_planner_qual'),
                        'old_build_cache': recovery.identity(old / 'CMakeCache.txt'),
                        'build_cache': recovery.identity(new / 'CMakeCache.txt')}
                    mappings.append(mapping)
                    for key in ('old_benchmark', 'old_planner', 'old_build_cache'):
                        item = mapping[key]
                        frozen[str(Path(item['path']).relative_to(root))] = item['sha256']
        selection_value = {'selected_prefill_chunk': 2048, 'sources': selected}
        selection = write('selection.json', selection_value)
        quality = write('quality.json', {'selected_prefill_chunk': 2048,
                                        'selected_prefill_chunk_authority': selection})
        executable = write('new-dense-16/src/ninfer_r9700_fp8_gate_up_qual', 'qualifier')
        oracle_source = write('tools/r9700/fp8_gate_up_qual.hip', 'public oracle')
        report = write('qualification/report.json', physical())
        inputs = {'sources': sources, 'executable': executable, 'qualifier_source': oracle_source,
                  'build_cache': mappings[0]['build_cache'],
                  'command': [executable['path'], '--shared-context', '--output', report['path']]}
        receipt = write('qualification/receipt.json', {
            'artifact_type': 'ninfer_r9700_fp8_context_physical_receipt', 'schema_version': 1,
            'pass': True, **inputs, 'report': report,
            'prelaunch_inputs': write('qualification/inputs.json', inputs)})
        retained = {}
        for i in range(40):
            item = write(f'old-capacity/{i}.json', {})
            retained[item['path']] = item['sha256']
        value = {
            'artifact_type': 'ninfer_r9700_fp8_context_resource_recovery', 'schema_version': 1,
            'scope': recovery.SCOPE, 'preserves': ['numerical_quality', 'selected_prefill_chunk'],
            'capacity_or_speed_reuse': False, 'selected_prefill_chunk': 2048,
            'chunk_selection': selection, 'quality_authorities': quality,
            'frozen_panel_inputs': write('frozen.json', {'files': frozen}),
            'stopped_capacity': write('closure.json', {'status': 'stopped_unaccounted_fp8_library_allocation'}),
            'source_review': review, 'qualification': receipt, 'hybrid_builds': mappings,
            'retained_nonhybrid_capacity': retained, 'model_startup': write('startup.json', {}),
        }
        return value, selection_value

    def test_bridge_requires_bound_source_physical_and_exact_four_builds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(recovery, 'REPO', root), \
                    patch.object(recovery, 'validate_model_startup') as startup:
                value, selection = self.fixture(root)
                recovery.validate_bridge_value(value, selection)
                startup.assert_called_once()
                for mutation in ('wrong_recipe', 'wrong_profile', 'missing_mapping', 'changed_artifact',
                                 'missing_physical', 'changed_original_benchmark'):
                    changed = copy.deepcopy(value)
                    first = changed['hybrid_builds'][0]
                    if mutation == 'wrong_recipe': first['weights_id'] = 'q4'
                    if mutation == 'wrong_profile': first['kv_value_group'] = 64
                    if mutation == 'missing_mapping': changed['hybrid_builds'].pop()
                    if mutation == 'changed_artifact': first['artifact']['sha256'] = '1' * 64
                    if mutation == 'missing_physical': changed['qualification']['path'] += '.absent'
                    if mutation == 'changed_original_benchmark': first['old_benchmark']['sha256'] = '1' * 64
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        recovery.validate_bridge_value(changed, selection)
                Path(value['hybrid_builds'][0]['new_benchmark']['path']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'bound file changed'):
                    recovery.validate_bridge_value(value, selection)

    def test_chunk_join_uses_new_hybrid_only_and_requires_bound_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(recovery, 'REPO', root):
                bridge, selection = self.fixture(root)
            provenance = [{'artifact': row['artifact'], 'cache_value_group': row['kv_value_group'],
                           'quality': {'representation': {'xattention_profile': row['xattention_profile']}},
                           'benchmark_executable': recovery.resolved_benchmark(row, bridge)}
                          for row in selection['sources']]
            with self.assertRaisesRegex(ValueError, 'selection identity'):
                validate_chunk_candidate_bindings(selection, provenance)
            with patch.object(recovery, 'validate_recovered_capacity') as capacity:
                validate_chunk_candidate_bindings(selection, provenance, bridge)
                self.assertEqual(capacity.call_count, 12)
                provenance[0]['benchmark_executable'] = bridge['hybrid_builds'][0]['new_benchmark']
                with self.assertRaisesRegex(ValueError, 'selection identity'):
                    validate_chunk_candidate_bindings(selection, provenance, bridge)

    def test_hybrid_capacity_requires_current_planner_and_observed_headroom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bench = {'path': '/new/bench', 'sha256': 'a' * 64, 'file_size_bytes': 8}
            planner = {'path': '/new/planner', 'sha256': 'b' * 64, 'file_size_bytes': 9}
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'bench': bench, 'hybrid_shared_workspace_authority': {'tool': planner}}))
            report = root / 'report.json'
            report.write_text(json.dumps({'memory': {'available_after_startup_bytes': 2 * 1024**3,
                'planned_slack_bytes': 2 * 1024**3, 'kv_capacity_headroom_bytes': 1024**3}}))
            source = {'artifact': {'weights_id': recovery.HYBRID}, 'cache_value_group': 16,
                      'quality': {'representation': {'xattention_profile': 'dense'}},
                      'matrices': {'pareto-capacity': {**recovery.identity(manifest),
                                                     'reports': [recovery.identity(report)]}}}
            bridge = {'hybrid_builds': [{'kv_value_group': 16, 'xattention_profile': 'dense',
                                        'new_benchmark': bench, 'new_planner': planner}]}
            recovery.validate_recovered_capacity(source, bridge)
            bridge['hybrid_builds'][0]['new_planner'] = {**planner, 'sha256': 'c' * 64}
            with self.assertRaisesRegex(ValueError, 'exact fresh'):
                recovery.validate_recovered_capacity(source, bridge)


if __name__ == '__main__':
    unittest.main()
