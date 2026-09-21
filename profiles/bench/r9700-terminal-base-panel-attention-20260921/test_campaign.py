"""CPU-only regression checks for current terminal-base orchestration."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('terminal_base_campaign', Path(__file__).with_name('campaign.py'))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTest(unittest.TestCase):
    def state(self):
        return {'bindings': {}, 'authority': {'selected_prefill_chunk': 2048, 'sha256': 'selection'},
                'quality': {'authorities': {
                    key: {'path': f'/quality/{key}/results.json'}
                    for key in campaign.EXPECTED_AUTHORITIES}}}

    def test_missing_quality_preflight_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(campaign, 'QUALITY', root / 'missing.json'), \
                    patch.object(campaign.subprocess, 'run') as execute:
                with self.assertRaisesRegex(ValueError, 'quality authority map'):
                    campaign.preflight()
                execute.assert_not_called()
                self.assertEqual(list(root.iterdir()), [])

    def test_matrix_commands_cover_exact_profiles_and_selected_chunk(self):
        cases = list(campaign.candidates())
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({tuple(campaign.identity(case)) for case in cases}), 12)
        for case in cases:
            for stage in ('capacity', 'whole'):
                args = campaign.command(case, stage, self.state())
                self.assertEqual(args[args.index('--prefill-chunk') + 1], '2048')
                self.assertEqual([args[i + 1] for i, v in enumerate(args) if v == '--concurrency'],
                                 ['1', '2', '3', '4'])
                self.assertEqual('--require-post-chunk-capacity' in args, stage == 'capacity')
                self.assertEqual('--require-fp8-hybrid' in args, 'four-role' in case['weights_id'])
                self.assertIn('--no-build', args)
                self.assertNotIn('--resume', args)
                self.assertNotIn('--repetitions', args)
                self.assertIn('build-r9700-selection-panel-', args[args.index('--bench') + 1])

    def test_preflight_rejects_changed_frozen_benchmark_artifact_or_corpus(self):
        for changed_kind in ('benchmark', 'artifact', 'corpus'):
            with self.subTest(changed_kind=changed_kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(campaign, 'REPO', root):
                    cases = list(campaign.candidates())
                corpus = root / 'corpus.ids'
                corpus.write_text('1 2 3')
                files = {corpus}
                artifacts = {}
                for case in cases:
                    case['artifact'].parent.mkdir(exist_ok=True)
                    case['artifact'].write_bytes(b'artifact')
                    artifacts[case['weights_id']] = {
                        'path': str(case['artifact']), 'weights_id': case['weights_id'],
                        'bytes': 8, 'sha256': campaign.run.file_sha256(case['artifact']),
                        'conversion_receipt': {'receipt': 'fixture'},
                    }
                    for name in ('bench/ninfer_bench', 'src/ninfer_r9700_runtime_planner_qual'):
                        path = case['build'] / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b'executable')
                        path.chmod(0o755)
                        files.add(path)
                selection, frozen, quality_path = (root / name for name in ('selection.json', 'inputs.json', 'quality.json'))
                selection.write_text('{}')
                quality_path.write_text('{}')
                authority = {'path': str(selection), 'sha256': campaign.run.file_sha256(selection), 'selected_prefill_chunk': 2048}
                quality = {'selected_prefill_chunk': 2048, 'selected_prefill_chunk_authority': {
                    key: authority[key] for key in ('path', 'sha256')}, 'authorities': {}}
                sources = []
                for case in cases:
                    artifact = artifacts[case['weights_id']]
                    summary = dict(artifact)
                    summary['file_size_bytes'] = summary.pop('bytes')
                    quality['authorities'][case['quality_name']] = {'artifact': {
                        key: summary[key] for key in ('weights_id', 'sha256', 'file_size_bytes', 'conversion_receipt')}}
                    bench = case['build'] / 'bench/ninfer_bench'
                    sources.append({'weights_id': case['weights_id'], 'kv_value_group': case['group'],
                                    'xattention_profile': case['attention'], 'artifact': summary,
                                    'benchmark_executable': {'path': str(bench), 'sha256': campaign.run.file_sha256(bench)}})
                frozen.write_text(json.dumps({'artifact_type': 'ninfer_r9700_chunk_campaign_inputs',
                    'schema_version': 1, 'artifacts': list(artifacts.values()),
                    'files': {str(path.relative_to(root)): campaign.run.file_sha256(path) for path in files}}))
                def inspect(path):
                    original = next(item for item in artifacts.values() if item['path'] == str(path))
                    return {**original, 'bytes': path.stat().st_size, 'sha256': campaign.run.file_sha256(path)}
                with patch.object(campaign, 'REPO', root), patch.object(campaign, 'CORPUS', corpus), \
                        patch.object(campaign, 'SELECTION', selection), patch.object(campaign, 'FROZEN', frozen), \
                        patch.object(campaign, 'QUALITY', quality_path), \
                        patch.object(campaign, 'validate_authority_map', return_value=quality), \
                        patch.object(campaign, 'validate_prefill_chunk_authority', return_value=(authority, {'sources': sources})), \
                        patch.object(campaign.run, 'inspect_candidate_artifact', side_effect=inspect), \
                        patch.object(campaign.subprocess, 'run') as execute:
                    campaign.preflight()
                    changed = {'benchmark': cases[0]['build'] / 'bench/ninfer_bench',
                               'artifact': cases[0]['artifact'], 'corpus': corpus}[changed_kind]
                    changed.write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError, 'differs from frozen'):
                        campaign.preflight()
                    execute.assert_not_called()

    def test_capacity_collects_all_outcomes_before_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            root = Path(directory)
            def execute(args, output, name):
                calls.append(name)
                return 1 if len(calls) == 1 else 0
            def validate(selection, matrices, *, executed):
                self.assertEqual(len(calls), 12)
                self.assertEqual(len(matrices), 12)
                self.assertTrue(executed)
                outcomes = json.loads((root / 'capacity/runner-outcomes.json').read_text())
                self.assertEqual(outcomes[0]['returncode'], 1)
                return {'whole_eligible_identities': []}
            with patch.object(campaign, 'PACKAGE', root), \
                    patch.object(campaign, 'execute', side_effect=execute), \
                    patch.object(campaign, 'validate_campaign', side_effect=validate):
                campaign.capacity(self.state())
                self.assertTrue((root / 'capacity/validation.json').exists())
                with self.assertRaisesRegex(ValueError, 'namespace already exists'):
                    campaign.capacity(self.state())
                self.assertEqual(len(calls), 12)

    def test_invalid_or_asymmetric_capacity_blocks_whole_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(campaign, 'PACKAGE', root), \
                    patch.object(campaign, 'capacity_evidence', side_effect=ValueError('asymmetric controls')), \
                    patch.object(campaign, 'execute') as execute:
                with self.assertRaisesRegex(ValueError, 'asymmetric'):
                    campaign.whole(self.state())
                execute.assert_not_called()
                self.assertEqual(list(root.iterdir()), [])

    def test_tuple_validator_identities_survive_publication_revalidation_and_whole(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = list(campaign.candidates())
            identities = [tuple(campaign.identity(case)) for case in cases]
            # Match the real validator's sorted list-of-tuples return shape, not JSON fixtures.
            result = {'identities': sorted(identities),
                      'capacity_eligible_identities': sorted(identities[:2]),
                      'whole_eligible_identities': sorted(identities[:2])}
            with patch.object(campaign, 'PACKAGE', root), \
                    patch.object(campaign, 'validate_campaign', return_value=result), \
                    patch.object(campaign, 'execute', return_value=0) as execute:
                campaign.capacity(self.state())
                recorded = json.loads((root / 'capacity/validation.json').read_text())
                self.assertEqual(campaign.capacity_evidence(self.state()), recorded)
                self.assertIsInstance(recorded['whole_eligible_identities'][0], list)
                execute.reset_mock()
                campaign.whole(self.state())
                self.assertEqual([call.args[2] for call in execute.call_args_list],
                                 [case['name'] for case in cases[:2]])

    def test_whole_runs_only_validated_eligible_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            eligible = [campaign.identity(case) for case in list(campaign.candidates())[:2]]
            with patch.object(campaign, 'PACKAGE', Path(directory)), \
                    patch.object(campaign, 'capacity_evidence', return_value={'whole_eligible_identities': eligible}), \
                    patch.object(campaign, 'execute', return_value=0) as execute:
                campaign.whole(self.state())
                self.assertEqual(execute.call_count, 2)
                self.assertEqual([call.args[2] for call in execute.call_args_list],
                                 [case['name'] for case in list(campaign.candidates())[:2]])

    def test_changed_stage_bindings_block_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'capacity').mkdir()
            (root / 'capacity/inputs.json').write_text('{"changed": "identity"}')
            with patch.object(campaign, 'PACKAGE', root), patch.object(campaign, 'validate_campaign') as validate:
                with self.assertRaisesRegex(ValueError, 'bindings differ'):
                    campaign.capacity_evidence(self.state())
                validate.assert_not_called()

    def test_select_passes_all_candidates_with_exclusions_and_exclusive_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.state()
            eligible = [campaign.identity(case) for case in list(campaign.candidates())[:2]]
            def execute(args, output, name):
                rows = [args[i + 1:i + 7] for i, value in enumerate(args) if value == '--candidate']
                self.assertEqual(len(rows), 12)
                self.assertEqual(sum(row[-1] == '-' for row in rows), 10)
                self.assertEqual(rows[0][3], state['quality']['authorities']['ALL_Q4_DENSE_QUALITY']['path'])
                Path(args[args.index('--out') + 1]).write_text('{}')
                return 0
            with patch.object(campaign, 'PACKAGE', root), \
                    patch.object(campaign, 'capacity_evidence', return_value={'whole_eligible_identities': eligible}), \
                    patch.object(campaign, 'require_stage_bindings'), \
                    patch.object(campaign, 'execute', side_effect=execute), \
                    patch.object(campaign.pareto, 'load_payload', return_value={}), \
                    patch.object(campaign.pareto, 'classify', return_value={'schema_version': 7}), \
                    patch.object(campaign, 'publish') as publish:
                campaign.select(state)
                publish.assert_called_once()
                pending_input, final_input, pending_result, final_result = publish.call_args.args
                result = json.loads(pending_result.read_text())
                self.assertEqual(result['pareto_input']['path'], str(final_input))
                self.assertEqual(result['pareto_input']['sha256'], campaign.run.file_sha256(pending_input))
                self.assertEqual(final_result, root / 'select/result.json')
                self.assertEqual(publish.call_args.kwargs['guard_sha256'], 'selection')


if __name__ == '__main__':
    unittest.main()
