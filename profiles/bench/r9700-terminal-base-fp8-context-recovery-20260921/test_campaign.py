"""CPU-only recovery orchestration checks; physical gates remain mandatory."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('accounted_campaign', Path(__file__).with_name('campaign.py'))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTest(unittest.TestCase):
    def state(self, root):
        proof = root / 'startup-proof.json'
        proof.write_text('{}')
        return {'authority': {'selected_prefill_chunk': 2048}, 'bindings': {},
                'recovery': {'model_startup': campaign.recovery.identity(proof)},
                'quality': {'authorities': {
                    name: {'path': f'/quality/{name}.json',
                           'eligibility_by_group': {'16': name != 'MIXED_XATTENTION_QUALITY',
                                                    '32': name != 'MIXED_XATTENTION_QUALITY'}}
                    for name in campaign.EXPECTED_AUTHORITIES}}}

    def test_exact_eight_old_capacity_and_four_new_builds(self):
        cases = list(campaign.candidates())
        old, new = [], []
        for case in cases:
            root = campaign.matrix(case, 'capacity')
            if case['weights_id'] == campaign.recovery.HYBRID:
                new.append(root)
                self.assertEqual(root.parent, campaign.PACKAGE / 'capacity')
                self.assertIn('build-r9700-fp8-accounted', str(case['build']))
            else:
                old.append(root)
                self.assertEqual(root.parent, campaign.OLD / 'capacity')
                self.assertEqual(case['build'], case['old_build'])
            self.assertEqual(campaign.matrix(case, 'whole').parent, campaign.PACKAGE / 'whole')
        self.assertEqual((len(old), len(new)), (8, 4))

    def test_commands_keep_original_semantics_without_resume_or_build(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.state(Path(directory))
            for case in campaign.candidates():
                for stage in ('capacity', 'whole'):
                    command = campaign.command(case, stage, state)
                    self.assertEqual(command[command.index('--bench') + 1],
                                     str(case['build'] / 'bench/ninfer_bench'))
                    self.assertEqual(command[command.index('--prefill-chunk') + 1], '2048')
                    self.assertEqual([command[i + 1] for i, item in enumerate(command)
                                      if item == '--concurrency'], ['1', '2', '3', '4'])
                    self.assertEqual('--require-fp8-hybrid' in command,
                                     case['weights_id'] == campaign.recovery.HYBRID)
                    self.assertIn('--no-build', command)
                    self.assertNotIn('--resume', command)

    def test_only_four_hybrid_matrices_execute_then_rebuild_all_twelve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.state(root)
            calls = []
            def execute(command, output, name):
                calls.append(name)
                return 0
            def validate(selection, roots, *, executed):
                self.assertEqual(len(calls), 4)
                self.assertTrue(all(name.startswith('four_role_') for name in calls))
                self.assertEqual(len(roots), 12)
                self.assertEqual(sum(path.parent.parent == campaign.OLD for path in roots), 8)
                return {'whole_eligible_identities': [tuple(campaign.identity(case))
                        for case in campaign.candidates()]}
            with patch.object(campaign, 'PACKAGE', root), \
                    patch.object(campaign.recovery, 'validate_model_startup'), \
                    patch.object(campaign, 'validate_recovered_headroom'), \
                    patch.object(campaign, 'validate_campaign', side_effect=validate), \
                    patch.object(campaign, 'execute', side_effect=execute):
                campaign.capacity(state)
                self.assertEqual(len(json.loads((root / 'capacity/runner-outcomes.json').read_text())), 4)
                evidence = campaign.capacity_evidence(state)
                self.assertIsInstance(evidence['whole_eligible_identities'][0], list)
                self.assertEqual(len(campaign.whole_eligible(state)), 10)
                with self.assertRaisesRegex(ValueError, 'namespace already exists'):
                    campaign.capacity(state)
                self.assertEqual(len(calls), 4)

    def test_first_model_accounting_failure_blocks_all_capacity_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.state(root)
            with patch.object(campaign, 'PACKAGE', root), \
                    patch.object(campaign.recovery, 'validate_model_startup',
                                 side_effect=ValueError('unaccounted planned slack')), \
                    patch.object(campaign, 'execute') as execute:
                with self.assertRaisesRegex(ValueError, 'unaccounted'):
                    campaign.capacity(state)
                execute.assert_not_called()
                self.assertFalse((root / 'capacity').exists())

    def test_missing_bridge_preflight_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(campaign, 'original_inputs', return_value={'selection': {}}), \
                    patch.object(campaign, 'BRIDGE', root / 'absent.json'), \
                    patch.object(campaign, 'execute') as execute:
                with self.assertRaises(FileNotFoundError):
                    campaign.preflight()
                execute.assert_not_called()
                self.assertEqual(list(root.iterdir()), [])

    def test_qualification_does_not_overwrite_existing_physical_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / 'failed.txt'
            existing.write_text('retained failure')
            with patch.object(campaign, 'QUALIFICATION', root), \
                    patch.object(campaign, 'ready_build'), \
                    patch.object(campaign, 'execute') as execute:
                with self.assertRaisesRegex(ValueError, 'namespace already exists'):
                    campaign.qualify()
                execute.assert_not_called()
                self.assertEqual(existing.read_text(), 'retained failure')

    def test_controls_resolve_new_hybrid_benchmark_only(self):
        with patch.dict(sys.modules, {'campaign': campaign}):
            spec = importlib.util.spec_from_file_location('accounted_controls', Path(__file__).with_name('controls.py'))
            controls = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(controls)
        case = next(case for case in campaign.candidates() if case['weights_id'] == campaign.recovery.HYBRID)
        old = {'path': str(case['old_build'] / 'bench/ninfer_bench'), 'sha256': 'a', 'file_size_bytes': 8}
        new = {'path': str(case['build'] / 'bench/ninfer_bench'), 'sha256': 'b', 'file_size_bytes': 9}
        source = {'weights_id': case['weights_id'], 'kv_value_group': case['group'],
                  'xattention_profile': case['attention'], 'artifact': {}, 'benchmark_executable': old}
        state = {'authority': {'selected_prefill_chunk': 2048}, 'recovery': {'hybrid_builds': [{
            **{key: source[key] for key in ('weights_id', 'kv_value_group', 'xattention_profile')},
            'old_benchmark': old, 'new_benchmark': new}]}}
        command = [new['path'], '--weights', str(case['artifact']), '--corpus', str(campaign.CORPUS), '--device', '0']
        manifest = {'artifact': {}, 'bench': new, 'corpus': str(campaign.CORPUS), 'corpus_sha256': 'corpus',
                    'prefill_chunk_authority': state['authority'], 'selected_prefill_chunk': 2048,
                    'commands': [{'command': command}]}
        controls.validate_graph_binding(manifest, case, state, source, 'corpus')
        with self.assertRaisesRegex(ValueError, 'current artifact/build'):
            controls.validate_graph_binding({**manifest, 'bench': old}, case, state, source, 'corpus')

    def test_startup_producer_real_receipt_roundtrip_and_strict_accounting_failure(self):
        from tools.ppl.test_fp8_context_recovery import RecoveryTest, startup_report
        from tools.bench import run_ninfer_bench_matrix as bench
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(campaign.recovery, 'REPO', root):
                    bridge, selection = RecoveryTest().fixture(root)
                    mapping = bridge['hybrid_builds'][0]
                    corpus = root / 'bench/fixtures/bench_corpus.ids'
                    corpus.parent.mkdir(parents=True)
                    corpus.write_text('1 2 3')
                    case = {'weights_id': campaign.recovery.HYBRID, 'group': 16, 'attention': 'dense',
                            'build': root / 'new-dense-16', 'old_build': root / 'old-dense-16',
                            'artifact': Path(mapping['artifact']['path'])}
                    report = startup_report()
                    if fails:
                        report['memory']['available_after_startup_bytes'] -= 1
                    def execute(command, output, name):
                        Path(command[command.index('--output-file') + 1]).write_text(json.dumps(report))
                        return 0
                    with patch.object(campaign, 'REPO', root), \
                            patch.object(campaign, 'STARTUP', root / 'startup-accounted'), \
                            patch.object(campaign, 'QUALIFICATION', root / 'qualification'), \
                            patch.object(campaign, 'SELECTION', root / 'selection.json'), \
                            patch.object(campaign, 'CORPUS', corpus), \
                            patch.object(campaign, 'candidates', return_value=iter([case])), \
                            patch.object(campaign, 'ready_build'), \
                            patch.object(campaign, 'validate_prefill_chunk_authority', return_value=(
                                {'selected_prefill_chunk': 2048}, selection)), \
                            patch.object(campaign, 'execute', side_effect=execute), \
                            patch.object(bench, 'inspect_artifact', return_value=mapping['artifact']), \
                            patch.object(bench, 'require_auto_power_profile'), \
                            patch.object(bench, 'require_hip_pci_device'), \
                            patch.object(bench, 'load_bench_report',
                                         side_effect=lambda path, *args: json.loads(path.read_text())) as loader:
                        if fails:
                            with self.assertRaisesRegex(ValueError, 'unaccounted planned slack'):
                                campaign.startup(root / 'review.json')
                            self.assertFalse((root / 'startup-accounted/result.json').exists())
                            self.assertTrue((root / 'startup-accounted/model-c1.json').exists())
                        else:
                            campaign.startup(root / 'review.json')
                            result = campaign.recovery.validate_model_startup(
                                root / 'startup-accounted/result.json', bridge)
                            self.assertEqual(result['memory'], report['memory'])
                            self.assertEqual(loader.call_args.args[1:6], (16, 8, 8, True, 1))
                            altered = json.loads((root / 'startup-accounted/inputs.json').read_text())
                            altered['command'][altered['command'].index('--concurrency') + 1] = '2'
                            (root / 'startup-accounted/inputs.json').write_text(json.dumps(altered))
                            with self.assertRaisesRegex(ValueError, 'bound file changed'):
                                campaign.recovery.validate_model_startup(
                                    root / 'startup-accounted/result.json', bridge)
                        with self.assertRaisesRegex(ValueError, 'namespace already exists'):
                            campaign.startup(root / 'review.json')


if __name__ == '__main__':
    unittest.main()
