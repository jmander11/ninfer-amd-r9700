"""CPU orchestration checks; no real artifact loads, builds or device execution."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('phase_sum_campaign', Path(__file__).with_name('campaign.py'))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
with patch.dict(sys.modules, {'campaign': campaign}):
    spec = importlib.util.spec_from_file_location('phase_sum_controls', Path(__file__).with_name('controls.py'))
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)


class CampaignTest(unittest.TestCase):
    def state(self):
        return {'authority': {'selected_prefill_chunk': 2048, 'sha256': 'a' * 64},
                'capacity': {'whole_eligible_identities': [campaign.identity(c) for c in campaign.candidates()]},
                'bindings': {}, 'quality': {'authorities': {
                    name: {'path': f'/quality/{name}.json', 'eligibility_by_group': {
                        str(g): name != 'MIXED_XATTENTION_QUALITY' for g in (16, 32)}}
                    for name in campaign.old.EXPECTED_AUTHORITIES}}}

    def test_twelve_identities_ten_eligible_and_all_capacity_paths_unchanged(self):
        cases = list(campaign.candidates())
        self.assertEqual(len(cases), 12)
        self.assertEqual(len(campaign.whole_eligible(self.state())), 10)
        self.assertEqual(len({c['build'] for c in cases}), 4)
        for case in cases:
            self.assertIn('build-r9700-phase-sum', str(case['build']))
            self.assertEqual(campaign.matrix(case, 'capacity'), campaign.old.matrix(case, 'capacity'))
            self.assertEqual(campaign.matrix(case, 'whole').parent, campaign.PACKAGE / 'whole')

    def test_whole_commands_use_all_four_and_matched_new_planner(self):
        for case in campaign.candidates():
            cmd = campaign.command(case, self.state())
            self.assertEqual([cmd[i + 1] for i, x in enumerate(cmd) if x == '--concurrency'], ['1', '2', '3', '4'])
            self.assertEqual(cmd[cmd.index('--bench') + 1], str(case['build'] / 'bench/ninfer_bench'))
            self.assertEqual(cmd[cmd.index('--prefill-chunk') + 1], '2048')
            self.assertIn('--no-build', cmd)
            self.assertNotIn('--resume', cmd)
            self.assertEqual('--require-fp8-hybrid' in cmd, case['weights_id'] == campaign.recovery.HYBRID)
            if '--hybrid-width-tool' in cmd:
                self.assertEqual(cmd[cmd.index('--hybrid-width-tool') + 1], str(case['build'] / 'src/ninfer_r9700_runtime_planner_qual'))

    def test_whole_runs_ten_and_failed_namespace_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(campaign, 'PACKAGE', Path(directory)):
            calls = []
            with patch.object(campaign, 'execute', side_effect=lambda cmd, root, name: calls.append(name) or 0), \
                 patch.object(campaign, 'validate_whole') as validate:
                campaign.whole(self.state())
                self.assertEqual(len(calls), 10)
                self.assertEqual(validate.call_count, 10)
                with self.assertRaises(Exception):
                    campaign.whole(self.state())
                self.assertEqual(len(calls), 10)
        with tempfile.TemporaryDirectory() as directory, patch.object(campaign, 'PACKAGE', Path(directory)):
            with patch.object(campaign, 'execute', return_value=1) as execute:
                with self.assertRaisesRegex(ValueError, 'preserve namespace'):
                    campaign.whole(self.state())
                with self.assertRaises(Exception):
                    campaign.whole(self.state())
                self.assertEqual(execute.call_count, 1)

    def test_whole_rejects_legacy_or_wrong_reporter(self):
        state = self.state()
        case = next(campaign.candidates())
        expected = {'path': '/new-bench', 'sha256': 'b' * 64}
        report = {'schema_version': 21,
                  'phase_timing_semantics': 'serial-lane-service-sum_shared-decode-max_v1'}
        from tools.ppl import assemble_pareto
        with patch.object(campaign, 'expected_benchmark', return_value=expected), \
             patch.object(assemble_pareto, '_manifest', return_value={'bench': expected}), \
             patch.object(assemble_pareto, '_reports', return_value={c: [report] for c in (1, 2, 3, 4)}):
            campaign.validate_whole(case, state)
            report['schema_version'] = 20
            with self.assertRaisesRegex(ValueError, 'corrected schema21'):
                campaign.validate_whole(case, state)

    def test_select_retains_all_twelve_and_only_measured_whole_paths(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(campaign, 'PACKAGE', Path(directory)):
            state = self.state()
            (Path(directory) / 'whole').mkdir()
            (Path(directory) / 'whole/inputs.json').write_text('{}')
            captured = []
            def execute(args, root, name):
                captured.extend(args)
                Path(args[args.index('--out') + 1]).write_text('{}')
                return 0
            with patch.dict(sys.modules, {'controls': controls}), \
                 patch.object(controls, 'validate_completed') as gate, \
                 patch.object(campaign, 'execute', side_effect=execute), \
                 patch.object(campaign.pareto, 'classify', return_value={}), \
                 patch.object(campaign, 'publish') as publish:
                campaign.select(state)
                gate.assert_called_once()
                publish.assert_called_once()
            self.assertEqual(captured.count('--candidate'), 12)
            self.assertEqual(captured.count('-'), 2)
            self.assertIn('--benchmark-reporting-recovery', captured)
            self.assertIn(str(campaign.CAPACITY_VALIDATION), captured)

    def test_controls_compare_all_graph_repetitions_and_exact_lane_tokens(self):
        def report(reps):
            return {'tests': [{'label': '8k', 'reps': [
                {'generated_token_ids_by_lane': [[11, 12], [21, 22]]} for _ in range(reps)]}]}
        graph, eager = report(3), report(1)
        self.assertTrue(controls.exact_tokens(graph, eager)[0]['exact'])
        graph['tests'][0]['reps'][1]['generated_token_ids_by_lane'][1][0] = 99
        self.assertFalse(controls.exact_tokens(graph, eager)[0]['exact'])
        cmd = ['bench', '--retain-token-ids', '-r', '3', '--warmup', '1', '--output-file', 'graph']
        transformed = controls.eager_command(cmd, Path('eager'))
        self.assertEqual(transformed[transformed.index('-r') + 1], '1')
        self.assertEqual(transformed[transformed.index('--warmup') + 1], '0')
        self.assertIn('--no-device-graph', transformed)
        self.assertEqual(cmd[-1], 'graph')

    def test_host_checks_publish_exact_bound_receipts_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(campaign, 'PACKAGE', Path(directory)):
            cases = []
            for case in campaign.candidates():
                if case['weights_id'] != campaign.recovery.HYBRID:
                    continue
                build = Path(directory) / case['build'].name
                tool = build / 'src/ninfer_r9700_runtime_planner_qual'
                tool.parent.mkdir(parents=True)
                tool.write_text('host tool fixture')
                cases.append({**case, 'build': build})
            def execute(cmd, root, name):
                self.assertEqual(cmd[1:], ['--host-split512-routing'])
                for suffix in ('stdout', 'stderr'):
                    (root / f'{name}.{suffix}.txt').write_text('PASS' if suffix == 'stdout' else '')
                return 0
            with patch.object(campaign, 'candidates', return_value=cases), \
                 patch.object(campaign, 'execute', side_effect=execute) as executed:
                campaign.host_checks()
                self.assertEqual(executed.call_count, 4)
                with self.assertRaises(Exception):
                    campaign.host_checks()
            for path in (Path(directory) / 'host-checks').glob('*.json'):
                value = json.loads(path.read_text())
                self.assertEqual(value['exit_code'], 0)
                campaign.recovery.checked_file(value['stdout'])
                campaign.recovery.checked_file(value['stderr'])


if __name__ == '__main__':
    unittest.main()
