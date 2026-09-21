#!/usr/bin/env python3
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import controls


class ControlsTest(unittest.TestCase):
    def test_changed_graph_inputs_rejected(self):
        source = {'artifact': {'weights_id': 'q4', 'sha256': 'a', 'conversion_receipt': {'sha256': 'b'}},
                  'benchmark_executable': {'path': 'bench', 'sha256': 'c'}}
        state = {'authority': {'path': 'chunk', 'sha256': 'd', 'selected_prefill_chunk': 2048}}
        candidate = {'artifact': Path('model')}
        manifest = {'artifact': source['artifact'], 'bench': source['benchmark_executable'],
                    'corpus': str(controls.campaign.CORPUS), 'corpus_sha256': 'e',
                    'prefill_chunk_authority': state['authority'], 'selected_prefill_chunk': 2048,
                    'commands': [{'command': ['bench', '--weights', 'model', '--corpus',
                                             str(controls.campaign.CORPUS), '--device', '0']}]}
        controls.validate_graph_binding(manifest, candidate, state, source, 'e')
        for field in ('artifact', 'bench', 'corpus', 'corpus_sha256', 'prefill_chunk_authority',
                      'selected_prefill_chunk'):
            altered = copy.deepcopy(manifest)
            altered[field] = None
            with self.subTest(field=field), self.assertRaises(ValueError):
                controls.validate_graph_binding(altered, candidate, state, source, 'e')
        altered = copy.deepcopy(manifest)
        altered['commands'][0]['command'][2] = 'stale-model'
        with self.assertRaises(ValueError):
            controls.validate_graph_binding(altered, candidate, state, source, 'e')

    def test_completed_control_rejects_missing_or_forged_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'inputs.json').write_text('{}')
            (output / 'result.json').write_text('{"artifact_type": "ninfer_r9700_terminal_graph_eager_controls",'
                                              '"schema_version":1,"pass":true,"timing_eligible":false,'
                                              '"inputs":{},"comparisons":[{"forged":true}]}')
            with patch.object(controls, 'OUTPUT', output), \
                    patch.object(controls, 'preflight', return_value=({}, [], {}, None)):
                with self.assertRaisesRegex(ValueError, 'every current eligible control'):
                    controls.validate_completed()

    def test_command_preserves_model_and_workload(self):
        command = ['bench', '--weights', 'model', '--corpus', 'ids', '--concurrency', '4',
                   '--whole-pg', '8192,256;32768,256', '--prefill-chunk', '2048',
                   '--draft-tokens', '0', '--retain-token-ids', '-r', '3', '--warmup', '1',
                   '--output-file', 'graph.json']
        actual = controls.eager_command(command, Path('eager.json'))
        expected = command.copy()
        expected[expected.index('-r') + 1] = '1'
        expected[expected.index('--warmup') + 1] = '0'
        expected[-1] = 'eager.json'
        self.assertEqual(actual, [*expected, '--no-device-graph'])
        self.assertEqual(command[-1], 'graph.json')

    def test_ambiguous_or_nonretaining_command_rejected(self):
        valid = ['bench', '--retain-token-ids', '-r', '3', '--warmup', '1',
                 '--output-file', 'graph.json']
        for command in (valid + ['-r', '4'], valid + ['--no-device-graph'], valid[2:]):
            with self.subTest(command=command), self.assertRaises(ValueError):
                controls.eager_command(command, Path('eager.json'))

    def reports(self):
        rep = {'generated_token_ids_by_lane': [[1, 2, 3], [4, 5, 6]]}
        graph = {'tests': [{'label': 'a', 'reps': [copy.deepcopy(rep) for _ in range(3)]}]}
        eager = {'tests': [{'label': 'a', 'reps': [copy.deepcopy(rep)]}]}
        return graph, eager

    def test_every_graph_repetition_and_lane_compared(self):
        graph, eager = self.reports()
        self.assertTrue(controls.exact_tokens(graph, eager)[0]['exact'])
        graph['tests'][0]['reps'][2]['generated_token_ids_by_lane'][1][2] = 7
        self.assertFalse(controls.exact_tokens(graph, eager)[0]['exact'])

    def test_eager_mismatch_retained_as_failure(self):
        graph, eager = self.reports()
        eager['tests'][0]['reps'][0]['generated_token_ids_by_lane'][1][0] = 7
        self.assertFalse(controls.exact_tokens(graph, eager)[0]['exact'])

    def test_missing_or_duplicate_tests_rejected(self):
        graph, eager = self.reports()
        for value in ({'tests': []}, {'tests': eager['tests'] * 2}):
            with self.assertRaises(ValueError):
                controls.exact_tokens(graph, value)

    def test_existing_failed_namespace_never_reexecuted(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'controls'
            output.mkdir()
            failure = output / 'failure.json'
            failure.write_text('retained failure')
            with patch.object(controls, 'OUTPUT', output), \
                    patch.object(controls, 'preflight', return_value=({}, [], {}, None)), \
                    patch.object(controls.bench, 'require_auto_power_profile') as power, \
                    patch.object(controls.campaign, 'execute') as execute:
                with self.assertRaises((ValueError, FileExistsError)):
                    controls.execute()
            execute.assert_not_called()
            power.assert_not_called()
            self.assertEqual(failure.read_text(), 'retained failure')


if __name__ == '__main__':
    unittest.main()
