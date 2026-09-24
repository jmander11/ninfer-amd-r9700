import unittest
from unittest.mock import patch

import run

OLD = run.ROOT/'profiles/bench/r9700-dflash-down-splitk-whole-ab-20260919/results'


class ReportTests(unittest.TestCase):
    def fixture(self):
        report = run.load(OLD/'k4w5-1-control.json')
        command = ['synthetic-benchmark', '--test']
        report['command'] = ' '.join(command)
        report['config']['dflash_down_scale_gather_candidate'] = False
        plan = {'artifact': {'path': report['artifact']['path'],
                             'bytes': report['artifact']['file_size_bytes']}}
        return report, plan, command

    def validate(self, mutate=lambda report: None):
        report, plan, command = self.fixture()
        mutate(report)
        def load(path):
            if path.name == 'plan.json':
                return plan
            if path.name.endswith('.process.json'):
                return {'command': command, 'exit_code': 0}
            return report
        with patch.object(run, 'load', side_effect=load), patch.object(run, 'command', return_value=command):
            return run.validate_report('synthetic', 'control', 4, 5)

    def test_valid_retained_report_and_all_timing_scopes(self):
        tokens, times, accounting, capacity = self.validate()
        self.assertEqual(len(tokens), 65)
        self.assertEqual(set(times), {'decode_seconds', 'total_seconds', 'prefill_seconds'})
        self.assertTrue(accounting[0] > 0 and capacity > 0)

    def test_reject_wrong_route(self):
        with self.assertRaises(RuntimeError):
            self.validate(lambda r: r['config'].update(dflash_down_scale_gather_candidate=True))

    def test_reject_different_tokens_within_process(self):
        with self.assertRaises(RuntimeError):
            self.validate(lambda r: r['tests'][0]['reps'][1]['generated_token_ids_by_lane'][0].__setitem__(0, 1))

    def test_reject_wrong_accounting(self):
        with self.assertRaises(RuntimeError):
            self.validate(lambda r: r['tests'][0]['reps'][0]['speculative'].update(accepted_tokens=9999))

    def test_reject_wrong_public_accounting(self):
        with self.assertRaises(RuntimeError):
            self.validate(lambda r: r['tests'][0]['reps'][0].update(decode_output_tokens=63))

    def test_reject_workspace_overrun(self):
        with self.assertRaises(RuntimeError):
            self.validate(lambda r: r['tests'][0].update(workspace_peak_bytes=10**15))

    def test_reject_graph_overrun(self):
        with self.assertRaises(RuntimeError):
            self.validate(lambda r: r['memory'].update(device_graph_observed_bytes=10**15))

    def test_reject_nonfinite_decode(self):
        with self.assertRaises(RuntimeError):
            self.validate(lambda r: r['tests'][0]['reps'][0]['timings'].update(decode_seconds=float('nan')))

    def records(self, ratios):
        return [{'role': role, 'times': {'decode_seconds': [value]*3}}
                for role, value in zip(('control', 'candidate', 'candidate', 'control'),
                                       (1, ratios[0], ratios[1], 1))]

    def test_material_win(self):
        self.assertTrue(run.speed_gate(self.records((.94, .945)), 'decode_seconds')['admitted'])

    def test_valid_loser_is_not_exception(self):
        self.assertFalse(run.speed_gate(self.records((1.01, 1.02)), 'decode_seconds')['admitted'])

    def test_immaterial_win_rejected(self):
        self.assertFalse(run.speed_gate(self.records((.995, .995)), 'decode_seconds')['admitted'])

    def test_order_instability_rejected(self):
        self.assertFalse(run.speed_gate(self.records((.90, .96)), 'decode_seconds')['admitted'])


if __name__ == '__main__':
    unittest.main()
