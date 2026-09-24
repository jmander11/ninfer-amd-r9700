import unittest
from unittest.mock import patch

import run

OLD = run.ROOT/'profiles/bench/r9700-dflash-down-splitk-whole-ab-20260919/results'


class ReportTests(unittest.TestCase):
    def fixture(self):
        report = run.load(OLD/'k4w5-1-control.json')
        command = ['synthetic-benchmark', '--test']
        report['command'] = ' '.join(command)
        for key in ('dflash_down_splitk_candidate', 'dflash_down_splitk_factor', 'dflash_down_scale_gather_candidate'):
            report['config'].pop(key, None)
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


    def test_confirmation_preserves_current_tokens_and_accounting(self):
        evidence = run.load(run.RETAINED/'summary.json')
        c = evidence['cells']['k4w5']
        times = {key: value['candidate_seconds'] for key, value in c['gates'].items()}
        accounting = tuple(c['speculative_accounting'][:-1])+(tuple(c['speculative_accounting'][-1]),)
        result = run.compare_confirmation(evidence['ordinary_tokens'], times, accounting,
                                           c['workspace_capacity_bytes'], 4, 5)
        self.assertTrue(result['confirmed'])
        times = {k: [v*1.1 for v in values] for k, values in times.items()}
        result = run.compare_confirmation(evidence['ordinary_tokens'], times, accounting,
                                           c['workspace_capacity_bytes'], 4, 5)
        self.assertFalse(result['confirmed'])
        with self.assertRaises(RuntimeError):
            run.compare_confirmation([1]*65, times, accounting, c['workspace_capacity_bytes'], 4, 5)


if __name__ == '__main__':
    unittest.main()
