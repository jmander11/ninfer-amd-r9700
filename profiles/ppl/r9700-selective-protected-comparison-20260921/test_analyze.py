import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('selective_compare', Path(__file__).with_name('analyze.py'))
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


class ComparisonTest(unittest.TestCase):
    def test_raw_command_binds_corpus_weights_and_scorer(self):
        campaign = {'candidate_artifact': {'path': '/weights'},
                    'scorers': {'r9700-g16': {'path': '/scorer'}}}
        reference = {'corpus': {'path': '/corpus'}}
        command = ['/scorer', '--weights', '/weights', '--ids', '/corpus']
        analyze.validate_command(command, campaign, reference)
        for index in (0, 2, 4):
            wrong = command.copy()
            wrong[index] = '/wrong'
            with self.assertRaises(ValueError):
                analyze.validate_command(wrong, campaign, reference)

    def test_paired_delta_sign_and_no_quality_waiver(self):
        result = analyze.quality([2., 3.], [1, 2], [1., 2.], [1, 2])
        self.assertEqual(result['delta_mean_nll_vs_bf16'], 1.)
        self.assertTrue(result['bf16_greedy_diagnostic']['argmax_exact'])
        self.assertFalse(result['quality_tiers']['accuracy']['pass'])
        self.assertFalse(result['quality_tiers']['capacity-speed']['pass'])

    def test_new_severe_membership_not_aggregate_count(self):
        reference = [11.] * 12 + [1.] * 4083
        candidate = [1.] * 12 + [11.] * 12 + [1.] * 4071
        result = analyze.quality(candidate, [1] * 4095, reference, [1] * 4095)
        self.assertEqual(result['delta_mean_nll_vs_bf16'], 0.)
        self.assertEqual(result['quality_tiers']['capacity-speed']['new_severe_positions'], 12)
        self.assertFalse(result['quality_tiers']['capacity-speed']['pass'])

    def test_reject_unaligned_or_nonfinite_sidecars(self):
        for nll in ([1.], [1., float('nan')]):
            with self.assertRaises(ValueError):
                analyze.quality(nll, [1, 2], [1., 2.], [1, 2])


if __name__ == '__main__':
    unittest.main()
