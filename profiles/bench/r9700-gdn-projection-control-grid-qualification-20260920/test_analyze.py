import copy
import unittest
import analyze


def report(ratio):
    return {'schema': 'ninfer.r9700.gdn-projection-control-grid.v1', 'status': 'qualified',
            'projection_max_relative_l2': 0.002, 'projection_max_bf16_steps': 1,
            'control_g_max_absolute_error': 0.001, 'control_beta_max_absolute_error': 0.0001,
            'codec_exact': True, 'incumbent_exact': True, 'graph_poison_stale_finite': True,
            'guards_and_immutability': True, 'malformed_cases': 16,
            'pci': '0000:13:00.0', 'power': 'auto', 'copies': 3,
            'scrub_bytes': 83886080, 'calls_per_token': 48,
            'samples': [{'allocation': i % 3, 'control_first': (i//3) % 2 == 0,
                         'control_ms': 0.12, 'candidate_ms': 0.12*ratio} for i in range(24)]}


class AdmissionTest(unittest.TestCase):
    def test_winner(self):
        result = analyze.analyze(report(0.9))
        self.assertEqual(result['disposition'], 'admitted')
        self.assertFalse(result['production_routing_authorized'])

    def test_loser_is_completed(self):
        result = analyze.analyze(report(1.1))
        self.assertEqual((result['status'], result['disposition']), ('completed', 'rejected'))

    def test_small_winner_rejected(self):
        self.assertEqual(analyze.analyze(report(0.99))['disposition'], 'rejected')

    def test_invalid_evidence(self):
        base = report(0.9)
        for field, value in (('incumbent_exact', False), ('projection_max_relative_l2', float('nan')),
                             ('control_g_max_absolute_error', 0.1)):
            bad = copy.deepcopy(base)
            bad[field] = value
            with self.assertRaises(ValueError):
                analyze.analyze(bad)
        for value in (0, float('nan'), True):
            bad = copy.deepcopy(base)
            bad['samples'][0]['candidate_ms'] = value
            with self.assertRaises(ValueError):
                analyze.analyze(bad)


if __name__ == '__main__':
    unittest.main()
