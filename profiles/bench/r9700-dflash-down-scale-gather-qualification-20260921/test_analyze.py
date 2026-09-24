import copy
import unittest
import analyze


def report(saving):
    return {'schema': 'ninfer.r9700.dflash-down-scale-gather.v1', 'status': 'qualified',
            'pci': '0000:13:00.0', 'power': 'auto', 'copies': 3, 'scrub_bytes': 83886080,
            'complete_boundary_graph': True, 'cells': [
                {'tokens': t, 'rows': 5120, 'columns': 17408,
                 'correctness': {'criterion': 'fp64_rel_l2_1e-2_gross_1e-2_refmax_plus_1e-5',
                                 'maximum_relative_l2': 0.002, 'maximum_absolute': 0.01,
                                 'maximum_gross_cap_fraction': 0.2,
                                 'exact_codec': True, 'exact_incumbent': True,
                                 'graph_poison_stale_finite': True, 'guards_immutability': True,
                                 'finite_cases': 5, 'malformed_cases': 19},
                 'samples': [{'allocation': i % 3, 'control_first': (i//3) % 2 == 0,
                              'control_ms': 0.21, 'candidate_ms': 0.21-saving}
                             for i in range(24)]} for t in (5, 6)]}


class GateTest(unittest.TestCase):
    def test_material_winner(self):
        r = analyze.analyze(report(0.03))
        self.assertEqual(r['disposition'], 'admitted')
        self.assertFalse(r['production_routing_authorized'])

    def test_valid_losers_completed(self):
        for saving in (-0.01, 0.005):
            r = analyze.analyze(report(saving))
            self.assertEqual((r['status'], r['disposition']), ('completed', 'rejected'))

    def test_every_width_required(self):
        r = report(0.03)
        for sample in r['cells'][1]['samples']:
            sample['candidate_ms'] = 0.22
        self.assertEqual(analyze.analyze(r)['disposition'], 'rejected')

    def test_malformed_evidence(self):
        for field, value in (('exact_incumbent', False), ('maximum_relative_l2', float('nan'))):
            r = report(0.03)
            r['cells'][0]['correctness'][field] = value
            with self.assertRaises(ValueError):
                analyze.analyze(r)
        for value in (0, True, float('nan')):
            r = report(0.03)
            r['cells'][0]['samples'][0]['candidate_ms'] = value
            with self.assertRaises(ValueError):
                analyze.analyze(r)

    def test_one_allocation_regression(self):
        r = report(0.05)
        for c in r['cells']:
            for sample in c['samples'][::3]:
                sample['candidate_ms'] = 0.211
        self.assertEqual(analyze.analyze(r)['disposition'], 'rejected')


if __name__ == '__main__':
    unittest.main()
