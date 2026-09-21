import unittest
from pathlib import Path
from tools.bench import xattention_keep_distribution as keep


def fixture(prompt=8192, group=16):
    rows = []
    for context in range(2048, prompt + 1, 2048):
        for _ in range(16):
            rows.append(dict(schema_version=1, diagnostic_only=True, timing_eligible=False,
                             dispatch=len(rows), context=context, query_rows=2048,
                             query_heads=24, kv_value_group=group, query_block_rows=128,
                             page_tokens=64, keep_counts=[2] * 384))
    return rows


class KeepDistributionTest(unittest.TestCase):
    def test_complete_real_geometry_and_distribution(self):
        for prompt in (8192, 32768):
            for group in (16, 32):
                result = keep.validate_rows(fixture(prompt, group), prompt, group)
                self.assertEqual(result['dispatches'], prompt // 2048 * 16)
                self.assertEqual(result['slots'], result['dispatches'] * 384)
                self.assertGreater(result['sparse_slots'], 0)
                self.assertFalse(result['timing_eligible'])

    def test_missing_layer_or_chunk_rejected(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            keep.validate_rows(fixture()[:-1], 8192, 16)

    def test_invalid_counts_and_false_dense_coverage_rejected(self):
        for count in (-1, 0, 3, True):
            rows = fixture()
            rows[0]['keep_counts'][0] = count
            with self.assertRaisesRegex(ValueError, 'count'):
                keep.validate_rows(rows, 8192, 16)
        rows = fixture()
        for row in rows:
            row['keep_counts'] = [(row['context'] - 2048 + (i % 16 + 1) * 128) // 64
                                  for i in range(384)]
        with self.assertRaisesRegex(ValueError, 'nontrivial'):
            keep.validate_rows(rows, 8192, 16)

    def test_wrong_group_order_or_timing_identity_rejected(self):
        for key, value in [('kv_value_group', 32), ('context', 4096),
                           ('dispatch', 1), ('timing_eligible', True)]:
            rows = fixture()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                keep.validate_rows(rows, 8192, 16)

    def test_command_uses_single_eager_diagnostic_prefill(self):
        cmd = keep.command(dict(executable='/bench', artifact='/weights', prompt=32768),
                           Path('/fresh'))
        for flag, value in [('--concurrency', '1'), ('--prefill-chunk', '2048'),
                            ('-r', '1'), ('--warmup', '0'), ('--draft-tokens', '0')]:
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn('--no-device-graph', cmd)
        self.assertEqual(cmd[cmd.index('--output-file') + 1], '/fresh/benchmark-diagnostic.json')


if __name__ == '__main__':
    unittest.main()
