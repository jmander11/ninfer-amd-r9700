import argparse
import copy
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from tools.bench import context_ladder as ladder


class ContextLadderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.contract = dict(kind='speed', contexts=[8192], chunk=2048,
                             weights=dict(path='/model.ninfer', bytes=123),
                             ids=dict(path='/corpus.ids', tokens=4096, sha256='corpus'),
                             max_context=131200, spec='dflash', draft_tokens=5,
                             lm_head_draft=True, warmup=0, repetitions=1, corpus_cycled=True)
        self.precision = dict(q4_activation_bits=8, q4_activation_profile='mixed',
                              q4_prefill_gate_up_a4=True, q4_prefill_cta_profile='production',
                              w8_activation_bits=8, kv_value_group=16, kv_plane_layouts={'key': 'fp8'},
                              fp8_qk_wmma_enabled=True, fp8_qk_wmma_profile='production',
                              fp8_qk_wmma_t1_min_context=64, fp8_qk_wmma_t2_min_context=320)
        config = dict(self.precision, xattention_qualification=False, concurrency=1,
                      prefill_chunk=2048, max_context=131200, spec='dflash', draft_tokens=5,
                      warmup=0, repetitions=1, corpus_path='/corpus.ids', corpus_tokens=4096,
                      proposal_head='optimized', kv_cache_format='fp8-k-int4-v',
                      use_device_graph=True, isolate_prompt_decode=False)
        self.raw = dict(config=config, artifact=dict(path='/model.ninfer', file_size_bytes=123),
                        load=dict(weights_id='selected'),
                        environment=dict(gpu_name='AMD Radeon AI PRO R9700', architecture_name='gfx1201',
                                         device_id=0, hip_runtime_version='10'),
                        tests=[dict(kind='pp', n_prompt=8192, n_gen=0,
                                    prefill_active_tok_s_mean=1000, prefill_active_tok_s_stddev=0,
                                    prefill_tok_s_mean=999, prefill_seconds_mean=8.192,
                                    reps=[dict(prefill_tail_by_lane=[dict(tok_s=500, window_s=1)])])])

    def retain(self, name, raw):
        report = self.root / (name + '.json')
        ladder.write(report, raw)
        attention = ladder.profile(raw['config'])
        row, precision, weights_id = ladder.speed_cell(raw, 8192, self.contract, attention)
        result = dict(schema=1, contract=self.contract, attention=attention, precision=precision,
                      weights_id=weights_id, environment=raw['environment'],
                      cells=[dict(row, report=str(report))])
        path = self.root / (name + '-ladder.json')
        ladder.write(path, result)
        return path

    def test_speed_comparison_and_changed_precision(self):
        dense = self.retain('dense', self.raw)
        other = copy.deepcopy(self.raw)
        other['config'].update(xattention_qualification=True, xattention_profile='b128-s16-tau900')
        other['tests'][0]['prefill_active_tok_s_mean'] = 1500
        other['tests'][0]['reps'][0]['prefill_tail_by_lane'][0]['tok_s'] = 1000
        candidate = self.retain('candidate', other)
        result = ladder.compare(dense, candidate)['cells'][0]
        self.assertEqual(result['average_speedup'], 1.5)
        self.assertEqual(result['tail_speedup'], 2)
        changed = ladder.read(candidate)
        changed['precision']['q4_activation_profile'] = 'all-a8'
        ladder.write(candidate, changed)
        with self.assertRaisesRegex(ValueError, 'precision'):
            ladder.compare(dense, candidate)

    def test_missing_tail_and_wrong_route_rejected(self):
        with self.assertRaisesRegex(ValueError, 'attention profile'):
            ladder.speed_cell(self.raw, 8192, self.contract, 'b128-s16-tau900')
        self.raw['tests'][0]['reps'][0]['prefill_tail_by_lane'] = []
        with self.assertRaisesRegex(ValueError, 'one lane'):
            ladder.speed_cell(self.raw, 8192, self.contract, 'dense')

    def test_ppl_sidecar_alignment_and_nonfinite(self):
        contract = dict(kind='ppl', weights=self.contract['weights'], chunk=2048, skip='half')
        raw = dict(self.precision, xattention_qualification=False, weights='/model.ninfer',
                   weights_id='selected', schedule='prefill', spec='none', draft_tokens=0,
                   device_graph=True, prefill_chunk=2048, prompt_tokens=8, skip_tokens=4,
                   tokens_scored=3, argmax_tokens=3, non_finite=0, kv_format='fp8-k-int4-v',
                   mean_nll=2.0, ppl=ladder.math.exp(2), terrible_nll=10)
        path = self.root / 'p8.json'
        path.with_suffix('.nllf32').write_bytes(struct.pack('<3f', 1, 2, 3))
        path.with_suffix('.argmaxi32').write_bytes(struct.pack('<3i', 1, 2, 3))
        result, _, _ = ladder.ppl_cell(raw, path, 8, contract, 'dense')
        self.assertEqual(result['tokens_scored'], 3)
        path.with_suffix('.nllf32').write_bytes(struct.pack('<3f', 1, float('nan'), 3))
        with self.assertRaisesRegex(ValueError, 'finite and aligned'):
            ladder.ppl_cell(raw, path, 8, contract, 'dense')
        path.with_suffix('.nllf32').write_bytes(struct.pack('<2f', 1, 2))
        with self.assertRaisesRegex(ValueError, 'finite and aligned'):
            ladder.ppl_cell(raw, path, 8, contract, 'dense')

    def test_short_ppl_corpus_cannot_be_cycled(self):
        ids = self.root / 'short.ids'
        ids.write_text('1 2 3 4')
        binary = self.root / 'scorer'
        binary.write_text('not executed')
        weights = self.root / 'weights'
        weights.write_text('not read')
        args = argparse.Namespace(binary=binary, weights=weights, ids=ids, out=self.root/'out',
                                  kind='ppl', contexts=[8], chunk=2048, repetitions=1, warmup=0,
                                  skip='half', spec='none', draft_tokens=0, lm_head_draft=False)
        with patch.object(ladder.subprocess, 'run') as execute:
            with self.assertRaisesRegex(ValueError, 'PPL never cycles'):
                ladder.run(args)
            execute.assert_not_called()

    def test_changed_retained_raw_report_rejected(self):
        dense = self.retain('dense', self.raw)
        other = copy.deepcopy(self.raw)
        other['config'].update(xattention_qualification=True, xattention_profile='b128-s16-tau900')
        candidate = self.retain('candidate', other)
        other['tests'][0]['prefill_active_tok_s_mean'] = 42
        ladder.write(self.root / 'candidate.json', other)
        with self.assertRaisesRegex(ValueError, 'retained report differs'):
            ladder.compare(dense, candidate)

    def test_resume_skips_complete_cell_but_rejects_failed_exit(self):
        ids = self.root / 'corpus.ids'
        ids.write_text('1 2 3 4')
        binary = self.root / 'bench'
        binary.write_text('mock binary')
        weights = self.root / 'model.ninfer'
        weights.write_bytes(b'x' * 123)
        power = self.root / 'power'
        power.write_text('auto')
        args = argparse.Namespace(binary=binary, weights=weights, ids=ids, out=self.root/'out',
                                  kind='speed', contexts=[8192], chunk=2048, max_context=131200,
                                  repetitions=1, warmup=0, spec='dflash', draft_tokens=5,
                                  lm_head_draft=True, allow_cyclic_speed_corpus=True,
                                  collect_only=False, attention='dense')
        raw = copy.deepcopy(self.raw)
        raw['artifact']['path'] = str(weights)
        raw['config'].update(corpus_path=str(ids), corpus_tokens=4)

        def execute(command, **kwargs):
            ladder.write(Path(command[-1]), raw)
            return argparse.Namespace(returncode=0)

        with patch.object(ladder, 'POWER', power), patch.object(ladder.subprocess, 'run', side_effect=execute) as proc:
            with patch('builtins.print'):
                ladder.run(args)
                ladder.run(args)
            self.assertEqual(proc.call_count, 1)
            self.assertEqual(len(ladder.read(args.out/'ladder.json')['cells']), 1)
            ladder.write(args.out/'p8192-exit.json', dict(returncode=1, power_after='auto'))
            with self.assertRaisesRegex(ValueError, 'prior cell did not complete'):
                ladder.run(args)
            self.assertEqual(proc.call_count, 1)

    def test_matched_ppl_comparison(self):
        contract = dict(kind='ppl', contexts=[8], weights=self.contract['weights'],
                        ids=self.contract['ids'], chunk=2048, skip='half')
        paths = []
        for name, values in (('dense', [1, 2, 3]), ('sparse', [1, 11, 3])):
            mean = sum(values)/3
            raw = dict(self.precision, xattention_qualification=name == 'sparse',
                       weights='/model.ninfer', weights_id='selected', schedule='prefill', spec='none',
                       draft_tokens=0, device_graph=True, prefill_chunk=2048, prompt_tokens=8,
                       skip_tokens=4, tokens_scored=3, argmax_tokens=3, non_finite=0,
                       kv_format='fp8-k-int4-v', mean_nll=mean, ppl=ladder.math.exp(mean), terrible_nll=10)
            if name == 'sparse':
                raw['xattention_profile'] = 'b128-s16-tau900'
            report = self.root / (name + '.json')
            ladder.write(report, raw)
            report.with_suffix('.nllf32').write_bytes(struct.pack('<3f', *values))
            report.with_suffix('.argmaxi32').write_bytes(struct.pack('<3i', 1, 2, 3))
            row, precision, weights_id = ladder.ppl_cell(raw, report, 8, contract, ladder.profile(raw))
            path = self.root / (name + '-ladder.json')
            ladder.write(path, dict(schema=1, contract=contract, attention=ladder.profile(raw),
                                   precision=precision, weights_id=weights_id, environment=None,
                                   cells=[dict(row, report=str(report))]))
            paths.append(path)
        row = ladder.compare(*paths)['cells'][0]
        self.assertEqual(row['mean_nll_delta'], 3)
        self.assertEqual(row['max_abs_nll_delta'], 9)
        self.assertEqual(row['new_severe_positions'], 1)
        changed = ladder.read(paths[1])
        changed['contract']['ids']['sha256'] = 'different corpus'
        ladder.write(paths[1], changed)
        with self.assertRaisesRegex(ValueError, 'contract'):
            ladder.compare(*paths)


if __name__ == '__main__':
    unittest.main()
