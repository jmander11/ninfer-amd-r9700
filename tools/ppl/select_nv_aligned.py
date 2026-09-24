"""Bounded NVIDIA-aligned protection / AMD activation comparison.

Uses the immutable three-text NVFP4 reference. It neither changes the default
artifact nor selects a global optimum. Each output cell is create-only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics

from tools.ppl import compare_nvfp4 as common

REFERENCE = common.ROOT / 'tools/ppl/fixtures/nvfp4-5090-20260922'
BUILD = common.ROOT / 'build-r9700-precision-search-20260923'
SELECTIVE = common.LOCAL / 'models/qwen3.8-27b-r9700-q4-fp8-selective-cap/qwen3.8-27b-r9700-q4-fp8-selective-cap-n16k16-eval.ninfer'
PROFILES = {
    'a8': 'uniform-a8',
    'gate-up': 'a8-except-n34816-k5120-tgt128-a4',
    'mlp': 'a8-except-mlp-tgt128-a4',
    'projections': 'a8-except-text-projections-tgt128-a4',
    'attn-input': 'a8-except-n7168-k5120-tgt128-a4',
    'gate-up-attn-input': 'a8-except-gate-up-attn-input-tgt128-a4',
}


def prepare(out):
    out.mkdir()
    (out/'bin').mkdir()
    reference = json.loads((REFERENCE/'reference.json').read_text())
    for name, row in reference['samples'].items():
        shutil.copyfile(REFERENCE/row['ids'], out/f'{name}.ids')
        for schedule, prefix in [('prefill', 'quality'), ('decode', 'decode-quality')]:
            cell = out/f'{prefix}-nvfp4-{name}'
            cell.mkdir()
            common.write_json(cell/'report.json', row[schedule]['report'])
            shutil.copyfile(REFERENCE/row[schedule]['nll_file'], cell/'report.nllf32')
    common.write_json(out/'manifest.json', dict(reference=str(REFERENCE),
        reference_sha256=common.digest(REFERENCE/'reference.json'),
        criterion='Each text/schedule <=2% PPL regression; report severe-position deltas separately',
        selective=str(SELECTIVE), profiles=PROFILES))


def summarize(out, name):
    result = dict(samples={})
    for schedule, prefix in [('prefill', 'quality'), ('decode', 'decode-quality')]:
        result['samples'][schedule] = {}
        for sample in common.SAMPLES:
            report, values = common.nlls(out/f'{prefix}-{name}_a8-{sample}', schedule)
            ref, baseline = common.nlls(out/f'{prefix}-nvfp4-{sample}', schedule)
            delta = statistics.mean(c-r for c, r in zip(values, baseline, strict=True))
            result['samples'][schedule][sample] = dict(ppl=report['ppl'],
                reference_ppl=ref['ppl'], ratio=math.exp(delta), mean_nll_delta=delta,
                severe=sum(c >= 10 for c in values),
                reference_severe=sum(r >= 10 for r in baseline),
                new_severe=sum(c >= 10 and r < 10 for c, r in zip(values, baseline)),
                resolved_severe=sum(c < 10 and r >= 10 for c, r in zip(values, baseline)))
    result['within_2pct'] = max(r['ratio'] for s in result['samples'].values() for r in s.values()) <= 1.02
    common.write_json(out/f'{name}-quality.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'snapshot', 'quality', 'summary', 'speed'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', choices=PROFILES, default='projections')
    parser.add_argument('--recipe', choices=['selective', 'nv-default', 'outputs-only', 'no-late-mlp'], default='selective')
    parser.add_argument('--binary-tag', help='Explicit create-only snapshot name; defaults to profile')
    args = parser.parse_args()
    out = args.out.resolve()
    name = f'{args.recipe}-{args.profile}'
    binary_tag = args.binary_tag or args.profile
    if args.action == 'prepare':
        prepare(out)
        return
    artifact = SELECTIVE if args.recipe == 'selective' else out/f'artifacts/{args.recipe}.ninfer'
    common.ARTIFACTS = {name: artifact}
    common.CONFIGS = [(name, 8)]
    if args.action == 'snapshot':
        for kind, rel in [('ppl', 'apps/ninfer-ppl'), ('bench', 'bench/ninfer_bench')]:
            dest = out/'bin'/f'{binary_tag}-{kind}'
            if dest.exists():
                raise FileExistsError(dest)
            shutil.copy2(BUILD/rel, dest)
        common.write_json(out/f'{binary_tag}-binaries.json', {
            kind: common.digest(out/'bin'/f'{binary_tag}-{kind}') for kind in ('ppl', 'bench')})
    elif args.action == 'quality':
        for schedule in ('prefill', 'decode'):
            common.quality(out, 'amd', schedule, binary_override=out/'bin'/f'{binary_tag}-ppl')
            prefix = 'quality' if schedule == 'prefill' else 'decode-quality'
            for sample in common.SAMPLES:
                report, _ = common.nlls(out/f'{prefix}-{name}_a8-{sample}', schedule)
                if report['q4_activation_profile'] != PROFILES[args.profile]:
                    raise ValueError('binary activation profile mismatch')
        summarize(out, name)
    elif args.action == 'summary':
        summarize(out, name)
    else:
        quality = json.loads((out/f'{name}-quality.json').read_text())
        if not quality['within_2pct']:
            raise ValueError('candidate outside chosen quality frontier; no timing')
        common.speed(out, binary_override=out/'bin'/f'{binary_tag}-bench')
        measured = {}
        for sample in common.SAMPLES:
            cell = out/f'speed-{name}_a8-{sample}'
            report = json.loads((cell/'report.json').read_text())
            if report['config']['q4_activation_profile'] != PROFILES[args.profile]:
                raise ValueError('benchmark activation profile mismatch')
            test, = report['tests']
            reps = test['reps']
            if test['n_prompt'] != 4096 or test['n_gen'] != 128 or len(reps) != 3:
                raise ValueError('benchmark geometry/repetitions mismatch')
            for rep in reps:
                if (rep['speculative']['enabled'] or rep['decode_output_tokens'] != 128 or
                        rep['generated_token_ids_by_lane'] != reps[0]['generated_token_ids_by_lane']):
                    raise ValueError('ordinary benchmark token-repeat check failed')
            prefill = [4096/r['timings']['prefill_seconds'] for r in reps]
            decode = [128/r['timings']['decode_seconds'] for r in reps]
            if not all(math.isfinite(v) and v > 0 for v in prefill+decode):
                raise ValueError('invalid timing')
            measured[sample] = dict(prefill_tok_s=statistics.median(prefill),
                decode_tok_s=statistics.median(decode), prefill_range=[min(prefill), max(prefill)],
                decode_range=[min(decode), max(decode)])
        common.write_json(out/f'{name}-speed.json', measured)
        print(json.dumps(measured, indent=2), flush=True)


if __name__ == '__main__':
    main()
