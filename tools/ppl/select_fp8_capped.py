"""Bounded fixed-recipe search against the immutable local 5090 NVFP4 reference."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics

from tools.ppl import compare_nvfp4 as common

NAMES = ('early-attention','all-attention','attention-gdn','selective-cap')
REFERENCE = common.ROOT/'tools/ppl/fixtures/nvfp4-5090-20260922'


def configure(out):
    common.ARTIFACTS = {name:out/'artifacts'/f'{name}.ninfer' for name in NAMES}
    common.CONFIGS = [(name,8) for name in NAMES]


def prepare(out):
    common.require_uniform_build(common.ROOT/'build-r9700', 8)
    reference=json.loads((REFERENCE/'reference.json').read_text())
    (out/'bin').mkdir()
    for kind,rel in [('ppl','apps/ninfer-ppl'),('bench','bench/ninfer_bench')]:
        shutil.copy2(common.ROOT/'build-r9700'/rel,out/'bin'/f'amd-a8-{kind}')
    for name,row in reference['samples'].items():
        shutil.copyfile(REFERENCE/row['ids'],out/f'{name}.ids')
        for schedule,prefix in [('prefill','quality'),('decode','decode-quality')]:
            cell=out/f'{prefix}-nvfp4-{name}';cell.mkdir()
            common.write_json(cell/'report.json',row[schedule]['report'])
            shutil.copyfile(REFERENCE/row[schedule]['nll_file'],cell/'report.nllf32')
    common.write_json(out/'manifest.json',dict(reference=str(REFERENCE),
        reference_sha256=common.digest(REFERENCE/'reference.json'),
        binaries={p.name:common.digest(p) for p in (out/'bin').iterdir()},
        quality='per-text PPL ratio <=1.02 preferred; <=1.05 secondary screen, report severe positions',
        recipes={k:str(v) for k,v in common.ARTIFACTS.items()}))


def screen(out):
    result={}
    for name in NAMES:
        row={}
        for schedule,prefix in [('prefill','quality'),('decode','decode-quality')]:
            row[schedule]={}
            for sample in common.SAMPLES:
                candidate,cv=common.nlls(out/f'{prefix}-{name}_a8-{sample}',schedule)
                reference,rv=common.nlls(out/f'{prefix}-nvfp4-{sample}',schedule)
                receipt=json.loads((out/f'{prefix}-{name}_a8-{sample}'/'receipt.json').read_text())
                assert receipt['exit_code']==0 and candidate['q4_activation_bits']==8
                delta=statistics.mean(c-r for c,r in zip(cv,rv))
                row[schedule][sample]=dict(ppl=candidate['ppl'],reference_ppl=reference['ppl'],
                    ratio=math.exp(delta),mean_nll_delta=delta,
                    new_severe=sum(c>=10 and r<10 for c,r in zip(cv,rv)),
                    resolved_severe=sum(c<10 and r>=10 for c,r in zip(cv,rv)))
        ratios=[v['ratio'] for samples in row.values() for v in samples.values()]
        row['within_2pct']=max(ratios)<=1.02
        row['within_5pct']=max(ratios)<=1.05
        row['file_gb']=common.ARTIFACTS[name].stat().st_size/1e9
        result[name]=row
    common.write_json(out/'quality-summary.json',result)
    print(json.dumps(result,indent=2))


def mixed(out, action):
    name='selective-cap-mixed'
    common.ARTIFACTS={name:out/'artifacts/selective-cap.ninfer'}
    common.CONFIGS=[(name,8)]
    if action=='snapshot-mixed':
        build=common.ROOT/'build-r9700-fp8-capped-mixed-20260923'
        records={}
        for kind,rel in [('ppl','apps/ninfer-ppl'),('bench','bench/ninfer_bench')]:
            target=out/'bin'/f'mixed-{kind}'
            if target.exists():raise FileExistsError(target)
            shutil.copy2(build/rel,target);records[target.name]=common.digest(target)
        common.write_json(out/'mixed-binaries.json',records)
        return
    if action in ('mixed-prefill','mixed-decode'):
        schedule=action.split('-')[1]
        common.quality(out,'amd',schedule,binary_override=out/'bin/mixed-ppl')
        prefix='quality' if schedule=='prefill' else 'decode-quality'
        for sample in common.SAMPLES:
            report,_=common.nlls(out/f'{prefix}-{name}_a8-{sample}',schedule)
            assert report['q4_prefill_gate_up_a4'] is True
            assert report['q4_activation_profile']=='a8-except-n34816-k5120-tgt128-a4'
        return
    if action=='mixed-speed':
        summary=json.loads((out/'mixed-quality-summary.json').read_text())
        if not summary['within_5pct']:
            raise ValueError('mixed profile not eligible for timing')
        common.speed(out,binary_override=out/'bin/mixed-bench')
        return
    result={}
    for schedule,prefix in [('prefill','quality'),('decode','decode-quality')]:
        cells=[out/f'{prefix}-{name}_a8-{sample}' for sample in common.SAMPLES]
        if schedule=='decode' and not any(p.exists() for p in cells):continue
        result[schedule]={}
        for sample,cell in zip(common.SAMPLES,cells):
            report,cv=common.nlls(cell,schedule)
            _,rv=common.nlls(out/f'{prefix}-nvfp4-{sample}',schedule)
            assert report['q4_prefill_gate_up_a4'] is True
            assert report['q4_activation_profile']=='a8-except-n34816-k5120-tgt128-a4'
            result[schedule][sample]=dict(ppl=report['ppl'],
                ratio=math.exp(statistics.mean(c-r for c,r in zip(cv,rv))),
                new_severe=sum(c>=10 and r<10 for c,r in zip(cv,rv)))
    ratios=[v['ratio'] for samples in result.values() for v in samples.values()]
    result['within_2pct']='decode' in result and max(ratios)<=1.02
    result['within_5pct']='decode' in result and max(ratios)<=1.05
    common.write_json(out/'mixed-quality-summary.json',result)
    print(json.dumps(result,indent=2))


def finish(out):
    quality=json.loads((out/'quality-summary.json').read_text())
    assert [n for n,r in quality.items() if r['within_2pct']]==['selective-cap']
    comparison=json.loads((out/'comparison.json').read_text())
    base=comparison['rows']['selective-cap_a8']
    mixed_quality=json.loads((out/'mixed-quality-summary.json').read_text())
    assert mixed_quality['within_2pct']
    mixed_speed={}
    for sample in common.SAMPLES:
        cell=out/f'speed-selective-cap-mixed_a8-{sample}'
        receipt=json.loads((cell/'receipt.json').read_text());assert receipt['exit_code']==0
        report=json.loads((cell/'report.json').read_text())
        assert report['config']['q4_prefill_gate_up_a4'] is True
        assert report['config']['q4_activation_profile']=='a8-except-n34816-k5120-tgt128-a4'
        test,=report['tests'];reps=test['reps']
        assert test['n_prompt']==4096 and test['n_gen']==128 and len(reps)==3
        assert all(not r['speculative']['enabled'] and r['decode_output_tokens']==128 for r in reps)
        assert all(r['generated_token_ids_by_lane']==reps[0]['generated_token_ids_by_lane'] for r in reps)
        pp=[4096/r['timings']['prefill_seconds'] for r in reps]
        tg=[128/r['timings']['decode_seconds'] for r in reps]
        mixed_speed[sample]=dict(prefill_tok_s=statistics.median(pp),decode_tok_s=statistics.median(tg),
                                prefill_range=[min(pp),max(pp)],decode_range=[min(tg),max(tg)])
        assert mixed_speed[sample]['prefill_tok_s']<base['samples'][sample]['prefill_tok_s']
    model=common.LOCAL/'models/qwen3.8-27b-r9700-q4-fp8-selective-cap/qwen3.8-27b-r9700-q4-fp8-selective-cap-n16k16-eval.ninfer'
    assert model.stat().st_size==common.ARTIFACTS['selective-cap'].stat().st_size
    result=dict(status='selected-for-further-optimization-not-final-production',
        artifact=str(model),file_bytes=model.stat().st_size,selected_execution='uniform-a8',
        reason='Only new capped recipe within2% per-text PPL in both schedules; mixed A4 is slower and adds severe technical-prefill positions.',
        reference=str(REFERENCE),base=base,mixed_quality=mixed_quality,mixed_speed=mixed_speed,
        limitations=['Three local texts; no universal quality equivalence or global optimality.',
                     'No DFlash companion or speculative throughput qualification for this artifact.',
                     'All-Q4/A8 still leads measured decode; four-role still leads measured prefill.'])
    common.write_json(out/'selection.json',result)
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','quality','decode-quality','screen','speed','analyze',
        'snapshot-mixed','mixed-prefill','mixed-decode','mixed-screen','mixed-speed','finish'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--only',choices=NAMES)
    args=p.parse_args();out=args.out.resolve();configure(out)
    only=args.only+'_a8' if args.only else None
    if 'mixed' in args.action:mixed(out,args.action)
    elif args.action=='finish':finish(out)
    elif args.action=='prepare':prepare(out)
    elif args.action in ('quality','decode-quality'):
        common.quality(out,'amd','prefill' if args.action=='quality' else 'decode',only=only)
    elif args.action=='screen':screen(out)
    elif args.action=='speed':
        summary=json.loads((out/'quality-summary.json').read_text())
        for name in NAMES:
            if args.only and name!=args.only:continue
            if summary[name]['within_5pct']:common.speed(out,name+'_a8')
            else:print(f'{name}: no timing, exceeds 5% per-text PPL screen')
    else:
        summary=json.loads((out/'quality-summary.json').read_text())
        common.CONFIGS=[(name,8) for name in NAMES if summary[name]['within_5pct']]
        common.analyze(out)


if __name__=='__main__':main()
