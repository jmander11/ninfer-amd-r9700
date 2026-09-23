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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','quality','decode-quality','screen','speed','analyze'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--only',choices=NAMES)
    args=p.parse_args();out=args.out.resolve();configure(out)
    only=args.only+'_a8' if args.only else None
    if args.action=='prepare':prepare(out)
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
