"""Endpoint precision quality matrix; theoretical costs only, never speed timing.

The preserved NVFP4 inputs/NLLs are the quality reference. Cost vectors are byte
and matrix-work proxies, not measured latency or a tok/s prediction.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import shutil

from tools.artifact.container import Artifact, TensorObject
from tools.ppl import compare_nvfp4 as common
from tools.ppl import select_nv_aligned as aligned

BUILD = common.ROOT/'build-r9700-precision-search-20260923'
PROFILES = aligned.PROFILES
FAMILIES = dict(a8=0, **{'gate-up':1, 'mlp':2, 'projections':3, 'attn-input':4, 'gate-up-attn-input':5})
RECIPES = tuple(f'{base}-{endpoint}' for base in ('cap26', 'cap22')
                for endpoint in ('embed', 'head', 'both')) + ('cap22-base',)
CAP22_BASE = common.ROOT/'profiles/ppl/r9700-nv-aligned-precision-20260923/artifacts/no-late-mlp.ninfer'


def weights_id(recipe):
    base, endpoint = recipe.split('-')
    body = 'selective-cap' if base == 'cap26' else 'selective-no-late-mlp'
    if endpoint == 'base':
        return f'r9700-q4-fp8-{body}-n16k16-eval'
    endpoint = 'endpoints' if endpoint == 'both' else endpoint
    return f'r9700-q4-fp8-{body}-{endpoint}-w8-n16k16-eval'


def uses_a4(profile, shape):
    n, k = shape
    gate = (n, k) == (34816, 5120)
    down = (n, k) == (5120, 17408)
    attention = (n, k) == (7168, 5120)
    other = (n, k) in ((4096, 5120), (12288, 5120), (5120, 6144))
    return {
        'a8':False, 'gate-up':gate, 'mlp':gate or down,
        'projections':gate or down or attention or other,
        'attn-input':attention, 'gate-up-attn-input':gate or attention,
    }[profile]


def cost(artifact, profile):
    with Artifact(artifact) as opened:
        objects = {o.name:o for o in opened.objects if isinstance(o, TensorObject)}
        layer_objects = [o for o in objects.values() if o.name.startswith('text/layers/')]
        head, embedding = objects['text/output_head'], objects['text/token_embedding']
        q4_macs = a4_macs = fp8_macs = 0
        for obj in layer_objects:
            if len(obj.shape) != 2:
                continue
            macs = math.prod(obj.shape)
            if obj.format == 'Q4G64_F16S':
                q4_macs += macs
                if uses_a4(profile, obj.shape):
                    a4_macs += macs
            elif obj.format == 'F8E4M3_ROW_F32S':
                fp8_macs += macs
        # Q4 A8 decomposes into unsigned-low/signed-high IU4 products; A4 uses one.
        work = 2*q4_macs-a4_macs
        layer_bytes = sum(o.bytes for o in layer_objects)
        embedding_row_bytes = embedding.bytes/embedding.shape[0]
        return dict(file_bytes=Path(artifact).stat().st_size,
            text_layer_weight_bytes=layer_bytes, head_weight_bytes=head.bytes,
            embedding_table_bytes=embedding.bytes, embedding_row_bytes=embedding_row_bytes,
            ordinary_weight_stream_bytes=layer_bytes+head.bytes+embedding_row_bytes+objects['text/final_norm'].bytes,
            q4_body_macs_per_prefill_token=q4_macs, a4_body_macs_per_prefill_token=a4_macs,
            fp8_body_macs_per_prefill_token=fp8_macs,
            q4_integer_product_work=work,
            q4_work_fraction_of_same_weights_a8=work/(2*q4_macs),
            head_format=head.format, embedding_format=embedding.format)


def snapshot(out, profile, w8_bits):
    tag = profile if w8_bits == 8 else f'{profile}-w8a16'
    dest = out/'bin'/f'{tag}-ppl'
    if dest.exists():
        raise FileExistsError(dest)
    shutil.copy2(BUILD/'apps/ninfer-ppl', dest)
    common.write_json(out/f'{tag}-binary.json', dict(path=str(dest), sha256=common.digest(dest),
        activation_profile=PROFILES[profile], w8_activation_bits=w8_bits))


def quality(out, recipe, profile, w8_bits):
    tag = profile if w8_bits == 8 else f'{profile}-w8a16'
    name = f'{recipe}-{tag}'
    binary = out/'bin'/f'{tag}-ppl'
    artifact = CAP22_BASE if recipe == 'cap22-base' else out/'artifacts'/f'{recipe}.ninfer'
    with Artifact(artifact) as opened:
        if opened.identity.model_id != 'qwen3.8-27b' or opened.identity.weights_id != weights_id(recipe):
            raise ValueError('artifact identity does not match endpoint label')
    for schedule, prefix in [('prefill','quality'), ('decode','decode-quality')]:
        for sample in common.SAMPLES:
            cell = out/f'{prefix}-{name}_a8-{sample}'
            if not cell.exists():
                print(f'PPL {name} {schedule} {sample}', flush=True)
                common.run([binary, '--weights', artifact, '--ids', out/f'{sample}.ids',
                    '--tokens','4096','--skip', '2048' if schedule == 'prefill' else '3967',
                    '--prefill-chunk','2048','--schedule',schedule,'--device','0',
                    '--scheme',name,'--out-json',cell/'report.json'], cell)
            receipt = json.loads((cell/'receipt.json').read_text())
            if receipt['exit_code'] != 0 or receipt['executable_sha256'] != common.digest(binary):
                raise ValueError(f'failed or mismatched cell retained: {cell}')
            report, _ = common.nlls(cell, schedule)
            if (report['q4_activation_profile'] != PROFILES[profile] or
                    report['q4_activation_bits'] != 8 or report['w8_activation_bits'] != w8_bits or
                    report['weights_id'] != weights_id(recipe) or
                    Path(report['weights']).resolve() != artifact.resolve()):
                raise ValueError('scorer precision/artifact does not match requested cell')
            print(f"  PPL {report['ppl']:.6f}", flush=True)
    if not (out/f'{name}-quality.json').exists():
        with contextlib.redirect_stdout(io.StringIO()):
            aligned.summarize(out, name)
    if not (out/f'{name}-cost.json').exists():
        common.write_json(out/f'{name}-cost.json',cost(artifact,profile))
    summary=json.loads((out/f'{name}-quality.json').read_text())
    worst=max(r['ratio'] for rows in summary['samples'].values() for r in rows.values())
    print(f"COMPLETE {name}: worst PPL delta={100*(worst-1):+.3f}%, within2%={summary['within_2pct']}",flush=True)


def report(out):
    rows = {}
    for recipe in RECIPES:
        for profile in PROFILES:
            for suffix in ('','-w8a16'):
                name=f'{recipe}-{profile}{suffix}'
                path=out/f'{name}-quality.json'
                if not path.exists():
                    continue
                quality=json.loads(path.read_text())
                costs=json.loads((out/f'{name}-cost.json').read_text())
                rows[name]=dict(quality=quality, theoretical=costs)
    lines=['# Endpoint precision: measured quality, theoretical cost', '',
        'Same three texts and six spans as the immutable NVFP4 reference. No speed benchmarks.',
        'Weight-stream bytes exclude KV, activations, rereads, workspaces and launch/reduction cost.',
        'Q4 work is the low/high integer-product count proxy, not whole-prefill latency; FP8 work is separate.', '',
        'Original Q4-endpoint cap26 A8/gate-up controls remain in `profiles/ppl/r9700-fp8-capped-selection-20260923/`.',
        'Passing this screen does not establish superiority to those controls or production admission.', '',
        '| Variant | File GB | Worst prefill PPL delta | Worst decode PPL delta | New severe prefill / decode (Wiki, technical, code) | Weight stream GB/step | Q4 work vs same-weight A8 | 2% screen |',
        '|---|---:|---:|---:|---|---:|---:|---|']
    for name,row in rows.items():
        q,c=row['quality'],row['theoretical']
        worst=[100*(max(r['ratio'] for r in q['samples'][s].values())-1) for s in ('prefill','decode')]
        severe=['/'.join(str(q['samples'][s][t]['new_severe']) for t in common.SAMPLES) for s in ('prefill','decode')]
        lines.append(f"|{name}|{c['file_bytes']/1e9:.3f}|{worst[0]:+.3f}%|{worst[1]:+.3f}%|{' ; '.join(severe)}|{c['ordinary_weight_stream_bytes']/1e9:.3f}|{c['q4_work_fraction_of_same_weights_a8']:.3f}|{'pass' if q['within_2pct'] else 'fail'}|")
    # Explicit output generation, not experiment-cell mutation. Reports can be refreshed.
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    (out/'comparison.json').write_text(json.dumps(dict(rows=rows,
        limitations=['No timing or tok/s estimate.', 'Finite three-text quality screen, not universal equivalence.',
            'Q4 work and FP8 MACs are separate; instruction counts alone do not predict speed.',
            'Prefill A4 does not reduce decode weight bytes; W8 embedding adds residency but only one row per step.',
            'PPL prefill scores many head rows; ordinary prefill needs only final logits.',
            'W8 head currently uses BF16 activations at T1-3 even in the W8 A8 build; changing that requires new quality checks.',
            'Head arithmetic is not in the body-work proxy; equal W8A8/W8A16 byte proxies are not a predicted speed tie.',
            'No speculative correctness, acceptance or DFlash throughput measured.']),indent=2)+'\n')
    print(f'{len(rows)} completed configurations; '+', '.join(n for n,r in rows.items() if r['quality']['within_2pct']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','snapshot','quality','report'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--profile',choices=PROFILES,default='a8')
    parser.add_argument('--recipe',choices=RECIPES)
    parser.add_argument('--w8-bits',type=int,choices=(8,16),default=8)
    args=parser.parse_args();out=args.out.resolve()
    if args.action=='prepare': aligned.prepare(out)
    elif args.action=='snapshot': snapshot(out,args.profile,args.w8_bits)
    elif args.action=='report': report(out)
    else:
        if not args.recipe: parser.error('--recipe required')
        quality(out,args.recipe,args.profile,args.w8_bits)


if __name__=='__main__': main()
