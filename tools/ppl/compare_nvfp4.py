#!/usr/bin/env python3
"""Bounded, matched-text AMD recipe comparison against the local 5090 NVFP4 model.

This is an evaluation report, not the terminal production/capacity selection campaign.
All outputs are create-only; successful cells may be resumed without rerunning them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT.parent
NV_REPO = LOCAL / 'ninfer-dylan2'
NV_MODEL = LOCAL / 'models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer'
ARTIFACTS = {
    'all_q4': ROOT / 'out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer',
    'mixed_mse': ROOT / 'out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer',
    'selective': ROOT / 'out/qwen3.8-27b-r9700-q4-selective-protected-n16k16-eval.ninfer',
    'four_role': ROOT / 'out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer',
    'selective_tiled': LOCAL / 'models/qwen3.8-27b-r9700-q4-selective-protected-dflash2-q4/qwen3.8-27b-r9700-q4-selective-protected-n16k16-dflash2-q4-head-n16k16-eval.ninfer',
}
CONFIGS = [(r, a) for r in ('all_q4', 'mixed_mse') for a in (4, 8)] + [('selective', 8), ('four_role', 8), ('selective_tiled', 8)]
SAMPLES = ('wiki', 'technical', 'code')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')


def run(command, directory):
    directory.mkdir()
    command = [str(x) for x in command]
    write_json(directory / 'command.json', command)
    start = time.time()
    with (directory / 'stdout.log').open('x') as out, (directory / 'stderr.log').open('x') as err:
        result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err)
    write_json(directory / 'receipt.json', dict(exit_code=result.returncode, seconds=time.time()-start,
               executable_sha256=digest(command[0])))
    if result.returncode:
        raise RuntimeError(f'failed cell retained: {directory}')


def prepare(out):
    out.mkdir()
    (out / 'bin').mkdir()
    # Preserve an immutable binary while the user continues development in the NVIDIA checkout.
    nv = out / 'bin/nvidia-ppl'
    shutil.copy2(NV_REPO / 'build/apps/ninfer-ppl', nv)
    sources = {
        'technical': ROOT / 'docs/maintainer/concurrent-inference-architecture.md',
        'code': ROOT / 'src/targets/qwen3/impl/runtime/program_impl.h',
    }
    manifest = dict(tokens=4096, skip=2048, chunk=2048, samples={},
                    nvidia_commit=subprocess.check_output(['git', '-C', str(NV_REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                    nvidia_binary_sha256=digest(nv), nvidia_model=str(NV_MODEL),
                    amd_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    for name in SAMPLES:
        if name == 'wiki':
            source = ROOT / 'tools/ppl/corpus.ids'
            ids = source.read_text().split()[:4096]
        else:
            source = sources[name]
            shutil.copy2(source, out / f'{name}.txt')
            for platform, binary, model in (
                ('amd', ROOT / 'build-r9700/apps/ninfer-ppl', ARTIFACTS['all_q4']),
                ('nvidia', nv, NV_MODEL),
            ):
                run([binary, '--encode', '--weights', model, '--text', out / f'{name}.txt',
                     '--ids', out / f'{name}-{platform}.ids', '--tokens', '4096'], out / f'encode-{name}-{platform}')
            ids = (out / f'{name}-amd.ids').read_text().split()
            assert ids == (out / f'{name}-nvidia.ids').read_text().split(), 'tokenizer mismatch'
        assert len(ids) == 4096
        with (out / f'{name}.ids').open('x') as f:
            f.write(' '.join(ids) + '\n')
        manifest['samples'][name] = dict(source=str(source), source_sha256=digest(source),
                                        ids_sha256=digest(out / f'{name}.ids'))
    write_json(out / 'manifest.json', manifest)


def require_uniform_build(build, bits):
    """Do not silently label the mixed delivery default as a uniform control."""
    settings = {}
    for line in (build/'CMakeCache.txt').read_text().splitlines():
        if line.startswith(('#', '//')) or '=' not in line:
            continue
        name, value = line.split('=', 1)
        settings[name.split(':', 1)[0]] = value
    expected = dict(NINFER_R9700_Q4_ACTIVATION_BITS=str(bits),
                    NINFER_R9700_Q4_PREFILL_A4_FAMILIES='0',
                    NINFER_R9700_W8_ACTIVATION_BITS='8')
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{build}: uniform comparison needs Q4 bits{bits}, prefill family0, W8 bits8; '
                         'the delivered family1 build is a mixed profile, not this control')


def snapshot_amd(out):
    records = {}
    for a in (4, 8):
        build = ROOT / ('build-r9700-pareto-a4-20260922' if a == 4 else 'build-r9700')
        require_uniform_build(build, a)
        for kind, rel in (('ppl', 'apps/ninfer-ppl'), ('bench', 'bench/ninfer_bench')):
            target = out / f'bin/amd-a{a}-{kind}'
            if target.exists():
                raise ValueError(f'snapshot exists: {target}')
            shutil.copy2(build / rel, target)
            records[target.name] = digest(target)
    write_json(out / 'amd-binaries.json', records)
    write_json(out / 'artifacts.json', {name: dict(path=str(path), bytes=path.stat().st_size,
               mtime_ns=path.stat().st_mtime_ns) for name,path in {**ARTIFACTS, 'nvfp4': NV_MODEL}.items()})


def quality(out, platform, schedule='prefill', only=None, binary_override=None, tag=None):
    rows = [('nvfp4', None)] if platform == 'nvidia' else CONFIGS
    for recipe, a in rows:
        name = recipe if a is None else f'{recipe}_a{a}'
        if only and name != only:
            continue
        binary = (binary_override.resolve() if binary_override else
                  out / ('bin/nvidia-ppl' if a is None else f'bin/amd-a{a}-ppl'))
        model = NV_MODEL if a is None else ARTIFACTS[recipe]
        selected = {}
        for sample in SAMPLES:
            prefix = 'quality' if schedule == 'prefill' else 'decode-quality'
            cell = out / (f'{prefix}-{name}-{sample}' + (f'-{tag}' if tag else ''))
            selected[sample] = cell.name
            if cell.exists():
                receipt = json.loads((cell / 'receipt.json').read_text())
                assert receipt['exit_code'] == 0 and receipt['executable_sha256'] == digest(binary)
                continue
            print(f'PPL {schedule} {name} {sample}', flush=True)
            command = [binary, '--weights', model, '--ids', out / f'{sample}.ids',
                       '--tokens', '4096', '--skip', '2048' if schedule == 'prefill' else '3967', '--prefill-chunk', '2048',
                       '--schedule', schedule, '--device', '0', '--scheme', name,
                       '--out-json', cell / 'report.json']
            if a is None:
                command += ['--kv-dtype', 'nvfp4']
            run(command, cell)
            report = json.loads((cell / 'report.json').read_text())
            assert report['non_finite'] == 0 and report['tokens_scored'] == (2047 if schedule == 'prefill' else 128)
            if a is not None:
                assert report['q4_activation_bits'] == a and report['w8_activation_bits'] == 8
            print(f"  PPL {report['ppl']:.6f}", flush=True)
        selection = out / f'quality-selection-{name}-{schedule}.json'
        if selection.exists():
            assert json.loads(selection.read_text()) == selected
        else:
            write_json(selection, selected)


def speed(out, only=None, binary_override=None, tag=None):
    for recipe, a in CONFIGS:
        name = f'{recipe}_a{a}'
        if only and name != only:
            continue
        binary = binary_override.resolve() if binary_override else out / f'bin/amd-a{a}-bench'
        selected = {}
        for sample in SAMPLES:
            cell = out / (f'speed-{name}-{sample}' + (f'-{tag}' if tag else ''))
            selected[sample] = cell.name
            if cell.exists():
                receipt = json.loads((cell / 'receipt.json').read_text())
                assert receipt['exit_code'] == 0 and receipt['executable_sha256'] == digest(binary)
                continue
            power = Path('/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level')
            assert power.read_text().strip() == 'auto'
            print(f'SPEED {name} {sample}', flush=True)
            run([binary, '--weights', ARTIFACTS[recipe], '--corpus', out / f'{sample}.ids',
                 '--device', '0', '--concurrency', '1', '--whole-pg', '4096,128',
                 '--prefill-chunk', '2048', '--max-ctx', '4224', '--kv-capacity', 'workload',
                 '--draft-tokens', '0', '--warmup', '1', '-r', '3', '--retain-token-ids',
                 '--output', 'json', '--output-file', cell / 'report.json'], cell)
            assert power.read_text().strip() == 'auto'
        selection = out / f'speed-selection-{name}.json'
        if selection.exists():
            assert json.loads(selection.read_text()) == selected
        else:
            write_json(selection, selected)


def nlls(cell, schedule='prefill'):
    report = json.loads((cell / 'report.json').read_text())
    count, skip = (2047, 2048) if schedule == 'prefill' else (128, 3967)
    assert report['prompt_tokens'] == 4096 and report['skip_tokens'] == skip
    assert report['tokens_scored'] == count and report['non_finite'] == 0
    assert report['schedule'] == schedule and report['prefill_chunk'] == 2048
    data = (cell / 'report.nllf32').read_bytes()
    assert len(data) == count * 4
    values = struct.unpack(f'<{count}f', data)
    assert all(math.isfinite(x) for x in values)
    assert math.isclose(statistics.mean(values), report['mean_nll'], abs_tol=2e-6)
    return report, values


def analyze(out):
    references = {s: nlls(out / f'quality-nvfp4-{s}') for s in SAMPLES}
    decode_references = {s: nlls(out / f'decode-quality-nvfp4-{s}', 'decode') for s in SAMPLES}
    rows = {}
    for recipe, a in CONFIGS:
        name = f'{recipe}_a{a}'
        row = dict(artifact=str(ARTIFACTS[recipe]), file_gb=ARTIFACTS[recipe].stat().st_size / 1e9,
                   q4_activation_bits=a, samples={})
        for sample in SAMPLES:
            def quality_path(schedule):
                selection = out / f'quality-selection-{name}-{schedule}.json'
                prefix = 'quality' if schedule == 'prefill' else 'decode-quality'
                return out / (json.loads(selection.read_text())[sample] if selection.exists()
                              else f'{prefix}-{name}-{sample}')
            quality, values = nlls(quality_path('prefill'))
            reference, ref = references[sample]
            deltas = [x-y for x,y in zip(values, ref)]
            decode, dv = nlls(quality_path('decode'), 'decode')
            decode_reference, dr = decode_references[sample]
            selection = out / f'speed-selection-{name}.json'
            directory = (json.loads(selection.read_text())[sample] if selection.exists()
                         else f'speed-{name}-{sample}')
            receipt = json.loads((out / directory / 'receipt.json').read_text())
            assert receipt['exit_code'] == 0
            report = json.loads((out / directory / 'report.json').read_text())
            test, = report['tests']
            reps = test['reps']
            assert len(reps) == 3 and test['n_prompt'] == 4096 and test['n_gen'] == 128
            assert all(not r['speculative']['enabled'] and r['decode_output_tokens'] == 128 for r in reps)
            assert all(r['generated_token_ids_by_lane'] == reps[0]['generated_token_ids_by_lane'] for r in reps)
            pp = [4096 / r['timings']['prefill_seconds'] for r in reps]
            tg = [128 / r['timings']['decode_seconds'] for r in reps]
            row['samples'][sample] = dict(
                ppl=quality['ppl'], nvfp4_ppl=reference['ppl'],
                mean_nll_delta=statistics.mean(deltas), ppl_ratio=math.exp(statistics.mean(deltas)),
                new_severe_positions=sum(x >= 10 and y < 10 for x,y in zip(values,ref)),
                resolved_severe_positions=sum(x < 10 and y >= 10 for x,y in zip(values,ref)),
                decode_ppl=decode['ppl'], nvfp4_decode_ppl=decode_reference['ppl'],
                decode_mean_nll_delta=statistics.mean(x-y for x,y in zip(dv,dr)),
                decode_ppl_ratio=decode['ppl']/decode_reference['ppl'],
                decode_new_severe_positions=sum(x >= 10 and y < 10 for x,y in zip(dv,dr)),
                prefill_tok_s=statistics.median(pp), decode_tok_s=statistics.median(tg),
                prefill_range=[min(pp),max(pp)], decode_range=[min(tg),max(tg)])
        row['pooled_ppl_ratio'] = math.exp(statistics.mean(v['mean_nll_delta'] for v in row['samples'].values()))
        row['prefill_within_2pct_each_text'] = all(v['ppl_ratio'] <= 1.02 for v in row['samples'].values())
        row['prefill_within_5pct_each_text'] = all(v['ppl_ratio'] <= 1.05 for v in row['samples'].values())
        row['decode_within_2pct_each_text'] = all(v['decode_ppl_ratio'] <= 1.02 for v in row['samples'].values())
        row['decode_within_5pct_each_text'] = all(v['decode_ppl_ratio'] <= 1.05 for v in row['samples'].values())
        rows[name] = row
    # Every text contributes its own quality and speed objectives; no averaged speed hides a loss.
    def objectives(row, include_decode=False):
        values = [value for s in SAMPLES for value in (
            row['samples'][s]['mean_nll_delta'], row['samples'][s]['new_severe_positions'],
            -row['samples'][s]['prefill_tok_s'], -row['samples'][s]['decode_tok_s'])]
        if include_decode:
            values += [value for s in SAMPLES for value in (
                row['samples'][s]['decode_mean_nll_delta'], row['samples'][s]['decode_new_severe_positions'])]
        return values
    for name, row in rows.items():
        for key, include_decode in [('dominated_by', False), ('decode_aware_dominated_by', True)]:
            v = objectives(row, include_decode)
            row[key] = [other for other, r in rows.items() if other != name and
                        all(x <= y for x,y in zip(objectives(r,include_decode),v)) and
                        any(x < y for x,y in zip(objectives(r,include_decode),v))]
    result = dict(rows=rows, measured_frontier=[n for n,r in rows.items() if not r['dominated_by']],
                  decode_aware_frontier=[n for n,r in rows.items() if not r['decode_aware_dominated_by']],
                  limitations=['Three local text samples, not a general capability benchmark.',
                               'PPL measures teacher-forced quality on the stated scoring spans, not DFlash verification qualification.',
                               'Different GPU cache/weight formats: product comparison, not isolated activation quantization.',
                               'A4 evaluator has fewer optimized routes; timings measure implemented paths, not hardware limits.',
                               'Pareto dominance uses measured medians without a statistical significance claim.'])
    write_json(out / 'comparison.json', result)
    lines = ['# Matched multi-text quality and AMD speed', '',
             'C1, P4096/G128, chunk2048, ordinary decode; warm1/r3 median. PPL scores2047 positions after2048 warmup tokens.', '',
             '## Summary', '',
             'Speeds span the three per-text medians. Worst PPL changes compare each scoring schedule only with its matched NVFP4 reference. Sizes are decimal GB.' +
             (' The tiled selective file includes its unused DFlash companion.' if 'selective_tiled_a8' in rows else ''), '',
             '| Configuration | File GB | Worst prefill PPL change | Worst decode PPL change | Prefill tok/s | Decode tok/s |',
             '|---|---:|---:|---:|---:|---:|']
    for name,row in rows.items():
        values = list(row['samples'].values())
        pp = [v['prefill_tok_s'] for v in values]
        tg = [v['decode_tok_s'] for v in values]
        lines.append(f"| {name} | {row['file_gb']:.2f} | {(max(v['ppl_ratio'] for v in values)-1)*100:+.2f}% | {(max(v['decode_ppl_ratio'] for v in values)-1)*100:+.2f}% | {min(pp):.0f}–{max(pp):.0f} | {min(tg):.2f}–{max(tg):.2f} |")
    lines += ['', '## Per-text prefill quality and speed', '',
             '| Configuration | Text | PPL | NVFP4 PPL | PPL ratio | New severe | Prefill tok/s | Decode tok/s |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,row in rows.items():
        for sample,v in row['samples'].items():
            lines.append(f"| {name} | {sample} | {v['ppl']:.5f} | {v['nvfp4_ppl']:.5f} | {v['ppl_ratio']:.4f} | {v['new_severe_positions']} | {v['prefill_tok_s']:.1f} | {v['decode_tok_s']:.2f} |")
    lines += ['', 'Measured prefill-quality frontier: '+', '.join(result['measured_frontier']),
              'Including short decode quality: '+', '.join(result['decode_aware_frontier']), ''] + result['limitations']
    lines += ['', '## Short teacher-forced decode check', '',
              'Same4096-token inputs; only last128 next-token positions scored (skip3967). Do not compare these PPLs with the larger prefill scoring span.', '',
              '| Configuration | Text | Decode PPL | NVFP4 decode PPL | Ratio | New severe |',
              '|---|---|---:|---:|---:|---:|']
    for name,row in rows.items():
        for sample,v in row['samples'].items():
            lines.append(f"| {name} | {sample} | {v['decode_ppl']:.5f} | {v['nvfp4_decode_ppl']:.5f} | {v['decode_ppl_ratio']:.4f} | {v['decode_new_severe_positions']} |")
    with (out / 'comparison.md').open('x') as f: f.write('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'snapshot-amd', 'quality-amd', 'quality-nvidia', 'decode-quality-amd', 'decode-quality-nvidia', 'speed', 'analyze'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--only', help='run only this explicit configuration name')
    parser.add_argument('--bench-bin', type=Path, help='explicit frozen benchmark for a host-only planner correction')
    parser.add_argument('--ppl-bin', type=Path, help='explicit frozen PPL binary for a host-only planner correction')
    parser.add_argument('--tag', help='fresh cell suffix; preserves failed or superseded evidence')
    args = parser.parse_args()
    if args.only and args.only not in {'nvfp4', *(f'{r}_a{a}' for r,a in CONFIGS)}:
        parser.error('unknown configuration: ' + args.only)
    out = args.out.resolve()
    if args.action == 'prepare': prepare(out)
    elif args.action == 'snapshot-amd': snapshot_amd(out)
    elif args.action.startswith('quality-'): quality(out, args.action.split('-')[1], only=args.only, binary_override=args.ppl_bin, tag=args.tag)
    elif args.action.startswith('decode-quality-'): quality(out, args.action.split('-')[2], 'decode', args.only, args.ppl_bin, args.tag)
    elif args.action == 'speed': speed(out, args.only, args.bench_bin, args.tag)
    else: analyze(out)


if __name__ == '__main__':
    main()
