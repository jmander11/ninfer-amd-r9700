#!/usr/bin/env python3
"""Serial C1 prefill speed / matched-token PPL ladders for dense and XAttention.

This is measurement tooling, not XAttention admission or a runtime route selector.
Explicit binaries own the attention profile. See context_ladder.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
POWER = Path('/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level')
PRECISION = ('q4_activation_bits', 'q4_activation_profile', 'q4_prefill_gate_up_a4',
             'q4_prefill_cta_profile', 'w8_activation_bits', 'kv_value_group',
             'kv_plane_layouts', 'fp8_qk_wmma_enabled', 'fp8_qk_wmma_profile',
             'fp8_qk_wmma_t1_min_context', 'fp8_qk_wmma_t2_min_context')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    # A crash must not leave a half-written summary that hides completed cells.
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def profile(report):
    enabled = report['xattention_qualification']
    require(type(enabled) is bool, 'missing/invalid attention qualification metadata')
    return report['xattention_profile'] if enabled else 'dense'


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def speed_cell(raw, context, contract, attention):
    config = raw['config']
    require(profile(config) == attention, 'binary attention profile differs from requested profile')
    require(raw['artifact']['path'] == contract['weights']['path'] and
            raw['artifact']['file_size_bytes'] == contract['weights']['bytes'], 'weight artifact mismatch')
    expected = dict(concurrency=1, prefill_chunk=contract['chunk'],
                    max_context=contract['max_context'], spec=contract['spec'],
                    draft_tokens=contract['draft_tokens'], warmup=contract['warmup'],
                    repetitions=contract['repetitions'], corpus_path=contract['ids']['path'],
                    corpus_tokens=contract['ids']['tokens'],
                    proposal_head='optimized' if contract['lm_head_draft'] else 'full')
    for key, value in expected.items():
        require(config[key] == value, f'speed report mismatch: {key}')
    require(config['kv_cache_format'] == 'fp8-k-int4-v', 'wrong cache format')
    require(config['use_device_graph'] is True and config['isolate_prompt_decode'] is False,
            'unexpected execution policy')
    require(raw['environment']['architecture_name'] == 'gfx1201' and
            raw['environment']['gpu_name'] == 'AMD Radeon AI PRO R9700' and
            raw['environment']['device_id'] == 0, 'not device-0 R9700/gfx1201 evidence')
    require(len(raw['tests']) == 1, 'expected one test per context')
    test = raw['tests'][0]
    require(test['kind'] == 'pp' and test['n_prompt'] == context and test['n_gen'] == 0,
            'not the requested prefill-only cell')
    require(len(test['reps']) == contract['repetitions'], 'incomplete repetitions')
    tails = []
    for rep in test['reps']:
        lanes = rep['prefill_tail_by_lane']
        require(len(lanes) == 1, 'expected one lane')
        require(positive(lanes[0]['tok_s']) and positive(lanes[0]['window_s']), 'invalid tail rate')
        tails.append(lanes[0])
    result = dict(context=context, average_tok_s=test['prefill_active_tok_s_mean'],
                  average_stddev=test['prefill_active_tok_s_stddev'],
                  wave_tok_s=test['prefill_tok_s_mean'], prefill_seconds=test['prefill_seconds_mean'],
                  tail_tok_s=statistics.mean(t['tok_s'] for t in tails), tail_reps=tails)
    require(all(positive(result[k]) for k in ('average_tok_s', 'wave_tok_s', 'prefill_seconds')),
            'invalid prefill timing')
    return result, {k: config[k] for k in PRECISION}, raw['load']['weights_id']


def sidecar(path, code):
    data = path.read_bytes()
    require(len(data) > 0 and len(data) % 4 == 0, f'invalid sidecar {path}')
    return list(struct.unpack('<' + code * (len(data) // 4), data))


def ppl_cell(raw, report_path, context, contract, attention):
    require(profile(raw) == attention, 'binary attention profile differs from requested profile')
    skip = context // 2 if contract['skip'] == 'half' else int(contract['skip'])
    expected = dict(weights=contract['weights']['path'], schedule='prefill', spec='none',
                    draft_tokens=0, device_graph=True, prefill_chunk=contract['chunk'],
                    prompt_tokens=context, skip_tokens=skip, tokens_scored=context-skip-1,
                    argmax_tokens=context-skip-1, non_finite=0, kv_format='fp8-k-int4-v')
    for key, value in expected.items():
        require(raw[key] == value, f'PPL report mismatch: {key}')
    nll = sidecar(report_path.with_suffix('.nllf32'), 'f')
    argmax = sidecar(report_path.with_suffix('.argmaxi32'), 'i')
    require(len(nll) == len(argmax) == context-skip-1 and all(math.isfinite(n) for n in nll),
            'NLL/argmax sidecars must be complete, finite and aligned')
    require(all(0 <= token < 248077 for token in argmax), 'invalid argmax token')
    mean = statistics.mean(nll)
    require(math.isclose(mean, raw['mean_nll'], rel_tol=1e-6, abs_tol=1e-6), 'NLL mean mismatch')
    require(math.isclose(math.exp(mean), raw['ppl'], rel_tol=1e-6), 'PPL mismatch')
    result = dict(context=context, skip_tokens=skip, tokens_scored=len(nll),
                  mean_nll=mean, ppl=math.exp(mean), terrible_nll=raw['terrible_nll'])
    return result, {k: raw[k] for k in PRECISION}, raw['weights_id']


def command_for(args, context, report):
    command = [str(args.binary), '--weights', str(args.weights)]
    if args.kind == 'speed':
        command += ['--corpus', str(args.ids), '--device', '0', '--concurrency', '1',
                    '-p', str(context), '--prefill-chunk', str(args.chunk),
                    '--max-ctx', str(args.max_context), '--kv-capacity', 'workload']
        if args.spec != 'none':
            command += ['--spec', args.spec, '--draft-tokens', str(args.draft_tokens)]
        if args.lm_head_draft:
            command += ['--lm-head-draft']
        command += ['--warmup', str(args.warmup), '-r', str(args.repetitions),
                    '--output', 'json', '--output-file', str(report)]
    else:
        command += ['--ids', str(args.ids), '--device', '0', '--schedule', 'prefill',
                    '--tokens', str(context), '--skip', args.skip,
                    '--prefill-chunk', str(args.chunk), '--out-json', str(report)]
    return command


def run(args):
    args.binary = args.binary.resolve(strict=True)
    args.weights = args.weights.resolve(strict=True)
    args.ids = args.ids.resolve(strict=True)
    args.out = args.out.resolve()
    require(args.chunk > 0 and args.repetitions > 0 and args.warmup >= 0, 'invalid chunk/repetitions')
    require(args.contexts == sorted(set(args.contexts)) and min(args.contexts) >= 2,
            'contexts must be increasing, unique, and >=2')
    ids = [int(word) for word in args.ids.read_text().split()]
    require(len(ids) >= 2 and all(0 <= token < 248077 for token in ids), 'invalid corpus')
    if args.kind == 'ppl':
        require(args.spec == 'none' and args.draft_tokens == 0 and not args.lm_head_draft,
                'PPL ladder is non-speculative; draft options are speed-only')
        require(len(ids) >= max(args.contexts), 'PPL never cycles IDs: supply a long enough natural corpus')
        require(args.skip == 'half' or (args.skip.isdecimal() and int(args.skip) < min(args.contexts)-1),
                'skip must leave scored tokens at every context')
    else:
        require(args.max_context > max(args.contexts), 'max-context must leave room for first output token')
        require(args.spec != 'none' or (args.draft_tokens == 0 and not args.lm_head_draft),
                'draft options require --spec dflash')
        require(args.spec == 'none' or args.draft_tokens in (4, 5), 'DFlash speed setup requires K4 or K5')
        require(len(ids) >= max(args.contexts) or args.allow_cyclic_speed_corpus,
                'short speed corpus requires explicit --allow-cyclic-speed-corpus')
    contract = dict(kind=args.kind, contexts=args.contexts, chunk=args.chunk,
                    weights=dict(path=str(args.weights), bytes=args.weights.stat().st_size),
                    ids=dict(path=str(args.ids), sha256=sha(args.ids), tokens=len(ids)))
    if args.kind == 'speed':
        contract.update(max_context=args.max_context, spec=args.spec, draft_tokens=args.draft_tokens,
                        lm_head_draft=args.lm_head_draft, warmup=args.warmup, repetitions=args.repetitions,
                        corpus_cycled=len(ids) < max(args.contexts))
    else:
        contract.update(skip=args.skip, schedule='prefill', spec='none')
    manifest = dict(schema=1, contract=contract, attention=args.attention, binary=str(args.binary),
                    binary_sha256=None if args.collect_only else sha(args.binary),
                    provenance='retained reports; historical power/binary provenance must accompany source package'
                    if args.collect_only else 'serial unprofiled; power auto checked before/after each cell')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists():
        require(read(manifest_path) == manifest, 'resume manifest differs; use a fresh output directory')
    else:
        write(manifest_path, manifest)
    results = []
    precision = weights_id = environment = None
    for context in args.contexts:
        report = args.out / f'p{context}.json'
        command = command_for(args, context, report)
        receipt = args.out / f'p{context}-command.json'
        completion = args.out / f'p{context}-exit.json'
        if report.exists():
            require(receipt.exists() and read(receipt) == command, 'existing report command differs')
            if not args.collect_only:
                require(completion.exists() and read(completion) == dict(returncode=0, power_after='auto'),
                        'prior cell did not complete successfully; retain it and use a fresh directory')
        else:
            require(not args.collect_only, f'missing retained report {report}')
            require(not receipt.exists(), f'previous cell did not complete: {receipt}; retain it and use a fresh directory')
            require(POWER.read_text().strip() == 'auto', 'R9700 power must already be auto')
            write(receipt, command)
            print(f'RUN {args.kind} {args.attention} P{context}', flush=True)
            with report.with_suffix('.log').open('x') as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            power_after = POWER.read_text().strip()
            write(completion, dict(returncode=process.returncode, power_after=power_after))
            require(process.returncode == 0, f'cell failed with exit {process.returncode}: {report.with_suffix(".log")}')
            require(power_after == 'auto', 'power changed during measurement')
        raw = read(report)
        if args.kind == 'speed':
            row, cell_precision, cell_weights_id = speed_cell(raw, context, contract, args.attention)
            if environment is not None:
                require(environment == raw['environment'], 'hardware/toolchain changed within ladder')
            environment = raw['environment']
        else:
            row, cell_precision, cell_weights_id = ppl_cell(raw, report, context, contract, args.attention)
        if precision is not None:
            require(precision == cell_precision and weights_id == cell_weights_id,
                    'model/precision changed within ladder')
        precision, weights_id = cell_precision, cell_weights_id
        row['report'] = str(report)
        results.append(row)
        write(args.out / 'ladder.json', dict(**manifest, precision=precision, weights_id=weights_id,
                                           environment=environment, cells=results))
        print(json.dumps(row), flush=True)


def compare(dense_path, candidate_path):
    dense, candidate = read(dense_path), read(candidate_path)
    require(dense['attention'] == 'dense' and candidate['attention'] != 'dense',
            'expected a dense control and an XAttention candidate')
    for key in ('schema', 'contract', 'precision', 'weights_id', 'environment'):
        require(dense[key] == candidate[key], f'unmatched comparison: {key}')
    require([r['context'] for r in dense['cells']] == dense['contract']['contexts'] and
            [r['context'] for r in candidate['cells']] == candidate['contract']['contexts'],
            'cannot compare incomplete ladders')
    rows = []
    for left, right in zip(dense['cells'], candidate['cells']):
        # Revalidate the retained raw reports and PPL sidecars, not just summaries.
        for cell, ladder in ((left, dense), (right, candidate)):
            path = Path(cell['report'])
            raw = read(path)
            if dense['contract']['kind'] == 'speed':
                row, precision, weights_id = speed_cell(raw, cell['context'], ladder['contract'], ladder['attention'])
                require(raw['environment'] == ladder['environment'], 'retained environment changed')
            else:
                row, precision, weights_id = ppl_cell(raw, path, cell['context'], ladder['contract'], ladder['attention'])
            require(dict(row, report=str(path)) == cell and precision == ladder['precision'] and
                    weights_id == ladder['weights_id'], 'retained report differs from ladder')
        row = dict(context=left['context'])
        if dense['contract']['kind'] == 'speed':
            row.update(average_speedup=right['average_tok_s']/left['average_tok_s'],
                       tail_speedup=right['tail_tok_s']/left['tail_tok_s'])
        else:
            require(left['terrible_nll'] == right['terrible_nll'], 'severe-position thresholds differ')
            l_nll = sidecar(Path(left['report']).with_suffix('.nllf32'), 'f')
            r_nll = sidecar(Path(right['report']).with_suffix('.nllf32'), 'f')
            row.update(mean_nll_delta=right['mean_nll']-left['mean_nll'],
                       ppl_change_percent=100*(right['ppl']/left['ppl']-1),
                       max_abs_nll_delta=max(abs(a-b) for a, b in zip(l_nll, r_nll)),
                       new_severe_positions=sum(a < left['terrible_nll'] <= b for a, b in zip(l_nll, r_nll)))
        rows.append(row)
    return dict(dense=str(dense_path), candidate=str(candidate_path), kind=dense['contract']['kind'],
                attention=candidate['attention'], cells=rows,
                admission='measurement only; no automatic quality gate or production promotion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run_parser = sub.add_parser('run', help='serial execution or retained-report collection')
    run_parser.add_argument('--kind', choices=('speed', 'ppl'), required=True)
    for name in ('binary', 'weights', 'ids', 'out'):
        run_parser.add_argument('--' + name, type=Path, required=True)
    run_parser.add_argument('--attention', required=True, help='dense or exact compiled XAttention profile')
    run_parser.add_argument('--contexts', nargs='+', type=int, required=True)
    run_parser.add_argument('--chunk', type=int, default=2048)
    run_parser.add_argument('--max-context', type=int, default=131200)
    run_parser.add_argument('--spec', choices=('none', 'dflash'), default='none')
    run_parser.add_argument('--draft-tokens', type=int, default=0)
    run_parser.add_argument('--lm-head-draft', action='store_true')
    run_parser.add_argument('--warmup', type=int, default=0)
    run_parser.add_argument('--repetitions', type=int, default=1)
    run_parser.add_argument('--skip', default='half')
    run_parser.add_argument('--allow-cyclic-speed-corpus', action='store_true')
    run_parser.add_argument('--collect-only', action='store_true', help='never launch; validate existing reports/commands')
    compare_parser = sub.add_parser('compare')
    compare_parser.add_argument('--dense', type=Path, required=True)
    compare_parser.add_argument('--candidate', type=Path, required=True)
    compare_parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'run':
        run(args)
    else:
        require(not args.out.exists(), 'comparison output exists; choose a fresh path')
        write(args.out, compare(args.dense, args.candidate))


if __name__ == '__main__':
    main()
