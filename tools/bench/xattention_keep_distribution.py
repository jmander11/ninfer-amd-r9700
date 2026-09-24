#!/usr/bin/env python3
"""Real-model C1 keep distributions; diagnostic runs are never timing evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.bench.prefill_chunk_authority import durable_create_json, validate_prefill_chunk_authority
from tools.ppl.quality_recovery_io import validate_authority_map
from tools.ppl import run as ppl
from tools.ppl.fp8_context_recovery import identity

SELECTION = REPO / 'profiles/bench/prefill-chunk-selection-panel-attention-20260921.json'
QUALITY = REPO / 'profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/quality-authorities-receipt-bound-n16k16.json'
CORPUS = REPO / 'bench/fixtures/bench_corpus.ids'
MACRO = '-DNINFER_R9700_XATTENTION_KEEP_DIAGNOSTIC=1'
SOURCES = [REPO / 'src/targets/qwen3_8_27b/impl' / name for name in (
    'r9700_full_attention.hip', 'xattention_keep_trace.h')]


def validate_rows(rows, prompt, group):
    if prompt not in (8192, 32768) or group not in (16, 32):
        raise ValueError('unsupported diagnostic workload')
    # Sixteen full-attention layers per chunk, one C1 prefill, no warmup/repeat.
    contexts = [context for context in range(2048, prompt + 1, 2048) for _ in range(16)]
    if len(rows) != len(contexts):
        raise ValueError('incomplete real-model dispatch coverage')
    histogram = Counter()
    fractions = []
    sparse_slots = 0
    for dispatch, (row, context) in enumerate(zip(rows, contexts)):
        expected = dict(schema_version=1, diagnostic_only=True, timing_eligible=False,
                        dispatch=dispatch, context=context, query_rows=2048,
                        query_heads=24, kv_value_group=group, query_block_rows=128,
                        page_tokens=64)
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError('keep trace workload/order/diagnostic identity differs')
        counts = row.get('keep_counts')
        if not isinstance(counts, list) or len(counts) != 24 * 16:
            raise ValueError('keep trace lacks every head/query-block slot')
        for slot, count in enumerate(counts):
            # A B128 query block shares its keep list; upper frontier includes
            # both B64 pages at the block end, not each individual query row.
            maximum = (context - 2048 + (slot % 16 + 1) * 128) // 64
            if type(count) is not int or not 0 < count <= maximum:
                raise ValueError('invalid sparse keep count')
            histogram[count] += 1
            fractions.append(count / maximum)
            sparse_slots += count < maximum
    if not sparse_slots:
        raise ValueError('real workload never exercised nontrivial sparse selection')
    return dict(dispatches=len(rows), slots=len(fractions), sparse_slots=sparse_slots,
                minimum_keep_fraction=min(fractions), maximum_keep_fraction=max(fractions),
                mean_keep_fraction=sum(fractions) / len(fractions),
                kept_page_count_histogram={str(k): v for k, v in sorted(histogram.items())},
                diagnostic_only=True, timing_eligible=False)


def cache(path):
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line:
            key, value = line.split('=', 1)
            key = key.split(':')[0]
            if (key.startswith('NINFER_R9700_') and not key.endswith('-STRINGS')) or key in (
                    'CMAKE_HIP_FLAGS', 'CMAKE_BUILD_TYPE', 'CMAKE_HIP_ARCHITECTURES'):
                values[key] = value
    return values


def preflight():
    authority, selection = validate_prefill_chunk_authority(SELECTION)
    quality = validate_authority_map(QUALITY)
    if (authority['selected_prefill_chunk'] != 2048 or
            quality['selected_prefill_chunk_authority'] !=
            {key: authority[key] for key in ('path', 'sha256')}):
        raise ValueError('diagnostic requires current shared selected chunk2048')
    cases = []
    for quality_name in ('ALL_Q4_XATTENTION_QUALITY', 'FOUR_ROLE_XATTENTION_QUALITY'):
        entry = quality['authorities'][quality_name]
        artifact = REPO / ('out/qwen3.8-27b-' + entry['artifact']['weights_id'] + '.ninfer')
        inspected = ppl.inspect_candidate_artifact(artifact)
        if inspected['sha256'] != entry['artifact']['sha256']:
            raise ValueError('diagnostic artifact differs from admitted quality')
        for group in (16, 32):
            if entry['eligibility_by_group'][str(group)] is not True:
                raise ValueError('requested sparse profile is quality-ineligible')
            build = REPO / f'build-r9700-xattention-keep-diagnostic-g{group}-20260921'
            old = REPO / f'build-r9700-selection-panel-xattention-g{group}-20260921'
            actual, expected = cache(build / 'CMakeCache.txt'), cache(old / 'CMakeCache.txt')
            expected['CMAKE_HIP_FLAGS'] = MACRO
            if actual != expected:
                raise ValueError('diagnostic build differs beyond capture compile flag')
            ready = subprocess.run(['cmake', '--build', str(build), '--target', 'ninfer_bench',
                                    '--', '-n'], check=True, capture_output=True, text=True)
            if ready.stdout.strip() != 'ninja: no work to do.':
                raise ValueError('diagnostic build is stale')
            executable = build / 'bench/ninfer_bench'
            for prompt in (8192, 32768):
                cases.append(dict(name=f'{quality_name.lower()}-g{group}-p{prompt}',
                                  group=group, prompt=prompt, artifact=str(artifact),
                                  executable=str(executable)))
    frozen = json.loads((REPO / 'profiles/bench/r9700-chunk-selection-panel-attention-20260921/inputs.json').read_text())
    if ppl.file_sha256(CORPUS) != frozen['files'][str(CORPUS.relative_to(REPO))]:
        raise ValueError('corpus differs from selected real-model workload')
    bindings = [identity(path) for path in [SELECTION, QUALITY, CORPUS, *SOURCES]]
    bindings += [identity(Path(path)) for path in sorted({c['executable'] for c in cases})]
    return cases, bindings


def command(case, output):
    return [case['executable'], '--weights', case['artifact'], '--corpus', str(CORPUS),
            '--device', '0', '--concurrency', '1', '-p', str(case['prompt']),
            '--prefill-chunk', '2048', '--max-ctx', '32768', '--draft-tokens', '0',
            '--no-device-graph', '-r', '1', '--warmup', '0', '--output', 'json',
            '--output-file', str(output / 'benchmark-diagnostic.json')]


def run(output):
    from tools.bench import run_ninfer_bench_matrix as bench
    cases, bindings = preflight()
    output.mkdir(exist_ok=False)
    durable_create_json(output / 'inputs.json', dict(bindings=bindings, cases=cases,
                                                   timing_eligible=False))
    summaries = []
    for case in cases:
        deadline = time.monotonic() + 10
        while True:
            bench.require_auto_power_profile()
            device = bench.R9700_POWER_PROFILE.parent
            if (int((device / 'mem_info_vram_used').read_text()) < 1024**3 and
                    int((device / 'gpu_busy_percent').read_text()) == 0):
                break
            if time.monotonic() >= deadline:
                raise ValueError('GPU busy; diagnostic outputs retained')
            time.sleep(.1)
        bench.require_hip_pci_device(0)
        root = output / case['name']
        root.mkdir()
        args = command(case, root)
        trace = root / 'keep-counts.jsonl'
        durable_create_json(root / 'command.json', dict(command=args,
                            environment={'NINFER_XATTENTION_KEEP_TRACE': str(trace)},
                            timing_eligible=False))
        with (root / 'stdout.txt').open('x') as stdout, (root / 'stderr.txt').open('x') as stderr:
            result = subprocess.run(args, cwd=REPO, env={**os.environ,
                                    'NINFER_XATTENTION_KEEP_TRACE': str(trace)},
                                    stdout=stdout, stderr=stderr)
        durable_create_json(root / 'exit.json', {'returncode': result.returncode})
        bench.require_auto_power_profile()
        if result.returncode:
            raise ValueError('diagnostic child failed; raw evidence retained')
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        summary = validate_rows(rows, case['prompt'], case['group'])
        summaries.append(dict(case=case, trace=identity(trace), **summary))
        durable_create_json(root / 'distribution.json', summary)
    if any(identity(Path(item['path'])) != item for item in bindings):
        raise ValueError('diagnostic inputs changed during execution')
    durable_create_json(output / 'result.json', dict(status='validated_real_model_keep_distributions',
                        timing_eligible=False, cases=summaries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'run'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'preflight':
        preflight()
        print('read-only keep-distribution preflight PASS')
    else:
        if args.output is None:
            parser.error('run requires a fresh --output directory')
        run(args.output.resolve())


if __name__ == '__main__':
    main()
