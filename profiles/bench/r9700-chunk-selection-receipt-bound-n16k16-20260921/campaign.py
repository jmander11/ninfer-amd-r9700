#!/usr/bin/env python3
"""Run the current twelve-profile chunk campaign through existing evidence owners."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
PYTHON = Path('/home/battlefront/.local/bin/python3.11')
sys.path.insert(0, str(REPO))

from tools.bench.prefill_chunk_authority import durable_create_json
from tools.bench.run_ninfer_bench_matrix import require_auto_power_profile
from tools.bench.select_prefill_chunk import validate_screening_record, validate_selection_record
from tools.ppl.run import inspect_candidate_artifact

SUFFIX = 'receipt-bound-n16k16-20260921'
RUNNER = REPO / 'tools/bench/run_ninfer_bench_matrix.py'
SELECTOR = REPO / 'tools/bench/select_prefill_chunk.py'
SCREENING = REPO / f'profiles/bench/prefill-chunk-screening-{SUFFIX}.json'
SELECTION = REPO / f'profiles/bench/prefill-chunk-selection-{SUFFIX}.json'
RECEIPT = PACKAGE / 'inputs.json'
RECIPES = {
    'all-q4': 'qwen3.8-27b-r9700-q4g64-n16k16-eval',
    'mixed': 'qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval',
    'four-role': 'qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval',
}


def profiles():
    for recipe, artifact in RECIPES.items():
        for group in (16, 32):
            for attention in ('dense', 'xattention'):
                yield recipe, artifact, group, attention


def build_root(group, attention):
    return REPO / f'build-r9700-selection-{attention}-g{group}-20260921'


def matrix_root(profile, stage):
    recipe, _, group, attention = profile
    tag = 'dense' if attention == 'dense' else 'xattention-s16-tau900'
    return REPO / f'profiles/bench/prefill-chunk-{stage}-{recipe}-g{group}-{tag}-{SUFFIX}'


def command(profile, stage, chunks):
    recipe, artifact, group, attention = profile
    build = build_root(group, attention)
    result = [str(PYTHON), str(RUNNER), '--preset', 'prefill-chunk',
              '--bench', str(build / 'bench/ninfer_bench'),
              '--weights', str(REPO / 'out' / f'{artifact}.ninfer'),
              '--corpus', str(REPO / 'bench/fixtures/bench_corpus.ids'),
              '--output-dir', str(matrix_root(profile, stage)), '--device', '0',
              '--concurrency', '1', '--expected-kv-value-group', str(group),
              '--expected-q4-activation-bits', '8', '--expected-w8-activation-bits', '8',
              '--expected-fp8-qk-wmma', '1', '--expected-xattention-profile',
              'dense' if attention == 'dense' else 'b128-s16-tau900',
              '--prefill-prompt', '8192' if stage == 'screen' else '32768', '--no-build']
    for chunk in chunks:
        result.extend(('--prefill-chunk', str(chunk)))
    if recipe == 'four-role':
        result.extend(('--require-fp8-hybrid', '--hybrid-width-tool',
                       str(build / 'src/ninfer_r9700_runtime_planner_qual')))
    return result


def run(argv):
    subprocess.run(argv, cwd=REPO, check=True)


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def inspect_inputs():
    artifacts = []
    for artifact in RECIPES.values():
        artifacts.append(inspect_candidate_artifact(REPO / 'out' / f'{artifact}.ninfer'))
    paths = [Path(__file__), PACKAGE / 'commands.sh', RUNNER, SELECTOR,
             REPO / 'tools/bench/matrix_contract.py',
             REPO / 'tools/bench/prefill_chunk_authority.py',
             REPO / 'tools/ppl/run.py', REPO / 'tools/bench/prepare_matrix_cell.py',
             REPO / 'bench/fixtures/bench_corpus.ids',
             REPO / 'bench/fixtures/bench_corpus.manifest.json']
    for group in (16, 32):
        for attention in ('dense', 'xattention'):
            build = build_root(group, attention)
            paths.extend(build / name for name in (
                'bench/ninfer_bench', 'apps/ninfer-ppl',
                'src/ninfer_r9700_runtime_planner_qual', 'CMakeCache.txt',
                'compile_commands.json'))
    return {
        'artifact_type': 'ninfer_r9700_chunk_campaign_inputs', 'schema_version': 1,
        'python': str(PYTHON), 'artifacts': artifacts,
        'files': {str(path.relative_to(REPO)): digest(path) for path in paths},
    }


def preflight():
    inputs = inspect_inputs()
    require_auto_power_profile()
    if os.path.lexists(RECEIPT):
        if json.loads(RECEIPT.read_text()) != inputs:
            raise ValueError('campaign inputs differ from frozen receipt')
    return inputs


def verify_inputs():
    receipt = json.loads(RECEIPT.read_text())
    for relative, expected in receipt['files'].items():
        if digest(REPO / relative) != expected:
            raise ValueError(f'campaign input changed: {relative}')
    artifacts = [inspect_candidate_artifact(REPO / 'out' / f'{artifact}.ninfer')
                 for artifact in RECIPES.values()]
    if artifacts != receipt['artifacts']:
        raise ValueError('campaign artifact or conversion receipt changed')


def prepare(stage, chunks):
    for profile in profiles():
        run([str(PYTHON), str(REPO / 'tools/bench/prepare_matrix_cell.py'),
             '--output-dir', str(matrix_root(profile, stage)), '--',
             *command(profile, stage, chunks), '--prepare-only'])


def require_idle_gpu():
    from tools.bench.run_ninfer_bench_matrix import R9700_POWER_PROFILE
    require_auto_power_profile()
    used = int((R9700_POWER_PROFILE.parent / 'mem_info_vram_used').read_text())
    if used > 1024 ** 3:
        raise ValueError(f'R9700 already has {used} bytes resident; release the GPU before measurement')


def execute(stage, chunks):
    for profile in profiles():
        require_idle_gpu()
        run([*command(profile, stage, chunks), '--resume'])


def screen_arguments():
    return [part for profile in profiles()
            for part in ('--screen', str(matrix_root(profile, 'screen')))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'freeze', 'prepare', 'screens', 'finalists', 'select'))
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error(f'use {PYTHON}')
    os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:/opt/rocm/core-10.0/lib'
    if args.phase == 'preflight':
        preflight()
        print('preflight passed; no files published')
        return
    if args.phase == 'freeze':
        durable_create_json(RECEIPT, preflight())
        return
    verify_inputs()
    if args.phase == 'prepare':
        prepare('screen', (1024, 2048, 4096, 8192))
    elif args.phase == 'screens':
        execute('screen', (1024, 2048, 4096, 8192))
    elif args.phase == 'finalists':
        if not os.path.lexists(SCREENING):
            run([str(PYTHON), str(SELECTOR), *screen_arguments(),
                 '--out', str(SCREENING), '--create-only'])
        screening = validate_screening_record(SCREENING,
            [matrix_root(profile, 'screen') for profile in profiles()])
        chunks = screening['finalist_chunks']
        # A prepared or partially executed finalist resumes through the runner's provenance checks.
        for profile in profiles():
            if not os.path.lexists(matrix_root(profile, 'finalist')):
                run([*command(profile, 'finalist', chunks), '--prepare-only'])
        execute('finalist', chunks)
    else:
        finalists = [part for profile in profiles()
                     for part in ('--finalist', str(matrix_root(profile, 'finalist')))]
        run([str(PYTHON), str(SELECTOR), *screen_arguments(), *finalists,
             '--out', str(SELECTION), '--create-only'])
        print(json.dumps(validate_selection_record(SELECTION), indent=2))


if __name__ == '__main__':
    main()
