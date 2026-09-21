#!/usr/bin/env python3
"""Create-only production confirmation; never builds or refreshes its inputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
PLAN = PACKAGE / 'plan.json'
ATTEMPT = PACKAGE / 'attempt-1'
PRIOR = ROOT / 'profiles/bench/r9700-projected-residual-t1-whole-ab-20260920'
FAILED = ROOT / 'profiles/bench/r9700-projected-residual-t1-production-confirmation-20260920'
spec = importlib.util.spec_from_file_location('retained_whole_ab', PRIOR / 'campaign.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
common.ATTEMPT = ATTEMPT
fail, load, identity, check = common.fail, common.load, common.identity, common.check
exclusive, digest = common.exclusive, common.digest
SCHEMA = 'ninfer.r9700.projected-residual-production-confirmation.v1'
ADMISSION = {'every_seconds_over_candidate_median_at_most': 1.01,
             'median_seconds_over_control_median_at_most': 0.99}
# Exporting compile commands was an A/B preparation choice, not a production
# execution contract. The fresh qualification binds this build's exact cache.
CACHE = {key: value for key, value in common.CACHE.items()
         if key != 'CMAKE_EXPORT_COMPILE_COMMANDS'}


def cache(build):
    values = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line and ':' in line.split('=', 1)[0]:
            key, value = line.split('=', 1)
            name, kind = key.split(':', 1)
            if name == common.SELECTOR: fail('retired selector remains in cache')
            if kind != 'INTERNAL': values[name] = value
    for key, value in CACHE.items():
        if values.get(key) != value: fail(f'production cache differs: {key}')
    for key, value in values.items():
        if key.startswith('NINFER_R9700_') and key.endswith(('_CANDIDATE', '_QUALIFICATION')):
            if value not in ('0', 'OFF'): fail(f'another candidate is active: {key}')
    return {key: value for key, value in values.items()
            if key.startswith(('NINFER_', 'CMAKE_HIP_', 'CMAKE_CXX_', 'CMAKE_EXE_LINKER_')) or key in CACHE}


def prior_evidence():
    directory = PRIOR / 'attempt-1'
    common.retained_closure(directory)
    check(load(directory / 'plan-identity.json'))
    old_plan = load(PRIOR / 'plan.json')
    summary = load(directory / 'summary.json')
    if (summary.get('status') != 'passed' or not summary.get('promotion_review_authorized')
            or not summary.get('exact_public_token_parity')):
        fail('retained whole A/B did not pass')
    records = []
    for index, role in enumerate(common.ORDER, 1):
        stem = f'run-{index}-{role}'
        report = load(directory / f'{stem}.json')
        if common.tokens(report) != common.tokens(load(common.AUTHORITY)):
            fail('retained A/B token mismatch')
        seconds = report['tests'][0]['reps'][0]['timings']['decode_seconds']
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            fail('invalid retained decode time')
        records.append({'role': role, 'stem': stem, 'decode_seconds': seconds,
                        'decode_output_tok_s': 256 / seconds})
    if common.summarize(records) != summary: fail('retained whole summary differs')
    direct = old_plan['production_qualification']
    direct_dir = Path(direct['attempt'])
    common.retained_closure(direct_dir, root_relative=True)
    for key in ('closure', 'manifest', 'qualification', 'plan'): check(direct[key])
    closure = load(direct_dir / 'closure.json')
    report = load(direct_dir / 'qualification.json')
    if (closure.get('status') != 'passed' or closure.get('plan') != direct['plan']
            or report.get('status') != 'qualified_for_whole_ab_only'):
        fail('retained production qualification did not pass')
    return {'plan': identity(PRIOR / 'plan.json'), 'summary': identity(directory / 'summary.json'),
            'manifest': identity(directory / 'result.sha256'), 'direct_qualification': direct,
            'candidate_median_seconds': statistics.median(r['decode_seconds'] for r in records if r['role'] == 'candidate'),
            'control_median_seconds': statistics.median(r['decode_seconds'] for r in records if r['role'] == 'control')}


def qualified_build(directory, build):
    common.retained_closure(directory, root_relative=True)
    closing = load(directory / 'closure.json')
    report = load(directory / 'qualification.json')
    if (closing.get('status') != 'passed'
            or report.get('status') != 'qualified_for_selector_free_production_confirmation'
            or report.get('decision') != 'prepare_selector_free_whole_c1_confirmation'
            or report.get('production_routing_authorized') is not False
            or report.get('correctness', {}).get('production_symbol') != 'ninfer::ops::projected_residual_t1'
            or report['correctness'].get('captured_graph_replays_per_shape') != 3):
        fail('fresh selector-free public-Op qualification did not pass')
    check(closing['plan'])
    qualification = {'attempt': str(directory), 'closure': identity(directory / 'closure.json'),
                     'manifest': identity(directory / 'result.sha256'),
                     'qualification': identity(directory / 'qualification.json'),
                     'plan': closing['plan']}
    bound = load(Path(qualification['plan']['path']))
    for item in bound['bound_inputs'] + bound['production_build']: check(item)
    inputs = {item['path']: item for item in bound['production_build']}
    required = [build / 'CMakeCache.txt', build / 'src/libninfer_r9700_core.a',
                build / 'src/ninfer_r9700_a8q4_projected_residual_t1_qual',
                build / 'src/ninfer_r9700_target_variant_projected_residual_qual']
    for path in required:
        if inputs.get(str(path)) != identity(path): fail(f'fresh qualification does not bind build input: {path}')
    return qualification


def failed_evidence():
    directory = FAILED / 'attempt-1'
    common.retained_closure(directory)
    check(load(directory / 'plan-identity.json'))
    failure = load(directory / 'failure.json')
    if failure != {'status': 'failed', 'error': 'RuntimeError: HIP device identity query failed'}:
        fail('original failed attempt differs')
    old_plan = load(FAILED / 'plan.json')
    for item in old_plan['sources']:
        if Path(item['path']).parent == FAILED: check(item)
    return {'plan': identity(FAILED / 'plan.json'),
            'manifest': identity(directory / 'result.sha256'),
            'failure': identity(directory / 'failure.json'),
            'reason': 'stale helper lacked source-supported --device-info; explicitly rebuilt helper'}


def query_helper(helper):
    # Read-only device discovery is allowed during preflight. Never create a
    # result directory, alter power, or run a numerical/benchmark workload here.
    check(helper['executable'])
    before = common.telemetry()
    completed = subprocess.run(helper['command'], cwd=ROOT, capture_output=True, text=True)
    samples = common.drain()
    check(helper['executable'])
    if completed.returncode or completed.stderr: fail('preflight HIP device identity query failed')
    device = json.loads(completed.stdout, parse_constant=lambda x: fail(f'nonfinite JSON: {x}'))
    common.validate_device_identity(device, helper['expected'])
    return {'command': helper['command'], 'executable': helper['executable'],
            'exit_code': completed.returncode, 'device': device,
            'before': before, 'drain_samples': samples}


def prepare(build, directory):
    if any(path.exists() or path.is_symlink() for path in (PLAN, ATTEMPT)):
        fail('prepare requires absent plan and attempt')
    qualification = qualified_build(directory, build)
    prior = prior_evidence()
    failed = failed_evidence()
    common.retained_closure(common.AUTHORITY.parent)
    if load(common.AUTHORITY.parent / 'summary.json').get('status') != 'passed':
        fail('retained token authority did not pass')
    sources = [identity(p) for p in common.SOURCES if p.parent != PRIOR]
    sources += [identity(ROOT / 'tools/r9700/a8q4_projected_residual_t1_qual.hip')]
    sources += [identity(PACKAGE / name) for name in ('campaign.py', 'commands.sh', 'README.md')]
    sources += [identity(PRIOR / 'campaign.py')]
    helper = build / 'src/ninfer_r9700_gdn_qual'
    helper_contract = {'executable': identity(helper), 'command': [str(helper), '--device-info'],
                       'expected': common.DEVICE_EXPECTED}
    helper_receipt = query_helper(helper_contract)
    plan = {'schema': SCHEMA, 'status': 'prepared_awaiting_independent_review',
            'sources': sources, 'production_qualification': qualification, 'retained_admission': prior,
            'failed_confirmation': failed, 'preparation_device_identity': helper_receipt,
            'build': {'directory': str(build), 'values': cache(build),
                      'cache': identity(build / 'CMakeCache.txt'),
                      'archive': identity(build / 'src/libninfer_r9700_core.a'),
                      'executable': identity(build / 'bench/ninfer_bench'),
                      'route_qualifier': identity(build / 'src/ninfer_r9700_target_variant_projected_residual_qual')},
            'device_identity_helper': helper_contract,
            'artifact': identity(ROOT / 'out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer'),
            'corpus': identity(ROOT / 'bench/fixtures/bench_corpus.ids'),
            'token_authority': identity(common.AUTHORITY), 'expected_tokens': common.tokens(load(common.AUTHORITY)),
            'runs': 3, 'admission': ADMISSION,
            'workload': {'concurrency': 1, 'whole_pg': '8192,256', 'prefill_chunk': 4096,
                         'spec': 'none', 'draft_tokens': 0, 'device_graph': True, 'warmup': 1, 'reps': 1}}
    exclusive(PLAN, plan)
    print('projected_residual_production_prepare: PASS; review before measurement')


def preflight():
    if ATTEMPT.exists() or ATTEMPT.is_symlink(): fail('attempt already exists')
    plan = load(PLAN)
    if plan.get('schema') != SCHEMA or plan.get('runs') != 3 or plan.get('admission') != ADMISSION:
        fail('plan contract differs')
    for item in plan['sources'] + [plan[k] for k in ('artifact', 'corpus', 'token_authority')]: check(item)
    build = Path(plan['build']['directory'])
    for key in ('cache', 'archive', 'executable', 'route_qualifier'): check(plan['build'][key])
    if cache(build) != plan['build']['values']: fail('production build configuration changed')
    if qualified_build(Path(plan['production_qualification']['attempt']), build) != plan['production_qualification']:
        fail('fresh qualification changed')
    if prior_evidence() != plan['retained_admission']: fail('retained admission changed')
    if failed_evidence() != plan['failed_confirmation']: fail('failed attempt changed')
    helper = plan['device_identity_helper']
    check(helper['executable'])
    if (helper['expected'] != common.DEVICE_EXPECTED or helper['command'] != [str(build / 'src/ninfer_r9700_gdn_qual'), '--device-info']
            or helper['executable']['path'] != helper['command'][0]): fail('HIP identity helper differs')
    if common.tokens(load(common.AUTHORITY)) != plan['expected_tokens']: fail('token authority differs')
    query_helper(helper)
    print('projected_residual_production_preflight: PASS')
    return plan


def command(plan, stem):
    return [plan['build']['executable']['path'], '--weights', plan['artifact']['path'],
            '--corpus', plan['corpus']['path'], '--device', '0', '--concurrency', '1', '--whole-pg',
            '8192,256', '--prefill-chunk', '4096', '--kv-capacity', 'workload', '--spec', 'mtp',
            '--draft-tokens', '0', '--retain-token-ids', '--output', 'json', '--output-file',
            str(ATTEMPT / f'{stem}.json'), '-r', '1', '--warmup', '1']


def validate_report(plan, stem):
    report = load(ATTEMPT / f'{stem}.json')
    env = report.get('environment', {})
    if (report.get('schema_version') != 20 or report.get('artifact_type') != 'ninfer_bench_report'
            or report.get('tool') != 'ninfer_bench' or env.get('device_id') != 0
            or env.get('gpu_name') != 'AMD Radeon AI PRO R9700' or env.get('architecture_name') != 'gfx1201'
            or report.get('artifact') != {'path': plan['artifact']['path'], 'file_size_bytes': plan['artifact']['bytes']}):
        fail(f'provenance differs: {stem}')
    config = report.get('config', {})
    if common.FIELD in config: fail('retired selector remains in benchmark report')
    expected = {'concurrency': 1, 'prefill_chunk': 4096, 'kv_cache_format': 'fp8-k-int4-v',
                'kv_value_group': 16, 'q4_activation_bits': 8, 'w8_activation_bits': 8,
                'fp8_qk_wmma_enabled': True, 'xattention_qualification': False,
                'spec': 'none', 'draft_tokens': 0, 'use_device_graph': True, 'retain_token_ids': True,
                'decode_path': 'device_graph', 'repetitions': 1, 'warmup': 1,
                'decode_graph_prime': {'primed': True, 'output_tokens': 3}}
    for key, value in expected.items():
        if config.get(key) != value: fail(f'config differs: {key}')
    if any(value is not False for key, value in config.items() if key.endswith('_candidate')):
        fail('another candidate is active')
    if common.tokens(report) != plan['expected_tokens']: fail('exact token mismatch')
    test = report['tests'][0]; rep = test['reps'][0]
    if (test.get('kind') != 'whole' or test.get('n_prompt') != 8192 or test.get('n_gen') != 256
            or test.get('requested_output_tokens') != 257 or rep.get('generated_output_tokens') != 257
            or rep.get('decode_output_tokens') != 256 or rep.get('decode_engine_tokens') != 256):
        fail('workload geometry differs')
    seconds = rep.get('timings', {}).get('decode_seconds')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0: fail('invalid decode time')
    return {'stem': stem, 'decode_seconds': seconds, 'decode_output_tok_s': 256 / seconds}


def measure():
    plan = preflight(); ATTEMPT.mkdir()
    try:
        exclusive(ATTEMPT / 'plan-identity.json', identity(PLAN))
        records = []
        for index in range(1, 4):
            stem = f'run-{index}-production'
            receipt = {'command': command(plan, stem), 'started_unix_ns': time.time_ns()}
            try:
                check(plan['build']['executable'])
                receipt['device_identity'] = common.device_identity(plan, stem)
                receipt['before'] = common.telemetry()
                completed = subprocess.run(receipt['command'], cwd=ROOT, capture_output=True, text=True)
                receipt.update(exit_code=completed.returncode, finished_unix_ns=time.time_ns())
                exclusive(ATTEMPT / f'{stem}.stdout', completed.stdout)
                exclusive(ATTEMPT / f'{stem}.stderr', completed.stderr)
                receipt['drain_samples'] = common.drain()
                if completed.returncode: fail(f'benchmark exited {completed.returncode}')
                check(plan['build']['executable'])
                records.append(validate_report(plan, stem))
            except BaseException as error:
                receipt['error'] = f'{type(error).__name__}: {error}'
                raise
            finally:
                exclusive(ATTEMPT / f'{stem}.process.json', receipt)
        reference = plan['retained_admission']
        ratios = [r['decode_seconds'] / reference['candidate_median_seconds'] for r in records]
        median_control_ratio = statistics.median(r['decode_seconds'] for r in records) / reference['control_median_seconds']
        passed = all(r <= ADMISSION['every_seconds_over_candidate_median_at_most'] for r in ratios) and median_control_ratio <= ADMISSION['median_seconds_over_control_median_at_most']
        exclusive(ATTEMPT / 'summary.json', {'status': 'passed' if passed else 'performance_gate_failed',
                  'selector_free_production_confirmed': passed, 'exact_public_token_parity': True,
                  'records': records, 'seconds_over_admitted_candidate_median': ratios,
                  'median_seconds_over_retained_control': median_control_ratio})
        if not passed: fail('production reproduction gate failed')
        print('projected_residual_production_confirmation: PASS')
    except BaseException as error:
        exclusive(ATTEMPT / 'failure.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        files = sorted(path for path in ATTEMPT.iterdir() if path.is_file())
        exclusive(ATTEMPT / 'result.sha256', ''.join(f'{digest(path)}  {path.name}\n' for path in files))


if __name__ == '__main__':
    if Path.cwd() != ROOT or sys.version_info[:2] != (3, 11): fail('run from repository root with Python 3.11')
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'preflight', 'measure'): group.add_argument('--' + name, action='store_true')
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--qualification-attempt', type=Path)
    args = parser.parse_args()
    if args.prepare:
        if any(p is None or not p.is_absolute() for p in (args.build_dir, args.qualification_attempt)):
            fail('prepare requires explicit absolute --build-dir and --qualification-attempt')
        prepare(args.build_dir, args.qualification_attempt)
    elif args.build_dir is not None or args.qualification_attempt is not None:
        fail('build and qualification arguments belong only to prepare')
    elif args.preflight: preflight()
    else: measure()
