#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/ssdpool2nvme/local_llm/ninfer-amd-r9700')
PACKAGE = Path(__file__).resolve().parent
PLAN = PACKAGE / 'plan.json'
ATTEMPT = PACKAGE / 'attempt-1'
SELECTOR = 'NINFER_R9700_PROJECTED_RESIDUAL_T1_CANDIDATE'
FIELD = 'projected_residual_t1_candidate'
BUILDS = {'control': ROOT / 'build-r9700-projected-residual-whole-control-20260920',
          'candidate': ROOT / 'build-r9700-projected-residual-whole-candidate-20260920'}
AUTHORITY = ROOT / 'profiles/bench/r9700-three-route-production-confirmation-20260920/results/c1.json'
PCI = Path('/sys/bus/pci/devices/0000:13:00.0')
DEVICE_EXPECTED = {'ordinal': 0, 'name': 'AMD Radeon AI PRO R9700',
                   'architecture': 'gfx1201', 'wavefront': 32, 'pci_bus_id': PCI.name}
ORDER = ['control', 'candidate', 'candidate', 'control', 'control', 'candidate']
CACHE = {'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_HIP_ARCHITECTURES': 'gfx1201',
         'CMAKE_HIP_COMPILER': '/opt/rocm/llvm/bin/clang++',
         'CMAKE_EXPORT_COMPILE_COMMANDS': 'ON',
         'NINFER_BUILD_APPS': 'ON', 'NINFER_BUILD_BENCHMARKS': 'ON',
         'NINFER_BUILD_R9700_CORE_QUALIFIER': 'ON',
         'NINFER_R9700_KV_VALUE_GROUP': '16', 'NINFER_R9700_Q4_ACTIVATION_BITS': '8',
         'NINFER_R9700_W8_ACTIVATION_BITS': '8', 'NINFER_R9700_FP8_QK_WMMA': '1',
         'NINFER_R9700_XATTENTION_QUALIFICATION': 'OFF',
         'NINFER_R9700_XATTENTION_STRIDE': '16', 'NINFER_R9700_XATTENTION_TAU_PERMILLE': '1000',
         'NINFER_R9700_DFLASH_DOWN_SPLITK_FACTOR': '8'}
SOURCES = [ROOT / name for name in (
    'CMakeLists.txt', 'src/CMakeLists.txt', 'include/ninfer/ops/projected_residual.h',
    'src/ops/r9700/linear/linear_tensor_op.cpp', 'src/ops/r9700/linear/r9700_linear.h',
    'src/ops/r9700/linear/r9700_linear.hip', 'src/ops/r9700/linear/r9700_q4_activation_profile.h',
    'src/targets/qwen3_8_27b/impl/variant.h', 'src/targets/qwen3_8_27b/impl/variant.cpp',
    'bench/targets/qwen3_8_27b/ninfer_bench_support.cpp',
    'tools/r9700/target_variant_projected_residual_qual.cpp', 'tools/r9700/gdn_op_qual.hip')]
SOURCES += [PACKAGE / name for name in ('campaign.py', 'commands.sh', 'README.md')]


def fail(message):
    raise RuntimeError(message)


def load(path):
    value = json.loads(path.read_text(), parse_constant=lambda x: fail(f'nonfinite JSON: {x}'))
    if not isinstance(value, dict): fail(f'not an object: {path}')
    return value


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    if not path.is_file() or path.is_symlink(): fail(f'not a regular file: {path}')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def check(item):
    if identity(Path(item['path'])) != item: fail(f'identity changed: {item["path"]}')


def exclusive(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(fd, 'w') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')


def retained_closure(directory, root_relative=False):
    seen = set()
    for line in (directory / 'result.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = ROOT / name if root_relative else directory / name
        if path.parent != directory or path.name in seen or digest(path) != expected:
            fail(f'retained closure differs: {path}')
        seen.add(path.name)
    actual = {p.name for p in directory.iterdir() if p.is_file() and p.name != 'result.sha256'}
    if actual != seen: fail(f'incomplete retained closure: {directory}')


def qualify(directory):
    retained_closure(directory, root_relative=True)
    closing = load(directory / 'closure.json')
    if closing.get('status') != 'passed': fail('production public-Op qualification did not pass')
    report = load(directory / 'qualification.json')
    if (report.get('status') != 'qualified_for_whole_ab_only'
            or report.get('correctness', {}).get('production_symbol') != 'ninfer::ops::projected_residual_t1'
            or report['correctness'].get('captured_graph_replays_per_shape') != 3):
        fail('production public-Op qualification did not pass')
    check(closing['plan'])
    bound = load(Path(closing['plan']['path']))
    for item in bound['bound_inputs']: check(item)
    return {'attempt': str(directory), 'closure': identity(directory / 'closure.json'),
            'manifest': identity(directory / 'result.sha256'),
            'qualification': identity(directory / 'qualification.json'),
            'plan': closing['plan']}


def cache(build, role):
    values = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line and ':' in line.split('=', 1)[0]:
            key, value = line.split('=', 1)
            name, kind = key.split(':', 1)
            if kind != 'INTERNAL': values[name] = value
    for key, value in {**CACHE, SELECTOR: str(int(role == 'candidate'))}.items():
        if values.get(key) != value: fail(f'{role} cache differs: {key}')
    for key, value in values.items():
        if key.startswith('NINFER_R9700_') and key.endswith(('_CANDIDATE', '_QUALIFICATION')) and key != SELECTOR:
            if value not in ('0', 'OFF'): fail(f'other candidate enabled: {key}')
    return {key: value for key, value in values.items()
            if key.startswith(('NINFER_', 'CMAKE_HIP_', 'CMAKE_CXX_', 'CMAKE_EXE_LINKER_')) or key in CACHE}


def tokens(report):
    tests = report.get('tests', [])
    if len(tests) != 1 or tests[0].get('label') != 'whole-pp8192+tg256': fail('token workload differs')
    reps = tests[0].get('reps', [])
    lanes = reps[0].get('generated_token_ids_by_lane') if len(reps) == 1 else None
    if (not isinstance(lanes, list) or len(lanes) != 1 or not isinstance(lanes[0], list)
            or len(lanes[0]) != 257 or any(type(x) is not int or x < 0 for x in lanes[0])):
        fail('token geometry differs')
    return lanes


def telemetry(require_low=True):
    sample = {'pci_bus_id': PCI.name,
              'power': (PCI / 'power_dpm_force_performance_level').read_text().strip(),
              'used_bytes': int((PCI / 'mem_info_vram_used').read_text()),
              'total_bytes': int((PCI / 'mem_info_vram_total').read_text()),
              'vendor': (PCI / 'vendor').read_text().strip(), 'device': (PCI / 'device').read_text().strip()}
    if (sample['power'] != 'auto' or sample['vendor'] != '0x1002' or sample['device'] != '0x7551'
            or sample['total_bytes'] < 32000000000 or not 0 <= sample['used_bytes'] <= sample['total_bytes']):
        fail(f'device/power/VRAM differs: {sample}')
    sample['low_vram'] = (sample['used_bytes'] <= 1073741824 and
                          sample['used_bytes'] * 1000 <= sample['total_bytes'] * 50)
    if require_low and not sample['low_vram']: fail(f'VRAM occupied: {sample}')
    return sample


def drain():
    started = time.monotonic(); samples = []
    while True:
        sample = telemetry(require_low=False); elapsed = time.monotonic() - started
        samples.append({'elapsed_seconds': elapsed, **sample})
        if elapsed > 30: fail('VRAM did not drain in 30 seconds')
        if sample['low_vram']: return samples
        time.sleep(min(0.1, 30-elapsed))


def validate_device_identity(device, expected):
    if any(device.get(key) != value for key, value in expected.items()):
        fail('HIP device 0 does not match the bound R9700 PCI function')
    if any(type(device.get(key)) is not int or device[key] <= 0
           for key in ('runtime_version', 'driver_version')):
        fail('HIP device identity lacks valid runtime/driver versions')


def device_identity(plan, stem):
    helper = plan['device_identity_helper']
    check(helper['executable'])
    receipt = {'command': helper['command'], 'executable': helper['executable'],
               'cwd': str(ROOT), 'started_unix_ns': time.time_ns()}
    try:
        # Fail before HIP initialization unless the bound PCI function is in auto.
        receipt['before'] = telemetry()
        completed = subprocess.run(helper['command'], cwd=ROOT, capture_output=True, text=True)
        receipt.update(exit_code=completed.returncode, finished_unix_ns=time.time_ns())
        exclusive(ATTEMPT / f'{stem}-device.stdout', completed.stdout)
        exclusive(ATTEMPT / f'{stem}-device.stderr', completed.stderr)
        receipt['drain_samples'] = drain()
        if completed.returncode or completed.stderr: fail('HIP device identity query failed')
        device = json.loads(completed.stdout, parse_constant=lambda x: fail(f'nonfinite JSON: {x}'))
        validate_device_identity(device, helper['expected'])
        device['power_profile_path'] = str(PCI / 'power_dpm_force_performance_level')
        exclusive(ATTEMPT / f'{stem}-device.json', device)
        receipt['device_identity'] = identity(ATTEMPT / f'{stem}-device.json')
        check(helper['executable'])
        return receipt['device_identity']
    except BaseException as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        exclusive(ATTEMPT / f'{stem}-device.process.json', receipt)


def prepare(directory):
    if any(path.exists() or path.is_symlink() for path in (PLAN, ATTEMPT, *BUILDS.values())):
        fail('prepare requires fresh plan, attempt, and build directories')
    qualification = qualify(directory)
    sources = [identity(path) for path in SOURCES]
    retained_closure(AUTHORITY.parent)
    if load(AUTHORITY.parent / 'summary.json').get('status') != 'passed': fail('token authority unaccepted')
    builds = {}
    for role, build in BUILDS.items():
        subprocess.run(['cmake', '-S', str(ROOT), '-B', str(build), '-GNinja'] +
                       [f'-D{key}={value}' for key, value in CACHE.items()] +
                       [f'-D{SELECTOR}={int(role == "candidate")}'], cwd=ROOT, check=True)
        targets = ['ninfer_bench', 'ninfer_r9700_target_variant_projected_residual_qual']
        if role == 'control': targets.append('ninfer_r9700_gdn_qual')
        subprocess.run(['cmake', '--build', str(build), '-j2', '--target', *targets], cwd=ROOT, check=True)
        qualifier = build / 'src/ninfer_r9700_target_variant_projected_residual_qual'
        receipt = subprocess.run([str(qualifier)], cwd=ROOT, check=True, capture_output=True, text=True)
        builds[role] = {'cache': identity(build / 'CMakeCache.txt'), 'values': cache(build, role),
                        'executable': identity(build / 'bench/ninfer_bench'),
                        'route_qualifier': identity(qualifier), 'route_stdout': receipt.stdout}
    left, right = (dict(builds[role]['values']) for role in ('control', 'candidate'))
    left.pop(SELECTOR); right.pop(SELECTOR)
    if left != right: fail('build configurations differ beyond the selector')
    for item in sources: check(item)
    qualify(directory)
    device_helper = BUILDS['control'] / 'src/ninfer_r9700_gdn_qual'
    exclusive(PLAN, {'schema': 'ninfer.r9700.projected-residual-whole-ab.v1',
                    'status': 'prepared_awaiting_independent_review', 'sources': sources,
                    'builds': builds, 'production_qualification': qualification,
                    'device_identity_helper': {'executable': identity(device_helper),
                        'command': [str(device_helper), '--device-info'], 'expected': DEVICE_EXPECTED},
                    'artifact': identity(ROOT / 'out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer'),
                    'corpus': identity(ROOT / 'bench/fixtures/bench_corpus.ids'),
                    'token_authority': identity(AUTHORITY), 'expected_tokens': tokens(load(AUTHORITY)),
                    'order': ORDER, 'admission': {'every_ratio_below': 1.0,
                       'mean_plus_2se_below': 1.0, 'median_at_most': 0.99},
                    'workload': {'concurrency': 1, 'whole_pg': '8192,256', 'prefill_chunk': 4096,
                        'spec': 'none', 'draft_tokens': 0, 'device_graph': True, 'warmup': 1, 'reps': 1}})
    print('projected_residual_whole_prepare: PASS; independent review required before measurement')


def preflight():
    if ATTEMPT.exists() or ATTEMPT.is_symlink(): fail('attempt already exists')
    plan = load(PLAN)
    if plan.get('schema') != 'ninfer.r9700.projected-residual-whole-ab.v1' or plan.get('order') != ORDER:
        fail('plan contract differs')
    for item in plan['sources'] + [plan[k] for k in ('artifact', 'corpus', 'token_authority')]: check(item)
    if qualify(Path(plan['production_qualification']['attempt'])) != plan['production_qualification']:
        fail('qualification changed')
    for role, build in plan['builds'].items():
        for key in ('cache', 'executable', 'route_qualifier'): check(build[key])
        if cache(BUILDS[role], role) != build['values']: fail('build values changed')
    helper = plan['device_identity_helper']
    check(helper['executable'])
    if (helper['expected'] != DEVICE_EXPECTED or
            helper['command'] != [str(BUILDS['control'] / 'src/ninfer_r9700_gdn_qual'), '--device-info'] or
            helper['executable']['path'] != helper['command'][0]):
        fail('bound HIP identity helper differs')
    if tokens(load(AUTHORITY)) != plan['expected_tokens']: fail('authority changed')
    telemetry()
    print('projected_residual_whole_preflight: PASS')
    return plan


def command(plan, stem, role):
    return [plan['builds'][role]['executable']['path'], '--weights', plan['artifact']['path'],
            '--corpus', plan['corpus']['path'], '--device', '0', '--concurrency', '1', '--whole-pg',
            '8192,256', '--prefill-chunk', '4096', '--kv-capacity', 'workload', '--spec', 'mtp',
            '--draft-tokens', '0', '--retain-token-ids', '--output', 'json', '--output-file',
            str(ATTEMPT / f'{stem}.json'), '-r', '1', '--warmup', '1']


def validate_report(plan, stem, role):
    report = load(ATTEMPT / f'{stem}.json'); env = report.get('environment', {})
    if (report.get('schema_version') != 20 or report.get('artifact_type') != 'ninfer_bench_report'
            or report.get('tool') != 'ninfer_bench' or env.get('device_id') != 0
            or env.get('gpu_name') != 'AMD Radeon AI PRO R9700' or env.get('architecture_name') != 'gfx1201'
            or report.get('artifact') != {'path': plan['artifact']['path'], 'file_size_bytes': plan['artifact']['bytes']}):
        fail(f'provenance differs: {stem}')
    config = report.get('config', {})
    expected = {'concurrency': 1, 'prefill_chunk': 4096, 'kv_cache_format': 'fp8-k-int4-v',
                'kv_value_group': 16, 'q4_activation_bits': 8, 'w8_activation_bits': 8,
                'fp8_qk_wmma_enabled': True, 'xattention_qualification': False,
                'spec': 'none', 'draft_tokens': 0, 'use_device_graph': True, 'retain_token_ids': True,
                'decode_path': 'device_graph', 'repetitions': 1, 'warmup': 1,
                'decode_graph_prime': {'primed': True, 'output_tokens': 3}, FIELD: role == 'candidate'}
    for key, value in expected.items():
        if config.get(key) != value: fail(f'config differs: {stem}: {key}')
    if any(value is not False for key, value in config.items() if key.endswith('_candidate') and key != FIELD):
        fail('another candidate is active')
    if tokens(report) != plan['expected_tokens']: fail(f'exact public token parity failed: {stem}')
    test = report['tests'][0]; rep = test['reps'][0]
    if (test.get('kind') != 'whole' or test.get('n_prompt') != 8192 or test.get('n_gen') != 256
            or test.get('requested_output_tokens') != 257 or rep.get('generated_output_tokens') != 257
            or rep.get('decode_output_tokens') != 256 or rep.get('decode_engine_tokens') != 256):
        fail(f'geometry differs: {stem}')
    seconds = rep.get('timings', {}).get('decode_seconds')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0: fail('invalid decode time')
    return {'role': role, 'stem': stem, 'decode_seconds': seconds, 'decode_output_tok_s': 256/seconds}


def summarize(records):
    pairs = []
    for index in range(3):
        pair = records[index*2:index*2+2]
        control = next(x for x in pair if x['role'] == 'control')
        candidate = next(x for x in pair if x['role'] == 'candidate')
        pairs.append({'order': [x['role'] for x in pair], 'control_seconds': control['decode_seconds'],
                      'candidate_seconds': candidate['decode_seconds'],
                      'candidate_over_control': candidate['decode_seconds']/control['decode_seconds']})
    ratios = [pair['candidate_over_control'] for pair in pairs]
    upper = statistics.mean(ratios) + 2*statistics.stdev(ratios)/math.sqrt(3)
    passed = all(x < 1 for x in ratios) and upper < 1 and statistics.median(ratios) <= 0.99
    return {'status': 'passed' if passed else 'performance_gate_failed',
            'production_routing_authorized': False, 'promotion_review_authorized': passed,
            'exact_public_token_parity': True, 'records': records, 'pairs': pairs,
            'paired_ratio_mean': statistics.mean(ratios), 'paired_ratio_upper_2se': upper,
            'paired_ratio_median': statistics.median(ratios)}


def measure():
    plan = preflight(); ATTEMPT.mkdir()
    try:
        exclusive(ATTEMPT / 'plan-identity.json', identity(PLAN))
        records = []
        for index, role in enumerate(ORDER, 1):
            stem = f'run-{index}-{role}'
            receipt = {'role': role, 'command': command(plan, stem, role), 'started_unix_ns': time.time_ns()}
            try:
                check(plan['builds'][role]['executable'])
                receipt['device_identity'] = device_identity(plan, stem)
                receipt['before'] = telemetry()
                completed = subprocess.run(receipt['command'], cwd=ROOT, capture_output=True, text=True)
                receipt.update(exit_code=completed.returncode, finished_unix_ns=time.time_ns())
                exclusive(ATTEMPT / f'{stem}.stdout', completed.stdout)
                exclusive(ATTEMPT / f'{stem}.stderr', completed.stderr)
                receipt['drain_samples'] = drain()
                if completed.returncode: fail(f'benchmark exited {completed.returncode}')
                check(plan['builds'][role]['executable'])
                records.append(validate_report(plan, stem, role))
            except BaseException as error:
                receipt['error'] = f'{type(error).__name__}: {error}'
                raise
            finally:
                exclusive(ATTEMPT / f'{stem}.process.json', receipt)
        summary = summarize(records)
        exclusive(ATTEMPT / 'summary.json', summary)
        if summary['status'] != 'passed': fail('whole C1 performance gate failed')
        print('projected_residual_whole_ab: PASS')
    except BaseException as error:
        exclusive(ATTEMPT / 'failure.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        files = sorted(path for path in ATTEMPT.iterdir() if path.is_file())
        exclusive(ATTEMPT / 'result.sha256', ''.join(f'{digest(path)}  {path.name}\n' for path in files))


if __name__ == '__main__':
    if Path.cwd() != ROOT: fail(f'must run from {ROOT}')
    if sys.version_info[:2] != (3, 11): fail('use Python 3.11')
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'preflight', 'measure'): group.add_argument('--'+name, action='store_true')
    parser.add_argument('--qualification-attempt', type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.qualification_attempt is None or not args.qualification_attempt.is_absolute():
            fail('prepare requires an explicit absolute --qualification-attempt')
        prepare(args.qualification_attempt)
    elif args.qualification_attempt is not None: fail('--qualification-attempt belongs only to --prepare')
    elif args.preflight: preflight()
    else: measure()
