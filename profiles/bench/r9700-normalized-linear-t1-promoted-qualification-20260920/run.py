#!/usr/bin/env python3
"""Create-only selector-free normalized-linear production qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
BUILD = ROOT / 'build-r9700'
TARGET = 'ninfer_r9700_a8q4_normalized_linear_t1_qual'
BINARY = BUILD / 'src' / TARGET
ARCHIVE = BUILD / 'src/libninfer_r9700_core.a'
PLAN = PACKAGE / 'plan.json'
ATTEMPT = PACKAGE / 'attempt-1'
LINEAR = ROOT / 'src/ops/r9700/linear/r9700_linear.hip'
CHECKER = ROOT / 'tools/r9700/check_a8q4_normalized_linear_t1_static.py'
PCI = Path('/sys/bus/pci/devices/0000:13:00.0')
POWER = PCI / 'power_dpm_force_performance_level'
BUILD_INPUTS = [BUILD / 'CMakeCache.txt', BUILD / 'compile_commands.json', ARCHIVE, BINARY]
SOURCE_INPUTS = [ROOT / name for name in (
    'CMakeLists.txt', 'src/CMakeLists.txt',
    'tools/r9700/a8q4_normalized_linear_t1_qual.hip',
    'tools/r9700/a8q4_shape_sweep_qual.hip',
    'tools/r9700/check_a8q4_normalized_linear_t1_static.py',
    'tools/bench/extract_embedded_code_object.py',
    'include/ninfer/ops/normalized_linear.h', 'include/ninfer/ops/linear.h',
    'include/ninfer/ops/rmsnorm.h', 'src/ops/r9700/linear/r9700_linear.hip',
    'src/ops/r9700/linear/r9700_linear.h',
    'src/ops/r9700/linear/r9700_q4_activation_profile.h',
    'src/ops/r9700/linear/linear_tensor_op.cpp',
    'src/ops/r9700/eager/eager_ops.hip', 'src/ops/r9700/eager/eager_ops.h',
    'src/core/device.hip', 'src/core/arena.hip', 'src/core/dtype.cpp',
    'src/core/tensor.cpp')]
AUTHORITY_INPUTS = [ROOT / 'profiles/bench' / package / 'attempt-1' / name
                    for package in (
                        'r9700-normalized-linear-t1-qualification-20260920',
                        'r9700-normalized-linear-t1-whole-ab-20260920')
                    for name in ('summary.json', 'closure.json', 'result.sha256')]
SCRIPT_INPUTS = [PACKAGE / name for name in ('commands.sh', 'run.py', 'analyze.py', 'README.md')]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    require(path.is_file() and not path.is_symlink(), f'not a regular file: {path}')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def exclusive(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                             0o644), 'w', encoding='utf-8')


def write_json(path, value):
    with exclusive(path) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def reject_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def read_json(path):
    return json.loads(path.read_text(), parse_constant=reject_constant)


def environment():
    require(Path.cwd() == ROOT and sys.version_info[:2] == (3, 11),
            f'run from {ROOT} with Python 3.11')
    forbidden = {key: value for key, value in os.environ.items() if value and
                 (key.startswith(('ROCP_', 'ROCPROF', 'HSA_TOOLS', 'ROCTRACER')) or
                  key in ('LD_PRELOAD', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
                          'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'))}
    require(not forbidden, f'profiler/device interception environment present: {sorted(forbidden)}')
    require((PCI / 'vendor').read_text().strip() == '0x1002' and
            (PCI / 'device').read_text().strip() == '0x7551', 'wrong physical PCI identity')
    require(POWER.read_text().strip() == 'auto', 'R9700 power must be auto')


def build_profile():
    values = {}
    for line in (BUILD / 'CMakeCache.txt').read_text().splitlines():
        if line and not line.startswith(('#', '//')) and ':' in line and '=' in line:
            key, value = line.split('=', 1)
            values[key.split(':', 1)[0]] = value
    expected = {'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_HIP_ARCHITECTURES': 'gfx1201',
                'NINFER_R9700_Q4_ACTIVATION_BITS': '8'}
    for key, value in expected.items():
        require(values.get(key) == value, f'wrong build profile: {key}')
    require('NINFER_R9700_NORMALIZED_LINEAR_T1_CANDIDATE' not in values,
            'obsolete normalized-linear selector remains in production cache')
    return expected


def assembly_command(directory):
    entries = [entry for entry in read_json(BUILD / 'compile_commands.json')
               if Path(entry['file']) == LINEAR]
    require(len(entries) == 1, 'ambiguous core linear compile command')
    entry = entries[0]
    require(Path(entry['directory']) == BUILD, 'unexpected compile working directory')
    argv = entry.get('arguments') or shlex.split(entry['command'])
    for flag in ('--offload-arch=gfx1201', '-DNINFER_R9700_Q4_ACTIVATION_BITS=8'):
        require(flag in argv, f'compile profile missing {flag}')
    require(not any('NINFER_R9700_NORMALIZED_LINEAR_T1_CANDIDATE' in arg for arg in argv),
            'obsolete normalized-linear selector remains in compile command')
    require(argv.count('-o') == 1 and argv.count('-c') == 1, 'unexpected compile command')
    output = argv.index('-o')
    argv = argv[:output] + argv[output + 2:]
    argv.remove('-c')
    return [*argv, '--offload-device-only', '-S', '-o', str(directory / 'qual.s')]


def link_command():
    result = subprocess.run(['ninja', '-C', str(BUILD), '-t', 'commands', f'src/{TARGET}'],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if f'-o src/{TARGET} ' in line]
    require(len(lines) == 1 and 'src/libninfer_r9700_core.a' in lines[0],
            'qualifier does not link the production core archive')
    return lines[0]


def current_plan():
    environment()
    relative = PACKAGE.relative_to(ROOT)
    for path in AUTHORITY_INPUTS:
        if path.name in ('summary.json', 'closure.json'):
            require(read_json(path)['status'] == 'passed', f'prior authority failed: {path}')
    whole = read_json(AUTHORITY_INPUTS[3])
    require(whole['promotion_review_authorized'] is True and
            whole['exact_public_token_parity'] is True, 'prior whole gate does not admit promotion')
    return {
        'schema': 'ninfer.r9700.normalized-linear-t1-promoted-plan.v1',
        'status': 'prepared_for_independent_review', 'production_routing_authorized': False,
        'claim': 'Requalify selector-free production normalized_linear against public RMSNorm+Linear at 64 MLP boundaries.',
        'bound_inputs': [identity(path) for path in SOURCE_INPUTS + SCRIPT_INPUTS],
        'prior_passing_authority': [identity(path) for path in AUTHORITY_INPUTS],
        'production_build': [identity(path) for path in BUILD_INPUTS],
        'build_profile': build_profile(), 'link_owner': 'ninfer_r9700_core',
        'link_command': link_command(),
        'source_assembly_command': assembly_command(ATTEMPT),
        'static_command': [sys.executable, str(CHECKER), str(ATTEMPT / 'qual.s'),
                           '--binary', str(BINARY), '--embedded-dir', str(ATTEMPT / 'embedded')],
        'qualification_command': [str(BINARY), '--out-json', str(ATTEMPT / 'qualification.json')],
        'analyzer_command': [sys.executable, str(PACKAGE / 'analyze.py'),
                             str(ATTEMPT / 'qualification.json')],
        'workload': {'device': 0, 'name': 'AMD Radeon AI PRO R9700', 'architecture': 'gfx1201',
                     'wave_size': 32, 'pci': '0000:13:00.0', 'vendor': '0x1002', 'device_id': '0x7551',
                     'power_path': str(POWER), 'power': 'auto', 'tokens': 1,
                     'rows': 34816, 'columns': 5120, 'copies': 3, 'paired_trials': 24,
                     'scrub_bytes': 83886080, 'calls_per_token': 64},
        'numerical_contract': {'oracle': 'FP64 RMSNorm -> explicit BF16 -> exact A8G64 -> FP64 signed-Q4 dot',
                               'criterion': 'normalized_linear_a8q4_fp64_rel_l2_1e-2_bf16_steps_2_v1',
                               'codec_exact': True, 'maximum_bf16_steps': 2, 'maximum_relative_l2': 0.01,
                               'graph_poison_stale_finite': True, 't2_independent_rows': True},
        'timing_contract': {'boundary': 'complete public-Op graph replay', 'every_allocation_wins': True,
                           'paired_ratio_mean_plus_2se_below': 1.0, 'minimum_saved_ms_per_token': 0.2,
                           'formula': '64*(median(control_ms)-median(candidate_ms))'},
        'static_contract': {'two_embedded_symbols': ['a8g64_normalized_prepare_t1_kernel',
                                                    'a8q4g64_linear_decode_dot8_t1_kernel'],
                            'wave_size': 32, 'scratch_spills': 0, 'prep_lds_allowed': True,
                            'consumer_native_iu4': True, 'consumer_vgpr_max': 32,
                            'consumer_sgpr_max': 64, 'consumer_lds': 0},
        'attempt': str(ATTEMPT), 'create_only': True,
        'exact_invocation': {mode: f'bash {relative}/commands.sh --{mode}'
                             for mode in ('prepare', 'preflight', 'measure')},
        'passing_authorizes': 'selector-free whole C1 production confirmation preparation only'}


def verify(fresh=True):
    plan = read_json(PLAN)
    require('UNBOUND' not in PLAN.read_text(), 'unbound plan')
    require(plan == current_plan(), 'bound source/build/profile/package changed after preparation')
    if fresh:
        require(not os.path.lexists(ATTEMPT), 'immutable attempt already exists')
    return plan


def ready_build():
    result = subprocess.run(['ninja', '-C', str(BUILD), '-n', TARGET],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    require(result.stdout.strip().endswith('ninja: no work to do.'),
            'production target is stale; build before preparing a fresh package')


def prepare():
    require(not os.path.lexists(PLAN) and not os.path.lexists(ATTEMPT), 'plan/attempt must be fresh')
    plan = current_plan()
    ready_build()
    write_json(PLAN, plan)
    print('normalized_linear_prepare: PASS; independent package review still required')


def preflight():
    verify()
    ready_build()
    # Compiler, disassembler and symbol reader only. Never invoke the HIP executable.
    with tempfile.TemporaryDirectory(prefix='normalized-linear-preflight-') as temporary:
        directory = Path(temporary)
        subprocess.run(assembly_command(directory), cwd=BUILD, check=True)
        subprocess.run([sys.executable, str(CHECKER), str(directory / 'qual.s'),
                        '--binary', str(BINARY), '--embedded-dir', str(directory / 'embedded')],
                       cwd=ROOT, check=True)
    verify()
    print('normalized_linear_preflight: PASS; no HIP execution; retained attempt absent')


def retained_step(name, argv, cwd=ROOT):
    stdout, stderr = ATTEMPT / f'{name}.stdout', ATTEMPT / f'{name}.stderr'
    record = {'argv': argv, 'cwd': str(cwd), 'started_unix_ns': time.time_ns(),
              'exit_code': None, 'error': None}
    try:
        with exclusive(stdout) as out, exclusive(stderr) as err:
            result = subprocess.run(argv, cwd=cwd, stdout=out, stderr=err, check=False)
        record['exit_code'] = result.returncode
        require(result.returncode == 0, f'{name} exited {result.returncode}; see {stderr}')
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record['finished_unix_ns'] = time.time_ns()
        record['stdout'] = identity(stdout) if stdout.exists() else None
        record['stderr'] = identity(stderr) if stderr.exists() else None
        write_json(ATTEMPT / f'{name}.process.json', record)


def interrupted(signum, _frame):
    raise InterruptedError(f'interrupted by signal {signum}')


def measure():
    plan = verify()
    ready_build()
    started = time.time_ns()
    ATTEMPT.mkdir()
    status, error_text = 'failed', None
    handlers = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        write_json(ATTEMPT / 'plan-identity.json', identity(PLAN))
        retained_step('device-compile', plan['source_assembly_command'], BUILD)
        retained_step('static-embedded-check', plan['static_command'])
        require(read_json(ATTEMPT / 'static-embedded-check.stdout')['status'] == 'passed',
                'static/embedded check did not pass')
        verify(fresh=False)
        retained_step('qualifier', plan['qualification_command'])
        verify(fresh=False)
        retained_step('analyzer', plan['analyzer_command'])
        summary = read_json(ATTEMPT / 'analyzer.stdout')
        require(summary['status'] == 'passed', 'independent raw-sample gate did not pass')
        write_json(ATTEMPT / 'summary.json', summary)
        status = 'passed'
    except BaseException as error:
        error_text = f'{type(error).__name__}: {error}'
        print(f'normalized_linear_measure: FAIL {error_text}', file=sys.stderr)
    finally:
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        write_json(ATTEMPT / 'closure.json', {
            'schema': 'ninfer.r9700.normalized-linear-t1-closure.v1',
            'status': status, 'error': error_text, 'started_unix_ns': started,
            'finished_unix_ns': time.time_ns(), 'plan': identity(PLAN),
            'create_only': True, 'production_routing_authorized': False,
            'artifacts': [identity(path) for path in sorted(ATTEMPT.rglob('*')) if path.is_file()]})
        with exclusive(ATTEMPT / 'result.sha256') as stream:
            for path in sorted(ATTEMPT.rglob('*')):
                if path.is_file() and path.name != 'result.sha256':
                    stream.write(f'{digest(path)}  {path.relative_to(ATTEMPT)}\n')
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    if status == 'passed':
        print('normalized_linear_measure: PASS; independent result review and selector-free whole C1 confirmation remain')
    return 0 if status == 'passed' else 1


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare', 'preflight', 'measure'):
        modes.add_argument(f'--{mode}', action='store_true')
    args = parser.parse_args()
    environment()
    if args.prepare:
        prepare()
    elif args.preflight:
        preflight()
    else:
        return measure()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
