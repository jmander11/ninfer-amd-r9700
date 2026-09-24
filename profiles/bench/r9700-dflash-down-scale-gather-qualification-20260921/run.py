#!/usr/bin/env python3
"""Create-only DFlash scale-gather complete-boundary qualification; preflight never launches HIP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
BUILD = ROOT/'tools/r9700/build'
BINARY = BUILD/'dflash_down_scale_gather_qual'
ASSEMBLY = [BUILD/name for name in ('dflash_down_scale_gather.s', 'a8q4_gdn_pair_t1.s')]
PLAN = PACKAGE/'plan.json'
ATTEMPT = PACKAGE/'attempt-1'
SOURCES = [ROOT/name for name in (
    'tools/r9700/Makefile', 'tools/r9700/dflash_down_scale_gather_qual.hip',
    'tools/r9700/a8q4_shape_sweep_qual.hip',
    'tools/r9700/check_dflash_down_scale_gather_static.py',
    'tools/r9700/check_gdn_projection_control_grid_static.py',
    'tools/r9700/check_a8q4_normalized_linear_t1_static.py',
    'tools/bench/extract_embedded_code_object.py',
    'src/ops/r9700/linear/dflash_down_scale_gather_qualification.hip',
    'src/ops/r9700/linear/dflash_down_scale_gather_qualification.h',
    'src/ops/r9700/linear/r9700_linear.h', 'src/ops/r9700/linear/r9700_linear.hip',
    'src/ops/r9700/linear/r9700_q4_activation_profile.h')]
AUTHORITY = [ROOT/'profiles/bench/r9700-dflash-down-splitk-whole-ab-20260919/results'/name
             for name in ('k4w5-1-control.json', 'k4w5-4-control.json',
                          'k5w6-2-control.json', 'k5w6-3-control.json')]
SCRIPTS = [PACKAGE/name for name in ('run.py', 'analyze.py', 'test_analyze.py', 'commands.sh', 'README.md')]
PCI = Path('/sys/bus/pci/devices/0000:13:00.0')


def require(ok, message):
    if not ok:
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


def write(path, value):
    with exclusive(path) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def environment():
    require(Path.cwd() == ROOT and sys.version_info[:2] == (3, 11), 'root/Python 3.11 required')
    bad = [k for k, v in os.environ.items() if v and
           (k.startswith(('ROCP_', 'ROCPROF', 'HSA_TOOLS', 'ROCTRACER')) or k in (
               'LD_PRELOAD', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
               'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'))]
    require(not bad, f'intercepted environment: {bad}')
    require((PCI/'vendor').read_text().strip() == '0x1002' and
            (PCI/'device').read_text().strip() == '0x7551', 'physical R9700 required')
    require((PCI/'power_dpm_force_performance_level').read_text().strip() == 'auto',
            'auto power required')


def current():
    environment()
    return {'schema': 'ninfer.r9700.dflash-down-scale-gather-plan.v1',
            'status': 'prepared_for_independent_review', 'create_only': True,
            'production_routing_authorized': False,
            'bound_inputs': [identity(p) for p in SOURCES+SCRIPTS+AUTHORITY+[BINARY]+ASSEMBLY],
            'compile_profile': 'gfx1201 wave32 O3 C++20 A8 standalone qualification only',
            'shape': {'tokens': [5, 6], 'columns': 17408, 'rows': 5120,
                      'groups': 272, 'grid': 320, 'threads': 32},
            'workload': {'pci': '0000:13:00.0', 'power': 'auto', 'copies': 3,
                         'paired_trials': 24, 'scrub_bytes': 83886080, 'layers': 64},
            'admission': {'minimum_projected_round_fraction_lower_2se': 0.01,
                          'retained_round_ms': {'5': 92.12005556976744, '6': 93.21258994047619},
                          'every_allocation_wins': True, 'ratio_mean_plus_2se_below': 1.0},
            'oracle': {'formula': 'FP64 decoded signed Q4 times exact A8G64 represented values',
                       'criterion': 'fp64_rel_l2_1e-2_gross_1e-2_refmax_plus_1e-5',
                       'codec_and_status_exact': True, 'incumbent_bf16_exact': True},
            'qualification_command': [str(BINARY), '--out-json', str(ATTEMPT/'qualification.json')],
            'static_command': static_command(ATTEMPT/'embedded'),
            'analyzer_command': [sys.executable, str(PACKAGE/'analyze.py'),
                                 str(ATTEMPT/'qualification.json')],
            'attempt': str(ATTEMPT),
            'valid_loser': 'completed/rejected with full immutable receipts',
            'passing_authorizes': 'whole-inference A/B preparation only'}


def static_command(output):
    return [sys.executable, str(ROOT/'tools/r9700/check_dflash_down_scale_gather_static.py'),
            *map(str, ASSEMBLY), '--binary', str(BINARY), '--embedded-dir', str(output)]


def ready():
    # -q never builds or invokes a recipe; CPU-only stale dependency check.
    result = subprocess.run(['make', '-q', '-C', str(ROOT/'tools/r9700'),
                             'build/dflash_down_scale_gather_qual',
                             *[str(p.relative_to(ROOT/'tools/r9700')) for p in ASSEMBLY]],
                            capture_output=True, text=True)
    require(result.returncode == 0, 'stale/missing build; build explicitly before preparation')


def verify(fresh=True):
    plan = json.loads(PLAN.read_text())
    require(plan == current(), 'plan/source/build/package drift')
    if fresh:
        require(not os.path.lexists(ATTEMPT), 'attempt already exists')
    ready()
    return plan


def step(name, argv):
    record = {'argv': argv, 'cwd': str(ROOT), 'started_unix_ns': time.time_ns(),
              'exit_code': None, 'error': None}
    stdout, stderr = ATTEMPT/f'{name}.stdout', ATTEMPT/f'{name}.stderr'
    try:
        with exclusive(stdout) as out, exclusive(stderr) as err:
            process = subprocess.run(argv, cwd=ROOT, stdout=out, stderr=err, check=False)
        record['exit_code'] = process.returncode
        require(process.returncode == 0, f'{name} failed; inspect retained stderr')
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record['finished_unix_ns'] = time.time_ns()
        record['outputs'] = [identity(p) for p in (stdout, stderr) if p.exists()]
        write(ATTEMPT/f'{name}.process.json', record)


def interrupted(signum, _):
    raise InterruptedError(f'signal {signum}')


def measure():
    plan = verify()
    ATTEMPT.mkdir()
    started = time.time_ns()
    status, disposition, error = 'failed', None, None
    handlers = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, interrupted)
        write(ATTEMPT/'plan-identity.json', identity(PLAN))
        step('static', plan['static_command'])
        require(json.loads((ATTEMPT/'static.stdout').read_text())['status'] == 'passed', 'static gate')
        verify(False)
        step('qualifier', plan['qualification_command'])
        verify(False)
        step('analyzer', plan['analyzer_command'])
        summary = json.loads((ATTEMPT/'analyzer.stdout').read_text())
        require(summary['status'] == 'completed', 'analysis incomplete')
        write(ATTEMPT/'summary.json', summary)
        status, disposition = 'completed', summary['disposition']
    except BaseException as problem:
        error = f'{type(problem).__name__}: {problem}'
        print(error, file=sys.stderr)
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        write(ATTEMPT/'closure.json', {'status': status, 'disposition': disposition,
              'error': error, 'started_unix_ns': started, 'finished_unix_ns': time.time_ns(),
              'plan': identity(PLAN), 'production_routing_authorized': False,
              'artifacts': [identity(p) for p in sorted(ATTEMPT.rglob('*')) if p.is_file()]})
        with exclusive(ATTEMPT/'result.sha256') as stream:
            for path in sorted(ATTEMPT.rglob('*')):
                if path.is_file() and path.name != 'result.sha256':
                    stream.write(f'{digest(path)}  {path.relative_to(ATTEMPT)}\n')
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(f'dflash_scale_gather_measure: {status}/{disposition}')
    return 0 if status == 'completed' else 1


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare', 'preflight', 'measure'):
        modes.add_argument('--'+mode, action='store_true')
    args = parser.parse_args()
    if args.prepare:
        require(not os.path.lexists(PLAN) and not os.path.lexists(ATTEMPT), 'fresh package required')
        ready()
        write(PLAN, current())
        print('prepared; independent review remains')
    elif args.preflight:
        verify()
        with tempfile.TemporaryDirectory(prefix='gdn-complete-static-') as temporary:
            subprocess.run(static_command(Path(temporary)/'embedded'), check=True, cwd=ROOT)
        verify()
        print('preflight passed; no HIP execution or retained attempt')
    else:
        return measure()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

