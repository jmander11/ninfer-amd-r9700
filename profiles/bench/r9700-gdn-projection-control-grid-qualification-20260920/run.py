#!/usr/bin/env python3
"""Create-only GDN complete-boundary qualification; preflight never launches HIP."""
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
BINARY = BUILD/'gdn_projection_control_grid_qual'
ASSEMBLY = [BUILD/name for name in ('gdn_projection_control_grid_qualification.s',
                                   'a8q4_gdn_pair_t1.s', 'gdn_ops.s')]
PLAN = PACKAGE/'plan.json'
ATTEMPT = PACKAGE/'attempt-1'
SOURCES = [ROOT/name for name in (
    'tools/r9700/Makefile', 'tools/r9700/gdn_projection_control_grid_qual.hip',
    'tools/r9700/a8q4_shape_sweep_qual.hip',
    'tools/r9700/check_gdn_projection_control_grid_static.py',
    'tools/r9700/check_a8q4_normalized_linear_t1_static.py',
    'tools/bench/extract_embedded_code_object.py',
    'src/ops/r9700/gdn/gdn_projection_control_grid_qualification.hip',
    'src/ops/r9700/gdn/gdn_projection_control_grid_qualification.h',
    'src/ops/r9700/gdn/gdn_ops.h', 'src/ops/r9700/gdn/gdn_ops.hip',
    'src/ops/r9700/linear/r9700_linear.h', 'src/ops/r9700/linear/r9700_linear.hip',
    'src/ops/r9700/linear/r9700_q4_activation_profile.h')]
SCRIPTS = [PACKAGE/name for name in ('run.py', 'static.py', 'analyze.py', 'test_analyze.py', 'commands.sh', 'README.md')]
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
    return {'schema': 'ninfer.r9700.gdn-projection-control-grid-plan.v1',
            'status': 'prepared_for_independent_review', 'create_only': True,
            'production_routing_authorized': False,
            'bound_inputs': [identity(p) for p in SOURCES+SCRIPTS+[BINARY]+ASSEMBLY],
            'compile_profile': 'gfx1201 wave32 O3 C++20 A8 standalone qualification only',
            'shape': {'tokens': 1, 'columns': 5120, 'qk_rows': 4096, 'value_z_rows': 12288,
                      'control_heads': 48, 'grid': 112, 'threads': 256},
            'workload': {'pci': '0000:13:00.0', 'power': 'auto', 'copies': 3,
                         'paired_trials': 24, 'scrub_bytes': 83886080, 'layers': 48},
            'admission': {'minimum_saved_ms_per_token': 0.2, 'every_allocation_wins': True,
                          'ratio_mean_plus_2se_below': 1.0},
            'oracle': {'projection': 'FP64 decoded signed Q4 times exact A8G64 represented values',
                       'projection_relative_l2': 0.01, 'projection_bf16_steps': 2,
                       'control': 'FP64 BF16 dot -> explicit BF16 a/b -> FP64 control formula',
                       'control_g_absolute_tolerance': 0.0625,
                       'control_beta_absolute_tolerance': 0.0003,
                       'codec_and_status_exact': True, 'incumbent_exact': True},
            'qualification_command': [str(BINARY), '--out-json', str(ATTEMPT/'qualification.json')],
            'static_command': static_command(ATTEMPT/'embedded'),
            'analyzer_command': [sys.executable, str(PACKAGE/'analyze.py'),
                                 str(ATTEMPT/'qualification.json')],
            'attempt': str(ATTEMPT),
            'valid_loser': 'completed/rejected with full immutable receipts',
            'passing_authorizes': 'whole-inference A/B preparation only'}


def static_command(output):
    return [sys.executable, str(PACKAGE/'static.py'), str(BINARY),
            *map(str, ASSEMBLY), str(output)]


def ready():
    # -q never builds or invokes a recipe; CPU-only stale dependency check.
    result = subprocess.run(['make', '-q', '-C', str(ROOT/'tools/r9700'),
                             'build/gdn_projection_control_grid_qual',
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
    print(f'gdn_projection_control_measure: {status}/{disposition}')
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
