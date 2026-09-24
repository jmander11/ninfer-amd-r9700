#!/usr/bin/env python3
"""Create the production-symbol qualification plan once; never refresh it."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
PLAN = PACKAGE / 'plan.json'
ATTEMPT = PACKAGE / 'attempt-1'
FAILED_PACKAGES = [ROOT / name for name in
                   ('profiles/bench/r9700-a8q4-projected-residual-t1-production-qualification-20260920',
                    'profiles/bench/r9700-a8q4-projected-residual-t1-production-qualification-retry1-20260920')]
PRIOR = ROOT / 'profiles/bench/r9700-a8q4-projected-residual-t1-design-retry2-20260920'
QUALIFIER = ROOT / 'tools/r9700/a8q4_projected_residual_t1_qual.hip'
LINEAR = ROOT / 'src/ops/r9700/linear/r9700_linear.hip'
EAGER = ROOT / 'src/ops/r9700/eager/eager_ops.hip'
CHECKER = ROOT / 'tools/r9700/check_a8q4_projected_residual_t1_static.py'
BUILD = ROOT / 'build-r9700'
BINARY = BUILD / 'src/ninfer_r9700_a8q4_projected_residual_t1_qual'
HOST_BINARY = BUILD / 'src/ninfer_r9700_target_variant_projected_residual_qual'
ARCHIVE = BUILD / 'src/libninfer_r9700_core.a'
BUILD_INPUTS = [BUILD / 'CMakeCache.txt', BUILD / 'compile_commands.json', ARCHIVE, BINARY, HOST_BINARY]
WRAPPER = ROOT / 'src/ops/r9700/linear/linear_tensor_op.cpp'
CORE_SOURCES = [ROOT / 'src/core' / name for name in
                ('device.hip', 'arena.hip', 'dtype.cpp', 'tensor.cpp')]
SOURCE_INPUTS = [QUALIFIER, ROOT / 'tools/r9700/target_variant_projected_residual_qual.cpp',
                 ROOT / 'src/targets/qwen3_8_27b/impl/variant.h', ROOT / 'tools/r9700/a8q4_shape_sweep_qual.hip',
                 CHECKER, LINEAR, LINEAR.with_suffix('.h'), EAGER, EAGER.with_suffix('.h'),
                 WRAPPER, ROOT / 'include/ninfer/ops/projected_residual.h',
                 ROOT / 'include/ninfer/ops/linear.h', *CORE_SOURCES,
                 ROOT / 'CMakeLists.txt', ROOT / 'src/CMakeLists.txt',
                 ROOT / 'tools/bench/extract_embedded_code_object.py']
SCRIPT_INPUTS = [PACKAGE / name for name in
                 ('commands.sh', 'prepare.py', 'preflight.py', 'measure.py',
                  'production_binary.py', 'README.md')]
EVIDENCE_INPUTS = [PRIOR / name for name in
                   ('plan.json', 'attempt-1/qualification.json', 'attempt-1/closure.json',
                    'attempt-1/result.sha256')]

FAILED_INPUTS = [package / name for package in FAILED_PACKAGES for name in
                 ('README.md', 'commands.sh', 'prepare.py', 'preflight.py', 'measure.py',
                  'production_binary.py', 'plan.json', 'attempt-1/closure.json',
                  'attempt-1/result.sha256')]

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def identity(path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f'expected regular file: {path}')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}

def open_exclusive(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                             0o644), 'w', encoding='utf-8')

def write_json(path, value):
    with open_exclusive(path) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def require_environment():
    if Path.cwd() != ROOT or sys.version_info[:2] != (3, 11):
        raise RuntimeError(f'run from {ROOT} with Python 3.11')

def compile_commands(build):
    common = ['/opt/rocm/bin/hipcc', '-O3', '-std=c++20', '--offload-arch=gfx1201',
              f'-I{ROOT / "src"}', f'-I{ROOT / "include"}', '-isystem', '/opt/rocm/include',
              f'-DNINFER_SOURCE_DIR="{ROOT}"']
    device = [*common, '-Wno-unused-command-line-argument', '--offload-device-only',
              '-S', str(LINEAR), '-o', str(build / 'qual.s')]
    return device, [sys.executable, str(CHECKER), str(build / 'qual.s')]

def host_validation_command():
    return ['/usr/bin/env', 'ROCR_VISIBLE_DEVICES=-1', 'HIP_VISIBLE_DEVICES=-1',
            str(HOST_BINARY)]

def verify_failed_diagnostics():
    for package in FAILED_PACKAGES:
        attempt = package / 'attempt-1'
        closure = json.loads((attempt / 'closure.json').read_text())
        if (closure['status'] != 'failed'
                or closure['plan'] != identity(package / 'plan.json')):
            raise RuntimeError(f'expected immutable failed closure: {package}')
        artifacts = sorted(p for p in attempt.iterdir()
                           if p.name not in ('closure.json', 'result.sha256'))
        if closure['artifacts'] != [identity(p) for p in artifacts]:
            raise RuntimeError(f'failed diagnostic artifacts changed: {package}')
        expected_manifest = ''.join(
            f'{digest(p)}  {p.relative_to(ROOT)}\n'
            for p in sorted([*artifacts, attempt / 'closure.json']))
        if (attempt / 'result.sha256').read_text() != expected_manifest:
            raise RuntimeError(f'failed diagnostic manifest changed: {package}')

def main():
    require_environment()
    if PLAN.exists() or ATTEMPT.exists() or PLAN.is_symlink() or ATTEMPT.is_symlink():
        raise RuntimeError('plan and attempt must be fresh')
    prior = json.loads((PRIOR / 'attempt-1/qualification.json').read_text())
    closure = json.loads((PRIOR / 'attempt-1/closure.json').read_text())
    if prior['status'] != 'qualified_for_whole_ab_only' or closure['status'] != 'passed':
        raise RuntimeError('retry2 evidence is not accepted')
    verify_failed_diagnostics()
    failed_plans = [json.loads((p / 'plan.json').read_text()) for p in FAILED_PACKAGES]
    relative = PACKAGE.relative_to(ROOT)
    plan = {
        'schema': 'ninfer.r9700.projected-residual-production-qualification-plan.v1',
        'status': 'prepared_for_review', 'production_routing_authorized': False,
        'claim': 'Qualify exact linked public ops::projected_residual_t1 before whole C1 A/B.',
        'bound_inputs': [identity(p) for p in SOURCE_INPUTS + SCRIPT_INPUTS],
        'production_build': [identity(p) for p in BUILD_INPUTS],
        'production_build_command': ['cmake', '--build', str(BUILD), '--target',
                                     'ninfer_r9700_a8q4_projected_residual_t1_qual',
                                     'ninfer_r9700_target_variant_projected_residual_qual', '-j', '4'],
        'link_owner': 'ninfer_r9700_core',
        'qualification_executable': str(BINARY),
        'host_validation_command': host_validation_command(),
        'accepted_retry2': [identity(p) for p in EVIDENCE_INPUTS],
        'failed_production_diagnostic': [identity(p) for p in FAILED_INPUTS],
        'retry_reason': 'Reject malformed public Op bindings before HIP launch; retain corrected PCI power path and unchanged admission criteria',
        'workload': {'device': 0, 'architecture': 'gfx1201', 'tokens': 1, 'rows': 5120,
                     'columns': [6144, 17408], 'power': 'auto', 'copies': 3,
                     'trials_per_arm': 12, 'cache_scrub_bytes': 83886080},
        'power_checks': ['before HIP initialization', 'after each shape correctness before timing',
                         'after timing'],
        'numerical_contract': {
            'oracle': 'independent FP64 represented A8 activation and signed packed-Q4 weight decode',
            'boundary': 'delta=BF16(dot); residual=BF16(FP32(residual)+FP32(delta))',
            'maximum_bf16_steps': 2, 'exact_incumbent_residual_bits': True,
            'captured_graph_replays_per_shape': 3,
            'poisoning_canaries_malformed_alias_checks': True},
        'timing_contract': {'balanced_allocation_and_arm_order': True,
                            'maximum_ratio_each_allocation': 1.01,
                            'minimum_saved_ms_per_token': 0.2,
                            'formula': '64*sum(incumbent_shape_median-candidate_shape_median)'},
        'static_contract': {'source': str(LINEAR), 'native_iu4_dot8': True, 'wave_size': 32,
                            'maximum_vgpr': 32, 'maximum_sgpr': 64, 'lds_scratch_spills': 0},
        'exact_invocation': {name: f'bash {relative}/commands.sh --{name}'
                             for name in ('preflight', 'measure')},
        'attempt': str(ATTEMPT), 'create_only': True,
        'passing_authorizes': 'whole C1 A/B; no production promotion'}
    for key in ('workload', 'power_checks', 'numerical_contract', 'timing_contract',
                'static_contract', 'passing_authorizes'):
        if any(plan[key] != failed_plan[key] for failed_plan in failed_plans):
            raise RuntimeError(f'retry changes admission contract: {key}')
    write_json(PLAN, plan)
    print('projected_residual_production_prepare: PASS')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
