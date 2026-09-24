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
PRIOR = ROOT / 'profiles/bench/r9700-a8q4-projected-residual-t1-production-qualification-retry2-20260920'
WHOLE = ROOT / 'profiles/bench/r9700-projected-residual-t1-whole-ab-20260920'
DIAGNOSTIC = ROOT / 'profiles/bench/r9700-a8q4-projected-residual-t1-promoted-qualification-20260920'
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
                 ROOT / 'src/targets/qwen3_8_27b/impl/variant.h',
                 ROOT / 'src/targets/qwen3_8_27b/impl/variant.cpp',
                 ROOT / 'src/ops/r9700/linear/r9700_q4_activation_profile.h', ROOT / 'tools/r9700/a8q4_shape_sweep_qual.hip',
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

WHOLE_INPUTS = [WHOLE / name for name in
                ('plan.json', 'attempt-1/summary.json', 'attempt-1/result.sha256')]

DIAGNOSTIC_INPUTS = [DIAGNOSTIC / name for name in
                     ('plan.json', 'attempt-1/qualification.json', 'attempt-1/closure.json',
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

def verify_accepted_evidence():
    prior = json.loads((PRIOR / 'attempt-1/qualification.json').read_text())
    closure = json.loads((PRIOR / 'attempt-1/closure.json').read_text())
    whole = json.loads((WHOLE / 'attempt-1/summary.json').read_text())
    if (prior['status'] != 'qualified_for_whole_ab_only' or closure['status'] != 'passed'
            or whole['status'] != 'passed' or whole['exact_public_token_parity'] is not True
            or whole['promotion_review_authorized'] is not True):
        raise RuntimeError('accepted production or whole evidence did not pass')
    diagnostic = json.loads((DIAGNOSTIC / 'attempt-1/qualification.json').read_text())
    diagnostic_closure = json.loads((DIAGNOSTIC / 'attempt-1/closure.json').read_text())
    if (diagnostic_closure['status'] != 'passed'
            or diagnostic['status'] != 'qualified_for_whole_ab_only'
            or diagnostic_closure['plan'] != identity(DIAGNOSTIC / 'plan.json')):
        raise RuntimeError('stale-report diagnostic disposition changed')
    for package in (PRIOR, WHOLE, DIAGNOSTIC):
        manifest_root = package / 'attempt-1' if package == WHOLE else ROOT
        for line in (package / 'attempt-1/result.sha256').read_text().splitlines():
            expected, relative = line.split('  ', 1)
            if digest(manifest_root / relative) != expected:
                raise RuntimeError(f'accepted result changed: {relative}')

def verify_selector_free_build():
    for path in (BUILD / 'CMakeCache.txt', BUILD / 'compile_commands.json'):
        if 'NINFER_R9700_PROJECTED_RESIDUAL_T1_CANDIDATE' in path.read_text():
            raise RuntimeError(f'obsolete candidate selector remains in {path}')

def main():
    require_environment()
    if PLAN.exists() or ATTEMPT.exists() or PLAN.is_symlink() or ATTEMPT.is_symlink():
        raise RuntimeError('plan and attempt must be fresh')
    verify_accepted_evidence()
    verify_selector_free_build()
    relative = PACKAGE.relative_to(ROOT)
    plan = {
        'schema': 'ninfer.r9700.projected-residual-promoted-qualification-plan.v1',
        'status': 'prepared_for_review', 'production_routing_authorized': False,
        'claim': 'Confirm linked public Op after selector-free promotion; whole production confirmation remains required.',
        'bound_inputs': [identity(p) for p in SOURCE_INPUTS + SCRIPT_INPUTS],
        'production_build': [identity(p) for p in BUILD_INPUTS],
        'production_build_command': ['cmake', '--build', str(BUILD), '--target',
                                     'ninfer_r9700_a8q4_projected_residual_t1_qual',
                                     'ninfer_r9700_target_variant_projected_residual_qual', '-j', '4'],
        'link_owner': 'ninfer_r9700_core',
        'qualification_executable': str(BINARY),
        'host_validation_command': host_validation_command(),
        'accepted_production_qualification': [identity(p) for p in EVIDENCE_INPUTS],
        'accepted_whole_ab': [identity(p) for p in WHOLE_INPUTS],
        'selector_free_default_build': True,
        'stale_report_diagnostic': [identity(p) for p in DIAGNOSTIC_INPUTS],
        'retry_reason': 'Correct active public qualifier disposition after selector-free promotion; retain prior numerical pass as immutable diagnostic.',
        'report_contract': {
            'status': 'qualified_for_selector_free_production_confirmation',
            'decision': 'prepare_selector_free_whole_c1_confirmation',
            'production_routing_authorized': False},
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
        'passing_authorizes': 'selector-free C1 production confirmation; promotion closure requires that result'}
    prior_plan = json.loads((PRIOR / 'plan.json').read_text())
    for key in ('workload', 'power_checks', 'numerical_contract', 'timing_contract',
                'static_contract'):
        if plan[key] != prior_plan[key]:
            raise RuntimeError(f'promotion confirmation changes accepted direct criteria: {key}')
    write_json(PLAN, plan)
    print('projected_residual_production_prepare: PASS')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
