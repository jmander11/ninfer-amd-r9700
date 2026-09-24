"""CPU-only create-only preparation and immutable benchmark preflight."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from run import ROOT, PACKAGE, RESULTS, PROFILES, ORDERS, fail, load, identity, json_write

PLAN = PACKAGE / 'plan.json'
BUILD = ROOT/'build-r9700-dflash-verify-down-production-20260921'
WHOLE = ROOT/'profiles/bench/r9700-dflash-down-scale-gather-whole-ab-20260921'
DIRECT = ROOT / 'profiles/bench/r9700-dflash-down-scale-gather-qualification-20260921'
ARTIFACT = ROOT / 'out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer'
PCI = Path('/sys/bus/pci/devices/0000:13:00.0')
TARGETS = ['ninfer_bench', 'ninfer_bench_support_test',
           'ninfer_r9700_target_variant_down_scale_gather_qual', 'ninfer_r9700_dflash_verify_down_qual']
COMMON = {'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_HIP_ARCHITECTURES': 'gfx1201',
          'GPU_BUILD_TARGETS': 'gfx1201', 'NINFER_BUILD_BENCHMARKS': 'ON',
          'CMAKE_EXPORT_COMPILE_COMMANDS': 'ON', 'NINFER_R9700_Q4_ACTIVATION_BITS': '8',
          'NINFER_R9700_W8_ACTIVATION_BITS': '8', 'NINFER_R9700_KV_VALUE_GROUP': '16',
          'NINFER_R9700_FP8_QK_WMMA': '1', 'NINFER_R9700_XATTENTION_QUALIFICATION': 'OFF'}
for flag in ('DFLASH_SMALL_T', 'DFLASH_MLP_DOWN_T5',
             'DFLASH_RMSNORM_ROWS56', 'TEXT_P129_WMMA_TAIL', 'GDN_VERIFY_WAVE_QK',
             'FP8_PREFIX_COMMON_ALGO'):
    COMMON['NINFER_R9700_' + flag + '_CANDIDATE'] = '0'


def require(ok, message):
    if not ok:
        fail(message)


def environment():
    require(Path.cwd() == ROOT and sys.version_info[:2] == (3, 11), 'root/Python 3.11 required')
    exact = {'LD_PRELOAD', 'LD_AUDIT', 'HSA_TOOLS_LIB', 'ROCPROFILER_TOOL_LIBRARIES',
             'ROCP_TOOL_LIBRARIES', 'HIP_FORCE_QUEUE_PROFILING', 'AMD_SERIALIZE_KERNEL',
             'AMD_SERIALIZE_COPY', 'ROC_SERIALIZE_KERNEL', 'HIP_VISIBLE_DEVICES',
             'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'}
    prefixes = ('ROCPROF', 'ROCP_', 'ROCTRACER_', 'ROCTX_', 'HSA_TOOLS_', 'HIP_TRACE_',
                'AQLPROFILE_', 'ATT_PROFILE')
    bad = [key for key, value in os.environ.items() if value and
           (key in exact or key.startswith(prefixes))]
    require(not bad, f'intercepted environment: {bad}')
    require((PCI/'vendor').read_text().strip() == '0x1002' and
            (PCI/'device').read_text().strip() == '0x7551', 'R9700 PCI identity differs')
    require((PCI/'power_dpm_force_performance_level').read_text().strip() == 'auto',
            'R9700 auto power required')


def cache(path):
    return {line.split(':', 1)[0]: line.split('=', 1)[1]
            for line in path.read_text().splitlines()
            if ':' in line and '=' in line and not line.startswith(('#', '//'))}


def builds():
    values = cache(BUILD/'CMakeCache.txt')
    for key, value in COMMON.items():
        require(values.get(key) == value, f'production cache differs: {key}')
    for key in ('NINFER_R9700_DFLASH_DOWN_SPLITK_CANDIDATE', 'NINFER_R9700_DFLASH_DOWN_SPLITK_FACTOR',
                'NINFER_R9700_DFLASH_DOWN_SCALE_GATHER_CANDIDATE'):
        require(key not in values, f'retired selector in fresh production cache: {key}')
    dry = subprocess.run(['ninja', '-C', str(BUILD), '-n', *TARGETS],
                         capture_output=True, text=True, check=True)
    require('ninja: no work to do.' in dry.stdout, 'production build is stale')
    return {'production': {'executable': str(BUILD/'bench/ninfer_bench'),
            'identities': [identity(BUILD/p) for p in (
                'CMakeCache.txt', 'compile_commands.json', 'build.ninja',
                'src/libninfer_r9700_core.a', 'src/libninfer_r9700_target_bindings.a',
                'bench/ninfer_bench', 'tests/ninfer_bench_support_test',
                'src/ninfer_r9700_target_variant_down_scale_gather_qual',
                'src/ninfer_r9700_dflash_verify_down_qual')]}}


def current():
    environment()
    direct = load(DIRECT/'attempt-1/summary.json')
    closure = load(DIRECT/'attempt-1/closure.json')
    require(direct.get('status') == 'completed' and direct.get('disposition') == 'admitted'
            and closure.get('disposition') == 'admitted', 'direct gate is not admitted')
    sources = [ROOT/p for p in (
        'CMakeLists.txt', 'src/CMakeLists.txt', 'src/targets/qwen3_8_27b/impl/variant.h',
        'src/targets/qwen3_8_27b/impl/variant.cpp',
        'src/ops/r9700/linear/linear_tensor_op.cpp',
        'src/ops/r9700/linear/r9700_q4_activation_profile.h',
        'src/ops/r9700/linear/dflash_verify_down.h',
        'src/ops/r9700/linear/dflash_verify_down.hip',
        'src/ops/r9700/linear/r9700_linear.hip', 'src/ops/r9700/linear/r9700_linear.h',
        'bench/targets/qwen3_8_27b/ninfer_bench_support.cpp',
        'bench/targets/qwen3_8_27b/ninfer_bench_support.h', 'tests/test_ninfer_bench_support.cpp',
        'tools/r9700/target_variant_down_scale_gather_qual.cpp',
        'tools/r9700/check_dflash_down_scale_gather_static.py',
        'tools/r9700/check_gdn_projection_control_grid_static.py',
        'tools/r9700/check_a8q4_normalized_linear_t1_static.py',
        'tools/bench/extract_embedded_code_object.py',
        'tools/r9700/build/dflash_down_scale_gather.s', 'tools/r9700/build/a8q4_gdn_pair_t1.s')]
    scripts = [PACKAGE/p for p in ('run.py', 'preflight.py', 'analyze.py', 'test_analyze.py', 'test_run.py', 'commands.sh', 'README.md')]
    authority = [DIRECT/p for p in ('plan.json', 'attempt-1/summary.json', 'attempt-1/closure.json',
                                   'attempt-1/result.sha256', 'attempt-1/qualification.json')]
    whole = load(WHOLE/'results/summary.json')
    require(whole.get('disposition') == 'admitted' and load(WHOLE/'results/closure.json').get('status') == 'completed',
            'whole evidence is not admitted')
    authority += [WHOLE/'plan.json'] + [p for p in sorted((WHOLE/'results').iterdir()) if p.is_file()]
    return {'schema': 'ninfer.r9700.dflash-verify-down-production-confirm-plan.v1',
            'status': 'prepared_for_independent_review', 'create_only': True,
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'artifact': identity(ARTIFACT), 'corpus': identity(ROOT/'bench/fixtures/bench_corpus.ids'),
            'builds': builds(), 'bound_inputs': [identity(p) for p in sources+scripts+authority],
            'profiles': {'ordinary': [0, 0], 'k4w5': [4, 5], 'k5w6': [5, 6]},
            'workload': {'concurrency': 1, 'prompt': 128, 'decode': 64, 'public_tokens': 65,
                         'repetitions': 3, 'warmup': 1, 'device_graph': True, 'power': 'auto'},
            'gates': {'tokens': 'exact to admitted ordinary all 65, every repetition',
                      'accounting': 'exact to admitted matched width', 'workspace_growth': False,
                      'linked_oracle': 'exact A8, decoded-Q4 FP64, incumbent BF16, guards, graph poison recovery',
                      'production_timing': 'median <=1.02 retained candidate and <=0.99 retained control, decode and total'},
            'confirmation_only': True,
            'limitation': 'production versus retained matched results is not a fresh A/B',
            'valid_loser': 'completed/rejected with immutable closure and zero exit status'}



def cpu_checks():
    hidden = dict(os.environ, HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1')
    for executable in ('tests/ninfer_bench_support_test',
                       'src/ninfer_r9700_target_variant_down_scale_gather_qual'):
        subprocess.run([str(BUILD/executable)], cwd=ROOT, env=hidden, check=True)
    for test in ('test_run.py', 'test_analyze.py'):
        subprocess.run([sys.executable, str(PACKAGE/test)], cwd=ROOT, check=True)
    for binary in ('bench/ninfer_bench', 'src/ninfer_r9700_dflash_verify_down_qual'):
        with tempfile.TemporaryDirectory(prefix='ninfer-dflash-production-static-') as directory:
            command = [sys.executable, str(ROOT/'tools/r9700/check_dflash_down_scale_gather_static.py'),
                       str(ROOT/'tools/r9700/build/dflash_down_scale_gather.s'),
                       str(ROOT/'tools/r9700/build/a8q4_gdn_pair_t1.s'),
                       '--binary', str(BUILD/binary), '--embedded-dir', str(Path(directory)/'embedded')]
            result = subprocess.run(command, capture_output=True, text=True)
            require(result.returncode == 0, f'embedded ISA check failed: {result.stderr}')


def prepare():
    require(not os.path.lexists(PLAN) and not os.path.lexists(RESULTS), 'package already prepared/executed')
    value = current()
    cpu_checks()
    json_write(PLAN, value)
    print('prepared: ' + identity(PLAN)['sha256'])


def verify():
    require(not os.path.lexists(RESULTS), 'results already exist; never overwrite')
    # Normalize tuples through the JSON representation before comparing with the sealed plan.
    import json
    require(load(PLAN) == json.loads(json.dumps(current())), 'plan/source/build/package drift')
    cpu_checks()
    print('CPU preflight PASS: ' + identity(PLAN)['sha256'])
