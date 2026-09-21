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
RESULTS = PACKAGE / 'results'
BUILD = ROOT / 'build-r9700-three-route-production-20260920'
BENCH = ROOT / 'profiles/bench'
WHOLE = {
    'gdn': BENCH / 'r9700-bf16-gdn-control-t1-whole-ab-retry2-20260920/results',
    'attention': BENCH / 'r9700-attention-q4-pair-t1-whole-ab-20260919/results',
    'paired': BENCH / 'r9700-paired-projection-c2c4-whole-ab-20260919/results',
}
AUTHORITY = {1: WHOLE['attention'] / 'run-2-candidate.json',
             2: WHOLE['paired'] / 'run-2-c2-candidate.json',
             3: WHOLE['paired'] / 'run-4-c3-candidate.json',
             4: WHOLE['paired'] / 'run-6-c4-candidate.json'}
REMOVED = ('BF16_GDN_CONTROL_T1_CANDIDATE', 'ATTENTION_Q4_PAIR_T1_CANDIDATE',
           'Q4_PAIR_WMMA_C2C4_CANDIDATE')
CACHE = {'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_HIP_ARCHITECTURES': 'gfx1201',
         'CMAKE_HIP_COMPILER': '/opt/rocm/llvm/bin/clang++',
         'NINFER_BUILD_APPS': 'ON', 'NINFER_BUILD_BENCHMARKS': 'ON',
         'NINFER_R9700_KV_VALUE_GROUP': '16', 'NINFER_R9700_Q4_ACTIVATION_BITS': '8',
         'NINFER_R9700_W8_ACTIVATION_BITS': '8', 'NINFER_R9700_FP8_QK_WMMA': '1',
         'NINFER_R9700_XATTENTION_QUALIFICATION': 'OFF',
         'NINFER_R9700_XATTENTION_STRIDE': '16', 'NINFER_R9700_XATTENTION_TAU_PERMILLE': '1000',
         'NINFER_R9700_DFLASH_DOWN_SPLITK_FACTOR': '8',
         **{'NINFER_R9700_' + name + '_CANDIDATE': '0' for name in (
             'DFLASH_SMALL_T', 'DFLASH_MLP_DOWN_T5', 'DFLASH_DOWN_SPLITK',
             'DFLASH_RMSNORM_ROWS56', 'TEXT_P129_WMMA_TAIL', 'GDN_VERIFY_WAVE_QK',
             'FP8_PREFIX_COMMON_ALGO')}}
QUALIFIERS = ('ninfer_r9700_target_variant_gdn_qual',
              'ninfer_r9700_target_variant_attention_projection_qual',
              'ninfer_r9700_target_variant_pair_c2c4_qual')
SOURCES = [ROOT / name for name in (
    'CMakeLists.txt', 'src/CMakeLists.txt',
    'src/targets/qwen3_8_27b/impl/variant.h', 'src/targets/qwen3_8_27b/impl/variant.cpp',
    'src/ops/r9700/linear/r9700_linear.h', 'src/ops/r9700/linear/r9700_linear.hip',
    'src/ops/r9700/gdn/gdn_gating.cpp', 'src/ops/r9700/gdn/gdn_ops.hip',
    'src/ops/r9700/attention_projection/attention_projection.h',
    'src/ops/r9700/attention_projection/attention_projection.hip',
    'src/ops/r9700/paired_projection/paired_projection.h',
    'src/ops/r9700/paired_projection/paired_projection.hip',
    'include/ninfer/ops/attention_projection.h', 'include/ninfer/ops/gdn_projection.h',
    'bench/targets/qwen3_8_27b/ninfer_bench_support.cpp',
    'tools/r9700/target_variant_gdn_qual.cpp',
    'tools/r9700/target_variant_attention_projection_qual.cpp',
    'tools/r9700/target_variant_pair_c2c4_qual.cpp')]
SOURCES += [PACKAGE / name for name in ('commands.sh', 'campaign.py', 'README.md')]
PCI = Path('/sys/bus/pci/devices/0000:13:00.0')
LIMITATIONS = ['Unprofiled whole decode does not establish physical bandwidth or stall freedom.',
               'Low pre/post VRAM does not exclude competing GPU activity during inference.',
               'One process per concurrency confirms composition; it is not a new speedup A/B.']


def fail(message):
    raise RuntimeError(message)


def load(path):
    value = json.loads(path.read_text(), parse_constant=lambda x: fail(f'nonfinite JSON: {x}'))
    if not isinstance(value, dict): fail(f'not a JSON object: {path}')
    return value


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    if not path.is_file() or path.is_symlink(): fail(f'not a regular file: {path}')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def check(item):
    if item != identity(Path(item['path'])): fail(f'identity changed: {item["path"]}')


def exclusive(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(fd, 'w') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')


def cache():
    result = {}
    for line in (BUILD / 'CMakeCache.txt').read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line and ':' in line.split('=', 1)[0]:
            key, value = line.split('=', 1)
            result[key.split(':', 1)[0]] = value
    for key, value in CACHE.items():
        if result.get(key) != value: fail(f'build cache differs: {key}')
    if any('NINFER_R9700_' + name in result for name in REMOVED):
        fail('removed selector remains in build cache')
    for key, value in result.items():
        if key.startswith('NINFER_R9700_') and key.endswith(('_CANDIDATE', '_QUALIFICATION')):
            if value not in ('0', 'OFF'): fail(f'candidate enabled: {key}')
    return {key: value for key, value in result.items()
            if key.startswith(('NINFER_R9700_', 'CMAKE_HIP_', 'CMAKE_CXX_', 'CMAKE_EXE_LINKER_'))
            or key in CACHE}


def closure(directory):
    manifest = directory / 'result.sha256'
    seen = set()
    for line in manifest.read_text().splitlines():
        expected, name = line.split('  ', 1)
        if Path(name).name != name or name in seen or digest(directory / name) != expected:
            fail(f'retained closure differs: {directory}/{name}')
        seen.add(name)
    actual = {p.name for p in directory.iterdir() if p.is_file() and p.name != 'result.sha256'}
    if actual != seen: fail(f'incomplete retained closure: {directory}')


def tokens(report, concurrency):
    tests = report.get('tests', [])
    if len(tests) != 1 or tests[0].get('label') != 'whole-pp8192+tg256': fail('token workload differs')
    reps = tests[0].get('reps', [])
    lanes = reps[0].get('generated_token_ids_by_lane') if len(reps) == 1 else None
    if (not isinstance(lanes, list) or len(lanes) != concurrency or
            any(not isinstance(lane, list) or len(lane) != 257 or
                any(type(token) is not int or token < 0 for token in lane) for lane in lanes)):
        fail('token lane geometry differs')
    return hashlib.sha256(json.dumps(lanes, separators=(',', ':')).encode()).hexdigest()


def telemetry(expected_total=None, require_low=True):
    # Sysfs reads only. The report independently confirms HIP device 0 is gfx1201/R9700.
    sample = {'pci': PCI.name, 'power': (PCI / 'power_dpm_force_performance_level').read_text().strip(),
              'used_bytes': int((PCI / 'mem_info_vram_used').read_text()),
              'total_bytes': int((PCI / 'mem_info_vram_total').read_text()),
              'device_vendor': (PCI / 'vendor').read_text().strip(),
              'device_id': (PCI / 'device').read_text().strip()}
    if (sample['power'] != 'auto' or sample['device_vendor'] != '0x1002' or
            sample['device_id'] != '0x7551' or sample['total_bytes'] < 32000000000 or
            not 0 <= sample['used_bytes'] <= sample['total_bytes'] or
            (expected_total is not None and sample['total_bytes'] != expected_total)):
        fail(f'device/power/VRAM telemetry differs: {sample}')
    sample['low_vram'] = (sample['used_bytes'] <= 1073741824 and
                          sample['used_bytes'] * 1000 <= sample['total_bytes'] * 50)
    if require_low and not sample['low_vram']: fail(f'VRAM is occupied: {sample}')
    return sample


def prepare():
    for path in (PLAN, RESULTS, BUILD):
        if path.exists() or path.is_symlink(): fail(f'prepare requires fresh path: {path}')
    before_sources = [identity(path) for path in SOURCES]
    retained = {}
    summaries = {}
    for role, directory in WHOLE.items():
        closure(directory)
        summary = load(directory / 'summary.json')
        if (summary.get('status') != 'passed' or summary.get('production_routing_authorized') is not True
                or summary.get('exact_public_token_parity') is not True): fail(f'admission differs: {role}')
        summaries[role] = summary
        retained[role] = {'summary': identity(directory / 'summary.json'),
                          'closure': identity(directory / 'result.sha256')}
    reference = {1: min(statistics.median(p['candidate_seconds'] for p in summaries[role]['pairs'])
                        for role in ('gdn', 'attention'))}
    reference.update({c: statistics.median(p['candidate_seconds']
                      for p in summaries['paired']['by_concurrency'][str(c)]['pairs']) for c in (2, 3, 4)})
    hardware = telemetry()
    subprocess.run(['cmake', '-S', str(ROOT), '-B', str(BUILD), '-GNinja'] +
                   [f'-D{key}={value}' for key, value in CACHE.items()], cwd=ROOT, check=True)
    subprocess.run(['cmake', '--build', str(BUILD), '-j2', '--target', 'ninfer_bench', *QUALIFIERS],
                   cwd=ROOT, check=True)
    receipts = []
    for name in QUALIFIERS:
        path = BUILD / 'src' / name
        result = subprocess.run([str(path)], cwd=ROOT, check=True, capture_output=True, text=True)
        receipts.append({'executable': identity(path), 'stdout': result.stdout})
    obj = BUILD / 'src/CMakeFiles/ninfer_r9700_target_bindings.dir/targets/qwen3_8_27b/impl/variant.cpp.o'
    symbols = subprocess.run(['nm', '-C', str(obj)], check=True, capture_output=True, text=True).stdout
    required = ('bf16_gdn_projected_gating_t1(', 'full_attention_projection_t1(',
                'full_attention_projection_decode(', 'gdn_input_projection_decode(')
    for name in required:
        if name not in symbols: fail(f'production symbol absent: {name}')
    for item in before_sources: check(item)
    plan = {'schema': 'ninfer.r9700.three-route-production-confirmation.v1',
            'status': 'prepared_awaiting_independent_review',
            'workload': {'concurrency': [1, 2, 3, 4], 'whole_pg': '8192,256', 'prefill_chunk': 4096,
                         'spec': 'none', 'device_graph': True, 'repetitions': 1, 'warmup': 1},
            'sources': before_sources, 'build_cache': identity(BUILD / 'CMakeCache.txt'),
            'cache_values': cache(), 'executable': identity(BUILD / 'bench/ninfer_bench'),
            'route_assertions': receipts, 'variant_object': identity(obj), 'route_symbols': list(required),
            'artifact': identity(ROOT / 'out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer'),
            'corpus': identity(ROOT / 'bench/fixtures/bench_corpus.ids'),
            'retained_campaigns': retained,
            'token_authorities': {str(c): {'authority': identity(path), 'sha256': tokens(load(path), c)}
                                  for c, path in AUTHORITY.items()},
            'admission': {str(c): {'retained_candidate_median_seconds': reference[c],
                         'maximum_ratio': 1.005 if c == 1 else 1.02,
                         'maximum_decode_seconds': reference[c] * (1.005 if c == 1 else 1.02)}
                         for c in (1, 2, 3, 4)},
            'hardware': hardware, 'post_exit_drain': {'maximum_seconds': 30, 'poll_seconds': 0.1},
            'limitations': LIMITATIONS}
    exclusive(PLAN, plan)
    print('three_route_production_prepare: PASS')


def preflight():
    if RESULTS.exists() or RESULTS.is_symlink(): fail('results path is not fresh')
    plan = load(PLAN)
    if (plan.get('schema') != 'ninfer.r9700.three-route-production-confirmation.v1' or
            plan.get('status') != 'prepared_awaiting_independent_review' or
            plan.get('cache_values') != cache()): fail('plan/build contract differs')
    for item in plan['sources'] + [plan[key] for key in
            ('build_cache', 'executable', 'variant_object', 'artifact', 'corpus')]: check(item)
    for receipt in plan['route_assertions']: check(receipt['executable'])
    for role, retained in plan['retained_campaigns'].items():
        check(retained['summary']); check(retained['closure']); closure(WHOLE[role])
    for c, item in plan['token_authorities'].items():
        check(item['authority'])
        if tokens(load(Path(item['authority']['path'])), int(c)) != item['sha256']: fail('token authority changed')
    telemetry(plan['hardware']['total_bytes'])
    print('three_route_production_preflight: PASS')
    return plan


def command(plan, c):
    return [plan['executable']['path'], '--weights', plan['artifact']['path'], '--corpus',
            plan['corpus']['path'], '--device', '0', '--concurrency', str(c), '--whole-pg', '8192,256',
            '--prefill-chunk', '4096', '--kv-capacity', 'workload', '--spec', 'mtp', '--draft-tokens', '0',
            '--retain-token-ids', '--output', 'json', '--output-file', str(RESULTS / f'c{c}.json'),
            '-r', '1', '--warmup', '1']


def report_result(plan, c):
    report = load(RESULTS / f'c{c}.json')
    environment = report.get('environment', {})
    if (report.get('schema_version') != 20 or report.get('artifact_type') != 'ninfer_bench_report' or
            report.get('tool') != 'ninfer_bench' or environment.get('device_id') != 0 or
            environment.get('gpu_name') != 'AMD Radeon AI PRO R9700' or
            environment.get('architecture_name') != 'gfx1201' or report.get('artifact') !=
            {'path': plan['artifact']['path'], 'file_size_bytes': plan['artifact']['bytes']}):
        fail(f'C{c} report provenance differs')
    config = report.get('config', {})
    expected = {'concurrency': c, 'prefill_chunk': 4096, 'kv_cache_format': 'fp8-k-int4-v',
                'kv_value_group': 16, 'q4_activation_bits': 8, 'w8_activation_bits': 8,
                'fp8_qk_wmma_enabled': True, 'xattention_qualification': False,
                'spec': 'none', 'draft_tokens': 0, 'use_device_graph': True, 'retain_token_ids': True,
                'decode_path': 'device_graph', 'repetitions': 1, 'warmup': 1,
                'decode_graph_prime': {'primed': True, 'output_tokens': 3}}
    for key, value in expected.items():
        if config.get(key) != value: fail(f'C{c} report config differs: {key}')
    if any(name.lower() in config for name in REMOVED): fail('removed selector leaked into report')
    if any(value is not False for key, value in config.items() if key.endswith('_candidate')):
        fail('other candidate active in report')
    if tokens(report, c) != plan['token_authorities'][str(c)]['sha256']: fail(f'C{c} token parity failed')
    test = report['tests'][0]; rep = test['reps'][0]
    if (test.get('kind') != 'whole' or test.get('n_prompt') != 8192 or test.get('n_gen') != 256 or
            test.get('requested_output_tokens') != 257 or rep.get('generated_output_tokens') != 257*c or
            rep.get('decode_output_tokens') != 256 or rep.get('decode_engine_tokens') != 256):
        fail(f'C{c} geometry differs')
    seconds = rep.get('timings', {}).get('decode_seconds')
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0 or
            seconds > plan['admission'][str(c)]['maximum_decode_seconds']):
        fail(f'C{c} decode timing exceeds retained-candidate regression ceiling: {seconds}')
    return {'concurrency': c, 'decode_seconds': seconds, 'decode_output_tok_s': 256*c/seconds,
            'per_lane_decode_tok_s': 256/seconds, 'exact_retained_tokens': True,
            'over_retained_candidate': seconds / plan['admission'][str(c)]['retained_candidate_median_seconds']}


def measure():
    plan = preflight()
    RESULTS.mkdir()
    try:
        exclusive(RESULTS / 'plan-identity.json', identity(PLAN))
        records = []
        for c in (1, 2, 3, 4):
            receipt = {'concurrency': c, 'command': command(plan, c), 'cwd': str(ROOT),
                       'executable': plan['executable'], 'started_unix_ns': time.time_ns()}
            try:
                receipt['before'] = telemetry(plan['hardware']['total_bytes'])
                completed = subprocess.run(receipt['command'], cwd=ROOT, capture_output=True, text=True)
                drain_started = time.monotonic()
                receipt.update(exit_code=completed.returncode, finished_unix_ns=time.time_ns())
                exclusive(RESULTS / f'c{c}.stdout', completed.stdout)
                exclusive(RESULTS / f'c{c}.stderr', completed.stderr)
                receipt['drain_samples'] = []
                while True:
                    sample = telemetry(plan['hardware']['total_bytes'], require_low=False)
                    elapsed = time.monotonic() - drain_started
                    receipt['drain_samples'].append({'elapsed_seconds': elapsed, **sample})
                    if elapsed > 30: fail('post-process VRAM drain exceeded 30 seconds')
                    if sample['low_vram']: break
                    time.sleep(min(0.1, 30-elapsed))
                receipt['stdout'] = identity(RESULTS / f'c{c}.stdout')
                receipt['stderr'] = identity(RESULTS / f'c{c}.stderr')
                if completed.returncode != 0: fail(f'C{c} process exited {completed.returncode}')
                receipt['report'] = identity(RESULTS / f'c{c}.json')
                check(plan['executable'])
                records.append(report_result(plan, c))
            except Exception as error:
                receipt['failure'] = f'{type(error).__name__}: {error}'
                raise
            finally:
                exclusive(RESULTS / f'c{c}.process.json', receipt)
        exclusive(RESULTS / 'summary.json', {'schema': 'ninfer.r9700.three-route-production-confirmation-result.v1',
                  'status': 'passed', 'production_composition_confirmed': True,
                  'exact_public_token_parity': True, 'by_concurrency': records, 'limitations': LIMITATIONS})
        print('three_route_production_confirmation: PASS')
    except Exception as error:
        exclusive(RESULTS / 'failure.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        files = sorted(path for path in RESULTS.iterdir() if path.is_file())
        exclusive(RESULTS / 'result.sha256', ''.join(f'{digest(path)}  {path.name}\n' for path in files))


if __name__ == '__main__':
    if Path.cwd() != ROOT: fail(f'must run from {ROOT}')
    if sys.version_info[:2] != (3, 11): fail('use the selected Python 3.11 interpreter')
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'preflight', 'measure'): group.add_argument('--'+name, action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.preflight: preflight()
    else: measure()
