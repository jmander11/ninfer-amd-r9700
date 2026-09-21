#!/usr/bin/env python3
"""Create-only gate-up prefetch qualification; no production routing change."""
from pathlib import Path
import importlib.util
import shlex
import signal
import sys
import time

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
REUSED = ROOT / 'profiles/bench/r9700-normalized-linear-t1-promoted-qualification-20260920/run.py'
spec = importlib.util.spec_from_file_location('qualification_lifecycle', REUSED)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.PACKAGE = PACKAGE
runner.PLAN = PACKAGE / 'plan.json'
runner.ATTEMPT = PACKAGE / 'attempt-1'
runner.TARGET = 'ninfer_r9700_a8q4_gate_up_prefetch_qual'
runner.BINARY = runner.BUILD / 'src' / runner.TARGET
runner.LINEAR = ROOT / 'src/ops/r9700/linear/a8q4_gate_up_prefetch_qualification.hip'
runner.CHECKER = PACKAGE / 'static.py'
runner.BUILD_INPUTS = [runner.BUILD / 'CMakeCache.txt', runner.BUILD / 'compile_commands.json',
                       runner.ARCHIVE, runner.BINARY]
runner.SOURCE_INPUTS += [REUSED, ROOT / 'tools/r9700/a8q4_gate_up_prefetch_qual.hip',
                         runner.LINEAR, runner.LINEAR.with_suffix('.h'),
                         ROOT / 'tools/r9700/check_a8q4_gate_up_prefetch_static.py']
runner.SCRIPT_INPUTS = [PACKAGE / name for name in
                        ('commands.sh', 'run.py', 'analyze.py', 'static.py', 'README.md')]


def assembly_command(directory):
    entries = [entry for entry in runner.read_json(runner.BUILD / 'compile_commands.json')
               if Path(entry['file']) == runner.LINEAR]
    runner.require(len(entries) == 1, 'ambiguous challenger compile command')
    entry = entries[0]
    runner.require(Path(entry['directory']) == runner.BUILD, 'wrong compiler working directory')
    argv = entry.get('arguments') or shlex.split(entry['command'])
    runner.require('--offload-arch=gfx1201' in argv and '-O3' in argv,
                   'wrong challenger compile profile')
    output = argv.index('-o')
    argv = argv[:output] + argv[output + 2:]
    argv.remove('-c')
    return [*argv, '--offload-device-only', '-S', '-o', str(directory / 'qual.s')]


def current_plan():
    runner.environment()
    return {
        'schema': 'ninfer.r9700.gate-up-prefetch-plan.v1',
        'status': 'prepared_for_independent_review', 'production_routing_authorized': False,
        'claim': 'Gate-up T1 prefetch challenger with the identical production normalized preparation in both arms.',
        'bound_inputs': [runner.identity(path) for path in runner.SOURCE_INPUTS + runner.SCRIPT_INPUTS],
        'production_build': [runner.identity(path) for path in runner.BUILD_INPUTS],
        'build_profile': runner.build_profile(), 'link_owner': 'ninfer_r9700_core',
        'link_command': runner.link_command(),
        'source_assembly_command': assembly_command(runner.ATTEMPT),
        'static_command': [sys.executable, str(runner.CHECKER), str(runner.ATTEMPT / 'qual.s'),
                           '--binary', str(runner.BINARY), '--embedded-dir', str(runner.ATTEMPT / 'embedded')],
        'qualification_command': [str(runner.BINARY), '--out-json', str(runner.ATTEMPT / 'qualification.json')],
        'analyzer_command': [sys.executable, str(PACKAGE / 'analyze.py'),
                             str(runner.ATTEMPT / 'qualification.json')],
        'workload': {'device': 0, 'architecture': 'gfx1201', 'wave_size': 32,
                     'pci': '0000:13:00.0', 'vendor': '0x1002', 'device_id': '0x7551',
                     'power_path': str(runner.POWER), 'power': 'auto', 'tokens': 1,
                     'rows': 34816, 'columns': 5120, 'copies': 3, 'paired_trials': 24,
                     'scrub_bytes': 83886080, 'calls_per_token': 64},
        'numerical_contract': {
            'oracle': 'FP64 RMSNorm -> explicit BF16 -> exact A8G64 -> FP64 signed-Q4 dot',
            'criterion': 'normalized_linear_a8q4_fp64_rel_l2_1e-2_bf16_steps_2_v1',
            'codec_exact': True, 'incumbent_bf16_exact': True, 'maximum_bf16_steps': 2,
            'maximum_relative_l2': 0.01, 'graph_poison_stale_finite': True,
            'finite_cases': 4, 'qualified_copies': 3, 'immutable_weights': True},
        'timing_contract': {'boundary': 'complete normalized_linear graph replay',
                           'same_production_preparation': True,
                           'only_terminal_consumer_replaced_from_complete_donor_params': True,
                           'every_allocation_wins': True, 'paired_ratio_mean_plus_2se_below': 1.0,
                           'minimum_saved_ms_per_token': 0.2,
                           'formula': '64*(median(control_ms)-median(candidate_ms))'},
        'static_contract': {'native_iu4': True, 'four_successor_b64_before_current_dot8': True,
                            'no_early_payload_drain': True, 'two_loop_generations_and_terminal': True,
                            'exact_embedded_instruction_bytes_match_checked_assembly': True,
                            'wave_size': 32, 'scratch_spills_lds': 0,
                            'vgpr_max': 32, 'sgpr_max': 64, 'occupancy': 16},
        'attempt': str(runner.ATTEMPT), 'create_only': True,
        'outcome_contract': {'valid_measurement_exit_code': 0, 'closure_status': 'completed',
                             'dispositions': ['passed', 'rejected'],
                             'operational_error_closure_status': 'failed'},
        'exact_invocation': {mode: f'bash {PACKAGE.relative_to(ROOT)}/commands.sh --{mode}'
                             for mode in ('prepare', 'preflight', 'measure')},
        'passing_authorizes': 'independent result review then whole C1 A/B preparation only'}


def measure():
    plan = runner.verify()
    runner.ready_build()
    started = time.time_ns()
    runner.ATTEMPT.mkdir()
    status, disposition, error_text = 'failed', None, None
    handlers = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, runner.interrupted)
        runner.write_json(runner.ATTEMPT / 'plan-identity.json', runner.identity(runner.PLAN))
        runner.retained_step('device-compile', plan['source_assembly_command'], runner.BUILD)
        runner.retained_step('static-embedded-check', plan['static_command'])
        runner.require(runner.read_json(runner.ATTEMPT / 'static-embedded-check.stdout')['status'] == 'passed',
                       'static/embedded check did not pass')
        runner.verify(fresh=False)
        runner.retained_step('qualifier', plan['qualification_command'])
        runner.verify(fresh=False)
        runner.retained_step('analyzer', plan['analyzer_command'])
        summary = runner.read_json(runner.ATTEMPT / 'analyzer.stdout')
        runner.require(summary['status'] in ('passed', 'rejected'), 'invalid analyzed disposition')
        runner.write_json(runner.ATTEMPT / 'summary.json', summary)
        disposition = summary['status']
        status = 'completed'
    except BaseException as error:
        error_text = f'{type(error).__name__}: {error}'
        print(f'gate_up_prefetch_measure: FAIL {error_text}', file=sys.stderr)
    finally:
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        runner.write_json(runner.ATTEMPT / 'closure.json', {
            'schema': 'ninfer.r9700.gate-up-prefetch-closure.v1',
            'status': status, 'disposition': disposition, 'error': error_text,
            'started_unix_ns': started, 'finished_unix_ns': time.time_ns(),
            'plan': runner.identity(runner.PLAN), 'create_only': True,
            'production_routing_authorized': False,
            'artifacts': [runner.identity(path) for path in sorted(runner.ATTEMPT.rglob('*'))
                          if path.is_file()]})
        with runner.exclusive(runner.ATTEMPT / 'result.sha256') as stream:
            for path in sorted(runner.ATTEMPT.rglob('*')):
                if path.is_file() and path.name != 'result.sha256':
                    stream.write(f'{runner.digest(path)}  {path.relative_to(runner.ATTEMPT)}\n')
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    if status == 'completed':
        if disposition == 'passed':
            print('gate_up_prefetch_measure: PASS; completed; independent result review then whole C1 A/B preparation remain')
        else:
            print('gate_up_prefetch_measure: REJECT; completed; direct timing did not admit whole C1 A/B')
    return 0 if status == 'completed' else 1


runner.assembly_command = assembly_command
runner.current_plan = current_plan
runner.measure = measure

if __name__ == '__main__':
    raise SystemExit(runner.main())
