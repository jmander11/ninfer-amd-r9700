#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import time
from preflight import verify
from prepare import (ATTEMPT, PACKAGE, PLAN, ROOT, BINARY, BUILD_INPUTS,
                     compile_commands, digest, identity,
                     open_exclusive, write_json)

def retained_step(name, command, stdout_path=None):
    stdout_path = stdout_path or ATTEMPT / f'{name}.stdout'
    stderr_path = ATTEMPT / f'{name}.stderr'
    record = {'name': name, 'argv': command, 'started_unix_ns': time.time_ns(), 'exit_code': None}
    process = None
    try:
        with open_exclusive(stdout_path) as stdout, open_exclusive(stderr_path) as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            record['pid'] = process.pid
            record['exit_code'] = process.wait()
        if record['exit_code'] != 0:
            raise RuntimeError(f'{name} exited {record["exit_code"]}')
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if process is not None:
            record['exit_code'] = process.returncode
        raise
    finally:
        record['finished_unix_ns'] = time.time_ns()
        record['stdout'] = identity(stdout_path) if stdout_path.exists() else None
        record['stderr'] = identity(stderr_path) if stderr_path.exists() else None
        write_json(ATTEMPT / f'{name}.process.json', record)
    return record

def interrupted(signum, _frame):
    raise InterruptedError(f'measurement interrupted by signal {signum}')

def reject_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')

def main():
    verify()
    started = time.time_ns()
    ATTEMPT.mkdir(mode=0o755)
    status, error_text = 'failed', None
    handlers = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        device, checker = compile_commands(ATTEMPT)
        static_receipt = ATTEMPT / 'static-receipt.txt'
        compile_receipt = ATTEMPT / 'compile-receipt.json'
        report = ATTEMPT / 'qualification.json'
        qualifier = [str(BINARY), '--out-json', str(report), '--assembly',
                     str(ATTEMPT / 'qual.s'), '--static-receipt', str(static_receipt),
                     '--compile-receipt', str(compile_receipt), '--package', str(PACKAGE)]
        steps = [retained_step('device-compile', device),
                 retained_step('static-check', checker, static_receipt),
                 retained_step('production-binary-check',
                               [sys.executable, str(PACKAGE / 'production_binary.py'), str(ATTEMPT)])]
        payload = ''.join(chr(i) for i in range(32)) + '"\\ UTF-8: λ界\n'
        serializer_input = ATTEMPT / 'serializer-input.txt'
        with open_exclusive(serializer_input) as stream:
            stream.write(payload)
        serializer_output = ATTEMPT / 'serializer-regression.stdout'
        retained_step('serializer-regression', [str(BINARY), '--serialize-json-text',
                                                str(serializer_input)], serializer_output)
        if json.loads(serializer_output.read_bytes(), parse_constant=reject_constant) != {'text': payload}:
            raise RuntimeError('strict serializer round-trip failed')
        write_json(compile_receipt, {
            'schema': 'ninfer.r9700.immutable-compile-process-receipt.v1',
            'attempt': str(ATTEMPT), 'create_only': True, 'plan': identity(PLAN), 'steps': steps,
            'production_build': [identity(p) for p in BUILD_INPUTS],
            'outputs': {'executable': identity(BINARY),
                        'assembly': identity(ATTEMPT / 'qual.s'),
                        'static_receipt': identity(static_receipt)},
            'measurement_process': {'argv': qualifier}})
        # Recheck bound build objects immediately before starting the GPU process.
        if json.loads(PLAN.read_text())['production_build'] != [identity(p) for p in BUILD_INPUTS]:
            raise RuntimeError('production build changed before qualification')
        retained_step('qualifier', qualifier)
        if json.loads(PLAN.read_text())['production_build'] != [identity(p) for p in BUILD_INPUTS]:
            raise RuntimeError('production build changed during qualification')
        qualification = json.loads(report.read_text(), parse_constant=reject_constant)
        if (qualification.get('status') != 'qualified_for_whole_ab_only'
                or qualification['correctness']['production_symbol'] != 'ninfer::ops::projected_residual_t1'
                or qualification['correctness']['captured_graph_replays_per_shape'] != 3
                or qualification['power_profile']['before_hip_initialization'] != 'auto'
                or qualification['power_profile']['after_correctness_before_timing'] != 'auto'):
            raise RuntimeError('production qualification contract did not pass')
        status = 'passed'
    except BaseException as error:
        error_text = f'{type(error).__name__}: {error}'
        print(f'projected_residual_production_measure: FAIL {error_text}', file=sys.stderr)
    finally:
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        write_json(ATTEMPT / 'closure.json', {
            'schema': 'ninfer.r9700.projected-residual-attempt-closure.v1',
            'status': status, 'error': error_text, 'started_unix_ns': started,
            'finished_unix_ns': time.time_ns(), 'plan': identity(PLAN), 'create_only': True,
            'artifacts': [identity(path) for path in sorted(ATTEMPT.iterdir()) if path.is_file()]})
        with open_exclusive(ATTEMPT / 'result.sha256') as stream:
            for path in sorted(ATTEMPT.iterdir()):
                if path.is_file() and path.name != 'result.sha256':
                    stream.write(f'{digest(path)}  {path.relative_to(ROOT)}\n')
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    if status != 'passed':
        return 1
    print('projected_residual_production_measure: PASS; whole C1 A/B only')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
