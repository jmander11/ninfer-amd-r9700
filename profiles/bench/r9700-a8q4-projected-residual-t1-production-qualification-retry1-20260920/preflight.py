#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import tempfile
from prepare import (ATTEMPT, PLAN, ROOT, BINARY, PACKAGE, BUILD_INPUTS,
                     SOURCE_INPUTS, SCRIPT_INPUTS, EVIDENCE_INPUTS, FAILED_INPUTS,
                     compile_commands, identity, require_environment)

def verify():
    require_environment()
    plan = json.loads(PLAN.read_text())
    if (plan.get('schema') != 'ninfer.r9700.projected-residual-production-qualification-plan.v1'
            or plan.get('production_routing_authorized') is not False
            or plan.get('status') != 'prepared_for_review'):
        raise RuntimeError('plan disposition differs')
    for key, paths in [('bound_inputs', SOURCE_INPUTS + SCRIPT_INPUTS),
                       ('production_build', BUILD_INPUTS),
                       ('accepted_retry2', EVIDENCE_INPUTS),
                       ('failed_production_diagnostic', FAILED_INPUTS)]:
        if plan[key] != [identity(path) for path in paths]:
            raise RuntimeError(f'{key} changed after binding')
    if ATTEMPT.exists() or ATTEMPT.is_symlink():
        raise RuntimeError('immutable attempt already exists')
    return plan

def serializer_check(binary, directory):
    payload = ''.join(chr(i) for i in range(32)) + '"\\ UTF-8: λ界\n'
    source = directory / 'serializer-input.txt'
    source.write_bytes(payload.encode('utf-8'))
    result = subprocess.run([str(binary), '--serialize-json-text', str(source)],
                            cwd=ROOT, check=True, capture_output=True)
    if json.loads(result.stdout) != {'text': payload}:
        raise RuntimeError('strict JSON serializer round-trip failed')

def main():
    verify()
    with tempfile.TemporaryDirectory(prefix='projected-residual-production-preflight-') as tmp:
        directory = Path(tmp)
        device, checker = compile_commands(directory)
        serializer_check(BINARY, directory)
        subprocess.run(device, cwd=ROOT, check=True)
        subprocess.run(checker, cwd=ROOT, check=True)
        subprocess.run([__import__('sys').executable, str(PACKAGE / 'production_binary.py'),
                        str(directory)], cwd=ROOT, check=True)
    print('projected_residual_production_preflight: PASS; no HIP initialized; attempt absent')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
