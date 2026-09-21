#!/usr/bin/env python3
"""Extract the selected executable's actual ISA; never execute it or infer dispatch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from tools.bench.extract_embedded_code_object import extract
from tools.bench.verify_selected_hardware_use import (
    AUDIT_SCHEMA, _machine_interval, selected_route, snapshot,
)

LLVM = Path('/opt/rocm/llvm/bin')
SPECS = {
    'q4_p2048_cta': ('linear_by_profile', 'a8q4g64_linear_prefill_cta_kernel', 'v_wmma_i32_16x16x32_iu4'),
    'q4_wave32': ('linear_by_profile', 'a8q4g64_linear_wmma32_kernel', 'v_wmma_i32_16x16x32_iu4'),
    'w8_p2048_cta': ('linear_by_profile', 'a8w8g32_linear_prefill_cta_kernel', 'v_wmma_i32_16x16x16_iu8'),
    'ordinary_fp8_qk': ('attention_by_profile', 'qk_wmma_kernelILb0EE', 'v_wmma_f32_16x16x16_fp8_fp8'),
    'dense_panel_qk': ('attention_by_profile', 'dense_full_score_qk_bk32_kernelILb0EE', 'v_wmma_f32_16x16x16_bf16'),
    'xattention_rank': ('xattention_by_profile', 'xattention_rank_kernelILj16EE', 'v_wmma_f32_16x16x16_bf16'),
    'xattention_flash_consumer': ('xattention_by_profile', '', 'v_wmma_f32_16x16x16_bf16'),
}


def unique_symbol(binary: bytes, token: str) -> str:
    matches = {item.decode('ascii') for item in re.findall(rb'_Z[A-Za-z0-9_]+', binary)
               if token.encode() in item and b'__device_stub__' not in item}
    if len(matches) != 1:
        raise ValueError(f'expected one selected code symbol for {token}; got {len(matches)}')
    return matches.pop()


def symbol_proof(symbol: str, opcode: str, disassembly: str, metadata: str) -> dict:
    # readelf's AMDGPU notes contain the emitted kernel YAML, not guessed occupancy.
    records = re.split(r'(?m)^\s*-\s*\.args:', metadata)
    records = [record for record in records
               if re.search(r'(?m)^\s*\.name:\s*' + re.escape(symbol) + r'\s*$', record)]
    if len(records) != 1:
        raise ValueError(f'expected one metadata record for {symbol}')
    def integer(field):
        values = re.findall(r'(?m)^\s*\.' + field + r':\s*(\d+)\s*$', records[0])
        if len(values) != 1:
            raise ValueError(f'missing unique {field} for {symbol}')
        return int(values[0])
    if integer('wavefront_size') != 32:
        raise ValueError('selected ISA is not wave32')
    return {'code_symbol': symbol, 'opcode': opcode,
            'opcode_sites': len(re.findall(r'\b' + re.escape(opcode) + r'\b', disassembly)),
            'vgpr': integer('vgpr_count'), 'lds_bytes': integer('group_segment_fixed_size'),
            'private_bytes': integer('private_segment_fixed_size'),
            'scratch_bytes': integer('private_segment_fixed_size')}


def inspect_symbol(code: Path, symbol: str, name: str) -> dict:
    opcode = SPECS[name][2]
    assembly = subprocess.run([str(LLVM / 'llvm-objdump'), '-d', '--mcpu=gfx1201',
        f'--disassemble-symbols={symbol}', str(code)], check=True, capture_output=True, text=True).stdout
    metadata = subprocess.run([str(LLVM / 'llvm-readelf'), '--notes', str(code)],
                              check=True, capture_output=True, text=True).stdout
    proof = {**symbol_proof(symbol, opcode, assembly, metadata), **_machine_interval(code, symbol)}
    if name == 'ordinary_fp8_qk':
        proof.update(fp8_wmma_opcode=opcode, fp8_wmma_sites=proof['opcode_sites'],
            packed_fp8_conversion_opcode='v_cvt_pk_fp8_f32',
            packed_fp8_conversion_sites=len(re.findall(r'\bv_cvt_pk_fp8_f32\b', assembly)))
    return proof


def prepare(selection: Path, out: Path) -> dict:
    if os.path.lexists(out):
        raise ValueError(f'refusing to overwrite {out}')
    route = selected_route(selection)
    executable = Path(route['executable']['path'])
    before = snapshot(executable)
    raw = executable.read_bytes()
    sparse = route['xattention_profile'] != 'dense'
    group = route['kv_value_group']
    profile = ('xattention-s16-tau900' if sparse else 'dense') + f'-g{group}'
    names = ['q4_p2048_cta', 'q4_wave32', 'ordinary_fp8_qk']
    if route['weights_id'] == 'r9700-q4-w8-mse-n16k16-eval':
        names.append('w8_p2048_cta')
    names += ['xattention_rank', 'xattention_flash_consumer'] if sparse else ['dense_panel_qk']
    proofs, embedded = {}, {}
    with tempfile.TemporaryDirectory(prefix='ninfer-selected-static-') as directory:
        for name in names:
            family, token, opcode = SPECS[name]
            if name == 'xattention_flash_consumer':
                token = f'xattention_flash_consumer_kernelILj{group}ELb0ELb0EE'
            symbol = unique_symbol(raw, token)
            code = Path(directory) / f'{name}.co'
            result = extract(executable, code, code_symbol=symbol)
            digest = result['code_object_sha256']
            existing = embedded.setdefault(family, {}).setdefault(profile, digest)
            if existing != digest:
                raise ValueError(f'selected {family} symbols do not share their owning code object')
            proofs[name] = inspect_symbol(code, symbol, name)
    if snapshot(executable) != before:
        raise ValueError('selected executable changed during static extraction')
    value = {'schema': AUDIT_SCHEMA, 'status': 'static_preflight_pass_terminal_dispatch_proof_pending',
        'scope': 'selected executable only; ISA presence, not execution or append-panel numerical proof',
        'benchmark_executables': [{'profile': profile, **before}],
        'embedded_code_objects': embedded, 'shared_symbol_proofs': proofs,
        'terminal_selection': route['terminal_selection'],
        'attention_scope': ('selected sparse rank/consumer ISA; real keep-distribution and dispatch evidence remain separate'
                            if sparse else 'BK32 QK implementation shared by initial and appended dense query panels; actual dispatch requires the separate trace'),
        'tools': {name: snapshot(LLVM / name) for name in ('llvm-objdump', 'llvm-readelf', 'llvm-nm')}}
    with out.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.selection.resolve(strict=True), args.out.absolute())


if __name__ == '__main__':
    main()
