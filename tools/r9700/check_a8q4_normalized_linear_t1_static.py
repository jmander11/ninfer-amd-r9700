#!/usr/bin/env python3
"""Check the fused RMS preparation and its unchanged IU4 projection consumer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from extract_embedded_code_object import extract

KERNELS = ('a8g64_normalized_prepare_t1_kernel',
           'a8q4g64_linear_decode_dot8_t1_kernel')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def metadata(block, label):
    values = {}
    for key in ('group_segment_fixed_size', 'private_segment_fixed_size',
                'sgpr_spill_count', 'vgpr_spill_count', 'wavefront_size',
                'vgpr_count', 'sgpr_count'):
        match = re.search(r'\.' + key + r':\s*(\d+)', block)
        require(match is not None, f'{label}: missing {key}')
        values[key] = int(match.group(1))
    require(values['wavefront_size'] == 32, f'{label}: not wave32')
    for key in ('private_segment_fixed_size', 'sgpr_spill_count', 'vgpr_spill_count'):
        require(values[key] == 0, f'{label}: nonzero {key}')
    # RMSNorm legitimately owns a shared reduction. Do not impose the consumer's
    # zero-LDS constraint on the preparation kernel.
    if KERNELS[1] in label:
        require(values['group_segment_fixed_size'] == 0, f'{label}: consumer uses LDS')
        require(values['vgpr_count'] <= 32 and values['sgpr_count'] <= 64,
                f'{label}: consumer resource profile regressed')
    return values


def source_check(assembly):
    text = assembly.read_text()
    result = {}
    for name in KERNELS:
        symbols = set(re.findall(r'^\s*\.globl\s+(\S*' + re.escape(name) + r'\S*)\s+;',
                                 text, re.MULTILINE))
        require(len(symbols) == 1, f'{name}: missing or ambiguous symbol')
        symbol = symbols.pop()
        require(symbol.startswith('_ZN6ninfer3ops5r97006linear'), f'{name}: wrong owner')
        match = re.search(r'^\s*\.globl\s+' + re.escape(symbol) +
                          r'.*?^\s*\.size\s+' + re.escape(symbol) + ',',
                          text, re.MULTILINE | re.DOTALL)
        require(match is not None, f'{name}: body missing')
        body = match.group(0)
        require('scratch_' not in body, f'{name}: scratch instructions')
        if name == KERNELS[1]:
            require('v_dot8_i32_iu4' in body, f'{name}: native IU4 missing')
        blocks = [b for b in re.findall(r'^  - \.args:.*?(?=^  - \.args:|^\.\.\.)',
                                        text, re.MULTILINE | re.DOTALL)
                  if re.search(r'\.name:\s+' + re.escape(symbol) + r'\s*$', b, re.MULTILINE)]
        require(len(blocks) == 1, f'{name}: metadata missing or ambiguous')
        result[name] = {'symbol': symbol, 'source_resources': metadata(blocks[0], name)}
    return result


def embedded_check(binary, directory, result):
    directory.mkdir()
    symbols = subprocess.run(['/opt/rocm/llvm/bin/llvm-nm', '-C', '--defined-only', str(binary)],
                             check=True, text=True, capture_output=True).stdout
    for name in ('ninfer::ops::normalized_linear(',
                 'ninfer::ops::normalized_linear_workspace_capacity_bytes(',
                 'ninfer::ops::rmsnorm(', 'ninfer::ops::linear('):
        require(name in symbols, f'linked public symbol missing: {name}')
    for name, kernel in result.items():
        code = directory / f'{name}.hsaco'
        kernel['extraction'] = extract(binary, code, code_symbol=kernel['symbol'])
        disassembly = subprocess.run(['/opt/rocm/llvm/bin/llvm-objdump', '-d', '--mcpu=gfx1201', str(code)],
                                     check=True, text=True, capture_output=True).stdout
        symbol = re.escape(kernel['symbol'])
        body = re.search(r'^\s*[0-9a-f]+ <' + symbol + r'>:\n(.*?)(?=^\s*[0-9a-f]+ <|\Z)',
                         disassembly, re.MULTILINE | re.DOTALL)
        require(body is not None and 'scratch_' not in body.group(1),
                f'{name}: embedded body missing or has scratch')
        if name == KERNELS[1]:
            require('v_dot8_i32_iu4' in body.group(1), f'{name}: embedded native IU4 missing')
        notes = subprocess.run(['/opt/rocm/llvm/bin/llvm-readelf', '--notes', str(code)],
                               check=True, text=True, capture_output=True).stdout
        blocks = [b for b in re.split(r'(?=^  - \.args:)', notes, flags=re.MULTILINE)
                  if re.search(r'\.name:\s+' + symbol + r'\s*$', b, re.MULTILINE)]
        require(len(blocks) == 1, f'{name}: embedded metadata missing or ambiguous')
        kernel['embedded_resources'] = metadata(blocks[0], name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('assembly', type=Path)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--embedded-dir', type=Path)
    args = parser.parse_args()
    require(bool(args.binary) == bool(args.embedded_dir), 'binary and embedded directory are a pair')
    result = source_check(args.assembly)
    if args.binary:
        embedded_check(args.binary, args.embedded_dir, result)
    print(json.dumps({'status': 'passed', 'kernels': result}, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
