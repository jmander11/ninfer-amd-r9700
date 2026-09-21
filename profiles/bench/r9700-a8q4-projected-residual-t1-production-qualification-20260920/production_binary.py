#!/usr/bin/env python3
"""Inspect the code object in the CMake-built qualifier without mutating it."""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path
from prepare import (ARCHIVE, BINARY, BUILD, ROOT, identity, open_exclusive, write_json)
sys.path.insert(0, str(ROOT / 'tools/bench'))
from extract_embedded_code_object import extract

def captured(command, path):
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    with open_exclusive(path) as stream:
        stream.write(result.stdout)
    return result.stdout

def main():
    directory = Path(sys.argv[1])
    before = identity(BINARY)
    assembly = (directory / 'qual.s').read_text()
    symbols = set(re.findall(r'^\s*\.globl\s+(\S*a8q4g64_projected_residual_t1_kernel\S*)\s+;',
                             assembly, re.MULTILINE))
    if len(symbols) != 1:
        raise RuntimeError('expected exact production assembly symbol')
    symbol = symbols.pop()
    code = directory / 'production.hsaco'
    extraction = extract(BINARY, code, code_symbol=symbol)
    write_json(directory / 'extraction.json', extraction)
    disassembly = captured(['/opt/rocm/llvm/bin/llvm-objdump', '-d', '--mcpu=gfx1201', str(code)],
                           directory / 'production.disassembly.txt')
    match = re.search(r'^\s*[0-9a-f]+ <' + re.escape(symbol) + r'>:\n(.*?)(?=^\s*[0-9a-f]+ <|\Z)',
                      disassembly, re.MULTILINE | re.DOTALL)
    if match is None or 'v_dot8_i32_iu4' not in match.group(1) or 'scratch_' in match.group(1):
        raise RuntimeError('embedded production kernel lacks IU4 or contains scratch traffic')
    notes = captured(['/opt/rocm/llvm/bin/llvm-readelf', '--notes', str(code)],
                     directory / 'production.notes.txt')
    blocks = [block for block in re.split(r'(?=^  - \.args:)', notes, flags=re.MULTILINE)
              if re.search(r'\.name:\s+' + re.escape(symbol) + r'\s*$', block, re.MULTILINE)]
    if len(blocks) != 1:
        raise RuntimeError('embedded kernel metadata missing or duplicated')
    metadata = blocks[0]
    values = {}
    for key, maximum in [('group_segment_fixed_size', 0), ('private_segment_fixed_size', 0),
                         ('sgpr_spill_count', 0), ('vgpr_spill_count', 0),
                         ('vgpr_count', 32), ('sgpr_count', 64), ('wavefront_size', 32)]:
        value = re.search(r'\.' + key + r':\s*(\d+)', metadata)
        if value is None:
            raise RuntimeError(f'embedded metadata missing {key}')
        values[key] = int(value.group(1))
        if values[key] > maximum or (key == 'wavefront_size' and values[key] != 32):
            raise RuntimeError(f'embedded production resource failure: {key}={values[key]}')
    host_symbols = captured(['/opt/rocm/llvm/bin/llvm-nm', '-C', '--defined-only', str(BINARY)],
                            directory / 'production.host-symbols.txt')
    for required in ('ninfer::ops::projected_residual_t1(',
                     'ninfer::ops::projected_residual_t1_workspace_capacity_bytes('):
        if required not in host_symbols:
            raise RuntimeError(f'public production symbol missing: {required}')
    commands = subprocess.run(['ninja', '-C', str(BUILD), '-t', 'commands',
                               'src/ninfer_r9700_a8q4_projected_residual_t1_qual'],
                              cwd=ROOT, check=True, capture_output=True, text=True).stdout
    links = [line for line in commands.splitlines()
             if '-o src/ninfer_r9700_a8q4_projected_residual_t1_qual ' in line]
    if len(links) != 1 or 'src/libninfer_r9700_core.a' not in links[0]:
        raise RuntimeError('qualifier CMake link does not own production archive')
    if identity(BINARY) != before:
        raise RuntimeError('production executable changed during inspection')
    write_json(directory / 'production-binary-receipt.json', {
        'status': 'passed', 'executable': before, 'archive': identity(ARCHIVE),
        'cache': identity(BUILD / 'CMakeCache.txt'), 'link_command': links[0],
        'embedded_code': identity(code), 'kernel_symbol': symbol,
        'native_iu4_dot8': True, 'resources': values,
        'source_assembly': identity(directory / 'qual.s')})
    print('production_binary: PASS public Op/capacity + ninfer_r9700_core + embedded gfx1201 IU4')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
