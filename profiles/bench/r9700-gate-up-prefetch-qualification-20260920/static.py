#!/usr/bin/env python3
"""Bind the reviewed load/compute schedule to exact embedded instruction bytes."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/r9700'))
sys.path.insert(0, str(ROOT / 'tools/bench'))
import check_a8q4_gate_up_prefetch_static as challenger
import check_a8q4_normalized_linear_t1_static as normalized
from extract_embedded_code_object import extract


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, capture_output=True, **kwargs).stdout


def instruction_bytes(path, symbol):
    text = run(['/opt/rocm/llvm/bin/llvm-objdump', '-d', '--mcpu=gfx1201', str(path)])
    body = re.search(r'^\s*[0-9a-f]+ <' + re.escape(symbol) +
                     r'>:\n(.*?)(?=^\s*[0-9a-f]+ <|\Z)', text, re.M | re.S)
    challenger.require(body is not None, 'embedded symbol missing')
    words = re.findall(r'//\s*[0-9A-Fa-f]+:\s*((?:[0-9A-Fa-f]{8}(?:\s+|$))+)', body[1])
    challenger.require(len(words) > 100, 'incomplete encoded instruction extraction')
    return ''.join(''.join(word.split()) for word in words).lower()


def check(assembly, binary, directory):
    directory.mkdir()
    result = challenger.check(assembly.read_text())
    static_object = directory / 'checked-assembly.o'
    run(['/opt/rocm/llvm/bin/clang', '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx1201',
         '-c', str(assembly), '-o', str(static_object)])
    code = directory / 'candidate.hsaco'
    result['extraction'] = extract(binary, code, code_symbol=challenger.SYMBOL)
    expected = instruction_bytes(static_object, challenger.SYMBOL)
    actual = instruction_bytes(code, challenger.SYMBOL)
    challenger.require(expected == actual, 'embedded instruction bytes differ from checked schedule')
    result['exact_embedded_instruction_bytes'] = True
    # Recheck the actual linked incumbent and normalization preparation as well.
    build = ROOT / 'build-r9700'
    source = ROOT / 'src/ops/r9700/linear/r9700_linear.hip'
    entries = [entry for entry in json.loads((build / 'compile_commands.json').read_text())
               if Path(entry['file']) == source]
    challenger.require(len(entries) == 1, 'ambiguous production compilation')
    entry = entries[0]
    argv = entry.get('arguments') or shlex.split(entry['command'])
    challenger.require('--offload-arch=gfx1201' in argv and
                        '-DNINFER_R9700_Q4_ACTIVATION_BITS=8' in argv, 'wrong production profile')
    output = argv.index('-o')
    argv = argv[:output] + argv[output + 2:]
    argv.remove('-c')
    production_assembly = directory / 'production.s'
    run([*argv, '--offload-device-only', '-S', '-o', str(production_assembly)], cwd=entry['directory'])
    incumbent = normalized.source_check(production_assembly)
    normalized.embedded_check(binary, directory / 'production', incumbent)
    return {'status': 'passed', 'candidate': result, 'production': incumbent}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('assembly', type=Path)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--embedded-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.assembly, args.binary, args.embedded_dir), indent=2))
