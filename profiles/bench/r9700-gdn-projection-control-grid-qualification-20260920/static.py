#!/usr/bin/env python3
"""Bind reviewed source schedule to all four kernels embedded in the executable."""
import collections
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools/r9700'))
sys.path.insert(0, str(ROOT/'tools/bench'))
from check_gdn_projection_control_grid_static import check, kernel, require
from check_a8q4_normalized_linear_t1_static import metadata
from extract_embedded_code_object import extract


def inspect(binary, assembly, output):
    candidate, linear, control = [p.read_text() for p in assembly]
    result = check(candidate, linear, control)
    output.mkdir()
    results = {}
    for text, name in ((candidate, 'gdn_projection_control_grid_qualification_kernel'),
                       (linear, 'a8q4g64_gdn_pair_t1_kernel'),
                       (linear, 'a8g64_quantize_activation_kernel'),
                       (control, 'bf16_projected_control_t1_kernel')):
        symbol, body, source_resources = kernel(text, name)
        code = output/f'{name}.hsaco'
        extraction = extract(binary, code, code_symbol=symbol)
        disassembly = subprocess.run(['/opt/rocm/llvm/bin/llvm-objdump', '-d',
            '--mcpu=gfx1201', str(code)], check=True, capture_output=True, text=True).stdout
        match = re.search(r'^\s*[0-9a-f]+ <'+re.escape(symbol)+
            r'>:\n(.*?)(?=^\s*[0-9a-f]+ <|\Z)', disassembly, re.M | re.S)
        require(match is not None, 'embedded body missing')
        embedded = match.group(1)
        # Complete ordered instruction-mnemonic schedule, including waits/barriers/loads.
        instructions = lambda value: re.findall(
            r'^\s*((?:s_|v_|global_|flat_|ds_|scratch_|buffer_)[a-z0-9_]+)\b', value, re.M)
        source_ops = instructions(body)
        embedded_ops = [op for op in instructions(embedded) if op != 's_code_end']
        require(source_ops == embedded_ops, f'{name}: embedded instruction schedule differs')
        notes = subprocess.run(['/opt/rocm/llvm/bin/llvm-readelf', '--notes', str(code)],
            check=True, capture_output=True, text=True).stdout
        blocks = [b for b in re.split(r'(?=^  - \.args:)', notes, flags=re.M)
            if re.search(r'\.name:\s*'+re.escape(symbol)+r'\s*$', b, re.M)]
        require(len(blocks) == 1, 'embedded metadata missing')
        resources = metadata(blocks[0], name)
        source_blocks = [b for b in re.split(r'(?=^  - \.args:)', text, flags=re.M)
            if re.search(r'\.name:\s*'+re.escape(symbol)+r'\s*$', b, re.M)]
        require(len(source_blocks) == 1, 'source metadata missing')
        require(resources == metadata(source_blocks[0], name), 'embedded/source metadata differ')
        results[name] = {'extraction': extraction, 'embedded_resources': resources,
            'ordered_instruction_schedule_exact': True,
            'instruction_counts': dict(collections.Counter(embedded_ops))}
    return {'status': 'passed', 'reviewed_static_schedule': result, 'kernels': results,
            'vector_load_issue_cost_requires_complete_boundary_timing': True}


if __name__ == '__main__':
    print(json.dumps(inspect(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:5]],
                             Path(sys.argv[5])), indent=2))
