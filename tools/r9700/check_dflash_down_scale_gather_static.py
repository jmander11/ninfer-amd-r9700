#!/usr/bin/env python3
"""Bind scale-gather mechanism, unchanged IU4/FMA order and embedded resources."""
import argparse
import collections
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/bench'))
from extract_embedded_code_object import extract
from check_gdn_projection_control_grid_static import kernel, require
from check_a8q4_normalized_linear_t1_static import metadata


def inspect(candidate, production, binary=None, directory=None):
    c, p = candidate.read_text(), production.read_text()
    require('amdgcn-amd-amdhsa--gfx1201' in c, 'wrong architecture')
    entries = []
    for t in (5, 6):
        fragment = f'dflash_down_scale_gather_kernelILj{t}EE'
        symbol, body, resources = kernel(c, fragment)
        require(body.count('v_wmma_i32_16x16x32_iu4') == 4, 'four native IU4 instructions required')
        require(body.count('neg_lo:[0,1,0]') == 2 and body.count('neg_lo:[1,1,0]') == 2,
                'IU4 signed decomposition changed')
        require(len(re.findall(r'^\s*global_load_d16_b16\b', body, re.M)) == 2,
                'must have one weight-scale and one activation-scale load')
        require(len(re.findall(r'^\s*ds_bpermute_b32\b', body, re.M)) == t,
                'one broadcast per live token required')
        require(len(re.findall(r'\bv_(?:dual_)?fmac_f32(?:_e32)?\b', body)) == t,
                'must retain one FP32 accumulated FMA per output per group')
        require(len(re.findall(r'^\s*v_fma_mix_f32\b', body, re.M)) == t,
                'scale product must remain separate from accumulated FMA')
        require(re.search(r's_cmp_eq_u32\s+(s\d+), 0x110', body) is not None,
                '272-group ordered loop missing')
        require(resources['group_segment_fixed_size'] == 0 and resources['next_free_vgpr'] <= 64 and
                resources['compiler_occupancy'] == 16, 'resource envelope regressed')
        require(not re.search(r'\bds_(?:load|store)|\bs_barrier', body), 'unexpected LDS/barrier')
        entries.append((fragment, c, symbol, body, resources))
    for fragment in ('a8q4g64_linear_wmma32_kernel', 'a8g64_quantize_activation_kernel'):
        symbol, body, resources = kernel(p, fragment)
        entries.append((fragment, p, symbol, body, resources))
    result = {}
    if binary:
        require(directory is not None, 'embedded directory missing')
        directory.mkdir()
    for name, text, symbol, body, resources in entries:
        facts = {'source_resources': resources, 'symbol': symbol}
        if binary:
            code = directory/f'{name}.hsaco'
            facts['extraction'] = extract(binary, code, code_symbol=symbol)
            assembly = subprocess.run(['/opt/rocm/llvm/bin/llvm-objdump', '-d', '--mcpu=gfx1201', str(code)],
                                       check=True, capture_output=True, text=True).stdout
            match = re.search(r'^\s*[0-9a-f]+ <'+re.escape(symbol)+r'>:\n(.*?)(?=^\s*[0-9a-f]+ <|\Z)',
                              assembly, re.M | re.S)
            require(match is not None, 'missing embedded body')
            # Preserve both instructions in dual-issue lines; exclude alignment padding.
            ops = lambda s: [op for op in re.findall(r'\b((?:s_|v_|global_|flat_|ds_|scratch_|buffer_)[a-z0-9_]+)\b', s)
                             if op != 's_code_end']
            require(ops(body) == ops(match.group(1)), 'embedded ordered instruction sequence changed')
            notes = subprocess.run(['/opt/rocm/llvm/bin/llvm-readelf', '--notes', str(code)],
                                    check=True, capture_output=True, text=True).stdout
            def fields(value):
                blocks = [b for b in re.split(r'(?=^  - \.args:)', value, flags=re.M)
                          if re.search(r'\.name:\s*'+re.escape(symbol)+r'\s*$', b, re.M)]
                require(len(blocks) == 1, 'metadata missing')
                return metadata(blocks[0], name)
            facts['embedded_resources'] = fields(notes)
            require(facts['embedded_resources'] == fields(text), 'embedded resources changed')
            facts['instruction_counts'] = dict(collections.Counter(ops(body)))
            facts['embedded_schedule_exact'] = True
        result[name] = facts
    return {'status': 'passed', 'kernels': result,
            'mechanism': 'one gathered activation-scale load plus T wave broadcasts per ordered G64 group',
            'broadcast_ds_wait_cost_included_in_timing': True,
            'production_routing_authorized': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('production', type=Path)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--embedded-dir', type=Path)
    a = parser.parse_args()
    require(bool(a.binary) == bool(a.embedded_dir), 'binary/directory pair required')
    print(json.dumps(inspect(a.candidate, a.production, a.binary, a.embedded_dir), indent=2))
