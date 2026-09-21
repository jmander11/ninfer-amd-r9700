#!/usr/bin/env python3
"""Check T1 pipeline overlap, loop-carried dependencies, and terminal waits.

Static feasibility only: numerical qualification and boundary timing are required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

SYMBOL = 'a8q4_gate_up_prefetch_qualification_kernel'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def registers(operand):
    match = re.fullmatch(r'v\[(\d+):(\d+)\]', operand)
    if match:
        return list(range(int(match[1]), int(match[2]) + 1))
    match = re.fullmatch(r'v(\d+)(?:\.[hl])?', operand)
    return [int(match[1])] if match else []


def check(text):
    require('amdgcn-amd-amdhsa--gfx1201' in text, 'wrong target')
    match = re.search(r'^' + SYMBOL + r':.*?^\s*\.size\s+' + SYMBOL + r',',
                      text, re.M | re.S)
    require(match is not None, 'kernel missing')
    body = match[0]
    require(not re.search(r'\b(?:scratch_|flat_load|buffer_load|ds_load|ds_store)', body),
            'unexpected memory family')
    heads = re.findall(r'^(\.LBB\d+_\d+):.*Inner Loop Header', body, re.M)
    require(len(heads) == 1, 'expected one serial group loop')
    head = heads[0]
    start = body.index(head + ':')
    back = re.search(r'\bs_cbranch_scc0\s+' + re.escape(head) + r'\b', body[start:])
    require(back is not None, 'serial loop backedge missing')
    loop = body[start:start + back.end()]
    loads = list(re.finditer(r'^\s*global_load_b64\s+.*$', loop, re.M))
    require(len(loads) == 4, 'must issue four successor B64 loads')
    dots = list(re.finditer(r'^\s*v_dot8_i32_iu4\s+.*$', loop, re.M))
    require(len(dots) == 16 and loads[-1].end() < dots[0].start(),
            'successor B64 loads must precede all current dot8 work')
    fmas = list(re.finditer(r'^\s*v_fmac_f32_e32\s+.*$', loop, re.M))
    require(len(fmas) == 1 and dots[-1].end() < fmas[0].start(),
            'serial group FP32 FMA missing')
    overlap = loop[loads[0].start():fmas[0].end()]
    require(not re.search(r'\bs_wait_(?:loadcnt|dscnt)', overlap),
            'successor loads drained before current dot8/FMA completes')
    require(len(re.findall(r'neg_lo:\[0,1,0\]', loop)) == 8 and
            len(re.findall(r'neg_lo:\[1,1,0\]', loop)) == 8,
            'signed Q4 or split signed/unsigned A8 interpretation changed')
    require('s_cmp_eq_u32' in loop and '0x4f' in loop,
            'expected 79 pipelined iterations plus terminal group')
    terminal = body[start + back.end():]
    terminal_dots = list(re.finditer(r'^\s*v_dot8_i32_iu4\s+.*$', terminal, re.M))
    require(len(terminal_dots) == 16, 'terminal group must consume all eight Q4 words')
    require(len(re.findall(r'neg_lo:\[0,1,0\]', terminal)) == 8 and
            len(re.findall(r'neg_lo:\[1,1,0\]', terminal)) == 8,
            'terminal A8/Q4 interpretation changed')
    terminal_fmas = list(re.finditer(r'^\s*v_fmac_f32_e32\s+.*$', terminal, re.M))
    require(len(terminal_fmas) == 1 and terminal_dots[-1].end() < terminal_fmas[0].start(),
            'terminal serial FP32 FMA missing')
    activation_loads = list(re.finditer(r'^\s*s_load_b256\s+.*$', terminal, re.M))
    require(len(activation_loads) == 2 and activation_loads[-1].end() < terminal_dots[0].start(),
            'terminal A8 scalar loads missing')
    require(re.search(r'\bs_wait_kmcnt\s+0x0\b',
                      terminal[activation_loads[-1].end():terminal_dots[0].start()]),
            'terminal A8 scalar operands consumed without KMC wait')

    # Track VMEM requests in issue order and each weight register's generation.
    # A wait retires all but N requests. MOV is a real dependency: it must not
    # copy an unretired predecessor. Dot8 must consume g, never g+1.
    pending = []
    values = {}
    next_id = 0
    dot_generations = []

    def walk(segment, load_generation, expected_dot_generation=None):
        nonlocal next_id
        for raw in segment.splitlines():
            line = raw.strip().split(';')[0].strip()
            if not line or line.endswith(':'):
                continue
            wait = re.match(r's_wait_loadcnt\s+(0x[0-9a-f]+|\d+)$', line)
            if wait:
                keep = int(wait[1], 0)
                if len(pending) > keep:
                    del pending[:len(pending) - keep]
                continue
            require(not line.startswith('s_wait_loadcnt_'), 'unmodeled combined wait')
            for instruction in line.split(' :: '):
                parts = instruction.split(None, 1)
                if len(parts) != 2:
                    continue
                op, args = parts
                operands = [p.strip() for p in args.split(',')]
                destination = registers(operands[0])
                if op.startswith('global_load_'):
                    next_id += 1
                    pending.append(next_id)
                    for reg in destination:
                        # Track scales as well as weights so the terminal FMA
                        # cannot silently consume either unretired scale load.
                        values[reg] = (next_id, load_generation if op == 'global_load_b64' else None)
                    continue
                if op.startswith(('v_mov_b32', 'v_dual_mov_b32')):
                    source = registers(operands[1])
                    value = values.get(source[0]) if source else None
                    if value is not None:
                        require(value[0] not in pending, 'weight copied before its VMEM wait')
                    for reg in destination:
                        values[reg] = value
                elif op == 'v_dot8_i32_iu4':
                    weight = registers(operands[2])
                    require(len(weight) == 1, 'unmodeled weight operand')
                    value = values.get(weight[0])
                    require(value is not None and value[0] not in pending,
                            'weight consumed without a retired B64 dependency')
                    require(value[1] == expected_dot_generation,
                            'dot8 consumed wrong weight generation')
                    dot_generations.append(value[1])
                    for reg in destination:
                        values[reg] = None
                elif op.startswith('v_'):
                    for operand in operands[1:]:
                        for reg in registers(operand):
                            value = values.get(reg)
                            require(value is None or value[0] not in pending,
                                    'vector operand consumed before its VMEM wait')
                    for reg in destination:
                        values[reg] = None

    walk(body[:start], 0)
    walk(loop, 1, 0)
    walk(loop, 2, 1)
    require(dot_generations == [0] * 16 + [1] * 16,
            'incomplete two-iteration dependency evidence')
    # Two body iterations establish the same loop-carried register mapping as
    # the final iteration. Its still-pending successor stands for group 79.
    walk(terminal, None, 2)
    require(dot_generations == [0] * 16 + [1] * 16 + [2] * 16,
            'incomplete terminal dependency evidence')
    require(not pending, 'terminal group left unretired VMEM requests')

    resources = {}
    for key in ('group_segment_fixed_size', 'private_segment_fixed_size',
                'sgpr_spill_count', 'vgpr_spill_count', 'wavefront_size',
                'vgpr_count', 'sgpr_count', 'max_flat_workgroup_size'):
        field = re.search(r'\.' + key + r':\s*(\d+)', text)
        require(field is not None, f'missing {key}')
        resources[key] = int(field[1])
    for key in ('group_segment_fixed_size', 'private_segment_fixed_size',
                'sgpr_spill_count', 'vgpr_spill_count'):
        require(resources[key] == 0, f'nonzero {key}')
    require(resources['wavefront_size'] == 32 and resources['max_flat_workgroup_size'] == 256,
            'wave/workgroup changed')
    require(resources['vgpr_count'] <= 32 and resources['sgpr_count'] <= 64,
            'register footprint outside admitted occupancy screen')
    occupancy = re.search(r'; Occupancy:\s*(\d+)', text)
    require(occupancy is not None and int(occupancy[1]) == 16, 'occupancy is not 16')
    return {'status': 'passed_static_feasibility_only', 'resources': resources,
            'occupancy': 16, 'successor_b64_loads': 4, 'overlapped_dot8': 16,
            'overlapped_serial_fp32_fma': 1, 'dependency_iterations_checked': 2,
            'terminal_group_dependencies_checked': True,
            'requires': 'independent numerical qualification and complete-boundary timing'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assembly', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.assembly.read_text()), indent=2))


if __name__ == '__main__':
    main()
