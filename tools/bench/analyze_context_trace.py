#!/usr/bin/env python3
"""Attribute one C1 selected-region prefill+decode trace, dense or XAttention.

Kernel service durations are attribution, not unprofiled throughput or counter evidence.
Chunk windows run from each serial chunk start to the next; the final window ends
at the first graph dispatch (and can include tail/setup). Do not use for C>1,
multiple measured requests/repetitions, eager decode, or overlapping streams.
"""
import argparse
import csv
import json
from pathlib import Path
import sqlite3


def families(rows):
    groups = {}
    for row in rows:
        name = row['display_name']
        group = groups.setdefault(name, dict(kernel=name, ms=0, calls=0, launch_profiles={}))
        duration = (row['end'] - row['start']) / 1e6
        group['ms'] += duration
        group['calls'] += 1
        blocks = 1
        for axis in ('x', 'y', 'z'):
            blocks *= (row['grid_size_' + axis] + row['workgroup_size_' + axis] - 1) // row['workgroup_size_' + axis]
        key = (blocks, row['arch_vgpr_count'], row['group_segment_size'], row['private_segment_size'])
        launch = group['launch_profiles'].setdefault(key, dict(blocks=blocks, vgpr=key[1],
                    lds_bytes=key[2], scratch_bytes=key[3], calls=0, ms=0))
        launch['calls'] += 1
        launch['ms'] += duration
    total = sum(g['ms'] for g in groups.values())
    result = sorted(groups.values(), key=lambda g: -g['ms'])
    for group in result:
        group['share'] = group['ms'] / total
        group['launch_profiles'] = list(group['launch_profiles'].values())
    return dict(kernel_ms=total, families=result)


def analyze(database, markers):
    with sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute('''select d.*, s.display_name, s.arch_vgpr_count
            from rocpd_kernel_dispatch d join rocpd_info_kernel_symbol s
            on d.kernel_id=s.id and d.guid=s.guid''').fetchall()
    with markers.open() as stream:
        ranges = list(csv.DictReader(stream))
    measured = [r for r in ranges if r['Function'].startswith('ninfer_bench_measured')]
    if len(measured) != 1:
        raise ValueError('expected exactly one measured region')
    if len({(r['pid'], r['agent_id'], r['queue_id']) for r in rows}) != 1:
        raise ValueError('expected one process/device/queue for serial attribution')
    chunks = sorted((r for r in ranges if r['Function'].startswith('ninfer.prefill.prefill.chunk ')),
                    key=lambda r: int(r['Start_Timestamp']))
    graphed = [r for r in rows if r['graph_exec_id'] != 0]
    if not chunks or not graphed:
        raise ValueError('expected prefill markers followed by graph decode; use --whole-pg P,16')
    first_graph = min(r['start'] for r in graphed)
    if first_graph <= int(chunks[-1]['Start_Timestamp']):
        raise ValueError('graph work overlaps prefill; not the supported single-request trace')
    result = dict(database=str(database.resolve()), markers=str(markers.resolve()),
                  scope='profiled kernel service only; not throughput, bandwidth or stall counters',
                  chunk_window='chunk start to next start; final ends at first graph dispatch, may include setup',
                  prefill_setup=families([r for r in rows if r['graph_exec_id'] == 0]),
                  decode_graph=families(graphed), chunks=[])
    for index, chunk in enumerate(chunks):
        start = int(chunk['Start_Timestamp'])
        end = int(chunks[index+1]['Start_Timestamp']) if index+1 < len(chunks) else first_graph
        result['chunks'].append(dict(tokens=int(chunk['Function'].split('payload=')[1]),
            **families([r for r in rows if r['graph_exec_id'] == 0 and start <= r['start'] < end])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--markers', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.database, args.markers)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for phase in ('prefill_setup', 'decode_graph'):
        print(phase, result[phase]['kernel_ms'], 'ms')
        for group in result[phase]['families'][:6]:
            print(f"  {100*group['share']:.2f}% {group['ms']:.2f}ms {group['kernel'][:130]}")


if __name__ == '__main__':
    main()
