#!/usr/bin/env python3
"""Recompute the complete-boundary prefetch gate from retained paired samples."""
import argparse
import json
import math
from pathlib import Path
import statistics


def require(value, message):
    if not value:
        raise ValueError(message)


def analyze(report):
    require(report['schema'] == 'ninfer.r9700.gate-up-prefetch-qualification.v1', 'wrong schema')
    require(report['shape'] == {'tokens': 1, 'rows': 34816, 'columns': 5120}, 'wrong shape')
    correctness = report['correctness']
    require(correctness['criterion'] == 'normalized_linear_a8q4_fp64_rel_l2_1e-2_bf16_steps_2_v1',
            'wrong numerical criterion')
    require(0 <= correctness['maximum_bf16_steps'] <= 2 and
            0 <= correctness['maximum_relative_l2'] <= 0.01 and
            math.isfinite(correctness['maximum_absolute']) and correctness['maximum_absolute'] >= 0,
            'oracle criterion failed')
    for name in ('exact_codec', 'exact_public_control', 'canaries', 'malformed_alias_rejections',
                 'immutable_inputs', 'immutable_weights'):
        require(correctness[name] is True, f'incomplete correctness gate: {name}')
    for name, expected in (('candidate_null_rejections', 8), ('poison_stale_finite_replays', 3),
                           ('qualified_copies', 3), ('finite_cases', 4)):
        require(correctness[name] == expected, f'incomplete correctness gate: {name}')
    require(report['power_profile'] == {'pci': '0000:13:00.0', 'before_hip': 'auto',
            'after_correctness': 'auto', 'after_timing': 'auto'}, 'wrong power identity')
    timing = report['timing']
    require(timing['complete_boundary_graph'] is True and
            timing['identical_production_preparation'] is True, 'wrong timed boundary')
    require(timing['copies'] == 3 and timing['scrub_bytes'] == 83886080, 'wrong cache policy')
    samples = timing['samples']
    require(len(samples) == 24, 'incomplete timing')
    control, candidate, ratios = [], [], []
    copies = {i: ([], []) for i in range(3)}
    for index, sample in enumerate(samples):
        require(sample['copy'] == index % 3 and sample['control_first'] is ((index // 3) % 2 == 0),
                'allocation/order imbalance')
        c, q = sample['control_ms'], sample['candidate_ms']
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and
                    math.isfinite(v) and v > 0 for v in (c, q)), 'invalid event sample')
        control.append(c); candidate.append(q); ratios.append(q / c)
        copies[index % 3][0].append(c); copies[index % 3][1].append(q)
    c, q = statistics.median(control), statistics.median(candidate)
    saving = 64 * (c - q)
    upper = statistics.mean(ratios) + 2 * statistics.stdev(ratios) / math.sqrt(24)
    copy_ratios = [statistics.median(qs) / statistics.median(cs) for cs, qs in copies.values()]
    every_copy = all(ratio < 1 for ratio in copy_ratios)
    for name, value in (('control_median_ms', c), ('candidate_median_ms', q),
                        ('saved_ms_per_token', saving), ('ratio_mean_plus_2se', upper)):
        require(math.isclose(timing[name], value, rel_tol=1e-12, abs_tol=1e-12), 'raw sample mismatch')
    require(timing['every_copy_wins'] is every_copy, 'allocation result mismatch')
    passed = every_copy and upper < 1 and saving >= 0.2
    require(report['status'] == ('qualified_for_whole_ab_only' if passed else 'direct_timing_rejected'),
            'disposition mismatch')
    return {'status': 'passed' if passed else 'rejected', 'control_median_ms': c,
            'candidate_median_ms': q, 'saved_ms_per_token': saving,
            'ratio_mean_plus_2se': upper, 'allocation_median_ratios': copy_ratios,
            'passing_authorizes': 'independent result review then whole C1 A/B preparation only'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.report.read_text()))
    print(json.dumps(result, indent=2, allow_nan=False))
    # Both dispositions are valid analyzed measurements; malformed evidence raises.
    raise SystemExit(0)
