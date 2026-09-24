#!/usr/bin/env python3
"""Recompute the normalized-linear direct gate from retained paired samples."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def reject_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(report):
    require(report['schema'] == 'ninfer.r9700.normalized-linear-t1-qualification.v1',
            'wrong qualification schema')
    require(report['shape'] == {'tokens': 1, 'rows': 34816, 'columns': 5120},
            'wrong decision point')
    correctness = report['correctness']
    require(correctness['criterion'] ==
            'normalized_linear_a8q4_fp64_rel_l2_1e-2_bf16_steps_2_v1',
            'wrong numerical criterion')
    require(0 <= correctness['maximum_bf16_steps'] <= 2, 'FP64 oracle failed')
    require(isinstance(correctness['maximum_relative_l2'], (float, int)) and
            not isinstance(correctness['maximum_relative_l2'], bool) and
            math.isfinite(correctness['maximum_relative_l2']) and
            0 <= correctness['maximum_relative_l2'] <= 1.0e-2,
            'FP64 normwise criterion failed')
    require(isinstance(correctness['maximum_absolute'], (float, int)) and
            not isinstance(correctness['maximum_absolute'], bool) and
            math.isfinite(correctness['maximum_absolute']) and
            correctness['maximum_absolute'] >= 0,
            'invalid gross pointwise diagnostic')
    for name in ('exact_codec', 'exact_public_control', 'canaries',
                 'malformed_alias_rejections', 't2_fallback', 'immutable_inputs'):
        require(correctness[name] is True, f'correctness gate failed: {name}')
    require(correctness['poison_stale_finite_replays'] == 3, 'replay gate incomplete')
    power = report['power_profile']
    require(power['pci'] == '0000:13:00.0', 'wrong PCI identity')
    for name in ('before_hip', 'after_correctness', 'after_timing'):
        require(power[name] == 'auto', f'power is not auto: {name}')
    timing = report['timing']
    require(timing['complete_boundary_graph'] is True, 'wrong timed boundary')
    require(timing['copies'] == 3 and timing['scrub_bytes'] == 83886080,
            'allocation or cache policy changed')
    samples = timing['samples']
    require(len(samples) == 24, 'expected 24 paired samples')
    controls, candidates, ratios = [], [], []
    per_copy = {i: {'control': [], 'candidate': [], 'control_first': 0}
                for i in range(3)}
    for index, sample in enumerate(samples):
        require(sample['copy'] == index % 3, 'allocation rotation differs')
        require(sample['control_first'] is ((index // 3) % 2 == 0),
                'arm order differs')
        control, candidate = sample['control_ms'], sample['candidate_ms']
        require(all(isinstance(v, (float, int)) and not isinstance(v, bool)
                    and math.isfinite(v) and v > 0 for v in (control, candidate)),
                'invalid event sample')
        controls.append(control)
        candidates.append(candidate)
        ratios.append(candidate / control)
        copy = per_copy[sample['copy']]
        copy['control'].append(control)
        copy['candidate'].append(candidate)
        copy['control_first'] += int(sample['control_first'])
    copy_ratios = []
    for copy in per_copy.values():
        require(len(copy['control']) == 8 and copy['control_first'] == 4,
                'allocation and arm order are not jointly balanced')
        copy_ratios.append(statistics.median(copy['candidate']) /
                           statistics.median(copy['control']))
    control = statistics.median(controls)
    candidate = statistics.median(candidates)
    saved = 64 * (control - candidate)
    upper = statistics.mean(ratios) + 2 * statistics.stdev(ratios) / math.sqrt(len(ratios))
    every_copy = all(ratio < 1 for ratio in copy_ratios)
    for name, calculated in (('control_median_ms', control),
                             ('candidate_median_ms', candidate),
                             ('saved_ms_per_token', saved),
                             ('ratio_mean_plus_2se', upper)):
        require(math.isclose(timing[name], calculated, rel_tol=1e-12, abs_tol=1e-12),
                f'reported {name} differs from raw samples')
    require(timing['every_copy_wins'] is every_copy, 'reported allocation result differs')
    passed = every_copy and upper < 1 and saved >= 0.2
    require(report['status'] == ('qualified_for_selector_free_production_confirmation' if passed else
                                 'direct_timing_rejected'), 'reported disposition differs')
    return {'status': 'passed' if passed else 'rejected',
            'control_median_ms': control, 'candidate_median_ms': candidate,
            'saved_ms_per_token': saved, 'ratio_mean_plus_2se': upper,
            'allocation_median_ratios': copy_ratios,
            'passing_authorizes': 'selector-free whole C1 production confirmation preparation only'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.report.read_text(), parse_constant=reject_constant))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
