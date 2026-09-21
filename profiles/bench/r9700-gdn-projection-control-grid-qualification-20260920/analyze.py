#!/usr/bin/env python3
"""Independent admission from complete-boundary paired event samples."""
import json
import math
from pathlib import Path
import statistics
import sys


def require(ok, message):
    if not ok:
        raise ValueError(message)


def analyze(report):
    require(report['schema'] == 'ninfer.r9700.gdn-projection-control-grid.v1', 'schema')
    require(report['status'] == 'qualified', 'numerical qualification failed')
    require(0 <= report['projection_max_relative_l2'] <= 0.01 and
            0 <= report['projection_max_bf16_steps'] <= 2, 'projection oracle')
    require(0 <= report['control_g_max_absolute_error'] <= 0.0625 and
            0 <= report['control_beta_max_absolute_error'] <= 0.0003, 'control oracle')
    for key in ('codec_exact', 'incumbent_exact', 'graph_poison_stale_finite',
                'guards_and_immutability'):
        require(report[key] is True, key)
    require(report['malformed_cases'] >= 14, 'malformed coverage')
    require(report['pci'] == '0000:13:00.0' and report['power'] == 'auto', 'physical profile')
    require(report['copies'] == 3 and report['scrub_bytes'] == 83886080 and
            report['calls_per_token'] == 48, 'timing geometry')
    samples = report['samples']
    require(len(samples) == 24, '24 pairs required')
    controls, candidates, ratios = [], [], []
    for i, sample in enumerate(samples):
        require(sample['allocation'] == i % 3 and
                sample['control_first'] is ((i // 3) % 2 == 0), 'unbalanced ordering')
        c, r = sample['control_ms'], sample['candidate_ms']
        require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and
                    math.isfinite(x) and x > 0 for x in (c, r)), 'invalid timing')
        controls.append(c)
        candidates.append(r)
        ratios.append(r/c)
    control = statistics.median(controls)
    candidate = statistics.median(candidates)
    allocation_ratios = [statistics.median(candidates[i::3]) /
                         statistics.median(controls[i::3]) for i in range(3)]
    upper = statistics.mean(ratios) + 2*statistics.stdev(ratios)/math.sqrt(len(ratios))
    saving = 48*(control-candidate)
    admitted = saving >= 0.2 and upper < 1 and all(r < 1 for r in allocation_ratios)
    return {'status': 'completed', 'disposition': 'admitted' if admitted else 'rejected',
            'production_routing_authorized': False, 'whole_ab_preparation_authorized': admitted,
            'control_median_ms': control, 'candidate_median_ms': candidate,
            'saved_ms_per_token_48_layers': saving, 'ratio_mean_plus_2se': upper,
            'allocation_median_ratios': allocation_ratios,
            'minimum_saved_ms_per_token': 0.2}


if __name__ == '__main__':
    print(json.dumps(analyze(json.loads(Path(sys.argv[1]).read_text())), indent=2, allow_nan=False))
