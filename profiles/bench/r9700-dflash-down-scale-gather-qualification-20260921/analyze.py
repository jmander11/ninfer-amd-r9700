#!/usr/bin/env python3
"""Independent finite-cell timing admission; valid losers are completed/rejected."""
import json
import math
from pathlib import Path
import statistics
import sys

ROUND_MS = {5: 92.12005556976744, 6: 93.21258994047619}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def analyze(report):
    require(report['schema'] == 'ninfer.r9700.dflash-down-scale-gather.v1' and
            report['status'] == 'qualified', 'wrong report')
    require(report['pci'] == '0000:13:00.0' and report['power'] == 'auto', 'wrong device/profile')
    require(report['copies'] == 3 and report['scrub_bytes'] == 83886080 and
            report['complete_boundary_graph'] is True, 'timing boundary')
    require([c['tokens'] for c in report['cells']] == [5, 6], 'exact two widths required')
    cells = []
    for cell in report['cells']:
        t = cell['tokens']
        require(cell['rows'] == 5120 and cell['columns'] == 17408, 'wrong shape')
        numeric = cell['correctness']
        require(numeric['criterion'] == 'fp64_rel_l2_1e-2_gross_1e-2_refmax_plus_1e-5', 'criterion')
        require(0 <= numeric['maximum_relative_l2'] <= 0.01 and
                math.isfinite(numeric['maximum_absolute']) and numeric['maximum_absolute'] >= 0 and
                0 <= numeric['maximum_gross_cap_fraction'] <= 1, 'oracle')
        for key in ('exact_codec', 'exact_incumbent', 'graph_poison_stale_finite', 'guards_immutability'):
            require(numeric[key] is True, key)
        require(numeric['finite_cases'] == 5 and numeric['malformed_cases'] >= 19, 'coverage')
        require(len(cell['samples']) == 24, 'paired sample count')
        controls, candidates, ratios, savings = [], [], [], []
        for i, sample in enumerate(cell['samples']):
            require(sample['allocation'] == i % 3 and sample['control_first'] is ((i//3) % 2 == 0), 'balance')
            c, r = sample['control_ms'], sample['candidate_ms']
            require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and
                        math.isfinite(v) and v > 0 for v in (c, r)), 'event sample')
            controls.append(c); candidates.append(r); ratios.append(r/c); savings.append(c-r)
        upper = statistics.mean(ratios)+2*statistics.stdev(ratios)/math.sqrt(24)
        lower_saving = statistics.mean(savings)-2*statistics.stdev(savings)/math.sqrt(24)
        allocation_ratios = [statistics.median(candidates[i::3])/statistics.median(controls[i::3]) for i in range(3)]
        round_saving = 64*lower_saving
        admitted = upper < 1 and all(r < 1 for r in allocation_ratios) and round_saving >= 0.01*ROUND_MS[t]
        cells.append({'tokens': t, 'disposition': 'admitted' if admitted else 'rejected',
                      'control_median_ms': statistics.median(controls),
                      'candidate_median_ms': statistics.median(candidates),
                      'ratio_mean_plus_2se': upper, 'allocation_median_ratios': allocation_ratios,
                      'round_saving_lower_ms': round_saving,
                      'minimum_round_saving_ms': 0.01*ROUND_MS[t],
                      'retained_control_round_ms': ROUND_MS[t]})
    admitted = all(c['disposition'] == 'admitted' for c in cells)
    return {'status': 'completed', 'disposition': 'admitted' if admitted else 'rejected',
            'cells': cells, 'production_routing_authorized': False,
            'whole_ab_preparation_authorized': admitted,
            'limitation': '64-call saved-time projection uses historical matched whole controls; no current whole throughput claim'}


if __name__ == '__main__':
    print(json.dumps(analyze(json.loads(Path(sys.argv[1]).read_text())), indent=2, allow_nan=False))
