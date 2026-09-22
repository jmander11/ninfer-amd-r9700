#!/usr/bin/env python3
"""Compare the fixed selective-protected experiment with retained dense/G16 PPL."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.ppl import run, compare_bf16_repeats
from tools.ppl.assemble_pareto import _replay_campaign_quality

BASELINES = REPO / 'profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921'
IDENTITIES = {
    'all_q4': 'r9700-q4g64-n16k16-eval',
    'four_role': 'r9700-q4g64-f8e4m3-four-role-n16k16-eval',
    'selective_protected': 'r9700-q4-selective-protected-n16k16-eval',
}
LENGTHS = (8192, 32768)


def quality(nll, argmax, ref_nll, ref_argmax):
    paired = run.paired_nll_stats(ref_nll, nll)
    if paired is None or len(nll) != len(argmax) or len(ref_nll) != len(ref_argmax):
        raise ValueError('incomplete paired sidecars')
    tiers = {}
    for tier, limits in run.QUALITY_TIERS.items():
        severe = run.severe_position_stats(nll, ref_nll, threshold=run.TERRIBLE_NLL,
            maximum_new_rate=limits['maximum_new_severe_rate'],
            minimum_budget=limits['minimum_new_severe_budget'])
        tiers[tier] = {
            'maximum_mean_nll_delta': limits['maximum_mean_nll_delta'],
            'new_severe_positions': severe['new_severe_positions'],
            'new_severe_position_budget': severe['new_severe_position_budget'],
            'pass': severe['new_severe_positions_pass'] and
                    paired['mean_delta_nll'] <= limits['maximum_mean_nll_delta'],
        }
    mean = math.fsum(nll) / len(nll)
    return {'tokens_scored': len(nll), 'mean_nll': mean, 'ppl': math.exp(mean),
            'delta_mean_nll_vs_bf16': paired['mean_delta_nll'],
            'ppl_ratio_vs_bf16': math.exp(paired['mean_delta_nll']),
            'paired_delta_standard_error': run.paired_delta_se(nll, ref_nll),
            'bf16_greedy_diagnostic': run.exact_argmax_stats(argmax, ref_argmax),
            'quality_tiers': tiers}


def validate_command(command, campaign, reference):
    for flag, expected in (('--ids', reference['corpus']['path']),
                           ('--weights', campaign['candidate_artifact']['path'])):
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            raise ValueError(f'missing/duplicate command binding: {flag}')
        if Path(command[command.index(flag) + 1]).resolve() != Path(expected).resolve():
            raise ValueError(f'command binding differs: {flag}')
    if not command or Path(command[0]).resolve() != Path(campaign['scorers']['r9700-g16']['path']).resolve():
        raise ValueError('command scorer differs')


def load_candidate(path, name, reference):
    campaign = json.loads(path.read_text())
    weights_id = IDENTITIES[name]
    if campaign.get('candidate_artifact', {}).get('weights_id') != weights_id:
        raise ValueError(f'{name}: wrong weight identity')
    if name != 'selective_protected':
        _replay_campaign_quality(campaign, weights_id)
    for field in ('model_id', 'corpus', 'reference_source', 'reference_execution',
                  'lengths', 'skip', 'prefill_chunk', 'schedules', 'spec', 'draft_tokens'):
        if campaign.get(field) != reference.get(field):
            raise ValueError(f'{name}: reference alignment differs at {field}')
    if campaign.get('xattention_profile') != 'dense':
        raise ValueError(f'{name}: not dense attention')
    cells = [c for c in campaign['cells'] if c['scheme'] == 'r9700-g16']
    if len(cells) != 2 or {c['prompt_tokens'] for c in cells} != set(LENGTHS):
        raise ValueError(f'{name}: incomplete/duplicate G16 cells')
    out = {}
    for cell in cells:
        tokens = cell['prompt_tokens']
        command = cell['command']
        validate_command(command, campaign, reference)
        if command.count('--out-json') != 1:
            raise ValueError('missing raw cell identity')
        raw_path = Path(command[command.index('--out-json') + 1])
        raw = json.loads(raw_path.read_text())
        if any(cell.get(k) != v for k, v in raw.items()):
            raise ValueError('raw cell differs from campaign')
        run.validate_cell_report(raw, profile_name='r9700-g16',
            weights=Path(campaign['candidate_artifact']['path']),
            ids=Path(reference['corpus']['path']), expected_weights_id=weights_id,
            schedule='prefill', skip='half', tokens=tokens, prefill_chunk=2048, extra=[],
            expected_q4_activation_bits=8, expected_w8_activation_bits=8,
            expected_fp8_qk_wmma=True, expected_xattention_profile='dense')
        for suffix, key in (('nllf32', 'nll_sha256'), ('argmaxi32', 'argmax_sha256')):
            if run.file_sha256(raw_path.with_suffix('.' + suffix)) != cell[key]:
                raise ValueError('changed sidecar')
        nll, argmax = run.load_nlls(raw_path), run.load_argmax(raw_path)
        if not run.cell_ok(raw, nll, argmax):
            raise ValueError('invalid candidate sidecars')
        compare_bf16_repeats._validate_sidecar_content(name, tokens, raw, raw_path)
        out[tokens] = (raw_path, nll, argmax)
    return campaign, out


def analyze(candidate_path):
    ref_path = BASELINES / 'bf16-chunk2048-a/results.json'
    reference, ref_cells = compare_bf16_repeats.load_campaign(ref_path)
    run.validate_bf16_repeat_comparison(BASELINES / 'bf16-chunk2048-repeat.json', ref_path)
    if (reference['lengths'] != list(LENGTHS) or reference['prefill_chunk'] != 2048
            or reference['skip'] != 'half' or reference['schedules'] != ['prefill']):
        raise ValueError('reference workload is not the declared experiment')
    sources = {
        'all_q4': BASELINES / 'all_q4_dense_quality/results.json',
        'four_role': BASELINES / 'four_role_dense_quality/results.json',
        'selective_protected': candidate_path.resolve(),
    }
    report = {'status': 'completed', 'scope': 'dense_g16_prefill_ppl_chunk2048_skip_half',
              'production_selection': False, 'reference': str(ref_path),
              'reference_sha256': run.file_sha256(ref_path), 'candidates': {}, 'paired': {}}
    loaded = {}
    for name, path in sources.items():
        campaign, loaded[name] = load_candidate(path, name, reference)
        rows = {}
        for tokens, (raw_path, nll, argmax) in loaded[name].items():
            _, bf_path = ref_cells[tokens]
            rows[str(tokens)] = {'raw_report': str(raw_path), **quality(nll, argmax,
                run.load_nlls(bf_path), run.load_argmax(bf_path))}
        report['candidates'][name] = {'campaign': str(path),
            'campaign_sha256': run.file_sha256(path),
            'artifact': campaign['candidate_artifact'], 'results': rows}
    for name in ('all_q4', 'four_role'):
        report['paired']['selective_minus_' + name] = {
            str(tokens): run.paired_nll_stats(loaded[name][tokens][1],
                                            loaded['selective_protected'][tokens][1])
            for tokens in LENGTHS}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.candidate)
    with args.out.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    for name, candidate in result['candidates'].items():
        print(name, {n: {'ppl': row['ppl'], 'delta_nll': row['delta_mean_nll_vs_bf16'],
                        'tiers': row['quality_tiers']} for n, row in candidate['results'].items()})


if __name__ == '__main__':
    main()
