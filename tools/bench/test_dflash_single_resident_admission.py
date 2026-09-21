"""Fixed-resident admission over recomputed evaluation, never per-C recipe mixing."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.bench import assemble_dflash_selection as selection
from tools.bench import validate_final_cutover_admission as cutover


class Campaign:
    """Disk-bound fixture at the existing numerical/capacity validator boundaries.

    assemble, evidence-path reconstruction, matched speed math, admission and the
    final-cutover consumer are real. Large artifact/GPU report validators are stubbed.
    """
    def __init__(self, root, *, missing_c=None, slow_c=None, failed_screen=False):
        self.root = root
        self.chosen = selection.RECIPES[0]
        self.base = self.write(root / 'base.json', {})
        self.bench = {'path': '/accounted/bench/ninfer_bench', 'sha256': 'a' * 64,
                      'file_size_bytes': 100}
        self.route = {'terminal_selection': selection._identity(self.base), 'winner': 'base-winner',
                      'base_artifact': {'path': '/base.ninfer', 'sha256': 'b' * 64, 'file_size_bytes': 200,
                                        'weights_id': 'r9700-q4g64-n16k16-eval'},
                      'base_benchmark': self.bench, 'cache_group': 16,
                      'text_prefill_attention_profile': 'dense', 'selected_prefill_chunk': 2048,
                      'prefill_chunk_authority': {}, 'hybrid_base_authority': None}
        self.recipes, self.capacities, self.screens, self.followups = [], [], [], []
        for recipe in selection.RECIPES:
            parent = root / recipe
            conversion = self.write(parent / 'conversion.json', {'recipe': recipe})
            shortlist = self.write(parent / 'shortlist/dflash-shortlist.json', {'recipe': recipe})
            self.recipes.append((recipe, conversion, shortlist.parent))
            for k, w in selection.DFLASH_PRODUCTION_PROFILES:
                chosen = recipe == self.chosen and k == 4
                parent = root / recipe / f'k{k}-w{w}'
                eligible = [c for c in (1, 2, 3, 4) if not (chosen and c == missing_c)]
                self.write(parent / 'capacity/manifest.json', {'recipe': recipe, 'k': k, 'w': w,
                                                             'eligible': eligible})
                self.capacities.append((recipe, k, w, parent / 'capacity'))
                if 1 not in eligible:
                    continue
                screen_pass = not (chosen and failed_screen)
                for stage, concurrency in (('c1', [1]), ('pareto', eligible)):
                    means = {str(c): (130. if chosen and c == 1 else 110. if chosen else 120.)
                             for c in concurrency}
                    if chosen and stage == 'pareto' and slow_c in concurrency:
                        means[str(slow_c)] = 101.
                    if stage == 'c1' and not screen_pass:
                        means['1'] = 101.
                    self.write(parent / stage / 'manifest.json', {
                        'recipe': recipe, 'k': k, 'w': w, 'concurrency': concurrency, 'means': means})
                    for name in ('parity', 'determinism', 'generated_quality'):
                        self.write(parent / stage / f'{name}.json', {'pass': True})
                self.screens.append((recipe, k, w, parent / 'c1'))
                if screen_pass:
                    self.followups.append((recipe, k, w, parent / 'pareto'))

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def evidence(self, route, recipe, conversion, root):
        if json.loads(conversion.read_text()) != {'recipe': recipe}:
            raise ValueError('conversion recipe mismatch')
        return {'recipe': recipe, 'benchmark': self.bench,
                'artifact': {'path': f'/companions/{recipe}.ninfer', 'weights_id': recipe,
                             'sha256': str(selection.RECIPES.index(recipe) + 1) * 64,
                             'file_size_bytes': 200, 'dflash_base_artifact': route['base_artifact']},
                'build': {'benchmark': self.bench}, 'conversion_report': selection._identity(conversion),
                'shortlist': selection._identity(root / 'dflash-shortlist.json')}

    @staticmethod
    def read_cell(evidence, k, w, root):
        value = json.loads((root / 'manifest.json').read_text())
        if (value['recipe'], value['k'], value['w']) != (evidence['recipe'], k, w):
            raise ValueError('mixed recipe/K/W evidence')
        return value

    def capacity(self, route, evidence, k, w, root):
        value = self.read_cell(evidence, k, w, root)
        return {'matrix': selection._identity(root / 'manifest.json'),
                'eligible_concurrency': value['eligible'], 'cells': {
                    str(c): {'eligible': c in value['eligible'], 'tokens': 200000,
                             'exclusion': {} if c in value['eligible'] else {'status': 'capacity_failure'}}
                    for c in (1, 2, 3, 4)}}

    def performance(self, route, evidence, k, w, root, concurrency):
        value = self.read_cell(evidence, k, w, root)
        if value['concurrency'] != concurrency:
            raise ValueError('missing declared concurrency')
        aux = {}
        for name in ('parity', 'determinism', 'generated_quality'):
            path = root / f'{name}.json'
            if json.loads(path.read_text()).get('pass') is not True:
                raise ValueError(f'raw {name} failed')
            aux[name] = selection._identity(path)
        gates = {}
        for c in concurrency:
            rows = []
            for prompt in (8192, 32768):
                common = {'n_prompt': prompt, 'n_gen': 256, 'requested_output_tokens': 257,
                          'concurrency': c}
                for suite, phase, mean in (
                    ('dflash_pareto_whole_inference', 'whole_output', value['means'][str(c)]),
                    ('dflash_pareto_whole_control', 'whole_output', 100.),
                    ('dflash_pareto_decode', 'decode_output', value['means'][str(c)]),
                    ('dflash_pareto_control', 'decode_output', 100.),
                ):
                    rows.append({**common, 'suite': suite, f'{phase}_tok_s_mean': mean,
                                 f'{phase}_tok_s_stddev': .1})
            gates[str(c)] = selection._matched_ordinary_speed_gate(rows, [c])
        return {'matrix': selection._identity(root / 'manifest.json'), 'declared_concurrency': concurrency,
                'matched_speed_by_concurrency': gates, **aux,
                'objectives': {str(c): {'whole': {'8K': value['means'][str(c)], '32K': value['means'][str(c)]},
                                      'acceptance': {'8K': 3., '32K': 3.}} for c in concurrency}}

    def patched(self):
        stack = ExitStack()
        stack.enter_context(patch.object(selection, 'selected_base_route', return_value=self.route))
        stack.enter_context(patch.object(selection, 'recipe_evidence', side_effect=self.evidence))
        stack.enter_context(patch.object(selection, 'capacity_cells', side_effect=self.capacity))
        stack.enter_context(patch.object(selection, 'performance_cells', side_effect=self.performance))
        return stack

    def evaluate(self):
        value = selection.assemble(self.base, self.recipes, self.capacities, self.screens, self.followups)
        path = self.write(self.root / 'evaluation.json', value)
        return path, value


class ResidentAdmissionTest(unittest.TestCase):
    def test_explicit_resident_all_four_even_when_not_each_frontier_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Campaign(Path(directory))
            with fixture.patched():
                path, evaluation = fixture.evaluate()
                key = selection.candidate_key(fixture.chosen, 4, 5)
                self.assertEqual(evaluation['per_concurrency']['1']['winner'], key)
                self.assertNotEqual(evaluation['per_concurrency']['4']['winner'], key)
                self.assertFalse(evaluation['production_selected'])
                output = Path(directory) / 'resident.json'
                selection.main(['admit', '--evaluation', str(path), '--recipe', fixture.chosen,
                                '--draft-tokens', '4', '--verify-width', '5', '--out', str(output)])
                admitted = cutover.revalidate_dflash(output)
                self.assertEqual(admitted['concurrency'], [1, 2, 3, 4])
                self.assertEqual(admitted['resident']['key'], key)
                self.assertTrue(admitted['production_selected'])
                self.assertEqual(selection.revalidate_evaluation(path), evaluation)
                snapshot = {'path': str(fixture.base), 'sha256': selection.file_sha256(fixture.base)}
                route = {'winner': 'base-winner', 'artifact': fixture.route['base_artifact'],
                         'executable': fixture.bench, 'cache_group': 16, 'attention_profile': 'dense',
                         'prefill_chunk': 2048}
                cutover.require_dflash_base(admitted, snapshot, route)
                with self.assertRaisesRegex(ValueError, 'same base selection'):
                    cutover.require_dflash_base(admitted, snapshot, {**route, 'cache_group': 32})
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    selection.main(['admit', '--evaluation', str(path), '--recipe', fixture.chosen,
                                    '--draft-tokens', '4', '--verify-width', '5', '--out', str(output)])

    def test_cannot_mix_per_concurrency_winners_or_shrink_contract(self):
        for kwargs in ({'missing_c': 4}, {'slow_c': 4}, {'failed_screen': True}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                fixture = Campaign(Path(directory), **kwargs)
                with fixture.patched():
                    path, evaluation = fixture.evaluate()
                    self.assertIsNotNone(evaluation['per_concurrency']['4']['winner'])
                    with self.assertRaisesRegex(ValueError, 'every C1..4'):
                        selection.admit_single_resident(path, fixture.chosen, 4, 5)
                    chosen = next(row for row in evaluation['candidates']
                                  if row['key'] == selection.candidate_key(fixture.chosen, 4, 5))
                    chosen['qualified_concurrency'] = [1, 2, 3, 4]
                    chosen['exclusions'] = {}
                    fixture.write(path, evaluation)
                    with self.assertRaisesRegex(ValueError, 'recomputed raw evidence'):
                        selection.admit_single_resident(path, fixture.chosen, 4, 5)

    def test_changed_raw_parity_or_proposal_proof_rejects_admission(self):
        for proof in ('parity', 'determinism', 'generated_quality'):
            with self.subTest(proof=proof), tempfile.TemporaryDirectory() as directory:
                fixture = Campaign(Path(directory))
                with fixture.patched():
                    path, _ = fixture.evaluate()
                    result = selection.admit_single_resident(path, fixture.chosen, 4, 5)
                    output = fixture.write(Path(directory) / 'resident.json', result)
                    fixture.write(Path(directory) / fixture.chosen / 'k4-w5/pareto' / f'{proof}.json', {'pass': False})
                    with self.assertRaisesRegex(ValueError, f'raw {proof} failed'):
                        cutover.revalidate_dflash(output)

    def test_missing_manifest_and_cross_recipe_proof_cannot_rebind(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Campaign(Path(directory))
            with fixture.patched():
                path, evaluation = fixture.evaluate()
                chosen = next(row for row in evaluation['candidates']
                              if row['key'] == selection.candidate_key(fixture.chosen, 4, 5))
                other = next(row for row in evaluation['candidates']
                             if row['recipe'] == selection.RECIPES[1] and row['draft_tokens'] == 4)
                chosen['performance'] = copy.deepcopy(other['performance'])
                fixture.write(path, evaluation)
                with self.assertRaisesRegex(ValueError, 'mixed recipe/K/W'):
                    selection.admit_single_resident(path, fixture.chosen, 4, 5)
                path, evaluation = fixture.evaluate()
                first_manifest = Path(evaluation['candidates'][0]['capacity']['matrix']['path'])
                first_manifest.unlink()
                with self.assertRaisesRegex(ValueError, 'capacity evidence changed'):
                    selection.admit_single_resident(path, fixture.chosen, 4, 5)

    def test_bare_evaluation_and_mutated_resident_never_authorize_cutover(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Campaign(Path(directory))
            with fixture.patched():
                path, _ = fixture.evaluate()
                with self.assertRaisesRegex(ValueError, 'single-resident'):
                    cutover.revalidate_dflash(path)
                result = selection.admit_single_resident(path, fixture.chosen, 4, 5)
                output = Path(directory) / 'resident.json'
                for field, value in (('concurrency', [1]), ('production_selected', False),
                                     ('resident_by_concurrency', {'1': fixture.chosen, '4': selection.RECIPES[1]})):
                    fixture.write(output, {**result, field: value})
                    with self.assertRaisesRegex(ValueError, 'recomputed evidence'):
                        cutover.revalidate_dflash(output)
                malformed = copy.deepcopy(result)
                malformed['resident']['recipe'] = selection.RECIPES[1]
                fixture.write(output, malformed)
                with self.assertRaisesRegex(ValueError, 'recomputed evidence'):
                    cutover.revalidate_dflash(output)


if __name__ == '__main__':
    unittest.main()
