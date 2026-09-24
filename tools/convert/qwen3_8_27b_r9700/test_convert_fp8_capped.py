"""Fixed protection inventory and real-container copy-only composition checks."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encoded_size
from tools.convert.qwen3.common.inventory import tensor_spec, Q4
from . import convert_fp8_capped as convert


class CappedConversionTest(unittest.TestCase):
    def test_protection_ablation_inventories(self):
        recipes = convert.recipes()
        output_only = recipes['r9700-q4-fp8-output-only-n16k16-eval']
        self.assertEqual(output_only, {'text/layers/3/attention/output',
            'text/layers/7/attention/output', 'text/layers/4/gdn/output'})
        no_late_mlp = recipes['r9700-q4-fp8-selective-no-late-mlp-n16k16-eval']
        late_mlp = {f'text/layers/{layer}/mlp/{role}'
                    for layer in (62, 63) for role in ('gate_up', 'down')}
        self.assertEqual(no_late_mlp,
            recipes['r9700-q4-fp8-selective-cap-n16k16-eval'] - late_mlp)
        self.assertEqual(len(no_late_mlp), 22)
        for selected in (output_only, no_late_mlp):
            specs = {s.name: s for s in convert.specs_for(selected)}
            for name in late_mlp | {'text/token_embedding', 'text/output_head'}:
                self.assertEqual(specs[name].format, Q4)

    def test_default_protected_inventory(self):
        recipes = convert.recipes()
        selected = recipes['r9700-q4-fp8-default-protected-n16k16-eval']
        expected = {f'text/layers/{layer}/attention/{role}'
                    for layer in (3, 7, 11, 15, 19, 23)
                    for role in ('query_key', 'gate_value')}
        expected |= {'text/layers/3/attention/output', 'text/layers/7/attention/output',
                     'text/layers/4/gdn/output'}
        self.assertEqual(selected, expected)
        self.assertTrue(selected < recipes['r9700-q4-fp8-selective-cap-n16k16-eval'])
        specs = {s.name: s for s in convert.specs_for(selected)}
        self.assertEqual(specs['text/token_embedding'].format, Q4)
        self.assertEqual(specs['text/output_head'].format, Q4)
        self.assertEqual(specs['text/layers/62/mlp/down'].format, Q4)

    def test_exact_capped_donor_copy(self):
        specs = tuple(tensor_spec(name, (16, 128), Q4) for name in ('keep', 'protect', 'lower'))
        recipes = {'donor': {'protect', 'lower'}, 'candidate': {'protect'}}
        with TemporaryDirectory() as temporary, \
             patch.object(convert, 'recipes', return_value=recipes), \
             patch.object(convert.q4_inventory, 'OBJECT_SPECS', specs):
            root = Path(temporary)
            base, donor, output = (root / name for name in ('base.ninfer', 'donor.ninfer', 'out.ninfer'))
            def write(path, identity, inventory, code):
                stored = tuple(TensorSpec(s.name, s.shape, s.format, s.layout) for s in inventory)
                with ArtifactWriter(path, ArtifactIdentity(convert.q4_inventory.MODEL_ID, identity), stored) as writer:
                    for s in inventory:
                        writer.write(s.name, bytes([code]) * encoded_size(s.layout, s.format, s.shape))
            write(base, convert.q4_inventory.WEIGHTS_ID, specs, 0x11)
            write(donor, 'donor', convert.specs_for(recipes['donor']), 0x22)
            args = SimpleNamespace(recipe='candidate', base=base, capped_donor=donor,
                                   out=output, model=None, four_role=None, selective=None)
            convert.convert(args)
            report = convert.validate(output)
            self.assertIsNone(report['source_model'])
            self.assertEqual([o['origin'] for o in report['objects']],
                             ['base-copy-exact', 'capped-copy-exact', 'base-copy-exact'])
            with Artifact(output) as artifact:
                for obj in artifact.objects:
                    expected = 0x22 if obj.name == 'protect' else 0x11
                    with artifact.payload(obj.name) as payload:
                        self.assertEqual(bytes(payload), bytes([expected]) * obj.bytes)
            with self.assertRaises(FileExistsError):
                convert.convert(args)
            args.recipe = 'donor'
            args.capped_donor = output
            args.out = root / 'rejected.ninfer'
            with self.assertRaisesRegex(ValueError, 'every selected FP8'):
                convert.convert(args)
            self.assertFalse(args.out.exists())


if __name__ == '__main__':
    unittest.main()
