"""Endpoint recipe formats and exact real-container payload composition."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec as StoredTensor
from tools.artifact.layouts import encoded_size
from tools.convert.qwen3.common.inventory import tensor_spec, TensorSpec, Q4
from . import compose_fp8_endpoints as compose


class EndpointCompositionTest(unittest.TestCase):
    def test_fixed_six_recipes(self):
        recipes = compose.recipes()
        self.assertEqual(len(recipes), 6)
        for identity, (base, embed, head) in recipes.items():
            specs = {s.name: s for s in compose.specs_for(identity)}
            base_specs = {s.name: s for s in compose.capped.specs_for(compose.capped.recipes()[base])}
            self.assertEqual(len(specs), 1124)
            self.assertFalse(any(name.startswith('dflash/') for name in specs))
            for name in specs:
                if name not in compose.endpoint_names(identity):
                    self.assertEqual(specs[name], base_specs[name])
            self.assertEqual(specs[compose.EMBEDDING].format, 'W8G32_F16S' if embed else Q4)
            self.assertEqual(specs[compose.HEAD].format, 'W8G32_F16S' if head else Q4)
            if embed:
                self.assertEqual(specs[compose.EMBEDDING].layout, 'row-split-k128-v1')
            if head:
                self.assertEqual(specs[compose.HEAD].layout, 'r9700-w8g32-n16-k16-v1')

    def test_exact_codes_scales_and_no_companion(self):
        base_specs = tuple(tensor_spec(name, (16, 128), Q4)
                           for name in (compose.EMBEDDING, compose.HEAD, 'text/unchanged'))
        donor_specs = (
            TensorSpec(compose.EMBEDDING, (16, 128), 'W8G32_F16S', 'row-split-k128-v1'),
            TensorSpec(compose.HEAD, (16, 128), 'W8G32_F16S', 'r9700-w8g32-n16-k16-v1'),
            tensor_spec('dflash/unused', (16, 128), Q4))
        records = {'embedding': ('base', True, False), 'head': ('base', False, True),
                   'both': ('base', True, True)}
        with TemporaryDirectory() as temporary, \
             patch.object(compose, 'recipes', return_value=records), \
             patch.object(compose.capped, 'recipes', return_value={'base': set()}), \
             patch.object(compose.capped.q4_inventory, 'OBJECT_SPECS', base_specs), \
             patch.object(compose.dflash, 'SELECTIVE_OBJECT_SPECS', donor_specs):
            root = Path(temporary)
            base, donor = root/'base.ninfer', root/'donor.ninfer'
            def write(path, identity, specs, seed):
                stored = tuple(StoredTensor(s.name, s.shape, s.format, s.layout) for s in specs)
                payloads = {}
                with ArtifactWriter(path, ArtifactIdentity(compose.dflash.MODEL_ID, identity), stored) as writer:
                    for spec in specs:
                        size = encoded_size(spec.layout, spec.format, spec.shape)
                        data = bytes((i + seed) % 256 for i in range(size))
                        writer.write(spec.name, data)
                        payloads[spec.name] = data
                return payloads
            base_payloads = write(base, 'base', base_specs, 7)
            donor_payloads = write(donor, compose.dflash.SELECTIVE_WEIGHTS_ID, donor_specs, 29)
            for recipe in records:
                output = root/f'{recipe}.ninfer'
                compose.compose(recipe, base, donor, output)
                with Artifact(output) as artifact:
                    self.assertEqual(len(artifact.objects), 3)
                    for obj in artifact.objects:
                        expected = donor_payloads if obj.name in compose.endpoint_names(recipe) else base_payloads
                        with artifact.payload(obj.name) as payload:
                            self.assertEqual(bytes(payload), expected[obj.name])
                with self.assertRaises(FileExistsError):
                    compose.compose(recipe, base, donor, output)
            with self.assertRaises(ValueError):
                compose.compose('both', donor, base, root/'wrong.ninfer')
            self.assertFalse((root/'wrong.ninfer').exists())


if __name__ == '__main__':
    unittest.main()
