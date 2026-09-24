"""Real-container checks for exact base/companion composition and rejected donors."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.convert.qwen3.common.inventory import tensor_spec, BF16
from . import compose_fp8_capped_dflash as compose


class CompositionTest(unittest.TestCase):
    def test_exact_payloads_and_rejections(self):
        base_specs = (tensor_spec('text/final_norm', (2,), BF16),)
        head_specs = (tensor_spec('dflash/selector/predecessor_codebook', (2,), BF16),
                      tensor_spec('dflash/selector/successor_codebook', (2,), BF16))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, donor, output = [root / name for name in ('base.ninfer', 'donor.ninfer', 'out.ninfer')]
            def write(path, identity, specs, payloads):
                stored = tuple(TensorSpec(s.name, s.shape, s.format, s.layout) for s in specs)
                with ArtifactWriter(path, ArtifactIdentity('qwen3.8-27b', identity), stored) as writer:
                    for spec, data in zip(specs, payloads, strict=True):
                        writer.write(spec.name, data)
            write(base, compose.BASE_ID, base_specs, (b'\x80?\x00@',))
            write(donor, compose.DONOR_ID, base_specs + head_specs,
                  (b'xxxx', b'\x00@\x80@', b'\x80?\x80\xbf'))
            with patch.object(compose, 'base_specs', return_value=base_specs), \
                 patch.object(compose.dflash, 'SELECTIVE_OBJECT_SPECS', base_specs + head_specs), \
                 patch.object(compose.dflash, 'TENSOR_SPECS', head_specs), \
                 patch.object(compose.capped, 'validate'):
                receipt = compose.compose(base, donor, output)
                self.assertEqual([r['origin'] for r in receipt['objects']],
                                 ['base', 'dflash-donor', 'dflash-donor'])
                with Artifact(output) as artifact:
                    for name, expected in [('text/final_norm', b'\x80?\x00@'),
                                           (head_specs[0].name, b'\x00@\x80@'),
                                           (head_specs[1].name, b'\x80?\x80\xbf')]:
                        with artifact.payload(name) as payload:
                            self.assertEqual(bytes(payload), expected)
                with self.assertRaises(FileExistsError):
                    compose.compose(base, donor, output)
                wrong = root / 'wrong.ninfer'
                write(wrong, compose.BASE_ID, base_specs + head_specs,
                      (b'xxxx', b'yyyy', b'zzzz'))
                with self.assertRaises(ValueError):
                    compose.compose(base, wrong, root / 'rejected.ninfer')
                self.assertFalse((root / 'rejected.ninfer').exists())
                # Corrupt a represented tensor, not framing: readback must detect it.
                with Artifact(output) as artifact:
                    descriptor = artifact.objects[-1]
                    with artifact.payload(descriptor.name) as payload:
                        expected = bytes(payload)
                raw = output.read_bytes()
                offset = raw.rfind(expected)
                self.assertGreater(offset, 0)
                with output.open('r+b') as file:
                    file.seek(offset)
                    file.write(b'bad!')
                with self.assertRaises(ValueError):
                    compose.validate(output)


if __name__ == '__main__':
    unittest.main()
