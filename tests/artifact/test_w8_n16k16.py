import struct
import tempfile
import unittest
from pathlib import Path

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encoded_size, transcode_w8_n16k16, w8_n16k16_geometry


class W8N16K16Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

    def test_exact_code_and_scale_permutation(self) -> None:
        for shape in ((16, 1), (32, 129), (48, 5120)):
            with self.subTest(shape=shape):
                self._check_exact_code_and_scale_permutation(shape)

    def _check_exact_code_and_scale_permutation(self, shape: tuple[int, int]) -> None:
        n, k = shape
        geometry = w8_n16k16_geometry(shape)
        groups = geometry.groups_per_row
        payload = bytearray(geometry.payload_bytes)
        for row in range(n):
            for column in range(geometry.k_pad):
                payload[row * geometry.k_pad + column] = (
                    ((row * 17 + column * 29) % 255 - 127) & 255 if column < k else 0
                )
            for group in range(groups):
                # Positive finite FP16 words, plus exact signed-zero preservation.
                word = 0x8000 if (row + group) % 13 == 0 else 0x0400 + (row * 31 + group) % 0x7000
                struct.pack_into(
                    "<H", payload, geometry.scale_offset + 2 * (row * groups + group), word
                )
        tiled = b"".join(transcode_w8_n16k16(payload, shape))
        # Independent scalar address oracle, not inverse-transform-only coverage.
        for row in range(n):
            for column in range(geometry.k_pad):
                index = (
                    ((row // 16 * groups + column // 32) * 2 + column % 32 // 16) * 16 + row % 16
                ) * 16 + column % 16
                self.assertEqual(tiled[index], payload[row * geometry.k_pad + column])
            for group in range(groups):
                target = geometry.scale_offset + ((row // 16 * groups + group) * 16 + row % 16) * 2
                source = geometry.scale_offset + (row * groups + group) * 2
                self.assertEqual(tiled[target : target + 2], payload[source : source + 2])
        self.assertEqual(b"".join(transcode_w8_n16k16(tiled, shape, inverse=True)), payload)
        layout = "r9700-w8g32-n16-k16-v1"
        self.assertEqual(encoded_size(layout, "W8G32_F16S", shape), len(payload))
        path = self.tmp_path / f"head-{n}x{k}.ninfer"
        with ArtifactWriter(
            path,
            ArtifactIdentity("test", "layout"),
            [TensorSpec("head", shape, "W8G32_F16S", layout)],
        ) as writer:
            writer.write("head", tiled)
        with Artifact.open(path) as artifact:
            view = artifact.payload("head")
            self.assertEqual(view, tiled)
            view.release()

    def test_invalid_geometry_format_and_extent(self) -> None:
        with self.assertRaises(ValueError):
            w8_n16k16_geometry((17, 128))
        with self.assertRaises(ValueError):
            encoded_size("r9700-w8g32-n16-k16-v1", "Q4G64_F16S", (16, 128))
        with self.assertRaises(ValueError):
            b"".join(transcode_w8_n16k16(b"short", (16, 128)))


if __name__ == "__main__":
    unittest.main()
