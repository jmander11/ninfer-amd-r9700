from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec
from tools.artifact.layouts import (decode_row_split_codes, encode_q4_n16k16,
                                    q4_n16k16_geometry)

from .transcode_embedding_rows import transcode


class EmbeddingRowTranscodeTest(unittest.TestCase):
    def test_permutes_only_the_embedding_and_carries_the_receipt(self) -> None:
        shape = (32, 256)
        geometry = q4_n16k16_geometry(shape)
        generator = torch.Generator().manual_seed(5)
        codes = torch.randint(-8, 8, (32, geometry.groups_per_row, 64), dtype=torch.int8,
                              generator=generator)
        scales = (torch.rand((32, geometry.groups_per_row), generator=generator) + 0.5).half()
        head = bytes(range(256)) * 2
        with TemporaryDirectory() as directory:
            source, output = Path(directory, "source.ninfer"), Path(directory, "out.ninfer")
            specs = (TensorSpec("text/token_embedding", shape, "Q4G64_F16S",
                                "r9700-q4g64-n16-k16-v1"),
                     ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", len(head)))
            with ArtifactWriter(source, ArtifactIdentity("qwen3.8-27b", "r9700-fp8lut4"),
                                specs) as writer:
                writer.write("text/token_embedding", [encode_q4_n16k16(codes, scales, shape)])
                writer.write("frontend/tokenizer.json", [head])
            Path(str(source) + ".conversion.json").write_text(json.dumps({
                "recipe": "r9700-fp8lut4", "objects": [
                    {"name": "text/token_embedding", "origin": "base-copy-exact", "sha256": "0"},
                    {"name": "frontend/tokenizer.json", "origin": "base-copy-exact",
                     "sha256": "1"}]}))
            result = transcode(source, output)
            with Artifact(output) as artifact:
                embedding = artifact.find("text/token_embedding")
                self.assertEqual(embedding.layout, "row-split-k128-v1")
                with artifact.payload(embedding) as payload:
                    decoded_scales, decoded_codes = decode_row_split_codes(
                        bytes(payload), "Q4G64_F16S", shape)
                with artifact.payload("frontend/tokenizer.json") as payload:
                    self.assertEqual(bytes(payload), head)
            self.assertTrue(torch.equal(decoded_codes, codes))
            self.assertTrue(torch.equal(decoded_scales, scales))
            receipt = json.loads(Path(str(output) + ".conversion.json").read_text())
            self.assertEqual(receipt["objects"][0], {
                "name": "text/token_embedding", "origin": "base-transcode-row-split",
                "sha256": result["embedding_sha256"]})
            self.assertEqual(receipt["objects"][1]["sha256"], "1")
            with self.assertRaises(FileExistsError):
                transcode(source, output)


if __name__ == "__main__":
    unittest.main()
