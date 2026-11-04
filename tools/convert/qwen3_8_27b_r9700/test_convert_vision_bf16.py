from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec,
    TensorSpec,
)
from tools.convert.qwen3.common.recipe import SourceTensor, TensorRecipe

from . import convert_vision_bf16 as converter


class VisionBf16ConversionTest(unittest.TestCase):
    def test_source_encoding_and_preserved_text_payloads(self) -> None:
        # Nontrivial BF16 source words exercise exact encoding, while miniature
        # signatures isolate copying from full checkpoint preflight.
        name = "vision/layers/0/mlp/fc2"
        shape = (16, 136)
        spec = TensorSpec(name, shape, "BF16", "contiguous-le-v1")
        source = (torch.arange(16 * 136).reshape(shape).float().sin() * 0.37).bfloat16()
        preserved = bytes(range(256))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base, output = root / "base.ninfer", root / "output.ninfer"
            (root / "config.json").write_text("{}")
            save_file({"source": source}, root / "weights.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"source": "weights.safetensors"}})
            )
            with ArtifactWriter(
                base,
                ArtifactIdentity("qwen3.8-27b", "r9700-lq4"),
                (
                    TensorSpec(name, shape, "Q4G64_F16S", "r9700-q4g64-n16-k16-v1"),
                    ResourceSpec("text/preserved", "raw-bytes-v1", len(preserved)),
                ),
            ) as writer:
                writer.write(name, [bytes(16 * 256 // 2 + 16 * 4 * 2)])
                writer.write("text/preserved", [preserved])
            with (
                patch.object(converter, "VISION_SPECS", (spec,)),
                patch.object(converter, "MATRICES", {name: spec}),
                patch.object(converter.source, "validate_config"),
                patch.object(
                    converter.source_recipe,
                    "preflight_sources",
                    return_value=SimpleNamespace(
                        source_shard_count=18, source_dtype_counts={"BF16": 1199}
                    ),
                ),
                patch.object(
                    converter.source_recipe,
                    "RECIPES_BY_NAME",
                    {name: TensorRecipe(name, SourceTensor("source", shape))},
                ),
            ):
                report = converter.convert(base, root, output)
                self.assertTrue(report["preserved_payloads_byte_exact"])
                with Artifact(output) as artifact:
                    self.assertEqual(artifact.identity.weights_id, converter.WEIGHTS_ID)
                    self.assertEqual(bytes(artifact.payload("text/preserved")), preserved)
                    with artifact.payload(name) as payload:
                        self.assertEqual(
                            bytes(payload), source.view(torch.uint16).numpy().tobytes()
                        )
                with self.assertRaises(FileExistsError):
                    converter.convert(base, root, output)


if __name__ == "__main__":
    unittest.main()
