"""Focused tests for shared Qwen3 conversion-report publication."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec,
    plan_objects,
)
from tools.convert.qwen3.common.conversion import build_conversion_report


class ConversionReportTest(unittest.TestCase):
    def test_report_binds_completed_artifact_sha256_and_size(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "fixture.ninfer"
            ranking = root / "ranking.i64"
            ranking.write_bytes(b"ranking")
            identity = ArtifactIdentity("fixture-model", "fixture-weights")
            specs = (ResourceSpec("frontend/fixture", "raw-bytes-v1", 7),)
            objects = plan_objects(specs)
            with ArtifactWriter(artifact, identity, specs) as writer:
                writer.write("frontend/fixture", b"payload")

            final_bytes = artifact.stat().st_size
            report = build_conversion_report(
                identity=identity,
                target_key="fixture-target",
                recipe_id="fixture-recipe",
                repo_root=root,
                model_dir=root,
                out_path=artifact,
                arguments={},
                config_summary={},
                source_preflight=SimpleNamespace(
                    recipe_count=1,
                    source_tensor_count=1,
                    source_shard_count=1,
                    source_dtype_counts={"BF16": 1},
                ),
                objects=objects,
                elapsed_seconds=1.0,
                final_bytes=final_bytes,
                device=torch.device("cpu"),
                ranking_path=ranking,
                revision="fixture-revision",
                environment_summary={},
            )
            self.assertEqual(report["artifact"]["bytes"], final_bytes)
            self.assertEqual(
                report["artifact"]["sha256"],
                hashlib.sha256(artifact.read_bytes()).hexdigest(),
            )

            with self.assertRaisesRegex(ValueError, "artifact size changed"):
                build_conversion_report(
                    identity=identity,
                    target_key="fixture-target",
                    recipe_id="fixture-recipe",
                    repo_root=root,
                    model_dir=root,
                    out_path=artifact,
                    arguments={},
                    config_summary={},
                    source_preflight=SimpleNamespace(
                        recipe_count=1,
                        source_tensor_count=1,
                        source_shard_count=1,
                        source_dtype_counts={"BF16": 1},
                    ),
                    objects=objects,
                    elapsed_seconds=1.0,
                    final_bytes=final_bytes - 1,
                    device=torch.device("cpu"),
                    ranking_path=ranking,
                    revision="fixture-revision",
                    environment_summary={},
                )


if __name__ == "__main__":
    unittest.main()
