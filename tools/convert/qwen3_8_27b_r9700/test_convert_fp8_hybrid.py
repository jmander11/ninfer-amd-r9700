"""Synthetic CPU conversion test for the four-role FP8/Q4 artifact boundary."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tools.artifact.container import Artifact
from tools.convert.qwen3.common import conversion as family_conversion
from tools.convert.qwen3.common.inventory import ResourceSpec, TensorSpec

from . import convert_fp8_hybrid, fp8_hybrid_inventory, source_recipe


class Fp8HybridConverterTest(unittest.TestCase):
    def _destination_preflight(self, root: Path):
        resource = family_conversion.ResourcePayload("frontend/tokenizer.json", b"{}")
        specs = (
            ResourceSpec(resource.name),
            TensorSpec("fixture/weight", (16, 128), "Q4G64_F16S", "r9700-q4g64-n16-k16-v1"),
        )
        plan = family_conversion.build_object_plan(specs, {resource.name: resource.data})
        ranking = root / "ranking.i64"
        ranking.write_bytes(b"ranking")
        return convert_fp8_hybrid.Fp8HybridConversionPreflight(
            model_dir=root,
            config_summary={"synthetic": True},
            source=source_recipe.SourcePreflight(1, 1199, 18, {"BF16": 1199}),
            resources=(resource,),
            draft=SimpleNamespace(ranking=ranking, n=1),
            draft_ranking=SimpleNamespace(
                ranking_path=ranking,
                ranking_sha256="1" * 64,
                sidecar_path=root / "ranking.i64.provenance.json",
                sidecar_sha256="2" * 64,
                total_tokens=1,
                distinct_token_ids=1,
            ),
            object_plan=plan,
            source_shards=convert_fp8_hybrid.SourceShardManifest(
                index_path=root / "model.safetensors.index.json",
                index_sha256="3" * 64,
                shard_names=convert_fp8_hybrid._SOURCE_SHARDS,
                shard_bytes=(1,) * 18,
            ),
        )

    def test_synthetic_conversion_has_one_plane_per_selected_name(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected_name = "text/layers/0/mlp/gate_up"
            q4_name = "fixture/unselected"
            selected_tensor = torch.linspace(-1, 1, 2048, dtype=torch.bfloat16).reshape(16, 128)
            q4_tensor = torch.linspace(-2, 2, 2048, dtype=torch.bfloat16).reshape(16, 128)
            tensors = {selected_name: selected_tensor, q4_name: q4_tensor}

            class FakeShardReader:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def get(self, name):
                    return tensors[name]
            ranking = root / "ranking.i64"
            ranking.write_bytes(b"synthetic-ranking")
            resource = family_conversion.ResourcePayload("frontend/tokenizer.json", b"{}")
            object_specs = (
                ResourceSpec(resource.name),
                TensorSpec(selected_name, (16, 128), "F8E4M3_ROW_F32S", "row-scaled-k128-v1"),
                TensorSpec(q4_name, (16, 128), "Q4G64_F16S", "r9700-q4g64-n16-k16-v1"),
            )
            plan = family_conversion.build_object_plan(
                object_specs, {resource.name: resource.data}
            )
            preflight = convert_fp8_hybrid.Fp8HybridConversionPreflight(
                model_dir=root,
                config_summary={"synthetic": True},
                source=source_recipe.SourcePreflight(2, 2, 1, {"BF16": 2}),
                resources=(resource,),
                draft=SimpleNamespace(ranking=ranking, n=1),
                draft_ranking=SimpleNamespace(
                    ranking_path=ranking,
                    ranking_sha256=hashlib.sha256(ranking.read_bytes()).hexdigest(),
                    sidecar_path=root / "ranking.i64.provenance.json",
                    sidecar_sha256="2" * 64,
                    total_tokens=2,
                    distinct_token_ids=2,
                ),
                object_plan=plan,
                source_shards=convert_fp8_hybrid.SourceShardManifest(
                    index_path=root / "model.safetensors.index.json",
                    index_sha256="3" * 64,
                    shard_names=("synthetic.safetensors",),
                    shard_bytes=(512,),
                ),
            )
            output = root / "hybrid.ninfer"
            with (
                mock.patch.object(
                    convert_fp8_hybrid, "preflight_conversion", return_value=preflight
                ),
                mock.patch.object(fp8_hybrid_inventory, "OBJECT_SPECS", object_specs),
                mock.patch.object(
                    convert_fp8_hybrid, "ShardReader", return_value=FakeShardReader()
                ),
                mock.patch.object(
                    convert_fp8_hybrid,
                    "preflight_destination",
                    return_value=convert_fp8_hybrid.DestinationPreflight(
                        output_path=output,
                        report_path=Path(str(output) + ".conversion.json"),
                        filesystem_path=root,
                        projected_file_bytes=1024,
                        headroom_bytes=1024,
                        required_free_bytes=2048,
                        available_free_bytes=4096,
                    ),
                ),
                mock.patch.object(
                    convert_fp8_hybrid.source,
                    "materialize_tensor",
                    side_effect=lambda spec, reader, draft: reader.get(spec.name),
                ),
                redirect_stdout(io.StringIO()),
            ):
                report_path = convert_fp8_hybrid.convert(
                    root, output, draft_ranking=ranking, device="cpu"
                )

            with Artifact.open(output) as artifact:
                self.assertEqual(artifact.identity.weights_id, fp8_hybrid_inventory.WEIGHTS_ID)
                self.assertEqual(len(artifact.objects), 3)
                self.assertEqual(
                    (artifact.find(selected_name).format, artifact.find(selected_name).layout),
                    ("F8E4M3_ROW_F32S", "row-scaled-k128-v1"),
                )
                self.assertEqual(
                    (artifact.find(q4_name).format, artifact.find(q4_name).layout),
                    ("Q4G64_F16S", "r9700-q4g64-n16-k16-v1"),
                )
                self.assertEqual(
                    len([obj for obj in artifact.objects if obj.name == selected_name]), 1
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["recipe_id"], fp8_hybrid_inventory.RECIPE_ID)
            self.assertEqual(
                report["candidate"]["selection_sha256"],
                fp8_hybrid_inventory.SELECTION_SHA256,
            )
            self.assertFalse(report["candidate"]["weight_recipe_selected"])
            validation = convert_fp8_hybrid.validate_completed_conversion(
                preflight, output
            )
            self.assertEqual(validation["status"], "valid")
            self.assertEqual(validation["objects"], 3)
            report["candidate"]["object_plan_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate receipt"):
                convert_fp8_hybrid.validate_completed_conversion(preflight, output)

    def test_destination_preflight_is_no_write_and_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checked = self._destination_preflight(root)
            output = root / "new" / "hybrid.ninfer"
            projected = convert_fp8_hybrid._projected_file_bytes(checked)
            available = projected + (2 << 30)
            with mock.patch.object(
                convert_fp8_hybrid.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=available),
            ):
                destination = convert_fp8_hybrid.preflight_destination(checked, output)
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())
            self.assertEqual(destination.projected_file_bytes, projected)
            self.assertEqual(
                destination.required_free_bytes,
                projected + convert_fp8_hybrid._DISK_HEADROOM_BYTES,
            )
            with (
                mock.patch.object(convert_fp8_hybrid.preflight_identity, "source_checkpoint",
                                  return_value={"indexed_tensor_count": 1199}),
                mock.patch.object(convert_fp8_hybrid.preflight_identity, "frontend_resources",
                                  return_value=[{"name": "fixture"}] * 6),
            ):
                summary = convert_fp8_hybrid.preflight_summary(checked, destination)
            self.assertEqual(summary["identity"]["weights_id"], fp8_hybrid_inventory.WEIGHTS_ID)
            self.assertEqual(summary["recipe_id"], fp8_hybrid_inventory.RECIPE_ID)
            self.assertEqual(summary["destination"]["projected_file_bytes"], projected)
            self.assertEqual(summary["conversion_argv"][-1], "cuda")
            self.assertEqual(summary["validation_argv"][-1], "--validate-only")

            with mock.patch.object(
                convert_fp8_hybrid.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=projected),
            ):
                with self.assertRaisesRegex(OSError, "insufficient free space"):
                    convert_fp8_hybrid.preflight_destination(checked, output)
            occupied = root / "occupied.ninfer"
            occupied.write_bytes(b"occupied")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                convert_fp8_hybrid.preflight_destination(checked, occupied)

    def test_source_manifest_requires_exact_nonempty_eighteen_shards(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = {
                f"tensor.{index}": convert_fp8_hybrid._SOURCE_SHARDS[index % 18]
                for index in range(1199)
            }
            index_path = root / "model.safetensors.index.json"
            index_path.write_text(json.dumps({"weight_map": mapping}), encoding="utf-8")
            for shard in convert_fp8_hybrid._SOURCE_SHARDS:
                (root / shard).write_bytes(b"represented-BF16-shard")
            manifest = convert_fp8_hybrid._source_shard_manifest(root)
            self.assertEqual(manifest.shard_names, convert_fp8_hybrid._SOURCE_SHARDS)
            self.assertEqual(len(manifest.index_sha256), 64)
            (root / convert_fp8_hybrid._SOURCE_SHARDS[-1]).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "source shard is missing"):
                convert_fp8_hybrid._source_shard_manifest(root)


if __name__ == "__main__":
    unittest.main()
