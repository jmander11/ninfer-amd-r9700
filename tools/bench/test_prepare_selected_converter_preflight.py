from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench import prepare_selected_converter_preflight as producer


class SelectedConverterPreflightTest(unittest.TestCase):
    def fixture(self, root: Path, weights: str) -> tuple[dict, Path, Path, Path]:
        selection = root / "selection.json"; selection.write_text("{}")
        artifact = root / "evaluation.ninfer"; artifact.write_bytes(b"evaluation")
        model = root / "model"; model.mkdir(); (model / "config.json").write_text("{}")
        (model / "model.safetensors.index.json").write_text("{}")
        ranking = root / "ranking.i64"; ranking.write_bytes(b"ranking")
        prospective = root / "staging" / "selected.ninfer"
        route = {"winner": "winner", "artifact": {"path": str(artifact),
                 "sha256": producer.sha(artifact), "file_size_bytes": artifact.stat().st_size,
                 "weights_id": weights}}
        return route, selection, model, ranking, prospective

    def summary(self, weights: str, model: Path, ranking: Path,
                prospective: Path | None = None) -> dict:
        inventory = producer.RECIPES[weights][1]
        branch = producer.RECIPES[weights][2]
        provenance = ranking.with_suffix(".manifest.json"); provenance.write_text("{}")
        value = {
            "artifact_type": producer.PREFLIGHT_TYPES[branch], "schema_version": 1,
            "identity": {"model_id": "qwen3.8-27b", "weights_id": weights},
            "target_key": inventory.TARGET_KEY, "recipe_id": inventory.RECIPE_ID,
            "model_path": str(model.resolve()),
            "source": {"tensors": 1199, "shards": 18, "dtypes": {"BF16": 1199}},
            "checkpoint": {"config": producer.file_receipt(model / "config.json", "config"),
                           "index": producer.file_receipt(model / "model.safetensors.index.json", "index"),
                           "indexed_tensor_count": 1199,
                           "index_total_tensor_bytes": producer.preflight_identity.bf16_protocol.SOURCE_TOTAL_BYTES,
                           "shards": [{"name": name, "bytes": 1}
                                      for name in producer.preflight_identity.SOURCE_SHARDS],
                           "shard_total_file_bytes": 18, "shard_payload_sha256": None,
                           "shard_payload_hash_policy": producer.SHARD_HASH_POLICY},
            "frontend_resources": [
                {"name": name, "bytes": 1, "sha256": digest}
                for name, digest in producer.target_resources.OFFICIAL_RESOURCE_SHA256.items()],
            "objects": {"count": 1124}, "object_plan_sha256": "b" * 64,
            "ranking": {"path": str(ranking.resolve()), "sha256": producer.sha(ranking),
                        "provenance_path": str(provenance.resolve()),
                        "provenance_sha256": producer.sha(provenance)},
            "format_counts": inventory.FORMAT_COUNTS,
            "format_encoded_bytes": inventory.FORMAT_ENCODED_BYTES,
            "tensor_encoded_bytes": inventory.TENSOR_ENCODED_BYTES,
            "device_arena_bytes": inventory.DEVICE_ARENA_BYTES,
        }
        if prospective is not None:
            value["selection_sha256"] = "c" * 64
            value["destination"] = {"artifact_path": str(prospective.absolute()),
                                    "destination_exists": False,
                                    "required_free_bytes": 1, "available_free_bytes": 2}
        return value

    def test_only_selected_all_q4_converter_executes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); route, selection, model, ranking, prospective = self.fixture(
                root, "r9700-q4g64-n16k16-eval")
            summary = self.summary(route["artifact"]["weights_id"], model, ranking)
            with patch.object(producer, "resolve", return_value=route), patch.object(
                    producer.convert_q4, "preflight_conversion", return_value=object()) as selected, \
                    patch.object(producer.convert_q4, "preflight_summary", return_value=summary), \
                    patch.object(producer.convert_q4_w8_mse, "preflight_conversion") as losing:
                value = producer.build_value(selection, model, ranking, prospective)
            selected.assert_called_once(); losing.assert_not_called()
            self.assertEqual(value["selected_route"]["branch"], "all_q4")
            self.assertFalse(prospective.exists())

    def test_hybrid_binds_absent_destination_without_writing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); route, selection, model, ranking, prospective = self.fixture(
                root, "r9700-q4g64-f8e4m3-four-role-n16k16-eval")
            summary = self.summary(route["artifact"]["weights_id"], model, ranking, prospective)
            with patch.object(producer, "resolve", return_value=route), patch.object(
                    producer.convert_fp8_hybrid, "preflight_conversion", return_value=object()), \
                    patch.object(producer.convert_fp8_hybrid, "preflight_destination",
                                 return_value=object()), patch.object(
                    producer.convert_fp8_hybrid, "preflight_summary", return_value=summary):
                value = producer.build_value(selection, model, ranking, prospective)
            self.assertEqual(value["prospective_final_artifact"], str(prospective.absolute()))
            self.assertFalse(prospective.exists()); self.assertFalse(prospective.parent.exists())

    def test_rejects_existing_prospective_artifact_before_converter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); route, selection, model, ranking, prospective = self.fixture(
                root, "r9700-q4-w8-mse-n16k16-eval")
            prospective.parent.mkdir(); prospective.write_bytes(b"occupied")
            with patch.object(producer, "resolve", return_value=route), patch.object(
                    producer.convert_q4_w8_mse, "preflight_conversion") as conversion, \
                    self.assertRaisesRegex(ValueError, "must be absent"):
                producer.build_value(selection, model, ranking, prospective)
            conversion.assert_not_called()

    def test_rejects_alias_ancestor_and_accepts_writable_absent_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prospective = root / "absent" / "selected.ninfer"
            self.assertEqual(
                producer.validate_prospective_artifact(prospective), prospective.absolute())
            self.assertFalse(prospective.parent.exists())
            alias = root / "alias"; alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real directory"):
                producer.validate_prospective_artifact(alias / "selected.ninfer")

    def test_rejects_reserved_conversion_report_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prospective = root / "selected.ninfer"
            Path(str(prospective) + ".conversion.json").symlink_to(root / "missing")
            with self.assertRaisesRegex(ValueError, "must be absent"):
                producer.validate_prospective_artifact(prospective)


if __name__ == "__main__": unittest.main()
