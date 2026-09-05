from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import publish_n16_migration_receipt as receipts


class MigrationReceiptTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, dict]:
        source = root / "legacy.ninfer"; output = root / "n16.ninfer"
        source.write_bytes(b"legacy"); output.write_bytes(b"n16")
        upstream = {
            "identity": {"model_id": "qwen3.8-27b", "weights_id": "r9700-q4g64-eval"},
            "target_key": "qwen3_8_27b_r9700", "recipe_id": "r9700-all-q4g64-eval-v0",
            "source": {"model_path": "/checkpoint"},
            "artifact": {"path": str(source), "bytes": 6},
            "candidate": {"status": "registered-evaluation-only",
                          "weight_recipe_selected": False},
        }
        Path(str(source) + ".conversion.json").write_text(json.dumps(upstream))
        validation = {
            "source": str(source), "output": str(output),
            "source_identity": {"model_id": "qwen3.8-27b", "weights_id": "r9700-q4g64-eval"},
            "identity": {"model_id": "qwen3.8-27b",
                         "weights_id": "r9700-q4g64-n16k16-eval"},
            "source_bytes": 6, "output_bytes": 3,
            "source_sha256": hashlib.sha256(b"legacy").hexdigest(),
            "output_sha256": hashlib.sha256(b"n16").hexdigest(),
            "objects": 1124, "q4_objects": 439,
            "source_object_plan_sha256": "a" * 64,
            "object_plan_sha256": "b" * 64,
            "source_file_identity": (source.stat().st_dev, source.stat().st_ino, source.stat().st_uid),
            "output_file_identity": (output.stat().st_dev, output.stat().st_ino, output.stat().st_uid),
        }
        return source, output, validation

    def test_common_receipt_round_trip_and_tamper_rejection(self) -> None:
        with TemporaryDirectory() as temporary:
            source, output, validation = self.fixture(Path(temporary))
            with patch.object(receipts.migration, "_validate_migration_pair",
                              return_value=validation):
                receipt, _ = receipts._payload(source, output)
            path = Path(str(output) + ".conversion.json")
            path.write_text(json.dumps(receipt))
            artifact = {"path": str(output), "bytes": 3,
                        "sha256": hashlib.sha256(b"n16").hexdigest(),
                        "weights_id": "r9700-q4g64-n16k16-eval"}
            with patch.object(receipts, "_validate_exact_plans",
                              return_value=("a" * 64, "b" * 64)):
                summary = receipts.validate_receipt(path, artifact)
            self.assertEqual(summary["object_plan_sha256"], "b" * 64)
            receipt["migration"]["q4_objects"] = 438
            path.write_text(json.dumps(receipt))
            with (patch.object(receipts, "_validate_exact_plans",
                               return_value=("a" * 64, "b" * 64)),
                  self.assertRaisesRegex(ValueError, "differs")):
                receipts.validate_receipt(path, artifact)

            receipt["migration"]["q4_objects"] = 439
            path.write_text(json.dumps(receipt))
            with (patch.object(receipts, "_validate_exact_plans",
                               return_value=("a" * 64, "c" * 64)),
                  self.assertRaisesRegex(ValueError, "differs")):
                receipts.validate_receipt(path, artifact)

    def test_publication_is_create_only(self) -> None:
        with TemporaryDirectory() as temporary:
            source, output, validation = self.fixture(Path(temporary))
            with (patch.object(receipts.migration, "_validate_migration_pair",
                               return_value=validation),
                  patch.object(receipts, "_validate_exact_plans",
                               return_value=("a" * 64, "b" * 64))):
                result = receipts.publish(source, output)
                self.assertEqual(result["recipe_id"], "r9700-all-q4g64-n16k16-eval-v1")
                with self.assertRaisesRegex(ValueError, "must not exist"):
                    receipts.publish(source, output)

    def test_validator_rejects_nonobject_receipt_and_ancestry(self) -> None:
        with TemporaryDirectory() as temporary:
            source, output, validation = self.fixture(Path(temporary))
            path = Path(str(output) + ".conversion.json")
            artifact = {"path": str(output), "bytes": 3,
                        "sha256": hashlib.sha256(b"n16").hexdigest(),
                        "weights_id": "r9700-q4g64-n16k16-eval"}
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "root must be an object"):
                receipts.validate_receipt(path, artifact)
            path.write_text(json.dumps({"migration": []}))
            with self.assertRaisesRegex(ValueError, "blocks must be objects"):
                receipts.validate_receipt(path, artifact)

            upstream = Path(str(source) + ".conversion.json")
            upstream.write_text("[]")
            with (patch.object(receipts.migration, "_validate_migration_pair",
                               return_value=validation),
                  self.assertRaisesRegex(ValueError, "root must be an object")):
                receipts._payload(source, output)

    def test_publication_leaf_comes_from_validated_lexical_output(self) -> None:
        with TemporaryDirectory() as temporary:
            source, output, validation = self.fixture(Path(temporary))

            class UnresolvableOutput:
                def resolve(self):
                    raise AssertionError("caller output must not derive the receipt leaf")

            with (patch.object(receipts.migration, "_validate_migration_pair",
                               return_value=validation),
                  patch.object(receipts, "_validate_exact_plans",
                               return_value=("a" * 64, "b" * 64))):
                result = receipts.publish(source, UnresolvableOutput())
            self.assertEqual(result["path"], str(output) + ".conversion.json")

    def test_late_owner_change_rolls_back_only_published_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            source, output, validation = self.fixture(Path(temporary))
            receipt_path = Path(str(output) + ".conversion.json")
            original_publish = receipts.migration._publish_json_create_only

            def publish_then_change(path, payload):
                result = original_publish(path, payload)
                output.write_bytes(b"changed")
                return result

            with (patch.object(receipts.migration, "_validate_migration_pair",
                               return_value=validation),
                  patch.object(receipts, "_validate_exact_plans",
                               return_value=("a" * 64, "b" * 64)),
                  patch.object(receipts.migration, "_publish_json_create_only",
                               side_effect=publish_then_change),
                  self.assertRaisesRegex(RuntimeError, "authorities changed")):
                receipts.publish(source, output)
            self.assertFalse(receipt_path.exists())

    def test_validator_reopens_exact_legacy_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            source, output, validation = self.fixture(Path(temporary))
            with patch.object(receipts.migration, "_validate_migration_pair",
                              return_value=validation):
                receipt, _ = receipts._payload(source, output)
            receipt_path = Path(str(output) + ".conversion.json")
            receipt_path.write_text(json.dumps(receipt))
            upstream_path = Path(str(source) + ".conversion.json")
            upstream = json.loads(upstream_path.read_text())
            upstream["recipe_id"] = "wrong-recipe"
            upstream_path.write_text(json.dumps(upstream))
            artifact = {"path": str(output), "bytes": 3,
                        "sha256": hashlib.sha256(b"n16").hexdigest(),
                        "weights_id": "r9700-q4g64-n16k16-eval"}
            with (patch.object(receipts, "_validate_exact_plans",
                               return_value=("a" * 64, "b" * 64)),
                  self.assertRaisesRegex(ValueError, "legacy conversion receipt differs")):
                receipts.validate_receipt(receipt_path, artifact)


if __name__ == "__main__":
    unittest.main()
