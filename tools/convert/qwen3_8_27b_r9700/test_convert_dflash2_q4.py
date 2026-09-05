"""Dependency-light tests for recoverable DFlash2 conversion publication."""

from __future__ import annotations

import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec, plan_objects
from tools.convert.qwen3_8_27b_r9700 import convert_dflash2_q4 as conversion
from tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 import (
    _atomic_json,
    _sha256,
    _validate_completed_artifact,
    preflight_summary,
)


class DFlash2ConversionPublicationTest(unittest.TestCase):
    def test_hybrid_base_authority_binds_conversion_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "hybrid.ninfer"
            base.write_bytes(b"hybrid-base")
            base_sha256 = _sha256(base)
            receipt = {
                "identity": {
                    "model_id": conversion.inventory.MODEL_ID,
                    "weights_id": conversion.inventory.HYBRID_BASE_WEIGHTS_ID,
                },
                "target_key": conversion.inventory.TARGET_KEY,
                "recipe_id": conversion.fp8_hybrid_inventory.RECIPE_ID,
                "candidate": {
                    "selection_sha256":
                        conversion.fp8_hybrid_inventory.SELECTION_SHA256,
                    "object_plan_sha256": "1" * 64,
                },
                "source": {"index_sha256": "2" * 64, "ranking_sha256": "3" * 64},
                "artifact": {
                    "path": str(base), "bytes": base.stat().st_size, "sha256": base_sha256,
                },
            }
            receipt_path = Path(str(base) + ".conversion.json")
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            authority = conversion._base_authority(
                base,
                ArtifactIdentity(
                    conversion.inventory.MODEL_ID,
                    conversion.inventory.HYBRID_BASE_WEIGHTS_ID,
                ),
                base_sha256,
            )
            self.assertEqual(authority["selection_sha256"], receipt["candidate"]["selection_sha256"])
            self.assertEqual(authority["receipt"]["sha256"], _sha256(receipt_path))
            receipt["artifact"]["sha256"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "authority differs"):
                conversion._base_authority(
                    base,
                    ArtifactIdentity(
                        conversion.inventory.MODEL_ID,
                        conversion.inventory.HYBRID_BASE_WEIGHTS_ID,
                    ),
                    base_sha256,
                )

    def test_hybrid_base_maps_to_its_own_companion(self) -> None:
        expected, output, arena = conversion._expected_base(ArtifactIdentity(
            conversion.inventory.MODEL_ID,
            conversion.inventory.HYBRID_BASE_WEIGHTS_ID,
        ))
        self.assertIs(expected, conversion.fp8_hybrid_inventory.OBJECT_SPECS)
        self.assertEqual(output.weights_id, conversion.inventory.HYBRID_WEIGHTS_ID)
        self.assertEqual(arena, conversion.inventory.HYBRID_DEVICE_ARENA_BYTES)

    def test_preflight_summary_binds_base_and_publication_paths(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.ninfer"
            base.write_bytes(b"exact-base")
            output = root / "planned.ninfer"
            identity = ArtifactIdentity("fixture-model", "fixture-base")
            checked = conversion.Preflight(
                base_path=base,
                dflash_model=root / "source",
                source={"safetensors_sha256": "1" * 64},
                base_identity=identity,
                output_identity=ArtifactIdentity("fixture-model", "fixture-dflash"),
                specs=(),
                objects=(),
                projected_file_bytes=123,
                projected_device_arena_bytes=456,
            )
            result = preflight_summary(checked, output)
            self.assertEqual(result["base"]["sha256"], _sha256(base))
            self.assertEqual(len(result["combined_plan"]["sha256"]), 64)
            self.assertEqual(result["publication"]["artifact_path"], str(output.resolve()))
            self.assertFalse(result["publication"]["destination_exists"])
            output.write_bytes(b"occupied")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                preflight_summary(checked, output)

    def test_atomic_report_and_completed_artifact_validation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "dflash-eval.ninfer"
            identity = ArtifactIdentity("fixture-model", "fixture-dflash")
            specs = (TensorSpec("weight", (4,), "I32", "contiguous-le-v1"),)
            with ArtifactWriter(artifact_path, identity, specs) as writer:
                writer.write("weight", struct.pack("<4i", 1, -2, 3, -4))

            objects = plan_objects(specs)
            expected_sha256 = _sha256(artifact_path)
            self.assertEqual(
                _validate_completed_artifact(
                    artifact_path,
                    identity,
                    objects,
                    artifact_path.stat().st_size,
                    expected_sha256,
                ),
                expected_sha256,
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _validate_completed_artifact(
                    artifact_path,
                    identity,
                    objects,
                    artifact_path.stat().st_size,
                    "0" * 64,
                )

            report_path = root / "dflash-eval.ninfer.conversion.json"
            _atomic_json(report_path, {"artifact": {"sha256": expected_sha256}})
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8")),
                {"artifact": {"sha256": expected_sha256}},
            )
            self.assertEqual(list(root.glob(f".{report_path.name}.*.tmp")), [])

    def test_finalize_report_recovers_from_completed_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = root / "base.ninfer"
            base_path.write_bytes(b"exact-base-artifact")
            output = root / "dflash-eval.ninfer"
            identity = ArtifactIdentity("fixture-model", "fixture-dflash")
            base_identity = ArtifactIdentity("fixture-model", "fixture-base")
            specs = (TensorSpec("weight", (4,), "I32", "contiguous-le-v1"),)
            with ArtifactWriter(output, identity, specs) as writer:
                writer.write("weight", struct.pack("<4i", 1, -2, 3, -4))
            output_sha256 = _sha256(output)
            checked = conversion.Preflight(
                base_path=base_path,
                dflash_model=root / "source",
                source={"safetensors_sha256": "1" * 64},
                base_identity=base_identity,
                output_identity=identity,
                specs=specs,
                objects=plan_objects(specs),
                projected_file_bytes=output.stat().st_size,
                projected_device_arena_bytes=16,
            )
            pending_path = conversion._pending_path(output)
            _atomic_json(
                pending_path,
                {
                    "schema": 1,
                    "status": "artifact_complete",
                    "identity": conversion._identity_record(identity),
                    "base": {
                        "identity": conversion._identity_record(base_identity),
                        "sha256": _sha256(base_path),
                    },
                    "dflash_source": checked.source,
                    "artifact": {"sha256": output_sha256},
                    "converter": {"mode": "convert", "device_resolved": "cuda"},
                },
            )
            with patch.object(conversion, "preflight", return_value=checked):
                report_path = conversion.finalize_report(
                    base_path,
                    root / "source",
                    output,
                    expected_output_sha256=output_sha256,
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["base"]["sha256"], _sha256(base_path))
            self.assertEqual(report["artifact"]["sha256"], output_sha256)
            self.assertTrue(report["converter"]["report_finalized_by_resume"])
            self.assertFalse(pending_path.exists())


if __name__ == "__main__":
    unittest.main()
