"""Dependency-light tests for recoverable DFlash2 conversion publication."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec, plan_objects
from tools.convert.qwen3_8_27b_r9700 import convert_dflash2_q4 as conversion
from tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 import (
    _atomic_json,
    _sha256,
    _validate_completed_artifact,
    preflight_summary,
)


class DFlash2ConversionPublicationTest(unittest.TestCase):
    def test_recipe_conversion_copies_base_and_bf16_payloads_exactly(self) -> None:
        # Exercise the real container/publication path with small represented payloads.
        # Numeric encoders have their independent exact oracle in test_dflash2_matrix_recipes.
        inv = conversion.inventory
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.ninfer"
            identity = ArtifactIdentity(inv.MODEL_ID, inv.ALL_Q4_BASE_WEIGHTS_ID)
            base_specs = (
                TensorSpec("text/draft_head", (2, 2), "BF16", "contiguous-le-v1"),
                TensorSpec("text/draft_head_token_ids", (2,), "I32", "contiguous-le-v1"),
            )
            base_payloads = (b"\x80\x3f\x00\x40\x40\x40\x80\x40", struct.pack("<2i", 9, 4))
            with ArtifactWriter(base, identity, base_specs) as writer:
                for spec, payload in zip(base_specs, base_payloads, strict=True):
                    writer.write(spec.name, payload)
            torch = ModuleType("torch")
            torch.__version__ = "fixture"
            torch.version = MagicMock(hip="fixture")
            quantize = ModuleType("tools.convert.common.quantize")
            quantize.pick_device = lambda device: device
            safetensors = ModuleType("tools.convert.common.safetensors")
            safetensors.ShardReader = MagicMock()
            family = ModuleType("tools.convert.qwen3.common.conversion")
            preserved = b"\x80\x3f\x00\x40"
            family.encode_tensor_payload = MagicMock(return_value=preserved)
            for recipe in conversion.dflash2_matrix_recipes.RECIPES:
                output = root / (recipe.key + ".ninfer")
                matrix = conversion.dflash2_matrix_recipes.tensor_spec(
                    "dflash/feature_projection", (16, 128), recipe.matrix_format)
                codebook = conversion.TensorSpec("dflash/selector/predecessor_codebook",
                    (2,), "BF16", "contiguous-le-v1")
                bindings = (inv.SourceBinding(matrix, ("matrix",)),
                            inv.SourceBinding(codebook, ("codebook",)))
                specs = base_specs + tuple(TensorSpec(spec.name, spec.shape, spec.format,
                    spec.layout) for spec in (matrix, codebook))
                objects = plan_objects(specs)
                out_identity = ArtifactIdentity(inv.MODEL_ID,
                    inv.companion_weights_id(identity.weights_id, recipe.key))
                file_bytes = conversion.align_up(16 + len(conversion.encode_directory(
                    out_identity, objects)), 4096) + objects[-1].offset + objects[-1].bytes
                checked = conversion.Preflight(base, root, {}, identity, out_identity,
                    specs, objects, file_bytes, 0, recipe.key)
                matrix_payload = bytes((index % 251 for index in range(objects[2].bytes)))
                with patch.dict(sys.modules, {"torch": torch,
                    "tools.convert.common.quantize": quantize,
                    "tools.convert.common.safetensors": safetensors,
                    "tools.convert.qwen3.common.conversion": family}), \
                    patch.object(conversion, "preflight", return_value=checked), \
                    patch.object(conversion, "_base_authority", return_value=None), \
                    patch.object(conversion, "_load_source_tensor", return_value=object()), \
                    patch.object(inv, "source_bindings_for_recipe", return_value=bindings), \
                    patch.object(conversion.dflash2_matrix_recipes, "encode_matrix_payload",
                                 return_value=matrix_payload) as encoder:
                    report_path = conversion.convert(base, root, output, device="cpu",
                                                     matrix_recipe=recipe.key)
                self.assertEqual(encoder.call_args.args[2], recipe.key)
                with conversion.Artifact.open(output) as artifact:
                    self.assertEqual(artifact.identity, out_identity)
                    for obj, payload in zip(artifact.objects[:2], base_payloads, strict=True):
                        self.assertEqual(bytes(artifact.payload(obj)), payload)
                    self.assertEqual(bytes(artifact.payload(artifact.objects[2])), matrix_payload)
                    self.assertEqual(bytes(artifact.payload(artifact.objects[3])), preserved)
                report = json.loads(report_path.read_text())
                self.assertEqual(report["recipe_id"], recipe.recipe_id)
                self.assertEqual(report["base"]["payload_copy"], "byte_exact")
                self.assertFalse(conversion._pending_path(output).exists())

    def test_all_recipe_plans_bind_distinct_identities_and_storage(self) -> None:
        inv = conversion.inventory
        identities = set()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for base_id in (inv.ALL_Q4_BASE_WEIGHTS_ID, inv.MIXED_BASE_WEIGHTS_ID,
                            inv.HYBRID_BASE_WEIGHTS_ID):
                base = root / (base_id + ".ninfer")
                base_identity = ArtifactIdentity(inv.MODEL_ID, base_id)
                fixture = TensorSpec("fixture", (1,), "BF16", "contiguous-le-v1")
                expected_base = conversion.TensorSpec(
                    fixture.name, fixture.shape, fixture.format, fixture.layout)
                with ArtifactWriter(base, base_identity, (fixture,)) as writer:
                    writer.write("fixture", b"\x80\x3f")
                for recipe in conversion.dflash2_matrix_recipes.RECIPES:
                    with patch.object(inv, "validate_source", return_value={}), patch.object(
                        conversion, "_expected_base",
                        return_value=((expected_base,), base_identity, inv.TENSOR_ENCODED_BYTES + 256)
                    ):
                        checked = conversion.preflight(base, root, recipe.key)
                    identities.add(checked.output_identity.weights_id)
                    expected = inv.matrix_recipe_summary(recipe.key)
                    self.assertEqual(checked.matrix_recipe, recipe.key)
                    self.assertEqual(checked.projected_device_arena_bytes,
                        256 + expected["tensor_encoded_bytes"])
                    appended = checked.objects[1:]
                    self.assertEqual(sum(obj.bytes for obj in appended),
                                     expected["tensor_encoded_bytes"])
                    self.assertEqual(sum(obj.format == recipe.matrix_format
                                         for obj in appended), 32)
                    self.assertEqual(sum(obj.format == "BF16" for obj in appended), 34)
                    self.assertEqual(checked.output_identity.weights_id,
                                     inv.companion_weights_id(base_id, recipe.key))
            self.assertEqual(len(identities), 9)
            self.assertIn(inv.ALL_Q4_WEIGHTS_ID, identities)
            self.assertIn(inv.MIXED_WEIGHTS_ID, identities)
            self.assertIn(inv.HYBRID_WEIGHTS_ID, identities)

    def test_hybrid_base_authority_binds_conversion_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "hybrid.ninfer"
            base.write_bytes(b"hybrid-base")
            base_sha256 = _sha256(base)
            receipt = {
                "path": str(base) + ".conversion.json", "sha256": "0" * 64,
                "recipe_id": conversion.fp8_hybrid_inventory.RECIPE_ID,
                "selection_sha256": conversion.fp8_hybrid_inventory.SELECTION_SHA256,
                "object_plan_sha256": "1" * 64, "source_index_sha256": "2" * 64,
                "source_ranking_sha256": "3" * 64, "source_artifact_sha256": "4" * 64,
                "source_receipt_sha256": "5" * 64, "transcoder_sha256": "6" * 64,
            }
            with patch("tools.ppl.run.validate_n16_conversion_receipt", return_value=receipt):
                authority = conversion._base_authority(
                    base,
                    ArtifactIdentity(
                        conversion.inventory.MODEL_ID,
                        conversion.inventory.HYBRID_BASE_WEIGHTS_ID,
                    ),
                    base_sha256,
                )
            self.assertEqual(authority["selection_sha256"], receipt["selection_sha256"])
            self.assertEqual(authority["receipt"]["sha256"], receipt["sha256"])

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
            self.assertEqual(
                tuple(
                    recipe["key"]
                    for recipe in result["dflash_plan"]["candidate_matrix_recipes"]
                ),
                ("source-mse-q4g64", "source-mse-w8g32"),
            )
            self.assertEqual(
                result["dflash_plan"]["recipe"]["materialization"],
                "recipe-aware-converter-route",
            )
            self.assertTrue(
                all(
                    recipe["materialization"] == "recipe-aware-converter-route"
                    for recipe in result["dflash_plan"]["candidate_matrix_recipes"]
                )
            )
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
                    "dflash_matrix_recipe": conversion.inventory.matrix_recipe_summary(
                        conversion._CANONICAL_RECIPE
                    ),
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

            # Recovery must not relabel canonical payloads as either source-MSE recipe,
            # even though canonical and MSE Q4 share exactly the same storage format.
            for recipe in conversion.dflash2_matrix_recipes.RECIPES[1:]:
                selected = replace(checked, matrix_recipe=recipe.key)
                _atomic_json(pending_path, {
                    "schema": 1, "identity": conversion._identity_record(identity),
                    "base": {"identity": conversion._identity_record(base_identity),
                             "sha256": _sha256(base_path)},
                    "dflash_source": selected.source,
                    "dflash_matrix_recipe": conversion.inventory.matrix_recipe_summary(
                        conversion._CANONICAL_RECIPE),
                })
                with patch.object(conversion, "preflight", return_value=selected):
                    with self.assertRaisesRegex(ValueError, "receipt differs"):
                        conversion.finalize_report(base_path, root / "source", output,
                                                   matrix_recipe=recipe.key)
                recipe_report = conversion._report_value(selected, output,
                    _sha256(base_path), output_sha256, {})
                self.assertEqual(recipe_report["recipe_id"], recipe.recipe_id)
                self.assertEqual(recipe_report["dflash_recipe"]["matrix_format"],
                                 recipe.matrix_format)
                self.assertFalse(recipe_report["weight_recipe_selected"])


if __name__ == "__main__":
    unittest.main()
