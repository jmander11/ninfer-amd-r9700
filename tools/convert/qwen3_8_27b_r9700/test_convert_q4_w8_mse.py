"""Focused converter-boundary tests for the same-format Q4/W8 MSE control."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools.convert.qwen3.common.inventory import Q4, Q5, W8, tensor_spec
from tools.convert.qwen3_8_27b_r9700 import q4_w8_inventory, q4_w8_mse_inventory

try:
    import torch

    from tools.convert.qwen3_8_27b_r9700 import codec, convert_q4_w8_mse
except ModuleNotFoundError:
    torch = None


class Q4W8MseInventoryTest(unittest.TestCase):
    def test_identity_and_byte_plan_are_stable(self) -> None:
        self.assertEqual(q4_w8_mse_inventory.WEIGHTS_ID,
                         "r9700-q4-w8-mse-n16k16-eval")
        self.assertEqual(
            q4_w8_mse_inventory.RECIPE_ID,
            "r9700-source-q4-n16k16-promoted-w8-source-mse8-eval-v1",
        )
        self.assertEqual(q4_w8_mse_inventory.OBJECT_SPECS, q4_w8_inventory.OBJECT_SPECS)
        self.assertEqual(
            q4_w8_mse_inventory.FORMAT_ENCODED_BYTES,
            q4_w8_inventory.FORMAT_ENCODED_BYTES,
        )
        self.assertEqual(q4_w8_mse_inventory.TENSOR_ENCODED_BYTES, 22_868_177_312)
        self.assertEqual(q4_w8_mse_inventory.DEVICE_ARENA_BYTES, 22_868_191_232)


@unittest.skipIf(torch is None, "requires the project Torch environment")
class Q4W8MseConverterTest(unittest.TestCase):
    def test_checkpoint_receipt_normalizes_only_the_exact_integral_byte_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"metadata": {"total_size": 55562855904.0}}))
            shard = root / "model-00001-of-00018.safetensors"
            shard.write_bytes(b"source")
            with mock.patch.object(
                convert_q4_w8_mse.bf16_protocol,
                "validate_checkpoint_files",
                return_value={"tensor": shard.name},
            ):
                receipt = convert_q4_w8_mse._checkpoint_receipt(root)
                self.assertEqual(receipt["index_total_tensor_bytes"], 55_562_855_904)
                self.assertIs(type(receipt["index_total_tensor_bytes"]), int)
                index.write_text(json.dumps({"metadata": {"total_size": 55562855904.5}}))
                with self.assertRaisesRegex(ValueError, "byte count"):
                    convert_q4_w8_mse._checkpoint_receipt(root)

    @staticmethod
    def synthetic_preflight(*, shards: int = 18):
        objects = tuple(
            SimpleNamespace(to_json=lambda index=index: {"name": f"object-{index}"})
            for index in range(len(q4_w8_mse_inventory.OBJECT_SPECS))
        )
        return SimpleNamespace(
            model_dir=Path("synthetic-model"),
            config_summary={"architecture": "Qwen3_5ForConditionalGeneration"},
            source=SimpleNamespace(
                recipe_count=1118, source_tensor_count=1199,
                source_shard_count=shards, source_dtype_counts={"BF16": 1199},
            ),
            resources=tuple(
                SimpleNamespace(name=name, data=f"resource-{index}".encode())
                for index, name in enumerate(convert_q4_w8_mse.resources.OFFICIAL_RESOURCE_SHA256)
            ),
            draft=SimpleNamespace(n=131072),
            draft_ranking=SimpleNamespace(
                ranking_path=Path("ranking.i64"), ranking_sha256="1" * 64,
                sidecar_path=Path("ranking.i64.provenance.json"),
                sidecar_sha256="2" * 64, total_tokens=32768,
                distinct_token_ids=3933,
            ),
            object_plan=SimpleNamespace(objects=objects),
        )

    def test_preflight_summary_binds_exact_source_recipe_and_complete_write_plan(self) -> None:
        statistics = {"count": len(q4_w8_mse_inventory.OBJECT_SPECS)}
        checkpoint = {"indexed_tensor_count": 1199, "shards": [{"name": "s", "bytes": 1}]}
        with mock.patch.object(
            convert_q4_w8_mse.family_conversion, "object_statistics", return_value=statistics
        ), mock.patch.object(
            convert_q4_w8_mse, "_checkpoint_receipt", return_value=checkpoint
        ):
            summary = convert_q4_w8_mse.preflight_summary(self.synthetic_preflight())
        self.assertEqual(summary["source"], {
            "recipes": 1118, "tensors": 1199, "shards": 18,
            "dtypes": {"BF16": 1199},
        })
        self.assertEqual(summary["objects"], statistics)
        self.assertEqual(summary["checkpoint"], checkpoint)
        self.assertEqual(
            [row["name"] for row in summary["frontend_resources"]],
            list(convert_q4_w8_mse.resources.OFFICIAL_RESOURCE_SHA256),
        )
        self.assertTrue(all(len(row["sha256"]) == 64 for row in summary["frontend_resources"]))
        self.assertEqual(len(summary["object_plan_sha256"]), 64)
        self.assertEqual(summary["format_counts"]["Q4G64_F16S"], 183)
        self.assertEqual(summary["format_counts"]["W8G32_F16S"], 256)
        with mock.patch.object(convert_q4_w8_mse, "_checkpoint_receipt", return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, "18-shard"):
                convert_q4_w8_mse.preflight_summary(self.synthetic_preflight(shards=17))

    def test_preflight_only_is_machine_readable_and_never_selects_device_or_output(self) -> None:
        summary = {
            "artifact_type": "ninfer_qwen3_8_27b_r9700_q4_w8_mse_conversion_preflight",
            "identity": {"weights_id": q4_w8_mse_inventory.WEIGHTS_ID},
        }
        stdout = io.StringIO()
        with (
            mock.patch.object(convert_q4_w8_mse, "preflight_conversion", return_value=object()),
            mock.patch.object(convert_q4_w8_mse, "preflight_summary", return_value=summary),
            mock.patch.object(convert_q4_w8_mse, "pick_device") as pick_device,
            redirect_stdout(stdout),
        ):
            convert_q4_w8_mse.main([
                "--model", "synthetic-model", "--draft-ranking", "ranking.i64",
                "--preflight-only",
            ])
        self.assertEqual(json.loads(stdout.getvalue()), summary)
        pick_device.assert_not_called()

    def test_preflight_only_rejects_output_and_conversion_requires_output_device(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                convert_q4_w8_mse.main([
                    "--model", "synthetic-model", "--draft-ranking", "ranking.i64",
                    "--out", "forbidden.ninfer", "--preflight-only",
                ])
            with self.assertRaises(SystemExit):
                convert_q4_w8_mse.main([
                    "--model", "synthetic-model", "--draft-ranking", "ranking.i64",
                ])

    def test_dispatches_each_integer_format_to_its_exact_mse_codec(self) -> None:
        values = [-8.0] + [-1.0] * 63 + [float((index % 11) - 5) for index in range(65)]
        matrix = torch.tensor(values * 16, dtype=torch.bfloat16).reshape(16, 129)
        represented = matrix.to(dtype=torch.float32).reshape(-1).tolist()

        q4_spec = tensor_spec("test/q4", (16, 129), Q4)
        self.assertEqual(
            convert_q4_w8_mse._encode_tensor(matrix, q4_spec, torch.device("cpu")),
            codec.encode_q4g64_n16k16_reference(represented, 16, 129, refined=True),
        )

        w8_spec = tensor_spec("test/w8", (1, 129), W8)
        w8_matrix = matrix[:1]
        w8_represented = w8_matrix.to(dtype=torch.float32).reshape(-1).tolist()
        self.assertEqual(
            convert_q4_w8_mse._encode_tensor(w8_matrix, w8_spec, torch.device("cpu")),
            codec.encode_w8g32_mse_reference(w8_represented, 1, 129),
        )

    def test_rejects_non_bf16_and_unassigned_integer_formats(self) -> None:
        q4_spec = tensor_spec("test/q4", (1, 64), Q4)
        with self.assertRaisesRegex(TypeError, "source must be BF16"):
            convert_q4_w8_mse._encode_tensor(
                torch.zeros((1, 64), dtype=torch.float32), q4_spec, torch.device("cpu")
            )

        q5_spec = tensor_spec("test/q5", (1, 64), Q5)
        with self.assertRaisesRegex(ValueError, "requires Q4G64 or W8G32"):
            convert_q4_w8_mse._encode_tensor(
                torch.zeros((1, 64), dtype=torch.bfloat16), q5_spec, torch.device("cpu")
            )

    def test_report_declares_source_only_same_format_control(self) -> None:
        metadata = convert_q4_w8_mse._candidate_metadata()
        self.assertFalse(metadata["weight_recipe_selected"])
        self.assertTrue(metadata["same_format_and_byte_plan_as_reference"])
        self.assertEqual(metadata["layout_reference_weights_id"],
                         "r9700-q4-w8-n16k16-eval")
        self.assertEqual(metadata["format_counts"], q4_w8_inventory.FORMAT_COUNTS)
        objective = metadata["scale_objective"]
        self.assertEqual(objective["represented_input"], "original source BF16 weight values")
        self.assertEqual(objective["calibration_inputs"], "none")
        self.assertEqual(
            objective["excluded_inputs"],
            [
                "calibration activations",
                "draft-ranking frequencies or source corpus tokens",
                "PPL NLL or argmax sidecars",
                "GPU measurements",
            ],
        )


if __name__ == "__main__":
    unittest.main()
