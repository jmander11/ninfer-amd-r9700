import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.bench.prepare_selected_vision_diagnostic import REPO, file_identity
from tools.bench.validate_selected_vision_diagnostic import validate


class ValidateSelectedVisionDiagnosticTest(unittest.TestCase):
    def fixture(self, root: Path):
        prepared = root / "prepared-input.safetensors"
        prepared.write_bytes(b"prepared")
        route = {
            "terminal_selection": {"path": str(root / "selection.json"), "sha256": "a" * 64},
            "artifact": {"path": str(root / "artifact.ninfer"), "sha256": "b" * 64,
                         "model_id": "qwen3.8-27b", "weights_id": "r9700-q4g64-n16k16-eval"},
            "maximum_runtime_concurrency": 4,
        }
        contract = {
            "prompt_length": 428, "image_tokens": 384, "video_tokens": 0,
            "image_grid_thw": [[1, 32, 48]], "pixel_values_shape": [1536, 1536],
            "rope_delta": -360, "thinking": False, "images": 1, "videos": 0,
        }
        source_identity = {"receipt": {"path": "receipt", "sha256": "c" * 64},
                           "source": "/source"}
        python_front = {"launcher": {"path": "front"}}
        python_gpu = {"launcher": {"path": "gpu"}}
        plan = {
            "artifact_type": "ninfer_r9700_selected_vision_diagnostic_plan",
            "schema_version": 1, "status": "command_only_not_executed",
            "terminal_route": route, "source_checkpoint": source_identity,
            "frontend_python": python_front, "gpu_python": python_gpu,
            "fixture": {
                "messages": file_identity(REPO / "examples/cli/messages/image_chart.json"),
                "media": file_identity(REPO / "examples/cli/media/visual_chart.png"),
            },
            "prepared_input": {**file_identity(prepared), "contract": contract},
            "workload": {
                "maximum_concurrency": 1, "thinking": False, "prefix_reuse": False,
                "speculative_decode": False, "images": 1, "videos": 0,
                "capture_names": ["block_00", "block_13", "block_26", "merger"],
                "gate": "diagnostic completion with finite exact-shape comparisons; no numeric threshold",
            },
            "outputs": {"raw": str(root / "vision.raw.json"),
                        "admission": str(root / "admission.json")},
        }
        source = {"path": "/source", "config_sha256": "d" * 64,
                  "index_sha256": "e" * 64, "indexed_tensor_count": 1199,
                  "shards": {"shard": {"bytes": 1}}}
        metric = {"rmse": 1.0, "relative_rmse": 0.1, "cosine": 0.99,
                  "actual_norm": 2.0, "reference_norm": 2.1}
        raw = {
            "format": "ninfer_vision_bf16_comparison_v3", "scope": "diagnostic-only",
            "execution": {"maximum_concurrency": 1, "thinking": False,
                          "prefix_reuse": False, "speculative_decode": False},
            "artifact": {"path": route["artifact"]["path"],
                         "sha256": route["artifact"]["sha256"],
                         "identity": {"model_id": "qwen3.8-27b",
                                      "weights_id": "r9700-q4g64-n16k16-eval"}},
            "source": source,
            "input": {"messages_path": plan["fixture"]["messages"]["path"],
                      "messages_sha256": plan["fixture"]["messages"]["sha256"],
                      "thinking": False,
                      "prepared_input": {"path": str(prepared),
                                         "sha256": plan["prepared_input"]["sha256"],
                                         "contract": contract}},
            "image_grid_thw": [[1, 32, 48]], "video_grid_thw": None,
            "vision": {"images": 1, "videos": 0, "raw_patches": 1536,
                       "llm_tokens": 384, "attention_pairs": 2359296},
            "comparisons": {
                **{name: {"shape": [1536, 1152], **metric}
                   for name in ("block_00", "block_13", "block_26")},
                "merger": {"shape": [384, 5120], **metric},
            },
        }
        (root / "selection.json").write_text("{}")
        (root / "artifact.ninfer").write_bytes(b"artifact")
        (root / "prepared.sha256").write_text("closure")
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        (root / "vision.raw.json").write_text(json.dumps(raw))
        return plan_path, route, source_identity, python_front, python_gpu, source, prepared

    def test_accepts_finite_exact_shape_diagnostic_without_threshold(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO / "build-r9700") as directory:
            root = Path(directory)
            values = self.fixture(root)
            plan_path, route, source_identity, front, gpu, source, prepared = values
            with mock.patch(
                "tools.bench.validate_selected_vision_diagnostic.validate_closure",
                return_value={plan_path.resolve(): "x", Path(__file__).resolve(): "y",
                              prepared.resolve(): "z"},
            ), mock.patch(
                "tools.bench.validate_selected_vision_diagnostic.validate_source_receipt",
                return_value=source_identity,
            ), mock.patch(
                "tools.bench.validate_selected_vision_diagnostic.inspect_python",
                side_effect=(front, gpu),
            ), mock.patch(
                "tools.bench.validate_selected_vision_diagnostic._expected_source",
                return_value=source,
            ):
                result = validate(plan_path, root, route_resolver=lambda _path: route)
            self.assertEqual(result["status"], "complete_diagnostic_no_numeric_threshold")

    def test_rejects_nonfinite_capture(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO / "build-r9700") as directory:
            root = Path(directory)
            values = self.fixture(root)
            plan_path, route, source_identity, front, gpu, source, prepared = values
            raw_path = root / "vision.raw.json"
            raw = json.loads(raw_path.read_text())
            raw["comparisons"]["block_13"]["cosine"] = float("nan")
            raw_path.write_text(json.dumps(raw))
            with mock.patch(
                "tools.bench.validate_selected_vision_diagnostic.validate_closure",
                return_value={plan_path.resolve(): "x", Path(__file__).resolve(): "y",
                              prepared.resolve(): "z"},
            ), mock.patch(
                "tools.bench.validate_selected_vision_diagnostic.validate_source_receipt",
                return_value=source_identity,
            ), mock.patch(
                "tools.bench.validate_selected_vision_diagnostic.inspect_python",
                side_effect=(front, gpu),
            ), mock.patch(
                "tools.bench.validate_selected_vision_diagnostic._expected_source",
                return_value=source,
            ), self.assertRaisesRegex(ValueError, "nonfinite"):
                validate(plan_path, root, route_resolver=lambda _path: route)


if __name__ == "__main__":
    unittest.main()
