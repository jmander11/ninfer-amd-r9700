import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.bench.prepare_selected_vision_diagnostic import REPO, file_identity
from tools.bench.validate_selected_vision_diagnostic import validate
from tools.parity.qwen3_8_27b.vision_contract import EXPECTED_TRACE_NAMES, GATE, REPORT_FORMAT, CRITERIA, trace_shapes


class ValidateSelectedVisionDiagnosticTest(unittest.TestCase):
    def fixture(self, root: Path):
        prepared = root / "prepared-input.safetensors"
        prepared.write_bytes(b"prepared")
        tracer = root / "vision-tracer"
        tracer.write_bytes(b"tracer")
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
            "schema_version": 2, "status": "command_only_not_executed",
            "terminal_route": route, "source_checkpoint": source_identity,
            "frontend_python": python_front, "gpu_python": python_gpu,
            "fixture": {
                "messages": file_identity(REPO / "examples/cli/messages/image_chart.json"),
                "media": file_identity(REPO / "examples/cli/media/visual_chart.png"),
            },
            "prepared_input": {**file_identity(prepared), "contract": contract},
            "trace_executable": file_identity(tracer),
            "workload": {
                "maximum_concurrency": 1, "thinking": False, "prefix_reuse": False,
                "speculative_decode": False, "images": 1, "videos": 0,
                "capture_names": list(EXPECTED_TRACE_NAMES),
                "gate": GATE,
            },
            "outputs": {"raw": str(root / "vision.raw.json"),
                        "admission": str(root / "admission.json")},
        }
        source = {"path": "/source", "config_sha256": "d" * 64,
                  "index_sha256": "e" * 64, "indexed_tensor_count": 1199,
                  "shards": {"shard": {"bytes": 1}}}
        metric = {"rmse": 0.0, "relative_rmse": 0.0, "cosine": 1.0,
                  "max_absolute": 0.0, "actual_rms": 2.0, "reference_rms": 2.0,
                  "actual_finite": True, "reference_finite": True,
                  **{key: {"worst_scaled_rmse": 0.0, "worst_cosine": 1.0,
                           "worst_scaled_rmse_index": 0, "worst_cosine_index": 0}
                     for key in ("tokens", "features")}}
        rows = [{"item": 0, "name": name, "shape": shape, **metric}
                for name, shape in trace_shapes(1536, 384).items()]
        raw = {
            "format": REPORT_FORMAT, "passed": True, "criteria": CRITERIA,
            "artifact": {"path": route["artifact"]["path"], "sha256": route["artifact"]["sha256"],
                         "identity": {"model_id": "qwen3.8-27b", "weights_id": "r9700-q4g64-n16k16-eval"}},
            "model_dir": "/source",
            "source_provenance": {k: v for k, v in source.items() if k != "path"},
            "messages": plan["fixture"]["messages"]["path"],
            "messages_sha256": plan["fixture"]["messages"]["sha256"],
            "prepared_input": {"path": str(prepared), "sha256": plan["prepared_input"]["sha256"],
                               "contract": contract},
            "trace_executable": {"path": str(tracer), "sha256": plan["trace_executable"]["sha256"]},
            **{key: {"passed": True, "comparisons": rows} for key in (
                "production_vs_artifact_reference", "production_local_op_oracles", "artifact_vs_source_bf16")},
            **{key: {"passed": True} for key in (
                "preprocessing", "artifact_weights_vs_source_bf16", "source_manual_vs_hf_final")},
        }
        (root / "selection.json").write_text("{}")
        (root / "artifact.ninfer").write_bytes(b"artifact")
        (root / "prepared.sha256").write_text("closure")
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        (root / "vision.raw.json").write_text(json.dumps(raw))
        return plan_path, route, source_identity, python_front, python_gpu, source, prepared

    def test_accepts_complete_intermediate_validation(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            values = self.fixture(root)
            plan_path, route, source_identity, front, gpu, source, prepared = values
            with mock.patch(
                "tools.bench.validate_selected_vision_diagnostic.validate_closure",
                return_value={plan_path.resolve(): "x", Path(__file__).resolve(): "y",
                              prepared.resolve(): "z", (root / "vision-tracer").resolve(): "t"},
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
            self.assertEqual(result["status"], "complete_intermediate_validation")

    def check_rejected_report(self, mutate, error) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            values = self.fixture(root)
            plan_path, route, source_identity, front, gpu, source, prepared = values
            raw_path = root / "vision.raw.json"
            raw = json.loads(raw_path.read_text())
            mutate(raw)
            raw_path.write_text(json.dumps(raw))
            with mock.patch(
                "tools.bench.validate_selected_vision_diagnostic.validate_closure",
                return_value={plan_path.resolve(): "x", Path(__file__).resolve(): "y",
                              prepared.resolve(): "z", (root / "vision-tracer").resolve(): "t"},
            ), mock.patch(
                "tools.bench.validate_selected_vision_diagnostic.validate_source_receipt",
                return_value=source_identity,
            ), mock.patch(
                "tools.bench.validate_selected_vision_diagnostic.inspect_python",
                side_effect=(front, gpu),
            ), mock.patch(
                "tools.bench.validate_selected_vision_diagnostic._expected_source",
                return_value=source,
            ), self.assertRaisesRegex(ValueError, error):
                validate(plan_path, root, route_resolver=lambda _path: route)

    def test_rejects_nonfinite_capture(self) -> None:
        self.check_rejected_report(
            lambda raw: raw["production_vs_artifact_reference"]["comparisons"][15].update(cosine=float("nan")),
            "nonfinite")

    def test_rejects_false_pass_metrics_and_changed_criteria(self) -> None:
        for section in ("production_vs_artifact_reference", "production_local_op_oracles",
                        "artifact_vs_source_bf16"):
            with self.subTest(section=section):
                self.check_rejected_report(
                    lambda raw: raw[section]["comparisons"][15].update(relative_rmse=1.0),
                    "failed numerical criteria")
        self.check_rejected_report(
            lambda raw: raw["criteria"].update(local_relative_rmse=1.0), "criteria differ")
        self.check_rejected_report(
            lambda raw: raw["production_local_op_oracles"]["comparisons"][15]["tokens"].update(worst_scaled_rmse=1.0),
            "failed numerical criteria")
        self.check_rejected_report(
            lambda raw: raw["production_vs_artifact_reference"]["comparisons"].pop(),
            "capture inventory")

if __name__ == "__main__":
    unittest.main()
