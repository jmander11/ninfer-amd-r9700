import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.bench.prepare_selected_vision_diagnostic import REPO, prepare


class PrepareSelectedVisionDiagnosticTest(unittest.TestCase):
    def test_prepares_one_bound_command_package(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO / "build-r9700") as directory:
            root = Path(directory)
            selection = root / "selection.json"
            artifact = root / "artifact.ninfer"
            capacity = root / "capacity.json"
            whole = root / "whole.json"
            cache = root / "CMakeCache.txt"
            ctest = root / "CTestTestfile.cmake"
            for path in (selection, artifact, capacity, whole, cache, ctest):
                path.write_text(path.name)
            output = root / "campaign"
            route = {
                "terminal_selection": {"path": str(selection), "sha256": "a" * 64},
                "artifact": {"path": str(artifact), "model_id": "qwen3.8-27b",
                             "weights_id": "r9700-q4g64-n16k16-eval", "sha256": "b" * 64},
                "source_matrices": {
                    "pareto-capacity": {"path": str(capacity), "sha256": "c" * 64},
                    "pareto-whole": {"path": str(whole), "sha256": "d" * 64},
                },
                "build_identity": {
                    "cmake_cache": {"path": str(cache), "sha256": "e" * 64},
                    "ctest_root": {"path": str(ctest), "sha256": "f" * 64},
                },
                "cache_profile": {"value_group": 16},
                "execution_profile": {"xattention_profile": "dense"},
                "selected_prefill_chunk": 4096,
                "hybrid_width_tool": None,
                "maximum_runtime_concurrency": 4,
            }
            contract = {
                "prompt_length": 428, "image_tokens": 384, "video_tokens": 0,
                "image_grid_thw": [[1, 32, 48]], "pixel_values_shape": [1536, 1536],
                "rope_delta": -360, "thinking": False, "images": 1, "videos": 0,
            }

            def run(command, **_kwargs):
                out = Path(command[command.index("--out") + 1])
                out.write_bytes(b"prepared-input")
                return subprocess.CompletedProcess(command, 0, json.dumps(contract) + "\n", "")

            with mock.patch(
                "tools.bench.prepare_selected_vision_diagnostic.resolve_route",
                return_value=route,
            ), mock.patch(
                "tools.bench.prepare_selected_vision_diagnostic.validate_source_receipt",
                return_value={"receipt": {"path": "receipt", "bytes": 1, "sha256": "1" * 64},
                              "source": "/source"},
            ), mock.patch(
                "tools.bench.prepare_selected_vision_diagnostic.inspect_python",
                side_effect=({"launcher": {"path": "front"}}, {"launcher": {"path": "gpu"}}),
            ), mock.patch(
                "tools.bench.prepare_selected_vision_diagnostic.subprocess.run", side_effect=run
            ):
                plan = prepare(selection, output)
            self.assertEqual(plan["terminal_route"], route)
            self.assertEqual(plan["prepared_input"]["contract"], contract)
            self.assertTrue((output / "commands.sh").is_file())
            self.assertTrue((output / "prepared.sha256").is_file())

    def test_rejects_dangling_output_before_route_resolution(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO / "build-r9700") as directory:
            root = Path(directory)
            selection = root / "selection.json"
            selection.write_text("{}")
            output = root / "campaign"
            output.symlink_to(root / "missing")
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(selection, output)
            self.assertTrue(output.is_symlink())


if __name__ == "__main__":
    unittest.main()
