#!/usr/bin/env python3

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench.prepare_selected_hardware_trace import _attach_hybrid_capture, prepare
from tools.bench.selected_loaded_code_objects import selected_loaded_fp8
from tools.bench.verify_selected_hardware_use import snapshot


class PrepareSelectedHardwareTraceTest(unittest.TestCase):
    def test_selected_fp8_dispatches_bind_exact_database_code_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); database = root / "trace.db"; capture = root / "capture"
            capture.mkdir()
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE kernels(dispatch_id INTEGER,kernel_id INTEGER,name TEXT,code_object_id INTEGER,"
                "start INTEGER,sgpr_count INTEGER,vgpr_count INTEGER,accum_vgpr_count INTEGER,"
                "lds_size INTEGER,scratch_size INTEGER);"
                "CREATE TABLE code_objects(id INTEGER,uri TEXT);"
                "INSERT INTO code_objects VALUES(7,'memory://11#offset=0x22&size=4');"
                "INSERT INTO kernels VALUES(1,17,'Cijk_Alik_Bljk_F8BS_exact.kd',7,10,128,192,0,25088,0);"
                "INSERT INTO kernels VALUES(2,17,'Cijk_Alik_Bljk_F8BS_exact.kd',7,20,128,192,0,25088,0);")
            connection.commit(); connection.close()
            selected = [
                {"rocprof_dispatch_id": 1, "symbol": "Cijk_Alik_Bljk_F8BS_exact.kd"},
                {"rocprof_dispatch_id": 2, "symbol": "Cijk_Alik_Bljk_F8BS_exact.kd"},
            ]
            code = {"uri": "memory://11#offset=0x22&size=4",
                    "source": {"path": str(root / "captured.co"), "bytes": 4,
                               "sha256": "unused"},
                    "sha256": "9f64a747e1b97f131fabb6b447296c9b6f0201e79fb3c5356e6c77e89b6a806a",
                    "bytes": 4}
            with patch("tools.bench.selected_loaded_code_objects._loaded_elf",
                       return_value=(b"\x01\x02\x03\x04", code)):
                result = selected_loaded_fp8(database, capture, selected)
            self.assertEqual(result[0]["dispatch_ids"], [1, 2])
            self.assertEqual(result[0]["kernel_id"], 17)
            self.assertEqual(result[0]["resources"], {
                "sgpr_count": 128, "architectural_vgpr_count": 192,
                "accum_vgpr_count": 0, "group_segment_lds_bytes": 25088,
                "private_segment_bytes": 0,
                "spill_counts": {"available": False, "sgpr": None, "vgpr": None},
            })
            selected.append({"rocprof_dispatch_id": 3,
                             "symbol": "Cijk_Alik_Bljk_F8BS_exact.kd"})
            with self.assertRaisesRegex(ValueError, "mapping is incomplete"):
                selected_loaded_fp8(database, capture, selected)

    def test_selected_fp8_dispatch_resource_variation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); database = root / "trace.db"; capture = root / "capture"
            capture.mkdir()
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE kernels(dispatch_id INTEGER,kernel_id INTEGER,name TEXT,code_object_id INTEGER,"
                "start INTEGER,sgpr_count INTEGER,vgpr_count INTEGER,accum_vgpr_count INTEGER,"
                "lds_size INTEGER,scratch_size INTEGER);"
                "CREATE TABLE code_objects(id INTEGER,uri TEXT);"
                "INSERT INTO code_objects VALUES(7,'memory://11#offset=0x22&size=4');"
                "INSERT INTO kernels VALUES(1,17,'Cijk_Alik_Bljk_F8BS_exact.kd',7,10,128,192,0,25088,0);"
                "INSERT INTO kernels VALUES(2,17,'Cijk_Alik_Bljk_F8BS_exact.kd',7,20,128,191,0,25088,0);")
            connection.commit(); connection.close()
            selected = [
                {"rocprof_dispatch_id": dispatch, "symbol": "Cijk_Alik_Bljk_F8BS_exact.kd"}
                for dispatch in (1, 2)
            ]
            with self.assertRaisesRegex(ValueError, "resources vary"):
                selected_loaded_fp8(database, capture, selected)

    def test_hybrid_capture_wraps_profiler_without_changing_benchmark_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "capture.so"; library.write_bytes(b"capture")
            plan = {"benchmark_command": ["bench", "--fixed"],
                    "profiler_command": ["rocprofv3", "--", "bench", "--fixed"]}
            (root / "plan.json").write_text(json.dumps(plan))
            (root / "commands.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n\n"
                "set +e\nrocprofv3 -- bench --fixed\n", encoding="utf-8")
            with patch("tools.bench.prepare_selected_hardware_trace.CAPTURE_LIBRARY", library):
                result = _attach_hybrid_capture(root, plan)
            self.assertEqual(result["benchmark_command"], ["bench", "--fixed"])
            self.assertEqual(result["profiler_command"][0], "/usr/bin/env")
            self.assertIn("LD_PRELOAD=", result["profiler_command"][1])
            command = (root / "commands.sh").read_text()
            self.assertIn("test ! -e", command)
            self.assertIn("NINFER_CODE_OBJECT_CAPTURE_DIR=", command)
            self.assertLess(command.index("test ! -e"), command.index("set +e"))

    def test_exact_terminal_route_builds_one_c1_p2048_trace_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_path = root / "selection.json"
            selection_path.write_text("{}", encoding="utf-8")
            artifact, bench, corpus = root / "model.ninfer", root / "ninfer_bench", root / "ids"
            artifact.write_text("artifact", encoding="utf-8")
            bench.write_text("bench", encoding="utf-8")
            corpus.write_text("tokens", encoding="utf-8")
            artifact_id = {**snapshot(artifact), "weights_id": "selected-weights"}
            bench_id = snapshot(bench)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "artifact": artifact_id, "bench": bench_id,
                "expected_kv_value_group": 32,
                "expected_xattention_profile": "b128-s16-tau900",
                "selected_prefill_chunk": 4096, "concurrency": [1, 2, 3, 4],
                "corpus": str(corpus), "corpus_sha256": snapshot(corpus)["sha256"],
                "corpus_tokens": 8192,
                "power_profile": {"required": "auto",
                                  "sysfs_path": "/sys/example/power_profile"},
            }), encoding="utf-8")
            selection = {"selected_prefill_chunk": 4096, "source_provenance": [{
                "candidate": "winner", "matrices": {"pareto-whole": {
                    "path": str(manifest_path), "sha256": snapshot(manifest_path)["sha256"]}},
            }]}
            route = {"winner": "winner", "artifact": artifact_id, "weights_id": "selected-weights",
                     "executable": bench_id, "kv_value_group": 32,
                     "xattention_profile": "b128-s16-tau900", "prefill_chunk": 4096,
                     "terminal_selection": snapshot(selection_path)}
            terminal = {"winner": "winner", "winner_artifact": artifact_id,
                        "winner_cache_profile": {"value_group": 32},
                        "winner_execution_profile": {
                            "xattention_profile": "b128-s16-tau900"}}
            with patch("tools.bench.prepare_selected_hardware_trace.load_payload",
                       return_value=selection), patch(
                "tools.bench.prepare_selected_hardware_trace.validate_terminal_production_authority",
                return_value=(terminal, {})), patch(
                "tools.bench.prepare_selected_hardware_trace.selected_route", return_value=route), patch(
                "tools.bench.prepare_selected_hardware_trace._write_plan",
                side_effect=lambda out, **kwargs: {"out": out, **kwargs}), patch(
                "tools.bench.prepare_selected_hardware_trace.file_sha256",
                side_effect=lambda path: snapshot(Path(path))["sha256"]):
                result = prepare(selection_path, root / "out")
            command = result["benchmark_command"]
            self.assertEqual(command[command.index("-p") + 1], "2048")
            self.assertEqual(command[command.index("--concurrency") + 1], "1")
            self.assertEqual(command[command.index("--draft-tokens") + 1], "0")
            self.assertEqual(command[command.index("--prefill-chunk") + 1], "4096")
            self.assertEqual(result["workload"]["xattention_profile"], "b128-s16-tau900")


if __name__ == "__main__":
    unittest.main()
