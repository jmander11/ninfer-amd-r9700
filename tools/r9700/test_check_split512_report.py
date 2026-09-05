from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.r9700.check_split512_report import (
    SOURCE_PATHS,
    SHAPES,
    split_workspace_bytes,
    validate_report,
)


def report() -> dict:
    iterations = 20
    measurements = []
    for group, context, rows, active in sorted(SHAPES):
        incumbent = [2.0] * iterations
        split = [1.0] * iterations
        measurements.append({
            "value_group": group,
            "context": context,
            "query_rows": rows,
            "active_rows": active,
            "iterations": iterations,
            "split_workspace_bytes": split_workspace_bytes(rows, context),
            "incumbent_route": (
                "fp8-q-wmma-score-plus-fp32-softmax-int4-pv" if rows == 1
                else "bf16-q-score-streaming-online-fp32-softmax-int4-pv"
            ),
            "incumbent_mean_ms": 2.0,
            "split_mean_ms": 1.0,
            "incumbent_over_split": 2.0,
            "incumbent_samples_ms": incumbent,
            "split_samples_ms": split,
        })
    maximum_context = 262144
    return {
        "schema": "ninfer_r9700_split512_attention_qualification",
        "schema_version": 2,
        "architecture": "gfx1201",
        "disposition": "unpromoted-qualification-candidate",
        "physical_qualification": {
            "status": "passed", "scope": "full-numerical-rejection-device-graph"
        },
        "power_profile": {
            "path": "/sys/class/drm/card2/device/power_dpm_force_performance_level",
            "value": "auto", "required": "auto",
        },
        "device": {
            "name": "AMD Radeon AI PRO R9700", "architecture": "gfx1201", "wave_size": 32,
            "runtime_version": 70000000, "driver_version": 70000000,
        },
        "executable": {"path": "/repo/build/split512", "sha256": "a" * 64},
        "source_root": "/repo",
        "sources": [
            {"role": role, "path": "/repo/" + relative,
             "sha256": f"{index:x}" * 64}
            for index, (role, relative) in enumerate(SOURCE_PATHS.items(), 1)
        ],
        "workspace_contract": {
            "status": "passed",
            "maximum_context": maximum_context,
            "t1_maximum_bytes": split_workspace_bytes(1, maximum_context),
            "t4_maximum_bytes": split_workspace_bytes(4, maximum_context),
            "maximum_plus_one_context": maximum_context + 1,
            "t1_maximum_plus_one_bytes": 0,
            "t4_maximum_plus_one_bytes": 0,
            "overflow_context": (1 << 64) - 1,
            "t1_overflow_bytes": 0,
            "t4_overflow_bytes": 0,
        },
        "sample_order": "alternating-interleaved",
        "warmup_iterations_per_route": 3,
        "incumbent_scope": "complete-production-semantic-attention-leaf",
        "split_scope": "complete-qk-plus-partial-softmax-pv-plus-merge-leaf",
        "measurements": measurements,
    }


class Split512ReportTest(unittest.TestCase):
    @staticmethod
    def validate(value: dict) -> None:
        digests = {value["executable"]["path"]: value["executable"]["sha256"]}
        digests.update({source["path"]: source["sha256"] for source in value["sources"]})
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.is_dir", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             patch("tools.r9700.check_split512_report._file_sha256",
                   side_effect=lambda path: digests[str(path)]):
            validate_report(value, Path("/repo/build/split512"), Path("/repo"),
                            lambda _path: "auto")

    def test_accepts_complete_report(self) -> None:
        self.validate(report())

    def test_rejects_missing_shape(self) -> None:
        value = report()
        value["measurements"].pop()
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.validate(value)

    def test_rejects_non_auto_profile(self) -> None:
        value = report()
        value["power_profile"]["value"] = "profile_standard"
        with self.assertRaisesRegex(ValueError, "auto"):
            self.validate(value)

    def test_rejects_non_auto_live_power(self) -> None:
        value = report()
        with self.assertRaisesRegex(ValueError, "live power"):
            validate_report(value, Path("/repo/build/split512"), Path("/repo"),
                            lambda _path: "profile_standard")

    def test_rejects_recomputed_speedup_drift(self) -> None:
        value = report()
        value["measurements"][0]["incumbent_over_split"] = 1.0
        with self.assertRaisesRegex(ValueError, "raw samples"):
            self.validate(value)

    def test_rejects_workspace_drift(self) -> None:
        value = report()
        value["measurements"][0]["split_workspace_bytes"] += 4
        with self.assertRaisesRegex(ValueError, "workspace"):
            self.validate(value)

    def test_rejects_workspace_boundary_drift(self) -> None:
        value = report()
        value["workspace_contract"]["t4_maximum_bytes"] -= 4
        with self.assertRaisesRegex(ValueError, "boundary"):
            self.validate(value)

    def test_rejects_duplicate_shape(self) -> None:
        value = report()
        value["measurements"][-1] = copy.deepcopy(value["measurements"][0])
        with self.assertRaisesRegex(ValueError, "shape"):
            self.validate(value)

    def test_rejects_noncanonical_iteration_count(self) -> None:
        value = report()
        for row in value["measurements"]:
            row["iterations"] = 1
            row["incumbent_samples_ms"] = [2.0]
            row["split_samples_ms"] = [1.0]
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            self.validate(value)

    def test_rejects_boolean_numeric(self) -> None:
        value = report()
        value["measurements"][0]["incumbent_mean_ms"] = True
        with self.assertRaisesRegex(ValueError, "raw samples"):
            self.validate(value)

    def test_rejects_device_identity_drift(self) -> None:
        value = report()
        value["device"]["wave_size"] = 64
        with self.assertRaisesRegex(ValueError, "device provenance"):
            self.validate(value)

    def test_rejects_float_or_boolean_integer_fields(self) -> None:
        for mutate in (
            lambda value: value.__setitem__("schema_version", 2.0),
            lambda value: value["device"].__setitem__("wave_size", 32.0),
            lambda value: value.__setitem__("warmup_iterations_per_route", 3.0),
            lambda value: value["workspace_contract"].__setitem__(
                "t1_maximum_plus_one_bytes", False),
        ):
            with self.subTest(mutate=mutate):
                value = report()
                mutate(value)
                with self.assertRaises(ValueError):
                    self.validate(value)

    def test_rejects_wrong_r9700_name_or_architecture(self) -> None:
        for key, replacement in (("name", "R9700"), ("architecture", "gfx1201:sramecc+")):
            with self.subTest(key=key):
                value = report()
                value["device"][key] = replacement
                with self.assertRaisesRegex(ValueError, "device provenance"):
                    self.validate(value)

    def test_rejects_changed_executable_bytes(self) -> None:
        value = report()
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             patch("tools.r9700.check_split512_report._file_sha256", return_value="0" * 64), \
             self.assertRaisesRegex(ValueError, "executable bytes changed"):
            validate_report(value, Path("/repo/build/split512"), Path("/repo"),
                            lambda _path: "auto")

    def test_rejects_same_bytes_at_different_executable_path(self) -> None:
        value = report()
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             self.assertRaisesRegex(ValueError, "path differs"):
            validate_report(value, Path("/retained/split512"), Path("/repo"),
                            lambda _path: "auto")

    def test_rejects_report_selected_source_root(self) -> None:
        value = report()
        value["source_root"] = "/forged"
        for source in value["sources"]:
            source["path"] = "/forged/" + SOURCE_PATHS[source["role"]]
        with self.assertRaisesRegex(ValueError, "expected exact canonical"):
            self.validate(value)

    def test_rejects_changed_source_bytes(self) -> None:
        value = report()
        digests = {value["executable"]["path"]: value["executable"]["sha256"]}
        digests.update({source["path"]: source["sha256"] for source in value["sources"]})
        digests[value["sources"][-1]["path"]] = "0" * 64
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.is_dir", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             patch("tools.r9700.check_split512_report._file_sha256",
                   side_effect=lambda path: digests[str(path)]), \
             self.assertRaisesRegex(ValueError, "source bytes changed"):
            validate_report(value, Path("/repo/build/split512"), Path("/repo"),
                            lambda _path: "auto")


if __name__ == "__main__":
    unittest.main()
