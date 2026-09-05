from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.r9700.check_xattention_production_report import (
    SOURCE_PATHS,
    validate_report,
    workspace_bytes,
)


def report(group: int = 16) -> dict:
    iterations = 3
    measurements = []
    for context in (8192, 32768):
        measurements.append({
            "context": context, "query_rows": 128, "iterations": iterations,
            "workspace_bytes": workspace_bytes(128, context), "mean_keep_fraction": 0.25,
            "sparse_median_ms": 2.0, "rank_median_ms": 0.5,
            "consumer_median_ms": 1.5, "dense_median_ms": 4.0,
            "dense_over_sparse": 2.0,
            "sparse_samples_ms": [2.0, 2.1, 1.9],
            "rank_samples_ms": [0.5, 0.6, 0.4],
            "consumer_samples_ms": [1.5, 1.6, 1.4],
            "dense_samples_ms": [4.0, 4.1, 3.9],
        })
    return {
        "schema": "ninfer_r9700_xattention_qualification", "schema_version": 6,
        "architecture": "gfx1201", "disposition": "qualification-candidate",
        "power_profile": {"path": "/sys/class/drm/card2/device/power_dpm_force_performance_level",
                          "value": "auto", "required": "auto"},
        "physical_oracle": {"status": "passed", "max_abs_error": 0.00015,
                            "max_normalized_error": 0.5,
                            "absolute_tolerance": 0.0003, "relative_tolerance": 0.0003,
                            "criterion": "per-element-absolute-or-relative"},
        "timing_fixture": "structured-concentrated-e4m3fn-v1", "hot_page_period": 16,
        "sample_order": "rotating-interleaved", "stage_samples_are_independent": True,
        "rank_stage": "fp8-k-pack-plus-two-pass-bf16-wmma-plus-b128-select",
        "consumer_stage": "b16-query-b16-key-bf16-wmma-online-fp32-softmax-direct-int4-v",
        "value_group": group, "stride": 16, "tau_permille": 900,
        "physical_qualification": {"status": "passed",
                                   "scope": "full-numerical-rejection-before-timing"},
        "device": {"name": "R9700", "architecture": "gfx1201", "wave_size": 32,
                   "runtime_version": 70000000, "driver_version": 70000000},
        "executable": {"path": "/repo/build/xattention", "sha256": "a" * 64},
        "source_root": "/repo",
        "sources": [{"role": role, "path": "/repo/" + path,
                     "sha256": f"{index:x}" * 64}
                    for index, (role, path) in enumerate(SOURCE_PATHS.items(), 1)],
        "measurements": measurements,
        "production_scale": {
            "fixture": "rotating-address-concentrated-e4m3fn-nonzero-int4-v1",
            "context": 8192, "query_rows": 4096, "iterations": iterations,
            "rotating_replicas": 4,
            "workspace_bytes_per_replica": workspace_bytes(4096, 8192),
            "value_fixture": "signed-int4-feature-1-minus2-token-head-varying-fp16-scale",
            "output_oracle": "all-elements-fp64-softmax-pv-from-device-keep-list",
            "oracle_relative_l2": 0.0001, "oracle_relative_l2_tolerance": 0.0003,
            "oracle_max_abs_error": 0.001, "oracle_gross_max_abs_tolerance": 0.002,
            "mean_keep_fraction": 0.25, "sparse_median_ms": 2.0,
            "rank_median_ms": 0.5, "consumer_median_ms": 1.5,
            "sparse_samples_ms": [2.0, 2.1, 1.9],
            "rank_samples_ms": [0.5, 0.6, 0.4],
            "consumer_samples_ms": [1.5, 1.6, 1.4],
        },
    }


class XAttentionProductionReportTest(unittest.TestCase):
    @staticmethod
    def validate(value: dict, group: int = 16) -> None:
        digests = {value["executable"]["path"]: value["executable"]["sha256"]}
        digests.update({source["path"]: source["sha256"] for source in value["sources"]})
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.is_dir", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             patch("tools.r9700.check_xattention_production_report.file_sha256",
                   side_effect=lambda path: digests[str(path)]):
            validate_report(value, group, Path("/repo/build/xattention"), Path("/repo"),
                            lambda _path: "auto")

    def test_accepts_complete_g16_and_g32_reports(self) -> None:
        self.validate(report(16), 16)
        self.validate(report(32), 32)

    def test_rejects_non_auto_profile(self) -> None:
        value = report()
        value["power_profile"]["value"] = "profile_standard"
        with self.assertRaisesRegex(ValueError, "power profile"):
            self.validate(value)

    def test_rejects_wrong_executable_digest(self) -> None:
        value = report()
        value["executable"]["sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.validate(value)

    def test_rejects_missing_source(self) -> None:
        value = report()
        value["sources"].pop()
        with self.assertRaisesRegex(ValueError, "source inventory"):
            self.validate(value)

    def test_rejects_missing_measurement_context(self) -> None:
        value = report()
        value["measurements"][-1] = copy.deepcopy(value["measurements"][0])
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.validate(value)

    def test_rejects_raw_sample_drift(self) -> None:
        value = report()
        value["production_scale"]["sparse_median_ms"] = 1.0
        with self.assertRaisesRegex(ValueError, "raw samples"):
            self.validate(value)

    def test_rejects_oracle_failure(self) -> None:
        value = report()
        value["production_scale"]["oracle_relative_l2"] = 0.001
        with self.assertRaisesRegex(ValueError, "oracle failed"):
            self.validate(value)

    def test_rejects_wrong_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "value group"):
            self.validate(report(16), 32)

    def test_rejects_changed_source_bytes(self) -> None:
        value = report()
        digests = {value["executable"]["path"]: value["executable"]["sha256"]}
        digests.update({source["path"]: source["sha256"] for source in value["sources"]})
        digests[value["sources"][-1]["path"]] = "0" * 64
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.is_dir", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             patch("tools.r9700.check_xattention_production_report.file_sha256",
                   side_effect=lambda path: digests[str(path)]), \
             self.assertRaisesRegex(ValueError, "source bytes changed"):
            validate_report(value, 16, Path("/repo/build/xattention"), Path("/repo"),
                            lambda _path: "auto")

    def test_rejects_same_bytes_at_different_executable_path(self) -> None:
        value = report()
        digests = {"/retained/xattention": value["executable"]["sha256"]}
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             patch("tools.r9700.check_xattention_production_report.file_sha256",
                   side_effect=lambda path: digests[str(path)]), \
             self.assertRaisesRegex(ValueError, "path differs"):
            validate_report(value, 16, Path("/retained/xattention"), Path("/repo"),
                            lambda _path: "auto")

    def test_rejects_report_selected_source_root(self) -> None:
        value = report()
        value["source_root"] = "/forged"
        for source in value["sources"]:
            source["path"] = "/forged/" + SOURCE_PATHS[source["role"]]
        with self.assertRaisesRegex(ValueError, "expected exact canonical"):
            self.validate(value)

    def test_rejects_non_auto_live_power(self) -> None:
        value = report()
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.is_dir", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             self.assertRaisesRegex(ValueError, "live power"):
            validate_report(value, 16, Path("/repo/build/xattention"), Path("/repo"),
                            lambda _path: "profile_standard")

    def test_rejects_boolean_numeric(self) -> None:
        value = report()
        value["production_scale"]["oracle_relative_l2"] = False
        with self.assertRaisesRegex(ValueError, "oracle failed"):
            self.validate(value)

    def test_rejects_forged_huge_physical_oracle_error(self) -> None:
        value = report()
        value["physical_oracle"]["max_abs_error"] = 1e100
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.validate(value)


if __name__ == "__main__":
    unittest.main()
