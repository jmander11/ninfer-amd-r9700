from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.r9700.check_prefill_cta_report import (
    CONFIG,
    DEVICE_NAME,
    POWER_PATH,
    TOKENS,
    validate_report,
)


def report(recipe: str) -> dict:
    cfg = CONFIG[recipe]
    cases = []
    for n, k in sorted(cfg["shapes"]):
        for tokens in sorted(TOKENS):
            row = {"tokens": tokens, "maximum_bf16_steps": 0,
                   "one_wave_samples_ms": [2.0] * cfg["trials"],
                   "cooperative_cta_samples_ms": [1.0] * cfg["trials"],
                   "cooperative_over_one_wave": 0.5}
            if recipe == "q4":
                row.update(rows=n, columns=k, one_wave_ms=2.0, cooperative_cta_ms=1.0,
                           exhaustive_oracle=False)
            else:
                roles = {(7168, 5120): "text_attention_gate_value",
                         (12288, 5120): "text_gdn_value_z",
                         (5120, 17408): "text_mlp_down",
                         (5120, 6144): "text_attention_gdn_output"}
                row.update(shape=[n, k], one_wave_median_ms=2.0,
                           cooperative_cta_median_ms=1.0, exhaustive_oracle=False,
                           role=roles[(n, k)])
            cases.append(row)
    roles = {
        "qualifier": cfg["qualifier"],
        "contract_header": "src/ops/r9700/linear/r9700_linear.h",
        "kernel": "src/ops/r9700/linear/r9700_linear.hip",
        "independent_oracle": cfg["qualifier"],
    }
    return {
        "schema": cfg["schema"], "schema_version": 3,
        "disposition": "unpromoted-qualification-candidate",
        "candidate": copy.deepcopy(cfg["candidate"]),
        "resources": copy.deepcopy(cfg["resources"]),
        "qualified_tuple_scope": copy.deepcopy(cfg["scope"]),
        "incumbent_scope": "post-quantization-a8-linear-leaf",
        "challenger_scope": "post-quantization-a8-linear-leaf",
        "physical_qualification": {"status": "passed", "scope": "full-numerical-rejection-before-timing"},
        "power_profile": {"path": POWER_PATH, "value": "auto", "required": "auto"},
        "hardware": {"device": DEVICE_NAME, "architecture": "gfx1201", "wave_size": 32,
                     "vram_bytes": 33554432,
                     "hip_runtime_version": 70000000, "hip_driver_version": 70000000},
        "toolchain": {"compiler": "clang", "offload_architecture": "gfx1201"},
        "executable": {"path": "/tmp/qual", "sha256": "a" * 64},
        "source_root": "/repo",
        "sources": [{"role": role, "path": "/repo/" + path,
                     "sha256": ("1" if path == cfg["qualifier"] else f"{i:x}") * 64}
                    for i, (role, path) in enumerate(roles.items(), 1)],
        "timing": {"sample_order": "alternating-interleaved",
                   "warmup_iterations_per_route": 1, "trials": cfg["trials"],
                   "method": "unprofiled HIP events, median",
                   "token_extents": [1024, 2048, 4096, 8192]},
        "edge_checks": {"nonzero_status_all_output_nan": True,
                        "partial_group_padded_tail_poison_ignored": True,
                        "packed_code_alignment_bytes": 4,
                        "host_owned_stored_weight_bytes": True,
                        "oracle_decodes_uploaded_bytes": True,
                        "tail_maximum_bf16_steps": 1},
        "oracle": {"kind": "independent FP64 complete quantized formula",
                   "small_case": "all outputs", "real_shapes": "8 rows x 8 tokens",
                   "maximum_bf16_steps_allowed": 2},
        "cases": cases,
    }


class PrefillCtaReportTest(unittest.TestCase):
    @staticmethod
    def validate(value: dict, recipe: str) -> None:
        digests = {value["executable"]["path"]: value["executable"]["sha256"]}
        digests.update({source["path"]: source["sha256"] for source in value["sources"]})
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.is_dir", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             patch("tools.r9700.check_prefill_cta_report.file_sha256",
                   side_effect=lambda path: digests[str(path)]):
            validate_report(value, recipe, Path("/tmp/qual"), Path("/repo"),
                            lambda _path: "auto")

    def test_accepts_both_complete_reports(self) -> None:
        for recipe in CONFIG:
            self.validate(report(recipe), recipe)

    def test_rejects_non_auto(self) -> None:
        value = report("q4")
        value["power_profile"]["value"] = "profile_standard"
        with self.assertRaisesRegex(ValueError, "auto"):
            self.validate(value, "q4")

    def test_rejects_obsolete_schema_v2(self) -> None:
        value = report("q4")
        value["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema"):
            self.validate(value, "q4")

    def test_rejects_non_integer_schema_version(self) -> None:
        value = report("w8")
        value["schema_version"] = 3.0
        with self.assertRaisesRegex(ValueError, "schema"):
            self.validate(value, "w8")

    def test_rejects_missing_case(self) -> None:
        value = report("w8")
        value["cases"].pop()
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.validate(value, "w8")

    def test_rejects_duplicate_case(self) -> None:
        value = report("q4")
        value["cases"][-1] = copy.deepcopy(value["cases"][0])
        with self.assertRaisesRegex(ValueError, "tuple"):
            self.validate(value, "q4")

    def test_rejects_hash_drift(self) -> None:
        value = report("w8")
        value["sources"][0]["sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.validate(value, "w8")

    def test_rejects_median_drift(self) -> None:
        value = report("q4")
        value["cases"][0]["cooperative_cta_ms"] = 0.5
        with self.assertRaisesRegex(ValueError, "median"):
            self.validate(value, "q4")

    def test_rejects_slow_candidate(self) -> None:
        value = report("w8")
        value["cases"][0]["cooperative_cta_samples_ms"] = [3.0] * 5
        value["cases"][0]["cooperative_cta_median_ms"] = 3.0
        value["cases"][0]["cooperative_over_one_wave"] = 1.5
        with self.assertRaisesRegex(ValueError, "admission"):
            self.validate(value, "w8")

    def test_rejects_candidate_geometry_drift(self) -> None:
        value = report("q4")
        value["candidate"] = dict(value["candidate"], tile_rows=32)
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.validate(value, "q4")

    def test_rejects_w8_role_drift(self) -> None:
        value = report("w8")
        value["cases"][0]["role"] = "wrong"
        with self.assertRaisesRegex(ValueError, "role"):
            self.validate(value, "w8")

    def test_rejects_changed_executable_bytes(self) -> None:
        value = report("q4")
        digests = {source["path"]: source["sha256"] for source in value["sources"]}
        digests[value["executable"]["path"]] = "0" * 64
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.is_dir", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             patch("tools.r9700.check_prefill_cta_report.file_sha256",
                   side_effect=lambda path: digests[str(path)]), \
             self.assertRaisesRegex(ValueError, "executable bytes changed"):
            validate_report(value, "q4", Path("/tmp/qual"), Path("/repo"),
                            lambda _path: "auto")

    def test_rejects_same_bytes_at_different_executable_path(self) -> None:
        value = report("q4")
        digests = {"/retained/qual": value["executable"]["sha256"]}
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             patch("tools.r9700.check_prefill_cta_report.file_sha256",
                   side_effect=lambda path: digests[str(path)]), \
             self.assertRaisesRegex(ValueError, "path differs"):
            validate_report(value, "q4", Path("/retained/qual"), Path("/repo"),
                            lambda _path: "auto")

    def test_rejects_changed_source_bytes(self) -> None:
        value = report("w8")
        digests = {value["executable"]["path"]: value["executable"]["sha256"]}
        digests.update({source["path"]: source["sha256"] for source in value["sources"]})
        digests[value["sources"][1]["path"]] = "0" * 64
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.is_dir", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             patch("tools.r9700.check_prefill_cta_report.file_sha256",
                   side_effect=lambda path: digests[str(path)]), \
             self.assertRaisesRegex(ValueError, "source bytes changed"):
            validate_report(value, "w8", Path("/tmp/qual"), Path("/repo"),
                            lambda _path: "auto")

    def test_rejects_forged_source_root(self) -> None:
        value = report("q4")
        value["source_root"] = "/forged"
        for source in value["sources"]:
            source["path"] = "/forged/" + source["path"].split("/repo/", 1)[1]
        with self.assertRaisesRegex(ValueError, "expected exact canonical"):
            self.validate(value, "q4")

    def test_rejects_non_auto_live_power(self) -> None:
        value = report("w8")
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.is_dir", return_value=True), \
             patch("pathlib.Path.resolve", lambda self, strict=False: self), \
             self.assertRaisesRegex(ValueError, "live power"):
            validate_report(value, "w8", Path("/tmp/qual"), Path("/repo"),
                            lambda _path: "profile_standard")

    def test_rejects_boolean_numeric(self) -> None:
        for recipe in CONFIG:
            value = report(recipe)
            value["cases"][0]["one_wave_samples_ms"][0] = True
            with self.assertRaisesRegex(ValueError, "raw timing"):
                self.validate(value, recipe)
        value = report("q4")
        value["hardware"]["hip_runtime_version"] = True
        with self.assertRaisesRegex(ValueError, "hardware"):
            self.validate(value, "q4")
        value = report("w8")
        value["edge_checks"]["tail_maximum_bf16_steps"] = True
        with self.assertRaisesRegex(ValueError, "tail numerical"):
            self.validate(value, "w8")

    def test_rejects_boolean_aliases_in_fixed_structures(self) -> None:
        mutations = (
            ("q4", "candidate", "fragment_major_lds_swizzle", 1, "geometry"),
            ("q4", "candidate", "production_dispatch", 0, "geometry"),
            ("w8", "resources", "local_bytes", False, "resources"),
            ("w8", "timing", "warmup_iterations_per_route", True, "timing protocol"),
        )
        for recipe, section, key, forged, message in mutations:
            with self.subTest(recipe=recipe, section=section, key=key):
                value = report(recipe)
                value[section][key] = forged
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(value, recipe)

    def test_rejects_wrong_device_or_invalid_vram(self) -> None:
        value = report("q4")
        value["hardware"]["device"] = "fake"
        with self.assertRaisesRegex(ValueError, "hardware"):
            self.validate(value, "q4")
        value = report("w8")
        value["hardware"]["vram_bytes"] = True
        with self.assertRaisesRegex(ValueError, "hardware"):
            self.validate(value, "w8")


if __name__ == "__main__":
    unittest.main()
