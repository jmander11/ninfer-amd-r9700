from __future__ import annotations

import hashlib
import json
import statistics
import tempfile
import unittest
from pathlib import Path

from tools.r9700.check_prefill_cta_lds_scope_report import (
    CONFIG, MAXIMUM_PER_TUPLE_RATIO, POWER_PATH, TOKENS, validate_report,
)


class PrefillCtaLdsScopeReportTest(unittest.TestCase):
    def fixture(self, root: Path, recipe: str) -> tuple[dict, Path]:
        config = CONFIG[recipe]
        executable = root / "qualifier"
        executable.write_bytes(b"elf")
        sources = []
        for role, relative in {
            "qualifier": config["qualifier"],
            "contract_header": "src/ops/r9700/linear/r9700_linear.h",
            "kernel": "src/ops/r9700/linear/r9700_linear.hip",
            "dispatch_profile": config["profile"],
        }.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(role)
            sources.append({"role": role, "path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        cases = []
        challenger_ratio = (0.6 if recipe == "q4-pingpong"
                            else 0.7 if recipe == "q4-m128n128" else 0.98)
        for rows, columns in config["shapes"]:
            for tokens in TOKENS:
                incumbent = 2.0 + rows / 100000.0 + tokens / 100000.0
                challenger = incumbent * challenger_ratio
                cases.append({
                    "rows": rows, "columns": columns, "tokens": tokens,
                    "bit_exact": True, "maximum_bf16_steps": 1,
                    "incumbent_forward_ms": [incumbent] * 7,
                    "incumbent_reverse_ms": [incumbent] * 7,
                    "incumbent_balanced_ms": [incumbent] * 7,
                    "challenger_forward_ms": [challenger] * 7,
                    "challenger_reverse_ms": [challenger] * 7,
                    "challenger_balanced_ms": [challenger] * 7,
                    "incumbent_median_ms": incumbent,
                    "challenger_median_ms": challenger,
                    "challenger_over_incumbent": challenger_ratio,
                })
        lookup = {(row["rows"], row["columns"]): row for row in cases if row["tokens"] == 2048}
        old = sum(count * lookup[shape]["incumbent_median_ms"]
                  for shape, count in config["weights"].items())
        new = sum(count * lookup[shape]["challenger_median_ms"]
                  for shape, count in config["weights"].items())
        report = {
            "artifact_type": config["artifact_type"],
            "schema_version": 1, "status": "passed", "recipe": recipe,
            "disposition": "qualification_only_unpromoted",
            "production_state": {"dispatch_changed": False,
                                 "selected_route": "M64xN128-pingpong"},
            "timing": {"method": "unprofiled HIP events", "warmup_iterations_per_route": 1,
                       "balanced_forward_reverse_pairs": 7,
                       "balanced_sample": "mean of same-route forward and reverse measurements",
                       "token_extents": list(TOKENS)},
            "numerical_qualification": {
                "completed_before_timing": True, "incumbent_challenger_bit_exact": True,
                "active_output_coverage":
                    "full rewrite and bit parity after non-incumbent poison",
                "independent_oracle":
                    "FP64 represented quantized formula at sampled coordinates",
                "maximum_bf16_steps_allowed": 2,
                "tail_status_and_alignment_gate": "passed"},
            "power_profile": {"path": POWER_PATH, "required": "auto",
                              "before": "auto", "after": "auto"},
            "hardware": {"device": "AMD Radeon AI PRO R9700", "architecture": "gfx1201",
                         "wave_size": 32, "vram_bytes": 33554432,
                         "hip_runtime_version": 70150000, "hip_driver_version": 70150000},
            "resources": {"incumbent_registers": 50, "challenger_registers": 50,
                          "incumbent_static_shared_bytes": config["incumbent_lds"],
                          "challenger_static_shared_bytes": config["challenger_lds"],
                          "incumbent_local_bytes": 0, "challenger_local_bytes": 0,
                          "challenger_max_threads_per_block":
                              1024 if recipe == "q4-m128n128" else 512},
            "toolchain": {"compiler": "clang", "offload_architecture": "gfx1201"},
            "executable": {"path": str(executable),
                           "sha256": hashlib.sha256(executable.read_bytes()).hexdigest()},
            "source_root": str(root), "sources": sources, "cases": cases,
            "failing_tuples": [],
            "decision": {"token_extent": 2048,
                         "weights": [{"rows": r, "columns": c, "count": n}
                                     for (r, c), n in config["weights"].items()],
                         "weighted_incumbent_ms": old, "weighted_challenger_ms": new,
                         "challenger_over_incumbent": new / old,
                         "maximum_per_tuple_ratio":
                             1.0 if recipe == "q4-pingpong"
                             else MAXIMUM_PER_TUPLE_RATIO,
                         "no_material_per_tuple_regression": True,
                         "minimum_weighted_saving_ms": 150 if recipe == "q4-m128n128" else 0,
                         "minimum_weighted_speedup":
                             1.5 if recipe == "q4-pingpong" else 1.0,
                         "weighted_saving_ms": old - new,
                         "accepted": True,
                         "weighted_challenger_faster": True,
                         "required_speedup_met": True},
        }
        return report, executable

    def validate(self, report: dict, recipe: str, executable: Path, root: Path) -> None:
        validate_report(report, recipe, executable, root, lambda _: "auto")

    def test_accepts_exact_q4_w8_and_q4_n128_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for recipe in CONFIG:
                root = Path(temporary) / recipe
                root.mkdir()
                report, executable = self.fixture(root, recipe)
                self.validate(report, recipe, executable, root)

    def test_pingpong_rejection_is_terminal_below_required_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, executable = self.fixture(root, "q4-pingpong")
            for row in report["cases"]:
                incumbent = row["incumbent_median_ms"]
                challenger = incumbent * 0.8
                for name in ("challenger_forward_ms", "challenger_reverse_ms",
                             "challenger_balanced_ms"):
                    row[name] = [challenger] * 7
                row["challenger_median_ms"] = challenger
                row["challenger_over_incumbent"] = 0.8
            decision = report["decision"]
            decision["weighted_challenger_ms"] = decision["weighted_incumbent_ms"] * 0.8
            decision["challenger_over_incumbent"] = 0.8
            decision["weighted_saving_ms"] = decision["weighted_incumbent_ms"] * 0.2
            decision["required_speedup_met"] = False
            decision["accepted"] = False
            report["status"] = "rejected"
            report["disposition"] = "qualification_rejected_production_unchanged"
            self.validate(report, "q4-pingpong", executable, root)

    def test_rejects_cartesian_raw_and_derived_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mutation in ("drop", "raw", "median", "ratio", "bool", "float-dimension"):
                report, executable = self.fixture(root, "w8")
                if mutation == "drop": report["cases"].pop()
                elif mutation == "raw": report["cases"][0]["challenger_balanced_ms"][0] *= 1.1
                elif mutation == "median": report["cases"][0]["incumbent_median_ms"] *= 1.1
                elif mutation == "ratio": report["cases"][0]["challenger_over_incumbent"] = 0.5
                elif mutation == "bool": report["timing"]["warmup_iterations_per_route"] = True
                else: report["cases"][0]["rows"] = float(report["cases"][0]["rows"])
                with self.assertRaises(ValueError, msg=mutation):
                    self.validate(report, "w8", executable, root)

    def test_rejects_decision_and_material_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, executable = self.fixture(root, "q4")
            report["decision"]["weighted_challenger_ms"] *= 0.5
            with self.assertRaises(ValueError): self.validate(report, "q4", executable, root)
            report, executable = self.fixture(root, "q4-m128n128")
            report["decision"]["minimum_weighted_saving_ms"] = 149
            with self.assertRaisesRegex(ValueError, "weighted saving contract"):
                self.validate(report, "q4-m128n128", executable, root)
            report, executable = self.fixture(root, "q4")
            row = report["cases"][0]
            for key in ("challenger_forward_ms", "challenger_reverse_ms", "challenger_balanced_ms"):
                row[key] = [row["incumbent_median_ms"] * 1.02] * 7
            row["challenger_median_ms"] = row["incumbent_median_ms"] * 1.02
            row["challenger_over_incumbent"] = 1.02
            with self.assertRaises(ValueError): self.validate(report, "q4", executable, root)

    def test_rejects_provenance_and_non_auto(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, executable = self.fixture(root, "q4")
            report["sources"][0]["sha256"] = "0" * 64
            with self.assertRaises(ValueError): self.validate(report, "q4", executable, root)

    def test_accepts_derived_rejected_m128n128_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, executable = self.fixture(root, "q4-m128n128")
            row = report["cases"][0]
            incumbent = row["incumbent_median_ms"]
            challenger = incumbent * 1.02
            for name in ("challenger_forward_ms", "challenger_reverse_ms",
                         "challenger_balanced_ms"):
                row[name] = [challenger] * 7
            row["challenger_median_ms"] = challenger
            row["challenger_over_incumbent"] = 1.02
            report["status"] = "rejected"
            report["disposition"] = "qualification_rejected_production_unchanged"
            report["failing_tuples"] = [{
                "rows": row["rows"], "columns": row["columns"], "tokens": row["tokens"],
                "challenger_over_incumbent": 1.02,
                "reason": "challenger_over_incumbent_exceeds_1.01",
            }]
            report["decision"]["no_material_per_tuple_regression"] = False
            report["decision"]["accepted"] = False
            self.validate(report, "q4-m128n128", executable, root)
            report, executable = self.fixture(root, "q4")
            with self.assertRaises(ValueError):
                validate_report(report, "q4", executable, root, lambda _: "profile_standard")
            report, executable = self.fixture(root, "q4")
            report["resources"]["incumbent_local_bytes"] = False
            with self.assertRaises(ValueError): self.validate(report, "q4", executable, root)


if __name__ == "__main__":
    unittest.main()
