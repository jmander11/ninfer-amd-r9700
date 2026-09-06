import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.r9700.check_rmsnorm_decode_report import main, validate


def fixture() -> dict:
    rows = []
    for extent in range(1, 5):
        trials = [{"index": index,
                   "order": "incumbent,candidate" if index % 2 == 0 else "candidate,incumbent",
                   "incumbent_interval_ms": 819.2,
                   "candidate_interval_ms": 409.6} for index in range(9)]
        rows.append({"rows": extent, "ordinary_calls_per_round": 129,
                     "incumbent_median_ms": .1, "candidate_median_ms": .05,
                     "incumbent_mad_ms": 0.0, "candidate_mad_ms": 0.0,
                     "incumbent_uncertainty_ms": 0.0, "candidate_uncertainty_ms": 0.0,
                     "robust_upper_ratio": .5, "robust_round_saving_lower_ms": 6.45,
                     "passed": True, "trials": trials})
    return {"schema": "ninfer.r9700.rmsnorm-k5120-rows4-qualification.v2",
            "status": "measured",
            "command": ["/tmp/rmsnorm_decode_qual", "--benchmark", "--out-json",
                        "/tmp/report.json"],
            "device": {"ordinal": 0, "name": "AMD Radeon AI PRO R9700",
                       "architecture": "gfx1201", "wave_size": 32,
                       "runtime_version": 1, "driver_version": 1},
            "power_profile": {
                "path": "/sys/class/drm/card2/device/power_dpm_force_performance_level",
                "required": "auto", "before_device": "auto", "after_numerical": "auto",
                "after_timing": "auto"},
            "correctness": {
                            "oracle": "independent FP64 complete represented-BF16 RMSNorm formula; BF16 output boundary",
                            "criterion": "maximum 2 BF16 representable steps; absolute and relative-L2 values are diagnostics",
                            "rows": [1, 2, 3, 4],
                            "fixtures": ["ordinary", "mixed_magnitude", "zero"],
                            "unit_offset": [False, True], "eps": [1e-8, 1e-6, 1e-4],
                            "maximum_bf16_steps_allowed": 2, "cases_per_route": 72,
                            "incumbent": {"maximum_bf16_steps": 1,
                                          "maximum_absolute_error": .01,
                                          "maximum_relative_l2": .001},
                            "candidate": {"maximum_bf16_steps": 2,
                                          "maximum_absolute_error": .01,
                                          "maximum_relative_l2": .001},
                            "graph_replays": 2, "output_canaries": True,
                            "malformed_cases": 11,
                            "malformed_case_names": [
                                "null_input", "null_weight", "null_output", "rows_zero",
                                "rows_five", "eps_zero", "eps_negative", "eps_nan",
                                "eps_positive_infinity", "eps_negative_infinity",
                                "null_stream"]},
            "timing": {"stream": "one owned nondefault stream; both routes and events ordered on it",
                       "method": "9 alternating paired trials; raw event intervals",
                       "iterations_per_interval": 8192, "minimum_interval_ms": 10,
                       "uncertainty": "4.4478*MAD"},
            "rows": rows,
            "decision": {"maximum_robust_ratio": 1.01,
                         "minimum_round_saving_lower_ms": 5.0,
                         "minimum_observed_round_saving_lower_ms": 6.45,
                         "all_rows_passed": True, "accepted": True}}


def rejected_fixture() -> dict:
    value = fixture()
    for row in value["rows"]:
        for trial in row["trials"]:
            trial["candidate_interval_ms"] = 800.0
        row.update({"candidate_median_ms": 800.0 / 8192.0,
                    "robust_upper_ratio": (800.0 / 8192.0) / .1,
                    "robust_round_saving_lower_ms": 129 * (.1 - 800.0 / 8192.0),
                    "passed": False})
    value["decision"].update({
        "minimum_observed_round_saving_lower_ms": 129 * (.1 - 800.0 / 8192.0),
        "all_rows_passed": False, "accepted": False})
    return value


class ReportCheckTest(unittest.TestCase):
    def test_accepts_recomputed_report(self):
        self.assertTrue(validate(fixture())["accepted"])

    def test_rejects_power_drift(self):
        value = fixture()
        value["power_profile"]["after_timing"] = "manual"
        with self.assertRaisesRegex(ValueError, "power checkpoints"):
            validate(value)

    def test_rejects_missing_raw_trial(self):
        value = fixture()
        value["rows"][0]["trials"].pop()
        with self.assertRaisesRegex(ValueError, "raw trial count"):
            validate(value)

    def test_rejects_forged_decision(self):
        value = fixture()
        for trial in value["rows"][0]["trials"]:
            trial["candidate_interval_ms"] = 200.0
        with self.assertRaises(ValueError):
            validate(value)

    def test_rejects_forged_statistic(self):
        value = copy.deepcopy(fixture())
        value["rows"][0]["robust_upper_ratio"] = .4
        with self.assertRaisesRegex(ValueError, "inconsistent robust_upper_ratio"):
            validate(value)

    def test_rejects_rejected_report(self):
        with self.assertRaisesRegex(ValueError, "report decision rejected"):
            validate(rejected_fixture())

    def test_cli_rejects_rejected_report(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(rejected_fixture()), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "report decision rejected"):
                main([str(path)])

    def test_rejects_contract_string_mutations(self):
        for section, key in (("correctness", "oracle"), ("correctness", "criterion"),
                             ("timing", "stream"), ("timing", "method")):
            with self.subTest(section=section, key=key):
                value = fixture()
                value[section][key] += " drift"
                with self.assertRaisesRegex(ValueError, "contract|inventory"):
                    validate(value)

    def test_rejects_negative_diagnostic(self):
        value = fixture()
        value["correctness"]["candidate"]["maximum_relative_l2"] = -0.01
        with self.assertRaisesRegex(ValueError, "negative relative L2"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
