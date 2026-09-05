from __future__ import annotations

import copy
import math
import unittest

import torch

from . import screen_selective_q4g128_mse_quality as screen


class SelectiveQ4G128MseQualityTest(unittest.TestCase):
    def test_scope_is_exact_and_excludes_mtp(self) -> None:
        specs = screen.selective_specs()
        self.assertEqual(len(specs), 160)
        self.assertTrue(all(spec.name.startswith("text/layers/") for spec in specs))
        self.assertFalse(any(spec.name.startswith("mtp/") for spec in specs))
        self.assertEqual(
            {role: sum(screen._role(spec.name) == role for spec in specs)
             for role in screen.ROLE_SHAPES_AND_COUNTS},
            {"gdn_output": 48, "gdn_value_z": 48, "mlp_down": 64},
        )

    def test_mse_g128_is_deterministic_and_improves_g128_absmax_sse(self) -> None:
        source = torch.tensor(
            [[float(((index * 37) % 257) - 128) / 13 for index in range(256)]],
            dtype=torch.bfloat16,
        )
        first = screen._mse_q4g128_decode(source)
        second = screen._mse_q4g128_decode(source)
        self.assertTrue(torch.equal(first, second))
        canonical = screen.quantize_dequantize(source, group_size=128)
        candidate = screen._metrics(source, first)
        control = screen._metrics(source, canonical)
        self.assertLessEqual(candidate["squared_error"], control["squared_error"])
        self.assertTrue(math.isfinite(candidate["relative_l2"]))

    @staticmethod
    def _record(spec, error: float) -> dict[str, object]:
        indices = screen.select_row_indices(spec.name, spec.shape[0], 1)
        metrics = {}
        for codec, multiplier in (
            ("q4g64_absmax", 1.0),
            ("q4g128_absmax", 1.2),
            ("q4g128_mse", error),
        ):
            squared_error = multiplier * multiplier
            metrics[codec] = {
                "relative_l2": multiplier,
                "max_abs": multiplier,
                "squared_error": squared_error,
                "squared_reference": 1.0,
            }
        return {
            "name": spec.name,
            "role": screen._role(spec.name),
            "shape": list(spec.shape),
            "sampled_row_indices": list(indices),
            "sampled_rows": 1,
            "sampled_elements": spec.shape[1],
            "zero_rows": 0,
            "metrics": metrics,
            "q4g128_mse_over_q4g64_relative_l2": error,
        }

    def test_report_recomputes_scope_aggregates_and_decision(self) -> None:
        records = [self._record(spec, 1.02) for spec in screen.selective_specs()]
        report = screen.assemble_report(
            tensors=records,
            source={"indexed_tensor_count": 1199},
            implementation={"tool.py": "1" * 64},
            command=["python", "-m", "screen"],
            rows_per_tensor=1,
        )
        screen.validate_report(report)
        self.assertTrue(report["decision"]["sampled_source_gate_pass"])
        self.assertEqual(report["aggregate"]["sampled_rows"], 160)

        tampered = copy.deepcopy(report)
        tampered["aggregate"]["q4g128_mse"]["squared_error"] += 1.0
        with self.assertRaisesRegex(ValueError, "aggregate"):
            screen.validate_report(tampered)

        tampered = copy.deepcopy(report)
        tampered["tensors"][0]["sampled_row_indices"] = [1]
        with self.assertRaisesRegex(ValueError, "row selection"):
            screen.validate_report(tampered)

    def test_report_fails_when_any_role_crosses_screen_bound(self) -> None:
        records = [
            self._record(spec, 1.04 if screen._role(spec.name) == "gdn_output" else 1.0)
            for spec in screen.selective_specs()
        ]
        report = screen.assemble_report(
            tensors=records, source={}, implementation={}, command=[], rows_per_tensor=1
        )
        screen.validate_report(report)
        self.assertFalse(report["decision"]["sampled_source_gate_pass"])


if __name__ == "__main__":
    unittest.main()
