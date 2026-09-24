#!/usr/bin/env python3

from __future__ import annotations

import unittest
from unittest import mock

from tools.bench.validate_fp8_gate_up_hardware_proof import (
    Q4_KERNEL,
    _hardware_profile,
    _is_fp8_poison_boundary,
    _validate_resource_binding,
    _validate_resources,
    _validate_qualifier_report,
)


class ValidateFp8GateUpHardwareProofTest(unittest.TestCase):
    def test_current_and_retained_poison_boundaries_are_explicit(self) -> None:
        self.assertTrue(_is_fp8_poison_boundary("scope_poison_nonfinite_output_args"))
        self.assertTrue(_is_fp8_poison_boundary("scope_poison_nonfinite_gate_up_output_args"))
        self.assertFalse(_is_fp8_poison_boundary("scope_unrelated_output_args"))

    def test_exact_production_symbol_excludes_named_variants(self) -> None:
        self.assertIn(Q4_KERNEL, "_ZN_scope_a8q4g64_linear_prefill_cta_kernel_args")
        self.assertNotIn(Q4_KERNEL, "a8q4g64_linear_prefill_cta_m64n128_regression_kernel")

    def test_existing_gate_plan_without_qualification_remains_valid(self) -> None:
        profile = _hardware_profile({
            "schema": "ninfer.r9700.fp8_gate_up_hardware_proof_plan.v1",
            "shape": {"tokens": 2048, "rows": 34816, "columns": 5120},
        })
        self.assertEqual(profile["qualification"], "gate_up")

    def test_attention_plan_and_validator_are_distinct(self) -> None:
        profile = _hardware_profile({
            "schema": "ninfer.r9700.fp8_attention_qk_gate_value_hardware_proof_plan.v1",
            "qualification": "attention_qk_gate_value",
            "shape": {"tokens": 2048, "rows": 7168, "columns": 5120},
        })
        self.assertEqual(profile["proof_schema"],
                         "ninfer.r9700.fp8_attention_qk_gate_value_hardware_proof.v1")
        with mock.patch(
            "tools.bench.validate_fp8_gate_up_hardware_proof.validate_projection"
        ) as validate:
            _validate_qualifier_report({}, "attention_qk_gate_value")
            validate.assert_called_once_with({}, "attention_qk_gate_value")

    def test_profile_shape_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "qualification or shape"):
            _hardware_profile({
                "schema": "ninfer.r9700.fp8_attention_qk_gate_value_hardware_proof_plan.v1",
                "qualification": "attention_qk_gate_value",
                "shape": {"tokens": 2048, "rows": 4096, "columns": 5120},
            })

    def test_fp8_resources_are_exact_and_stable(self) -> None:
        columns = {"sgpr_count", "arch_vgpr_count", "accum_vgpr_count",
                   "group_segment_size", "private_segment_size"}
        row = {"sgpr_count": 128, "arch_vgpr_count": 192,
               "accum_vgpr_count": 0, "group_segment_size": 25088,
               "private_segment_size": 0}
        expected = _hardware_profile({
            "schema": "ninfer.r9700.fp8_gate_up_hardware_proof_plan.v1",
            "shape": {"tokens": 2048, "rows": 34816, "columns": 5120},
        })["resources"]
        self.assertEqual(_validate_resources([row, dict(row)], columns, expected), expected)
        changed = dict(row, arch_vgpr_count=191)
        with self.assertRaisesRegex(ValueError, "stable kernel resource"):
            _validate_resources([row, changed], columns, expected)
        with self.assertRaisesRegex(ValueError, "admitted envelope"):
            _validate_resources([changed], columns, expected)
        spill_columns = columns | {"sgpr_spill_count", "vgpr_spill_count"}
        spill_row = dict(row, sgpr_spill_count=2, vgpr_spill_count=3)
        spill_expected = dict(expected, spill_counts={
            "available": True, "sgpr": 2, "vgpr": 3})
        self.assertEqual(
            _validate_resources([spill_row], spill_columns, spill_expected), spill_expected)

    def test_resource_rows_bind_exact_kernel_code_object_and_uri(self) -> None:
        row = {"kernel_id": 825, "code_object_id": 4,
               "uri": "memory://1433014#offset=0x10&size=32"}
        fp8 = {"kernel_id": 825, "code_object_id": 4,
               "code_object": {"uri": row["uri"]}}
        _validate_resource_binding([row, dict(row)], fp8)
        for field, value in (("kernel_id", 826), ("code_object_id", 5)):
            changed = dict(fp8, **{field: value})
            with self.assertRaisesRegex(ValueError, "kernel/code-object/URI"):
                _validate_resource_binding([row], changed)
        changed = dict(fp8, code_object={"uri": "memory://other#offset=0x10&size=32"})
        with self.assertRaisesRegex(ValueError, "kernel/code-object/URI"):
            _validate_resource_binding([row], changed)


if __name__ == "__main__":
    unittest.main()
