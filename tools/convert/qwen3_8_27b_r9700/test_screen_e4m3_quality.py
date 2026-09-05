from __future__ import annotations

import unittest

import torch

from tools.convert.qwen3.common.recipe import Concat, GatherRows, Reshape, Slice, SourceTensor

from .screen_e4m3_quality import (
    _sample_expression,
    assemble_report,
    canonical_json,
    select_row_indices,
)


class _Reader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def rows(self, source: SourceTensor, indices: list[int]) -> torch.Tensor:
        tensor = self.tensors[source.name]
        return tensor[indices].reshape(len(indices), -1).contiguous()


class E4M3QualityScreenTest(unittest.TestCase):
    def test_row_selection_is_fixed_bounded_and_includes_endpoints(self) -> None:
        selected = select_row_indices("text/layers/0/mlp/gate_up", 34816, 8)
        self.assertEqual(selected, select_row_indices("text/layers/0/mlp/gate_up", 34816, 8))
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 34815)
        self.assertEqual(len(set(selected)), 8)
        self.assertEqual(select_row_indices("scalar", 1, 8), (0,))

    def test_sampler_preserves_concat_slice_reshape_and_gather_rows(self) -> None:
        matrix = torch.arange(48, dtype=torch.float32).reshape(12, 4).to(torch.bfloat16)
        other = (100 + torch.arange(24, dtype=torch.float32)).reshape(6, 4).to(torch.bfloat16)
        reader = _Reader({"matrix": matrix, "other": other})

        query = Reshape(
            Slice(Reshape(SourceTensor("matrix", (12, 4)), (3, 4, 4)), 1, 1, 3),
            (6, 4),
        )
        expression = Concat((query, Slice(SourceTensor("other", (6, 4)), 0, 1, 5)), 0)
        sampled = _sample_expression(expression, [0, 5, 6, 9], reader, [])
        expected = torch.stack((matrix[1], matrix[10], other[1], other[4]))
        self.assertTrue(torch.equal(sampled, expected))

        gathered = GatherRows(SourceTensor("matrix", (12, 4)), "ids", 3)
        sampled_gather = _sample_expression(gathered, [0, 2], reader, [9, 4, 1])
        self.assertTrue(torch.equal(sampled_gather, torch.stack((matrix[9], matrix[1]))))

    def test_report_bytes_are_independent_of_input_order(self) -> None:
        def record(name: str, relative: float, absolute: float) -> dict[str, object]:
            return {
                "name": name,
                "shape": [2, 2],
                "sampled_row_indices": [0],
                "sampled_rows": 1,
                "sampled_elements": 2,
                "nonfinite_values": 0,
                "zero_rows": 0,
                "max_abs": absolute,
                "relative_l2": relative,
                "squared_error": relative * relative,
                "squared_reference": 1.0,
            }

        tensors = [record("z", 0.25, 0.5), record("a", 0.5, 0.25)]
        arguments = {
            "rows_per_tensor": 1,
            "source": {"config": {"sha256": "1"}},
            "implementation": {"b": "2", "a": "1"},
            "worst_count": 2,
        }
        forward = assemble_report(tensors=tensors, **arguments)
        reverse = assemble_report(tensors=list(reversed(tensors)), **arguments)
        self.assertEqual(canonical_json(forward), canonical_json(reverse))
        self.assertEqual([item["name"] for item in forward["tensors"]], ["a", "z"])
        self.assertEqual(
            forward["worst_tensors"]["relative_l2"],
            [{"name": "a", "value": 0.5}, {"name": "z", "value": 0.25}],
        )


if __name__ == "__main__":
    unittest.main()
