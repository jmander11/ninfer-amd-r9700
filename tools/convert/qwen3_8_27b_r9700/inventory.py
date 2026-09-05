"""Exact provisional R9700 integer-candidate inventory for Qwen3.8-27B.

This is a direct-BF16-source artifact inventory, not a product registration.
Its object order and logical shapes deliberately match Qwen3.8's complete
1118-tensor inventory. Every matrix that was quantized in the established
storage inventory is encoded as W8G32_F16S; direct BF16, FP32, and I32 state
objects remain direct. That one explicit candidate is useful for converter
and kernel bring-up, but is not a selected production weight recipe.
"""

from __future__ import annotations

from collections import Counter

from tools.convert.qwen3.common.inventory import (
    BF16,
    DIRECT_FORMATS,
    FP32,
    I32,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    W8,
    tensor_spec,
)
from . import source_inventory


MODEL_ID = "qwen3.8-27b"
# This identity is provisional and cannot be mistaken for the final R9700
# production artifact while weight-recipe selection remains open.
WEIGHTS_ID = "r9700-int-candidate"
TARGET_KEY = "qwen3_8_27b_r9700"
RECIPE_ID = "r9700-w8g32-candidate-v0"


def _candidate_tensor(spec: TensorSpec) -> TensorSpec:
    # The source inventory already makes the model-role decision: its direct
    # objects are norms, state, and exact transforms; its non-direct objects
    # are all persistent matrix roles. This target candidate replaces only
    # the latter format choice with one explicit integer representation.
    numeric_format = spec.format if spec.format in DIRECT_FORMATS else W8
    return tensor_spec(spec.name, spec.shape, numeric_format)


RESOURCE_SPECS: tuple[ResourceSpec, ...] = source_inventory.RESOURCE_SPECS
TENSOR_SPECS: tuple[TensorSpec, ...] = tuple(
    _candidate_tensor(spec) for spec in source_inventory.TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS: dict[str, int] = dict(Counter(spec.format for spec in TENSOR_SPECS))


def validate_inventory() -> None:
    """Prove this candidate has the exact Qwen3.8 object/shape inventory."""

    if len(TENSOR_SPECS) != 1118 or len(OBJECT_SPECS) != 1124:
        raise ValueError("R9700 candidate inventory count is not Qwen3.8-27B complete")
    if tuple(spec.name for spec in TENSOR_SPECS) != tuple(
        spec.name for spec in source_inventory.TENSOR_SPECS
    ):
        raise ValueError("R9700 candidate tensor order differs from Qwen3.8 source inventory")
    if tuple(spec.shape for spec in TENSOR_SPECS) != tuple(
        spec.shape for spec in source_inventory.TENSOR_SPECS
    ):
        raise ValueError("R9700 candidate tensor shapes differ from Qwen3.8 source inventory")
    expected = {BF16: 582, FP32: 96, I32: 1, W8: 439}
    if FORMAT_COUNTS != expected:
        raise ValueError(
            f"R9700 candidate format counts differ: expected {expected}, got {FORMAT_COUNTS}"
        )


validate_inventory()
