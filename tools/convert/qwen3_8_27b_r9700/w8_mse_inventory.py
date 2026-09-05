"""In-size W8G32 source-MSE evaluation inventory for Qwen3.8-27B."""

from __future__ import annotations

from collections import Counter

from tools.artifact.layouts import align_up, encoded_size
from tools.convert.qwen3.common.inventory import (
    BF16,
    DIRECT_FORMATS,
    FP32,
    I32,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    W8,
)

from . import inventory


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "r9700-w8g32-mse-eval"
TARGET_KEY = "qwen3_8_27b_r9700"
RECIPE_ID = "r9700-w8g32-source-mse8-eval-v0"

RESOURCE_SPECS: tuple[ResourceSpec, ...] = inventory.RESOURCE_SPECS
TENSOR_SPECS: tuple[TensorSpec, ...] = inventory.TENSOR_SPECS
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS
FORMAT_COUNTS: dict[str, int] = dict(Counter(spec.format for spec in TENSOR_SPECS))
TENSOR_ENCODED_BYTES = sum(
    encoded_size(spec.layout, spec.format, spec.shape) for spec in TENSOR_SPECS
)


def _device_arena_bytes() -> int:
    cursor = 0
    for spec in TENSOR_SPECS:
        cursor = align_up(cursor, 256)
        cursor += encoded_size(spec.layout, spec.format, spec.shape)
    return cursor


DEVICE_ARENA_BYTES = _device_arena_bytes()


def validate_inventory() -> None:
    inventory.validate_inventory()
    if len(TENSOR_SPECS) != 1118 or len(OBJECT_SPECS) != 1124:
        raise ValueError("W8 source-MSE evaluator is not Qwen3.8-27B complete")
    if FORMAT_COUNTS != {BF16: 582, FP32: 96, I32: 1, W8: 439}:
        raise ValueError("W8 source-MSE evaluator format counts changed")
    if TENSOR_ENCODED_BYTES != 30_260_413_792:
        raise ValueError("W8 source-MSE evaluator tensor byte total changed")
    if DEVICE_ARENA_BYTES != 30_260_425_984:
        raise ValueError("W8 source-MSE evaluator device arena changed")


validate_inventory()


__all__ = [
    "DEVICE_ARENA_BYTES",
    "DIRECT_FORMATS",
    "FORMAT_COUNTS",
    "MODEL_ID",
    "OBJECT_SPECS",
    "RECIPE_ID",
    "RESOURCE_SPECS",
    "ResourceSpec",
    "TARGET_KEY",
    "TENSOR_ENCODED_BYTES",
    "TENSOR_SPECS",
    "TensorSpec",
    "WEIGHTS_ID",
    "validate_inventory",
]
