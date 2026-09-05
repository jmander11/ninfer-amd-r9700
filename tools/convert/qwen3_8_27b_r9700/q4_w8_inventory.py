"""Explicit source-Q4 plus promoted-W8 evaluation inventory for Qwen3.8.

Source Q4 roles remain Q4G64. Every source Q5G64, Q6G64, or W8G32 role is
stored as W8G32, providing a registered evaluation compromise against all-Q4 and
all-W8 without admitting Q5/Q6 fallback formats prematurely.
"""

from __future__ import annotations

from collections import Counter

from tools.artifact.layouts import align_up, encoded_size
from tools.convert.qwen3.common.inventory import (
    BF16,
    DIRECT_FORMATS,
    FP32,
    I32,
    Q4,
    W8,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    tensor_spec,
)

from . import source_inventory


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "r9700-q4-w8-n16k16-eval"
TARGET_KEY = "qwen3_8_27b_r9700"
RECIPE_ID = "r9700-source-q4-n16k16-promoted-w8g32-eval-v1"


def _candidate_tensor(spec: TensorSpec) -> TensorSpec:
    if spec.format in DIRECT_FORMATS:
        numeric_format = spec.format
    else:
        numeric_format = Q4 if spec.format == Q4 else W8
    return tensor_spec(spec.name, spec.shape, numeric_format)


RESOURCE_SPECS: tuple[ResourceSpec, ...] = source_inventory.RESOURCE_SPECS
TENSOR_SPECS: tuple[TensorSpec, ...] = tuple(
    _candidate_tensor(spec) for spec in source_inventory.TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS
FORMAT_COUNTS: dict[str, int] = dict(Counter(spec.format for spec in TENSOR_SPECS))
FORMAT_ENCODED_BYTES: dict[str, int] = dict(
    sorted(
        {
            numeric_format: sum(
                encoded_size(spec.layout, spec.format, spec.shape)
                for spec in TENSOR_SPECS
                if spec.format == numeric_format
            )
            for numeric_format in FORMAT_COUNTS
        }.items()
    )
)
TENSOR_ENCODED_BYTES = sum(FORMAT_ENCODED_BYTES.values())


def _device_arena_bytes() -> int:
    cursor = 0
    for spec in TENSOR_SPECS:
        cursor = align_up(cursor, 256)
        cursor += encoded_size(spec.layout, spec.format, spec.shape)
    return cursor


DEVICE_ARENA_BYTES = _device_arena_bytes()


def validate_inventory() -> None:
    if len(TENSOR_SPECS) != 1118 or len(OBJECT_SPECS) != 1124:
        raise ValueError("Q4/W8 evaluation inventory is not Qwen3.8-27B complete")
    if tuple(spec.name for spec in TENSOR_SPECS) != tuple(
        spec.name for spec in source_inventory.TENSOR_SPECS
    ) or tuple(spec.shape for spec in TENSOR_SPECS) != tuple(
        spec.shape for spec in source_inventory.TENSOR_SPECS
    ):
        raise ValueError("Q4/W8 evaluation inventory differs from the source plan")
    expected_counts = {BF16: 582, FP32: 96, I32: 1, Q4: 183, W8: 256}
    if FORMAT_COUNTS != expected_counts:
        raise ValueError(
            f"Q4/W8 format counts differ: expected {expected_counts}, got {FORMAT_COUNTS}"
        )
    expected_bytes = {
        BF16: 58_647_008,
        FP32: 18_432,
        I32: 524_288,
        Q4: 7_392_236_480,
        W8: 15_416_751_104,
    }
    if FORMAT_ENCODED_BYTES != dict(sorted(expected_bytes.items())):
        raise ValueError("Q4/W8 encoded byte totals differ from the fixed tensor plan")
    if TENSOR_ENCODED_BYTES != 22_868_177_312:
        raise ValueError("Q4/W8 tensor payload byte total differs")
    if DEVICE_ARENA_BYTES != 22_868_191_232:
        raise ValueError("Q4/W8 aligned device arena byte total differs")


validate_inventory()


__all__ = [
    "DEVICE_ARENA_BYTES",
    "DIRECT_FORMATS",
    "FORMAT_COUNTS",
    "FORMAT_ENCODED_BYTES",
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
