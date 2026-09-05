"""Target-private all-rowwise-E4M3 evaluation inventory for Qwen3.8-27B."""

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
)

from . import source_inventory


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "r9700-f8e4m3-row-eval"
TARGET_KEY = "qwen3_8_27b_r9700"
RECIPE_ID = "r9700-all-rowwise-f8e4m3-f32s-eval-v0"
F8E4M3_ROW_F32S = "F8E4M3_ROW_F32S"
ROW_SCALED_LAYOUT = "row-scaled-k128-v1"


def _candidate_tensor(spec: TensorSpec) -> TensorSpec:
    if spec.format in DIRECT_FORMATS:
        return spec
    return TensorSpec(
        name=spec.name,
        shape=spec.shape,
        format=F8E4M3_ROW_F32S,
        layout=ROW_SCALED_LAYOUT,
    )


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
        raise ValueError("rowwise-E4M3 evaluation inventory is not Qwen3.8-27B complete")
    if tuple(spec.name for spec in TENSOR_SPECS) != tuple(
        spec.name for spec in source_inventory.TENSOR_SPECS
    ) or tuple(spec.shape for spec in TENSOR_SPECS) != tuple(
        spec.shape for spec in source_inventory.TENSOR_SPECS
    ):
        raise ValueError("rowwise-E4M3 evaluation inventory differs from the source plan")
    expected_counts = {BF16: 582, FP32: 96, I32: 1, F8E4M3_ROW_F32S: 439}
    if FORMAT_COUNTS != expected_counts:
        raise ValueError(
            "rowwise-E4M3 format counts differ: "
            f"expected {expected_counts}, got {FORMAT_COUNTS}"
        )
    expected_bytes = {
        BF16: 58_647_008,
        FP32: 18_432,
        I32: 524_288,
        F8E4M3_ROW_F32S: 28_444_178_368,
    }
    if FORMAT_ENCODED_BYTES != dict(sorted(expected_bytes.items())):
        raise ValueError("rowwise-E4M3 encoded byte totals differ from the fixed plan")
    if TENSOR_ENCODED_BYTES != 28_503_368_096:
        raise ValueError("rowwise-E4M3 tensor payload byte total differs")
    if DEVICE_ARENA_BYTES != 28_503_382_016:
        raise ValueError("rowwise-E4M3 aligned device arena byte total differs")


validate_inventory()


__all__ = [
    "DEVICE_ARENA_BYTES",
    "F8E4M3_ROW_F32S",
    "FORMAT_COUNTS",
    "FORMAT_ENCODED_BYTES",
    "MODEL_ID",
    "OBJECT_SPECS",
    "RECIPE_ID",
    "RESOURCE_SPECS",
    "ROW_SCALED_LAYOUT",
    "TARGET_KEY",
    "TENSOR_ENCODED_BYTES",
    "TENSOR_SPECS",
    "WEIGHTS_ID",
    "validate_inventory",
]
