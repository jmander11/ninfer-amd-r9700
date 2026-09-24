"""Decision-owned four-role rowwise-E4M3/all-other-Q4 evaluation inventory."""

from __future__ import annotations

from collections import Counter

from tools.artifact.layouts import align_up, encoded_size
from tools.convert.qwen3.common.inventory import (
    BF16,
    DIRECT_FORMATS,
    FP32,
    I32,
    Q4,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
)

from . import fp8_hybrid_decision, q4_inventory
from .e4m3_inventory import F8E4M3_ROW_F32S, ROW_SCALED_LAYOUT


MODEL_ID = q4_inventory.MODEL_ID
TARGET_KEY = q4_inventory.TARGET_KEY
WEIGHTS_ID = fp8_hybrid_decision.DECISION.weights_id
RECIPE_ID = fp8_hybrid_decision.DECISION.recipe_id
SELECTION_SHA256 = fp8_hybrid_decision.DECISION.selection_sha256
SELECTED_MATRIX_NAMES = fp8_hybrid_decision.selected_matrix_names()


def _candidate_tensor(spec: TensorSpec) -> TensorSpec:
    if spec.name not in SELECTED_MATRIX_NAMES:
        return spec
    return TensorSpec(
        name=spec.name,
        shape=spec.shape,
        format=F8E4M3_ROW_F32S,
        layout=ROW_SCALED_LAYOUT,
    )


RESOURCE_SPECS: tuple[ResourceSpec, ...] = q4_inventory.RESOURCE_SPECS
TENSOR_SPECS: tuple[TensorSpec, ...] = tuple(
    _candidate_tensor(spec) for spec in q4_inventory.TENSOR_SPECS
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
        raise ValueError("FP8/Q4 hybrid inventory is not Qwen3.8-27B complete")
    if tuple((spec.name, spec.shape) for spec in TENSOR_SPECS) != tuple(
        (spec.name, spec.shape) for spec in q4_inventory.TENSOR_SPECS
    ):
        raise ValueError("FP8/Q4 hybrid inventory differs from the all-Q4 object plan")
    q4_names = {spec.name for spec in q4_inventory.TENSOR_SPECS if spec.format == Q4}
    if not SELECTED_MATRIX_NAMES <= q4_names:
        raise ValueError("FP8/Q4 hybrid selection contains a non-Q4 source object")
    represented_selected = {
        spec.name for spec in TENSOR_SPECS if spec.format == F8E4M3_ROW_F32S
    }
    if represented_selected != SELECTED_MATRIX_NAMES:
        raise ValueError("FP8/Q4 hybrid inventory differs from its decision-owned selection")
    expected_counts = {BF16: 582, FP32: 96, I32: 1, Q4: 295, F8E4M3_ROW_F32S: 144}
    if FORMAT_COUNTS != expected_counts:
        raise ValueError(
            f"FP8/Q4 hybrid format counts differ: expected {expected_counts}, got {FORMAT_COUNTS}"
        )
    expected_bytes = {
        BF16: 58_647_008,
        FP32: 18_432,
        I32: 524_288,
        Q4: 7_881_166_272,
        F8E4M3_ROW_F32S: 13_600_161_792,
    }
    if FORMAT_ENCODED_BYTES != dict(sorted(expected_bytes.items())):
        raise ValueError("FP8/Q4 hybrid encoded byte totals differ from the fixed plan")
    if TENSOR_ENCODED_BYTES != 21_540_517_792:
        raise ValueError("FP8/Q4 hybrid tensor payload byte total differs")
    if DEVICE_ARENA_BYTES != 21_540_531_712:
        raise ValueError("FP8/Q4 hybrid aligned device arena byte total differs")


validate_inventory()


__all__ = [
    "DEVICE_ARENA_BYTES",
    "DIRECT_FORMATS",
    "F8E4M3_ROW_F32S",
    "FORMAT_COUNTS",
    "FORMAT_ENCODED_BYTES",
    "MODEL_ID",
    "OBJECT_SPECS",
    "Q4",
    "RECIPE_ID",
    "RESOURCE_SPECS",
    "ROW_SCALED_LAYOUT",
    "SELECTION_SHA256",
    "SELECTED_MATRIX_NAMES",
    "TARGET_KEY",
    "TENSOR_ENCODED_BYTES",
    "TENSOR_SPECS",
    "TensorSpec",
    "WEIGHTS_ID",
    "validate_inventory",
]
