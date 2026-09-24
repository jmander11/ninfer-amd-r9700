"""W8G32 plus BF16 GDN query/key evaluation inventory for Qwen3.8-27B.

The query/key input projection in every one of the 48 GDN layers retains its
represented source BF16 values. Every other persistent matrix remains W8G32.
This is one complete source-Q4 semantic family, never a selected layer subset.
"""

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
    tensor_spec,
)

from . import source_inventory


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "r9700-w8-bf16-gdn-qk-eval"
TARGET_KEY = "qwen3_8_27b_r9700"
RECIPE_ID = "r9700-w8g32-bf16-gdn-query-key-eval-v0"


def _is_promoted(spec: TensorSpec) -> bool:
    return any(
        spec.name == f"text/layers/{layer}/gdn/query_key"
        for layer in source_inventory.GDN_LAYERS
    )


def _candidate_tensor(spec: TensorSpec) -> TensorSpec:
    if _is_promoted(spec):
        numeric_format = BF16
    else:
        numeric_format = spec.format if spec.format in DIRECT_FORMATS else W8
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
        raise ValueError("W8/BF16-GDN-QK evaluator is not Qwen3.8-27B complete")
    if tuple(spec.name for spec in TENSOR_SPECS) != tuple(
        spec.name for spec in source_inventory.TENSOR_SPECS
    ) or tuple(spec.shape for spec in TENSOR_SPECS) != tuple(
        spec.shape for spec in source_inventory.TENSOR_SPECS
    ):
        raise ValueError("W8/BF16-GDN-QK inventory differs from the source plan")
    expected_counts = {BF16: 630, FP32: 96, I32: 1, W8: 391}
    if FORMAT_COUNTS != expected_counts:
        raise ValueError(
            "W8/BF16-GDN-QK format counts differ: "
            f"expected {expected_counts}, got {FORMAT_COUNTS}"
        )
    expected_bytes = {
        BF16: 2_071_912_928,
        FP32: 18_432,
        I32: 524_288,
        W8: 29_131_676_544,
    }
    if FORMAT_ENCODED_BYTES != dict(sorted(expected_bytes.items())):
        raise ValueError("W8/BF16-GDN-QK encoded byte totals differ from the fixed plan")
    if TENSOR_ENCODED_BYTES != 31_204_132_192:
        raise ValueError("W8/BF16-GDN-QK tensor payload byte total differs")
    if DEVICE_ARENA_BYTES != 31_204_144_384:
        raise ValueError("W8/BF16-GDN-QK aligned device arena byte total differs")
    promoted = tuple(spec for spec in TENSOR_SPECS if _is_promoted(spec))
    if len(promoted) != 48 or any(spec.format != BF16 for spec in promoted):
        raise ValueError("W8/BF16-GDN-QK promoted role family changed")
    for source, candidate in zip(
        source_inventory.TENSOR_SPECS, TENSOR_SPECS, strict=True
    ):
        expected = BF16 if _is_promoted(source) else (
            source.format if source.format in DIRECT_FORMATS else W8
        )
        if candidate.format != expected:
            raise ValueError(f"W8/BF16-GDN-QK format mismatch for {candidate.name}")


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
