"""Same-format source-MSE Q4G64/W8G32 evaluator for Qwen3.8-27B.

The object names, order, shapes, formats, layouts, and byte plan are identical
to the explicit source-Q4/promoted-W8 evaluator.  Only conversion-time scale
selection differs; no calibration input or runtime format selector is added.
"""

from __future__ import annotations

from tools.convert.qwen3.common.inventory import (
    DIRECT_FORMATS,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
)

from . import q4_w8_inventory


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "r9700-q4-w8-mse-n16k16-eval"
TARGET_KEY = "qwen3_8_27b_r9700"
RECIPE_ID = "r9700-source-q4-n16k16-promoted-w8-source-mse8-eval-v1"

RESOURCE_SPECS: tuple[ResourceSpec, ...] = q4_w8_inventory.RESOURCE_SPECS
TENSOR_SPECS: tuple[TensorSpec, ...] = q4_w8_inventory.TENSOR_SPECS
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS
FORMAT_COUNTS: dict[str, int] = dict(q4_w8_inventory.FORMAT_COUNTS)
FORMAT_ENCODED_BYTES: dict[str, int] = dict(q4_w8_inventory.FORMAT_ENCODED_BYTES)
TENSOR_ENCODED_BYTES = q4_w8_inventory.TENSOR_ENCODED_BYTES
DEVICE_ARENA_BYTES = q4_w8_inventory.DEVICE_ARENA_BYTES


def validate_inventory() -> None:
    q4_w8_inventory.validate_inventory()
    if len(TENSOR_SPECS) != 1118 or len(OBJECT_SPECS) != 1124:
        raise ValueError("Q4/W8 source-MSE evaluator is not Qwen3.8-27B complete")
    if TENSOR_SPECS != q4_w8_inventory.TENSOR_SPECS:
        raise ValueError("Q4/W8 source-MSE evaluator changed the tensor plan")
    if RESOURCE_SPECS != q4_w8_inventory.RESOURCE_SPECS:
        raise ValueError("Q4/W8 source-MSE evaluator changed the resource plan")
    if FORMAT_COUNTS != q4_w8_inventory.FORMAT_COUNTS:
        raise ValueError("Q4/W8 source-MSE evaluator changed format counts")
    if FORMAT_ENCODED_BYTES != q4_w8_inventory.FORMAT_ENCODED_BYTES:
        raise ValueError("Q4/W8 source-MSE evaluator changed encoded format bytes")
    if TENSOR_ENCODED_BYTES != 22_868_177_312:
        raise ValueError("Q4/W8 source-MSE evaluator changed tensor bytes")
    if DEVICE_ARENA_BYTES != 22_868_191_232:
        raise ValueError("Q4/W8 source-MSE evaluator changed device arena bytes")


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
