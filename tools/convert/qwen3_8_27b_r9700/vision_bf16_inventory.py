"""Exact BF16 Vision signatures on unchanged LQ4 Text weights."""

from tools.convert.qwen3.common.inventory import DIRECT_FORMATS, tensor_spec

from . import source_inventory

WEIGHTS_ID = "r9700-lq4-vision-bf16"
VISION_SPECS = tuple(
    tensor_spec(s.name, s.shape, "BF16") for s in source_inventory.VISION_TENSOR_SPECS
)
MATRICES = {
    s.name: s
    for s, original in zip(VISION_SPECS, source_inventory.VISION_TENSOR_SPECS, strict=True)
    if original.format not in DIRECT_FORMATS
}
