"""Fixed selective-protected AMD evaluation recipe; never a production selection."""

from collections import Counter

from tools.artifact.layouts import align_up, encoded_size
from tools.convert.qwen3.common.inventory import BF16, W8, TensorSpec, tensor_spec
from . import q4_inventory
from .e4m3_inventory import F8E4M3_ROW_F32S, ROW_SCALED_LAYOUT

MODEL_ID = q4_inventory.MODEL_ID
TARGET_KEY = q4_inventory.TARGET_KEY
WEIGHTS_ID = "r9700-q4-selective-protected-n16k16-eval"
RECIPE_ID = "r9700-q4-selective-protected-n16k16-v1"
W8_NAMES = frozenset(("text/token_embedding", "text/output_head"))
BF16_NAMES = frozenset(
    [f"text/layers/{layer}/attention/{role}"
     for layer in (3, 7, 11, 15, 19, 23) for role in ("query_key", "gate_value")]
    + [f"text/layers/{layer}/attention/output" for layer in (3, 7)]
    + ["text/layers/4/gdn/output"]
)
FP8_NAMES = frozenset(
    ["text/layers/11/attention/output"]
    + [f"text/layers/{layer}/attention/{role}"
       for layer in (27, 31, 51) for role in ("query_key", "gate_value")]
    + [f"text/layers/{layer}/mlp/{role}"
       for layer in (62, 63) for role in ("gate_up", "down")]
)
CHANGED_FORMATS = {**dict.fromkeys(W8_NAMES, W8), **dict.fromkeys(BF16_NAMES, BF16),
                   **dict.fromkeys(FP8_NAMES, F8E4M3_ROW_F32S)}


def _replace(spec):
    fmt = CHANGED_FORMATS.get(spec.name)
    if fmt is None:
        return spec
    if fmt == F8E4M3_ROW_F32S:
        return TensorSpec(spec.name, spec.shape, fmt, ROW_SCALED_LAYOUT)
    return tensor_spec(spec.name, spec.shape, fmt)


RESOURCE_SPECS = q4_inventory.RESOURCE_SPECS
TENSOR_SPECS = tuple(_replace(spec) for spec in q4_inventory.TENSOR_SPECS)
OBJECT_SPECS = RESOURCE_SPECS + TENSOR_SPECS
FORMAT_COUNTS = dict(Counter(spec.format for spec in TENSOR_SPECS))
FORMAT_ENCODED_BYTES = {
    fmt: sum(encoded_size(s.layout, s.format, s.shape) for s in TENSOR_SPECS if s.format == fmt)
    for fmt in sorted(FORMAT_COUNTS)
}
TENSOR_ENCODED_BYTES = sum(FORMAT_ENCODED_BYTES.values())
DEVICE_ARENA_BYTES = 0
for _spec in TENSOR_SPECS:
    DEVICE_ARENA_BYTES = align_up(DEVICE_ARENA_BYTES, 256) + encoded_size(
        _spec.layout, _spec.format, _spec.shape)


def validate_inventory():
    base = {spec.name: spec for spec in q4_inventory.TENSOR_SPECS}
    if (len(W8_NAMES), len(BF16_NAMES), len(FP8_NAMES), len(CHANGED_FORMATS)) != (2, 15, 11, 28):
        raise ValueError("selective-protected recipe must change exactly 2 W8 + 15 BF16 + 11 FP8 objects")
    if any(name not in base or base[name].format != q4_inventory.Q4 for name in CHANGED_FORMATS):
        raise ValueError("selective-protected replacement is not an existing Q4 matrix")
    if len(OBJECT_SPECS) != 1124 or len(TENSOR_SPECS) != 1118:
        raise ValueError("selective-protected inventory is incomplete")


validate_inventory()
