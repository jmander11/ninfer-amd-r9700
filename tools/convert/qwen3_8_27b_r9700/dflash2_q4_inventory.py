"""Explicit BF16-source Q4G64 DFlash2 companion inventories for R9700.

The DFlash matrices are persistent signed Q4G64 weights. Their represented-BF16 inputs are
quantized by the compile-selected adaptive A8G64 Linear profile at execution time. Selector
predecessor/successor codebooks and every non-matrix value remain source BF16.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct

from tools.artifact.layouts import encoded_size
from tools.convert.qwen3.common.inventory import BF16, Q4, TensorSpec, tensor_spec

from . import fp8_hybrid_inventory, q4_inventory, q4_w8_mse_inventory


MODEL_ID = "qwen3.8-27b"
ALL_Q4_BASE_WEIGHTS_ID = q4_inventory.WEIGHTS_ID
MIXED_BASE_WEIGHTS_ID = q4_w8_mse_inventory.WEIGHTS_ID
HYBRID_BASE_WEIGHTS_ID = fp8_hybrid_inventory.WEIGHTS_ID
ALL_Q4_WEIGHTS_ID = "r9700-q4g64-n16k16-dflash2-q4-eval"
MIXED_WEIGHTS_ID = "r9700-q4-w8-mse-n16k16-dflash2-q4-eval"
HYBRID_WEIGHTS_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval"
TARGET_KEY = "qwen3_8_27b_r9700"
RECIPE_ID = "r9700-dflash2-all-q4g64-n16k16-bf16-codebook-eval-v1"

DFLASH2_LAYERS = 5
HIDDEN = 5120
INTERMEDIATE = 17408
FEATURE_ROWS = 25600
QKV_ROWS = 6144
QUERY_SIZE = 4096
CONV_PROJECTION_ROWS = 1280
SELECTOR_RANK = 256
VOCAB = 248320


@dataclass(frozen=True, slots=True)
class SourceBinding:
    artifact: TensorSpec
    sources: tuple[str, ...]


def _direct(name: str, shape: tuple[int, ...], source: str) -> SourceBinding:
    return SourceBinding(tensor_spec(name, shape, BF16), (source,))


def _matrix(name: str, shape: tuple[int, ...], *sources: str) -> SourceBinding:
    return SourceBinding(tensor_spec(name, shape, Q4), tuple(sources))


def _bindings() -> tuple[SourceBinding, ...]:
    result: list[SourceBinding] = [
        _matrix("dflash/feature_projection", (HIDDEN, FEATURE_ROWS), "fc.weight"),
        _direct("dflash/context_norm", (HIDDEN,), "hidden_norm.weight"),
    ]
    for layer in range(DFLASH2_LAYERS):
        out = f"dflash/layers/{layer}/"
        src = f"layers.{layer}."
        result.extend(
            (
                _direct(out + "input_norm", (HIDDEN,), src + "input_layernorm.weight"),
                _matrix(
                    out + "attention/query_key_value",
                    (QKV_ROWS, HIDDEN),
                    src + "self_attn.q_proj.weight",
                    src + "self_attn.k_proj.weight",
                    src + "self_attn.v_proj.weight",
                ),
                _direct(out + "attention/query_norm", (128,), src + "self_attn.q_norm.weight"),
                _direct(out + "attention/key_norm", (128,), src + "self_attn.k_norm.weight"),
                _matrix(out + "attention/output", (HIDDEN, QUERY_SIZE),
                        src + "self_attn.o_proj.weight"),
                _direct(out + "attention_conv/base_kernel", (HIDDEN, 2, 2),
                        src + "attention_conv.base_kernel"),
                _matrix(out + "attention_conv/kernel_projection",
                        (CONV_PROJECTION_ROWS, HIDDEN),
                        src + "attention_conv.kernel_projection.weight"),
                _direct(out + "post_attention_norm", (HIDDEN,),
                        src + "post_attention_layernorm.weight"),
                _matrix(
                    out + "mlp/gate_up",
                    (2 * INTERMEDIATE, HIDDEN),
                    src + "mlp.gate_proj.weight",
                    src + "mlp.up_proj.weight",
                ),
                _matrix(out + "mlp/down", (HIDDEN, INTERMEDIATE),
                        src + "mlp.down_proj.weight"),
                _direct(out + "mlp_conv/base_kernel", (HIDDEN, 2, 2),
                        src + "mlp_conv.base_kernel"),
                _matrix(out + "mlp_conv/kernel_projection",
                        (CONV_PROJECTION_ROWS, HIDDEN),
                        src + "mlp_conv.kernel_projection.weight"),
            )
        )
    result.extend(
        (
            _direct("dflash/final_norm", (HIDDEN,), "norm.weight"),
            _matrix("dflash/selector/hidden_projection", (SELECTOR_RANK, HIDDEN),
                    "candidate_selector.hidden_projection.weight"),
            _direct("dflash/selector/predecessor_codebook", (SELECTOR_RANK, VOCAB),
                    "candidate_selector.predecessor_codebook"),
            _direct("dflash/selector/successor_codebook", (SELECTOR_RANK, VOCAB),
                    "candidate_selector.successor_codebook"),
        )
    )
    return tuple(result)


SOURCE_BINDINGS = _bindings()
TENSOR_SPECS = tuple(binding.artifact for binding in SOURCE_BINDINGS)
SOURCE_NAMES = tuple(name for binding in SOURCE_BINDINGS for name in binding.sources)
FORMAT_COUNTS = dict(Counter(spec.format for spec in TENSOR_SPECS))
FORMAT_ENCODED_BYTES = {
    numeric_format: sum(
        encoded_size(spec.layout, spec.format, spec.shape)
        for spec in TENSOR_SPECS
        if spec.format == numeric_format
    )
    for numeric_format in sorted(FORMAT_COUNTS)
}
TENSOR_ENCODED_BYTES = sum(FORMAT_ENCODED_BYTES.values())

ALL_Q4_OBJECT_SPECS = q4_inventory.OBJECT_SPECS + TENSOR_SPECS
MIXED_OBJECT_SPECS = q4_w8_mse_inventory.OBJECT_SPECS + TENSOR_SPECS
HYBRID_OBJECT_SPECS = fp8_hybrid_inventory.OBJECT_SPECS + TENSOR_SPECS
ALL_Q4_TENSOR_BYTES = q4_inventory.TENSOR_ENCODED_BYTES + TENSOR_ENCODED_BYTES
MIXED_TENSOR_BYTES = q4_w8_mse_inventory.TENSOR_ENCODED_BYTES + TENSOR_ENCODED_BYTES
HYBRID_TENSOR_BYTES = fp8_hybrid_inventory.TENSOR_ENCODED_BYTES + TENSOR_ENCODED_BYTES
ALL_Q4_DEVICE_ARENA_BYTES = q4_inventory.DEVICE_ARENA_BYTES + TENSOR_ENCODED_BYTES
MIXED_DEVICE_ARENA_BYTES = q4_w8_mse_inventory.DEVICE_ARENA_BYTES + TENSOR_ENCODED_BYTES
HYBRID_DEVICE_ARENA_BYTES = fp8_hybrid_inventory.DEVICE_ARENA_BYTES + TENSOR_ENCODED_BYTES


def _safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    with path.open("rb") as source:
        length_raw = source.read(8)
        if len(length_raw) != 8:
            raise ValueError("DFlash2 safetensors prefix is truncated")
        header_bytes = struct.unpack("<Q", length_raw)[0]
        if header_bytes == 0 or header_bytes > path.stat().st_size - 8:
            raise ValueError("DFlash2 safetensors header length is invalid")
        value = json.loads(source.read(header_bytes))
    if not isinstance(value, dict):
        raise ValueError("DFlash2 safetensors header must be an object")
    value.pop("__metadata__", None)
    return value


def _source_shape_for(binding: SourceBinding, name: str) -> tuple[int, ...]:
    rows, *rest = binding.artifact.shape
    if len(binding.sources) == 1:
        if binding.artifact.name.endswith("codebook"):
            return (VOCAB, SELECTOR_RANK)
        if binding.artifact.name.endswith("base_kernel"):
            return (2, 2, HIDDEN)
        return binding.artifact.shape
    if binding.artifact.name.endswith("query_key_value"):
        index = binding.sources.index(name)
        return ((QUERY_SIZE, 1024, 1024)[index], HIDDEN)
    if binding.artifact.name.endswith("gate_up"):
        return (rows // 2, *rest)
    raise ValueError(f"unhandled fused DFlash2 source binding: {binding.artifact.name}")


def validate_source(model_dir: str | Path) -> dict[str, object]:
    root = Path(model_dir)
    config_path = root / "config.json"
    readme_path = root / "README.md"
    tensor_path = root / "model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "architectures": ["DFlash2DraftModel"],
        "dtype": "bfloat16",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": DFLASH2_LAYERS,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "sliding_window": 2048,
        "vocab_size": VOCAB,
    }
    mismatches = [
        f"{key}: expected {wanted!r}, got {config.get(key)!r}"
        for key, wanted in expected.items()
        if config.get(key) != wanted
    ]
    dflash_expected = {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 248070,
        "selector_rank": SELECTOR_RANK,
        "selector_top_k": 16,
        "target_layer_ids": [5, 19, 33, 47, 61],
    }
    dflash = config.get("dflash_config")
    if not isinstance(dflash, dict):
        mismatches.append("dflash_config is missing")
    else:
        mismatches.extend(
            f"dflash_config.{key}: expected {wanted!r}, got {dflash.get(key)!r}"
            for key, wanted in dflash_expected.items()
            if dflash.get(key) != wanted
        )
    if mismatches:
        raise ValueError("DFlash2 config mismatch:\n  " + "\n  ".join(mismatches))

    header = _safetensors_header(tensor_path)
    if set(header) != set(SOURCE_NAMES):
        missing = sorted(set(SOURCE_NAMES) - set(header))
        extra = sorted(set(header) - set(SOURCE_NAMES))
        raise ValueError(f"DFlash2 source inventory mismatch: missing={missing}, extra={extra}")
    for binding in SOURCE_BINDINGS:
        for name in binding.sources:
            entry = header[name]
            expected_shape = _source_shape_for(binding, name)
            if entry.get("dtype") != "BF16" or tuple(entry.get("shape", ())) != expected_shape:
                raise ValueError(
                    f"{name}: expected BF16 {expected_shape}, got "
                    f"{entry.get('dtype')} {entry.get('shape')}"
                )
    with tensor_path.open("rb") as source:
        source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    with config_path.open("rb") as source:
        config_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    with readme_path.open("rb") as source:
        readme_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "config": expected | {"dflash_config": dflash_expected},
        "config_sha256": config_sha256,
        "readme_sha256": readme_sha256,
        "tensor_count": len(header),
        "safetensors_bytes": tensor_path.stat().st_size,
        "safetensors_sha256": source_sha256,
    }


def validate_inventory() -> None:
    if len(SOURCE_BINDINGS) != 66 or len(SOURCE_NAMES) != 81:
        raise ValueError("DFlash2 fused/source inventory counts differ")
    if len(set(SOURCE_NAMES)) != len(SOURCE_NAMES):
        raise ValueError("DFlash2 source inventory contains duplicate names")
    if FORMAT_COUNTS != {BF16: 34, Q4: 32}:
        raise ValueError(f"DFlash2 format counts differ: {FORMAT_COUNTS}")
    if FORMAT_ENCODED_BYTES != {BF16: 254_814_720, Q4: 954_654_720}:
        raise ValueError(f"DFlash2 encoded byte totals differ: {FORMAT_ENCODED_BYTES}")
    if TENSOR_ENCODED_BYTES != 1_209_469_440:
        raise ValueError("DFlash2 tensor byte total differs")
    if any(len(specs) != 1190 for specs in (
        ALL_Q4_OBJECT_SPECS, MIXED_OBJECT_SPECS, HYBRID_OBJECT_SPECS,
    )):
        raise ValueError("DFlash2 combined object inventory is incomplete")
    if ALL_Q4_DEVICE_ARENA_BYTES != 16_369_285_120:
        raise ValueError("all-Q4 DFlash2 arena projection differs")
    if MIXED_DEVICE_ARENA_BYTES != 24_077_660_672:
        raise ValueError("mixed DFlash2 arena projection differs")
    if HYBRID_DEVICE_ARENA_BYTES != 22_750_001_152:
        raise ValueError("four-role hybrid DFlash2 arena projection differs")


validate_inventory()


__all__ = [
    "ALL_Q4_BASE_WEIGHTS_ID",
    "ALL_Q4_DEVICE_ARENA_BYTES",
    "ALL_Q4_OBJECT_SPECS",
    "ALL_Q4_TENSOR_BYTES",
    "ALL_Q4_WEIGHTS_ID",
    "FORMAT_COUNTS",
    "FORMAT_ENCODED_BYTES",
    "HYBRID_BASE_WEIGHTS_ID",
    "HYBRID_DEVICE_ARENA_BYTES",
    "HYBRID_OBJECT_SPECS",
    "HYBRID_TENSOR_BYTES",
    "HYBRID_WEIGHTS_ID",
    "MIXED_BASE_WEIGHTS_ID",
    "MIXED_DEVICE_ARENA_BYTES",
    "MIXED_OBJECT_SPECS",
    "MIXED_TENSOR_BYTES",
    "MIXED_WEIGHTS_ID",
    "MODEL_ID",
    "RECIPE_ID",
    "SOURCE_BINDINGS",
    "SOURCE_NAMES",
    "TARGET_KEY",
    "TENSOR_ENCODED_BYTES",
    "TENSOR_SPECS",
    "validate_inventory",
    "validate_source",
]
