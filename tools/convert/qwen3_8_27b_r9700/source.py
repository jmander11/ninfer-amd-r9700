"""Qwen3.8 source-checkpoint validation and logical tensor materialization."""

from __future__ import annotations

from typing import Mapping

import torch

from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3.common import conversion

from . import draft_head, inventory, source_inventory, source_recipe


_ROOT_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "language_model_only": False,
    "tie_word_embeddings": False,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
}
_TEXT_CONFIG = {
    "num_hidden_layers": 64,
    "full_attention_interval": 4,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "mamba_ssm_dtype": "float32",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "tie_word_embeddings": False,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
}
_ROPE_CONFIG = {"rope_theta": 10000000, "mrope_section": [11, 11, 10]}
_VISION_CONFIG = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "out_hidden_size": 5120,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
}


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    conversion.check_members("config", config, _ROOT_CONFIG)
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("config.json must contain text_config and vision_config")
    conversion.check_members("text_config", text, _TEXT_CONFIG)
    expected_layer_types = tuple(
        "full_attention" if layer in source_inventory.FULL_ATTENTION_LAYERS else "linear_attention"
        for layer in range(64)
    )
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != expected_layer_types:
        raise ValueError("text_config.layer_types does not match the 64-layer schedule")
    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    conversion.check_members("text_config.rope_parameters", rope, _ROPE_CONFIG)
    conversion.check_members("vision_config", vision, _VISION_CONFIG)
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": {name: text[name] for name in _TEXT_CONFIG},
        "layer_types": {
            "layers": len(layer_types),
            "full_attention": len(source_inventory.FULL_ATTENTION_LAYERS),
            "linear_attention": 64 - len(source_inventory.FULL_ATTENTION_LAYERS),
            "full_attention_layers": list(source_inventory.FULL_ATTENTION_LAYERS),
        },
        "rope": {name: rope[name] for name in _ROPE_CONFIG},
        "vision": {name: vision[name] for name in _VISION_CONFIG},
        "mtp_num_hidden_layers": text["mtp_num_hidden_layers"],
        "vision_token_ids": {
            name: config[name]
            for name in ("vision_start_token_id", "vision_end_token_id", "image_token_id", "video_token_id")
        },
    }


def materialize_tensor(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    draft: draft_head.DraftHeadContext,
) -> torch.Tensor:
    derived = None
    if spec.name in (draft_head.DRAFT_HEAD_OBJECT, draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT):
        derived = {
            draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT:
                draft_head.materialize_draft_head_token_ids(draft)
        }
    tensor = source_recipe.materialize_recipe(
        source_recipe.RECIPES_BY_NAME[spec.name], reader, derived
    )
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


__all__ = ["materialize_tensor", "validate_config"]
