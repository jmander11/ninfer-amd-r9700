"""Exact frontend resources admitted by the Qwen3.8-27B R9700 converter."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tools.convert.qwen3.common import conversion

from . import inventory


OFFICIAL_RESOURCE_SHA256 = {
    "frontend/tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "frontend/tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "frontend/chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "frontend/generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    "frontend/preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "frontend/video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
}


def load_resources(model_dir: str | Path) -> tuple[conversion.ResourcePayload, ...]:
    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    spec_names = tuple(spec.name for spec in inventory.RESOURCE_SPECS)
    if spec_names != expected_names:
        raise ValueError("converter resource inventory differs from Qwen3.8")
    resources = conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)
    if tuple(resource.name for resource in resources) != expected_names:
        raise ValueError("Qwen3.8 frontend resource set is incomplete")
    for resource in resources:
        actual = hashlib.sha256(resource.data).hexdigest()
        expected = OFFICIAL_RESOURCE_SHA256[resource.name]
        if actual != expected:
            filename = resource.name.removeprefix("frontend/")
            raise ValueError(
                f"official Qwen3.8 resource hash mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )
    return resources


__all__ = ["OFFICIAL_RESOURCE_SHA256", "load_resources"]
