#!/usr/bin/env python3
"""Freeze one selected-artifact, single-image Qwen3.8 Vision input on CPU."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from safetensors.torch import save

from tools.reference.qwen3.common.frontend import Frontend
from tools.reference.qwen3.common.multimodal import load_messages
from tools.reference.qwen3_8_27b.bindings import VisionArtifactBinding


def prepare(weights: Path, messages: Path, output: Path) -> dict[str, object]:
    weights = weights.resolve(strict=True)
    messages = messages.resolve(strict=True)
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    if os.path.lexists(output):
        raise ValueError(f"refusing to overwrite prepared input: {output}")
    with VisionArtifactBinding.open(weights) as binding:
        batch = Frontend(binding).process(load_messages(messages), thinking=False)
    if (
        batch.image_grid_thw is None
        or batch.pixel_values is None
        or batch.video_grid_thw is not None
        or batch.pixel_values_videos is not None
        or int(batch.image_grid_thw.shape[0]) != 1
        or batch.image_tokens <= 0
        or batch.video_tokens != 0
    ):
        raise ValueError("selected Vision fixture must contain exactly one image and no video")
    contract = {
        "prompt_length": batch.prompt_length,
        "image_tokens": batch.image_tokens,
        "video_tokens": 0,
        "image_grid_thw": batch.image_grid_thw.tolist(),
        "pixel_values_shape": list(batch.pixel_values.shape),
        "rope_delta": batch.rope_delta,
        "thinking": False,
        "images": 1,
        "videos": 0,
    }
    tensors = {
        "input_ids": batch.input_ids.contiguous(),
        "mm_token_type_ids": batch.mm_token_type_ids.contiguous(),
        "position_ids": batch.position_ids.contiguous(),
        "pixel_values": batch.pixel_values.contiguous(),
        "image_grid_thw": batch.image_grid_thw.contiguous(),
    }
    pending: Path | None = None
    owner: tuple[int, int] | None = None
    published = False
    validated = False
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.pending-", dir=parent)
        pending = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(save(tensors, metadata={"contract": json.dumps(contract, sort_keys=True)}))
            stream.flush()
            os.fsync(stream.fileno())
            stat = os.fstat(stream.fileno())
            owner = (stat.st_dev, stat.st_ino)
        pending_stat = os.stat(pending, follow_symlinks=False)
        if (pending_stat.st_dev, pending_stat.st_ino) != owner:
            raise ValueError("prepared input pending namespace changed")
        os.link(pending, output)
        published = True
        target = os.stat(output, follow_symlinks=False)
        pending_stat = os.stat(pending, follow_symlinks=False)
        if (target.st_dev, target.st_ino) != owner or (
            pending_stat.st_dev, pending_stat.st_ino
        ) != owner:
            raise ValueError("published prepared input does not retain its pending inode")
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        validated = True
    finally:
        if pending is not None and owner is not None:
            try:
                current = os.stat(pending, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == owner:
                    pending.unlink()
            except FileNotFoundError:
                pass
        if published and owner is not None and not validated:
            try:
                target = os.stat(output, follow_symlinks=False)
                if (target.st_dev, target.st_ino) == owner:
                    output.unlink()
            except FileNotFoundError:
                pass
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--messages", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        contract = prepare(args.weights, args.messages, args.out)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(contract, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
