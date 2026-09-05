"""Compare native artifact Vision activations with the source BF16 vision tower."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open

from tools.reference.qwen3.common.frontend import Frontend
from tools.reference.qwen3.common.multimodal import MultimodalBatch, load_messages
from tools.reference.qwen3_8_27b.bindings import VisionArtifactBinding
from tools.reference.qwen3_8_27b.vision import VisionEncoder
from tools.reference.qwen3_8_27b_bf16.protocol import validate_checkpoint_files


CAPTURE_LAYERS = {0, 13, 26}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(model_dir: Path) -> dict[str, object]:
    config = model_dir / "config.json"
    index = model_dir / "model.safetensors.index.json"
    weight_map = validate_checkpoint_files(model_dir)
    shard_names = sorted(set(weight_map.values()))
    return {
        "config_sha256": sha256_file(config),
        "index_sha256": sha256_file(index),
        "indexed_tensor_count": len(weight_map),
        "shards": {
            name: {"bytes": (model_dir / name).stat().st_size}
            for name in shard_names
        },
    }


def load_prepared_batch(path: Path) -> tuple[MultimodalBatch, dict[str, object]]:
    with safe_open(path, framework="pt", device="cpu") as source:
        if set(source.keys()) != {
            "image_grid_thw", "input_ids", "mm_token_type_ids", "pixel_values",
            "position_ids",
        }:
            raise ValueError("prepared Vision input has an unexpected tensor inventory")
        metadata = source.metadata()
        if set(metadata) != {"contract"}:
            raise ValueError("prepared Vision input lacks its exact contract")
        contract = json.loads(metadata["contract"])
        tensors = {name: source.get_tensor(name) for name in source.keys()}
    batch = MultimodalBatch(
        input_ids=tensors["input_ids"],
        mm_token_type_ids=tensors["mm_token_type_ids"],
        position_ids=tensors["position_ids"],
        rope_delta=int(contract["rope_delta"]),
        pixel_values=tensors["pixel_values"],
        image_grid_thw=tensors["image_grid_thw"],
        pixel_values_videos=None,
        video_grid_thw=None,
    )
    actual = {
        "prompt_length": batch.prompt_length,
        "image_tokens": batch.image_tokens,
        "video_tokens": batch.video_tokens,
        "image_grid_thw": batch.image_grid_thw.tolist(),
        "pixel_values_shape": list(batch.pixel_values.shape),
        "rope_delta": batch.rope_delta,
        "thinking": False,
        "images": 1,
        "videos": 0,
    }
    if contract != actual:
        raise ValueError("prepared Vision tensors differ from their contract")
    return batch, contract


def publish_json_create_only(path: Path, value: dict[str, object]) -> None:
    """Durably publish one report without following or replacing a namespace."""
    parent = path.parent.resolve(strict=True)
    destination = parent / path.name
    if os.path.lexists(destination):
        raise ValueError(f"refusing to overwrite output: {destination}")
    pending: Path | None = None
    owner: tuple[int, int] | None = None
    published = False
    validated = False
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.pending-", dir=parent)
        pending = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
            stat = os.fstat(output.fileno())
            owner = (stat.st_dev, stat.st_ino)
        os.link(pending, destination)
        published = True
        target = os.stat(destination, follow_symlinks=False)
        if (target.st_dev, target.st_ino) != owner:
            raise ValueError("published report does not retain the pending inode")
        if json.loads(destination.read_text(encoding="utf-8")) != value:
            raise ValueError("published report differs from computed evidence")
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
        if published and not validated and owner is not None:
            try:
                current = os.stat(destination, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == owner:
                    destination.unlink()
            except FileNotFoundError:
                pass


def load_hf_vision(model_dir: Path):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_config(config.vision_config).to(dtype=torch.bfloat16)
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.startswith("model.visual."):
            shards.setdefault(shard, []).append(name)
    state = {}
    for shard, names in sorted(shards.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as source:
            for name in names:
                state[name.removeprefix("model.visual.")] = source.get_tensor(name)
    missing, unexpected = model.load_state_dict(state)
    if missing or unexpected:
        raise RuntimeError(f"HF vision state mismatch: missing={missing}, unexpected={unexpected}")
    return model


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    if actual.shape != expected.shape:
        raise ValueError(
            f"capture shape mismatch: artifact={tuple(actual.shape)}, source={tuple(expected.shape)}"
        )
    af, ef = actual.float(), expected.float()
    if not torch.isfinite(af).all() or not torch.isfinite(ef).all():
        raise ValueError("capture contains a non-finite value")
    diff = af - ef
    rmse = diff.square().mean().sqrt()
    reference_rms = ef.square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(af.flatten(), ef.flatten(), dim=0)
    result = {
        "shape": list(actual.shape),
        "rmse": float(rmse),
        "relative_rmse": float(rmse / reference_rms),
        "cosine": float(cosine),
        "actual_norm": float(af.norm()),
        "reference_norm": float(ef.norm()),
    }
    if not all(math.isfinite(result[name]) for name in (
        "rmse", "relative_rmse", "cosine", "actual_norm", "reference_norm"
    )):
        raise ValueError("capture metrics are not finite")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--prepared-input")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    weights_path = Path(args.weights).resolve()
    messages_path = Path(args.messages).resolve()
    artifact_sha256 = sha256_file(weights_path)
    source = source_provenance(model_dir)
    messages_sha256 = sha256_file(messages_path)
    ninfer_captures: dict[str, torch.Tensor] = {}
    with VisionArtifactBinding.open(weights_path) as binding, torch.inference_mode():
        artifact_identity = {
            "model_id": binding.identity.model_id,
            "weights_id": binding.identity.weights_id,
        }
        if args.prepared_input:
            if args.thinking:
                raise ValueError("prepared selected Vision input requires thinking off")
            batch, prepared_contract = load_prepared_batch(Path(args.prepared_input))
        else:
            batch = Frontend(binding).process(
                load_messages(args.messages), thinking=args.thinking
            )
            prepared_contract = None
        encoder = VisionEncoder(binding, device=device, compile_codec=True)
        try:
            ninfer_output = encoder.encode(
                batch.pixel_values, batch.image_grid_thw,
                batch.pixel_values_videos, batch.video_grid_thw,
                tap=lambda name, value: ninfer_captures.__setitem__(
                    name, value.detach().to(device="cpu", dtype=torch.bfloat16)
                ),
            )
        finally:
            encoder.close()
        ninfer_captures["merger"] = torch.cat(
            [
                value
                for value in (ninfer_output.image_embeddings, ninfer_output.video_embeddings)
                if value is not None
            ]
        ).detach().to(device="cpu", dtype=torch.bfloat16)
        vision_stats = ninfer_output.stats
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_captures: dict[str, torch.Tensor] = {}
    hf = load_hf_vision(model_dir).to(device=device, dtype=torch.bfloat16).eval()
    handles = []
    for layer in sorted(CAPTURE_LAYERS):
        name = f"block_{layer:02d}"
        handles.append(
            hf.blocks[layer].register_forward_hook(
                lambda _module, _inputs, output, name=name: hf_captures.__setitem__(
                    name, output.detach().to(device="cpu", dtype=torch.bfloat16)
                )
            )
        )
    pixels = [value for value in (batch.pixel_values, batch.pixel_values_videos) if value is not None]
    grids = [value for value in (batch.image_grid_thw, batch.video_grid_thw) if value is not None]
    with torch.inference_mode():
        output = hf(
            torch.cat(pixels).to(device=device, dtype=torch.bfloat16),
            grid_thw=torch.cat(grids).to(device=device),
            return_dict=True,
        )
        hf_captures["merger"] = output.pooler_output.detach().to(
            device="cpu", dtype=torch.bfloat16
        )
    for handle in handles:
        handle.remove()

    comparisons = {
        name: metrics(ninfer_captures[name], hf_captures[name])
        for name in ("block_00", "block_13", "block_26", "merger")
    }
    report = {
        "format": "ninfer_vision_bf16_comparison_v3",
        "scope": "diagnostic-only",
        "execution": {
            "maximum_concurrency": 1,
            "thinking": args.thinking,
            "prefix_reuse": False,
            "speculative_decode": False,
        },
        "artifact": {
            "path": str(weights_path),
            "sha256": artifact_sha256,
            "identity": artifact_identity,
        },
        "source": {
            "path": str(model_dir.resolve()),
            **source,
        },
        "input": {
            "messages_path": str(messages_path),
            "messages_sha256": messages_sha256,
            "thinking": args.thinking,
            "prepared_input": (
                None if not args.prepared_input else {
                    "path": str(Path(args.prepared_input).resolve()),
                    "sha256": sha256_file(Path(args.prepared_input)),
                    "contract": prepared_contract,
                }
            ),
        },
        "image_grid_thw": None if batch.image_grid_thw is None else batch.image_grid_thw.tolist(),
        "video_grid_thw": None if batch.video_grid_thw is None else batch.video_grid_thw.tolist(),
        "vision": {
            "images": vision_stats.images,
            "videos": vision_stats.videos,
            "raw_patches": vision_stats.raw_patches,
            "llm_tokens": vision_stats.llm_tokens,
            "attention_pairs": vision_stats.attention_pairs,
        },
        "comparisons": comparisons,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        publish_json_create_only(Path(args.output), report)
    print(rendered)


if __name__ == "__main__":
    main()
