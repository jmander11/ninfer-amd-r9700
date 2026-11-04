"""Create an artifact with LQ4 Text unchanged and source-derived BF16 Vision.

CPU-only and create-only. No Text, MTP, DFlash2, frontend or direct Vision control is
re-encoded. The 111 Vision matrices use exact original BF16 words.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec,
    TensorObject,
    TensorSpec,
    plan_objects,
)
from tools.convert.common.safetensors import ShardReader

from . import source, source_recipe
from .convert_lq4 import copied, recorded
from .vision_bf16_inventory import MATRICES, VISION_SPECS, WEIGHTS_ID


def output_specs(base: Artifact) -> tuple:
    if base.identity != ArtifactIdentity("qwen3.8-27b", "r9700-lq4"):
        raise ValueError("base must be qwen3.8-27b/r9700-lq4")
    for spec in VISION_SPECS:
        obj = base.find(spec.name)
        expected_format = "Q4G64_F16S" if spec.name in MATRICES else spec.format
        expected_layout = "r9700-q4g64-n16-k16-v1" if spec.name in MATRICES else spec.layout
        if not isinstance(obj, TensorObject) or (obj.shape, obj.format, obj.layout) != (
            spec.shape,
            expected_format,
            expected_layout,
        ):
            raise ValueError(f"base Vision signature differs: {spec.name}")
    return tuple(
        TensorSpec(o.name, o.shape, "BF16", "contiguous-le-v1")
        if o.name in MATRICES
        else TensorSpec(o.name, o.shape, o.format, o.layout)
        if isinstance(o, TensorObject)
        else ResourceSpec(o.name, o.encoding, o.bytes)
        for o in base.objects
    )


def verify(base: Path, output: Path, records: list[dict]) -> None:
    """Verify encoded digests and every preserved payload, including all Text weights."""
    import hashlib

    with Artifact(base) as before, Artifact(output) as after:
        if after.identity != ArtifactIdentity("qwen3.8-27b", WEIGHTS_ID):
            raise ValueError("incorrect output identity")
        if after.objects != plan_objects(output_specs(before)):
            raise ValueError("output directory differs from the conversion plan")
        for obj, record in zip(after.objects, records, strict=True):
            if obj.name != record["name"]:
                raise ValueError("output object order differs")
            with after.payload(obj.name) as payload:
                if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                    raise ValueError(f"output digest differs: {obj.name}")
                if obj.name not in MATRICES:
                    with before.payload(obj.name) as original:
                        for begin in range(0, len(payload), 8 << 20):
                            if bytes(original[begin : begin + (8 << 20)]) != bytes(
                                payload[begin : begin + (8 << 20)]
                            ):
                                raise ValueError(f"preserved payload differs: {obj.name}")


def convert(base: Path, model: Path, output: Path) -> dict:
    import torch

    base, model, output = base.resolve(), model.resolve(), output.absolute()
    receipt = Path(str(output) + ".conversion.json")
    if output.exists() or receipt.exists():
        raise FileExistsError("output and conversion receipt must be fresh")
    source.validate_config(json.loads((model / "config.json").read_text()))
    metadata = source_recipe.preflight_sources(model)
    if metadata.source_shard_count != 18 or metadata.source_dtype_counts != {"BF16": 1199}:
        raise ValueError("requires the complete original BF16 source")
    torch.set_num_threads(8)
    records = []
    with Artifact(base) as artifact, ShardReader(model) as reader:
        specs = output_specs(artifact)
        with ArtifactWriter(output, ArtifactIdentity("qwen3.8-27b", WEIGHTS_ID), specs) as writer:
            for obj in artifact.objects:
                record = {"name": obj.name}
                if obj.name in MATRICES:
                    tensor = source_recipe.materialize_recipe(
                        source_recipe.RECIPES_BY_NAME[obj.name], reader
                    )
                    if tensor.dtype != torch.bfloat16:
                        raise TypeError(f"{obj.name}: expected BF16 source")
                    payload = tensor.contiguous().view(torch.uint16).numpy().tobytes()
                    record["origin"] = "original-bf16-source-exact"
                    writer.write(obj.name, recorded((payload,), record))
                    print(obj.name, flush=True)
                else:
                    record["origin"] = "base-copy-exact"
                    writer.write(obj.name, recorded(copied(artifact, obj.name), record))
                records.append(record)
    verify(base, output, records)
    report = {
        "recipe": WEIGHTS_ID,
        "base": str(base),
        "source_model": str(model),
        "vision_matrices": len(MATRICES),
        "preserved_payloads_byte_exact": True,
        "artifact_bytes": output.stat().st_size,
        "objects": records,
    }
    with receipt.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = convert(args.base, args.model, args.out)
    print(f"verified {report['vision_matrices']} BF16 Vision matrices: {args.out}")


if __name__ == "__main__":
    main()
