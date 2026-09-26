"""Create-only conversion of the CB4G32 Text recipe on the selective-cap DFlash2 base.

Every Text-layer projection the base stores as Q4G64 is re-encoded from the original BF16
checkpoint as CB4G32_F32S (`cb4_codec`), MLP gate/up rows in the interleaved SiLU-pair order
(`cb4_codec.interleave_gate_up`); every other object (FP8 protections, embeddings, head,
MTP, DFlash2 companion, Vision, resources) is copied byte-exact from the base artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.artifact.container import (
    Artifact, ArtifactIdentity, ArtifactWriter, TensorObject,
    TensorSpec as StoredTensor, ResourceSpec as StoredResource,
)

BASE_WEIGHTS_ID = "r9700-q4-fp8-selective-cap-n16k16-dflash2-q4-eval"
WEIGHTS_ID = "r9700-cb4-fp8-selective-cap-dflash2-q4-eval"
CB4 = "CB4G32_F32S"
CB4_LAYOUT = "r9700-cb4g32-n16k64-v1"


def selected(obj) -> bool:
    return (isinstance(obj, TensorObject) and obj.name.startswith("text/layers/")
            and obj.format == "Q4G64_F16S")


def recorded(chunks, record):
    digest = hashlib.sha256()
    for part in chunks:
        digest.update(part)
        yield part
    record["sha256"] = digest.hexdigest()


def copied(artifact, name):
    with artifact.payload(name) as payload:
        for begin in range(0, len(payload), 8 << 20):
            yield bytes(payload[begin:begin + (8 << 20)])


def convert(args) -> None:
    import torch
    from tools.convert.common.safetensors import ShardReader
    from . import cb4_codec, source, source_recipe

    receipt = Path(str(args.out) + ".conversion.json")
    if args.out.exists() or receipt.exists():
        raise FileExistsError(args.out)
    source.validate_config(json.loads((args.model / "config.json").read_text()))
    metadata = source_recipe.preflight_sources(args.model)
    if metadata.source_shard_count != 18 or metadata.source_dtype_counts != {"BF16": 1199}:
        raise ValueError("requires the complete original BF16 source")
    torch.set_num_threads(8)
    report = dict(recipe=WEIGHTS_ID, base=str(args.base.resolve()),
                  source_model=str(args.model.resolve()), objects=[])
    with Artifact(args.base) as base, ShardReader(args.model) as reader:
        if base.identity.weights_id != BASE_WEIGHTS_ID:
            raise ValueError(f"base must be {BASE_WEIGHTS_ID}, got {base.identity.weights_id}")
        stored = tuple(
            StoredTensor(o.name, o.shape, CB4, CB4_LAYOUT) if selected(o)
            else StoredTensor(o.name, o.shape, o.format, o.layout) if isinstance(o, TensorObject)
            else StoredResource(o.name, o.encoding, o.bytes)
            for o in base.objects)
        identity = ArtifactIdentity(base.identity.model_id, WEIGHTS_ID)
        with ArtifactWriter(args.out, identity, stored) as writer:
            for obj in base.objects:
                record = dict(name=obj.name)
                if selected(obj):
                    tensor = source_recipe.materialize_recipe(
                        source_recipe.RECIPES_BY_NAME[obj.name], reader)
                    if obj.name.endswith("/mlp/gate_up"):
                        tensor = cb4_codec.interleave_gate_up(tensor)
                    record["origin"] = "original-bf16-source-cb4"
                    writer.write(obj.name, recorded(
                        cb4_codec.encode_chunks(tensor, device=args.device), record))
                    del tensor
                    print(obj.name, flush=True)
                else:
                    record["origin"] = "base-copy-exact"
                    writer.write(obj.name, recorded(copied(base, obj.name), record))
                report["objects"].append(record)
    report["artifact"] = dict(path=str(args.out.resolve()), bytes=args.out.stat().st_size)
    with receipt.open("x") as handle:
        json.dump(report, handle, indent=1)
        handle.write("\n")
    validate(args.out)
    print(json.dumps(report["artifact"]), flush=True)


def validate(path: Path) -> None:
    report = json.loads(Path(str(path) + ".conversion.json").read_text())
    with Artifact(path) as artifact:
        if artifact.identity.weights_id != WEIGHTS_ID:
            raise ValueError("incorrect output identity")
        names = [o.name for o in artifact.objects]
        if names != [row["name"] for row in report["objects"]]:
            raise ValueError("receipt object order differs")
        for row in report["objects"]:
            digest = hashlib.sha256()
            for part in copied(artifact, row["name"]):
                digest.update(part)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError(f"payload differs: {row['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cpu", help="torch device for the codebook search")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate:
        validate(args.validate)
        print("PASS: identity and every payload digest")
        return
    for key in ("base", "model", "out"):
        if getattr(args, key) is None:
            parser.error(f"--{key} is required")
    convert(args)


if __name__ == "__main__":
    main()
