"""Create-only conversion of the FP8LUT4 Text recipe on the selective-cap DFlash2 base.

Every Text-layer projection the base stores as Q4G64 is re-encoded from the original BF16
checkpoint as FP8LUT4 (`fp8lut4_codec`), MLP gate/up rows in the interleaved SiLU-pair order
(`fp8lut4_codec.interleave_gate_up`), with GPTQ error-compensated rounding against the input
second moments of the calibration sequences (`calibration.InputMoments`, evaluated layer-major
through the BF16 reference in lock-step with the object order). With `--head`, the output head
is re-encoded the same way against the final-norm moments (evaluation identity
`r9700-fp8lut4-head-eval`). Every other object (FP8 protections, embeddings, draft head, MTP,
DFlash2 companion, Vision, resources) is copied byte-exact from the base artifact.
`--reuse-layers` copies the Text-layer FP8LUT4 objects byte-exact from an existing
`r9700-fp8lut4` artifact converted with the same calibration and damping.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path

from tools.artifact.container import (
    Artifact, ArtifactIdentity, ArtifactWriter, TensorObject,
    TensorSpec as StoredTensor, ResourceSpec as StoredResource,
)

# Mean-diagonal GPTQ damping per calibrated input, selected on held-out calibration sequences:
# the 17408-wide MLP down input overfits at low damping.
DAMPING = {"mlp/down": 0.3}
DEFAULT_DAMPING = 0.1
HEAD = "text/output_head"
HEAD_DAMPING = 0.1

BASE_WEIGHTS_ID = "r9700-q4-fp8-selective-cap-n16k16-dflash2-q4-eval"
WEIGHTS_ID = "r9700-fp8lut4"
HEAD_WEIGHTS_ID = "r9700-fp8lut4-head-eval"
FP8LUT4 = "FP8LUT4"
FP8LUT4_LAYOUT = "r9700-fp8lut4-n16k64-v1"


def selected(obj, head: bool) -> bool:
    return (isinstance(obj, TensorObject) and obj.format == "Q4G64_F16S" and
            (obj.name.startswith("text/layers/") or (head and obj.name == HEAD)))


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
    from . import fp8lut4_codec, source, source_recipe
    from tools.reference.qwen3_8_27b_bf16.protocol import LAYERS
    from .calibration import InputMoments

    receipt = Path(str(args.out) + ".conversion.json")
    if args.out.exists() or receipt.exists():
        raise FileExistsError(args.out)
    source.validate_config(json.loads((args.model / "config.json").read_text()))
    metadata = source_recipe.preflight_sources(args.model)
    if metadata.source_shard_count != 18 or metadata.source_dtype_counts != {"BF16": 1199}:
        raise ValueError("requires the complete original BF16 source")
    torch.set_num_threads(8)
    device = torch.device(args.device)
    moments = InputMoments(args.model, args.calibration, device)
    weights_id = HEAD_WEIGHTS_ID if args.head else WEIGHTS_ID
    calibration_sha = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    reuse = None
    if args.reuse_layers is not None:
        prior = json.loads(Path(str(args.reuse_layers) + ".conversion.json").read_text())
        if (prior["recipe"] != WEIGHTS_ID or prior["calibration"]["sha256"] != calibration_sha or
                prior["calibration"]["damping"] != dict(DAMPING, default=DEFAULT_DAMPING) or
                prior["base"] != str(args.base.resolve()) or
                prior["source_model"] != str(args.model.resolve())):
            raise ValueError("--reuse-layers needs an r9700-fp8lut4 artifact of this calibration")
        prior_digests = {row["name"]: row["sha256"] for row in prior["objects"]}
        reuse = Artifact(args.reuse_layers)
        if reuse.identity.weights_id != WEIGHTS_ID:
            reuse.close()
            raise ValueError("--reuse-layers artifact identity is not r9700-fp8lut4")
    report = dict(recipe=weights_id, base=str(args.base.resolve()),
                  source_model=str(args.model.resolve()),
                  calibration=dict(ids=str(args.calibration.resolve()),
                                   sha256=calibration_sha, tokens=moments.tokens,
                                   rounding="gptq-block128",
                                   damping=dict(DAMPING, default=DEFAULT_DAMPING,
                                                **({"output_head": HEAD_DAMPING} if args.head else {}))),
                  objects=[])
    with Artifact(args.base) as base, ShardReader(args.model) as reader, \
            (reuse if reuse is not None else contextlib.nullcontext()):
        if base.identity.weights_id != BASE_WEIGHTS_ID:
            raise ValueError(f"base must be {BASE_WEIGHTS_ID}, got {base.identity.weights_id}")
        stored = tuple(
            StoredTensor(o.name, o.shape, FP8LUT4, FP8LUT4_LAYOUT) if selected(o, args.head)
            else StoredTensor(o.name, o.shape, o.format, o.layout) if isinstance(o, TensorObject)
            else StoredResource(o.name, o.encoding, o.bytes)
            for o in base.objects)
        identity = ArtifactIdentity(base.identity.model_id, weights_id)
        layer_moments: dict[str, torch.Tensor] = {}
        with ArtifactWriter(args.out, identity, stored) as writer:
            for obj in base.objects:
                record = dict(name=obj.name)
                if selected(obj, args.head) and obj.name == HEAD:
                    while moments.next_layer < LAYERS:
                        moments.layer(moments.next_layer)
                    calibration = fp8lut4_codec.Calibration(moments.final(), HEAD_DAMPING)
                    tensor = source_recipe.materialize_recipe(
                        source_recipe.RECIPES_BY_NAME[obj.name], reader)
                    record["origin"] = "original-bf16-source-fp8lut4-gptq"
                    writer.write(obj.name, recorded(fp8lut4_codec.encode_chunks(
                        tensor, device=device, rows_per_chunk=16384,
                        calibration=calibration), record))
                    del tensor, calibration
                    print(obj.name, flush=True)
                elif selected(obj, args.head) and reuse is not None:
                    layer = int(obj.name.split("/")[2])
                    while moments.next_layer <= layer:
                        moments.layer(moments.next_layer)
                    record["origin"] = "reused-fp8lut4-gptq"
                    if reuse.find(obj.name).format != FP8LUT4:
                        raise ValueError(f"reused {obj.name} is not FP8LUT4")
                    writer.write(obj.name, recorded(copied(reuse, obj.name), record))
                    if record["sha256"] != prior_digests.get(obj.name):
                        raise ValueError(f"reused {obj.name} differs from its receipt")
                elif selected(obj, args.head):
                    layer = int(obj.name.split("/")[2])
                    while moments.next_layer <= layer:
                        layer_moments = moments.layer(moments.next_layer)
                    if moments.next_layer != layer + 1:
                        raise ValueError(f"{obj.name} is out of layer-major order")
                    tensor = source_recipe.materialize_recipe(
                        source_recipe.RECIPES_BY_NAME[obj.name], reader)
                    if obj.name.endswith("/mlp/gate_up"):
                        tensor = fp8lut4_codec.interleave_gate_up(tensor)
                    role = "/".join(obj.name.split("/")[3:])
                    calibration = None if role.startswith("attention/") else fp8lut4_codec.Calibration(
                        layer_moments[role], DAMPING.get(role, DEFAULT_DAMPING))
                    record["origin"] = ("original-bf16-source-fp8lut4" if calibration is None
                                        else "original-bf16-source-fp8lut4-gptq")
                    writer.write(obj.name, recorded(fp8lut4_codec.encode_chunks(
                        tensor, device=device, rows_per_chunk=tensor.shape[0],
                        calibration=calibration), record))
                    del tensor, calibration
                    print(obj.name, flush=True)
                else:
                    record["origin"] = "base-copy-exact"
                    writer.write(obj.name, recorded(copied(base, obj.name), record))
                report["objects"].append(record)
    if reuse is not None:
        report["reused_layers"] = str(args.reuse_layers.resolve())
    report["artifact"] = dict(path=str(args.out.resolve()), bytes=args.out.stat().st_size)
    with receipt.open("x") as handle:
        json.dump(report, handle, indent=1)
        handle.write("\n")
    validate(args.out)
    print(json.dumps(report["artifact"]), flush=True)


def validate(path: Path) -> None:
    report = json.loads(Path(str(path) + ".conversion.json").read_text())
    with Artifact(path) as artifact:
        if artifact.identity.weights_id != report["recipe"]:
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
    parser.add_argument("--calibration", type=Path,
                        help="calibration token ids, one equal-length sequence per line")
    parser.add_argument("--device", default="cuda:0",
                        help="torch device for the BF16 calibration pass and GPTQ")
    parser.add_argument("--head", action="store_true",
                        help="also GPTQ-encode the output head (evaluation identity)")
    parser.add_argument("--reuse-layers", type=Path,
                        help="copy the Text-layer FP8LUT4 objects from this r9700-fp8lut4 artifact")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate:
        validate(args.validate)
        print("PASS: identity and every payload digest")
        return
    for key in ("base", "model", "out", "calibration"):
        if getattr(args, key) is None:
            parser.error(f"--{key} is required")
    convert(args)


if __name__ == "__main__":
    main()
