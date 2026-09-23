"""Create-only, head-only lossless layout revision of the selective DFlash artifact."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, plan_objects
from tools.artifact.layouts import transcode_w8_n16k16
from . import dflash2_q4_inventory as inventory, selective_protected_inventory
from .convert_dflash2_q4 import _artifact_spec, _chunks, _validate_base_objects

HEAD = "text/output_head"
LAYOUT = "r9700-w8g32-n16-k16-v1"
IDENTITY = ArtifactIdentity(inventory.MODEL_ID, inventory.SELECTIVE_WEIGHTS_ID)


def inspect_source(artifact: Artifact):
    if artifact.identity != IDENTITY:
        raise ValueError("head transcode requires the selective canonical-Q4 DFlash recipe")
    # This is the explicit offline source contract, not a runtime fallback.
    _validate_base_objects(artifact.objects,
                          selective_protected_inventory.OBJECT_SPECS + inventory.TENSOR_SPECS)
    return tuple(replace(_artifact_spec(obj), layout=LAYOUT)
                 if obj.name == HEAD else _artifact_spec(obj) for obj in artifact.objects)


def verify(source: Path, output: Path) -> dict:
    """Compare every unchanged byte and every inverse-permuted head byte."""
    with Artifact.open(source) as before, Artifact.open(output) as after:
        specs = inspect_source(before)
        if after.identity != before.identity or after.objects != plan_objects(specs):
            raise ValueError("transcoded artifact directory differs from the exact output plan")
        copied = 0
        for obj in before.objects:
            original = before.payload(obj)
            encoded = after.payload(obj.name)
            try:
                if obj.name == HEAD:
                    offset = 0
                    for block in transcode_w8_n16k16(encoded, obj.shape, inverse=True):
                        if original[offset:offset + len(block)] != block:
                            raise ValueError("output head codes/scales were not preserved exactly")
                        offset += len(block)
                    if offset != len(original):
                        raise ValueError("inverse head transform extent mismatch")
                else:
                    for begin in range(0, len(original), 8 * 1024 * 1024):
                        if original[begin:begin + 8 * 1024 * 1024] != encoded[begin:begin + 8 * 1024 * 1024]:
                            raise ValueError(f"non-head payload changed: {obj.name}")
                    copied += 1
            finally:
                original.release()
                encoded.release()
        return {"unchanged_payloads": copied, "head_inverse_byte_exact": True,
                "numeric_format": "W8G32_F16S", "layout": LAYOUT,
                "duplicate_resident_head": False, "requantized": False}


def transcode(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.absolute()
    receipt = Path(str(output) + ".head-layout.json")
    if source == output or output.exists() or output.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise FileExistsError("source and create-only output/receipt must be distinct and fresh")
    before_stat = source.stat()
    with Artifact.open(source) as artifact:
        specs = inspect_source(artifact)
        with ArtifactWriter(output, artifact.identity, specs) as writer:
            for obj in artifact.objects:
                view = artifact.payload(obj)
                try:
                    writer.write(obj.name, transcode_w8_n16k16(view, obj.shape)
                                 if obj.name == HEAD else _chunks(view))
                finally:
                    view.release()
    result = verify(source, output)
    after_stat = source.stat()
    if (before_stat.st_dev, before_stat.st_ino, before_stat.st_size, before_stat.st_mtime_ns) != (
            after_stat.st_dev, after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns):
        raise ValueError("source changed during head transformation")
    def digest(path):
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()
    report = {"schema": "ninfer.r9700.lossless-w8-head.v1",
              "identity": {"model_id": IDENTITY.model_id, "weights_id": IDENTITY.weights_id},
              "source": {"path": str(source), "bytes": before_stat.st_size, "sha256": digest(source)},
              "artifact": {"path": str(output), "bytes": output.stat().st_size, "sha256": digest(output)},
              "verification": result}
    with receipt.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        if args.output is not None:
            parser.error("--preflight-only does not create an output")
        with Artifact.open(args.source) as artifact:
            specs = inspect_source(artifact)
            print(json.dumps({"objects": len(specs), "changed_object": HEAD,
                              "layout": LAYOUT, "recipe_identity_unchanged": True,
                              "payload_bytes_unchanged": True}, indent=2))
    else:
        if args.output is None:
            parser.error("--output is required unless --preflight-only")
        print(json.dumps(transcode(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
