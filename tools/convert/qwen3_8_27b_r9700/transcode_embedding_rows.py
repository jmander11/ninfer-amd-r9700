"""Create-only, embedding-only lossless layout revision of a Qwen3.8-27B R9700 artifact.

Rewrites the Q4G64 `text/token_embedding` from `r9700-q4g64-n16-k16-v1` to `row-split-k128-v1`,
the layout `convert_fp8lut4` now writes: the embedding is only row-gathered from pinned host
memory, and a row-split row is one contiguous run of codes and of scales. Codes and FP16 scale
words are permuted, never re-encoded; every other object is copied byte-exact. When the source
has a `.conversion.json` receipt, the output receives the same receipt with the embedding's
origin and digest updated, so `convert_fp8lut4 --validate` holds for it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.artifact.container import (
    Artifact,
    ArtifactWriter,
    ResourceSpec,
    TensorObject,
    TensorSpec,
    plan_objects,
)
from tools.artifact.layouts import transcode_q4_n16k16
from .convert_fp8lut4 import EMBEDDING, ROW_SPLIT, copied, recorded, row_split

N16K16 = "r9700-q4g64-n16-k16-v1"
CHUNK = 8 << 20


def output_specs(artifact: Artifact) -> tuple:
    specs = []
    for obj in artifact.objects:
        if not isinstance(obj, TensorObject):
            specs.append(ResourceSpec(obj.name, obj.encoding, obj.bytes))
        elif obj.name == EMBEDDING:
            if obj.format != "Q4G64_F16S" or obj.layout != N16K16:
                raise ValueError(f"{EMBEDDING} is not Q4G64 {N16K16}")
            specs.append(TensorSpec(obj.name, obj.shape, obj.format, ROW_SPLIT))
        else:
            specs.append(TensorSpec(obj.name, obj.shape, obj.format, obj.layout))
    return tuple(specs)


def verify(source: Path, output: Path) -> None:
    """Every unchanged payload is byte-identical; the forward permutation of the output
    embedding reproduces the source embedding exactly."""
    with Artifact(source) as before, Artifact(output) as after:
        if after.identity != before.identity or after.objects != plan_objects(output_specs(before)):
            raise ValueError("output directory differs from the exact transcode plan")
        for obj in before.objects:
            with before.payload(obj.name) as original, after.payload(obj.name) as encoded:
                if obj.name == EMBEDDING:
                    offset = 0
                    for block in transcode_q4_n16k16(encoded, obj.shape):
                        if original[offset : offset + len(block)] != block:
                            raise ValueError("embedding codes/scales were not preserved exactly")
                        offset += len(block)
                    if offset != len(original):
                        raise ValueError("embedding transform extent mismatch")
                else:
                    for begin in range(0, len(original), CHUNK):
                        if original[begin : begin + CHUNK] != encoded[begin : begin + CHUNK]:
                            raise ValueError(f"unchanged payload differs: {obj.name}")


def transcode(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.absolute()
    source_receipt = Path(str(source) + ".conversion.json")
    output_receipt = Path(str(output) + ".conversion.json")
    if source == output or output.exists() or output_receipt.exists():
        raise FileExistsError("output and its receipt must be fresh and distinct from the source")
    embedding_record = {"name": EMBEDDING, "origin": "base-transcode-row-split"}
    with Artifact(source) as artifact:
        specs = output_specs(artifact)
        with ArtifactWriter(output, artifact.identity, specs) as writer:
            for obj in artifact.objects:
                if obj.name == EMBEDDING:
                    writer.write(obj.name, recorded(row_split(artifact, obj), embedding_record))
                else:
                    writer.write(obj.name, copied(artifact, obj.name))
    verify(source, output)
    if source_receipt.exists():
        receipt = json.loads(source_receipt.read_text())
        rows = [row for row in receipt["objects"] if row["name"] == EMBEDDING]
        if len(rows) != 1:
            raise ValueError("source receipt has no single embedding record")
        rows[0].update(embedding_record)
        receipt["artifact"] = dict(path=str(output), bytes=output.stat().st_size)
        with output_receipt.open("x") as handle:
            json.dump(receipt, handle, indent=1)
            handle.write("\n")
    return {
        "artifact": str(output),
        "bytes": output.stat().st_size,
        "embedding_layout": ROW_SPLIT,
        "embedding_sha256": embedding_record["sha256"],
        "unchanged_payloads_byte_exact": True,
        "embedding_inverse_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(transcode(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
