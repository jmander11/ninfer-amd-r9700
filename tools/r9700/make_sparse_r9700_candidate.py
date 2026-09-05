"""Create a sparse metadata-complete R9700 candidate artifact for binder qualification.

The file has the exact 1,124-object candidate directory and logical payload extents, but only the
directory and draft-token IDs consume blocks. It is not an inference artifact and must never be
used for numerical or performance evidence.
"""

from __future__ import annotations

import argparse
import json
from math import prod
from pathlib import Path
import struct
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.convert.qwen3_8_27b_r9700 import (
    dflash2_q4_inventory,
    fp8_hybrid_inventory,
    inventory,
    q4_inventory,
    q4_w8_inventory,
    q4_w8_mse_inventory,
    w8_bf16_attention_qk_inventory,
    w8_bf16_attention_vo_inventory,
    w8_bf16_embedding_inventory,
    w8_bf16_gdn_qk_inventory,
    w8_mse_inventory,
)


MAGIC = b"NINFER\x00\x02"
PREFIX = struct.Struct("<8sQ")
PREFIX_BYTES = PREFIX.size
PAYLOAD_ALIGNMENT = 4096


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def encoded_size(layout: str, numeric_format: str, shape: tuple[int, ...]) -> int:
    if layout == "contiguous-le-v1":
        word_bytes = {"BF16": 2, "FP32": 4, "I32": 4}[numeric_format]
        return prod(shape) * word_bytes
    if layout == "row-scaled-k128-v1" and numeric_format == "F8E4M3_ROW_F32S" and len(shape) == 2:
        rows, columns = shape
        code_bytes = rows * align_up(columns, 128)
        return align_up(code_bytes, 256) + rows * 4
    if layout != "row-split-k128-v1" or len(shape) != 2:
        raise ValueError(f"unsupported sparse fixture layout: {layout}")
    group, bits = {
        "W8G32_F16S": (32, 8),
        "Q4G64_F16S": (64, 4),
        "Q6G64_F16S": (64, 6),
    }[numeric_format]
    rows, columns = shape
    groups_per_row = align_up(columns, 128) // group
    base_group_bytes = group if bits == 8 else group // 2
    high_group_bytes = 0 if bits in (4, 8) else group * (bits - 4) // 8
    base_bytes = rows * groups_per_row * base_group_bytes
    high_bytes = rows * groups_per_row * high_group_bytes
    scale_bytes = rows * groups_per_row * 2
    high_offset = align_up(base_bytes, 256)
    scale_offset = high_offset + align_up(high_bytes, 256)
    return scale_offset + scale_bytes


def build_objects(specs: tuple[object, ...], *, wrong_token_format: bool) -> tuple[dict[str, object], ...]:
    objects: list[dict[str, object]] = []
    cursor = 0
    for spec in specs:
        if hasattr(spec, "format"):
            numeric_format = spec.format
            if wrong_token_format and spec.name == "text/token_embedding":
                numeric_format = "Q6G64_F16S"
            size = encoded_size(spec.layout, numeric_format, spec.shape)
            offset = align_up(cursor, 256)
            obj = {
                "name": spec.name,
                "kind": "tensor",
                "shape": list(spec.shape),
                "format": numeric_format,
                "layout": spec.layout,
                "offset": offset,
                "bytes": size,
            }
        else:
            offset = cursor
            obj = {
                "name": spec.name,
                "kind": "resource",
                "encoding": "raw-bytes-v1",
                "offset": offset,
                "bytes": 1,
            }
        objects.append(obj)
        cursor = offset + size if hasattr(spec, "format") else offset + 1
    return tuple(objects)


def write_sparse(path: Path, *, profile: str, wrong_token_format: bool) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sparse qualifier artifact: {path}")
    model_id, weights_id, object_specs = {
        "w8": (inventory.MODEL_ID, inventory.WEIGHTS_ID, inventory.OBJECT_SPECS),
        "w8-mse": (w8_mse_inventory.MODEL_ID, w8_mse_inventory.WEIGHTS_ID,
                    w8_mse_inventory.OBJECT_SPECS),
        "w8-bf16-embedding": (
            w8_bf16_embedding_inventory.MODEL_ID,
            w8_bf16_embedding_inventory.WEIGHTS_ID,
            w8_bf16_embedding_inventory.OBJECT_SPECS,
        ),
        "w8-bf16-attention-qk": (
            w8_bf16_attention_qk_inventory.MODEL_ID,
            w8_bf16_attention_qk_inventory.WEIGHTS_ID,
            w8_bf16_attention_qk_inventory.OBJECT_SPECS,
        ),
        "w8-bf16-attention-vo": (
            w8_bf16_attention_vo_inventory.MODEL_ID,
            w8_bf16_attention_vo_inventory.WEIGHTS_ID,
            w8_bf16_attention_vo_inventory.OBJECT_SPECS,
        ),
        "w8-bf16-gdn-qk": (
            w8_bf16_gdn_qk_inventory.MODEL_ID,
            w8_bf16_gdn_qk_inventory.WEIGHTS_ID,
            w8_bf16_gdn_qk_inventory.OBJECT_SPECS,
        ),
        "q4": (q4_inventory.MODEL_ID, q4_inventory.WEIGHTS_ID,
               q4_inventory.OBJECT_SPECS),
        "fp8-q4-hybrid": (
            fp8_hybrid_inventory.MODEL_ID,
            fp8_hybrid_inventory.WEIGHTS_ID,
            fp8_hybrid_inventory.OBJECT_SPECS,
        ),
        "q4-w8": (q4_w8_inventory.MODEL_ID, q4_w8_inventory.WEIGHTS_ID,
                  q4_w8_inventory.OBJECT_SPECS),
        "q4-w8-mse": (
            q4_w8_mse_inventory.MODEL_ID,
            q4_w8_mse_inventory.WEIGHTS_ID,
            q4_w8_mse_inventory.OBJECT_SPECS,
        ),
        "q4-dflash2-q4": (
            dflash2_q4_inventory.MODEL_ID,
            dflash2_q4_inventory.ALL_Q4_WEIGHTS_ID,
            dflash2_q4_inventory.ALL_Q4_OBJECT_SPECS,
        ),
        "q4-w8-mse-dflash2-q4": (
            dflash2_q4_inventory.MODEL_ID,
            dflash2_q4_inventory.MIXED_WEIGHTS_ID,
            dflash2_q4_inventory.MIXED_OBJECT_SPECS,
        ),
    }[profile]
    objects = build_objects(object_specs, wrong_token_format=wrong_token_format)
    directory = json.dumps(
        {
            "identity": {
                "model_id": model_id,
                "weights_id": weights_id,
            },
            "objects": objects,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    payload_offset = align_up(PREFIX_BYTES + len(directory), PAYLOAD_ALIGNMENT)
    logical_payload_bytes = int(objects[-1]["offset"]) + int(objects[-1]["bytes"])
    draft_ids = next(obj for obj in objects if obj["name"] == "text/draft_head_token_ids")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(PREFIX.pack(MAGIC, len(directory)))
        output.write(directory)
        output.write(b"\x00" * (payload_offset - PREFIX_BYTES - len(directory)))
        output.truncate(payload_offset + logical_payload_bytes)
        output.seek(payload_offset + int(draft_ids["offset"]))
        for token_id in range(131072):
            output.write(struct.pack("<I", token_id))

    print(
        f"sparse_candidate path={path} objects={len(objects)} "
        f"logical_bytes={payload_offset + logical_payload_bytes} "
        f"profile={profile} wrong_token_format={wrong_token_format}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--wrong-token-format", action="store_true")
    parser.add_argument(
        "--profile",
        choices=(
            "w8",
            "w8-mse",
            "w8-bf16-embedding",
            "w8-bf16-attention-qk",
            "w8-bf16-attention-vo",
            "w8-bf16-gdn-qk",
            "q4",
            "fp8-q4-hybrid",
            "q4-w8",
            "q4-w8-mse",
            "q4-dflash2-q4",
            "q4-w8-mse-dflash2-q4",
        ),
        default="w8",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace only the explicitly named output (for repeatable build-tree qualification)",
    )
    args = parser.parse_args()
    if args.replace:
        args.out.unlink(missing_ok=True)
    write_sparse(args.out, profile=args.profile, wrong_token_format=args.wrong_token_format)


if __name__ == "__main__":
    main()
