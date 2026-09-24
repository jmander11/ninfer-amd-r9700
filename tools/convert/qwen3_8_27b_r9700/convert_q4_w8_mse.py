"""Build the same-format source-MSE-refined Q4G64/W8G32 evaluator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.artifact.layouts import encode_direct
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3.common import conversion as family_conversion
from tools.convert.qwen3.common.inventory import Q4, W8
from tools.reference.qwen3_8_27b_bf16 import protocol as bf16_protocol

from . import (
    build_draft_ranking,
    draft_head,
    mse_quantize,
    q4_w8_mse_inventory,
    resources,
    source,
    source_recipe,
)


@dataclass(frozen=True, slots=True)
class Q4W8MseConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: source_recipe.SourcePreflight
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    draft_ranking: build_draft_ranking.RankingProvenance
    object_plan: family_conversion.ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    with path.open("rb") as source_file:
        return hashlib.file_digest(source_file, "sha256").hexdigest()


def _checkpoint_receipt(model: Path) -> dict[str, object]:
    weight_map = bf16_protocol.validate_checkpoint_files(model)
    config = model / "config.json"
    index = model / "model.safetensors.index.json"
    index_value = family_conversion.load_json(index)
    total_tensor_bytes = index_value.get("metadata", {}).get("total_size")
    if (
        isinstance(total_tensor_bytes, bool)
        or not isinstance(total_tensor_bytes, (int, float))
        or not float(total_tensor_bytes).is_integer()
        or int(total_tensor_bytes) != bf16_protocol.SOURCE_TOTAL_BYTES
    ):
        raise ValueError("BF16 source index has invalid exact tensor byte count")
    shard_names = sorted(set(weight_map.values()))
    shards = [
        {"name": name, "bytes": (model / name).stat().st_size} for name in shard_names
    ]
    return {
        "config": {"path": str(config.resolve()), "bytes": config.stat().st_size,
                   "sha256": _sha256(config)},
        "index": {"path": str(index.resolve()), "bytes": index.stat().st_size,
                  "sha256": _sha256(index)},
        "indexed_tensor_count": len(weight_map),
        "index_total_tensor_bytes": int(total_tensor_bytes),
        "shards": shards,
        "shard_total_file_bytes": sum(row["bytes"] for row in shards),
        "shard_payload_sha256": None,
        "shard_payload_hash_policy": (
            "not computed: exact names and nonempty regular files are the target-owned contract"
        ),
    }


def preflight_conversion(
    model_dir: str | Path, draft_ranking: str | Path
) -> Q4W8MseConversionPreflight:
    model = Path(model_dir)
    q4_w8_mse_inventory.validate_inventory()
    config_summary = source.validate_config(
        family_conversion.load_json(model / "config.json")
    )
    source_preflight = source_recipe.preflight_sources(model)
    frontend_resources = resources.load_resources(model)
    resource_map = {resource.name: resource.data for resource in frontend_resources}
    object_plan = family_conversion.build_object_plan(
        q4_w8_mse_inventory.OBJECT_SPECS, resource_map
    )
    ranking = build_draft_ranking.validate_ranking_provenance(draft_ranking)
    draft = draft_head.compute_shortlist(ranking.ranking_path, model)
    return Q4W8MseConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source_preflight,
        resources=frontend_resources,
        draft=draft,
        draft_ranking=ranking,
        object_plan=object_plan,
    )


def preflight_summary(preflight: Q4W8MseConversionPreflight) -> dict[str, object]:
    """Render the exact mixed source-to-artifact write plan without device setup."""

    objects = preflight.object_plan.objects
    object_plan_bytes = json.dumps(
        [obj.to_json() for obj in objects], sort_keys=True, separators=(",", ":")
    ).encode()
    source_summary = {
        "recipes": preflight.source.recipe_count,
        "tensors": preflight.source.source_tensor_count,
        "shards": preflight.source.source_shard_count,
        "dtypes": preflight.source.source_dtype_counts,
    }
    if source_summary["tensors"] != 1199 or source_summary["shards"] != 18 or source_summary[
        "dtypes"
    ] != {"BF16": 1199}:
        raise ValueError("source is not the exact 1,199-BF16-tensor/18-shard checkpoint")
    statistics = family_conversion.object_statistics(objects)
    if statistics["count"] != len(q4_w8_mse_inventory.OBJECT_SPECS):
        raise ValueError("mixed source-MSE object plan is incomplete")
    checkpoint = _checkpoint_receipt(preflight.model_dir)
    resource_receipt = [
        {"name": resource.name, "bytes": len(resource.data),
         "sha256": hashlib.sha256(resource.data).hexdigest()}
        for resource in preflight.resources
    ]
    if [row["name"] for row in resource_receipt] != list(resources.OFFICIAL_RESOURCE_SHA256):
        raise ValueError("frontend resource receipt is not in canonical order")
    return {
        "artifact_type": "ninfer_qwen3_8_27b_r9700_q4_w8_mse_conversion_preflight",
        "schema_version": 1,
        "identity": {
            "model_id": q4_w8_mse_inventory.MODEL_ID,
            "weights_id": q4_w8_mse_inventory.WEIGHTS_ID,
        },
        "target_key": q4_w8_mse_inventory.TARGET_KEY,
        "recipe_id": q4_w8_mse_inventory.RECIPE_ID,
        "model_path": str(preflight.model_dir.resolve()),
        "config_summary": preflight.config_summary,
        "source": source_summary,
        "checkpoint": checkpoint,
        "frontend_resources": resource_receipt,
        "ranking": {
            "path": str(preflight.draft_ranking.ranking_path.resolve()),
            "sha256": preflight.draft_ranking.ranking_sha256,
            "provenance_path": str(preflight.draft_ranking.sidecar_path.resolve()),
            "provenance_sha256": preflight.draft_ranking.sidecar_sha256,
            "tokens": preflight.draft_ranking.total_tokens,
            "distinct_token_ids": preflight.draft_ranking.distinct_token_ids,
            "selected_rows": preflight.draft.n,
        },
        "objects": statistics,
        "object_plan_sha256": hashlib.sha256(object_plan_bytes).hexdigest(),
        "format_counts": q4_w8_mse_inventory.FORMAT_COUNTS,
        "format_encoded_bytes": q4_w8_mse_inventory.FORMAT_ENCODED_BYTES,
        "tensor_encoded_bytes": q4_w8_mse_inventory.TENSOR_ENCODED_BYTES,
        "device_arena_bytes": q4_w8_mse_inventory.DEVICE_ARENA_BYTES,
    }


def _encode_tensor(
    tensor: torch.Tensor,
    spec: q4_w8_mse_inventory.TensorSpec,
    device: torch.device,
) -> bytes:
    if spec.format in q4_w8_mse_inventory.DIRECT_FORMATS:
        return encode_direct(tensor, spec.format)
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{spec.name}: Q4/W8 source must be BF16, got {tensor.dtype}")
    if spec.format == Q4:
        return mse_quantize.quantize_and_encode_q4g64_mse(tensor, device=device)
    if spec.format == W8:
        return mse_quantize.quantize_and_encode_mse(tensor, device=device)
    raise ValueError(f"{spec.name}: source-MSE evaluator requires Q4G64 or W8G32")


def _candidate_metadata() -> dict[str, object]:
    return {
        "status": "registered-evaluation-only",
        "weight_recipe_selected": False,
        "comparison_role": (
            "same-format control for deterministic source-only Q4G64 and W8G32 "
            "group-scale refinement"
        ),
        "format_counts": q4_w8_mse_inventory.FORMAT_COUNTS,
        "format_encoded_bytes": q4_w8_mse_inventory.FORMAT_ENCODED_BYTES,
        "tensor_encoded_bytes": q4_w8_mse_inventory.TENSOR_ENCODED_BYTES,
        "device_arena_bytes": q4_w8_mse_inventory.DEVICE_ARENA_BYTES,
        "layout_reference_weights_id": "r9700-q4-w8-n16k16-eval",
        "same_format_and_byte_plan_as_reference": True,
        "scale_objective": {
            "represented_input": "original source BF16 weight values",
            "q4g64_baseline": "FP16(maximum absolute group value divided by 7)",
            "w8g32_baseline": "FP16(maximum absolute group value divided by 127)",
            "refinement": "eight alternating signed-code and least-squares scale steps",
            "selection": "minimum decoded-weight SSE among trajectory; earliest tie",
            "stored_scale": "FP16 before code selection",
            "calibration_inputs": "none",
            "excluded_inputs": [
                "calibration activations",
                "draft-ranking frequencies or source corpus tokens",
                "PPL NLL or argmax sidecars",
                "GPU measurements",
            ],
        },
        "reason": (
            "measure scale-objective quality at the identical Q4G64/W8G32 persistent "
            "format, layout, byte size, binder profile, and runtime dispatch"
        ),
    }


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    draft_ranking: str | Path,
    device: str | torch.device,
) -> Path:
    output = Path(out_path)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing Q4/W8 source-MSE evaluation artifact: {output}"
        )
    started = time.perf_counter()
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model_dir, draft_ranking)
    output.parent.mkdir(parents=True, exist_ok=True)
    resource_payloads = {resource.name: resource.data for resource in preflight.resources}

    with ShardReader(preflight.model_dir) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(
                q4_w8_mse_inventory.MODEL_ID, q4_w8_mse_inventory.WEIGHTS_ID
            ),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError(
                    "Q4/W8 source-MSE writer plan differs from completed preflight"
                )
            for index, spec in enumerate(q4_w8_mse_inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, q4_w8_mse_inventory.ResourceSpec):
                    payload = resource_payloads[spec.name]
                else:
                    tensor = source.materialize_tensor(spec, reader, preflight.draft)
                    payload = _encode_tensor(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{len(q4_w8_mse_inventory.OBJECT_SPECS)}] {spec.name}",
                    flush=True,
                )

    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(
            q4_w8_mse_inventory.MODEL_ID, q4_w8_mse_inventory.WEIGHTS_ID
        ),
        target_key=q4_w8_mse_inventory.TARGET_KEY,
        recipe_id=q4_w8_mse_inventory.RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.model_dir,
        out_path=output,
        arguments={
            "model": str(model_dir),
            "out": str(out_path),
            "draft_ranking": str(draft_ranking),
            "device": requested_device,
        },
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=time.perf_counter() - started,
        final_bytes=output.stat().st_size,
        device=resolved_device,
        ranking_path=preflight.draft.ranking,
    )
    report["source"].update(
        {
            "ranking_provenance_path": str(preflight.draft_ranking.sidecar_path),
            "ranking_provenance_sha256": preflight.draft_ranking.sidecar_sha256,
            "ranking_tokens": preflight.draft_ranking.total_tokens,
            "ranking_distinct_token_ids": preflight.draft_ranking.distinct_token_ids,
        }
    )
    report["candidate"] = _candidate_metadata()
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--draft-ranking",
        required=True,
        type=Path,
        help="validated Qwen3.8 total-frequency I64 ranking and sibling provenance",
    )
    parser.add_argument("--device", help="Torch device, e.g. cpu or cuda for ROCm Torch")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate source, ranking, recipe, and complete object plan; write no artifact",
    )
    args = parser.parse_args(argv)
    if args.preflight_only:
        if args.out is not None or args.device is not None:
            parser.error("--preflight-only does not accept --out or --device")
        print(json.dumps(preflight_summary(preflight_conversion(args.model, args.draft_ranking)), indent=2))
        return
    if args.out is None or args.device is None:
        parser.error("conversion requires --out and --device")
    report = convert(
        args.model,
        args.out,
        draft_ranking=args.draft_ranking,
        device=args.device,
    )
    print(f"Q4/W8 source-MSE evaluation conversion report: {report}")


if __name__ == "__main__":
    main()
