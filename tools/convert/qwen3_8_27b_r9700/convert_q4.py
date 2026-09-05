"""Build the registered, evaluation-only all-Q4G64 Qwen3.8 R9700 artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3.common import conversion as family_conversion

from . import (
    build_draft_ranking,
    draft_head,
    preflight_identity,
    q4_inventory,
    resources,
    source,
    source_recipe,
)


@dataclass(frozen=True, slots=True)
class Q4ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: source_recipe.SourcePreflight
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    draft_ranking: build_draft_ranking.RankingProvenance
    object_plan: family_conversion.ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def preflight_conversion(
    model_dir: str | Path, draft_ranking: str | Path
) -> Q4ConversionPreflight:
    model = Path(model_dir)
    q4_inventory.validate_inventory()
    config_summary = source.validate_config(
        family_conversion.load_json(model / "config.json")
    )
    source_preflight = source_recipe.preflight_sources(model)
    frontend_resources = resources.load_resources(model)
    resource_map = {resource.name: resource.data for resource in frontend_resources}
    object_plan = family_conversion.build_object_plan(
        q4_inventory.OBJECT_SPECS, resource_map
    )
    ranking = build_draft_ranking.validate_ranking_provenance(draft_ranking)
    draft = draft_head.compute_shortlist(ranking.ranking_path, model)
    return Q4ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source_preflight,
        resources=frontend_resources,
        draft=draft,
        draft_ranking=ranking,
        object_plan=object_plan,
    )


def preflight_summary(preflight: Q4ConversionPreflight) -> dict[str, object]:
    """Render the exact all-Q4 source-to-artifact plan without materializing payloads."""

    source_summary = {
        "recipes": preflight.source.recipe_count,
        "tensors": preflight.source.source_tensor_count,
        "shards": preflight.source.source_shard_count,
        "dtypes": preflight.source.source_dtype_counts,
    }
    if (source_summary["tensors"] != 1199 or source_summary["shards"] != 18
            or source_summary["dtypes"] != {"BF16": 1199}):
        raise ValueError("source is not the exact 1,199-BF16-tensor/18-shard checkpoint")
    objects = preflight.object_plan.objects
    statistics = family_conversion.object_statistics(objects)
    if statistics["count"] != len(q4_inventory.OBJECT_SPECS):
        raise ValueError("all-Q4 object plan is incomplete")
    return {
        "artifact_type": "ninfer_qwen3_8_27b_r9700_q4_conversion_preflight",
        "schema_version": 1,
        "identity": {
            "model_id": q4_inventory.MODEL_ID,
            "weights_id": q4_inventory.WEIGHTS_ID,
        },
        "target_key": q4_inventory.TARGET_KEY,
        "recipe_id": q4_inventory.RECIPE_ID,
        "model_path": str(preflight.model_dir.resolve()),
        "config_summary": preflight.config_summary,
        "source": source_summary,
        "checkpoint": preflight_identity.source_checkpoint(preflight.model_dir),
        "frontend_resources": preflight_identity.frontend_resources(preflight.resources),
        "ranking": {
            "path": str(preflight.draft_ranking.ranking_path),
            "sha256": preflight.draft_ranking.ranking_sha256,
            "provenance_path": str(preflight.draft_ranking.sidecar_path),
            "provenance_sha256": preflight.draft_ranking.sidecar_sha256,
            "tokens": preflight.draft_ranking.total_tokens,
            "distinct_token_ids": preflight.draft_ranking.distinct_token_ids,
            "selected_rows": preflight.draft.n,
        },
        "objects": statistics,
        "object_plan_sha256": preflight_identity.object_plan_sha256(objects),
        "format_counts": q4_inventory.FORMAT_COUNTS,
        "format_encoded_bytes": q4_inventory.FORMAT_ENCODED_BYTES,
        "tensor_encoded_bytes": q4_inventory.TENSOR_ENCODED_BYTES,
        "device_arena_bytes": q4_inventory.DEVICE_ARENA_BYTES,
    }


def _materialize_tensor(
    spec: q4_inventory.TensorSpec,
    reader: ShardReader,
    draft: draft_head.DraftHeadContext,
) -> torch.Tensor:
    return source.materialize_tensor(spec, reader, draft)


def _encode_tensor(
    tensor: torch.Tensor,
    spec: q4_inventory.TensorSpec,
    device: torch.device,
) -> bytes:
    if spec.format not in q4_inventory.DIRECT_FORMATS and tensor.dtype != torch.bfloat16:
        raise TypeError(f"{spec.name}: Q4 source must be BF16, got {tensor.dtype}")
    return family_conversion.encode_tensor_payload(tensor, spec, device)


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
            f"refusing to overwrite existing all-Q4 evaluation artifact: {output}"
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
            ArtifactIdentity(q4_inventory.MODEL_ID, q4_inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("all-Q4 writer plan differs from completed preflight")
            for index, spec in enumerate(q4_inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, q4_inventory.ResourceSpec):
                    payload = resource_payloads[spec.name]
                else:
                    tensor = _materialize_tensor(spec, reader, preflight.draft)
                    payload = _encode_tensor(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{len(q4_inventory.OBJECT_SPECS)}] {spec.name}", flush=True)

    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(q4_inventory.MODEL_ID, q4_inventory.WEIGHTS_ID),
        target_key=q4_inventory.TARGET_KEY,
        recipe_id=q4_inventory.RECIPE_ID,
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
    report["candidate"] = {
        "status": "registered-evaluation-only",
        "weight_recipe_selected": False,
        "comparison_role": "all-Q4G64 capacity floor",
        "tensor_encoded_bytes": q4_inventory.TENSOR_ENCODED_BYTES,
        "device_arena_bytes": q4_inventory.DEVICE_ARENA_BYTES,
        "reason": "requires real-model accuracy, exact-token, and whole-inference selection",
    }
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
        help="validate source, frontend, ranking, and exact object plan; write no artifact",
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
    print(f"all-Q4 evaluation conversion report: {report}")


if __name__ == "__main__":
    main()
