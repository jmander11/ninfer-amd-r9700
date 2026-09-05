"""Build the evaluation-only all-rowwise-E4M3 Qwen3.8 R9700 artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, Payload
from tools.artifact.layouts import encode_direct
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3.common import conversion as family_conversion

from . import (
    build_draft_ranking,
    draft_head,
    e4m3_inventory,
    resources,
    source,
    source_recipe,
)
from .e4m3_rowwise import encode_e4m3_rowwise_chunks


@dataclass(frozen=True, slots=True)
class E4M3ConversionPreflight:
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
) -> E4M3ConversionPreflight:
    model = Path(model_dir)
    e4m3_inventory.validate_inventory()
    config_summary = source.validate_config(
        family_conversion.load_json(model / "config.json")
    )
    source_preflight = source_recipe.preflight_sources(model)
    frontend_resources = resources.load_resources(model)
    resource_map = {resource.name: resource.data for resource in frontend_resources}
    object_plan = family_conversion.build_object_plan(
        e4m3_inventory.OBJECT_SPECS, resource_map
    )
    ranking = build_draft_ranking.validate_ranking_provenance(draft_ranking)
    draft = draft_head.compute_shortlist(ranking.ranking_path, model)
    return E4M3ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source_preflight,
        resources=frontend_resources,
        draft=draft,
        draft_ranking=ranking,
        object_plan=object_plan,
    )


def preflight_summary(preflight: E4M3ConversionPreflight) -> dict[str, object]:
    return {
        "artifact_type": "ninfer_qwen3_8_27b_r9700_e4m3_conversion_preflight",
        "schema_version": 1,
        "identity": {
            "model_id": e4m3_inventory.MODEL_ID,
            "weights_id": e4m3_inventory.WEIGHTS_ID,
        },
        "target_key": e4m3_inventory.TARGET_KEY,
        "recipe_id": e4m3_inventory.RECIPE_ID,
        "model_path": str(preflight.model_dir.resolve()),
        "config_summary": preflight.config_summary,
        "source": {
            "recipes": preflight.source.recipe_count,
            "tensors": preflight.source.source_tensor_count,
            "shards": preflight.source.source_shard_count,
            "dtypes": preflight.source.source_dtype_counts,
        },
        "frontend_resources": len(preflight.resources),
        "ranking": {
            "path": str(preflight.draft_ranking.ranking_path),
            "sha256": preflight.draft_ranking.ranking_sha256,
            "provenance_path": str(preflight.draft_ranking.sidecar_path),
            "provenance_sha256": preflight.draft_ranking.sidecar_sha256,
            "tokens": preflight.draft_ranking.total_tokens,
            "distinct_token_ids": preflight.draft_ranking.distinct_token_ids,
            "selected_rows": preflight.draft.n,
        },
        "objects": family_conversion.object_statistics(preflight.object_plan.objects),
        "tensor_encoded_bytes": e4m3_inventory.TENSOR_ENCODED_BYTES,
        "device_arena_bytes": e4m3_inventory.DEVICE_ARENA_BYTES,
    }


def _encode_tensor(tensor: torch.Tensor, spec: e4m3_inventory.TensorSpec) -> Payload:
    if spec.format in e4m3_inventory.DIRECT_FORMATS:
        return encode_direct(tensor, spec.format)
    if spec.format != e4m3_inventory.F8E4M3_ROW_F32S:
        raise ValueError(f"{spec.name}: unsupported E4M3 evaluator format {spec.format}")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{spec.name}: E4M3 source must be BF16, got {tensor.dtype}")
    return encode_e4m3_rowwise_chunks(tensor)


def _candidate_metadata() -> dict[str, object]:
    return {
        "status": "registered-evaluation-only",
        "weight_recipe_selected": False,
        "format_counts": e4m3_inventory.FORMAT_COUNTS,
        "format_encoded_bytes": e4m3_inventory.FORMAT_ENCODED_BYTES,
        "tensor_encoded_bytes": e4m3_inventory.TENSOR_ENCODED_BYTES,
        "device_arena_bytes": e4m3_inventory.DEVICE_ARENA_BYTES,
        "scale": "one little-endian FP32 max(abs(row))/448 multiplier per output row",
        "reason": "requires single-object Op timing and 8K/32K quality admission",
    }


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    draft_ranking: str | Path,
) -> Path:
    output = Path(out_path)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing rowwise-E4M3 evaluation artifact: {output}"
        )
    started = time.perf_counter()
    preflight = preflight_conversion(model_dir, draft_ranking)
    output.parent.mkdir(parents=True, exist_ok=True)
    resource_payloads = {resource.name: resource.data for resource in preflight.resources}

    with ShardReader(preflight.model_dir) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(e4m3_inventory.MODEL_ID, e4m3_inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("E4M3 writer plan differs from completed preflight")
            for index, spec in enumerate(e4m3_inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, e4m3_inventory.ResourceSpec):
                    payload = resource_payloads[spec.name]
                else:
                    tensor = source.materialize_tensor(spec, reader, preflight.draft)
                    payload = _encode_tensor(tensor, spec)
                writer.write(spec.name, payload)
                if not isinstance(spec, e4m3_inventory.ResourceSpec):
                    del tensor
                del payload
                print(
                    f"[{index}/{len(e4m3_inventory.OBJECT_SPECS)}] {spec.name}",
                    flush=True,
                )

    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(e4m3_inventory.MODEL_ID, e4m3_inventory.WEIGHTS_ID),
        target_key=e4m3_inventory.TARGET_KEY,
        recipe_id=e4m3_inventory.RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.model_dir,
        out_path=output,
        arguments={
            "model": str(model_dir),
            "out": str(out_path),
            "draft_ranking": str(draft_ranking),
            "device": "cpu",
        },
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=time.perf_counter() - started,
        final_bytes=output.stat().st_size,
        device=torch.device("cpu"),
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
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate source, frontend, ranking, and exact object plan; write no artifact",
    )
    args = parser.parse_args(argv)
    if args.preflight_only:
        if args.out is not None:
            parser.error("--preflight-only does not accept --out")
        summary = preflight_summary(
            preflight_conversion(args.model, args.draft_ranking)
        )
        print(json.dumps(summary, indent=2))
        return
    if args.out is None:
        parser.error("conversion requires --out")
    report = convert(args.model, args.out, draft_ranking=args.draft_ranking)
    print(f"rowwise-E4M3 evaluation conversion report: {report}")


if __name__ == "__main__":
    main()
