"""Build the W8G32 plus BF16 full-attention value/output evaluator."""

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
    resources,
    source,
    source_recipe,
    w8_bf16_attention_vo_inventory,
)


@dataclass(frozen=True, slots=True)
class W8Bf16AttentionValueOutputConversionPreflight:
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
) -> W8Bf16AttentionValueOutputConversionPreflight:
    model = Path(model_dir)
    w8_bf16_attention_vo_inventory.validate_inventory()
    config_summary = source.validate_config(
        family_conversion.load_json(model / "config.json")
    )
    source_preflight = source_recipe.preflight_sources(model)
    frontend_resources = resources.load_resources(model)
    resource_map = {resource.name: resource.data for resource in frontend_resources}
    object_plan = family_conversion.build_object_plan(
        w8_bf16_attention_vo_inventory.OBJECT_SPECS, resource_map
    )
    ranking = build_draft_ranking.validate_ranking_provenance(draft_ranking)
    draft = draft_head.compute_shortlist(ranking.ranking_path, model)
    return W8Bf16AttentionValueOutputConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source_preflight,
        resources=frontend_resources,
        draft=draft,
        draft_ranking=ranking,
        object_plan=object_plan,
    )


def _materialize_tensor(
    spec: w8_bf16_attention_vo_inventory.TensorSpec,
    reader: ShardReader,
    draft: draft_head.DraftHeadContext,
) -> torch.Tensor:
    return source.materialize_tensor(spec, reader, draft)


def _encode_tensor(
    tensor: torch.Tensor,
    spec: w8_bf16_attention_vo_inventory.TensorSpec,
    device: torch.device,
) -> bytes:
    if (
        spec.format not in w8_bf16_attention_vo_inventory.DIRECT_FORMATS
        and tensor.dtype != torch.bfloat16
    ):
        raise TypeError(f"{spec.name}: W8 source must be BF16, got {tensor.dtype}")
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
            f"refusing to overwrite existing W8/BF16-attention evaluation artifact: {output}"
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
                w8_bf16_attention_vo_inventory.MODEL_ID,
                w8_bf16_attention_vo_inventory.WEIGHTS_ID,
            ),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError(
                    "W8/BF16-attention writer plan differs from completed preflight"
                )
            for index, spec in enumerate(
                w8_bf16_attention_vo_inventory.OBJECT_SPECS, start=1
            ):
                if isinstance(spec, w8_bf16_attention_vo_inventory.ResourceSpec):
                    payload = resource_payloads[spec.name]
                else:
                    tensor = _materialize_tensor(spec, reader, preflight.draft)
                    payload = _encode_tensor(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{len(w8_bf16_attention_vo_inventory.OBJECT_SPECS)}] "
                    f"{spec.name}",
                    flush=True,
                )

    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(
            w8_bf16_attention_vo_inventory.MODEL_ID,
            w8_bf16_attention_vo_inventory.WEIGHTS_ID,
        ),
        target_key=w8_bf16_attention_vo_inventory.TARGET_KEY,
        recipe_id=w8_bf16_attention_vo_inventory.RECIPE_ID,
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
        "comparison_role": (
            "all-W8G32 except represented source-BF16 gate/value and output projections "
            "in every full-attention layer"
        ),
        "tensor_encoded_bytes": w8_bf16_attention_vo_inventory.TENSOR_ENCODED_BYTES,
        "device_arena_bytes": w8_bf16_attention_vo_inventory.DEVICE_ARENA_BYTES,
        "reason": (
            "complete source-Q5 full-attention value/output role family within the fixed "
            "promotion and 32K-headroom budget"
        ),
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--draft-ranking",
        required=True,
        type=Path,
        help="validated Qwen3.8 total-frequency I64 ranking and sibling provenance",
    )
    parser.add_argument(
        "--device", required=True, help="Torch device, e.g. cpu or cuda for ROCm Torch"
    )
    args = parser.parse_args(argv)
    report = convert(
        args.model,
        args.out,
        draft_ranking=args.draft_ranking,
        device=args.device,
    )
    print(f"W8/BF16-attention value/output conversion report: {report}")


if __name__ == "__main__":
    main()
