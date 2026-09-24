"""Build the qualification-only four-role rowwise-E4M3/all-other-Q4 artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Sequence

import torch

from tools.artifact.container import (
    PREFIX_BYTES,
    PAYLOAD_ALIGNMENT,
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
    Payload,
    encode_directory,
)
from tools.artifact.layouts import align_up, encode_direct
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3.common import conversion as family_conversion

from . import (
    build_draft_ranking,
    draft_head,
    fp8_hybrid_inventory,
    preflight_identity,
    resources,
    source,
    source_recipe,
)
from .e4m3_rowwise import encode_e4m3_rowwise_chunks


_SOURCE_SHARDS = tuple(
    f"model-{index:05d}-of-00018.safetensors" for index in range(1, 19)
)
_EXPECTED_SOURCE_TENSORS = 1199
_DISK_HEADROOM_BYTES = 1 << 30


@dataclass(frozen=True, slots=True)
class SourceShardManifest:
    index_path: Path
    index_sha256: str
    shard_names: tuple[str, ...]
    shard_bytes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DestinationPreflight:
    output_path: Path
    report_path: Path
    filesystem_path: Path
    projected_file_bytes: int
    headroom_bytes: int
    required_free_bytes: int
    available_free_bytes: int


@dataclass(frozen=True, slots=True)
class Fp8HybridConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: source_recipe.SourcePreflight
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    draft_ranking: build_draft_ranking.RankingProvenance
    object_plan: family_conversion.ObjectPlan
    source_shards: SourceShardManifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _source_shard_manifest(model: Path) -> SourceShardManifest:
    index_path = model / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    value = json.loads(index_bytes)
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != _EXPECTED_SOURCE_TENSORS:
        raise ValueError(
            f"BF16 source index must contain exactly {_EXPECTED_SOURCE_TENSORS} tensors"
        )
    shard_names = tuple(sorted(set(weight_map.values())))
    if shard_names != _SOURCE_SHARDS:
        raise ValueError("BF16 source index does not reference the exact 18-shard set")
    shard_bytes: list[int] = []
    for name in shard_names:
        path = model / name
        if not path.is_file():
            raise FileNotFoundError(f"BF16 source shard is missing: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"BF16 source shard is empty: {path}")
        shard_bytes.append(size)
    return SourceShardManifest(
        index_path=index_path.resolve(),
        index_sha256=hashlib.sha256(index_bytes).hexdigest(),
        shard_names=shard_names,
        shard_bytes=tuple(shard_bytes),
    )


def _projected_file_bytes(preflight: Fp8HybridConversionPreflight) -> int:
    identity = ArtifactIdentity(
        fp8_hybrid_inventory.MODEL_ID, fp8_hybrid_inventory.WEIGHTS_ID
    )
    directory = encode_directory(identity, preflight.object_plan.objects)
    payload_offset = align_up(PREFIX_BYTES + len(directory), PAYLOAD_ALIGNMENT)
    last = preflight.object_plan.objects[-1]
    return payload_offset + last.offset + last.bytes


def _object_plan_sha256(preflight: Fp8HybridConversionPreflight) -> str:
    payload = json.dumps(
        [obj.to_json() for obj in preflight.object_plan.objects],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _existing_filesystem_path(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(f"no existing ancestor for conversion output: {path}")
        candidate = parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


def preflight_destination(
    preflight: Fp8HybridConversionPreflight, out_path: str | Path
) -> DestinationPreflight:
    output = Path(out_path).resolve()
    if output.suffix != ".ninfer":
        raise ValueError("FP8/Q4 hybrid output must use the .ninfer suffix")
    report = Path(str(output) + ".conversion.json")
    if output.exists():
        raise FileExistsError(f"preflight output already exists: {output}")
    if report.exists():
        raise FileExistsError(f"preflight conversion report already exists: {report}")
    filesystem_path = _existing_filesystem_path(output.parent)
    if not os.access(filesystem_path, os.W_OK | os.X_OK):
        raise PermissionError(f"conversion output filesystem is not writable: {filesystem_path}")
    projected = _projected_file_bytes(preflight)
    required = projected + _DISK_HEADROOM_BYTES
    available = shutil.disk_usage(filesystem_path).free
    if available < required:
        raise OSError(
            f"insufficient free space for FP8/Q4 hybrid conversion: "
            f"required {required}, available {available}"
        )
    return DestinationPreflight(
        output_path=output,
        report_path=report,
        filesystem_path=filesystem_path.resolve(),
        projected_file_bytes=projected,
        headroom_bytes=_DISK_HEADROOM_BYTES,
        required_free_bytes=required,
        available_free_bytes=available,
    )


def preflight_conversion(
    model_dir: str | Path, draft_ranking: str | Path
) -> Fp8HybridConversionPreflight:
    model = Path(model_dir)
    fp8_hybrid_inventory.validate_inventory()
    config_summary = source.validate_config(
        family_conversion.load_json(model / "config.json")
    )
    source_preflight = source_recipe.preflight_sources(model)
    if (
        source_preflight.source_tensor_count != _EXPECTED_SOURCE_TENSORS
        or source_preflight.source_shard_count != len(_SOURCE_SHARDS)
        or source_preflight.source_dtype_counts != {"BF16": _EXPECTED_SOURCE_TENSORS}
    ):
        raise ValueError("source metadata is not the exact 1,199-BF16-tensor/18-shard contract")
    source_shards = _source_shard_manifest(model)
    frontend_resources = resources.load_resources(model)
    resource_map = {resource.name: resource.data for resource in frontend_resources}
    object_plan = family_conversion.build_object_plan(
        fp8_hybrid_inventory.OBJECT_SPECS, resource_map
    )
    ranking = build_draft_ranking.validate_ranking_provenance(draft_ranking)
    draft = draft_head.compute_shortlist(ranking.ranking_path, model)
    return Fp8HybridConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source_preflight,
        resources=frontend_resources,
        draft=draft,
        draft_ranking=ranking,
        object_plan=object_plan,
        source_shards=source_shards,
    )


def preflight_summary(
    preflight: Fp8HybridConversionPreflight, destination: DestinationPreflight
) -> dict[str, object]:
    conversion_argv = [
        sys.executable,
        "-m",
        "tools.convert.qwen3_8_27b_r9700.convert_fp8_hybrid",
        "--model",
        str(preflight.model_dir.resolve()),
        "--draft-ranking",
        str(preflight.draft_ranking.ranking_path.resolve()),
        "--out",
        str(destination.output_path),
        "--device",
        "cuda",
    ]
    validation_argv = conversion_argv[:-2] + ["--validate-only"]
    return {
        "artifact_type": "ninfer_qwen3_8_27b_r9700_fp8_q4_hybrid_conversion_preflight",
        "schema_version": 1,
        "identity": {
            "model_id": fp8_hybrid_inventory.MODEL_ID,
            "weights_id": fp8_hybrid_inventory.WEIGHTS_ID,
        },
        "target_key": fp8_hybrid_inventory.TARGET_KEY,
        "recipe_id": fp8_hybrid_inventory.RECIPE_ID,
        "selection_sha256": fp8_hybrid_inventory.SELECTION_SHA256,
        "model_path": str(preflight.model_dir.resolve()),
        "config_summary": preflight.config_summary,
        "source": {
            "recipes": preflight.source.recipe_count,
            "tensors": preflight.source.source_tensor_count,
            "shards": preflight.source.source_shard_count,
            "dtypes": preflight.source.source_dtype_counts,
            "index_path": str(preflight.source_shards.index_path),
            "index_sha256": preflight.source_shards.index_sha256,
            "shard_names": list(preflight.source_shards.shard_names),
            "shard_bytes": list(preflight.source_shards.shard_bytes),
            "total_shard_bytes": sum(preflight.source_shards.shard_bytes),
        },
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
        "objects": family_conversion.object_statistics(preflight.object_plan.objects),
        "format_counts": fp8_hybrid_inventory.FORMAT_COUNTS,
        "format_encoded_bytes": fp8_hybrid_inventory.FORMAT_ENCODED_BYTES,
        "tensor_encoded_bytes": fp8_hybrid_inventory.TENSOR_ENCODED_BYTES,
        "device_arena_bytes": fp8_hybrid_inventory.DEVICE_ARENA_BYTES,
        "object_plan_sha256": _object_plan_sha256(preflight),
        "destination": {
            "artifact_path": str(destination.output_path),
            "report_path": str(destination.report_path),
            "filesystem_path": str(destination.filesystem_path),
            "destination_exists": False,
            "projected_file_bytes": destination.projected_file_bytes,
            "headroom_bytes": destination.headroom_bytes,
            "required_free_bytes": destination.required_free_bytes,
            "available_free_bytes": destination.available_free_bytes,
        },
        "conversion_argv": conversion_argv,
        "validation_argv": validation_argv,
    }


def _encode_tensor(
    tensor: torch.Tensor,
    spec: fp8_hybrid_inventory.TensorSpec,
    device: torch.device,
) -> Payload:
    if spec.format in fp8_hybrid_inventory.DIRECT_FORMATS:
        return encode_direct(tensor, spec.format)
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{spec.name}: quantized source must be BF16, got {tensor.dtype}")
    if spec.format == fp8_hybrid_inventory.F8E4M3_ROW_F32S:
        return encode_e4m3_rowwise_chunks(tensor)
    if spec.format == fp8_hybrid_inventory.Q4:
        return family_conversion.encode_tensor_payload(tensor, spec, device)
    raise ValueError(f"{spec.name}: unsupported FP8/Q4 hybrid format {spec.format}")


def _candidate_metadata() -> dict[str, object]:
    return {
        "status": "registered-evaluation-only",
        "weight_recipe_selected": False,
        "selection_sha256": fp8_hybrid_inventory.SELECTION_SHA256,
        "format_counts": fp8_hybrid_inventory.FORMAT_COUNTS,
        "format_encoded_bytes": fp8_hybrid_inventory.FORMAT_ENCODED_BYTES,
        "tensor_encoded_bytes": fp8_hybrid_inventory.TENSOR_ENCODED_BYTES,
        "device_arena_bytes": fp8_hybrid_inventory.DEVICE_ARENA_BYTES,
        "reason": "requires fresh executed-path proofs and whole-inference qualification",
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_completed_conversion(
    preflight: Fp8HybridConversionPreflight, out_path: str | Path
) -> dict[str, object]:
    output = Path(out_path).resolve()
    report_path = Path(str(output) + ".conversion.json")
    projected_bytes = _projected_file_bytes(preflight)
    if output.stat().st_size != projected_bytes:
        raise ValueError("completed FP8/Q4 hybrid artifact size differs from preflight")
    expected_identity = ArtifactIdentity(
        fp8_hybrid_inventory.MODEL_ID, fp8_hybrid_inventory.WEIGHTS_ID
    )
    with Artifact.open(output) as artifact:
        if artifact.identity != expected_identity:
            raise ValueError("completed FP8/Q4 hybrid artifact identity differs")
        if artifact.objects != preflight.object_plan.objects:
            raise ValueError("completed FP8/Q4 hybrid artifact object plan differs")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact_sha256 = _sha256(output)
    expected_report_members = {
        "identity": {
            "model_id": fp8_hybrid_inventory.MODEL_ID,
            "weights_id": fp8_hybrid_inventory.WEIGHTS_ID,
        },
        "target_key": fp8_hybrid_inventory.TARGET_KEY,
        "recipe_id": fp8_hybrid_inventory.RECIPE_ID,
    }
    for name, expected in expected_report_members.items():
        if report.get(name) != expected:
            raise ValueError(f"completed FP8/Q4 hybrid report {name} differs")
    candidate = report.get("candidate")
    if not isinstance(candidate, dict) or (
        candidate.get("weight_recipe_selected") is not False
        or candidate.get("selection_sha256") != fp8_hybrid_inventory.SELECTION_SHA256
        or candidate.get("object_plan_sha256") != _object_plan_sha256(preflight)
    ):
        raise ValueError("completed FP8/Q4 hybrid candidate receipt differs")
    artifact_record = report.get("artifact")
    if not isinstance(artifact_record, dict) or (
        artifact_record.get("bytes") != projected_bytes
        or artifact_record.get("sha256") != artifact_sha256
    ):
        raise ValueError("completed FP8/Q4 hybrid artifact digest receipt differs")
    source_record = report.get("source")
    if not isinstance(source_record, dict) or (
        source_record.get("ranking_sha256") != preflight.draft_ranking.ranking_sha256
        or source_record.get("index_sha256") != preflight.source_shards.index_sha256
    ):
        raise ValueError("completed FP8/Q4 hybrid source receipt differs")
    return {
        "artifact_type": "ninfer_qwen3_8_27b_r9700_fp8_q4_hybrid_validation",
        "schema_version": 1,
        "artifact_path": str(output),
        "report_path": str(report_path),
        "identity": expected_report_members["identity"],
        "recipe_id": fp8_hybrid_inventory.RECIPE_ID,
        "selection_sha256": fp8_hybrid_inventory.SELECTION_SHA256,
        "object_plan_sha256": _object_plan_sha256(preflight),
        "artifact_bytes": projected_bytes,
        "artifact_sha256": artifact_sha256,
        "objects": len(preflight.object_plan.objects),
        "status": "valid",
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
            f"refusing to overwrite existing FP8/Q4 hybrid evaluation artifact: {output}"
        )
    started = time.perf_counter()
    requested_device = str(device)
    preflight = preflight_conversion(model_dir, draft_ranking)
    destination = preflight_destination(preflight, output)
    resolved_device = pick_device(device)
    output = destination.output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    resource_payloads = {resource.name: resource.data for resource in preflight.resources}

    with ShardReader(preflight.model_dir) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(fp8_hybrid_inventory.MODEL_ID, fp8_hybrid_inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("FP8/Q4 hybrid writer plan differs from completed preflight")
            for index, spec in enumerate(fp8_hybrid_inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, fp8_hybrid_inventory.ResourceSpec):
                    payload = resource_payloads[spec.name]
                else:
                    tensor = source.materialize_tensor(spec, reader, preflight.draft)
                    payload = _encode_tensor(tensor, spec, resolved_device)
                writer.write(spec.name, payload)
                if not isinstance(spec, fp8_hybrid_inventory.ResourceSpec):
                    del tensor
                del payload
                print(
                    f"[{index}/{len(fp8_hybrid_inventory.OBJECT_SPECS)}] {spec.name}",
                    flush=True,
                )

    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(
            fp8_hybrid_inventory.MODEL_ID, fp8_hybrid_inventory.WEIGHTS_ID
        ),
        target_key=fp8_hybrid_inventory.TARGET_KEY,
        recipe_id=fp8_hybrid_inventory.RECIPE_ID,
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
            "index_path": str(preflight.source_shards.index_path),
            "index_sha256": preflight.source_shards.index_sha256,
            "source_shards": list(preflight.source_shards.shard_names),
        }
    )
    report["candidate"] = _candidate_metadata()
    report["candidate"]["object_plan_sha256"] = _object_plan_sha256(preflight)
    report_path = destination.report_path
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--draft-ranking", required=True, type=Path)
    parser.add_argument("--device", help="Torch device for Q4 encoding, e.g. cpu or cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        if args.out is None:
            parser.error("--preflight-only requires --out")
        if args.device is not None:
            parser.error("--preflight-only does not accept --device")
        checked = preflight_conversion(args.model, args.draft_ranking)
        destination = preflight_destination(checked, args.out)
        print(json.dumps(preflight_summary(checked, destination), indent=2))
        return
    if args.validate_only:
        if args.out is None:
            parser.error("--validate-only requires --out")
        if args.device is not None:
            parser.error("--validate-only does not accept --device")
        checked = preflight_conversion(args.model, args.draft_ranking)
        print(json.dumps(validate_completed_conversion(checked, args.out), indent=2))
        return
    if args.out is None or args.device is None:
        parser.error("conversion requires --out and --device")
    report = convert(
        args.model,
        args.out,
        draft_ranking=args.draft_ranking,
        device=args.device,
    )
    print(f"FP8/Q4 hybrid evaluation conversion report: {report}")


if __name__ == "__main__":
    main()
    Artifact,
