"""Append the explicit all-Q4G64/BF16-codebook DFlash2 evaluator to an R9700 base.

Accepted bases are exactly the all-Q4G64 evaluator, the source-MSE mixed Q4/W8 evaluator, and
the authority-bound four-role rowwise-FP8/all-other-Q4 evaluator.
Every existing payload byte is copied unchanged; only the 66 source-derived DFlash2 objects are
encoded and appended under a new identity.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterator, Sequence

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
    ResourceObject,
    ResourceSpec as ArtifactResourceSpec,
    TensorObject,
    TensorSpec as ArtifactTensorSpec,
    encode_directory,
    plan_objects,
)
from tools.artifact.layouts import align_up
from tools.convert.qwen3.common.inventory import ResourceSpec, TensorSpec

from . import dflash2_matrix_recipes, dflash2_q4_inventory as inventory
from . import fp8_hybrid_inventory, q4_inventory, q4_w8_mse_inventory


_COPY_CHUNK = 64 * 1024 * 1024
_CANONICAL_RECIPE = dflash2_matrix_recipes.CANONICAL_Q4G64


@dataclass(frozen=True, slots=True)
class Preflight:
    base_path: Path
    dflash_model: Path
    source: dict[str, object]
    base_identity: ArtifactIdentity
    output_identity: ArtifactIdentity
    specs: tuple[ArtifactResourceSpec | ArtifactTensorSpec, ...]
    objects: tuple[ArtifactObject, ...]
    projected_file_bytes: int
    projected_device_arena_bytes: int


def _artifact_spec(obj: ArtifactObject) -> ArtifactResourceSpec | ArtifactTensorSpec:
    if isinstance(obj, TensorObject):
        return ArtifactTensorSpec(obj.name, obj.shape, obj.format, obj.layout)
    if isinstance(obj, ResourceObject):
        return ArtifactResourceSpec(obj.name, obj.encoding, obj.bytes)
    raise TypeError(f"unsupported base artifact object: {type(obj).__name__}")


def _expected_base(identity: ArtifactIdentity):
    if identity == ArtifactIdentity(inventory.MODEL_ID, inventory.ALL_Q4_BASE_WEIGHTS_ID):
        return (
            q4_inventory.OBJECT_SPECS,
            ArtifactIdentity(inventory.MODEL_ID, inventory.ALL_Q4_WEIGHTS_ID),
            inventory.ALL_Q4_DEVICE_ARENA_BYTES,
        )
    if identity == ArtifactIdentity(inventory.MODEL_ID, inventory.MIXED_BASE_WEIGHTS_ID):
        return (
            q4_w8_mse_inventory.OBJECT_SPECS,
            ArtifactIdentity(inventory.MODEL_ID, inventory.MIXED_WEIGHTS_ID),
            inventory.MIXED_DEVICE_ARENA_BYTES,
        )
    if identity == ArtifactIdentity(inventory.MODEL_ID, inventory.HYBRID_BASE_WEIGHTS_ID):
        return (
            fp8_hybrid_inventory.OBJECT_SPECS,
            ArtifactIdentity(inventory.MODEL_ID, inventory.HYBRID_WEIGHTS_ID),
            inventory.HYBRID_DEVICE_ARENA_BYTES,
        )
    raise ValueError(
        "DFlash2 Q4 evaluator base must be qwen3.8-27b/"
        f"{inventory.ALL_Q4_BASE_WEIGHTS_ID} or qwen3.8-27b/"
        f"{inventory.MIXED_BASE_WEIGHTS_ID}, or qwen3.8-27b/"
        f"{inventory.HYBRID_BASE_WEIGHTS_ID}; got {identity.model_id}/{identity.weights_id}"
    )


def _validate_base_objects(actual: tuple[ArtifactObject, ...], expected) -> None:
    if len(actual) != len(expected):
        raise ValueError("base artifact object count differs from its registered inventory")
    for obj, spec in zip(actual, expected, strict=True):
        if obj.name != spec.name:
            raise ValueError(f"base artifact order differs at {obj.name!r}/{spec.name!r}")
        if isinstance(spec, TensorSpec):
            if not isinstance(obj, TensorObject) or (
                obj.shape,
                obj.format,
                obj.layout,
            ) != (spec.shape, spec.format, spec.layout):
                raise ValueError(f"base tensor contract differs for {spec.name}")
        elif isinstance(spec, ResourceSpec):
            if not isinstance(obj, ResourceObject) or obj.encoding != spec.encoding:
                raise ValueError(f"base resource contract differs for {spec.name}")
        else:
            raise TypeError(f"unsupported registered base spec: {type(spec).__name__}")


def preflight(base: str | Path, dflash_model: str | Path) -> Preflight:
    base_path = Path(base)
    model_path = Path(dflash_model)
    inventory.validate_inventory()
    source = inventory.validate_source(model_path)
    with Artifact.open(base_path) as artifact:
        expected, output_identity, device_bytes = _expected_base(artifact.identity)
        _validate_base_objects(artifact.objects, expected)
        specs = tuple(_artifact_spec(obj) for obj in artifact.objects) + tuple(
            ArtifactTensorSpec(spec.name, spec.shape, spec.format, spec.layout)
            for spec in inventory.TENSOR_SPECS
        )
        base_identity = artifact.identity
    objects = plan_objects(specs)
    directory = encode_directory(output_identity, objects)
    payload_offset = align_up(16 + len(directory), 4096)
    projected_file_bytes = payload_offset + objects[-1].offset + objects[-1].bytes
    return Preflight(
        base_path=base_path,
        dflash_model=model_path,
        source=source,
        base_identity=base_identity,
        output_identity=output_identity,
        specs=specs,
        objects=objects,
        projected_file_bytes=projected_file_bytes,
        projected_device_arena_bytes=device_bytes,
    )


def _chunks(payload: memoryview) -> Iterator[memoryview]:
    raw = payload.cast("B")
    for begin in range(0, len(raw), _COPY_CHUNK):
        yield raw[begin : begin + _COPY_CHUNK]


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _base_authority(path: Path, identity: ArtifactIdentity, artifact_sha256: str) -> dict | None:
    from tools.ppl.run import N16_MIGRATION_PROFILES
    if identity.weights_id not in N16_MIGRATION_PROFILES:
        return None
    receipt_path = Path(str(path.resolve()) + ".conversion.json")
    if identity.weights_id == inventory.HYBRID_BASE_WEIGHTS_ID:
        from tools.ppl.run import validate_n16_conversion_receipt
        receipt = validate_n16_conversion_receipt(path, {
            "path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": artifact_sha256, "weights_id": identity.weights_id,
        })
    else:
        from .publish_n16_migration_receipt import validate_receipt
        receipt = validate_receipt(receipt_path, {
            "path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": artifact_sha256, "weights_id": identity.weights_id,
        })
    if not isinstance(receipt, dict):
        raise ValueError("N16 base migration authority differs")
    values = {"receipt": {"path": receipt["path"], "sha256": receipt["sha256"]},
              "recipe_id": receipt["recipe_id"],
              "object_plan_sha256": receipt["object_plan_sha256"],
              "source_artifact_sha256": receipt["source_artifact_sha256"],
              "source_receipt_sha256": receipt["source_receipt_sha256"],
              "transcoder_sha256": receipt["transcoder_sha256"]}
    if identity.weights_id == inventory.HYBRID_BASE_WEIGHTS_ID:
        values.update({key: receipt[key] for key in (
            "selection_sha256", "source_index_sha256", "source_ranking_sha256",
        )})
    else:
        values["receipt_producer_sha256"] = receipt["receipt_producer_sha256"]
    return values


def preflight_summary(checked: Preflight, output: Path | None = None) -> dict[str, object]:
    """Bind the complete no-output DFlash conversion plan and its publication paths."""

    publication: dict[str, object] = {
        "writer": "atomic-no-replace",
        "pending_receipt_schema": 1,
        "conversion_report": "atomic-replace-after-completed-artifact-validation",
    }
    if output is not None:
        output = output.resolve()
        if output.exists():
            raise FileExistsError(f"preflight output already exists: {output}")
        publication.update(
            {
                "artifact_path": str(output),
                "pending_receipt_path": str(_pending_path(output)),
                "report_path": str(_report_path(output)),
                "destination_exists": False,
            }
        )
    plan_payload = json.dumps(
        [asdict(obj) for obj in checked.objects],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tensor_objects = [obj for obj in checked.objects if isinstance(obj, TensorObject)]
    resource_objects = [obj for obj in checked.objects if isinstance(obj, ResourceObject)]
    base_sha256 = _sha256(checked.base_path)
    return {
        "artifact_type": "ninfer_qwen3_8_27b_r9700_dflash2_q4_conversion_preflight",
        "schema_version": 1,
        "base": {
            "path": str(checked.base_path.resolve()),
            "identity": _identity_record(checked.base_identity),
            "bytes": checked.base_path.stat().st_size,
            "sha256": base_sha256,
            "authority": _base_authority(
                checked.base_path, checked.base_identity, base_sha256
            ),
            "objects": len(checked.objects) - len(inventory.TENSOR_SPECS),
            "payload_copy": "byte_exact",
        },
        "output_identity": _identity_record(checked.output_identity),
        "dflash_source": {
            "path": str(checked.dflash_model.resolve()),
            **checked.source,
        },
        "dflash_plan": {
            "recipe": inventory.matrix_recipe_summary(_CANONICAL_RECIPE),
            "candidate_matrix_recipes": tuple(
                recipe
                for recipe in inventory.MATRIX_RECIPE_SUMMARIES
                if recipe["key"] != _CANONICAL_RECIPE
            ),
            "source_tensors": len(inventory.SOURCE_NAMES),
            "appended_objects": len(inventory.TENSOR_SPECS),
            "combined_objects": len(checked.objects),
            "format_counts": inventory.FORMAT_COUNTS,
            "format_encoded_bytes": inventory.FORMAT_ENCODED_BYTES,
            "tensor_encoded_bytes": inventory.TENSOR_ENCODED_BYTES,
            "selector_codebooks": [
                {
                    "name": spec.name,
                    "shape": list(spec.shape),
                    "format": spec.format,
                    "layout": spec.layout,
                }
                for spec in inventory.TENSOR_SPECS
                if spec.name.endswith("codebook")
            ],
            "runtime_repack": False,
        },
        "combined_plan": {
            "objects": len(checked.objects),
            "tensors": len(tensor_objects),
            "resources": len(resource_objects),
            "formats": dict(sorted(Counter(obj.format for obj in tensor_objects).items())),
            "layouts": dict(sorted(Counter(obj.layout for obj in tensor_objects).items())),
            "encodings": dict(
                sorted(Counter(obj.encoding for obj in resource_objects).items())
            ),
            "object_bytes": sum(obj.bytes for obj in checked.objects),
            "payload_span_bytes": (
                checked.objects[-1].offset + checked.objects[-1].bytes
                if checked.objects else 0
            ),
            "sha256": hashlib.sha256(plan_payload).hexdigest(),
        },
        "projected_file_bytes": checked.projected_file_bytes,
        "projected_device_arena_bytes": checked.projected_device_arena_bytes,
        "publication": publication,
    }


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_completed_artifact(
    path: Path,
    identity: ArtifactIdentity,
    objects: tuple[ArtifactObject, ...],
    expected_bytes: int,
    expected_sha256: str | None = None,
) -> str:
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError("completed DFlash2 artifact size differs from preflight")
    with Artifact.open(path) as artifact:
        if artifact.identity != identity:
            raise ValueError("completed DFlash2 artifact identity differs from preflight")
        if artifact.objects != objects:
            raise ValueError("completed DFlash2 artifact object plan differs from preflight")
    actual_sha256 = _sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("completed DFlash2 artifact SHA-256 differs from expected value")
    return actual_sha256


def _identity_record(identity: ArtifactIdentity) -> dict[str, str]:
    return {"model_id": identity.model_id, "weights_id": identity.weights_id}


def _report_value(
    checked: Preflight,
    output: Path,
    base_sha256: str,
    output_sha256: str,
    converter: dict[str, object],
) -> dict[str, object]:
    recipe = dict(inventory.matrix_recipe_summary(_CANONICAL_RECIPE))
    recipe.update(
        {
            "activation_profile": "compile_selected_adaptive_A8G64",
            "objects": len(inventory.TENSOR_SPECS),
            "source_tensors": len(inventory.SOURCE_NAMES),
        }
    )
    return {
        "identity": _identity_record(checked.output_identity),
        "target_key": inventory.TARGET_KEY,
        "recipe_id": inventory.RECIPE_ID,
        "status": "registered-evaluation-only",
        "weight_recipe_selected": False,
        "base": {
            "path": str(checked.base_path),
            "identity": _identity_record(checked.base_identity),
            "bytes": checked.base_path.stat().st_size,
            "sha256": base_sha256,
            "authority": _base_authority(
                checked.base_path, checked.base_identity, base_sha256
            ),
            "payload_copy": "byte_exact",
        },
        "dflash_source": checked.source,
        "dflash_recipe": recipe,
        "artifact": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": output_sha256,
            "projected_bytes": checked.projected_file_bytes,
            "projected_device_arena_bytes": checked.projected_device_arena_bytes,
        },
        "converter": converter,
    }


def _pending_path(output: Path) -> Path:
    return Path(str(output) + ".conversion.pending.json")


def _report_path(output: Path) -> Path:
    return Path(str(output) + ".conversion.json")


def _load_source_tensor(binding, reader, torch):
    parts = [reader.get(name) for name in binding.sources]
    tensor = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    expected_elements = 1
    for extent in binding.artifact.shape:
        expected_elements *= extent
    if tensor.dtype != torch.bfloat16 or tensor.numel() != expected_elements:
        raise ValueError(
            f"{binding.artifact.name}: expected BF16 with {expected_elements} elements, got "
            f"{tensor.dtype} {tensor.numel()}"
        )
    return tensor


def convert(
    base: str | Path,
    dflash_model: str | Path,
    out: str | Path,
    *,
    device: str,
) -> Path:
    output = Path(out)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite DFlash2 evaluation artifact: {output}; "
            "use --finalize-report to recover a missing report"
        )
    checked = preflight(base, dflash_model)
    started = time.perf_counter()
    base_sha256 = _sha256(checked.base_path)

    # Heavy dependencies are deliberately deferred so inventory/preflight remains available in
    # dependency-light maintainer environments.
    import torch

    from tools.convert.common.quantize import pick_device
    from tools.convert.common.safetensors import ShardReader
    from tools.convert.qwen3.common import conversion as family_conversion

    resolved_device = pick_device(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending_path = _pending_path(output)
    pending = {
        "schema": 1,
        "status": "conversion_started",
        "identity": _identity_record(checked.output_identity),
        "base": {
            "path": str(checked.base_path),
            "identity": _identity_record(checked.base_identity),
            "bytes": checked.base_path.stat().st_size,
            "sha256": base_sha256,
            "authority": _base_authority(
                checked.base_path, checked.base_identity, base_sha256
            ),
        },
        "dflash_source": checked.source,
        "dflash_matrix_recipe": inventory.matrix_recipe_summary(_CANONICAL_RECIPE),
        "converter": {
            "mode": "convert",
            "device_requested": device,
            "device_resolved": str(resolved_device),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "elapsed_seconds": None,
        },
    }
    _atomic_json(pending_path, pending)
    source_path = checked.dflash_model / "model.safetensors"
    with Artifact.open(checked.base_path) as source_artifact:
        with ShardReader.from_file(source_path) as reader:
            with ArtifactWriter(output, checked.output_identity, checked.specs) as writer:
                if writer.objects != checked.objects:
                    raise RuntimeError("DFlash2 writer plan differs from completed preflight")
                for obj in source_artifact.objects:
                    writer.write(obj.name, _chunks(source_artifact.payload(obj)))
                for binding in inventory.source_bindings_for_recipe(_CANONICAL_RECIPE):
                    tensor = _load_source_tensor(binding, reader, torch)
                    if binding.artifact.format == "BF16":
                        payload = family_conversion.encode_tensor_payload(
                            tensor, binding.artifact, resolved_device
                        )
                    else:
                        payload = dflash2_matrix_recipes.encode_matrix_payload(
                            tensor, binding.artifact, _CANONICAL_RECIPE, resolved_device
                        )
                    writer.write(binding.artifact.name, payload)
                    del payload, tensor

    output_sha256 = _validate_completed_artifact(
        output, checked.output_identity, checked.objects, checked.projected_file_bytes
    )
    pending["status"] = "artifact_complete"
    pending["artifact"] = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": output_sha256,
    }
    converter = pending["converter"]
    if not isinstance(converter, dict):
        raise RuntimeError("internal DFlash2 conversion receipt is malformed")
    converter["elapsed_seconds"] = time.perf_counter() - started
    _atomic_json(pending_path, pending)
    report = _report_value(checked, output, base_sha256, output_sha256, converter)
    report_path = _report_path(output)
    _atomic_json(report_path, report)
    pending_path.unlink(missing_ok=True)
    return report_path


def finalize_report(
    base: str | Path,
    dflash_model: str | Path,
    out: str | Path,
    *,
    expected_output_sha256: str | None = None,
) -> Path:
    output = Path(out)
    checked = preflight(base, dflash_model)
    started = time.perf_counter()
    base_sha256 = _sha256(checked.base_path)
    pending_path = _pending_path(output)
    pending: dict[str, object] | None = None
    if pending_path.is_file():
        value = json.loads(pending_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("DFlash2 pending conversion receipt must be a JSON object")
        pending = value
        pending_base = pending.get("base")
        if (
            pending.get("schema") != 1
            or pending.get("identity") != _identity_record(checked.output_identity)
            or not isinstance(pending_base, dict)
            or pending_base.get("identity") != _identity_record(checked.base_identity)
            or pending_base.get("sha256") != base_sha256
            or pending_base.get("authority")
            != _base_authority(checked.base_path, checked.base_identity, base_sha256)
            or pending.get("dflash_source") != checked.source
            or pending.get("dflash_matrix_recipe")
            != inventory.matrix_recipe_summary(_CANONICAL_RECIPE)
        ):
            raise ValueError("DFlash2 pending conversion receipt differs from current preflight")
        pending_artifact = pending.get("artifact")
        if isinstance(pending_artifact, dict) and pending_artifact.get("sha256") is not None:
            receipt_sha256 = pending_artifact.get("sha256")
            if expected_output_sha256 is not None and receipt_sha256 != expected_output_sha256:
                raise ValueError("requested output SHA-256 differs from pending conversion receipt")
            expected_output_sha256 = str(receipt_sha256)

    output_sha256 = _validate_completed_artifact(
        output,
        checked.output_identity,
        checked.objects,
        checked.projected_file_bytes,
        expected_output_sha256,
    )
    if pending is not None and isinstance(pending.get("converter"), dict):
        converter = dict(pending["converter"])
        converter["report_finalized_by_resume"] = True
        converter["report_finalize_seconds"] = time.perf_counter() - started
    else:
        converter = {
            "mode": "finalize_existing",
            "device_requested": "unavailable_without_pending_receipt",
            "device_resolved": "unavailable_without_pending_receipt",
            "torch": "unavailable_without_pending_receipt",
            "torch_hip": "unavailable_without_pending_receipt",
            "elapsed_seconds": None,
            "report_finalized_by_resume": True,
            "report_finalize_seconds": time.perf_counter() - started,
        }
    report = _report_value(checked, output, base_sha256, output_sha256, converter)
    report_path = _report_path(output)
    _atomic_json(report_path, report)
    pending_path.unlink(missing_ok=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--dflash-model", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--finalize-report", action="store_true")
    parser.add_argument("--expected-output-sha256")
    args = parser.parse_args(argv)
    if args.preflight_only:
        checked = preflight(args.base, args.dflash_model)
        print(json.dumps(preflight_summary(checked, args.out), indent=2))
        return
    if args.out is None:
        parser.error("--out is required unless --preflight-only is selected")
    if args.expected_output_sha256 is not None and not args.finalize_report:
        parser.error("--expected-output-sha256 requires --finalize-report")
    if args.finalize_report:
        report = finalize_report(
            args.base,
            args.dflash_model,
            args.out,
            expected_output_sha256=args.expected_output_sha256,
        )
    else:
        report = convert(args.base, args.dflash_model, args.out, device=args.device)
    print(f"DFlash2 Q4 evaluation conversion report: {report}")


if __name__ == "__main__":
    main()
