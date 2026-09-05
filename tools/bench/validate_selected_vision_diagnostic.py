#!/usr/bin/env python3
"""Validate and publish the selected-artifact Vision diagnostic completion."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

from tools.bench.prepare_selected_vision_diagnostic import (
    FRONTEND_PYTHON,
    GPU_PYTHON,
    MEDIA,
    MESSAGES,
    REPO,
    REFERENCE_AUTHORITIES,
    SOURCE,
    file_identity,
    inspect_python,
    resolve_route,
    sha,
    validate_source_receipt,
)


CAPTURES = ("block_00", "block_13", "block_26", "merger")


def validate_closure(root: Path, plan_path: Path) -> dict[Path, str]:
    closure_path = root / "prepared.sha256"
    if closure_path.is_symlink() or not closure_path.is_file():
        raise ValueError("prepared closure must be a regular non-symlink file")
    entries: dict[Path, str] = {}
    for line in closure_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError("prepared closure is malformed")
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("prepared closure path is invalid")
        candidate = REPO / relative
        if candidate.is_symlink():
            raise ValueError("prepared closure authority must not be a symlink")
        path = candidate.resolve(strict=True)
        if path in entries:
            raise ValueError("prepared closure contains a duplicate path")
        entries[path] = parts[0]
    for path, digest in entries.items():
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"prepared closure bytes changed: {path}")
    required = {
        plan_path.resolve(), Path(__file__).resolve(),
        *[path.resolve(strict=True) for path in REFERENCE_AUTHORITIES],
    }
    if not required <= set(entries):
        raise ValueError("prepared closure lacks selected Vision authorities")
    return entries


def _expected_source() -> dict[str, object]:
    identity = validate_source_receipt()
    receipt = json.loads(Path(identity["receipt"]["path"]).read_text(encoding="utf-8"))
    return {
        "path": str(SOURCE.resolve()),
        "config_sha256": receipt["metadata"]["config"]["sha256"],
        "index_sha256": receipt["metadata"]["index"]["sha256"],
        "indexed_tensor_count": receipt["metadata"]["indexed_tensor_count"],
        "shards": {
            row["name"]: {"bytes": row["bytes"]} for row in receipt["shards"]["files"]
        },
    }


def validate(plan_path: Path, root: Path, *, route_resolver=resolve_route) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Vision campaign root must be a regular directory namespace")
    root = root.resolve(strict=True)
    if plan_path.is_symlink():
        raise ValueError("Vision plan must not be a symlink")
    plan_path = plan_path.resolve(strict=True)
    if plan_path != root / "plan.json":
        raise ValueError("Vision plan must belong to the supplied campaign root")
    closure = validate_closure(root, plan_path)
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    route = plan.get("terminal_route")
    if not isinstance(route, dict):
        raise ValueError("Vision plan lacks terminal route")
    recomputed = route_resolver(Path(route["terminal_selection"]["path"]))
    if recomputed != route:
        raise ValueError("Vision route differs from current schema-v7 winner")
    source_identity = validate_source_receipt()
    if plan.get("source_checkpoint") != source_identity:
        raise ValueError("Vision plan BF16 source receipt changed")
    if plan.get("frontend_python") != inspect_python(
        FRONTEND_PYTHON, ("torch", "safetensors", "transformers", "torchvision", "PIL")
    ) or plan.get("gpu_python") != inspect_python(
        GPU_PYTHON, ("torch", "safetensors", "transformers")
    ):
        raise ValueError("Vision Python execution environment changed")
    fixture = plan.get("fixture", {})
    if fixture != {"messages": file_identity(MESSAGES), "media": file_identity(MEDIA)}:
        raise ValueError("Vision committed image fixture changed")
    prepared = plan.get("prepared_input", {})
    prepared_name = Path(prepared.get("path", ""))
    if prepared_name.is_symlink():
        raise ValueError("Vision prepared input must not be a symlink")
    prepared_path = prepared_name.resolve(strict=True)
    if prepared_path != root / "prepared-input.safetensors" or file_identity(prepared_path) != {
        key: prepared[key] for key in ("path", "bytes", "sha256")
    } or prepared_path not in closure:
        raise ValueError("Vision prepared input differs from frozen plan")
    contract = prepared.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("thinking") is not False
        or contract.get("images") != 1
        or contract.get("videos") != 0
        or contract.get("video_tokens") != 0
        or contract.get("image_tokens", 0) <= 0
        or contract.get("prompt_length", 0) <= contract.get("image_tokens", 0)
        or not isinstance(contract.get("image_grid_thw"), list)
        or len(contract["image_grid_thw"]) != 1
    ):
        raise ValueError("Vision prepared input contract is not one committed image")
    grid = contract["image_grid_thw"][0]
    if not isinstance(grid, list) or len(grid) != 3 or any(
        not isinstance(value, int) or value <= 0 for value in grid
    ) or grid[1] % 2 or grid[2] % 2:
        raise ValueError("Vision prepared input grid is invalid")
    patches = math.prod(grid)
    tokens = patches // 4
    if contract["image_tokens"] != tokens or contract.get("pixel_values_shape") != [patches, 1536]:
        raise ValueError("Vision prepared input shapes differ from grid semantics")
    expected_workload = {
        "maximum_concurrency": 1, "thinking": False, "prefix_reuse": False,
        "speculative_decode": False, "images": 1, "videos": 0,
        "capture_names": list(CAPTURES),
        "gate": "diagnostic completion with finite exact-shape comparisons; no numeric threshold",
    }
    expected_outputs = {
        "raw": str(root / "vision.raw.json"), "admission": str(root / "admission.json")
    }
    if (
        plan.get("artifact_type") != "ninfer_r9700_selected_vision_diagnostic_plan"
        or plan.get("schema_version") != 1
        or plan.get("status") != "command_only_not_executed"
        or plan.get("workload") != expected_workload
        or plan.get("outputs") != expected_outputs
        or route.get("maximum_runtime_concurrency") != 4
    ):
        raise ValueError("Vision plan contract differs")
    raw_name = Path(plan["outputs"]["raw"])
    if raw_name.is_symlink():
        raise ValueError("Vision raw report namespace is invalid")
    raw_path = raw_name.resolve(strict=True)
    if raw_path.parent != root:
        raise ValueError("Vision raw report namespace is invalid")
    raw_bytes = raw_path.read_bytes()
    raw = json.loads(raw_bytes)
    artifact = raw.get("artifact", {})
    raw_input = raw.get("input", {})
    execution = raw.get("execution")
    if (
        raw.get("format") != "ninfer_vision_bf16_comparison_v3"
        or raw.get("scope") != "diagnostic-only"
        or artifact.get("path") != route["artifact"]["path"]
        or artifact.get("sha256") != route["artifact"]["sha256"]
        or artifact.get("identity") != {
            "model_id": route["artifact"]["model_id"],
            "weights_id": route["artifact"]["weights_id"],
        }
        or raw.get("source") != _expected_source()
        or raw_input.get("messages_path") != fixture["messages"]["path"]
        or raw_input.get("messages_sha256") != fixture["messages"]["sha256"]
        or raw_input.get("thinking") is not False
        or raw_input.get("prepared_input") != {
            "path": prepared["path"], "sha256": prepared["sha256"], "contract": contract
        }
        or execution != {
            "maximum_concurrency": 1, "thinking": False, "prefix_reuse": False,
            "speculative_decode": False,
        }
        or raw.get("image_grid_thw") != [grid]
        or raw.get("video_grid_thw") is not None
    ):
        raise ValueError("Vision raw identity or execution contract differs")
    stats = raw.get("vision", {})
    if stats != {
        "images": 1, "videos": 0, "raw_patches": patches,
        "llm_tokens": tokens, "attention_pairs": grid[0] * (grid[1] * grid[2]) ** 2,
    }:
        raise ValueError("Vision raw geometry differs from prepared input")
    comparisons = raw.get("comparisons")
    if not isinstance(comparisons, dict) or list(comparisons) != list(CAPTURES):
        raise ValueError("Vision raw report lacks exact capture inventory")
    for name, row in comparisons.items():
        expected_shape = [tokens, 5120] if name == "merger" else [patches, 1152]
        if not isinstance(row, dict) or row.get("shape") != expected_shape:
            raise ValueError(f"Vision capture {name} has wrong shape")
        numeric = [row.get(key) for key in (
            "rmse", "relative_rmse", "cosine", "actual_norm", "reference_norm"
        )]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
            raise ValueError(f"Vision capture {name} has nonfinite metrics")
        if row["rmse"] < 0 or row["relative_rmse"] < 0 or row["actual_norm"] <= 0 \
                or row["reference_norm"] <= 0 or not -1.000001 <= row["cosine"] <= 1.000001:
            raise ValueError(f"Vision capture {name} has invalid metrics")
    if sha(plan_path) != hashlib_sha(plan_raw) or sha(raw_path) != hashlib_sha(raw_bytes):
        raise ValueError("Vision evidence changed while validating")
    return {
        "artifact_type": "ninfer_r9700_selected_vision_diagnostic_completion",
        "schema_version": 1,
        "status": "complete_diagnostic_no_numeric_threshold",
        "terminal_route": route,
        "campaign": {
            "plan": {"path": str(plan_path), "sha256": sha(plan_path)},
            "prepared_closure": {"path": str(root / "prepared.sha256"),
                                 "sha256": sha(root / "prepared.sha256")},
        },
        "evidence": {"path": str(raw_path), "sha256": sha(raw_path)},
        "source_checkpoint": source_identity,
        "gate": "finite_exact_shape_diagnostic_completed_without_numeric_threshold",
    }


def hashlib_sha(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def publish(plan: Path, root: Path, output: Path) -> dict[str, object]:
    if os.path.lexists(output):
        raise ValueError(f"refusing to overwrite {output}")
    result = validate(plan, root)
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    pending: Path | None = None
    owner: tuple[int, int] | None = None
    published = False
    revalidated = False
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.pending-", dir=parent)
        pending = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            stat = os.fstat(stream.fileno())
            owner = (stat.st_dev, stat.st_ino)
        os.link(pending, output)
        published = True
        target = os.stat(output, follow_symlinks=False)
        if (target.st_dev, target.st_ino) != owner:
            raise ValueError("Vision admission hard link changed inode")
        if validate(plan, root) != result:
            raise ValueError("published Vision completion does not revalidate")
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("published Vision completion differs")
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        revalidated = True
        return result
    finally:
        for path in (pending, output if published and not revalidated else None):
            if path is None or owner is None:
                continue
            try:
                current = os.stat(path, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == owner:
                    path.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        publish(args.plan, args.root, args.out)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
