"""Deterministic CPU-only Q4G64 versus row-scaled-Q4 BF16 weight screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors import safe_open

from tools.convert.qwen3.common.recipe import (
    Concat, Expression, GatherRows, Reshape, Slice, SourceTensor, expression_shape,
)

from . import draft_head, q4_inventory, source_recipe
from .q4_row_scaled import compare_sample


SCHEMA = "ninfer.qwen3_8_27b.q4-row-scaled-sampled-quality.v1"
SELECTION_ALGORITHM = "endpoints-plus-sha256-counter-v1"
DEFAULT_ROWS_PER_TENSOR = 8
DEFAULT_WORST_COUNT = 16


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def select_row_indices(name: str, rows: int, count: int) -> tuple[int, ...]:
    if rows <= 0 or count <= 0:
        raise ValueError("row selection requires positive rows and count")
    wanted = min(rows, count)
    selected = {0}
    if wanted > 1:
        selected.add(rows - 1)
    seed = hashlib.sha256(
        f"{SELECTION_ALGORITHM}\0{name}\0{rows}".encode("utf-8")
    ).digest()
    counter = 0
    while len(selected) < wanted:
        digest = hashlib.sha256(seed + counter.to_bytes(8, "little")).digest()
        selected.add(int.from_bytes(digest[:8], "little") % rows)
        counter += 1
    return tuple(sorted(selected))


class SampledShardReader:
    def __init__(self, model_dir: Path, weight_map: Mapping[str, str]) -> None:
        self.model_dir = model_dir
        self.weight_map = dict(weight_map)
        self._shard: str | None = None
        self._context = None
        self._handle = None

    def _open(self, shard: str):
        if shard == self._shard:
            return self._handle
        self.close()
        self._context = safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
        self._handle = self._context.__enter__()
        self._shard = shard
        return self._handle

    def rows(self, source: SourceTensor, indices: Sequence[int]) -> torch.Tensor:
        if source.name not in self.weight_map:
            raise ValueError(f"source tensor is absent from index: {source.name}")
        tensor_slice = self._open(self.weight_map[source.name]).get_slice(source.name)
        if tuple(tensor_slice.get_shape()) != source.shape:
            raise ValueError(f"source shape differs for {source.name}")
        return torch.cat(
            [tensor_slice[index:index + 1] for index in indices], dim=0
        ).reshape(len(indices), -1).contiguous()

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
        self._shard = None
        self._context = None
        self._handle = None

    def __enter__(self) -> "SampledShardReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _sample_expression(expression: Expression, indices: Sequence[int],
                       reader: SampledShardReader,
                       draft_ids: Sequence[int]) -> torch.Tensor:
    if isinstance(expression, SourceTensor):
        return reader.rows(expression, indices)
    if isinstance(expression, GatherRows):
        return reader.rows(expression.source, [int(draft_ids[index]) for index in indices])
    if isinstance(expression, Slice) and expression.axis == 0:
        return _sample_expression(
            expression.source, [expression.begin + index for index in indices],
            reader, draft_ids,
        )
    if isinstance(expression, Concat) and expression.axis == 0:
        shapes = [expression_shape(part) for part in expression.sources]
        rows = []
        for index in indices:
            offset = 0
            for part, shape in zip(expression.sources, shapes):
                if index < offset + shape[0]:
                    rows.append(_sample_expression(
                        part, [index - offset], reader, draft_ids))
                    break
                offset += shape[0]
            else:
                raise IndexError(f"concat row {index} is out of range")
        return torch.cat(rows, dim=0)
    if isinstance(expression, Reshape):
        output_shape = expression_shape(expression)
        source_shape = expression_shape(expression.source)
        if len(output_shape) == 2 and output_shape[0] == source_shape[0]:
            return _sample_expression(
                expression.source, indices, reader, draft_ids
            ).reshape(len(indices), output_shape[1])
        sliced = expression.source
        if (len(output_shape) == 2 and isinstance(sliced, Slice) and sliced.axis == 1
                and isinstance(sliced.source, Reshape)
                and isinstance(sliced.source.source, SourceTensor)):
            inner_shape = sliced.source.shape
            width = sliced.end - sliced.begin
            if output_shape[0] == inner_shape[0] * width:
                physical = [(index // width) * inner_shape[1] + sliced.begin + index % width
                            for index in indices]
                return reader.rows(sliced.source.source, physical)
    raise TypeError(
        f"row sampler does not admit {type(expression).__name__} "
        f"with shape {expression_shape(expression)}")


def measure(name: str, shape: tuple[int, int], indices: tuple[int, ...],
            source: torch.Tensor) -> dict[str, object]:
    if source.dtype != torch.bfloat16 or tuple(source.shape) != (len(indices), shape[1]):
        raise ValueError(f"{name}: sampled BF16 source has wrong shape or dtype")
    nonfinite = int((~torch.isfinite(source)).sum().item())
    if nonfinite:
        raise ValueError(f"{name}: sampled source contains {nonfinite} nonfinite values")
    comparison = compare_sample(source)
    return {
        "name": name,
        "shape": list(shape),
        "sampled_row_indices": list(indices),
        "sampled_rows": len(indices),
        "sampled_elements": source.numel(),
        "zero_rows": int((source == 0).all(dim=1).sum().item()),
        **comparison,
    }


def _aggregate(tensors: Sequence[Mapping[str, object]], key: str) -> dict[str, float]:
    metrics = [item[key] for item in tensors]
    squared_error = sum(float(item["squared_error"]) for item in metrics)
    squared_reference = sum(float(item["squared_reference"]) for item in metrics)
    return {
        "relative_l2": math.sqrt(squared_error / squared_reference)
        if squared_reference else 0.0,
        "max_abs": max((float(item["max_abs"]) for item in metrics), default=0.0),
        "squared_error": squared_error,
        "squared_reference": squared_reference,
    }


def assemble_report(*, rows_per_tensor: int, tensors: Sequence[dict[str, object]],
                    source: Mapping[str, object], implementation: Mapping[str, str],
                    worst_count: int = DEFAULT_WORST_COUNT) -> dict[str, object]:
    ordered = sorted(tensors, key=lambda item: str(item["name"]))
    grouped = _aggregate(ordered, "q4g64")
    row = _aggregate(ordered, "q4_row_scaled")
    ratios = {
        "relative_l2": row["relative_l2"] / grouped["relative_l2"],
        "max_abs": row["max_abs"] / grouped["max_abs"],
    }
    by_ratio = sorted(
        ordered,
        key=lambda item: (-(float(item["ratios"]["relative_l2"])
                            if item["ratios"]["relative_l2"] is not None else -1.0),
                          str(item["name"])),
    )[:worst_count]
    return {
        "schema": SCHEMA,
        "status": "diagnostic_only_no_artifact_or_runtime_change",
        "formats": {
            "control": "signed-Q4G64-FP16-scale",
            "candidate": "signed-Q4-row-FP16-scale",
            "rounding": "FP16 scale; round-to-nearest-even codes; clamp[-8,7]",
        },
        "sampling": {
            "algorithm": SELECTION_ALGORITHM,
            "rows_per_tensor": rows_per_tensor,
            "candidate_tensor_count": len(ordered),
        },
        "aggregate": {
            "sampled_rows": sum(int(item["sampled_rows"]) for item in ordered),
            "sampled_elements": sum(int(item["sampled_elements"]) for item in ordered),
            "zero_rows": sum(int(item["zero_rows"]) for item in ordered),
            "q4g64": grouped,
            "q4_row_scaled": row,
            "candidate_over_control": ratios,
        },
        "worst_candidate_over_control_relative_l2": [
            {"name": item["name"], "ratio": item["ratios"]["relative_l2"],
             "q4g64": item["q4g64"]["relative_l2"],
             "q4_row_scaled": item["q4_row_scaled"]["relative_l2"]}
            for item in by_ratio
        ],
        "source": dict(source),
        "implementation_sha256": dict(sorted(implementation.items())),
        "tensors": ordered,
    }


def canonical_json(report: Mapping[str, object]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def run_screen(model_dir: Path, ranking: Path, output: Path, *, rows_per_tensor: int,
               worst_count: int) -> dict[str, object]:
    if rows_per_tensor <= 0 or worst_count <= 0:
        raise ValueError("rows-per-tensor and worst-count must be positive")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite quality report: {output}")
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = dict(index["weight_map"])
    shard_names = sorted(set(weight_map.values()))
    draft_ids = draft_head.compute_shortlist(ranking, model_dir).selected.tolist()
    candidates = [spec for spec in q4_inventory.TENSOR_SPECS
                  if spec.format == "Q4G64_F16S"]
    if len(candidates) != 439:
        raise ValueError(f"expected 439 Q4 candidates, got {len(candidates)}")
    records = []
    with SampledShardReader(model_dir, weight_map) as reader:
        for ordinal, spec in enumerate(candidates, 1):
            indices = select_row_indices(spec.name, spec.shape[0], rows_per_tensor)
            sampled = _sample_expression(
                source_recipe.RECIPES_BY_NAME[spec.name].expression,
                indices, reader, draft_ids,
            )
            records.append(measure(spec.name, spec.shape, indices, sampled))
            if ordinal % 50 == 0 or ordinal == len(candidates):
                print(f"sampled {ordinal}/{len(candidates)} matrices", flush=True)
    root = Path(__file__).resolve().parents[3]
    implementation_paths = (
        Path(__file__).resolve(),
        root / "tools/convert/qwen3_8_27b_r9700/q4_row_scaled.py",
        root / "tools/convert/qwen3_8_27b_r9700/q4_inventory.py",
        root / "tools/convert/qwen3_8_27b_r9700/source_recipe.py",
    )
    report = assemble_report(
        rows_per_tensor=rows_per_tensor,
        tensors=records,
        source={
            "model_dir": str(model_dir.resolve()),
            "config": {"path": config_path.name, "sha256": sha256_file(config_path)},
            "index": {"path": index_path.name, "sha256": sha256_file(index_path)},
            "ranking": {"path": str(ranking.resolve()), "sha256": sha256_file(ranking)},
            "shards": [{"path": shard, "bytes": (model_dir / shard).stat().st_size,
                        "sha256": sha256_file(model_dir / shard)}
                       for shard in shard_names],
        },
        implementation={str(path.relative_to(root)): sha256_file(path)
                        for path in implementation_paths},
        worst_count=worst_count,
    )
    write_no_clobber(output, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-tensor", type=int, default=DEFAULT_ROWS_PER_TENSOR)
    parser.add_argument("--worst-count", type=int, default=DEFAULT_WORST_COUNT)
    args = parser.parse_args()
    report = run_screen(args.model_dir, args.ranking, args.output,
                        rows_per_tensor=args.rows_per_tensor,
                        worst_count=args.worst_count)
    print(json.dumps(report["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
