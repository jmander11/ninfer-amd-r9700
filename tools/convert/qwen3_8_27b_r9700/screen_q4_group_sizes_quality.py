"""CPU-only sampled quality screen for signed Q4 G128/G256/G512 versus Q4G64."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from . import draft_head, q4_inventory, source_recipe
from .q4_row_scaled import error_metrics, quantize_dequantize
from .screen_q4_row_scaled_quality import (
    SampledShardReader, _sample_expression, canonical_json, select_row_indices,
    sha256_file, write_no_clobber,
)


SCHEMA = "ninfer.qwen3_8_27b.q4-group-size-sampled-quality.v1"
GROUPS = (64, 128, 256, 512)
ROWS_PER_TENSOR = 8
WORST_COUNT = 16
PRODUCTION_SHAPE_COUNTS = {
    (7168, 5120): 32, (4096, 5120): 48, (12288, 5120): 48,
    (5120, 6144): 65, (34816, 5120): 65, (5120, 17408): 65,
    (5120, 10240): 1, (1024, 5120): 0,
}
PRODUCTION_TOKENS = 2048
CTA_M = 64
CTA_N = 128


def _codec_metrics(source: torch.Tensor, group: int) -> dict[str, float]:
    return error_metrics(source, quantize_dequantize(source, group_size=group))


def measure(name: str, shape: tuple[int, int], indices: tuple[int, ...],
            source: torch.Tensor) -> dict[str, object]:
    if source.dtype != torch.bfloat16 or tuple(source.shape) != (len(indices), shape[1]):
        raise ValueError(f"{name}: sampled BF16 source has wrong shape or dtype")
    if not bool(torch.isfinite(source).all()):
        raise ValueError(f"{name}: sampled source contains nonfinite values")
    codecs = {f"q4g{group}": _codec_metrics(source, group) for group in GROUPS}
    control = codecs["q4g64"]
    return {
        "name": name, "shape": list(shape),
        "sampled_row_indices": list(indices), "sampled_rows": len(indices),
        "sampled_elements": source.numel(),
        "zero_rows": int((source == 0).all(dim=1).sum().item()),
        "codecs": codecs,
        "candidate_over_q4g64": {
            f"q4g{group}": {
                metric: codecs[f"q4g{group}"][metric] / control[metric]
                if control[metric] else None
                for metric in ("relative_l2", "max_abs")
            }
            for group in GROUPS[1:]
        },
    }


def _aggregate(records: Sequence[Mapping[str, object]], codec: str) -> dict[str, float]:
    metrics = [record["codecs"][codec] for record in records]
    error = sum(float(metric["squared_error"]) for metric in metrics)
    reference = sum(float(metric["squared_reference"]) for metric in metrics)
    return {
        "relative_l2": math.sqrt(error / reference) if reference else 0.0,
        "max_abs": max(float(metric["max_abs"]) for metric in metrics),
        "squared_error": error, "squared_reference": reference,
    }


def _persistent_weight_scale_bytes(group: int) -> int:
    total = 0
    for spec in q4_inventory.TENSOR_SPECS:
        if spec.format != "Q4G64_F16S":
            continue
        rows, columns = spec.shape
        alignment = math.lcm(128, group)
        padded = ((columns + alignment - 1) // alignment) * alignment
        total += rows * (padded // group) * 2
    return total


def _production_model(group: int) -> dict[str, int | float]:
    scale_bytes = 0
    accumulations = 0
    for (rows, columns), count in PRODUCTION_SHAPE_COUNTS.items():
        groups = columns // group
        blocks_m = (PRODUCTION_TOKENS + CTA_M - 1) // CTA_M
        blocks_n = (rows + CTA_N - 1) // CTA_N
        # Every CTA stages one A scale/token and one W scale/row per K group.
        scale_bytes += count * groups * 2 * (
            blocks_n * PRODUCTION_TOKENS + blocks_m * rows)
        accumulations += count * PRODUCTION_TOKENS * rows * groups
    conservative_i32 = group * (15 * 8 + 16 * 8 * 8)
    direct_i32 = group * 127 * 8
    return {
        "scale_request_bytes": scale_bytes,
        "fp32_group_accumulations": accumulations,
        "direct_signed_dot_max_abs": direct_i32,
        "conservative_split_low_plus_16_high_max_abs": conservative_i32,
        "int32_limit": 2**31 - 1,
    }


def assemble_report(*, records: Sequence[dict[str, object]], source: Mapping[str, object],
                    implementation: Mapping[str, str]) -> dict[str, object]:
    ordered = sorted(records, key=lambda item: str(item["name"]))
    aggregates = {f"q4g{group}": _aggregate(ordered, f"q4g{group}")
                  for group in GROUPS}
    control = aggregates["q4g64"]
    comparisons = {}
    for group in GROUPS[1:]:
        codec = f"q4g{group}"
        candidates = [item for item in ordered]
        comparisons[codec] = {
            "aggregate_candidate_over_q4g64": {
                metric: aggregates[codec][metric] / control[metric]
                for metric in ("relative_l2", "max_abs")
            },
            "matrices_with_higher_relative_l2": sum(
                item["candidate_over_q4g64"][codec]["relative_l2"] > 1
                for item in candidates),
            "worst_relative_l2_ratios": [
                {"name": item["name"],
                 "ratio": item["candidate_over_q4g64"][codec]["relative_l2"],
                 "q4g64": item["codecs"]["q4g64"]["relative_l2"],
                 "candidate": item["codecs"][codec]["relative_l2"]}
                for item in sorted(
                    candidates,
                    key=lambda item: (-item["candidate_over_q4g64"][codec]["relative_l2"],
                                      item["name"]),
                )[:WORST_COUNT]
            ],
        }
    baseline_model = _production_model(64)
    models = {}
    for group in GROUPS:
        model = _production_model(group)
        model["scale_request_fraction_of_q4g64"] = (
            model["scale_request_bytes"] / baseline_model["scale_request_bytes"])
        model["fp32_accumulation_fraction_of_q4g64"] = (
            model["fp32_group_accumulations"] /
            baseline_model["fp32_group_accumulations"])
        model["persistent_weight_scale_bytes_439_matrices"] = (
            _persistent_weight_scale_bytes(group))
        models[f"q4g{group}"] = model
    return {
        "schema": SCHEMA,
        "status": "diagnostic_only_no_artifact_or_runtime_change",
        "sampling": {"algorithm": "endpoints-plus-sha256-counter-v1",
                     "rows_per_tensor": ROWS_PER_TENSOR,
                     "candidate_tensor_count": len(ordered)},
        "aggregate": {"sampled_rows": sum(item["sampled_rows"] for item in ordered),
                      "sampled_elements": sum(item["sampled_elements"] for item in ordered),
                      "codecs": aggregates, "comparisons": comparisons},
        "production_p2048_model": {
            "tokens": PRODUCTION_TOKENS, "tile": "M64xN128",
            "shape_call_counts": [{"rows": key[0], "columns": key[1], "count": value}
                                  for key, value in PRODUCTION_SHAPE_COUNTS.items()],
            "groups": models,
        },
        "source": dict(source),
        "implementation_sha256": dict(sorted(implementation.items())),
        "tensors": ordered,
    }


def run_screen(model_dir: Path, ranking: Path, output: Path) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite quality report: {output}")
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    weight_map = dict(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"])
    shards = sorted(set(weight_map.values()))
    draft_ids = draft_head.compute_shortlist(ranking, model_dir).selected.tolist()
    candidates = [spec for spec in q4_inventory.TENSOR_SPECS
                  if spec.format == "Q4G64_F16S"]
    if len(candidates) != 439:
        raise ValueError(f"expected 439 Q4 candidates, got {len(candidates)}")
    records = []
    with SampledShardReader(model_dir, weight_map) as reader:
        for ordinal, spec in enumerate(candidates, 1):
            indices = select_row_indices(spec.name, spec.shape[0], ROWS_PER_TENSOR)
            source = _sample_expression(source_recipe.RECIPES_BY_NAME[spec.name].expression,
                                        indices, reader, draft_ids)
            records.append(measure(spec.name, spec.shape, indices, source))
            if ordinal % 50 == 0 or ordinal == len(candidates):
                print(f"sampled {ordinal}/{len(candidates)} matrices", flush=True)
    root = Path(__file__).resolve().parents[3]
    paths = (Path(__file__).resolve(),
             root / "tools/convert/qwen3_8_27b_r9700/q4_row_scaled.py",
             root / "tools/convert/qwen3_8_27b_r9700/screen_q4_row_scaled_quality.py",
             root / "tools/convert/qwen3_8_27b_r9700/q4_inventory.py",
             root / "tools/convert/qwen3_8_27b_r9700/source_recipe.py")
    report = assemble_report(
        records=records,
        source={"model_dir": str(model_dir.resolve()),
                "config": {"path": config_path.name, "sha256": sha256_file(config_path)},
                "index": {"path": index_path.name, "sha256": sha256_file(index_path)},
                "ranking": {"path": str(ranking.resolve()), "sha256": sha256_file(ranking)},
                "shards": [{"path": shard, "bytes": (model_dir / shard).stat().st_size,
                            "sha256": sha256_file(model_dir / shard)} for shard in shards]},
        implementation={str(path.relative_to(root)): sha256_file(path) for path in paths},
    )
    write_no_clobber(output, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_screen(args.model_dir, args.ranking, args.output)
    print(json.dumps({"aggregate": report["aggregate"],
                      "production_p2048_model": report["production_p2048_model"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
