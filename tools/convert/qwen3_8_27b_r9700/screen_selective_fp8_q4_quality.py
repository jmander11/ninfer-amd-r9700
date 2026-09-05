"""CPU-only sampled E4M3 versus Q4G64 screen for the selected FP8 role set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from . import draft_head, e4m3_inventory, source_recipe
from .e4m3_rowwise import decode_e4m3_rowwise, encode_e4m3_rowwise
from .q4_row_scaled import error_metrics, quantize_dequantize
from .screen_e4m3_quality import (
    DEFAULT_ROWS_PER_TENSOR,
    DEFAULT_WORST_COUNT,
    SELECTION_ALGORITHM,
    SampledShardReader,
    _sample_expression,
    canonical_json,
    select_row_indices,
    sha256_file,
    write_no_clobber,
)


SCHEMA = "ninfer.qwen3_8_27b.selective-e4m3-vs-q4g64-sampled-quality.v1"
SELECTED_ROLES = (
    "text.attention.gate_value",
    "text.attention.query_key",
    "text.gdn.query_key",
    "text.mlp.gate_up",
)


def role_of(name: str) -> str:
    parts = name.split("/")
    if len(parts) != 5 or parts[:2] != ["text", "layers"]:
        raise ValueError(f"selected tensor has no admitted Text role: {name}")
    return f"text.{parts[3]}.{parts[4]}"


def selected_specs():
    specs = [
        spec for spec in e4m3_inventory.TENSOR_SPECS
        if spec.format == e4m3_inventory.F8E4M3_ROW_F32S
        and spec.name.startswith("text/layers/")
        and role_of(spec.name) in SELECTED_ROLES
    ]
    counts = {role: sum(role_of(spec.name) == role for spec in specs) for role in SELECTED_ROLES}
    expected = {
        "text.attention.gate_value": 16,
        "text.attention.query_key": 16,
        "text.gdn.query_key": 48,
        "text.mlp.gate_up": 64,
    }
    if counts != expected or len(specs) != 144:
        raise ValueError(f"selected FP8 inventory differs: {counts}")
    return sorted(specs, key=lambda spec: spec.name)


def measure(
    name: str, shape: tuple[int, int], indices: tuple[int, ...], source: torch.Tensor
) -> dict[str, object]:
    if source.dtype != torch.bfloat16 or tuple(source.shape) != (len(indices), shape[1]):
        raise ValueError(f"{name}: sampled BF16 source has wrong shape or dtype")
    nonfinite = int((~torch.isfinite(source)).sum().item())
    base = {
        "name": name,
        "role": role_of(name),
        "shape": list(shape),
        "sampled_row_indices": list(indices),
        "sampled_rows": len(indices),
        "sampled_elements": source.numel(),
        "nonfinite_values": nonfinite,
        "zero_rows": int((source == 0).all(dim=1).sum().item()),
    }
    if nonfinite:
        return {**base, "e4m3": None, "q4g64": None, "e4m3_over_q4g64": None}
    e4m3 = error_metrics(
        source, decode_e4m3_rowwise(encode_e4m3_rowwise(source), len(indices), shape[1])
    )
    q4 = error_metrics(source, quantize_dequantize(source, group_size=64))
    ratios = {
        metric: e4m3[metric] / q4[metric] if q4[metric] else None
        for metric in ("relative_l2", "max_abs")
    }
    return {**base, "e4m3": e4m3, "q4g64": q4, "e4m3_over_q4g64": ratios}


def _aggregate(tensors: Sequence[Mapping[str, object]], codec: str) -> dict[str, float]:
    metrics = [item[codec] for item in tensors if item[codec] is not None]
    squared_error = sum(float(item["squared_error"]) for item in metrics)
    squared_reference = sum(float(item["squared_reference"]) for item in metrics)
    return {
        "relative_l2": math.sqrt(squared_error / squared_reference) if squared_reference else 0.0,
        "max_abs": max((float(item["max_abs"]) for item in metrics), default=0.0),
        "squared_error": squared_error,
        "squared_reference": squared_reference,
    }


def _summary(tensors: Sequence[Mapping[str, object]]) -> dict[str, object]:
    e4m3 = _aggregate(tensors, "e4m3")
    q4 = _aggregate(tensors, "q4g64")
    return {
        "object_count": len(tensors),
        "sampled_rows": sum(int(item["sampled_rows"]) for item in tensors),
        "sampled_elements": sum(int(item["sampled_elements"]) for item in tensors),
        "nonfinite_values": sum(int(item["nonfinite_values"]) for item in tensors),
        "zero_rows": sum(int(item["zero_rows"]) for item in tensors),
        "e4m3": e4m3,
        "q4g64": q4,
        "e4m3_over_q4g64": {
            metric: e4m3[metric] / q4[metric] if q4[metric] else None
            for metric in ("relative_l2", "max_abs")
        },
    }


def assemble_report(
    *, rows_per_tensor: int, tensors: Sequence[dict[str, object]], source: Mapping[str, object],
    implementation: Mapping[str, str], all_matrix_reference: Mapping[str, object],
    worst_count: int = DEFAULT_WORST_COUNT,
) -> dict[str, object]:
    ordered = sorted(tensors, key=lambda item: str(item["name"]))
    if len(ordered) != 144 or {str(item["role"]) for item in ordered} != set(SELECTED_ROLES):
        raise ValueError("quality records differ from the exact 144-object selected role set")
    by_role = {
        role: _summary([item for item in ordered if item["role"] == role])
        for role in SELECTED_ROLES
    }
    finite = [item for item in ordered if item["e4m3"] is not None]
    def ranked(metric: str, reverse: bool) -> list[Mapping[str, object]]:
        return sorted(
            finite,
            key=lambda item: (
                -float(item["e4m3_over_q4g64"][metric])
                if reverse else float(item["e4m3_over_q4g64"][metric]),
                str(item["name"]),
            ),
        )[:worst_count]
    aggregate = _summary(ordered)
    full_tensors = all_matrix_reference.get("tensors")
    if not isinstance(full_tensors, list):
        raise ValueError("all-matrix E4M3 reference lacks tensor records")
    full_by_name = {str(item["name"]): item for item in full_tensors}
    for item in finite:
        prior = full_by_name.get(str(item["name"]))
        if (
            not isinstance(prior, Mapping)
            or prior.get("sampled_row_indices") != item["sampled_row_indices"]
            or not math.isclose(float(prior["relative_l2"]), float(item["e4m3"]["relative_l2"]), rel_tol=1e-12)
            or not math.isclose(float(prior["max_abs"]), float(item["e4m3"]["max_abs"]), rel_tol=1e-12)
        ):
            raise ValueError(f"{item['name']}: deterministic E4M3 result differs from all-matrix reference")
    full_finite = [item for item in full_tensors if item.get("relative_l2") is not None]
    global_rel_worst = max(full_finite, key=lambda item: (float(item["relative_l2"]), str(item["name"])))
    global_abs_worst = max(full_finite, key=lambda item: (float(item["max_abs"]), str(item["name"])))
    selected_names = {str(item["name"]) for item in ordered}
    full_aggregate = all_matrix_reference["aggregate"]
    concentration = {
        "selected_e4m3_relative_l2_over_all_matrix": (
            aggregate["e4m3"]["relative_l2"] / float(full_aggregate["relative_l2"])
        ),
        "selected_e4m3_max_abs_over_all_matrix": (
            aggregate["e4m3"]["max_abs"] / float(full_aggregate["max_abs"])
        ),
        "contains_global_e4m3_relative_l2_worst": global_rel_worst["name"] in selected_names,
        "contains_global_e4m3_max_abs_worst": global_abs_worst["name"] in selected_names,
        "global_e4m3_relative_l2_worst": {
            "name": global_rel_worst["name"], "value": global_rel_worst["relative_l2"]
        },
        "global_e4m3_max_abs_worst": {
            "name": global_abs_worst["name"], "value": global_abs_worst["max_abs"]
        },
    }
    compact = lambda item, metric: {
        "name": item["name"], "role": item["role"],
        "metric": metric,
        "e4m3": item["e4m3"][metric],
        "q4g64": item["q4g64"][metric],
        "ratio": item["e4m3_over_q4g64"][metric],
    }
    return {
        "schema": SCHEMA,
        "status": "source_only_diagnostic_no_artifact_or_runtime_change",
        "selection": {"roles": list(SELECTED_ROLES), "object_count": 144},
        "sampling": {"algorithm": SELECTION_ALGORITHM, "rows_per_tensor": rows_per_tensor},
        "formats": {
            "candidate": "rowwise-E4M3-FP32-scale",
            "control": "signed-Q4G64-FP16-scale",
        },
        "aggregate": aggregate,
        "per_role": by_role,
        "comparative_object_counts": {
            metric: {
                "e4m3_better": sum(float(item["e4m3_over_q4g64"][metric]) < 1.0 for item in finite),
                "equal": sum(float(item["e4m3_over_q4g64"][metric]) == 1.0 for item in finite),
                "e4m3_worse": sum(float(item["e4m3_over_q4g64"][metric]) > 1.0 for item in finite),
            }
            for metric in ("relative_l2", "max_abs")
        },
        "worst_e4m3_over_q4g64": {
            metric: [compact(item, metric) for item in ranked(metric, True)]
            for metric in ("relative_l2", "max_abs")
        },
        "best_e4m3_over_q4g64": {
            metric: [compact(item, metric) for item in ranked(metric, False)]
            for metric in ("relative_l2", "max_abs")
        },
        "quality_concentration": concentration,
        "source": dict(source),
        "all_matrix_e4m3_reference": dict(all_matrix_reference["identity"]),
        "implementation_sha256": dict(sorted(implementation.items())),
        "tensors": ordered,
    }


def run_screen(
    model_dir: Path, ranking: Path, all_matrix_report_path: Path, output: Path,
    *, rows_per_tensor: int, worst_count: int,
) -> dict[str, object]:
    if rows_per_tensor <= 0 or worst_count <= 0:
        raise ValueError("rows-per-tensor and worst-count must be positive")
    all_matrix_report = json.loads(all_matrix_report_path.read_text(encoding="utf-8"))
    if (
        all_matrix_report.get("schema") != "ninfer.qwen3_8_27b.e4m3-rowwise-sampled-quality.v1"
        or all_matrix_report.get("sampling", {}).get("algorithm") != SELECTION_ALGORITHM
        or all_matrix_report.get("sampling", {}).get("rows_per_tensor") != rows_per_tensor
    ):
        raise ValueError("all-matrix E4M3 report uses a different sampling contract")
    all_matrix_report = {
        **all_matrix_report,
        "identity": {"path": str(all_matrix_report_path.resolve()),
                     "sha256": sha256_file(all_matrix_report_path)},
    }
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = dict(index["weight_map"])
    shards = sorted(set(weight_map.values()))
    draft_ids = draft_head.compute_shortlist(ranking, model_dir).selected.tolist()
    records = []
    specs = selected_specs()
    with SampledShardReader(model_dir, weight_map) as reader:
        for ordinal, spec in enumerate(specs, 1):
            indices = select_row_indices(spec.name, spec.shape[0], rows_per_tensor)
            sampled = _sample_expression(
                source_recipe.RECIPES_BY_NAME[spec.name].expression, indices, reader, draft_ids
            )
            records.append(measure(spec.name, spec.shape, indices, sampled))
            if ordinal % 32 == 0 or ordinal == len(specs):
                print(f"sampled {ordinal}/{len(specs)} selected matrices", flush=True)
    root = Path(__file__).resolve().parents[3]
    implementation_paths = (
        Path(__file__).resolve(),
        root / "tools/convert/qwen3_8_27b_r9700/e4m3_rowwise.py",
        root / "tools/convert/qwen3_8_27b_r9700/q4_row_scaled.py",
        root / "tools/convert/qwen3_8_27b_r9700/screen_e4m3_quality.py",
        root / "tools/convert/qwen3_8_27b_r9700/e4m3_inventory.py",
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
            "shards": [
                {"path": shard, "bytes": (model_dir / shard).stat().st_size,
                 "sha256": sha256_file(model_dir / shard)} for shard in shards
            ],
        },
        implementation={str(path.relative_to(root)): sha256_file(path) for path in implementation_paths},
        all_matrix_reference=all_matrix_report,
        worst_count=worst_count,
    )
    write_no_clobber(output, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--all-matrix-e4m3-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-tensor", type=int, default=DEFAULT_ROWS_PER_TENSOR)
    parser.add_argument("--worst-count", type=int, default=DEFAULT_WORST_COUNT)
    args = parser.parse_args()
    report = run_screen(
        args.model_dir, args.ranking, args.all_matrix_e4m3_report, args.output,
        rows_per_tensor=args.rows_per_tensor, worst_count=args.worst_count,
    )
    print(json.dumps({"aggregate": report["aggregate"], "quality_concentration": report["quality_concentration"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
