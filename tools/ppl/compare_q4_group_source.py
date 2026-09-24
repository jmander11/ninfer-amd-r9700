#!/usr/bin/env python3
"""Compare source-only Q4G64/Q4G128 diagnostics with one BF16 raw authority cell."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.ppl.q4_group_source_diagnostic import (
    ARTIFACT_TYPE, COMPARISON_TYPE, MAXIMUM_MEAN_NLL_DELTA,
    MAXIMUM_NEW_SEVERE_RATE, SCHEMA_VERSION, TERRIBLE_NLL, _atomic_new, sha256_file,
)
from tools.ppl.run import exact_argmax_stats, severe_position_stats


def _exact_int(value, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _sidecar_values(path: Path, suffix: str, code: str) -> list[int | float]:
    data = path.with_suffix(suffix).read_bytes()
    size = struct.calcsize(code)
    if len(data) % size:
        raise ValueError(f"{path.with_suffix(suffix)} is not {code}-aligned")
    return list(struct.unpack("<" + code * (len(data) // size), data))


def _load_diagnostic(path: Path, group: int) -> tuple[dict, list[float], list[int]]:
    report = json.loads(path.read_text())
    if (report.get("artifact_type") != ARTIFACT_TYPE
            or type(report.get("schema_version")) is not int
            or report["schema_version"] != SCHEMA_VERSION
            or type(report.get("group_size")) is not int
            or report["group_size"] != group):
        raise ValueError(f"{path}: wrong diagnostic identity")
    nlls = _sidecar_values(path, ".nllf32", "f")
    argmax = _sidecar_values(path, ".argmaxi32", "i")
    for kind, suffix in (("nll", ".nllf32"), ("argmax", ".argmaxi32")):
        sidecar = report["sidecars"][kind]
        target = path.with_suffix(suffix)
        if sidecar != {"path": target.name, "sha256": sha256_file(target)}:
            raise ValueError(f"{path}: {kind} sidecar identity mismatch")
    count = _exact_int(report["result"].get("tokens_scored"), "tokens_scored")
    if count != len(nlls) or _exact_int(report["result"].get("argmax_tokens"), "argmax_tokens") != len(argmax):
        raise ValueError(f"{path}: sidecar cardinality mismatch")
    if not nlls or not all(math.isfinite(value) for value in nlls):
        raise ValueError(f"{path}: NLL sidecar is empty or nonfinite")
    derived = {
        "non_finite": 0, "terrible_tokens": sum(v >= TERRIBLE_NLL for v in nlls),
        "sum_nll": sum(nlls), "mean_nll": sum(nlls) / len(nlls), "max_nll": max(nlls),
    }
    for key, value in derived.items():
        actual = report["result"].get(key)
        if actual != value and not (type(value) is float and type(actual) is float
                                    and math.isclose(actual, value, rel_tol=1e-7, abs_tol=1e-7)):
            raise ValueError(f"{path}: result {key} does not match raw sidecar")
    return report, nlls, argmax


def _load_bf16(path: Path) -> tuple[dict, list[float], list[int]]:
    report = json.loads(path.read_text())
    if (report.get("weights_id") != "bf16-source"
            or report.get("formula_profile") != "checkpoint-direct-qwen3.8-source-bf16"
            or report.get("schedule") != "prefill"
            or report.get("skip_tokens") != report.get("prompt_tokens") // 2):
        raise ValueError("BF16 input is not the exact prefill/half source authority cell")
    nlls = _sidecar_values(path, ".nllf32", "f")
    argmax = _sidecar_values(path, ".argmaxi32", "i")
    expected_nll = report.get("nll_sha256")
    expected_argmax = report.get("argmax_sha256")
    if expected_nll is None or expected_argmax is None:
        campaign_path = path.parent / "results.json"
        campaign = json.loads(campaign_path.read_text())
        matches = [
            cell for cell in campaign.get("cells", [])
            if cell.get("scheme") == "bf16-reference"
            and cell.get("weights_id") == report["weights_id"]
            and cell.get("formula_profile") == report["formula_profile"]
            and cell.get("schedule") == report["schedule"]
            and cell.get("prompt_tokens") == report["prompt_tokens"]
            and cell.get("skip_tokens") == report["skip_tokens"]
            and cell.get("tokens_scored") == report["tokens_scored"]
            and cell.get("source_config_sha256") == report["source_config_sha256"]
            and cell.get("source_index_sha256") == report["source_index_sha256"]
            and cell.get("source_shards_sha256") == report["source_shards_sha256"]
            and cell.get("corpus_ids_sha256") == report["corpus_ids_sha256"]
        ]
        if len(matches) != 1:
            raise ValueError("BF16 raw authority has no unique enclosing campaign cell")
        expected_nll = matches[0].get("nll_sha256")
        expected_argmax = matches[0].get("argmax_sha256")
    if (sha256_file(path.with_suffix(".nllf32")) != expected_nll
            or sha256_file(path.with_suffix(".argmaxi32")) != expected_argmax):
        raise ValueError("BF16 authority sidecar hash mismatch")
    if len(nlls) != report.get("tokens_scored") or len(argmax) != report.get("argmax_tokens"):
        raise ValueError("BF16 authority sidecar cardinality mismatch")
    return report, nlls, argmax


def _source_key(report: dict) -> tuple:
    source = report["source"]
    shards = source["shards_sha256"]
    if isinstance(shards, dict):
        shard_identity = tuple(sorted(shards.items()))
    elif (isinstance(shards, list)
          and all(isinstance(item, list) and len(item) == 2 for item in shards)):
        shard_identity = tuple(sorted((item[0], item[1]) for item in shards))
    else:
        raise ValueError("source shard identity has an invalid representation")
    return (source["config_sha256"], source["index_sha256"], shard_identity,
            source["corpus_ids_sha256"])


def _against(reference: list[float], candidate: list[float], argmax_ref: list[int],
             argmax_candidate: list[int]) -> dict:
    if len(reference) != len(candidate) or len(argmax_ref) != len(argmax_candidate):
        raise ValueError("comparison sidecars are not exactly aligned")
    severe = severe_position_stats(
        candidate, reference, threshold=TERRIBLE_NLL,
        maximum_new_rate=MAXIMUM_NEW_SEVERE_RATE, minimum_budget=0,
    )
    delta = sum(candidate) / len(candidate) - sum(reference) / len(reference)
    return {
        "mean_nll_delta": delta,
        "maximum_mean_nll_delta": MAXIMUM_MEAN_NLL_DELTA,
        "mean_nll_delta_pass": delta <= MAXIMUM_MEAN_NLL_DELTA,
        **severe,
        **exact_argmax_stats(argmax_candidate, argmax_ref),
        "argmax_identity_is_gate": False,
    }


def compare(g64_path: Path, g128_path: Path, bf16_path: Path) -> dict:
    g64, n64, a64 = _load_diagnostic(g64_path, 64)
    g128, n128, a128 = _load_diagnostic(g128_path, 128)
    bf16, nb, ab = _load_bf16(bf16_path)
    if _source_key(g64) != _source_key(g128):
        raise ValueError("G64 and G128 source/corpus identities differ")
    bf16_source = _source_key({"source": {
        "config_sha256": bf16["source_config_sha256"],
        "index_sha256": bf16["source_index_sha256"],
        "shards_sha256": bf16["source_shards_sha256"],
        "corpus_ids_sha256": bf16["corpus_ids_sha256"],
    }})
    if _source_key(g64) != bf16_source:
        raise ValueError("diagnostic and BF16 source/corpus identities differ")
    if g64["workload"] != g128["workload"] or g64["execution"] != g128["execution"]:
        raise ValueError("G64 and G128 workloads/execution profiles differ")
    rows = {"q4g64": _against(nb, n64, ab, a64),
            "q4g128": _against(nb, n128, ab, a128)}
    incremental = _against(n64, n128, a64, a128)
    return {
        "artifact_type": COMPARISON_TYPE, "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_weight_codec_gate_not_product_admission",
        "inputs": {
            "q4g64": {"path": str(g64_path.resolve()), "sha256": sha256_file(g64_path)},
            "q4g128": {"path": str(g128_path.resolve()), "sha256": sha256_file(g128_path)},
            "bf16": {"path": str(bf16_path.resolve()), "sha256": sha256_file(bf16_path)},
        },
        "workload": g64["workload"], "against_bf16": rows,
        "q4g128_minus_q4g64": incremental,
        "pass": all(row["mean_nll_delta_pass"] and row["new_severe_positions_pass"]
                    for row in rows.values()),
        "limitations": [
            "Weights are decoded canonical Q4, but activations and matmul remain BF16.",
            "Argmax flips are exact-token diagnostics and are not a lossy-codec gate.",
            "A passing report permits artifact/runtime implementation; it is not product PPL evidence.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q4g64", type=Path, required=True)
    parser.add_argument("--q4g128", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        report = compare(args.q4g64, args.q4g128, args.bf16)
        _atomic_new(args.out, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
        return 0 if report["pass"] else 1
    except (FileExistsError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"compare-q4-group-source: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
