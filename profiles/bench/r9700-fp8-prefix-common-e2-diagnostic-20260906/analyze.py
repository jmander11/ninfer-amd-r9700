#!/usr/bin/env python3
"""Fail-closed analysis of retained FP8 common-algorithm diagnostic results."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
E2 = "e2e00100000000000000000000000000"
CATALOG = sorted(f"{value:02x}e001" + "00" * 13 for value in (0xdb, 0xdc, 0xdd, 0xde, 0xdf, 0xe0, 0xe1, 0xe2))
LIMITATIONS = [
    "no retained shell-history command, process exit receipt, stdout, stderr, or independent timestamp receipt exists",
    "no retained build transcript independently proves the asserted source-commit to executable association; both identities are retained exactly",
    "the result self-reports exact hardware and auto power but lacks an external power/process receipt",
    "all timing values are diagnostic and cannot support a production performance claim",
    "the preliminary catalog is retained only to explain selection among the exact eight returned identities",
    "operator prefix exactness does not establish whole-Text append-versus-fresh parity",
]


def fail(message: str) -> None:
    raise ValueError(message)


def load_json(path: Path) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                fail(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        fail(f"nonfinite JSON value {value!r} in {path}")

    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=constant)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def finite_number(value: object, name: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{name} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{name} is nonfinite")
    return result


def check_timing(value: object, name: str) -> dict[str, float]:
    require(isinstance(value, dict), f"{name} is not an object")
    defaults = value.get("default_ms")
    selected = value.get("selected_ms")
    require(isinstance(defaults, list) and isinstance(selected, list) and
            len(defaults) == len(selected) == 14, f"{name} sample count differs")
    d = [finite_number(item, f"{name}.default") for item in defaults]
    s = [finite_number(item, f"{name}.selected") for item in selected]
    require(all(item > 0 for item in d + s), f"{name} has nonpositive sample")
    dmed = statistics.median(d)
    smed = statistics.median(s)
    ratio = smed / dmed
    for field, actual in (("default_median_ms", dmed), ("selected_median_ms", smed),
                          ("selected_over_default", ratio)):
        require(math.isclose(finite_number(value.get(field), f"{name}.{field}"), actual,
                             rel_tol=1e-12, abs_tol=1e-12), f"{name}.{field} differs")
    return {"default_median_ms": dmed, "selected_median_ms": smed,
            "selected_over_default": ratio}


def check_result(value: object, expected_selected: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    require(isinstance(value, dict), "result is not an object")
    require(value.get("artifact_type") == "ninfer_r9700_fp8_linear_prefix_discriminator_result" and
            value.get("schema_version") == 2 and value.get("diagnostic_only") is True and
            value.get("timing_evidence_eligible") is False and
            value.get("production_routing_authorized") is False and
            value.get("common_algorithm_candidate_selector") == 1,
            "result authority contract differs")
    require(value.get("qualified_common_algorithm_fingerprint") == expected_selected,
            "qualified fingerprint differs")
    require(sorted(value.get("gate_up_common_catalog_contract", [])) == CATALOG,
            "catalog contract differs")
    artifact = value.get("artifact")
    device = value.get("device")
    require(isinstance(artifact, dict) and artifact.get("bytes") == 22763026944 and
            artifact.get("sha256") == "d8fc77c36cf17c92e96d67b9a6b5a1826a1ade4f59d59c003b2368fe981fc512",
            "artifact identity differs")
    require(device == {"ordinal": 0, "name": "AMD Radeon AI PRO R9700",
                       "architecture": "gfx1201", "wave_size": 32,
                       "pci_bus_id": "0000:13:00.0", "performance_level": "auto"},
            "device identity differs")
    cells = value.get("cells")
    require(isinstance(cells, list) and len(cells) == 2, "cell inventory differs")
    gate = cells[0]
    query = cells[1]
    require(isinstance(gate, dict) and gate.get("weight_name") == "text/layers/0/mlp/gate_up" and
            gate.get("rows") == 34816 and gate.get("columns") == 5120,
            "gate/up cell differs")
    require(isinstance(query, dict) and query.get("weight_name") == "text/layers/0/gdn/query_key" and
            query.get("rows") == 4096 and query.get("columns") == 5120,
            "query-key cell differs")
    for name, cell in (("gate", gate), ("query", query)):
        require(cell.get("activation_code_prefix_exact") is True and
                cell.get("activation_scale_prefix_exact") is True and
                cell.get("bf16_output_prefix_exact") is True and
                cell.get("bf16_output_prefix_mismatch_count") == 0,
                f"{name} prefix correctness differs")
        oracle = cell.get("represented_fp64_oracle")
        require(isinstance(oracle, dict) and oracle.get("probe_count") == 28 and
                oracle.get("maximum_bf16_steps") == 0 and
                finite_number(oracle.get("maximum_error_bound_ratio"), f"{name}.oracle") <= 1.05,
                f"{name} oracle differs")
    profiles = gate.get("profiles")
    require(isinstance(profiles, list) and len(profiles) == 2 and
            [item.get("tokens") for item in profiles] == [128, 129] and
            all(item.get("algorithm_fingerprint") == expected_selected for item in profiles),
            "selected gate/up profiles differ")
    trials = gate.get("common_algorithm_trials")
    require(isinstance(trials, list) and len(trials) == 8 and
            sorted(item.get("algorithm_fingerprint") for item in trials) == CATALOG,
            "gate/up trial catalog differs")
    summaries: list[dict[str, object]] = []
    for trial in trials:
        fingerprint = trial.get("algorithm_fingerprint")
        require(trial.get("eligible") is True and trial.get("bf16_output_prefix_exact") is True and
                trial.get("bf16_output_prefix_mismatch_count") == 0,
                f"trial {fingerprint} correctness differs")
        oracle = trial.get("represented_fp64_oracle")
        require(isinstance(oracle, dict) and oracle.get("probe_count") == 28 and
                oracle.get("maximum_bf16_steps") == 0 and
                finite_number(oracle.get("maximum_error_bound_ratio"), f"trial {fingerprint} oracle") <= 1.05,
                f"trial {fingerprint} oracle differs")
        timing = trial.get("balanced_timings")
        require(isinstance(timing, list) and [item.get("tokens") for item in timing] == [128, 129],
                f"trial {fingerprint} timing inventory differs")
        summaries.append({"fingerprint": fingerprint,
                          "t128": check_timing(timing[0], f"{fingerprint}.t128"),
                          "t129": check_timing(timing[1], f"{fingerprint}.t129")})
    return gate, summaries


def analyze() -> dict[str, object]:
    plan = load_json(ROOT / "plan.json")
    expected_plan = {
        "artifact_type": "ninfer_r9700_fp8_prefix_common_e2_diagnostic_plan",
        "schema_version": 1,
        "claim": "operator-level real-weight FP8 T128/T129 prefix qualification only",
        "production_routing_authorized": False,
        "timing_evidence_eligible": False,
        "source": {"commit": "25a27bb293e6a32134fbc6b66cecff5444739ccc",
                   "tree": "40703b4636e339604728e37f080225a78960495c"},
        "executable": {"path": "retained-bin/fp8_linear_prefix_discriminator",
                       "bytes": 317712,
                       "sha256": "28d9937f7a3c966b3b49eb3f62449df52dad9d6c22a3e604b42d2ef087c76eff"},
        "artifact": {
            "path": "/ssdpool2nvme/local_llm/ninfer-amd-r9700/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer",
            "bytes": 22763026944,
            "sha256": "d8fc77c36cf17c92e96d67b9a6b5a1826a1ade4f59d59c003b2368fe981fc512"},
        "results": {
            "e2": {"path": "results/e2-selector1.json", "bytes": 26580,
                   "sha256": "e1d349b972baf48dc4c1174caac5d497533a7cf2c18ef5b16fd00894a9ad08ae"},
            "catalog_preliminary": {"path": "results/catalog-selector1-preliminary.json",
                                    "bytes": 26571,
                                    "sha256": "b51468755fd9f8c65e7e4af8df9bbd26df46521d292efa296a8af78619a4bb9c"}},
        "hardware": {"device": 0, "name": "AMD Radeon AI PRO R9700",
                     "architecture": "gfx1201", "wave_size": 32,
                     "pci_bus": "0000:13:00.0", "power_profile_reported_by_tool": "auto"},
        "decision": {"fingerprint": E2, "t128_action": "preserve_default",
                     "t129_action": "select_same_viable_identity",
                     "selection_score": "arithmetic_mean_of_t128_and_t129_selected_over_default",
                     "ranking": "ascending_selection_score_then_ascending_fingerprint",
                     "expected_rank": 1, "whole_text_parity": "pending"},
        "provenance_limitations": LIMITATIONS,
    }
    require(plan == expected_plan, "full plan semantic contract differs")
    for section, name in (("executable", None), ("results", "e2"), ("results", "catalog_preliminary")):
        record = plan[section] if name is None else plan[section][name]
        path = ROOT / record["path"]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == record["bytes"] and
                sha256(path) == record["sha256"], f"identity differs for {path}")
    e2 = load_json(ROOT / plan["results"]["e2"]["path"])
    catalog = load_json(ROOT / plan["results"]["catalog_preliminary"]["path"])
    gate, e2_trials = check_result(e2, E2)
    _, catalog_trials = check_result(catalog, "dde00100000000000000000000000000")
    selected = next(item for item in catalog_trials if item["fingerprint"] == E2)
    confirm = next(item for item in e2_trials if item["fingerprint"] == E2)
    ranked = []
    for item in catalog_trials:
        score = (item["t128"]["selected_over_default"] +
                 item["t129"]["selected_over_default"]) / 2.0
        ranked.append({"fingerprint": item["fingerprint"], "selection_score": score})
    ranked.sort(key=lambda item: (item["selection_score"], item["fingerprint"]))
    require(ranked[0]["fingerprint"] == E2, "predeclared e2 selection is not rank one")
    return {
        "artifact_type": "ninfer_r9700_fp8_prefix_common_e2_diagnostic_summary",
        "schema_version": 1,
        "decision": "e2_operator_qualified_whole_text_pending",
        "timing_evidence_eligible": False,
        "production_routing_authorized": False,
        "fingerprint": E2,
        "selection_rule": plan["decision"]["selection_score"],
        "ranking_rule": plan["decision"]["ranking"],
        "selected_rank": 1,
        "catalog_ranking": ranked,
        "prefix_mismatch_count": gate["bf16_output_prefix_mismatch_count"],
        "oracle_maximum_bf16_steps": gate["represented_fp64_oracle"]["maximum_bf16_steps"],
        "selection_screen_diagnostic": selected,
        "confirmatory_trial_diagnostic": confirm,
        "whole_text_parity": "pending",
        "provenance_limitations": LIMITATIONS,
    }


if __name__ == "__main__":
    json.dump(analyze(), sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")
