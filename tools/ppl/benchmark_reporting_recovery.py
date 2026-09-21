"""Reporting-only bridge: retain capacity/numerics, never old concurrent phase rates."""
from __future__ import annotations

import json
from pathlib import Path

from tools.ppl import fp8_context_recovery as fp8

REPO = Path(__file__).resolve().parents[2]
SCOPE = "serial_lane_phase_reporting_only"
SOURCE_FILES = (
    "bench/targets/qwen3_8_27b/ninfer_bench.cpp",
    "bench/targets/qwen3_8_27b/ninfer_bench_support.h",
    "bench/targets/qwen3_8_27b/ninfer_bench_support.cpp",
)


def check_review(value: dict, resource: dict) -> None:
    if (value.get("artifact_type") != "ninfer_r9700_benchmark_reporting_source_review"
            or value.get("schema_version") != 1 or value.get("status") != "reporting_only"
            or value.get("scope") != SCOPE
            or any(value.get(field) is not False for field in (
                "model_runtime_changed", "arithmetic_changed", "kernel_or_recipe_changed"))
            or value.get("base_source_review") != resource["source_review"]):
        raise ValueError("reporting bridge requires unchanged model/runtime/arithmetic review")
    fp8.check_review(json.loads(fp8.checked_file(value["base_source_review"]).read_text()))
    sources = value.get("sources")
    if (not isinstance(sources, list) or len(sources) != len(SOURCE_FILES)
            or {str(fp8.checked_file(item)) for item in sources}
            != {str(REPO / name) for name in SOURCE_FILES}):
        raise ValueError("reporting review must bind exactly the three host reporter sources")


def mapping(bridge: dict, group: int, attention: str) -> dict:
    matches = [row for row in bridge["profiles"]
               if (row["kv_value_group"], row["xattention_profile"]) == (group, attention)]
    if len(matches) != 1:
        raise ValueError("reporting bridge lacks unique compiled profile")
    return matches[0]


def source_mapping(source: dict, bridge: dict) -> dict:
    return mapping(bridge, source["cache_value_group"],
                   source["quality"]["representation"]["xattention_profile"])


def expected_benchmark(source: dict, preset: str, bridge: dict | None) -> dict:
    if bridge is None:
        return source["benchmark_executable"]
    row = source_mapping(source, bridge)
    if preset == "pareto-whole":
        return row["new_benchmark"]
    if preset != "pareto-capacity":
        raise ValueError("reporting bridge supports only capacity and whole matrices")
    kind = "hybrid" if source["artifact"]["weights_id"] == fp8.HYBRID else "nonhybrid"
    return row[f"old_{kind}_benchmark"]


def validate_source_matrices(source: dict, bridge: dict) -> None:
    row = source_mapping(source, bridge)
    if source["benchmark_executable"] != row["new_benchmark"]:
        raise ValueError("active benchmark differs from reporting bridge")
    for preset, binding in source["matrices"].items():
        manifest = json.loads(fp8.checked_file(
            {key: binding[key] for key in ("path", "sha256")}).read_text())
        if (manifest.get("bench") != expected_benchmark(source, preset, bridge)
                or manifest.get("artifact") != source["artifact"]):
            raise ValueError("reporting bridge matrix artifact/benchmark differs")
        if source["artifact"]["weights_id"] == fp8.HYBRID:
            planner = row["new_planner" if preset == "pareto-whole" else "old_hybrid_planner"]
            if manifest.get("hybrid_shared_workspace_authority", {}).get("tool") != planner:
                raise ValueError("reporting bridge matrix uses another hybrid planner")


def validate_bridge(path: Path, chunk_selection: dict | None = None) -> dict:
    return validate_bridge_value(json.loads(path.read_text()), chunk_selection)


def bound_bridge(sources: list[dict]) -> dict | None:
    bindings = [source.get("benchmark_reporting_recovery") for source in sources]
    if not any(binding is not None for binding in bindings):
        return None
    if any(binding != bindings[0] for binding in bindings):
        raise ValueError("reporting bridge binding differs across candidates")
    bridge = validate_bridge(fp8.checked_file(bindings[0]))
    if any(source.get("fp8_context_resource_recovery") != bridge["fp8_context_resource_recovery"]
           for source in sources):
        raise ValueError("reporting and FP8 resource bridges differ")
    return bridge


def validate_bridge_value(value: dict, chunk_selection: dict | None = None) -> dict:
    if (value.get("artifact_type") != "ninfer_r9700_benchmark_reporting_recovery"
            or value.get("schema_version") != 1 or value.get("scope") != SCOPE
            or value.get("preserves") != ["numerical_quality", "selected_prefill_chunk", "capacity"]
            or value.get("reuses_concurrent_prefill_metrics") is not False):
        raise ValueError("unsupported reporting-only bridge")
    resource = fp8.validate_bridge(fp8.checked_file(value.get("fp8_context_resource_recovery")),
                                   chunk_selection)
    selection = json.loads(fp8.checked_file(resource["chunk_selection"]).read_text())
    frozen = json.loads(fp8.checked_file(resource["frozen_panel_inputs"]).read_text())
    check_review(json.loads(fp8.checked_file(value.get("source_review")).read_text()), resource)
    profiles = value.get("profiles")
    expected = {(group, attention) for group in (16, 32)
                for attention in ("dense", "b128-s16-tau900")}
    if (not isinstance(profiles, list) or len(profiles) != 4
            or {(row.get("kv_value_group"), row.get("xattention_profile"))
                for row in profiles} != expected):
        raise ValueError("reporting bridge requires exactly four compiled profiles")
    for row in profiles:
        key = (row["kv_value_group"], row["xattention_profile"])
        old_sources = [source for source in selection["sources"] if
                       (source["kv_value_group"], source["xattention_profile"]) == key
                       and source["weights_id"] != fp8.HYBRID]
        if (len(old_sources) != 2 or any(source["benchmark_executable"]
                != row.get("old_nonhybrid_benchmark") for source in old_sources)):
            raise ValueError("reporting bridge changed retained nonhybrid benchmark")
        hybrid, = [item for item in resource["hybrid_builds"] if
                   (item["kv_value_group"], item["xattention_profile"]) == key]
        for suffix, resource_field in (("benchmark", "new_benchmark"),
                                       ("planner", "new_planner"), ("build_cache", "build_cache")):
            if row.get(f"old_hybrid_{suffix}") != hybrid[resource_field]:
                raise ValueError("reporting bridge changed retained accounted hybrid build")
        for prefix in ("old_nonhybrid_", "old_hybrid_", "new_"):
            bench = fp8.checked_file(row.get(prefix + "benchmark"))
            planner = fp8.checked_file(row.get(prefix + "planner"))
            cache_field = "build_cache" if prefix == "new_" else prefix + "build_cache"
            cache = fp8.checked_file(row.get(cache_field))
            build = bench.parent.parent
            if (bench != build / "bench/ninfer_bench"
                    or planner != build / "src/ninfer_r9700_runtime_planner_qual"
                    or cache != build / "CMakeCache.txt"):
                raise ValueError("reporting bridge executable/planner/cache roots differ")
            if prefix == "old_nonhybrid_":
                for item in (row[prefix + "benchmark"], row[prefix + "planner"], row[cache_field]):
                    if frozen["files"].get(str(Path(item["path"]).relative_to(REPO))) != item["sha256"]:
                        raise ValueError("reporting bridge nonhybrid build differs from frozen panel")
            if fp8.cache_profile(cache) != fp8.cache_profile(Path(row["build_cache"]["path"])):
                raise ValueError("reporting bridge changed compiled profile")
        if row["new_benchmark"] in (row["old_nonhybrid_benchmark"], row["old_hybrid_benchmark"]):
            raise ValueError("reporting bridge requires a fresh reporter executable")
        check = json.loads(fp8.checked_file(row.get("host_planner_check")).read_text())
        if (check.get("command") != [row["new_planner"]["path"], "--host-split512-routing"]
                or type(check.get("exit_code")) is not int or check["exit_code"] != 0):
            raise ValueError("reporting bridge lacks the matching host planner check")
        fp8.checked_file(check.get("stdout"))
        fp8.checked_file(check.get("stderr"))
    return value
