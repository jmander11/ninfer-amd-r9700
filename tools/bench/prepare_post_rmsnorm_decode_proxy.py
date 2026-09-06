#!/usr/bin/env python3
"""Prepare the bounded post-RMSNorm C1 ordinary-decode PMC diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sqlite3
import stat
import tempfile
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "build-r9700-rmsnorm-production-final-20260906/bench/ninfer_bench"
BENCH_SHA256 = "a7c9303bd213fa3dbdb29ca0cee73addef1de8ab6a6e6b231509e25239776425"
FINAL_REPORT = REPO / "profiles/bench/r9700-rmsnorm-production-final-p8192-g256-c1-20260906.json"
FINAL_REPORT_SHA256 = "b05db0068a4f1c73ce9c2092443b42f9f48b0fdb80ff8b3335db60cd5bdca74b"
ARTIFACT = REPO / "out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
ARTIFACT_SHA256 = "60719c5d5bfe978376c7de94db460bcd82d965ec447870cc47f0a02bf8d59953"
CORPUS = REPO / "bench/fixtures/bench_corpus.ids"
CORPUS_SHA256 = "27e4f63c17efe3f89b5cf278b3b1a42a737316ed4044d7d0d1d52437059d1002"
TRACE_REPORT = REPO / "profiles/rocprof/r9700-rmsnorm-production-selected-trace-20260906/benchmark-report.json"
TRACE_REPORT_SHA256 = "d3266256f79970d49f122bfdcf7b6deae6f2ee644f06f4e9453d86e0e0d52b15"
TRACE_DB = REPO / "profiles/rocprof/r9700-rmsnorm-production-selected-trace-20260906/raw/rmsnorm-production-selected_results.db"
TRACE_DB_SHA256 = "8fe71be97e77c2651cb0c75fe203cedb13f200f8ac76082e4310bbfb855e5370"
ROCPROF = Path("/opt/rocm/core-10.0/bin/rocprofv3")
ROCPROF_AVAIL = Path("/opt/rocm/bin/rocprofv3-avail")
STREAM_PROBE = REPO / "tools/r9700/build/hbm_bandwidth_probe"
STREAM_PROBE_SHA256 = "060e110bb3bf69678d3e81b52f3c3c457d17770f8210e2e165f3494169ade51b"
POWER = Path("/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level")
ANALYZER = REPO / "tools/bench/analyze_post_rmsnorm_decode_proxy.py"
PRODUCER = Path(__file__).resolve()

PASSES = {
    "cache-wait": (
        "SQ_WAIT_ANY", "SQ_WAIT_INST_ANY", "SQ_WAVE_CYCLES", "SQ_WAVES",
        "GL2C_HIT", "GL2C_MISS", "TCP_REQ", "TCP_REQ_MISS", "GL2C_EA_RDREQ",
        "GRBM_GUI_ACTIVE", "TA_TA_BUSY", "GL2C_MC_WRREQ_STALL",
    ),
    "issue-lds": (
        "SQ_INST_CYCLES_VALU", "SQ_INST_CYCLES_VMEM", "SQ_INSTS_LDS",
        "SQ_INSTS_TEX_LOAD", "SQ_INSTS_TEX_STORE", "SQC_LDS_BANK_CONFLICT",
        "SQC_LDS_IDX_ACTIVE", "SQ_WAVES", "GRBM_GUI_ACTIVE", "TA_TA_BUSY",
    ),
}


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def identity(path: Path, expected: str | None = None) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"unsafe input: {resolved}")
    result = {"path": str(resolved), "file_size_bytes": resolved.stat().st_size,
              "sha256": sha(resolved)}
    if expected is not None and result["sha256"] != expected:
        raise ValueError(f"pinned input changed: {resolved}")
    return result


def one_table(connection: sqlite3.Connection, prefix: str) -> str:
    rows = [row[0] for row in connection.execute(
        "select name from sqlite_master where type='table' and name like ?", (prefix + "%",))]
    if len(rows) != 1:
        raise ValueError(f"trace lacks one {prefix} table")
    return rows[0]


def trace_inventory() -> list[dict[str, object]]:
    connection = sqlite3.connect(f"file:{TRACE_DB.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        dispatch = one_table(connection, "rocpd_kernel_dispatch_")
        symbol = one_table(connection, "rocpd_info_kernel_symbol_")
        strings = one_table(connection, "rocpd_string_")
        rows = connection.execute(f'''select s.display_name symbol,
                    k.grid_size_x*k.grid_size_y*k.grid_size_z grid_size,
                    k.workgroup_size_x*k.workgroup_size_y*k.workgroup_size_z workgroup_size,
                    k.group_segment_size static_lds_bytes,
                    k.private_segment_size scratch_bytes,
                    s.arch_vgpr_count+s.accum_vgpr_count vgpr
                from "{dispatch}" k join "{symbol}" s on s.id=k.kernel_id
                join "{strings}" r on r.id=k.region_name_id
                where r.string='ninfer.decode.decode.ordinary_round payload=8192' ''')
        counts = Counter(tuple(row) for row in rows)
        if not counts:
            raise ValueError("selected trace has no ordinary frontier-8192 inventory")
        return [{"symbol": key[0], "grid_size": key[1], "workgroup_size": key[2],
                 "static_lds_bytes": key[3], "scratch_bytes": key[4], "vgpr": key[5],
                 "dispatch_count": count} for key, count in sorted(counts.items())]
    finally:
        connection.close()


def validate_final() -> tuple[dict, list[int]]:
    report = json.loads(FINAL_REPORT.read_text(encoding="utf-8"))
    test = report.get("tests", [{}])[0]
    rep = test.get("reps", [{}])[0]
    config = report.get("config", {})
    if (report.get("artifact_type") != "ninfer_bench_report"
            or report.get("schema_version") != 20
            or report.get("environment", {}).get("architecture_name") != "gfx1201"
            or config.get("concurrency") != 1 or config.get("spec") != "none"
            or config.get("draft_tokens") != 0 or config.get("use_device_graph") is not True
            or config.get("decode_path") != "device_graph"
            or config.get("kv_cache_format") != "fp8-k-int4-v"
            or config.get("kv_value_group") != 16 or config.get("prefill_chunk") != 4096
            or config.get("xattention_qualification") is not False
            or test.get("kind") != "whole" or test.get("n_prompt") != 8192
            or test.get("n_gen") != 256 or test.get("requested_output_tokens") != 257
            or rep.get("generated_output_tokens") != 257
            or rep.get("decode_output_tokens") != 256
            or rep.get("decode_engine_tokens") != 256):
        raise ValueError("final report is not exact post-RMSNorm C1/P8192+G256 ordinary authority")
    lanes = rep.get("generated_token_ids_by_lane")
    if (not isinstance(lanes, list) or len(lanes) != 1 or len(lanes[0]) != 257
            or not all(type(token) is int and token >= 0 for token in lanes[0])):
        raise ValueError("final report lacks exact 257-token identity")
    return report, lanes[0]


def benchmark_command(report: Path) -> list[str]:
    return [str(BENCH.resolve()), "--weights", str(ARTIFACT.resolve()), "--corpus",
            str(CORPUS.resolve()), "--device", "0", "--concurrency", "1", "--whole-pg",
            "8192,256", "--prefill-chunk", "4096", "--draft-tokens", "0",
            "--retain-token-ids", "--output", "json", "--output-file", str(report),
            "-r", "1", "--warmup", "1", "--profile-measured"]


def shell(plan_dir: Path) -> str:
    quoted = shlex.quote
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "",
             f"plan_dir={quoted(str(plan_dir))}", f"power={quoted(str(POWER))}",
             f"bench={quoted(str(BENCH.resolve()))}", f"artifact={quoted(str(ARTIFACT.resolve()))}",
             f"corpus={quoted(str(CORPUS.resolve()))}", "current_after=", "",
             "test \"$(sha256sum \"$bench\" | cut -d' ' -f1)\" = " + BENCH_SHA256,
             "test \"$(sha256sum \"$artifact\" | cut -d' ' -f1)\" = " + ARTIFACT_SHA256,
             "test \"$(sha256sum \"$corpus\" | cut -d' ' -f1)\" = " + CORPUS_SHA256,
             f"test \"$(sha256sum {quoted(str(STREAM_PROBE.resolve()))} | cut -d' ' -f1)\" = {STREAM_PROBE_SHA256}",
             f"test \"$(sha256sum {quoted(str(ROCPROF.resolve()))} | cut -d' ' -f1)\" = {sha(ROCPROF)}",
             f"test \"$(sha256sum {quoted(str(ROCPROF_AVAIL.resolve()))} | cut -d' ' -f1)\" = {sha(ROCPROF_AVAIL)}",
             "test \"$(cat \"$power\")\" = auto", "sudo -v", "",
             "restore_auto() {", "  printf '%s\\n' auto | sudo tee \"$power\" >/dev/null",
             "  test \"$(cat \"$power\")\" = auto",
             "  if [[ -n \"$current_after\" ]]; then printf '%s\\n' auto >\"$current_after\"; fi",
             "}", "trap restore_auto EXIT", "",
             "# Preflight both bounded counter sets under the required profiling power state.",
             "printf '%s\\n' profile_standard | sudo tee \"$power\" >/dev/null",
             "test \"$(cat \"$power\")\" = profile_standard"]
    for label, counters in PASSES.items():
        lines.append(f"{quoted(str(ROCPROF_AVAIL))} --device 0 pmc-check " + " ".join(counters))
    lines += ["restore_auto", "", "run_pass() {", "  local label=$1; shift",
              "  local report=\"$plan_dir/benchmark-$label.json\"",
              "  local raw=\"$plan_dir/raw-$label\"",
              "  local before=\"$plan_dir/power-$label-before.txt\"",
              "  local after=\"$plan_dir/power-$label-after.txt\"", "  current_after=$after",
              "  test ! -e \"$raw\" && test ! -e \"$report\" && test ! -e \"$before\" && test ! -e \"$after\"",
              "  test \"$(cat \"$power\")\" = auto",
              "  printf '%s\\n' profile_standard | sudo tee \"$power\" >/dev/null",
              "  test \"$(cat \"$power\")\" = profile_standard",
              "  printf '%s\\n' profile_standard >\"$before\"",
              f"  {quoted(str(ROCPROF))} --selected-regions -f csv rocpd -d \"$raw\" -o \"post-rmsnorm-$label\" --marker-trace --kernel-trace --pmc \"$@\" -- \\",
              "    \"$bench\" --weights \"$artifact\" --corpus \"$corpus\" --device 0 --concurrency 1 --whole-pg 8192,256 --prefill-chunk 4096 --draft-tokens 0 --retain-token-ids --output json --output-file \"$report\" -r 1 --warmup 1 --profile-measured",
              "  restore_auto", "  current_after=", "}", ""]
    for label, counters in PASSES.items():
        lines.append(f"run_pass {label} " + " ".join(counters))
    lines += ["restore_auto", "",
              "# Same-session auto-state streaming ceiling; this is not inference traffic.",
              "current_after=\"$plan_dir/power-stream-after.txt\"",
              "test ! -e \"$plan_dir/stream-probe.stdout\" && test ! -e \"$plan_dir/stream-probe.stderr\"",
              "test ! -e \"$plan_dir/power-stream-before.txt\" && test ! -e \"$current_after\"",
              "test \"$(cat \"$power\")\" = auto",
              "printf '%s\\n' auto >\"$plan_dir/power-stream-before.txt\"",
              f"{quoted(str(STREAM_PROBE.resolve()))} --size-gib 4 --trials 5 >\"$plan_dir/stream-probe.stdout\" 2>\"$plan_dir/stream-probe.stderr\"",
              "test \"$(cat \"$power\")\" = auto", "printf '%s\\n' auto >\"$current_after\"",
              "current_after=", "trap - EXIT",
              f"python3 {quoted(str(ANALYZER))} --plan \"$plan_dir/plan.json\" --root \"$plan_dir\" --out \"$plan_dir/analysis.json\"", ""]
    return "\n".join(lines)


def publish(path: Path, data: bytes, mode: int = 0o644) -> None:
    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.write(descriptor, data); os.fsync(descriptor); os.fchmod(descriptor, mode)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("unsafe pending package inode")
        os.link(temporary, path)
    finally:
        os.close(descriptor); Path(temporary).unlink(missing_ok=True)


def prepare(out: Path) -> None:
    out = Path(os.path.abspath(out))
    if os.path.lexists(out):
        raise ValueError(f"output already exists: {out}")
    for path, expected in ((BENCH, BENCH_SHA256), (FINAL_REPORT, FINAL_REPORT_SHA256),
                           (ARTIFACT, ARTIFACT_SHA256), (CORPUS, CORPUS_SHA256),
                           (TRACE_REPORT, TRACE_REPORT_SHA256), (TRACE_DB, TRACE_DB_SHA256),
                           (STREAM_PROBE, STREAM_PROBE_SHA256)):
        identity(path, expected)
    final, tokens = validate_final()
    trace_report = json.loads(TRACE_REPORT.read_text(encoding="utf-8"))
    if (trace_report.get("artifact_type") != "ninfer_bench_report"
            or trace_report.get("environment", {}).get("architecture_name") != "gfx1201"
            or trace_report.get("tests", [{}])[0].get("n_prompt") != 8192
            or trace_report.get("tests", [{}])[0].get("n_gen") != 1):
        raise ValueError("selected-region trace report identity differs")
    out.mkdir(parents=True)
    commands = {label: benchmark_command(out / f"benchmark-{label}.json")
                for label in PASSES}
    plan = {
        "artifact_type": "ninfer_r9700_post_rmsnorm_decode_proxy_plan", "schema_version": 1,
        "status": "command_only_not_executed", "profile_timing_admissible": False,
        "workload": {"concurrency": 1, "prompt_tokens": 8192, "generated_tokens": 256,
                     "requested_output_tokens": 257, "spec": "none"},
        "terminal_timing": {"report": identity(FINAL_REPORT, FINAL_REPORT_SHA256),
                            "executable": identity(BENCH, BENCH_SHA256),
                            "generated_token_ids": tokens,
                            "decode_output_tok_s": final["tests"][0]["decode_output_tok_s_mean"]},
        "artifact": identity(ARTIFACT, ARTIFACT_SHA256),
        "corpus": identity(CORPUS, CORPUS_SHA256),
        "selected_region_trace": {"report": identity(TRACE_REPORT, TRACE_REPORT_SHA256),
                                  "database": identity(TRACE_DB, TRACE_DB_SHA256),
                                  "frontier": 8192, "inventory": trace_inventory()},
        "profiler": identity(ROCPROF), "counter_preflight": identity(ROCPROF_AVAIL),
        "stream_probe": {"executable": identity(STREAM_PROBE, STREAM_PROBE_SHA256),
                         "arguments": ["--size-gib", "4", "--trials", "5"],
                         "power_profile": "auto", "working_set_gib_per_buffer": 4,
                         "trials_per_method": 5},
        "power_profile": {"path": str(POWER), "required_during_capture": "profile_standard",
                          "required_before_after": "auto"},
        "passes": {label: {"counters": list(counters), "benchmark_command": commands[label],
                           "raw_directory": str(out / f"raw-{label}"),
                           "benchmark_report": str(out / f"benchmark-{label}.json")}
                   for label, counters in PASSES.items()},
        "measurement_contract": {
            "scope": "selected ROCTX ordinary rounds from one C1/P8192+G256 request",
            "counter_values": "native rocprofiler per-dispatch aggregates",
            "cross_pass_join": "exact dispatch resource-inventory multiset, never dispatch id",
            "physical_memory_bandwidth_bytes_per_second": None,
            "physical_peak_fraction": None, "stall_freedom": None,
            "conclusion": "diagnostic relative cache, request, wait, issue, and LDS proxies only",
        },
        "producer": identity(PRODUCER), "analyzer": identity(ANALYZER),
    }
    publish(out / "plan.json", (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode())
    publish(out / "commands.sh", shell(out).encode(), 0o755)
    closure = "".join(f"{sha(path)}  {path.relative_to(REPO)}\n" for path in
                      (PRODUCER, ANALYZER, out / "plan.json", out / "commands.sh"))
    publish(out / "prepared.sha256", closure.encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        prepare(args.out)
    except (OSError, sqlite3.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
