#!/usr/bin/env python3
"""Audit the installed ROCm tools and retained gfx1201 profiler evidence.

This command is deliberately GPU-independent.  It compiles two tiny HIP objects but never loads
the HIP runtime or launches a kernel.  Profiler databases are opened read-only and are named
explicitly by the caller so an unrelated or newest-by-mtime capture cannot enter the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "ninfer.r9700.rocm-tool-capability-audit.v1"
ROCM_ROOT = Path("/opt/rocm")


def _run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _one_table(tables: list[str], prefix: str) -> str:
    matches = [table for table in tables if table.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one {prefix!r} table, found {matches}")
    return matches[0]


def summarize_database(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"profiler database does not exist: {path}")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        tables = [row[0] for row in connection.execute(
            "select name from sqlite_master where type='table' order by name")]
        process = _one_table(tables, "rocpd_info_process_")
        agent = _one_table(tables, "rocpd_info_agent_")
        metadata = _one_table(tables, "rocpd_metadata_")
        dispatch = _one_table(tables, "rocpd_kernel_dispatch_")
        commands = [row[0] for row in connection.execute(
            f'select command from "{process}" order by id')]
        agents = [
            {
                "type": row[0],
                "logical_index": row[1],
                "name": row[2],
                "model_name": row[3],
                "product_name": row[4],
            }
            for row in connection.execute(
                f'select type, logical_index, name, model_name, product_name '
                f'from "{agent}" order by id')
        ]
        row_counts: dict[str, int] = {}
        for prefix, label in (
            ("rocpd_kernel_dispatch_", "kernel_dispatch"),
            ("rocpd_memory_copy_", "memory_copy"),
            ("rocpd_region_", "runtime_region"),
        ):
            matches = [table for table in tables if table.startswith(prefix)]
            row_counts[label] = (connection.execute(
                f'select count(*) from "{matches[0]}"').fetchone()[0] if matches else 0)

        counters: dict[str, dict[str, int | float]] = {}
        info_matches = [table for table in tables if table.startswith("rocpd_info_pmc_")]
        event_matches = [table for table in tables if table.startswith("rocpd_pmc_event_")]
        if len(info_matches) == 1 and len(event_matches) == 1:
            query = f'''select i.symbol, count(e.value),
                               sum(case when e.value != 0 then 1 else 0 end),
                               coalesce(min(e.value), 0), coalesce(max(e.value), 0),
                               coalesce(sum(e.value), 0)
                        from "{info_matches[0]}" i
                        left join "{event_matches[0]}" e on e.pmc_id = i.id
                        group by i.id, i.symbol order by i.symbol'''
            for symbol, samples, nonzero, minimum, maximum, total in connection.execute(query):
                if samples:
                    counters[symbol] = {
                        "samples": samples,
                        "nonzero_samples": nonzero,
                        "minimum": minimum,
                        "maximum": maximum,
                        "sum": total,
                    }

        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "schema_metadata": dict(connection.execute(
                f'select tag, value from "{metadata}" order by id')),
            "commands": commands,
            "agents": agents,
            "row_counts": row_counts,
            "counters": counters,
            "dispatch_rows": connection.execute(
                f'select count(*) from "{dispatch}"').fetchone()[0],
        }
    finally:
        connection.close()


def _require_r9700(summary: dict[str, Any]) -> None:
    if not any(agent["type"] == "GPU" and agent["name"] == "gfx1201" and
               agent["product_name"] == "AMD Radeon AI PRO R9700"
               for agent in summary["agents"]):
        raise ValueError(f"database is not an R9700/gfx1201 capture: {summary['path']}")
    if summary["dispatch_rows"] <= 0:
        raise ValueError(f"database has no kernel dispatches: {summary['path']}")


def _compile_sanitizer_probes(hipcc: Path) -> dict[str, Any]:
    source_text = "#include <hip/hip_runtime.h>\n__global__ void probe(int* p) { p[threadIdx.x] = 1; }\n"
    with tempfile.TemporaryDirectory(prefix="ninfer-rocm-audit-") as temporary:
        directory = Path(temporary)
        source = directory / "probe.hip"
        source.write_text(source_text, encoding="utf-8")
        plain = _run([str(hipcc), "-std=c++20", "--offload-arch=gfx1201",
                      "-fsanitize=address", "-c", str(source), "-o", str(directory / "plain.o")])
        xnack = _run([str(hipcc), "-std=c++20", "--offload-arch=gfx1201:xnack+",
                      "-fsanitize=address", "-c", str(source), "-o", str(directory / "xnack.o")])
        temporary_prefix = str(directory)
        for result in (plain, xnack):
            result["command"] = [part.replace(temporary_prefix, "<temporary>")
                                 for part in result["command"]]
            result["output"] = result["output"].replace(temporary_prefix, "<temporary>")
    plain_warning = "ignoring '-fsanitize=address' option for offload arch 'gfx1201'" in plain["output"]
    xnack_rejected = "invalid target ID 'gfx1201:xnack+'" in xnack["output"]
    if plain["returncode"] != 0 or not plain_warning:
        raise ValueError("plain gfx1201 sanitizer probe did not produce the expected unsupported warning")
    if xnack["returncode"] == 0 or not xnack_rejected:
        raise ValueError("gfx1201:xnack+ sanitizer probe was not rejected as an invalid target")
    return {
        "plain_gfx1201": plain,
        "plain_device_asan_instrumented": False,
        "xnack_gfx1201": xnack,
        "xnack_target_available": False,
        "classification": "device AddressSanitizer unavailable for installed gfx1201 toolchain",
    }


def _supported_compute_architectures(help_text: str) -> list[str]:
    return sorted(set(re.findall(r"\bgfx[0-9a-z]+\b", help_text)))


def build_report(arguments: argparse.Namespace) -> dict[str, Any]:
    hipcc = ROCM_ROOT / "bin/hipcc"
    rocprofv3 = ROCM_ROOT / "bin/rocprofv3"
    compute = ROCM_ROOT / "bin/rocprof-compute"
    sys_avail = ROCM_ROOT / "bin/rocprof-sys-avail"
    rocgdb = ROCM_ROOT / "bin/rocgdb"
    tools = {
        "hipcc": _run([str(hipcc), "--version"]),
        "rocprofv3": _run([str(rocprofv3), "--version"]),
        "rocprof_compute": _run([str(compute), "--version"]),
        "rocprof_sys": _run([str(sys_avail), "--version"]),
        "rocgdb": _run([str(rocgdb), "--version"]),
    }
    for name, result in tools.items():
        if result["returncode"] != 0:
            raise ValueError(f"installed tool probe failed for {name}: {result['output']}")

    compute_help = _run([str(compute), "--help"])
    supported_architectures = _supported_compute_architectures(compute_help["output"])
    if "gfx1201" in supported_architectures:
        raise ValueError("rocprof-compute now advertises gfx1201; unavailable classification is stale")

    trace = summarize_database(arguments.trace_db)
    pmc = summarize_database(arguments.pmc_db)
    _require_r9700(trace)
    _require_r9700(pmc)
    if trace["commands"] != [arguments.expected_trace_command]:
        raise ValueError(
            f"trace workload command mismatch: expected {arguments.expected_trace_command!r}, "
            f"found {trace['commands']!r}")
    if pmc["commands"] != [arguments.expected_pmc_command]:
        raise ValueError(
            f"PMU workload command mismatch: expected {arguments.expected_pmc_command!r}, "
            f"found {pmc['commands']!r}")
    if trace["row_counts"]["memory_copy"] <= 0 or trace["row_counts"]["runtime_region"] <= 0:
        raise ValueError("trace database lacks required memory-copy or runtime records")
    for symbol in ("SQ_BUSY_CYCLES", "SQ_WAVES"):
        if pmc["counters"].get(symbol, {}).get("nonzero_samples", 0) <= 0:
            raise ValueError(f"validated PMU counter is missing or all-zero: {symbol}")

    zero_probes: dict[str, Any] = {}
    for specification in arguments.zero_probe:
        symbol, separator, raw_path = specification.partition("=")
        if not separator or not symbol or not raw_path:
            raise ValueError(f"invalid --zero-probe {specification!r}; expected SYMBOL=PATH")
        summary = summarize_database(Path(raw_path))
        _require_r9700(summary)
        counter = summary["counters"].get(symbol)
        if counter is None or counter["samples"] <= 0:
            raise ValueError(f"zero probe did not collect {symbol}: {raw_path}")
        if counter["nonzero_samples"] != 0 or counter["sum"] != 0:
            raise ValueError(f"counter is no longer identically zero: {symbol} in {raw_path}")
        zero_probes[symbol] = summary

    unavailable_counters = sorted(zero_probes)
    compute_soc_root = (ROCM_ROOT / "core-10.0/libexec/rocprofiler-compute/"
                        "rocprof_compute_soc")
    compute_soc_modules = sorted(path.name for path in compute_soc_root.glob("soc_gfx*.py"))
    optional_tools = {
        name: shutil.which(name)
        for name in ("amd-smi", "compute-sanitizer", "rocprof", "rocprofv2", "rocm-gdb",
                     "rocm-smi", "valgrind")
    }
    return {
        "schema": SCHEMA,
        "target": {
            "architecture": "gfx1201",
            "product": "AMD Radeon AI PRO R9700",
            "wavefront_size": 32,
        },
        "tools": tools,
        "optional_tool_paths": optional_tools,
        "host": {
            "kernel_release": os.uname().release,
            "perf_event_paranoid": Path("/proc/sys/kernel/perf_event_paranoid").read_text(
                encoding="utf-8").strip(),
        },
        "rocprof_compute": {
            "supported_architectures": supported_architectures,
            "soc_modules": compute_soc_modules,
            "gfx1201_supported": False,
        },
        "sanitizer": _compile_sanitizer_probes(hipcc),
        "retained_evidence": {
            "trace": trace,
            "pmc": pmc,
            "zero_counter_probes": zero_probes,
        },
        "classification": {
            "runtime_kernel_memory_trace": "available",
            "validated_pmu_counters": ["SQ_BUSY_CYCLES", "SQ_WAVES"],
            "unavailable_pmu_counters": unavailable_counters,
            "rocprof_compute_gfx1201_analysis": "unavailable in installed release",
            "device_address_sanitizer": "unavailable in installed toolchain/target",
            "host_address_sanitizer": "available through make -C tools/r9700 sanitize",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-db", required=True, type=Path)
    parser.add_argument("--pmc-db", required=True, type=Path)
    parser.add_argument("--expected-trace-command", required=True)
    parser.add_argument("--expected-pmc-command", required=True)
    parser.add_argument("--zero-probe", action="append", default=[], metavar="SYMBOL=PATH")
    parser.add_argument("--output", type=Path, help="atomically write JSON instead of stdout")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report = build_report(arguments)
    except (OSError, sqlite3.Error, ValueError) as error:
        raise SystemExit(f"ROCm capability audit failed: {error}") from error
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_name(arguments.output.name + f".tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, arguments.output)
        print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
