#!/usr/bin/env python3
"""Parse the exact ATT options with the installed rocprofv3 without executing an app."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
from pathlib import Path


def validate_parser(rocprof: Path, arguments: list[str]) -> None:
    source = rocprof.read_text(encoding="utf-8")
    conflict = "--selected-regions and --att-consecutive-kernels are mutually exclusive"
    if conflict not in source:
        raise ValueError("installed rocprofv3 conflict contract differs")
    loader = importlib.machinery.SourceFileLoader(
        "ninfer_installed_rocprofv3_parser", str(rocprof))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ValueError("cannot load installed rocprofv3 parser")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    parsed, application = module.parse_arguments(arguments)
    expected = {
        "selected_regions": True,
        "advanced_thread_trace": True,
        "att_target_cu": "1",
        "att_simd_select": "0",
        "att_buffer_size": "1610612736",
        "att_shader_engine_mask": "0x1",
        "att_gpu_index": "0",
        "att_consecutive_kernels": None,
        "kernel_include_regex": "a8q4g64_linear_prefill_cta_kernel",
        "kernel_iteration_range": ["[3]"],
        "marker_trace": True,
        "kernel_trace": True,
        "output_format": ["csv", "rocpd"],
        "output_file": "production-q4-p2048-att",
    }
    if any(getattr(parsed, key, None) != value for key, value in expected.items()):
        raise ValueError("installed rocprofv3 did not parse the exact ATT contract")
    if application != ["/usr/bin/true"]:
        raise ValueError("ATT parser preflight application sentinel differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocprof", required=True, type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    validate_parser(args.rocprof, arguments)
    print("rocprofv3 ATT argument parser: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
