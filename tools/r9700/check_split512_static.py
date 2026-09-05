#!/usr/bin/env python3
"""Failing gfx1201 ISA/resource regression gates for production split-512 attention."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


BEGIN = re.compile(r"-- Begin function\s+(\S+)")
INTEGER_FIELDS = {
    "lds": re.compile(r"\.amdhsa_group_segment_fixed_size\s+(\d+)"),
    "private": re.compile(r"\.amdhsa_private_segment_fixed_size\s+(\d+)"),
    "vgpr": re.compile(r"\.amdhsa_next_free_vgpr\s+(\d+)"),
    "scratch": re.compile(r"ScratchSize:\s*(\d+)"),
}


@dataclass(frozen=True)
class Kernel:
    name: str
    body: str

    @property
    def kind(self) -> str | None:
        if "qk_wmma_kernel" in self.name:
            return "t1_qk"
        if "split512_qk_bf16_t4_kernel" in self.name:
            return "qk"
        if "split512_softmax_pv_partial_kernel" in self.name:
            return "partial"
        if "split512_merge_kernel" in self.name:
            return "merge"
        return None

    def integer(self, field: str) -> int:
        match = INTEGER_FIELDS[field].search(self.body)
        if match is None:
            raise SystemExit(f"{self.name}: missing {field} metadata")
        return int(match.group(1))


def kernels(assembly: str) -> list[Kernel]:
    matches = list(BEGIN.finditer(assembly))
    selected = [
        Kernel(match.group(1), assembly[match.start() : matches[index + 1].start()])
        for index, match in enumerate(matches[:-1])
        if "split512_" in match.group(1) or "qk_wmma_kernel" in match.group(1)
    ]
    if matches and ("split512_" in matches[-1].group(1) or
                    "qk_wmma_kernel" in matches[-1].group(1)):
        selected.append(Kernel(matches[-1].group(1), assembly[matches[-1].start() :]))
    return selected


def check_isa(selected: list[Kernel]) -> None:
    t1_qk = [kernel for kernel in selected if kernel.kind == "t1_qk"]
    qk = [kernel for kernel in selected if kernel.kind == "qk"]
    if len(t1_qk) != 2:
        raise SystemExit(
            f"split-512 static gate expected 2 retained T1 QK specializations, got {len(t1_qk)}"
        )
    if len(qk) != 2:
        raise SystemExit(
            f"split-512 static gate expected 2 T4 QK specializations, got {len(qk)}"
        )
    for kernel in t1_qk:
        if "v_wmma_f32_16x16x16_fp8_fp8" not in kernel.body:
            raise SystemExit(f"{kernel.name}: missing native FP8 WMMA instruction")
    for kernel in qk:
        if "v_wmma_f32_16x16x16_bf16" not in kernel.body:
            raise SystemExit(f"{kernel.name}: missing native BF16 WMMA instruction")
    print(
        "split-512 ISA gate passed: "
        f"{len(t1_qk)} retained T1 FP8-QK and {len(qk)} T4 BF16-QK specializations"
    )


def check_resources(selected: list[Kernel]) -> None:
    limits = {
        "t1_qk": {"lds": 0, "vgpr": 24},
        "qk": {"lds": 9220, "vgpr": 33},
        "partial": {"lds": 6224, "vgpr": 26},
        "merge": {"lds": 2124, "vgpr": 25},
    }
    counts = {kind: 0 for kind in limits}
    for kernel in selected:
        kind = kernel.kind
        if kind is None:
            continue
        counts[kind] += 1
        values = {field: kernel.integer(field) for field in INTEGER_FIELDS}
        if values["private"] != 0 or values["scratch"] != 0:
            raise SystemExit(
                f"{kernel.name}: private={values['private']} scratch={values['scratch']}"
            )
        for field, limit in limits[kind].items():
            if values[field] > limit:
                raise SystemExit(
                    f"{kernel.name}: {field}={values[field]} exceeds {limit}"
                )
        print(
            f"{kind}: lds={values['lds']} vgpr={values['vgpr']} "
            f"private={values['private']} scratch={values['scratch']}"
        )
    expected_counts = {"t1_qk": 2, "qk": 2, "partial": 8, "merge": 2}
    mismatched = [
        f"{kind}={counts[kind]} (expected {expected})"
        for kind, expected in expected_counts.items()
        if counts[kind] != expected
    ]
    if mismatched:
        raise SystemExit("split-512 static gate specialization mismatch: " + ", ".join(mismatched))
    print("split-512 resource gate passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("isa", "resources"))
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args()
    assembly = args.assembly.read_text(encoding="utf-8")
    selected = kernels(assembly)
    if args.mode == "isa":
        check_isa(selected)
    else:
        check_resources(selected)


if __name__ == "__main__":
    main()
