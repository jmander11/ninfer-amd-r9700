#!/usr/bin/env python3
"""Compare fresh and prefix-append P129 Text-prefill boundary diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def fail(message: str) -> None:
    raise RuntimeError(message)


def load(path: Path) -> dict:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=reject_duplicates,
                       parse_constant=lambda token: fail(f"nonfinite JSON value {token}"))
    if not isinstance(value, dict):
        fail(f"trace is not an object: {path}")
    return value


def bf16_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def ordered_bf16(bits: int) -> int:
    return 0x8000 - (bits & 0x7FFF) if bits & 0x8000 else 0x8000 + bits


def integer(value: object, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        fail(f"{label} is invalid")
    return value


def validate(trace: dict, role: str) -> dict:
    expected_base, expected_tokens = (0, 129) if role == "fresh" else (128, 1)
    exact = {
        "artifact_type": "ninfer_qwen3_text_prefill_p129_tail_trace",
        "schema_version": 1,
        "diagnostic_only": True,
        "timing_evidence_eligible": False,
        "absolute_frontier": 129,
        "prefill_base": expected_base,
        "prefill_tokens": expected_tokens,
        "tail_hidden_kind": "final_rmsnorm_bf16",
        "target_logits_kind": "full_lm_head_bf16",
        "token_domain": 248077,
    }
    for key, value in exact.items():
        if trace.get(key) != value:
            fail(f"{role} {key} differs")
    hidden = trace.get("tail_hidden_bf16_bits")
    if not isinstance(hidden, list) or len(hidden) != 5120:
        fail(f"{role} tail hidden has invalid extent")
    checked_hidden = tuple(integer(value, f"{role} hidden[{index}]", 0, 0xFFFF)
                           for index, value in enumerate(hidden))
    if any(not math.isfinite(bf16_float(bits)) for bits in checked_hidden):
        fail(f"{role} tail hidden contains a nonfinite value")

    ranked = []
    for name in ("top1", "top2"):
        item = trace.get(name)
        if not isinstance(item, dict) or set(item) != {"token", "bf16_bits", "value"}:
            fail(f"{role} {name} is invalid")
        token = integer(item["token"], f"{role} {name} token", 0, 248076)
        bits = integer(item["bf16_bits"], f"{role} {name} bits", 0, 0xFFFF)
        value = item["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                not math.isfinite(value) or value != bf16_float(bits):
            fail(f"{role} {name} value differs from its BF16 bits")
        ranked.append((token, bits, float(value)))
    if ranked[0][0] == ranked[1][0] or ranked[0][2] < ranked[1][2] or \
            (ranked[0][2] == ranked[1][2] and ranked[0][0] > ranked[1][0]):
        fail(f"{role} top-two ordering is invalid")
    margin = trace.get("top1_top2_margin")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or \
            not math.isfinite(margin) or not math.isclose(
                float(margin), ranked[0][2] - ranked[1][2], rel_tol=0.0, abs_tol=1e-7):
        fail(f"{role} top-two margin is inconsistent")
    return {"hidden": checked_hidden, "top1": ranked[0], "top2": ranked[1],
            "margin": float(margin)}


def compare(fresh_trace: dict, append_trace: dict) -> dict:
    fresh = validate(fresh_trace, "fresh")
    append = validate(append_trace, "append")
    mismatches = [index for index, (left, right) in enumerate(
                  zip(fresh["hidden"], append["hidden"], strict=True)) if left != right]
    maximum_steps = max((abs(ordered_bf16(fresh["hidden"][index]) -
                             ordered_bf16(append["hidden"][index]))
                         for index in mismatches), default=0)
    maximum_absolute = max((abs(bf16_float(fresh["hidden"][index]) -
                                bf16_float(append["hidden"][index]))
                            for index in mismatches), default=0.0)
    if mismatches:
        classification = "final_normalized_tail_hidden_differs"
    elif fresh["top1"] != append["top1"] or fresh["top2"] != append["top2"]:
        classification = "same_tail_hidden_but_target_top2_differs"
    else:
        classification = "p129_prefill_boundary_is_exact"
    first = None if not mismatches else {
        "index": mismatches[0],
        "fresh_bits": fresh["hidden"][mismatches[0]],
        "append_bits": append["hidden"][mismatches[0]],
    }
    return {
        "artifact_type": "ninfer_qwen3_text_prefill_p129_tail_comparison",
        "schema_version": 1,
        "diagnostic_only": True,
        "timing_evidence_eligible": False,
        "classification": classification,
        "tail_hidden": {
            "exact": not mismatches,
            "mismatch_count": len(mismatches),
            "first_mismatch": first,
            "maximum_bf16_steps": maximum_steps,
            "maximum_absolute_difference": maximum_absolute,
        },
        "fresh": {"top1_token": fresh["top1"][0], "top1_bits": fresh["top1"][1],
                  "top2_token": fresh["top2"][0], "top2_bits": fresh["top2"][1],
                  "top1_top2_margin": fresh["margin"]},
        "append": {"top1_token": append["top1"][0], "top1_bits": append["top1"][1],
                   "top2_token": append["top2"][0], "top2_bits": append["top2"][1],
                   "top1_top2_margin": append["margin"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        fail(f"refusing to overwrite {args.out}")
    result = compare(load(args.fresh), load(args.append))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
