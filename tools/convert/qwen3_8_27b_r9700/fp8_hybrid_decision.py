"""Read the target-owned four-role FP8/Q4 qualification decision record."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


_IDENTITY = re.compile(
    r'NINFER_QWEN38_FP8_HYBRID_IDENTITY\(\s*"([^"]+)",\s*"([^"]+)",\s*"([0-9a-f]{64})"\s*\)',
    re.MULTILINE,
)
_MATRIX = re.compile(r'NINFER_QWEN38_FP8_HYBRID_MATRIX\("([^"]+)"\)')


@dataclass(frozen=True, slots=True)
class Fp8HybridDecision:
    weights_id: str
    recipe_id: str
    selection_sha256: str
    matrix_names: tuple[str, ...]


def _authority_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "src/targets/qwen3_8_27b/impl/load/fp8_hybrid_selection.inc"
    )


def load_decision() -> Fp8HybridDecision:
    text = _authority_path().read_text(encoding="utf-8")
    identity = _IDENTITY.findall(text)
    if len(identity) != 1:
        raise ValueError("FP8 hybrid authority must contain exactly one identity record")
    names = tuple(_MATRIX.findall(text))
    weights_id, recipe_id, expected_digest = identity[0]
    actual_digest = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()
    if len(names) != 144 or len(set(names)) != len(names):
        raise ValueError("FP8 hybrid authority must select exactly 144 unique matrices")
    if actual_digest != expected_digest:
        raise ValueError("FP8 hybrid authority matrix-name digest differs")
    return Fp8HybridDecision(weights_id, recipe_id, expected_digest, names)


DECISION = load_decision()


def selected_matrix_names() -> frozenset[str]:
    return frozenset(DECISION.matrix_names)


__all__ = ["DECISION", "Fp8HybridDecision", "load_decision", "selected_matrix_names"]
