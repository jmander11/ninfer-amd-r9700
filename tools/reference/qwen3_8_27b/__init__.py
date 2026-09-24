"""Qwen3.8-27B R9700-candidate implementation-profile diagnostic.

The artifact contract remains importable without the optional model-execution
dependencies. Model and weight-store exports are loaded only when requested.
"""

from __future__ import annotations

from typing import Any

from .bindings import (
    ArtifactBinding,
    AxisView,
    BindingError,
    BoundResource,
    LogicalRowView,
    PhysicalBlock,
    VisionArtifactBinding,
    WeightObject,
)


def __getattr__(name: str) -> Any:
    if name == "RefModel":
        from .model import RefModel

        return RefModel
    if name in {"MemoryPlan", "WeightStore"}:
        from .weights import MemoryPlan, WeightStore

        return {"MemoryPlan": MemoryPlan, "WeightStore": WeightStore}[name]
    raise AttributeError(name)

__all__ = [
    "ArtifactBinding",
    "AxisView",
    "BindingError",
    "BoundResource",
    "LogicalRowView",
    "MemoryPlan",
    "PhysicalBlock",
    "VisionArtifactBinding",
    "RefModel",
    "WeightObject",
    "WeightStore",
]
