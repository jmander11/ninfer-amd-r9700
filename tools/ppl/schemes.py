"""Fixed R9700 PPL profile registry.

Profiles select separate binaries in ``run.py``; they never select a runtime KV
codec. The BF16 authority is an independently built reference implementation.
G16 and G32 are separately configured FP8-K/INT4-V executables over the same
explicit R9700 artifact, used only until the accuracy/performance gate selects
one static product ABI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    reference: bool
    value_group: int | None


BASELINE = "bf16-reference"

PROFILES: dict[str, Profile] = {
    BASELINE: Profile(BASELINE, True, None),
    "r9700-g16": Profile("r9700-g16", False, 16),
    "r9700-g32": Profile("r9700-g32", False, 32),
}

ORDER: tuple[str, ...] = (BASELINE, "r9700-g16", "r9700-g32")
