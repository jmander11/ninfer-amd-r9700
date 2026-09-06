#!/usr/bin/env python3
"""Reopen the retained BF16 evidence reusable by an exact chunk-4096 campaign."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.ppl import run


BF16 = REPO / "profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json"
REPEAT = REPO / "profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json"
def main() -> None:
    repeat = run.validate_bf16_repeat_comparison(REPEAT, BF16)
    print(json.dumps({
        "bf16_chunk4096": repeat,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
