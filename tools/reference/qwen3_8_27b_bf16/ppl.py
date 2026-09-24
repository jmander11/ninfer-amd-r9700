#!/usr/bin/env python3
"""Executable wrapper for tools.ppl.run's scorer-binary protocol."""

from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.reference.qwen3_8_27b_bf16.scorer import main


if __name__ == "__main__":
    raise SystemExit(main())
