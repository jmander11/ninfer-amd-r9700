#!/usr/bin/env python3
"""Revalidate a frozen group-major A8G64 qualification report."""
from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

from tools.r9700.run_a8q4_group_major_activation_gate import OUTPUT, validate_report


def main() -> None:
    parser=argparse.ArgumentParser()
    args=parser.parse_args()
    path=OUTPUT
    info=os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError("report must be one regular inode")
    with path.open("r") as handle:value=json.load(handle)
    if not isinstance(value,dict):raise ValueError("report root")
    validate_report(value)
    print("PASS group-major A8G64 report revalidated")

if __name__=="__main__":main()
