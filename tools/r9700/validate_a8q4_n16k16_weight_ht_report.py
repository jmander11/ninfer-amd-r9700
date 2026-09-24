#!/usr/bin/env python3
"""Revalidate an immutable W-only DEVICE_HT qualification report."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from tools.r9700.run_a8q4_n16k16_weight_ht_gate import validate_report_payload
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("report",type=Path);a=p.parse_args()
 if not a.report.is_file() or a.report.is_symlink():raise RuntimeError("report must be a regular non-symlink")
 validate_report_payload(json.loads(a.report.read_text()));print("PASS")
if __name__=="__main__":main()
