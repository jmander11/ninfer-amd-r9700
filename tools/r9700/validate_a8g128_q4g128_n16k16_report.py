#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, stat
from pathlib import Path
from tools.r9700.run_a8g128_q4g128_n16k16_gate import validate_report_payload

def main()->None:
 ap=argparse.ArgumentParser();ap.add_argument("report",type=Path);a=ap.parse_args()
 p=Path(os.path.abspath(a.report));s=os.lstat(p)
 if not stat.S_ISREG(s.st_mode) or p.is_symlink() or s.st_nlink<1:raise RuntimeError("report must be a real regular file")
 validate_report_payload(json.loads(p.read_text()));print("PASS report_recomputed=true hashes_reopened=true")
if __name__=="__main__":main()
