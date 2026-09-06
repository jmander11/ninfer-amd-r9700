#!/usr/bin/env bash
set -euo pipefail

repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
root="$repo/profiles/rocprof/r9700-retained-production-p2048-trace-plan-20260906"
test "$PWD" = "$repo"
sha256sum -c "$root/analysis-resume.sha256"
test ! -e "$root/evidence.json"
test -f "$root/benchmark-report.json"
test -f "$root/raw/retained-production-p2048_results.db"
test "$(cat "$root/power-profile-before.txt")" = auto
test "$(cat "$root/power-profile-after.txt")" = auto
python3 "$repo/tools/bench/analyze_retained_prefill_trace.py" \
  --plan "$root/plan.json" --root "$root" --repair "$root/analysis-repair.json" \
  --out "$root/evidence.json"
