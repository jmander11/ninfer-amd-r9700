#!/usr/bin/env bash
set -euo pipefail

root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
plan_dir="$root/profiles/bench/r9700-fp8-gate-up-m128n256-retained-reopen-20260906"
attempt="$plan_dir/attempt-4"
temporary=$(mktemp)
trap 'rm -f -- "$temporary"' EXIT

cd "$root"
sha256sum --check "$plan_dir/prepared.sha256"
python3 "$plan_dir/qualification-source/validate_retained.py" \
  --assembly "$plan_dir/retained-fp8-gate-up-m128n256.s" \
  --object "$plan_dir/retained-fp8-gate-up-m128n256.hip.o" \
  --resource-report "$attempt/resource-report.json" \
  --resource-stderr "$attempt/resource-stderr.log" --resource-exit-code 0 \
  --report "$attempt/report.json" --stderr "$attempt/stderr.log" \
  --ldd "$attempt/ldd.txt" --exit-code 1 --output "$temporary"
cmp "$temporary" "$attempt/decision.json"
echo "terminal retained M128N256 rejection validates"
