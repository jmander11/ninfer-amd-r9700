#!/usr/bin/env bash
set -euo pipefail

repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
root="$repo/profiles/rocprof/r9700-retained-production-p2048-trace-plan-20260906"
power=/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level
bench="$repo/build-r9700-rmsnorm-production-final-20260906/bench/ninfer_bench"
artifact="$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer"
corpus="$repo/bench/fixtures/bench_corpus.ids"
profiler=/opt/rocm/core-10.0/bin/rocprofv3
analyzer="$repo/tools/bench/analyze_retained_prefill_trace.py"
authority="$repo/profiles/bench/r9700-scalar-base-production-p2048-c1-ab-20260905.json"
report="$root/benchmark-report.json"
raw="$root/raw"
database="$raw/retained-production-p2048_results.db"
before="$root/power-profile-before.txt"
after="$root/power-profile-after.txt"
evidence="$root/evidence.json"

test "$PWD" = "$repo"
sha256sum -c "$root/prepared.sha256"
test "$(sha256sum "$bench" | cut -d' ' -f1)" = a7c9303bd213fa3dbdb29ca0cee73addef1de8ab6a6e6b231509e25239776425
test "$(sha256sum "$artifact" | cut -d' ' -f1)" = 040c6e7ed29c856718a638c00181975710d987b7d5f49f4cafbdf68911f7e7d2
test "$(sha256sum "$corpus" | cut -d' ' -f1)" = 27e4f63c17efe3f89b5cf278b3b1a42a737316ed4044d7d0d1d52437059d1002
test "$(sha256sum "$profiler" | cut -d' ' -f1)" = 2d220ded2167097e480e6866a916b575548d56bed8a823a4e505f7133868f0a6
test "$(sha256sum "$authority" | cut -d' ' -f1)" = 08463f48aae28aa0dd5ad4f5f05155a11deacda2d458db38e2f512a974be8e9f
test "$(sha256sum "$repo/tools/bench/analyze_whole_profile.py" | cut -d' ' -f1)" = 3df20e4ffe8bb1a8a7cdb228088cd180330a7829d02720a8c2e5f8a366a855a1
test "$(sha256sum "$analyzer" | cut -d' ' -f1)" = 8b02f8bb10173afb98d8e719ad25bac3a8c54ed81b54d1e28ceb08111e0c6a3c
test ! -e "$raw" && test ! -e "$report" && test ! -e "$before" && test ! -e "$after" && test ! -e "$evidence"
test "$(cat "$power")" = auto
(set -C; printf '%s\n' auto >"$before")

capture_after() {
  local rc=$?
  local value
  value=$(cat "$power" 2>/dev/null || true)
  (set -C; printf '%s\n' "$value" >"$after") || true
  if [[ "$value" != auto ]]; then exit 1; fi
  exit "$rc"
}
trap capture_after EXIT

"$profiler" --selected-regions -f rocpd -d "$raw" -o retained-production-p2048 \
  --marker-trace --kernel-trace --memory-copy-trace -- \
  "$bench" --weights "$artifact" --corpus "$corpus" --device 0 --concurrency 1 \
  -p 2048 --prefill-chunk 4096 --draft-tokens 0 --output json --output-file "$report" \
  -r 1 --warmup 1 --profile-measured

test "$(cat "$power")" = auto
(set -C; printf '%s\n' auto >"$after")
trap - EXIT
test -f "$database"
python3 "$analyzer" --plan "$root/plan.json" --root "$root" --out "$evidence"
