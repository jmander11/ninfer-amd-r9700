#!/usr/bin/env bash
set -euo pipefail

repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
root="$repo/profiles/rocprof/r9700-dflash-q4-c1-k4k5-owner-trace-plan-20260906"
bench="$repo/build-r9700-dense-selection-g32/bench/ninfer_bench"
artifact="$repo/out/qwen3.8-27b-r9700-q4g64-dflash2-q4-eval.ninfer"
corpus="$repo/bench/fixtures/bench_corpus.ids"
profiler=/opt/rocm/core-10.0/bin/rocprofv3
power=/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level
analyzer="$repo/tools/bench/analyze_dflash_trace.py"

test "$PWD" = "$repo"
test "$#" -eq 1
case "$1" in
  k4w5) k=4; w=5 ;;
  k5w6) k=5; w=6 ;;
  *) echo "usage: $0 k4w5|k5w6" >&2; exit 2 ;;
esac
cell="$1"
cell_root="$root/$cell"
test ! -e "$cell_root"
sha256sum -c "$root/prepared.sha256"
test "$(sha256sum "$bench" | cut -d' ' -f1)" = 62853a231c801bbfe696fad8d42bb566ae683bb230d4f52b6349b0fbc2132cf6
test "$(sha256sum "$artifact" | cut -d' ' -f1)" = ba39608b8a70e78e038e29154983c45017437803d9a77ea308a288c7908f9cdc
test "$(sha256sum "$artifact.conversion.json" | cut -d' ' -f1)" = b9d5511eb2f14f089091c9ae2c2cdc40626bcca4702fd5ac6f70b21a285973c5
test "$(sha256sum "$corpus" | cut -d' ' -f1)" = 27e4f63c17efe3f89b5cf278b3b1a42a737316ed4044d7d0d1d52437059d1002
test "$(sha256sum "$profiler" | cut -d' ' -f1)" = 2d220ded2167097e480e6866a916b575548d56bed8a823a4e505f7133868f0a6
test "$(sha256sum "$repo/build-r9700-dense-selection-g32/CMakeCache.txt" | cut -d' ' -f1)" = 1ff03c7e8a94beea3caaf66548da90c8e97422384b3c17efc30ef3d9d922a327
test "$(sha256sum "$repo/build-r9700-dense-selection-g32/compile_commands.json" | cut -d' ' -f1)" = be12b0405615531c82f27f074525cf45e737e35d865ef4044d5ca8df91dbb391
grep -qx 'NINFER_R9700_KV_VALUE_GROUP:STRING=32' "$repo/build-r9700-dense-selection-g32/CMakeCache.txt"
grep -qx 'NINFER_R9700_Q4_ACTIVATION_BITS:STRING=8' "$repo/build-r9700-dense-selection-g32/CMakeCache.txt"
grep -qx 'NINFER_R9700_W8_ACTIVATION_BITS:STRING=8' "$repo/build-r9700-dense-selection-g32/CMakeCache.txt"
test "$(cat "$power")" = auto
mkdir "$cell_root"
(set -C; printf '%s\n' auto >"$cell_root/power-before.txt")

capture_after() {
  local rc=$?
  local value
  value=$(cat "$power" 2>/dev/null || true)
  (set -C; printf '%s\n' "$value" >"$cell_root/power-after.txt") || true
  if [[ "$value" != auto ]]; then exit 1; fi
  exit "$rc"
}
trap capture_after EXIT

"$profiler" --selected-regions -f rocpd -d "$cell_root/raw" -o "$cell" \
  --marker-trace --kernel-trace --memory-copy-trace -- \
  "$bench" --weights "$artifact" --corpus "$corpus" --device 0 --concurrency 1 \
  --whole-pg 128,64 --max-ctx 256 --kv-capacity workload --prefill-chunk 4096 \
  --spec dflash --draft-tokens "$k" --dflash-verify-width "$w" --lm-head-draft \
  --no-device-graph --output json --output-file "$cell_root/benchmark-report.json" \
  -r 1 --warmup 1 --profile-measured

test "$(cat "$power")" = auto
(set -C; printf '%s\n' auto >"$cell_root/power-after.txt")
trap - EXIT
python3 "$analyzer" --plan "$root/plan.json" --cell "$cell" --out "$cell_root/evidence.json"
