#!/usr/bin/env bash
set -euo pipefail

repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$repo/profiles/bench/r9700-fp8-target-gate-up-t5t6-qualification-20260906"
binary="$repo/tools/r9700/build/fp8_gate_up_small_t_qual"
assembly="$repo/tools/r9700/build/fp8_gate_up_small_t.s"
checker="$repo/tools/r9700/check_fp8_gate_up_small_t_static.py"
source_commit=337940345d20cd0211fccd10bc4f7e9d0181bc2d
source_tree=69601f7080e187f4c33d51b9cc7c232bad1581f6
runtime_path=/opt/rocm/lib

test "$PWD" = "$repo"
test "$#" -eq 0
sha256sum -c --strict "$package/prepared.sha256"
test "$(git rev-parse "$source_commit^{commit}")" = "$source_commit"
test "$(git rev-parse "$source_commit^{tree}")" = "$source_tree"
git diff --quiet "$source_commit" -- \
  tools/r9700/fp8_gate_up_small_t_qual.h \
  tools/r9700/fp8_gate_up_small_t_qual.hip \
  tools/r9700/fp8_gate_up_small_t_harness.hip \
  tools/r9700/check_fp8_gate_up_small_t_static.py \
  tools/r9700/test_check_fp8_gate_up_small_t_static.py \
  tools/r9700/fp8_gate_up_small_t.mk \
  src/ops/r9700/linear/fp8_activation.hip \
  src/ops/r9700/linear/linear_execution.hip

test "$(sha256sum "$binary" | cut -d' ' -f1)" = 4611e36dc2f1a5324a94a0da7f00c402b052f986dbf8ac4c504831d4327f4ece
test "$(stat -Lc %s "$binary")" = 113272
test "$(sha256sum "$assembly" | cut -d' ' -f1)" = a463e37e04512213013302769fa946c2fd1a57565533782f60797f107f46071d
test "$(stat -Lc %s "$assembly")" = 63959
test "$(readlink -f /opt/rocm/core-10.0/lib/llvm/bin/clang++)" = /opt/rocm/core-10.0/lib/llvm/bin/clang-23
test "$(sha256sum /opt/rocm/core-10.0/lib/llvm/bin/clang++ | cut -d' ' -f1)" = 241bf4da7ec39bc00b68ed74f6be751516d9892ed990e8c7fa372bad18500247
test "$(readlink -f /opt/rocm/bin/hipcc)" = /opt/rocm/core-10.0/bin/hipcc
test "$(sha256sum /opt/rocm/bin/hipcc | cut -d' ' -f1)" = 7b95d430bb8c4d4237f9b4935dbd6440e2067fde0979cc69bffb26ebd021464c
test "$(readlink -f /usr/bin/python3)" = /usr/bin/python3.12
test "$(sha256sum /usr/bin/python3 | cut -d' ' -f1)" = 1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118
test "$(readlink -f /opt/rocm/lib/libhipblaslt.so.1)" = /opt/rocm/core-10.0/lib/libhipblaslt.so.1.4
test "$(sha256sum /opt/rocm/lib/libhipblaslt.so.1 | cut -d' ' -f1)" = 1e8545626323f9b22c65d36492e1686b831cb85c8f7c8be3fc21c0d6cbb0d055
test "$(readlink -f /opt/rocm/lib/libamdhip64.so.7)" = /opt/rocm/core-10.0/lib/libamdhip64.so.7.15.26333-0000000
test "$(sha256sum /opt/rocm/lib/libamdhip64.so.7 | cut -d' ' -f1)" = 817aeadfd9f62b68831ad89993163c7f1f470f30595e7e0da5fdc942193142a8
ldd_output=$(LD_LIBRARY_PATH="$runtime_path" ldd "$binary")
! grep -q 'not found' <<<"$ldd_output"
grep -Fq 'libhipblaslt.so.1 => /opt/rocm/lib/libhipblaslt.so.1' <<<"$ldd_output"
grep -Fq 'libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7' <<<"$ldd_output"

if env | grep -Eq '^(LD_PRELOAD|LD_AUDIT|HIP_FORCE_QUEUE_PROFILING|AMD_SERIALIZE_KERNEL|AMD_SERIALIZE_COPY|GPU_DUMP_CODE_OBJECT|ROCPROF|ROCP_|ROCTRACER_|ROCTX_|HSA_TOOLS_|HIP_TRACE_|AQLPROFILE_|ATT_PROFILE)'; then
  echo 'profiling/injection environment is incompatible with unprofiled HIP-event authority' >&2
  exit 2
fi
test "$(cat /sys/class/drm/card2/device/vendor)" = 0x1002
test "$(cat /sys/class/drm/card2/device/device)" = 0x7551
test "$(basename "$(readlink -f /sys/class/drm/card2/device)")" = 0000:13:00.0
test "$(cat /sys/class/drm/card2/device/power_dpm_force_performance_level)" = auto

outputs=(regression.stdout regression.stderr regression.exit benchmark.stdout benchmark.stderr
  benchmark.exit static.stdout hardware.txt power-before.txt power-after.txt cell-t5.json
  cell-t6.json summary.json result.sha256)
/usr/bin/python3 "$package/preflight.py"
for output in "${outputs[@]}"; do
  test ! -e "$package/$output"
  test ! -L "$package/$output"
done

printf 'device=0\nname=AMD Radeon AI PRO R9700\narchitecture=gfx1201\nwavefront=32\npci=0000:13:00.0\nvendor=0x1002\ndevice_id=0x7551\n' > "$package/hardware.txt"
cat /sys/class/drm/card2/device/power_dpm_force_performance_level > "$package/power-before.txt"
/usr/bin/python3 "$checker" "$assembly" > "$package/static.stdout"

close_available() {
  local files=()
  local name
  for name in regression.stdout regression.stderr regression.exit benchmark.stdout \
    benchmark.stderr benchmark.exit static.stdout hardware.txt power-before.txt power-after.txt \
    cell-t5.json cell-t6.json summary.json; do
    if test -f "$package/$name"; then files+=("$package/$name"); fi
  done
  sha256sum "${files[@]}" > "$package/result.sha256"
  sha256sum -c --strict "$package/result.sha256"
}

set +e
LD_LIBRARY_PATH="$runtime_path" "$binary" --regression \
  > "$package/regression.stdout" 2> "$package/regression.stderr"
regression_status=$?
set -e
printf '%s\n' "$regression_status" > "$package/regression.exit"
test "$(cat /sys/class/drm/card2/device/power_dpm_force_performance_level)" = auto
if test "$regression_status" -ne 0; then
  cat /sys/class/drm/card2/device/power_dpm_force_performance_level > "$package/power-after.txt"
  close_available
  exit "$regression_status"
fi

set +e
LD_LIBRARY_PATH="$runtime_path" "$binary" --benchmark \
  > "$package/benchmark.stdout" 2> "$package/benchmark.stderr"
benchmark_status=$?
set -e
printf '%s\n' "$benchmark_status" > "$package/benchmark.exit"
cat /sys/class/drm/card2/device/power_dpm_force_performance_level > "$package/power-after.txt"
test "$(cat "$package/power-after.txt")" = auto
if test "$benchmark_status" -ne 0; then
  close_available
  exit "$benchmark_status"
fi

set +e
/usr/bin/python3 "$package/analyze.py" \
  --raw "$package/benchmark.stdout" \
  --cell-t5 "$package/cell-t5.json" \
  --cell-t6 "$package/cell-t6.json" \
  --summary "$package/summary.json"
analysis_status=$?
set -e
if test "$analysis_status" -ne 0 -a "$analysis_status" -ne 1; then
  close_available
  exit "$analysis_status"
fi
close_available
exit "$analysis_status"
