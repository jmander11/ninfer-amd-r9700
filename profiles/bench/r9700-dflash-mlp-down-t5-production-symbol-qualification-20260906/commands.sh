#!/usr/bin/env bash
set -euo pipefail

repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$repo/profiles/bench/r9700-dflash-mlp-down-t5-production-symbol-qualification-20260906"
binary="$repo/tools/r9700/build/a8q4_dflash_mlp_down_small_t_qual"
assembly="$repo/tools/r9700/build/a8q4_dflash_mlp_down_small_t.s"
checker="$repo/tools/r9700/check_a8q4_dflash_mlp_down_small_t_static.py"
source_commit=d1a9b6fb33a843363fba3ad46d6569071df1ab66

test "$PWD" = "$repo"
test "$#" -eq 0
sha256sum -c --strict "$package/prepared.sha256"
test "$(git rev-parse "$source_commit^{commit}")" = "$source_commit"
test "$(git rev-parse "$source_commit^{tree}")" = 557bbda04f2926319bf7c25f26eb9ec193d74824
git diff --quiet "$source_commit" -- CMakeLists.txt \
  src/ops/r9700/linear/r9700_linear.h src/ops/r9700/linear/r9700_linear.hip \
  src/ops/r9700/linear/r9700_q4_activation_profile.h \
  tools/r9700/Makefile tools/r9700/a8q4_dflash_mlp_down_small_t_qual.hip \
  tools/r9700/a8q4_dflash_small_t_qual.hip tools/r9700/a8q4_shape_sweep_qual.hip \
  tools/r9700/check_a8q4_dflash_mlp_down_small_t_static.py \
  tools/r9700/test_check_a8q4_dflash_mlp_down_small_t_static.py
test "$(sha256sum "$binary" | cut -d' ' -f1)" = 2add29d2c5065ff22215a7ffd76fd80ce65020fe6c99ebd80ba58bfde45d95b6
test "$(sha256sum "$assembly" | cut -d' ' -f1)" = bc04258672694a512128300a3303f72a1975bce1b89c80c3b0326eead6a9639a
ldd_output=$(LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib ldd "$binary")
! grep -q 'not found' <<<"$ldd_output"
grep -Fq 'libamdhip64.so.7 => /opt/rocm/core-10.0/lib/libamdhip64.so.7' <<<"$ldd_output"
for injected in HSA_TOOLS_LIB ROCPROFILER_TOOL_LIBRARIES ROCP_TOOL_LIBRARIES \
  ROCPROFILER_OUTPUT_PATH ROCPROFILER_OUTPUT_FILE_NAME LD_PRELOAD; do
  test -z "${!injected-}"
done
if env | grep -Eq '^(LD_PRELOAD|LD_AUDIT|HIP_FORCE_QUEUE_PROFILING|AMD_SERIALIZE_KERNEL|AMD_SERIALIZE_COPY|GPU_DUMP_CODE_OBJECT|ROCPROF|ROCP_|ROCTRACER_|ROCTX_|HSA_TOOLS_|HIP_TRACE_|AQLPROFILE_|ATT_PROFILE)'; then
  echo 'profiling/injection environment is incompatible with unprofiled HIP-event authority' >&2
  exit 2
fi
test "$(cat /sys/class/drm/card2/device/vendor)" = 0x1002
test "$(cat /sys/class/drm/card2/device/device)" = 0x7551
test "$(cat /sys/class/drm/card2/device/power_dpm_force_performance_level)" = auto
for output in cell-t5-production.json cell-t5-production.stdout cell-t5-production.stderr \
  cell-t5-production.exit summary.json static.stdout result.sha256; do
  test ! -e "$package/$output"
done
/usr/bin/python3 "$checker" "$assembly" > "$package/static.stdout"
set +e
"$binary" 5120 17408 5 --out-json "$package/cell-t5-production.json" \
  > "$package/cell-t5-production.stdout" 2> "$package/cell-t5-production.stderr"
status=$?
set -e
printf '%s\n' "$status" > "$package/cell-t5-production.exit"
test "$(cat /sys/class/drm/card2/device/power_dpm_force_performance_level)" = auto
if test "$status" -ne 0 -a "$status" -ne 1; then exit "$status"; fi
set +e
/usr/bin/python3 "$package/analyze.py" --report "$package/cell-t5-production.json" \
  --summary "$package/summary.json"
analysis_status=$?
set -e
test "$analysis_status" -eq "$status"
sha256sum "$package/cell-t5-production.json" "$package/cell-t5-production.stdout" \
  "$package/cell-t5-production.stderr" "$package/cell-t5-production.exit" \
  "$package/static.stdout" "$package/summary.json" > "$package/result.sha256"
sha256sum -c --strict "$package/result.sha256"
exit "$analysis_status"
