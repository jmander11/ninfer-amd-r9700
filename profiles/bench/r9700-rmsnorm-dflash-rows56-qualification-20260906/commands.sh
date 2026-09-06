#!/usr/bin/env bash
set -euo pipefail
repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$repo/profiles/bench/r9700-rmsnorm-dflash-rows56-qualification-20260906"
binary="$repo/tools/r9700/build/rmsnorm_dflash_rows56_qual"
assembly="$repo/tools/r9700/build/eager_ops.s"
test "$PWD" = "$repo"
test "$#" -le 1
test "$#" -eq 0 || test "$1" = --preflight-only
/usr/bin/python3 "$package/preflight.py"
sha256sum -c "$package/prepared.sha256"
if test "$#" -eq 1; then
  exit 0
fi
for variable in HSA_TOOLS_LIB ROCPROFILER_TOOL_LIBRARIES ROCP_TOOL_LIBRARIES \
  ROCPROFILER_OUTPUT_PATH ROCPROFILER_OUTPUT_FILE_NAME LD_PRELOAD LD_AUDIT \
  AMD_SERIALIZE_KERNEL ROC_SERIALIZE_KERNEL; do
  test -z "${!variable-}"
done
if env | grep -Eq '^(ROCP_|ROCPROF|ROCPROFILER|ROCTRACER|AQLPROFILE|ATT_PROFILE|HSA_TOOLS|ROCTX|HIP_TRACE|AMD_SERIALIZE_KERNEL|ROC_SERIALIZE_KERNEL|LD_PRELOAD|LD_AUDIT)='; then
  echo "profiling/injection environment is incompatible with timing authority" >&2
  exit 2
fi
test "$(cat /sys/class/drm/card2/device/vendor)" = 0x1002
test "$(cat /sys/class/drm/card2/device/device)" = 0x7551
test "$(cat /sys/class/drm/card2/device/power_dpm_force_performance_level)" = auto
for output in cells.json benchmark.stdout benchmark.stderr benchmark.exit summary.json static.stdout \
  hardware-before.txt hardware-after.txt result.sha256; do
  test ! -e "$package/$output" && test ! -L "$package/$output"
done
{
  readlink -f /sys/class/drm/card2/device
  cat /sys/class/drm/card2/device/vendor
  cat /sys/class/drm/card2/device/device
  cat /sys/class/drm/card2/device/power_dpm_force_performance_level
} > "$package/hardware-before.txt"
/usr/bin/python3 "$repo/tools/r9700/check_rmsnorm_dflash_rows56_static.py" "$assembly" > "$package/static.stdout"
set +e
"$binary" --benchmark --out-json "$package/cells.json" > "$package/benchmark.stdout" 2> "$package/benchmark.stderr"
status=$?
set -e
printf '%s\n' "$status" > "$package/benchmark.exit"
test "$status" -eq 0
test "$(cat /sys/class/drm/card2/device/power_dpm_force_performance_level)" = auto
{
  readlink -f /sys/class/drm/card2/device
  cat /sys/class/drm/card2/device/vendor
  cat /sys/class/drm/card2/device/device
  cat /sys/class/drm/card2/device/power_dpm_force_performance_level
} > "$package/hardware-after.txt"
cmp "$package/hardware-before.txt" "$package/hardware-after.txt"
/usr/bin/python3 "$package/analyze.py" --report "$package/cells.json" --summary "$package/summary.json"
(
  cd "$package"
  sha256sum prepared.sha256 plan.json preflight.py analyze.py test_analyze.py commands.sh \
    cells.json benchmark.stdout benchmark.stderr benchmark.exit summary.json static.stdout \
    hardware-before.txt hardware-after.txt \
    > result.sha256
)
