#!/usr/bin/env bash
set -euo pipefail

repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$repo/profiles/bench/r9700-dflash-mlp-down-t5-qualification-20260906"
binary="$repo/tools/r9700/build/a8q4_dflash_mlp_down_small_t_qual"
assembly="$repo/tools/r9700/build/a8q4_dflash_mlp_down_small_t.s"
checker="$repo/tools/r9700/check_a8q4_dflash_mlp_down_small_t_static.py"

test "$PWD" = "$repo"
test "$#" -eq 0
sha256sum -c "$package/prepared.sha256"
test "$(git rev-parse HEAD)" = 8bd921d757e08a576c59e5d7401c4ea36f1fbc9c
git diff --quiet HEAD -- tools/r9700/a8q4_dflash_mlp_down_small_t_qual.hip \
  tools/r9700/a8q4_dflash_small_t_qual.hip tools/r9700/a8q4_shape_sweep_qual.hip \
  tools/r9700/check_a8q4_dflash_mlp_down_small_t_static.py \
  src/ops/r9700/linear/r9700_linear.hip src/ops/r9700/linear/r9700_linear.h
test "$(sha256sum "$binary" | cut -d' ' -f1)" = 560ad12b7dace9817a6a056a9241fe59dfd599a14a7dc1fdc37eb9ecbfe33f47
test "$(sha256sum "$assembly" | cut -d' ' -f1)" = 92c71dad66dcc0e3e52f6de33779509ff0632de6ede4a928ffc306e2eea04079
test "$(sha256sum /usr/bin/python3 | cut -d' ' -f1)" = 1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118
test "$(sha256sum /opt/rocm/llvm/bin/clang++ | cut -d' ' -f1)" = 241bf4da7ec39bc00b68ed74f6be751516d9892ed990e8c7fa372bad18500247
test "$(sha256sum /opt/rocm/core-10.0/lib/libamdhip64.so.7 | cut -d' ' -f1)" = 817aeadfd9f62b68831ad89993163c7f1f470f30595e7e0da5fdc942193142a8
ldd_output=$(LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib ldd "$binary")
! grep -q 'not found' <<<"$ldd_output"
grep -Fq 'libamdhip64.so.7 => /opt/rocm/core-10.0/lib/libamdhip64.so.7' <<<"$ldd_output"
for injected in HSA_TOOLS_LIB ROCPROFILER_TOOL_LIBRARIES ROCP_TOOL_LIBRARIES \
  ROCPROFILER_OUTPUT_PATH ROCPROFILER_OUTPUT_FILE_NAME LD_PRELOAD; do
  test -z "${!injected-}"
done
if env | grep -Eq '^(ROCPROF|ROCPROFILER|HSA_TOOLS|ROCTX|HIP_TRACE|LD_PRELOAD)'; then
  echo 'profiling/injection environment is incompatible with unprofiled HIP-event authority' >&2
  exit 2
fi
test "$(cat /sys/class/drm/card2/device/vendor)" = 0x1002
test "$(cat /sys/class/drm/card2/device/device)" = 0x7551
test "$(cat /sys/class/drm/card2/device/power_dpm_force_performance_level)" = auto
for output in cell-t5.json cell-t5.stdout cell-t5.stderr cell-t5.exit summary.json static.stdout; do
  test ! -e "$package/$output"
done
/usr/bin/python3 "$checker" "$assembly" > "$package/static.stdout"
set +e
"$binary" 5120 17408 5 --out-json "$package/cell-t5.json" \
  > "$package/cell-t5.stdout" 2> "$package/cell-t5.stderr"
status=$?
set -e
printf '%s\n' "$status" > "$package/cell-t5.exit"
test "$(cat /sys/class/drm/card2/device/power_dpm_force_performance_level)" = auto
if test "$status" -ne 0 -a "$status" -ne 1; then
  exit "$status"
fi
set +e
/usr/bin/python3 "$package/analyze.py" --report "$package/cell-t5.json" \
  --summary "$package/summary.json"
analysis_status=$?
set -e
test "$analysis_status" -eq "$status"
exit "$analysis_status"
