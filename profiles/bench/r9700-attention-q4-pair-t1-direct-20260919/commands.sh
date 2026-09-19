#!/usr/bin/env bash
set -euo pipefail

repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$repo/profiles/bench/r9700-attention-q4-pair-t1-direct-20260919"
build="$package/build"
report="$package/qualification.json"
test "$PWD" = "$repo"

case "${1:-}" in
  --preflight)
    test "$#" -eq 1
    python3 "$package/preflight.py"
    ;;
  --measure)
    test "$#" -eq 1
    python3 "$package/preflight.py"
    test ! -e "$report"
    test ! -L "$report"
    test "$(cat /sys/class/drm/card2/device/power_dpm_force_performance_level)" = auto
    mkdir -p "$build"

    /opt/rocm/bin/hipcc -O3 -std=c++20 --offload-arch=gfx1201 \
      -I"$repo/src" -isystem /opt/rocm/include -Wall -Wextra -Werror \
      -Wno-unused-function -DNINFER_SOURCE_DIR='"/ssdpool2nvme/local_llm/ninfer-amd-r9700"' \
      "$repo/tools/r9700/a8q4_attention_pair_t1_qual.hip" \
      "$repo/src/ops/r9700/linear/r9700_linear.hip" \
      -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -o "$build/a8q4_attention_pair_t1_qual"

    /opt/rocm/bin/hipcc -O3 -std=c++20 --offload-arch=gfx1201 \
      -I"$repo/src" -isystem /opt/rocm/include -Wno-unused-command-line-argument \
      -DNINFER_SOURCE_DIR='"/ssdpool2nvme/local_llm/ninfer-amd-r9700"' \
      --offload-device-only -S "$repo/tools/r9700/a8q4_attention_pair_t1_qual.hip" \
      -o "$build/a8q4_attention_pair_t1_qual.s"

    python3 "$repo/tools/r9700/check_a8q4_attention_pair_t1_static.py" \
      "$build/a8q4_attention_pair_t1_qual.s" > "$build/static-receipt.txt"

    "$build/a8q4_attention_pair_t1_qual" --out-json "$report" \
      --assembly "$build/a8q4_attention_pair_t1_qual.s"
    ;;
  *)
    echo "usage: $0 --preflight|--measure" >&2
    exit 2
    ;;
esac
