#!/usr/bin/env bash
set -euo pipefail

root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$root/profiles/bench/r9700-a8q4-pair-wmma-c2c4-design-20260919"
build="$package/build"
report="$package/qualification.json"
test "$PWD" = "$root"

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
    mkdir -p "$build"
    /opt/rocm/bin/hipcc -O3 -std=c++20 --offload-arch=gfx1201 \
      -I"$root/src" -isystem /opt/rocm/include -Wall -Wextra -Werror \
      -Wno-unused-function -DNINFER_SOURCE_DIR='"/ssdpool2nvme/local_llm/ninfer-amd-r9700"' \
      "$root/tools/r9700/a8q4_pair_wmma_c2c4_qual.hip" \
      "$root/src/ops/r9700/linear/r9700_linear.hip" \
      -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -o "$build/qual"
    /opt/rocm/bin/hipcc -O3 -std=c++20 --offload-arch=gfx1201 \
      -I"$root/src" -isystem /opt/rocm/include -Wno-unused-command-line-argument \
      -DNINFER_SOURCE_DIR='"/ssdpool2nvme/local_llm/ninfer-amd-r9700"' \
      --offload-device-only -S "$root/tools/r9700/a8q4_pair_wmma_c2c4_qual.hip" \
      -o "$build/qual.s"
    python3 "$root/tools/r9700/check_a8q4_pair_wmma_c2c4_static.py" \
      "$build/qual.s" > "$build/static-receipt.txt"
    "$build/qual" --out-json "$report" --assembly "$build/qual.s" \
      --static-receipt "$build/static-receipt.txt" \
      --whole-baselines "$package/whole-baselines.json"
    ;;
  *)
    echo "usage: $0 --preflight|--measure" >&2
    exit 2
    ;;
esac
