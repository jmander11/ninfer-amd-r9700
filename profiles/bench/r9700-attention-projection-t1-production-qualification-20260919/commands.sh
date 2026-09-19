#!/usr/bin/env bash
set -euo pipefail

root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$root/profiles/bench/r9700-attention-projection-t1-production-qualification-20260919"
build="$package/build"
report="$package/qualification.json"
test "$PWD" = "$root"

compile_host() {
  /opt/rocm/bin/hipcc -O3 -std=c++20 --offload-arch=gfx1201 \
    -I"$root/include" -I"$root/src" -I"$root/tools/r9700" \
    -isystem /opt/rocm/include -Wall -Wextra -Werror -Wno-unused-function \
    -DNINFER_SOURCE_DIR='"/ssdpool2nvme/local_llm/ninfer-amd-r9700"' \
    "$root/tools/r9700/attention_projection_t1_op_qual.hip" \
    "$root/src/ops/r9700/attention_projection/attention_projection.hip" \
    "$root/src/ops/r9700/linear/r9700_linear.hip" \
    "$root/src/core/device.hip" "$root/src/core/tensor.cpp" "$root/src/core/dtype.cpp" \
    -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -o "$1"
}

compile_device() {
  /opt/rocm/bin/hipcc -O3 -std=c++20 --offload-arch=gfx1201 \
    -I"$root/include" -I"$root/src" -isystem /opt/rocm/include \
    -Wno-unused-command-line-argument --offload-device-only -S \
    "$root/src/ops/r9700/attention_projection/attention_projection.hip" -o "$1"
}

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
    compile_host "$build/qual"
    compile_device "$build/op.s"
    python3 "$root/tools/r9700/check_attention_projection_t1_op_static.py" \
      "$build/op.s" > "$build/static-receipt.txt"
    "$build/qual" --out-json "$report" --assembly "$build/op.s" \
      --static-receipt "$build/static-receipt.txt"
    ;;
  *)
    echo "usage: $0 --preflight|--measure" >&2
    exit 2
    ;;
esac
