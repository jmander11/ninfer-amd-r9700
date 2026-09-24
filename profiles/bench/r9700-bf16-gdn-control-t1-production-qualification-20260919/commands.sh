#!/usr/bin/env bash
set -euo pipefail

root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$root/profiles/bench/r9700-bf16-gdn-control-t1-production-qualification-20260919"
build="$root/build-r9700-bf16-gdn-control-t1-direct"
artifacts="$package/artifacts"
report="$package/report.json"

preflight() {
  test "$PWD" = "$root"
  test -x /opt/rocm/bin/hipcc
  test ! -e "$build"
  test ! -e "$report"
  test ! -e "$artifacts"
  python3 -m json.tool "$package/plan.json" >/dev/null
  PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile "$package/run_qualification.py" \
    "$package/write_report.py" \
    "$root/tools/r9700/check_bf16_gdn_projected_control_t1_static.py"
  local unresolved_marker='UN''BOUND'
  if rg -n "$unresolved_marker" "$package"; then
    echo "package contains an unresolved placeholder" >&2
    return 1
  fi
  echo "PASS: fresh BF16 GDN T1 production-symbol package; device-bound power check occurs at measurement"
}

case "${1:-}" in
  --preflight)
    test "$#" -eq 1
    preflight
    ;;
  --measure)
    test "$#" -eq 1
    preflight
    cmake -S "$root" -B "$build" -GNinja \
      -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
      -DCMAKE_BUILD_TYPE=Release \
      -DNINFER_BUILD_APPS=OFF \
      -DNINFER_BUILD_BENCHMARKS=OFF \
      -DNINFER_R9700_BF16_GDN_CONTROL_T1_CANDIDATE=1
    cmake --build "$build" --target ninfer_r9700_gdn_qual -j4
    mkdir "$artifacts"
    /opt/rocm/bin/hipcc -std=c++20 -O3 --offload-arch=gfx1201 \
      -I"$root/include" -I"$root/src" -I"$root/src/targets/qwen3/export" \
      -I"$root/src/targets/qwen3_8_27b/export" -I"$root/third_party" \
      -S "$root/src/ops/r9700/gdn/gdn_ops.hip" -o "$artifacts/gdn_ops.s"
    python3 "$root/tools/r9700/check_bf16_gdn_projected_control_t1_static.py" \
      "$artifacts/gdn_ops.s" >"$artifacts/static-receipt.txt"
    python3 "$package/run_qualification.py"
    python3 "$package/write_report.py"
    ;;
  *)
    echo "usage: $0 --preflight|--measure" >&2
    exit 2
    ;;
esac
