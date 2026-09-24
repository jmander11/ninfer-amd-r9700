#!/usr/bin/env bash
set -euo pipefail

root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$root/profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919"
control="$root/build-r9700-bf16-gdn-control-t1-control"
candidate="$root/build-r9700-bf16-gdn-control-t1-candidate"

preflight() {
  test "$PWD" = "$root"
  test -s "$root/profiles/bench/r9700-bf16-gdn-control-t1-production-qualification-20260919/report.json"
  test ! -e "$control"
  test ! -e "$candidate"
  test ! -e "$package/plan.json"
  test ! -e "$package/results"
  PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile "$package/prepare.py" "$package/run.py"
  python3 -m json.tool "$package/package-plan.json" >/dev/null
  local unresolved_marker='UN''BOUND'
  if rg -n "$unresolved_marker" "$package"; then
    echo "package contains an unresolved placeholder" >&2
    return 1
  fi
  echo "PASS: fresh BF16 GDN T1 whole A/B package; device-bound power check occurs at measurement"
}

configure_build() {
  local directory=$1 selector=$2
  cmake -S "$root" -B "$directory" -GNinja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
    -DNINFER_BUILD_BENCHMARKS=ON \
    -DNINFER_R9700_BF16_GDN_CONTROL_T1_CANDIDATE="$selector"
  cmake --build "$directory" --target ninfer_bench -j4
}

case "${1:-}" in
  --preflight)
    test "$#" -eq 1
    preflight
    ;;
  --measure)
    test "$#" -eq 1
    preflight
    configure_build "$control" 0
    configure_build "$candidate" 1
    python3 "$package/prepare.py"
    python3 "$package/run.py"
    ;;
  *)
    echo "usage: $0 --preflight|--measure" >&2
    exit 2
    ;;
esac
