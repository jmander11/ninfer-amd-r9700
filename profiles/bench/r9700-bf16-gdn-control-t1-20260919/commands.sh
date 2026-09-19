#!/usr/bin/env bash
set -euo pipefail

root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$root/profiles/bench/r9700-bf16-gdn-control-t1-20260919"
source_file="$root/tools/r9700/bf16_gdn_control_t1_qual.hip"
checker="$root/tools/r9700/check_bf16_gdn_control_t1_static.py"
artifacts="$package/artifacts"
executable="$artifacts/bf16_gdn_control_t1_qual"
assembly="$artifacts/bf16_gdn_control_t1_qual.s"
receipt="$artifacts/static-receipt.txt"
report="$package/report.json"
profile=/sys/class/drm/card2/device/power_dpm_force_performance_level
source_sha=01d4f15a73408801e61ad83568a1a5ba09f1db70e23d7da5a4ef11dd33c2ab27
checker_sha=d77d5c0033d072083e8c352bade23d2ea048ea1656793cd15ee330cae67f5354

preflight() {
  test "$PWD" = "$root"
  test -x /opt/rocm/bin/hipcc
  test -x /usr/bin/sha256sum
  test -r "$profile"
  test "$(<"$profile")" = auto
  test ! -e "$report"
  test ! -e "$artifacts"
  test "$(sha256sum "$source_file" | cut -d' ' -f1)" = "$source_sha"
  test "$(sha256sum "$checker" | cut -d' ' -f1)" = "$checker_sha"
  python3 -m json.tool "$package/plan.json" >/dev/null
  local unresolved_marker='UN''BOUND'
  if rg -n "$unresolved_marker" "$package"; then
    echo "package contains an unresolved placeholder" >&2
    return 1
  fi
  echo "PASS: fresh BF16 GDN control T1 package; power profile auto"
}

case "${1:-}" in
  --preflight)
    test "$#" -eq 1
    preflight
    ;;
  --measure)
    test "$#" -eq 1
    preflight
    mkdir "$artifacts"
    /opt/rocm/bin/hipcc -std=c++20 -O3 --offload-arch=gfx1201 -Wall -Wextra -Werror \
      "$source_file" -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -o "$executable"
    /opt/rocm/bin/hipcc -std=c++20 -O3 --offload-arch=gfx1201 -S \
      "$source_file" -o "$assembly"
    python3 "$checker" "$assembly" >"$receipt"
    "$executable" --output "$report" --assembly "$assembly" --static-receipt "$receipt"
    test -s "$report"
    test "$(<"$profile")" = auto
    ;;
  *)
    echo "usage: $0 --preflight|--measure" >&2
    exit 2
    ;;
esac
