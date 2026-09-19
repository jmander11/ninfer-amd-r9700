#!/usr/bin/env bash
set -euo pipefail

root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$root/profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919"
test "$PWD" = "$root"

case "${1:-}" in
  --preflight)
    test "$#" -eq 1
    python3 "$package/preflight.py"
    ;;
  --measure)
    test "$#" -eq 1
    python3 "$package/preflight.py"
    python3 "$package/measure.py"
    ;;
  *)
    echo "usage: $0 --preflight|--measure" >&2
    exit 2
    ;;
esac
