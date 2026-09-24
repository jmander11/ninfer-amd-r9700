#!/usr/bin/env bash
set -euo pipefail

root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$root/profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-retry-20260919"
test "$PWD" = "$root"

case "${1:-}" in
  --preflight)
    test "$#" -eq 1
    PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile "$package/prepare.py" "$package/run.py"
    python3 "$package/prepare.py" --preflight
    ;;
  --measure)
    test "$#" -eq 1
    PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile "$package/prepare.py" "$package/run.py"
    python3 "$package/prepare.py" --write-plan
    python3 "$package/run.py"
    ;;
  *)
    echo "usage: $0 --preflight|--measure" >&2
    exit 2
    ;;
esac
