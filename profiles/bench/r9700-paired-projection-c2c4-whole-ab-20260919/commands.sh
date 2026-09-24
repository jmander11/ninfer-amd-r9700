#!/usr/bin/env bash
set -euo pipefail
root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$root/profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919"
test "$PWD" = "$root"
case "${1:-}" in
  --prepare) test "$#" -eq 1; python3 "$package/prepare.py" ;;
  --preflight) test "$#" -eq 1; python3 "$package/validate.py" --require-fresh ;;
  --measure)
    test "$#" -eq 1
    python3 "$package/validate.py" --require-fresh
    python3 "$package/run.py"
    ;;
  *) echo "usage: $0 --prepare|--preflight|--measure" >&2; exit 2 ;;
esac
