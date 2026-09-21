#!/usr/bin/env bash
set -euo pipefail
package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
python=/home/battlefront/.local/bin/python3.11
test "$#" -eq 1
case "$1" in
  --prepare) "$python" "$package/prepare.py" ;;
  --preflight) "$python" "$package/preflight.py" ;;
  --measure) "$python" "$package/measure.py" ;;
  *) echo "usage: $0 --prepare|--preflight|--measure" >&2; exit 2 ;;
esac
