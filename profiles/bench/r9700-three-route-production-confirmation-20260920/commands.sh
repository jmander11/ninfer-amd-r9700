#!/usr/bin/env bash
set -euo pipefail
root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$root/profiles/bench/r9700-three-route-production-confirmation-20260920"
python=/home/battlefront/.local/bin/python3.11
test "$PWD" = "$root"
test "$#" -eq 1
case "$1" in
  --prepare|--preflight|--measure) "$python" -B "$package/campaign.py" "$1" ;;
  *) echo "usage: $0 --prepare|--preflight|--measure" >&2; exit 2 ;;
esac
