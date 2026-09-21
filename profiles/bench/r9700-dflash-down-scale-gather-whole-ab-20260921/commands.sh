#!/usr/bin/env bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
case "${1:---preflight}" in
  --prepare) action=prepare ;;
  --preflight) action=preflight ;;
  --measure) action=measure ;;
  *) echo 'usage: commands.sh [--prepare|--preflight|--measure]' >&2; exit 2 ;;
esac
exec /home/battlefront/.local/bin/python3.11 profiles/bench/r9700-dflash-down-scale-gather-whole-ab-20260921/run.py "$action"
