#!/usr/bin/env bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
export PYTHONDONTWRITEBYTECODE=1
exec /home/battlefront/.local/bin/python3.11 \
  profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/campaign.py "$@"
