#!/usr/bin/env bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
export PYTHONDONTWRITEBYTECODE=1
exec /home/battlefront/.local/bin/python3.11 profiles/bench/r9700-terminal-base-phase-sum-20260921/campaign.py "$@"
