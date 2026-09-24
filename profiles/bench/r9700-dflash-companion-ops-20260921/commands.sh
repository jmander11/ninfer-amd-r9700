#!/usr/bin/env bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib
exec /home/battlefront/.local/bin/python3.11 profiles/bench/r9700-dflash-companion-ops-20260921/campaign.py "${1:-preflight}"
