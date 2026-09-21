#!/usr/bin/env bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
export PYTHONDONTWRITEBYTECODE=1
exec /home/battlefront/.local/bin/python3.11 \
  profiles/bench/r9700-chunked-attention-candidate-leaf-retry1-20260921/retry.py "$@"
