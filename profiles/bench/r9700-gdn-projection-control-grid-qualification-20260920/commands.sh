#!/usr/bin/env bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
exec /home/battlefront/.local/bin/python3.11 profiles/bench/r9700-gdn-projection-control-grid-qualification-20260920/run.py "$@"
