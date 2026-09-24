#!/usr/bin/env bash
set -euo pipefail
root=/ssdpool2nvme/local_llm/ninfer-amd-r9700
test "$PWD" = "$root"
exec /home/battlefront/.local/bin/python3.11 -B "$root/profiles/bench/r9700-normalized-linear-t1-production-confirmation-20260920/campaign.py" "$@"
