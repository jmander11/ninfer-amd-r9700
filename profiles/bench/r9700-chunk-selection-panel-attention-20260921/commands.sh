#!/usr/bin/env bash
set -euo pipefail
readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly py=/home/battlefront/.local/bin/python3.11
readonly campaign="$repo/profiles/bench/r9700-chunk-selection-panel-attention-20260921/campaign.py"
cd "$repo"
# preflight validates files, builds, artifacts and power without publishing files.
# freeze explicitly creates the input receipt; prepare is CPU-only.
# screens and finalists execute GPU work; select publishes the validated shared chunk authority.
export PYTHONDONTWRITEBYTECODE=1
exec "$py" "$campaign" "$@"
