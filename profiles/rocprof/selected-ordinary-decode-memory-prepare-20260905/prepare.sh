#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/rocprof/selected-ordinary-decode-memory-prepare-20260905"
readonly selection="$repo/profiles/bench/pareto-result-post-promotion-20260905.json"
readonly receipt="$repo/profiles/rocprof/selected-ordinary-decode-profile-build-20260905/receipt.json"
readonly output="$repo/profiles/rocprof/selected-ordinary-decode-memory-20260905"

cd "$repo"
sha256sum --check --strict "$package/prepared.sha256"
[[ -f "$selection" && -f "$receipt" && ! -e "$output" && ! -L "$output" ]]
python3 -m tools.bench.prepare_selected_decode_memory_profile \
  --selection "$selection" --profile-build-receipt "$receipt" --out "$output"
echo "prepared only; future GPU command: bash $output/commands.sh"
