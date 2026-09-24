#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/bench/post-terminal-niah-prepare-20260905"
readonly selection="$repo/profiles/bench/pareto-result-post-promotion-20260905.json"
readonly output="$repo/profiles/bench/post-terminal-niah-20260905"

cd "$repo"
sha256sum --check --strict "$package/prepared.sha256"
if [[ ! -f "$selection" ]]; then
  echo "validated terminal schema-v7 selection is not available: $selection" >&2
  exit 1
fi
if [[ -e "$output" || -L "$output" ]]; then
  echo "refusing to overwrite prepared NIAH campaign: $output" >&2
  exit 1
fi
python3 -m tools.bench.prepare_selected_niah --selection "$selection" --out "$output"
echo "prepared only; future GPU command: bash $output/commands.sh"
