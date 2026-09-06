#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/ppl/post-terminal-exact-token-prepare-20260905"
readonly selection="$repo/profiles/bench/pareto-result-post-promotion-20260905.json"
readonly output="$repo/profiles/ppl/post-terminal-selected-exact-token-20260905"

cd "$repo"
sha256sum --check --strict "$package/prepared.sha256"
if [[ ! -f "$selection" ]]; then
  echo "validated terminal schema-v7 selection is not available: $selection" >&2
  exit 1
fi
if [[ -e "$output" || -L "$output" ]]; then
  echo "refusing to overwrite selected exact-token campaign: $output" >&2
  exit 1
fi
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  -m tools.ppl.prepare_selected_exact_token --selection "$selection" --out "$output"
echo "prepared only; future C1 GPU command: bash $output/commands.sh"
