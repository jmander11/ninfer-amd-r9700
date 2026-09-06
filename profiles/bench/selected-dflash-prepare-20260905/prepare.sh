#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/bench/selected-dflash-prepare-20260905"
readonly selection="$repo/profiles/bench/pareto-result-post-promotion-20260905.json"
readonly output="$repo/profiles/bench/selected-dflash-20260905"

cd "$repo"
sha256sum --check --strict "$package/prepared.sha256"
[[ -f "$selection" ]] || { echo "terminal schema-v7 selection is unavailable" >&2; exit 1; }
[[ ! -e "$output" && ! -L "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 1; }
python3 -m tools.bench.prepare_selected_dflash prepare --selection "$selection" --out "$output"
echo "prepared only; future staged GPU/conversion command: bash $output/commands.sh"
