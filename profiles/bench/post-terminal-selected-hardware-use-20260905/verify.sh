#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/bench/post-terminal-selected-hardware-use-20260905"
readonly selection="$repo/profiles/bench/pareto-result-post-promotion-20260905.json"
readonly audit="$repo/profiles/bench/r9700-twelve-candidate-hardware-path-static-audit-20260905.json"
readonly output="$repo/profiles/bench/selected-hardware-use-20260905.json"

if [[ $# -ne 1 && $# -ne 3 ]]; then
  echo "usage: $0 RECONCILIATION [FP8_GATE_PROOF FP8_ATTENTION_PROOF]" >&2
  exit 2
fi

cd "$repo"
sha256sum --check --strict "$package/prepared.sha256"
test -f "$selection"
test ! -e "$output" && test ! -L "$output"

args=(
  --selection "$selection"
  --reconciliation "$(realpath -e "$1")"
  --static-audit "$audit"
  --out "$output"
)
if [[ $# -eq 3 ]]; then
  args+=(--fp8-proof "$(realpath -e "$2")" --fp8-proof "$(realpath -e "$3")")
fi
python3 -m tools.bench.verify_selected_hardware_use "${args[@]}"
