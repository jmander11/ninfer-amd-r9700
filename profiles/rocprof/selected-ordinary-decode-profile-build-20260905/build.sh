#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/rocprof/selected-ordinary-decode-profile-build-20260905"
readonly selection="$repo/profiles/bench/pareto-result-post-promotion-20260905.json"

cd "$repo"
sha256sum --check --strict "$package/prepared.sha256"
[[ -f "$selection" ]] || { echo "terminal schema-v7 selection is unavailable" >&2; exit 1; }
if pgrep -f 'prefill-chunk-selection-pipeline-20260905/run-finalists.sh --execute-gpu-campaign' >/dev/null; then
  echo "live 32K finalist campaign forbids instrumentation rebuild" >&2
  exit 1
fi
python3 -m tools.bench.build_selected_decode_profile \
  --selection "$selection" --out "$package/receipt.json"
echo "published instrumentation-only profile build receipt; timing is inadmissible"
