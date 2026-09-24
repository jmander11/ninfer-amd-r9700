#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/bench/final-artifact-cutover-admission-prepare-20260905"
readonly python=/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python
cd "$repo"
sha256sum --check --strict "$package/prepared.sha256"
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"$python" -m tools.bench.validate_final_cutover_admission \
  --plan "$package/plan.json" \
  --out "$repo/profiles/bench/final-artifact-cutover-admission-20260905.json"
