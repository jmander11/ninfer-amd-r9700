#!/usr/bin/env bash
set -euo pipefail

cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
sha256sum --check --strict profiles/bench/post-terminal-selected-vision-prepare-20260905/prepared.sha256
python3 -m tools.bench.prepare_selected_vision_diagnostic \
  --selection profiles/bench/pareto-result-post-promotion-20260905.json \
  --out profiles/bench/post-terminal-selected-vision-20260905
