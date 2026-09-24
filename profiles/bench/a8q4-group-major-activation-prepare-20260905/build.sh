#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../.." && pwd -P)"
cd "$repo"
test "$#" -eq 0
bash tools/r9700/build_a8q4_group_major_activation.sh
sha256sum --check --strict profiles/bench/a8q4-group-major-activation-prepare-20260905/prepared.sha256
