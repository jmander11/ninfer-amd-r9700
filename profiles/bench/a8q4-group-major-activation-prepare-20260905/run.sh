#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../.." && pwd -P)"
cd "$repo"
test "$#" -eq 0
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
sha256sum --check --strict profiles/bench/a8q4-group-major-activation-prepare-20260905/prepared.sha256
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python -m tools.r9700.run_a8q4_group_major_activation_gate --benchmark
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python -m tools.r9700.validate_a8q4_group_major_activation_report
