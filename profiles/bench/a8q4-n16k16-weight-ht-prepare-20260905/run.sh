#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
if (( $# != 0 )); then
  echo "run.sh accepts no arguments" >&2
  exit 2
fi
PY=/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python
REPORT=profiles/bench/r9700-a8q4-n16k16-weight-device-ht-p2048-ab-20260905.json
/usr/bin/sha256sum --check --strict profiles/bench/a8q4-n16k16-weight-ht-prepare-20260905/prepared.sha256
test ! -e "$REPORT" && test ! -L "$REPORT"
"$PY" -m tools.r9700.run_a8q4_n16k16_weight_ht_gate --benchmark --output "$REPORT"
"$PY" -m tools.r9700.validate_a8q4_n16k16_weight_ht_report "$REPORT"
