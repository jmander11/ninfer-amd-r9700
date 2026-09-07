#!/usr/bin/env bash
set -euo pipefail
package=/ssdpool2nvme/local_llm/ninfer-amd-r9700/profiles/bench/r9700-qwen3-layer3-attention-trace-617672eb-20260906
[[ $# -le 1 && ( $# -eq 0 || $1 == --preflight-only ) ]] || { echo 'usage: commands.sh [--preflight-only]' >&2; exit 2; }
cd "$package"
sha256sum --check --strict prepared.sha256
exec /usr/bin/python3 run.py "$@"
