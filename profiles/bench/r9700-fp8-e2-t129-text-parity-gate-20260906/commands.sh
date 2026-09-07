#!/usr/bin/env bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700/profiles/bench/r9700-fp8-e2-t129-text-parity-gate-20260906
if [[ $# -gt 1 ]] || [[ $# -eq 1 && $1 != --preflight-only ]]; then
  echo "usage: $0 [--preflight-only]" >&2
  exit 2
fi
sha256sum -c --strict prepared.sha256
exec /usr/bin/python3 -B run.py "$@"
