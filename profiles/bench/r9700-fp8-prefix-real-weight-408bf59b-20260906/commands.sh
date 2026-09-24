#!/usr/bin/env bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
if [[ "$#" -gt 1 ]] || [[ "$#" -eq 1 && "$1" != "--preflight-only" ]]; then
  echo "usage: $0 [--preflight-only]" >&2
  exit 2
fi
sha256sum -c --strict profiles/bench/r9700-fp8-prefix-real-weight-408bf59b-20260906/prepared.sha256
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 profiles/bench/r9700-fp8-prefix-real-weight-408bf59b-20260906/run.py "$@"
