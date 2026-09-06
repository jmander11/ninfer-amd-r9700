#!/usr/bin/env bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
if [[ "$#" -gt 1 ]] || [[ "$#" -eq 1 && "$1" != "--preflight-only" ]]; then echo "usage: $0 [--preflight-only]" >&2; exit 2; fi
sha256sum -c --strict profiles/bench/r9700-dflash-rmsnorm-rows56-token-parity-20260906/prepared.sha256
/usr/bin/python3 profiles/bench/r9700-dflash-rmsnorm-rows56-token-parity-20260906/run.py "$@"
