#!/usr/bin/env bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
if [[ "$#" -gt 1 ]] || [[ "$#" -eq 1 && "$1" != "--preflight-only" && "$1" != "--analyze-only" ]]; then
  echo "usage: $0 [--preflight-only|--analyze-only]" >&2
  exit 2
fi
sha256sum --check --strict profiles/bench/r9700-dflash-p129-isolation-discriminator-20260906/prepared.sha256
python3 profiles/bench/r9700-dflash-p129-isolation-discriminator-20260906/run.py "$@"
