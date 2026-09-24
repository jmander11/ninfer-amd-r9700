#!/usr/bin/env bash
set -euo pipefail

repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$repo/profiles/bench/r9700-dflash-small-t-n16-whole-ab-c1-20260906"
cd "$repo"
test "$#" -eq 0
sha256sum --check --strict "$package/prepared.sha256"
sha256sum --check --strict \
  "$repo/profiles/bench/r9700-dflash-four-role-n16k16-canonical-q4-preflight-20260906/prepared.sha256"
python3 "$package/run.py"
