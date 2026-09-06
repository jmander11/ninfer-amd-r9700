#!/usr/bin/env bash
set -euo pipefail

repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
package="$repo/profiles/bench/r9700-dflash-four-role-n16k16-canonical-q4-preflight-20260906"
cd "$repo"
sha256sum --check --strict "$package/prepared.sha256"
python3 "$package/validate.py"
env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib \
  /ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  -m tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 \
  --base "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" \
  --dflash-model /ssdpool2nvme/local_llm/models/qwen3.8-27b-dflash2 \
  --out "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer" \
  --device cuda
