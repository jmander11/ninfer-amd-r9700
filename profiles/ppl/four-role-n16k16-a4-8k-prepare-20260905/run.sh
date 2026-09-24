#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/ppl/four-role-n16k16-a4-8k-prepare-20260905"
readonly output="$repo/profiles/ppl/four-role-n16k16-a4-8k-20260905"
readonly python=/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python
readonly power=/sys/class/drm/card2/device/power_dpm_force_performance_level

if [[ $# -ne 1 || $1 != --execute-gpu-quality ]]; then
  echo "usage: $0 --execute-gpu-quality" >&2
  exit 2
fi
cd "$repo"
sha256sum --check --strict "$package/prepared.sha256"
[[ ! -e "$output" && ! -L "$output" ]]
test "$(cat "$power")" = auto
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/core-10.0/lib

status=0
"$python" "$repo/tools/ppl/run.py" \
  --bf16-reference-ppl-bin "$repo/tools/reference/qwen3_8_27b_bf16/ppl.py" \
  --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --g16-ppl-bin "$repo/build-r9700-a4q4-n16k16/apps/ninfer-ppl" \
  --g16-weights "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" \
  --ids "$repo/tools/ppl/corpus.ids" --tokens 8192 --profiles bf16-reference,r9700-g16 \
  --schedule prefill --skip half --prefill-chunk 4096 --spec none --no-extras \
  --quality-tier capacity-speed --gate r9700-g16=0.04879016416943205 \
  --expected-q4-activation-bits 4 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 --expected-xattention-profile dense \
  --reuse-bf16-campaign "$repo/profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json" \
  --bf16-repeat-comparison "$repo/profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json" \
  --require-fp8-hybrid --device 0 --out "$output" || status=$?
if [[ $status -gt 1 ]]; then
  exit "$status"
fi
test "$(cat "$power")" = auto
"$python" "$package/validate_result.py" --output "$output"
