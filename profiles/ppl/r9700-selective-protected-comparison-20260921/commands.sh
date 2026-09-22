#!/usr/bin/env bash
set -euo pipefail
cd /ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly experiment=profiles/ppl/r9700-selective-protected-comparison-20260921
readonly prior=profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921
readonly scorer=build-r9700-selective-protected-g16-20260921/apps/ninfer-ppl
readonly artifact=out/qwen3.8-27b-r9700-q4-selective-protected-n16k16-eval.ninfer
readonly python311=/home/battlefront/.local/bin/python3.11
unset PYTHONPATH PYTHONHOME
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib
case "${1:-}" in
  ppl)
    test -x "$scorer"
    test -f "$artifact"
    test ! -e "$experiment/candidate" && test ! -L "$experiment/candidate"
    test "$(tr -d '\n' </sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level)" = auto
    status=0
    "$python311" tools/ppl/run.py \
      --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
      --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
      --g16-ppl-bin "$scorer" --g16-weights "$artifact" \
      --ids tools/ppl/corpus.ids --profiles bf16-reference,r9700-g16 \
      --schedule prefill --skip half --spec none --no-extras \
      --quality-tier accuracy --gate r9700-g16=0.02 \
      --prefill-chunk 2048 --device 0 \
      --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
      --expected-fp8-qk-wmma 1 --expected-xattention-profile dense \
      --reuse-bf16-campaign "$prior/bf16-chunk2048-a/results.json" \
      --bf16-repeat-comparison "$prior/bf16-chunk2048-repeat.json" \
      --out "$experiment/candidate" || status=$?
    test "$status" -le 1
    test "$(tr -d '\n' </sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level)" = auto
    test -f "$experiment/candidate/results.json"
    # A complete measured quality failure is a result, not permission to loosen its gate.
    "$python311" "$experiment/analyze.py" --candidate "$experiment/candidate/results.json" \
      --out "$experiment/comparison.json"
    ;;
  analyze)
    "$python311" "$experiment/analyze.py" --candidate "$experiment/candidate/results.json" \
      --out "$experiment/comparison.json"
    ;;
  *) echo 'usage: commands.sh ppl|analyze (only after conversion and numerical qualification)' >&2; exit 2 ;;
esac
