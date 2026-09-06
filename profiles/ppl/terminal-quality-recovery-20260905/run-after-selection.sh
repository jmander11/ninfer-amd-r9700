#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/ppl/terminal-quality-recovery-20260905"
readonly selection="$repo/profiles/bench/prefill-chunk-selection-20260905.json"
readonly power=/sys/class/drm/card2/device/power_dpm_force_performance_level
readonly py=/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python
readonly run="$repo/tools/ppl/run.py"
readonly compare="$repo/tools/ppl/compare_bf16_repeats.py"
readonly bf16_scorer="$repo/tools/reference/qwen3_8_27b_bf16/ppl.py"
readonly bf16_weights=/ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16
readonly recovery_io="$repo/tools/ppl/quality_recovery_io.py"
readonly ids="$repo/tools/ppl/corpus.ids"
readonly dense_g16="$repo/build-r9700-dense-selection-g16/apps/ninfer-ppl"
readonly dense_g32="$repo/build-r9700-dense-selection-g32/apps/ninfer-ppl"
readonly sparse_g16="$repo/build-r9700-xattention-model-s16-tau900/apps/ninfer-ppl"
readonly sparse_g32="$repo/build-r9700-xattention-model-s16-tau900-g32/apps/ninfer-ppl"
readonly all_q4="$repo/out/qwen3.8-27b-r9700-q4g64-eval.ninfer"
readonly mixed="$repo/out/qwen3.8-27b-r9700-q4-w8-mse-eval.ninfer"
readonly four_role="$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-eval.ninfer"
readonly authority_map="$package/quality-authorities.json"
readonly pending_authority_map="$package/quality-authorities.pending.json"

if [[ $# -ne 1 || $1 != --execute-gpu-quality ]]; then
  echo "usage: $0 --execute-gpu-quality" >&2
  exit 2
fi
cd "$repo"
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib
sha256sum --check --strict "$package/prepared.sha256"
for input in "$py" "$run" "$compare" "$bf16_scorer" "$recovery_io" "$ids" \
  "$dense_g16" "$dense_g32" "$sparse_g16" "$sparse_g32" \
  "$all_q4" "$mixed" "$four_role"; do
  [[ -f "$input" ]]
done
test "$(cat "$power")" = auto

selection_sha="$(sha256sum "$selection" | cut -d' ' -f1)"
selected_chunk="$($py -c 'from pathlib import Path; from tools.bench.select_prefill_chunk import validate_selection_record; print(validate_selection_record(Path("profiles/bench/prefill-chunk-selection-20260905.json"))["selected_prefill_chunk"])')"
case "$selected_chunk" in
  1024|2048|4096|8192) ;;
  *) echo "unsupported selected prefill chunk: $selected_chunk" >&2; exit 1 ;;
esac

readonly out_all_q4_dense="$repo/profiles/ppl/terminal-quality-all-q4-dense-20260905"
readonly out_all_q4_sparse="$repo/profiles/ppl/terminal-quality-all-q4-xattention-20260905"
readonly out_mixed_dense="$repo/profiles/ppl/terminal-quality-mixed-dense-20260905"
readonly out_mixed_sparse="$repo/profiles/ppl/terminal-quality-mixed-xattention-20260905"
readonly out_four_role_dense="$repo/profiles/ppl/terminal-quality-four-role-dense-20260905"
readonly out_four_role_sparse="$repo/profiles/ppl/terminal-quality-four-role-xattention-20260905"

# Establish the complete publication namespace and validate all 18 BF16 shards before the first
# GPU-capable subprocess. There is
# no in-place resume: an interrupted directory is non-authoritative and must be preserved or
# explicitly moved away before a fresh attempt. Only the final exclusive hard-link publishes.
preflight_args=(
  preflight --checkpoint "$bf16_weights"
  --absent "$authority_map" --absent "$pending_authority_map"
  --absent "$out_all_q4_sparse" --absent "$out_mixed_dense"
  --absent "$out_mixed_sparse" --absent "$out_four_role_dense"
  --absent "$out_four_role_sparse"
)
if [[ "$selected_chunk" != 4096 ]]; then
  preflight_args+=(--absent "$out_all_q4_dense")
  for output in \
    "$repo/profiles/ppl/bf16-reference-selected-chunk-${selected_chunk}-a-20260905" \
    "$repo/profiles/ppl/bf16-reference-selected-chunk-${selected_chunk}-b-20260905" \
    "$repo/profiles/ppl/bf16-reference-selected-chunk-${selected_chunk}-repeat-20260905.json"; do
    preflight_args+=(--absent "$output")
  done
fi
"$py" "$recovery_io" "${preflight_args[@]}"

if [[ "$selected_chunk" == 4096 ]]; then
  bf16_authority="$repo/profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json"
  bf16_repeat="$repo/profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json"
  all_q4_dense="$repo/profiles/ppl/xattention-dense-q4g64-v3-rebase-20260904/results.json"
  "$py" "$package/validate_retained.py" >/dev/null
else
  bf16_a="$repo/profiles/ppl/bf16-reference-selected-chunk-${selected_chunk}-a-20260905"
  bf16_b="$repo/profiles/ppl/bf16-reference-selected-chunk-${selected_chunk}-b-20260905"
  bf16_repeat="$repo/profiles/ppl/bf16-reference-selected-chunk-${selected_chunk}-repeat-20260905.json"
  all_q4_dense="$repo/profiles/ppl/terminal-quality-all-q4-dense-20260905/results.json"
  for output in "$bf16_a" "$bf16_b"; do
    "$py" "$run" \
      --bf16-reference-ppl-bin "$bf16_scorer" --bf16-reference-weights "$bf16_weights" \
      --ids "$ids" --profiles bf16-reference --schedule prefill --spec none --no-extras \
      --prefill-chunk "$selected_chunk" --device 0 \
      --quality-tier accuracy --allow-ungated --out "$output"
    test "$(cat "$power")" = auto
  done
  "$py" "$compare" --first "$bf16_a/results.json" --second "$bf16_b/results.json" --out "$bf16_repeat"
  bf16_authority="$bf16_a/results.json"
fi

run_quality() {
  local route=$1 weights=$2 tier=$3 gate=$4 g16=$5 g32=$6 output=$7 hybrid=$8
  shift 8
  local hybrid_args=()
  if [[ "$hybrid" == yes ]]; then hybrid_args+=(--require-fp8-hybrid); fi
  "$py" "$run" \
    --bf16-reference-ppl-bin "$bf16_scorer" --bf16-reference-weights "$bf16_weights" \
    --g16-ppl-bin "$g16" --g32-ppl-bin "$g32" \
    --g16-weights "$weights" --g32-weights "$weights" --ids "$ids" \
    --profiles bf16-reference,r9700-g16,r9700-g32 \
    --schedule prefill --spec none --no-extras --prefill-chunk "$selected_chunk" --device 0 \
    --quality-tier "$tier" --gate "r9700-g16=$gate" --gate "r9700-g32=$gate" \
    --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
    --expected-fp8-qk-wmma 1 --expected-xattention-profile "$route" \
    --reuse-bf16-campaign "$bf16_authority" --bf16-repeat-comparison "$bf16_repeat" \
    "${hybrid_args[@]}" "$@" --out "$output"
  test "$(cat "$power")" = auto
}

if [[ "$selected_chunk" != 4096 ]]; then
  run_quality dense "$all_q4" capacity-speed 0.048790164169432 \
    "$dense_g16" "$dense_g32" "$out_all_q4_dense" no
fi
run_quality b128-s16-tau900 "$all_q4" capacity-speed 0.048790164169432 \
  "$sparse_g16" "$sparse_g32" "$out_all_q4_sparse" no
run_quality dense "$mixed" accuracy 0.02 \
  "$dense_g16" "$dense_g32" "$out_mixed_dense" no
run_quality b128-s16-tau900 "$mixed" accuracy 0.02 \
  "$sparse_g16" "$sparse_g32" "$out_mixed_sparse" no
four_role_dense_reuse=()
if [[ "$selected_chunk" == 4096 ]]; then
  four_role_dense_reuse+=(
    --reuse-candidate-campaign "$repo/profiles/ppl/fp8-hybrid-product-8k-20260904/results.json"
    --reuse-candidate-campaign "$repo/profiles/ppl/fp8-hybrid-product-32k-20260904/results.json"
  )
fi
run_quality dense "$four_role" capacity-speed 0.048790164169432 \
  "$dense_g16" "$dense_g32" "$out_four_role_dense" yes "${four_role_dense_reuse[@]}"
run_quality b128-s16-tau900 "$four_role" capacity-speed 0.048790164169432 \
  "$sparse_g16" "$sparse_g32" "$out_four_role_sparse" yes

export NINFER_SELECTED_CHUNK="$selected_chunk"
export NINFER_CHUNK_SELECTION="$selection"
export NINFER_CHUNK_SELECTION_SHA="$selection_sha"
export NINFER_ALL_Q4_DENSE="$all_q4_dense"
export NINFER_ALL_Q4_SPARSE="$out_all_q4_sparse/results.json"
export NINFER_MIXED_DENSE="$out_mixed_dense/results.json"
export NINFER_MIXED_SPARSE="$out_mixed_sparse/results.json"
export NINFER_FOUR_ROLE_DENSE="$out_four_role_dense/results.json"
export NINFER_FOUR_ROLE_SPARSE="$out_four_role_sparse/results.json"
"$py" - <<'PY' >"$pending_authority_map"
import json, os
from pathlib import Path
from tools.ppl.assemble_pareto import _campaign_quality_candidate
from tools.ppl.run import file_sha256

chunk = int(os.environ["NINFER_SELECTED_CHUNK"])
entries = [
    ("ALL_Q4_DENSE_QUALITY", "r9700-q4g64-eval", os.environ["NINFER_ALL_Q4_DENSE"]),
    ("ALL_Q4_XATTENTION_QUALITY", "r9700-q4g64-eval", os.environ["NINFER_ALL_Q4_SPARSE"]),
    ("MIXED_DENSE_QUALITY", "r9700-q4-w8-mse-eval", os.environ["NINFER_MIXED_DENSE"]),
    ("MIXED_XATTENTION_QUALITY", "r9700-q4-w8-mse-eval", os.environ["NINFER_MIXED_SPARSE"]),
    ("FOUR_ROLE_DENSE_QUALITY", "r9700-q4g64-f8e4m3-four-role-eval", os.environ["NINFER_FOUR_ROLE_DENSE"]),
    ("FOUR_ROLE_XATTENTION_QUALITY", "r9700-q4g64-f8e4m3-four-role-eval", os.environ["NINFER_FOUR_ROLE_SPARSE"]),
]
authorities = {}
for name, weights_id, text in entries:
    path = Path(text).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    for group in (16, 32):
        _campaign_quality_candidate(value, weights_id, group, chunk)
    authorities[name] = {"path": str(path), "sha256": file_sha256(path)}
print(json.dumps({
    "artifact_type": "ninfer_r9700_terminal_quality_authority_map",
    "schema_version": 1,
    "selected_prefill_chunk": chunk,
    "selected_prefill_chunk_authority": {
        "path": str(Path(os.environ["NINFER_CHUNK_SELECTION"]).resolve(strict=True)),
        "sha256": os.environ["NINFER_CHUNK_SELECTION_SHA"],
    },
    "concurrency": 1,
    "authorities": authorities,
}, indent=2, sort_keys=True))
PY
test "$(sha256sum "$selection" | cut -d' ' -f1)" = "$selection_sha"
"$py" "$recovery_io" publish --pending "$pending_authority_map" --final "$authority_map"
test "$(cat "$power")" = auto
