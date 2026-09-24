#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/bench/post-chunk-twelve-candidate-20260905"
readonly screen_campaign="$repo/profiles/bench/prefill-chunk-screen-twelve-candidate-receipt-bound-n16k16-20260905"
readonly finalist_pipeline="$repo/profiles/bench/prefill-chunk-selection-pipeline-receipt-bound-n16k16-20260905"
readonly runner="$repo/tools/bench/run_ninfer_bench_matrix.py"
readonly corpus="$repo/bench/fixtures/bench_corpus.ids"
readonly selection="$repo/profiles/bench/prefill-chunk-selection-receipt-bound-n16k16-20260905.json"
readonly pending_selection="$package/selection.pending.json"
readonly capacity_validation="$package/executed-capacity-validation.json"
readonly power=/sys/class/drm/card2/device/power_dpm_force_performance_level
readonly concurrency=(--concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4)
capacity_validation_args=(--authority "$selection")

path_absent() { [[ ! -e "$1" && ! -L "$1" ]]; }

if [[ $# -ne 1 || $1 != --execute-gpu-campaign ]]; then
  echo "usage: $0 --execute-gpu-campaign" >&2
  exit 2
fi
cd "$repo"
python3 -m tools.bench.validate_n16_artifact_identity --artifact \
  "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer" \
  --weights-id r9700-q4g64-n16k16-eval
python3 -m tools.bench.validate_n16_artifact_identity --artifact \
  "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer" \
  --weights-id r9700-q4-w8-mse-n16k16-eval
python3 -m tools.bench.validate_n16_artifact_identity --artifact \
  "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" \
  --weights-id r9700-q4g64-f8e4m3-four-role-n16k16-eval
sha256sum --check --strict "$package/prepared.sha256"
sha256sum --check --strict "$screen_campaign/prepared.sha256"
sha256sum --check --strict "$finalist_pipeline/prepared.sha256"
path_absent "$pending_selection"
[[ -f "$selection" && ! -L "$selection" ]]
readonly selection_sha="$(sha256sum "$selection" | cut -d' ' -f1)"
if ! selected_chunk="$(python3 -c 'from pathlib import Path; from tools.bench.select_prefill_chunk import validate_selection_record; print(validate_selection_record(Path("profiles/bench/prefill-chunk-selection-receipt-bound-n16k16-20260905.json"))["selected_prefill_chunk"])')"; then
  echo "selected-chunk authority failed standalone revalidation" >&2
  exit 1
fi
[[ $selected_chunk == 1024 || $selected_chunk == 2048 || $selected_chunk == 4096 || $selected_chunk == 8192 ]]
test "$(cat "$power")" = auto

run_capacity() {
  local group=$1 attention=$2 tag=$3 recipe=$4 bench=$5 weights=$6
  shift 6
  local capacity="$repo/profiles/bench/pareto-capacity-post-promotion-${tag}-${recipe}-g${group}-receipt-bound-n16k16-20260905"
  local common=(
    --bench "$bench" --no-build --weights "$weights" --corpus "$corpus"
    --prefill-chunk "$selected_chunk" --prefill-chunk-authority "$selection"
    "${concurrency[@]}"
    --expected-kv-value-group "$group" --expected-q4-activation-bits 8
    --expected-w8-activation-bits 8 --expected-fp8-qk-wmma 1
    --expected-xattention-profile "$attention" --resume
  )
  capacity_validation_args+=(--matrix "$capacity")
  test "$(cat "$power")" = auto
  if ! python3 "$runner" --preset pareto-capacity --require-post-chunk-capacity \
    "${common[@]}" --output-dir "$capacity" "$@"; then
    # Retain the failure and acquire every remaining capacity candidate. The campaign validator
    # below admits only exact structured memory-admission failures before any whole timing starts.
    test "$(cat "$power")" = auto
    return 0
  fi
  test "$(cat "$power")" = auto
}

run_capacity 16 dense dense all-q4 "$repo/build-r9700-dense-selection-g16/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
run_capacity 32 dense dense all-q4 "$repo/build-r9700-dense-selection-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
run_capacity 16 b128-s16-tau900 xattention-s16-tau900 all-q4 "$repo/build-r9700-xattention-model-s16-tau900/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
run_capacity 32 b128-s16-tau900 xattention-s16-tau900 all-q4 "$repo/build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
run_capacity 16 dense dense mixed "$repo/build-r9700-dense-selection-g16/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer"
run_capacity 32 dense dense mixed "$repo/build-r9700-dense-selection-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer"
run_capacity 16 b128-s16-tau900 xattention-s16-tau900 mixed "$repo/build-r9700-xattention-model-s16-tau900/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer"
run_capacity 32 b128-s16-tau900 xattention-s16-tau900 mixed "$repo/build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer"
run_capacity 16 dense dense four-role "$repo/build-r9700-dense-selection-g16/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" --require-fp8-hybrid --hybrid-width-tool "$repo/build-r9700-dense-selection-g16/src/ninfer_r9700_runtime_planner_qual"
run_capacity 32 dense dense four-role "$repo/build-r9700-dense-selection-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" --require-fp8-hybrid --hybrid-width-tool "$repo/build-r9700-dense-selection-g32/src/ninfer_r9700_runtime_planner_qual"
run_capacity 16 b128-s16-tau900 xattention-s16-tau900 four-role "$repo/build-r9700-xattention-model-s16-tau900/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" --require-fp8-hybrid --hybrid-width-tool "$repo/build-r9700-xattention-model-s16-tau900/src/ninfer_r9700_runtime_planner_qual"
run_capacity 32 b128-s16-tau900 xattention-s16-tau900 four-role "$repo/build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" --require-fp8-hybrid --hybrid-width-tool "$repo/build-r9700-xattention-model-s16-tau900-g32/src/ninfer_r9700_runtime_planner_qual"

python3 -m tools.bench.validate_post_chunk_capacity_campaign \
  "${capacity_validation_args[@]}" --executed --out "$capacity_validation"

whole_is_eligible() {
  python3 - "$capacity_validation" "$1" "$2" "$3" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
identity = [sys.argv[2], int(sys.argv[3]), sys.argv[4]]
raise SystemExit(0 if identity in result["whole_eligible_identities"] else 1)
PY
}

run_whole() {
  local group=$1 attention=$2 tag=$3 recipe=$4 weights_id=$5 bench=$6 weights=$7
  shift 7
  if ! whole_is_eligible "$weights_id" "$group" "$attention"; then
    return 0
  fi
  local whole="$repo/profiles/bench/pareto-whole-post-promotion-${tag}-${recipe}-g${group}-receipt-bound-n16k16-20260905"
  python3 "$runner" --preset pareto-whole --bench "$bench" --no-build \
    --weights "$weights" --corpus "$corpus" --prefill-chunk "$selected_chunk" \
    --prefill-chunk-authority "$selection" "${concurrency[@]}" \
    --expected-kv-value-group "$group" --expected-q4-activation-bits 8 \
    --expected-w8-activation-bits 8 --expected-fp8-qk-wmma 1 \
    --expected-xattention-profile "$attention" --resume --output-dir "$whole" "$@"
  test "$(cat "$power")" = auto
}

run_whole 16 dense dense all-q4 r9700-q4g64-n16k16-eval "$repo/build-r9700-dense-selection-g16/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
run_whole 32 dense dense all-q4 r9700-q4g64-n16k16-eval "$repo/build-r9700-dense-selection-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
run_whole 16 b128-s16-tau900 xattention-s16-tau900 all-q4 r9700-q4g64-n16k16-eval "$repo/build-r9700-xattention-model-s16-tau900/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
run_whole 32 b128-s16-tau900 xattention-s16-tau900 all-q4 r9700-q4g64-n16k16-eval "$repo/build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
run_whole 16 dense dense mixed r9700-q4-w8-mse-n16k16-eval "$repo/build-r9700-dense-selection-g16/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer"
run_whole 32 dense dense mixed r9700-q4-w8-mse-n16k16-eval "$repo/build-r9700-dense-selection-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer"
run_whole 16 b128-s16-tau900 xattention-s16-tau900 mixed r9700-q4-w8-mse-n16k16-eval "$repo/build-r9700-xattention-model-s16-tau900/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer"
run_whole 32 b128-s16-tau900 xattention-s16-tau900 mixed r9700-q4-w8-mse-n16k16-eval "$repo/build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer"
run_whole 16 dense dense four-role r9700-q4g64-f8e4m3-four-role-n16k16-eval "$repo/build-r9700-dense-selection-g16/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" --require-fp8-hybrid --hybrid-width-tool "$repo/build-r9700-dense-selection-g16/src/ninfer_r9700_runtime_planner_qual"
run_whole 32 dense dense four-role r9700-q4g64-f8e4m3-four-role-n16k16-eval "$repo/build-r9700-dense-selection-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" --require-fp8-hybrid --hybrid-width-tool "$repo/build-r9700-dense-selection-g32/src/ninfer_r9700_runtime_planner_qual"
run_whole 16 b128-s16-tau900 xattention-s16-tau900 four-role r9700-q4g64-f8e4m3-four-role-n16k16-eval "$repo/build-r9700-xattention-model-s16-tau900/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" --require-fp8-hybrid --hybrid-width-tool "$repo/build-r9700-xattention-model-s16-tau900/src/ninfer_r9700_runtime_planner_qual"
run_whole 32 b128-s16-tau900 xattention-s16-tau900 four-role r9700-q4g64-f8e4m3-four-role-n16k16-eval "$repo/build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" --require-fp8-hybrid --hybrid-width-tool "$repo/build-r9700-xattention-model-s16-tau900-g32/src/ninfer_r9700_runtime_planner_qual"

test "$(sha256sum "$selection" | cut -d' ' -f1)" = "$selection_sha"
sha256sum --check --strict "$package/prepared.sha256"
sha256sum --check --strict "$screen_campaign/prepared.sha256"
sha256sum --check --strict "$finalist_pipeline/prepared.sha256"
test "$(cat "$power")" = auto
