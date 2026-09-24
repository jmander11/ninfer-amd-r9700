#!/usr/bin/env bash
set -euo pipefail

readonly repo=/ssdpool2nvme/local_llm/ninfer-amd-r9700
readonly package="$repo/profiles/bench/post-chunk-twelve-candidate-20260905"
readonly screen_campaign="$repo/profiles/bench/prefill-chunk-screen-twelve-candidate-receipt-bound-n16k16-20260905"
readonly finalist_pipeline="$repo/profiles/bench/prefill-chunk-selection-pipeline-receipt-bound-n16k16-20260905"
readonly selector="$repo/tools/bench/select_prefill_chunk.py"
readonly publisher="$repo/tools/bench/prefill_chunk_selection_io.py"
readonly runner="$repo/tools/bench/run_ninfer_bench_matrix.py"
readonly corpus="$repo/bench/fixtures/bench_corpus.ids"
readonly selection="$repo/profiles/bench/prefill-chunk-selection-receipt-bound-n16k16-20260905.json"
readonly pending_selection="$package/selection.pending.json"
readonly concurrency=(--concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4)
capacity_validation_args=(--authority "$selection")

path_absent() { [[ ! -e "$1" && ! -L "$1" ]]; }

readonly screens=(
  "$repo/profiles/bench/prefill-chunk-screen-all-q4-g16-dense-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-screen-all-q4-g32-dense-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-screen-all-q4-g16-xattention-s16-tau900-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-screen-all-q4-g32-xattention-s16-tau900-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-screen-mixed-g16-dense-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-screen-mixed-g32-dense-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-screen-mixed-g16-xattention-s16-tau900-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-screen-mixed-g32-xattention-s16-tau900-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-screen-four-role-g16-dense-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-screen-four-role-g32-dense-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-screen-four-role-g16-xattention-s16-tau900-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-screen-four-role-g32-xattention-s16-tau900-receipt-bound-n16k16-20260905"
)
readonly finalists=(
  "$repo/profiles/bench/prefill-chunk-finalist-all-q4-g16-dense-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-finalist-all-q4-g32-dense-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-finalist-all-q4-g16-xattention-s16-tau900-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-finalist-all-q4-g32-xattention-s16-tau900-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-finalist-mixed-g16-dense-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-finalist-mixed-g32-dense-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-finalist-mixed-g16-xattention-s16-tau900-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-finalist-mixed-g32-xattention-s16-tau900-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-finalist-four-role-g16-dense-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-finalist-four-role-g32-dense-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-finalist-four-role-g16-xattention-s16-tau900-receipt-bound-n16k16-20260905"
  "$repo/profiles/bench/prefill-chunk-finalist-four-role-g32-xattention-s16-tau900-receipt-bound-n16k16-20260905"
)

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
path_absent "$selection"
path_absent "$pending_selection"
for recipe in all-q4 mixed four-role; do
  for group in 16 32; do
    for tag in dense xattention-s16-tau900; do
      path_absent "$repo/profiles/bench/pareto-capacity-post-promotion-${tag}-${recipe}-g${group}-receipt-bound-n16k16-20260905"
      path_absent "$repo/profiles/bench/pareto-whole-post-promotion-${tag}-${recipe}-g${group}-receipt-bound-n16k16-20260905"
    done
  done
done

selection_args=()
for index in "${!screens[@]}"; do
  selection_args+=(--screen "${screens[index]}" --finalist "${finalists[index]}")
done
python3 "$selector" "${selection_args[@]}" --out "$pending_selection" --create-only
readonly selection_sha="$(sha256sum "$pending_selection" | cut -d' ' -f1)"
if ! selected_chunk="$(python3 -c 'from pathlib import Path; from tools.bench.select_prefill_chunk import validate_selection_record; print(validate_selection_record(Path("profiles/bench/post-chunk-twelve-candidate-20260905/selection.pending.json"))["selected_prefill_chunk"])')"; then
  echo "fresh selected-chunk authority failed standalone revalidation" >&2
  exit 1
fi
[[ $selected_chunk == 1024 || $selected_chunk == 2048 || $selected_chunk == 4096 || $selected_chunk == 8192 ]]
# Publish the independently valid selection before creating any runnable matrix. Every schema-v14
# manifest below then binds this exact canonical path and SHA, rather than a package-private file
# that disappears after preparation.
python3 -m tools.bench.prefill_chunk_selection_io "$pending_selection" "$selection"
python3 -c 'from pathlib import Path; from tools.bench.select_prefill_chunk import validate_selection_record; validate_selection_record(Path("profiles/bench/prefill-chunk-selection-receipt-bound-n16k16-20260905.json"))'
test "$(sha256sum "$selection" | cut -d' ' -f1)" = "$selection_sha"

prepare_pair() {
  local group=$1 attention=$2 tag=$3 recipe=$4 bench=$5 weights=$6
  shift 6
  local capacity="$repo/profiles/bench/pareto-capacity-post-promotion-${tag}-${recipe}-g${group}-receipt-bound-n16k16-20260905"
  local whole="$repo/profiles/bench/pareto-whole-post-promotion-${tag}-${recipe}-g${group}-receipt-bound-n16k16-20260905"
  [[ ! -e "$capacity" && ! -e "$whole" ]]
  local common=(
    --bench "$bench" --no-build --weights "$weights" --corpus "$corpus"
    --prefill-chunk "$selected_chunk" --prefill-chunk-authority "$selection"
    "${concurrency[@]}"
    --expected-kv-value-group "$group" --expected-q4-activation-bits 8
    --expected-w8-activation-bits 8 --expected-fp8-qk-wmma 1
    --expected-xattention-profile "$attention" --prepare-only
  )
  python3 "$runner" --preset pareto-capacity --require-post-chunk-capacity \
    "${common[@]}" --output-dir "$capacity" "$@"
  capacity_validation_args+=(--matrix "$capacity")
  python3 "$runner" --preset pareto-whole "${common[@]}" --output-dir "$whole" "$@"
}

prepare_pair 16 dense dense all-q4 "$repo/build-r9700-dense-selection-g16/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
prepare_pair 32 dense dense all-q4 "$repo/build-r9700-dense-selection-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
prepare_pair 16 b128-s16-tau900 xattention-s16-tau900 all-q4 "$repo/build-r9700-xattention-model-s16-tau900/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
prepare_pair 32 b128-s16-tau900 xattention-s16-tau900 all-q4 "$repo/build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
prepare_pair 16 dense dense mixed "$repo/build-r9700-dense-selection-g16/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer"
prepare_pair 32 dense dense mixed "$repo/build-r9700-dense-selection-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer"
prepare_pair 16 b128-s16-tau900 xattention-s16-tau900 mixed "$repo/build-r9700-xattention-model-s16-tau900/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer"
prepare_pair 32 b128-s16-tau900 xattention-s16-tau900 mixed "$repo/build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer"
prepare_pair 16 dense dense four-role "$repo/build-r9700-dense-selection-g16/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" --require-fp8-hybrid --hybrid-width-tool "$repo/build-r9700-dense-selection-g16/src/ninfer_r9700_runtime_planner_qual"
prepare_pair 32 dense dense four-role "$repo/build-r9700-dense-selection-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" --require-fp8-hybrid --hybrid-width-tool "$repo/build-r9700-dense-selection-g32/src/ninfer_r9700_runtime_planner_qual"
prepare_pair 16 b128-s16-tau900 xattention-s16-tau900 four-role "$repo/build-r9700-xattention-model-s16-tau900/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" --require-fp8-hybrid --hybrid-width-tool "$repo/build-r9700-xattention-model-s16-tau900/src/ninfer_r9700_runtime_planner_qual"
prepare_pair 32 b128-s16-tau900 xattention-s16-tau900 four-role "$repo/build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench" "$repo/out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer" --require-fp8-hybrid --hybrid-width-tool "$repo/build-r9700-xattention-model-s16-tau900-g32/src/ninfer_r9700_runtime_planner_qual"

if ! verified_chunk="$(python3 -c 'from pathlib import Path; from tools.bench.select_prefill_chunk import validate_selection_record; print(validate_selection_record(Path("profiles/bench/prefill-chunk-selection-receipt-bound-n16k16-20260905.json"))["selected_prefill_chunk"])')"; then
  echo "selected-chunk authority changed while preparing post-chunk matrices" >&2
  exit 1
fi
[[ $verified_chunk == "$selected_chunk" ]]
python3 -m tools.bench.validate_post_chunk_capacity_campaign \
  "${capacity_validation_args[@]}" --prepared
test "$(sha256sum "$selection" | cut -d' ' -f1)" = "$selection_sha"
