# tools/bench

Offline helper for the `ninfer_bench` throughput tool. Correctness/parity tooling lives separately
under [`tools/parity`](../parity).

## Corpus baker

`ninfer_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with the local Qwen3.8-27B tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, and source description
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `NINFER_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.8-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.8-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Check that the committed .ids and its descriptive manifest agree; no tokenizer or source needed.
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## NInfer performance matrix

`run_ninfer_bench_matrix.py` runs the layered public-Engine `ninfer_bench` matrix against the native
`.ninfer` artifact and stores its local reports under `profiles/bench/`. The exact selected artifact
is required rather than inferred from a default path. The other defaults are:

```text
binary:   build-r9700/bench/ninfer_bench
corpus:   bench/fixtures/bench_corpus.ids
```

The matrix treats MTP `k=3` with the optimized proposal head as the primary path, keeps `k=0` and
`k=5` as controls, and sweeps `k=0..5` on representative context-decode cases. Decode-bearing cases
cover Device Graph and eager execution; prefill-only cases vary prompt length and prefill chunk.

```bash
# Configure the benchmark targets once; they are off in the default public build.
cmake -S . -B build-r9700 -DNINFER_BUILD_BENCHMARKS=ON

# Inspect commands without running the model. The path need not exist for a dry run.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core \
  --weights /absolute/path/to/selected.ninfer --dry-run

# Main run. Builds build-r9700/bench/ninfer_bench first, then writes JSON and summary.csv.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core \
  --weights /absolute/path/to/selected.ninfer

# Qualification run: reject a binary/report from the wrong compiled cache group.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full \
  --weights /absolute/path/to/selected.ninfer \
  --expected-kv-value-group 16

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full \
  --weights /absolute/path/to/selected.ninfer

# Optional isolated phase diagnostic: 8K/32K MTP3 prefill and prefix-reuse-seeded decode.
# Static-profile selection does not require this diagnostic; pareto-whole supplies its
# decision-relevant timings from ordinary fresh-request repetitions.
python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto \
  --weights /absolute/path/to/selected.ninfer \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1

# Complete base-selection speed evidence: ordinary spec-none fresh-prompt prefill, decode,
# makespan, and output throughput for the same Pareto profiles. MTP3 is optional diagnostic
# exact-token/state/graph regression evidence and is neither required nor ranked. DFlash remains
# the required downstream speculative admission for the selected base profile.
python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-whole \
  --weights /absolute/path/to/selected.ninfer \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1

# Required 32K workload feasibility under the production MTP3 startup plan.
python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-feasibility \
  --weights /absolute/path/to/selected.ninfer \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1

# Exact effective maximum at the model-native 262,144-token per-request ceiling.
python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-capacity \
  --weights /absolute/path/to/selected.ninfer \
  --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1

After chunk selection, prepare the four-role FP8 hybrid whole and capacity matrices without
launching either benchmark. These commands inspect and hash the real artifact, its adjacent
conversion receipt, the compile-matched benchmark executable, and the host-only runtime planner
from the same shared-workspace build. They require one supported selected chunk and exactly C=1..4.

```bash
common=(
  --bench build-r9700-dense-selection-g16/bench/ninfer_bench --no-build
  --weights out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer
  --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK"
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4
  --expected-kv-value-group 16
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8
  --expected-fp8-qk-wmma 1 --expected-xattention-profile dense
  --require-fp8-hybrid
  --hybrid-width-tool build-r9700-dense-selection-g16/src/ninfer_r9700_runtime_planner_qual
)

python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-capacity \
  "${common[@]}" --prepare-only \
  --output-dir profiles/bench/fp8-hybrid-capacity-native-c1-4-20260904

python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-whole \
  "${common[@]}" --prepare-only \
  --output-dir profiles/bench/fp8-hybrid-whole-p8192-p32768-c1-4-20260904
```

Review each `manifest.json` and `commands.sh`. After the GPU queue is explicitly released, rerun
the corresponding command with `--resume` in place of `--prepare-only`. Resume revalidates the
artifact/receipt, benchmark and planner hashes, fixed C1..4 inventories, command matrix, and current
`auto` power state before executing missing cells. `pareto-whole` contains one spec-none ordinary
fresh-request row per concurrency and is the base-selection timing input. `pareto-capacity` is the
model-native capacity matrix. Neither prepared manifest is performance evidence.

# Production prefill-chunk selection starts with all twelve candidates at 8K, C=1, using the
# spec-none ordinary Text-prefill path with 3/1 timing.
# The runner fails before creating the output directory unless the R9700 sysfs power profile is
# `auto`; dry runs record the requirement but do not inspect hardware state.
# Invoke this template once for every recipe x G16/G32 x dense/sparse tuple. Each output directory
# must be identity-qualified and unique. For the four-role artifact add `--require-fp8-hybrid`
# and the matching `--hybrid-width-tool`; the runner derives and records the artifact receipt and
# planner-capacity authority in every screen/finalist manifest.
python3 tools/bench/run_ninfer_bench_matrix.py --preset prefill-chunk \
  --bench "$BENCH" --no-build --weights "$RECIPE_ARTIFACT" \
  --expected-kv-value-group "$GROUP" --expected-q4-activation-bits 8 \
  --expected-w8-activation-bits 8 --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile "$ATTENTION" --output-dir "$SCREEN_DIR"

# Pass all twelve completed 8K directories. This emits the same two global 32K finalists for every
# tuple; it does not select per-candidate chunks.
python3 tools/bench/select_prefill_chunk.py \
  --screen profiles/bench/prefill-chunk-screen-all-q4-g16-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-all-q4-g32-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-all-q4-g16-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-all-q4-g32-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-mixed-g16-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-mixed-g32-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-mixed-g16-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-mixed-g32-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-four-role-g16-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-four-role-g32-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-four-role-g16-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-four-role-g32-xattention-s16-tau900-20260904 \
  --out profiles/bench/prefill-chunk-screening-20260904.json

# Reopen the exact twelve 8K roots, recompute the complete screening authority, and compare it
# byte-for-structure with the retained schema-v2 record before reading its two finalists. Capture
# the command substitution status explicitly; process substitution would mask verifier failure.
set -euo pipefail
if ! PREFILL_FINALIST_TEXT="$(python3 tools/bench/select_prefill_chunk.py \
  --screen profiles/bench/prefill-chunk-screen-all-q4-g16-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-all-q4-g32-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-all-q4-g16-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-all-q4-g32-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-mixed-g16-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-mixed-g32-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-mixed-g16-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-mixed-g32-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-four-role-g16-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-four-role-g32-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-four-role-g16-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-four-role-g32-xattention-s16-tau900-20260904 \
  --verify-screening profiles/bench/prefill-chunk-screening-20260904.json)"; then
  echo "prefill-chunk screening verification failed" >&2
  exit 1
fi
mapfile -t PREFILL_FINALISTS <<<"$PREFILL_FINALIST_TEXT"
[[ "${#PREFILL_FINALISTS[@]}" -eq 2 ]]
FINALIST_A="${PREFILL_FINALISTS[0]}"
FINALIST_B="${PREFILL_FINALISTS[1]}"
[[ "$FINALIST_A" != "$FINALIST_B" ]]

# Fail before the first launch if any fixed input is absent or output already exists. The runner
# independently checks each explicit artifact/executable/group/profile identity before creating
# its output directory.
for BENCH in \
  build-r9700-dense-selection-g16/bench/ninfer_bench \
  build-r9700-dense-selection-g32/bench/ninfer_bench \
  build-r9700-xattention-model-s16-tau900/bench/ninfer_bench \
  build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench; do
  [[ -x "$BENCH" ]]
done
for ARTIFACT in \
  out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer \
  out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer; do
  [[ -f "$ARTIFACT" ]]
done
for FINALIST_DIR in \
  profiles/bench/prefill-chunk-finalist-all-q4-g16-dense-20260904 \
  profiles/bench/prefill-chunk-finalist-all-q4-g32-dense-20260904 \
  profiles/bench/prefill-chunk-finalist-all-q4-g16-xattention-s16-tau900-20260904 \
  profiles/bench/prefill-chunk-finalist-all-q4-g32-xattention-s16-tau900-20260904 \
  profiles/bench/prefill-chunk-finalist-mixed-g16-dense-20260904 \
  profiles/bench/prefill-chunk-finalist-mixed-g32-dense-20260904 \
  profiles/bench/prefill-chunk-finalist-mixed-g16-xattention-s16-tau900-20260904 \
  profiles/bench/prefill-chunk-finalist-mixed-g32-xattention-s16-tau900-20260904 \
  profiles/bench/prefill-chunk-finalist-four-role-g16-dense-20260904 \
  profiles/bench/prefill-chunk-finalist-four-role-g32-dense-20260904 \
  profiles/bench/prefill-chunk-finalist-four-role-g16-xattention-s16-tau900-20260904 \
  profiles/bench/prefill-chunk-finalist-four-role-g32-xattention-s16-tau900-20260904; do
  [[ ! -e "$FINALIST_DIR" ]]
done

run_prefill_finalist() {
  local BENCH="$1" ARTIFACT="$2" GROUP="$3" ATTENTION="$4" FINALIST_DIR="$5"
  local WIDTH_TOOL="${6:-}"
  local HYBRID_ARGS=()
  if [[ -n "$WIDTH_TOOL" ]]; then
    HYBRID_ARGS=(--require-fp8-hybrid --hybrid-width-tool "$WIDTH_TOOL")
  fi
  python3 tools/bench/run_ninfer_bench_matrix.py --preset prefill-chunk \
    --prefill-prompt 32768 --prefill-chunk "$FINALIST_A" --prefill-chunk "$FINALIST_B" \
    --bench "$BENCH" --no-build --weights "$ARTIFACT" \
    --expected-kv-value-group "$GROUP" --expected-q4-activation-bits 8 \
    --expected-w8-activation-bits 8 --expected-fp8-qk-wmma 1 \
    --expected-xattention-profile "$ATTENTION" "${HYBRID_ARGS[@]}" \
    --output-dir "$FINALIST_DIR"
}

run_prefill_finalist build-r9700-dense-selection-g16/bench/ninfer_bench \
  out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer 16 dense \
  profiles/bench/prefill-chunk-finalist-all-q4-g16-dense-20260904
run_prefill_finalist build-r9700-dense-selection-g32/bench/ninfer_bench \
  out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer 32 dense \
  profiles/bench/prefill-chunk-finalist-all-q4-g32-dense-20260904
run_prefill_finalist build-r9700-xattention-model-s16-tau900/bench/ninfer_bench \
  out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer 16 b128-s16-tau900 \
  profiles/bench/prefill-chunk-finalist-all-q4-g16-xattention-s16-tau900-20260904
run_prefill_finalist build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench \
  out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer 32 b128-s16-tau900 \
  profiles/bench/prefill-chunk-finalist-all-q4-g32-xattention-s16-tau900-20260904
run_prefill_finalist build-r9700-dense-selection-g16/bench/ninfer_bench \
  out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer 16 dense \
  profiles/bench/prefill-chunk-finalist-mixed-g16-dense-20260904
run_prefill_finalist build-r9700-dense-selection-g32/bench/ninfer_bench \
  out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer 32 dense \
  profiles/bench/prefill-chunk-finalist-mixed-g32-dense-20260904
run_prefill_finalist build-r9700-xattention-model-s16-tau900/bench/ninfer_bench \
  out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer 16 b128-s16-tau900 \
  profiles/bench/prefill-chunk-finalist-mixed-g16-xattention-s16-tau900-20260904
run_prefill_finalist build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench \
  out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer 32 b128-s16-tau900 \
  profiles/bench/prefill-chunk-finalist-mixed-g32-xattention-s16-tau900-20260904
run_prefill_finalist build-r9700-dense-selection-g16/bench/ninfer_bench \
  out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer 16 dense \
  profiles/bench/prefill-chunk-finalist-four-role-g16-dense-20260904 \
  build-r9700-dense-selection-g16/src/ninfer_r9700_runtime_planner_qual
run_prefill_finalist build-r9700-dense-selection-g32/bench/ninfer_bench \
  out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer 32 dense \
  profiles/bench/prefill-chunk-finalist-four-role-g32-dense-20260904 \
  build-r9700-dense-selection-g32/src/ninfer_r9700_runtime_planner_qual
run_prefill_finalist build-r9700-xattention-model-s16-tau900/bench/ninfer_bench \
  out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer 16 b128-s16-tau900 \
  profiles/bench/prefill-chunk-finalist-four-role-g16-xattention-s16-tau900-20260904 \
  build-r9700-xattention-model-s16-tau900/src/ninfer_r9700_runtime_planner_qual
run_prefill_finalist build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench \
  out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer 32 b128-s16-tau900 \
  profiles/bench/prefill-chunk-finalist-four-role-g32-xattention-s16-tau900-20260904 \
  build-r9700-xattention-model-s16-tau900-g32/src/ninfer_r9700_runtime_planner_qual

# Pair --screen and --finalist directories in the same order to emit the final authority.
python3 tools/bench/select_prefill_chunk.py \
  --screen profiles/bench/prefill-chunk-screen-all-q4-g16-dense-20260904 \
  --finalist profiles/bench/prefill-chunk-finalist-all-q4-g16-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-all-q4-g32-dense-20260904 \
  --finalist profiles/bench/prefill-chunk-finalist-all-q4-g32-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-all-q4-g16-xattention-s16-tau900-20260904 \
  --finalist profiles/bench/prefill-chunk-finalist-all-q4-g16-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-all-q4-g32-xattention-s16-tau900-20260904 \
  --finalist profiles/bench/prefill-chunk-finalist-all-q4-g32-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-mixed-g16-dense-20260904 \
  --finalist profiles/bench/prefill-chunk-finalist-mixed-g16-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-mixed-g32-dense-20260904 \
  --finalist profiles/bench/prefill-chunk-finalist-mixed-g32-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-mixed-g16-xattention-s16-tau900-20260904 \
  --finalist profiles/bench/prefill-chunk-finalist-mixed-g16-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-mixed-g32-xattention-s16-tau900-20260904 \
  --finalist profiles/bench/prefill-chunk-finalist-mixed-g32-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-four-role-g16-dense-20260904 \
  --finalist profiles/bench/prefill-chunk-finalist-four-role-g16-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-four-role-g32-dense-20260904 \
  --finalist profiles/bench/prefill-chunk-finalist-four-role-g32-dense-20260904 \
  --screen profiles/bench/prefill-chunk-screen-four-role-g16-xattention-s16-tau900-20260904 \
  --finalist profiles/bench/prefill-chunk-finalist-four-role-g16-xattention-s16-tau900-20260904 \
  --screen profiles/bench/prefill-chunk-screen-four-role-g32-xattention-s16-tau900-20260904 \
  --finalist profiles/bench/prefill-chunk-finalist-four-role-g32-xattention-s16-tau900-20260904 \
  --out profiles/bench/prefill-chunk-selection-20260904.json

NINFER_SELECTED_PREFILL_CHUNK=$(python3 -c \
  'from pathlib import Path; from tools.bench.select_prefill_chunk import validate_selection_record; print(validate_selection_record(Path("profiles/bench/prefill-chunk-selection-20260904.json"))["selected_prefill_chunk"])')

# After schema-v7 terminal selection, acquire and validate the separate dense C1 low-context
# prefill ladder. This entry point resolves the winner's exact artifact/value group and its unique
# compile-matched dense control directly from the authority; it also reopens the selected-chunk
# planner for the hybrid recipe. It runs five non-speculative 3/1 reports under auto and publishes
# the evaluation only after complete recomputation. A valid result below 2,000 tok/s at P=2,048 is
# retained with a failed gate so profiling can continue from the selected route.
bash profiles/bench/low-context-selected-ladder-20260905/run-and-publish.sh \
  --execute-gpu-campaign

export NINFER_LOW_CONTEXT_EVALUATION=profiles/bench/low-context-prefill-evaluation-20260905.json
NINFER_SELECTED_ARTIFACT=$(python3 -c 'import json,os; print(json.load(open(os.environ["NINFER_LOW_CONTEXT_EVALUATION"]))["artifact"]["path"])')
NINFER_SELECTED_WEIGHTS_ID=$(python3 -c 'import json,os; print(json.load(open(os.environ["NINFER_LOW_CONTEXT_EVALUATION"]))["artifact"]["weights_id"])')
NINFER_SELECTED_GROUP=$(python3 -c 'import json,os; print(json.load(open(os.environ["NINFER_LOW_CONTEXT_EVALUATION"]))["expected_kv_value_group"])')
NINFER_SELECTED_PREFILL_CHUNK=$(python3 -c 'import json,os; print(json.load(open(os.environ["NINFER_LOW_CONTEXT_EVALUATION"]))["selected_prefill_chunk"])')
NINFER_SELECTED_DENSE_BENCH=$(python3 -c 'import json,os; print(json.load(open(os.environ["NINFER_LOW_CONTEXT_EVALUATION"]))["bench"]["path"])')

# If P2048 remains an unresolved selected-route bottleneck, prepare its one-repetition selected-
# region trace from the exact validated ladder/evaluation/terminal-selection authorities. This
# does not rerun the 3/1 timing authority or require that the explicit throughput gate passed.
python3 -m tools.bench.prepare_whole_profile \
  --low-context-manifest profiles/bench/low-context-prefill-selected-20260905/manifest.json \
  --low-context-evaluation profiles/bench/low-context-prefill-evaluation-20260905.json \
  --terminal-selection profiles/bench/pareto-result-post-promotion-20260905.json \
  --executable "$NINFER_SELECTED_DENSE_BENCH" --artifact "$NINFER_SELECTED_ARTIFACT" \
  --concurrency 1 --prompt-tokens 2048 --generated-tokens 0 \
  --expected-weights-id "$NINFER_SELECTED_WEIGHTS_ID" --expected-kv-value-group "$NINFER_SELECTED_GROUP" \
  --expected-xattention-profile dense --expected-prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
  --kind trace --question "which production kernel owns selected dense P2048 prefill?" \
  --out profiles/rocprof/selected-dense-p2048-trace-20260905

# Only after that trace names a material kernel family, prepare a fresh dispatch-scoped counter
# plan. Replace the regex with the exact trace-observed family or bounded alternation. Its generated
# script requires auto initially, temporarily selects profile_standard, and restores auto with an
# EXIT trap; its duration is attribution-only.
python3 -m tools.bench.prepare_whole_profile \
  --low-context-manifest profiles/bench/low-context-prefill-selected-20260905/manifest.json \
  --low-context-evaluation profiles/bench/low-context-prefill-evaluation-20260905.json \
  --terminal-selection profiles/bench/pareto-result-post-promotion-20260905.json \
  --executable "$NINFER_SELECTED_DENSE_BENCH" --artifact "$NINFER_SELECTED_ARTIFACT" \
  --concurrency 1 --prompt-tokens 2048 --generated-tokens 0 \
  --expected-weights-id "$NINFER_SELECTED_WEIGHTS_ID" --expected-kv-value-group "$NINFER_SELECTED_GROUP" \
  --expected-xattention-profile dense --expected-prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
  --kind dispatch-pmc --kernel-include-regex 'TRACE_NAMED_KERNEL_FAMILY_REGEX' \
  --question "what are the selected P2048 kernel cache-hit ratios and wave count?" \
  --out profiles/rocprof/selected-dense-p2048-pmc-20260905
```

Both preparation modes revalidate the full five-row schema-v14 ladder, schema-v1 evaluation, and
schema-v7 terminal selection; bind and rehash the selected executable, artifact, corpus, P2048 raw
report, and every authority; and require the dense C1 semantic `spec=none` P2048 3/1 source row.
Only `-r` becomes `1`, the report output moves under the fresh profile directory, and
`--profile-measured` is appended. The original warmup remains outside the measured region. The
generated trace script guards `auto`; the generated PMC script requires `auto`, temporarily
selects `profile_standard`, and restores and verifies `auto` with an EXIT trap.
Each script writes fresh `power-profile-before.txt` and `power-profile-after.txt` endpoint evidence;
the after value is captured even when rocprof fails, while the script preserves rocprof's status.

Validate the completed `auto` trace before analyzing or assigning any dispatch role:

```bash
python3 -m tools.bench.validate_profile_trace \
  --plan profiles/rocprof/selected-dense-p2048-trace-20260905/plan.json \
  --benchmark-report profiles/rocprof/selected-dense-p2048-trace-20260905/benchmark-report.json \
  --database /explicit/path/to/trace-results.db \
  --power-before profiles/rocprof/selected-dense-p2048-trace-20260905/power-profile-before.txt \
  --power-after profiles/rocprof/selected-dense-p2048-trace-20260905/power-profile-after.txt \
  --terminal-selection profiles/bench/pareto-result-post-promotion-20260905.json \
  --artifact "$NINFER_SELECTED_ARTIFACT" --executable "$NINFER_SELECTED_DENSE_BENCH" \
  --corpus bench/fixtures/bench_corpus.ids \
  --out profiles/rocprof/selected-dense-p2048-trace-20260905/evidence.json
```

The schema-v1 `ninfer_r9700_selected_profile_trace` authority mirrors the PMC validator's complete
low-context provenance checks, but requires the exact marker/kernel/memory-copy trace command and
`auto` endpoint evidence. It reopens and rehashes the plan, schema-v20 capture report, database,
terminal selection, selected artifact and executable, corpus, low-context manifest/evaluation, and
unprofiled P2048 3/1 report. The database must contain one R9700/gfx1201 process running the exact
planned command. Every kernel dispatch is retained without model-role inference: stable trace and
numeric rocprof IDs, symbol, raw ROCTX region, nanosecond interval, stream, grid/workgroup geometry,
and available VGPR/accumulator-VGPR/LDS/scratch fields. Count and independent service time must
agree exactly with both the database and the embedded legacy analyzer; wall-union time is retained
separately. `selected_route` exposes the exact artifact/executable/group/profile/chunk identity and
the bound unprofiled P2048 gate result for downstream reconciliation. Trace durations remain
attribution-only. Missing resources remain explicit null values, ambiguous ROCTX is preserved for
the reconciler to reject, and the validator refuses overwrite and rehashes every input before
returning.

Validate a completed P2048 PMC capture using only explicit paths. The native counter CSV and rocpd
database must come from that same invocation; never pair either with the separate `auto` trace:

```bash
python3 -m tools.bench.validate_profile_pmc \
  --plan profiles/rocprof/selected-dense-p2048-pmc-20260905/plan.json \
  --benchmark-report profiles/rocprof/selected-dense-p2048-pmc-20260905/benchmark-report.json \
  --counter-csv /explicit/path/to/counter_collection.csv \
  --database /explicit/path/to/results.db \
  --power-before profiles/rocprof/selected-dense-p2048-pmc-20260905/power-profile-before.txt \
  --power-after profiles/rocprof/selected-dense-p2048-pmc-20260905/power-profile-after.txt \
  --terminal-selection profiles/bench/pareto-result-post-promotion-20260905.json \
  --artifact "$NINFER_SELECTED_ARTIFACT" --executable "$NINFER_SELECTED_DENSE_BENCH" \
  --corpus bench/fixtures/bench_corpus.ids \
  --out profiles/rocprof/selected-dense-p2048-pmc-20260905/evidence.json
```

The validator refuses overwrite, rehashes the complete authority/input chain, requires the exact
R9700/gfx1201 workload and counter inventory, and compares Decimal-parsed CSV aggregates with the
same dispatches in the database. It derives GL2 and TCP hit ratios only from positive base-event
denominators, requires positive SQ wave activity, retains collected all-zero counters as
`observed_zero`, and never converts missing or zero-denominator data into a cache claim. ROCTX stage
identity is accepted only through the same database's PMC-event to kernel-event/region join. The
profile-standard durations and endpoint-only power observations remain attribution-only.

```bash
# XAttention admission needs current dense G16/G32 controls in addition to the isolated sparse
# matrices below. Keep these builds separate from retained historical binaries. Each invocation
# emits four capacity and four whole reports from one unchanged executable. Each whole report retains
# both 8K/32K rows and their separately timed prefill/decode phases, avoiding a redundant eight-point
# phase campaign without dropping a selection objective.
cmake -S . -B build-r9700-dense-selection-g16 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON \
  -DNINFER_R9700_KV_VALUE_GROUP=16 \
  -DNINFER_R9700_Q4_ACTIVATION_BITS=8 -DNINFER_R9700_W8_ACTIVATION_BITS=8 \
  -DNINFER_R9700_FP8_QK_WMMA=1 -DNINFER_R9700_XATTENTION_STRIDE=16 \
  -DNINFER_R9700_XATTENTION_TAU_PERMILLE=900 \
  -DNINFER_R9700_XATTENTION_QUALIFICATION=OFF
cmake -S . -B build-r9700-dense-selection-g32 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON \
  -DNINFER_R9700_KV_VALUE_GROUP=32 \
  -DNINFER_R9700_Q4_ACTIVATION_BITS=8 -DNINFER_R9700_W8_ACTIVATION_BITS=8 \
  -DNINFER_R9700_FP8_QK_WMMA=1 -DNINFER_R9700_XATTENTION_STRIDE=16 \
  -DNINFER_R9700_XATTENTION_TAU_PERMILLE=900 \
  -DNINFER_R9700_XATTENTION_QUALIFICATION=OFF
cmake --build build-r9700-dense-selection-g16 --target ninfer_bench --parallel 4
cmake --build build-r9700-dense-selection-g32 --target ninfer_bench --parallel 4

run_dense_selection_pair() {
  local group="$1" build="$2"
  local common=(
    --bench "${build}/bench/ninfer_bench" --no-build
    --weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer
    --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK"
    --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4
    --expected-kv-value-group "${group}"
    --expected-q4-activation-bits 8 --expected-w8-activation-bits 8
    --expected-fp8-qk-wmma 1 --expected-xattention-profile dense
  )
  python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-capacity "${common[@]}" \
    --output-dir "profiles/bench/pareto-capacity-post-promotion-dense-all-q4-g${group}-20260904"
  python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-whole "${common[@]}" \
    --output-dir "profiles/bench/pareto-whole-post-promotion-dense-all-q4-g${group}-20260904"
}
run_dense_selection_pair 16 build-r9700-dense-selection-g16
run_dense_selection_pair 32 build-r9700-dense-selection-g32

# Compile-isolated XAttention uses the same presets but must bind its private profile.
# Point --bench at the matching G16 or G32 qualification build and use a fresh output directory.
python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-capacity \
  --bench build-r9700-xattention-model-s16-tau900/bench/ninfer_bench --no-build \
  --weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-kv-value-group 16 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile b128-s16-tau900 \
  --output-dir profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-all-q4-g16-20260904

python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-capacity \
  --bench build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench --no-build \
  --weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-kv-value-group 32 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile b128-s16-tau900 \
  --output-dir profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-all-q4-g32-20260904

# After an interrupted run, repeat its exact command with --resume. The existing schema-v14
# manifest must match the executable bytes and full campaign contract; valid cells are skipped.
# Resume reruns an invalid or partial report in place; preserve that root and use a fresh output
# directory for the complete candidate instead when retained failure evidence must remain immutable.
# A non-resume launch rejects an existing output directory instead of overwriting retained evidence.
# Each completed capacity directory contains the C=1..4 matrix (four reports). Never target the
# retained markerfree-layout dense capacity directories with either sparse command.

python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-whole \
  --bench build-r9700-xattention-model-s16-tau900/bench/ninfer_bench --no-build \
  --weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-kv-value-group 16 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile b128-s16-tau900 \
  --output-dir profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-all-q4-g16-20260904

python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-whole \
  --bench build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench --no-build \
  --weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-kv-value-group 32 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile b128-s16-tau900 \
  --output-dir profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-all-q4-g32-20260904

# The C<=4 product domain restores the mixed Q4/W8-MSE recipe to the base decision. Measure its
# complete dense/sparse Cartesian controls with the same binaries and presets; recipe changes only
# the bound artifact and fresh output namespace.
run_mixed_selection_pair() {
  local group="$1" dense_build="$2" sparse_build="$3"
  local profile build tag
  for profile in dense b128-s16-tau900; do
    if [[ "$profile" == dense ]]; then
      build="$dense_build"
      tag=dense
    else
      build="$sparse_build"
      tag=xattention-s16-tau900
    fi
    local common=(
      --bench "${build}/bench/ninfer_bench" --no-build
      --weights out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer
      --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK"
      --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4
      --expected-kv-value-group "$group"
      --expected-q4-activation-bits 8 --expected-w8-activation-bits 8
      --expected-fp8-qk-wmma 1 --expected-xattention-profile "$profile"
    )
    python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-capacity "${common[@]}" \
      --output-dir "profiles/bench/pareto-capacity-post-promotion-${tag}-mixed-g${group}-20260904"
    python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-whole "${common[@]}" \
      --output-dir "profiles/bench/pareto-whole-post-promotion-${tag}-mixed-g${group}-20260904"
  done
}
run_mixed_selection_pair 16 build-r9700-dense-selection-g16 \
  build-r9700-xattention-model-s16-tau900
run_mixed_selection_pair 32 build-r9700-dense-selection-g32 \
  build-r9700-xattention-model-s16-tau900-g32

# Run a group's whole matrix only after its capacity matrix completes. Do not rebuild or replace
# the artifact between the two runs: the assembler requires byte-identical benchmark and artifact
# provenance. Pareto-whole produces four commands: one ordinary spec-none report at each C=1..4,
# each with two 8K/32K fresh-request rows. Both v14 manifests
# must be complete with no failures.json. The in-progress/retained pareto-phase directories are
# diagnostic history and are not inputs to static-profile selection. Use --resume only with the
# exact original command in the corresponding capacity or whole directory.
# Pareto-whole is terminal speed evidence: the runner requires card2's live power level to be
# `auto` before it creates the output directory, records the corpus hash and observed level, emits
# the same precheck in commands.sh, and rechecks `auto` after the complete matrix.

# Keep pareto-whole at one discarded warmup plus three measured repetitions for each 8K/32K row.
# The Engine and decode Device Graph are constructed and primed before the tests, and every whole
# repetition disables prefix reuse and submits a fresh C-lane prompt set. That makes the repetitions
# semantically matched, but graph priming does not exercise the context-dependent Text-prefill grid
# or establish its thermal/clock state. There is no retained zero-warmup whole-inference evidence
# that would justify counting that first traversal. Three measured values are also the minimum that
# can identify one transient while retaining a center and spread; the schema-v7 selector compares
# exact per-cell dominance and maximin ratios without a noise tolerance, including potentially close
# G16/G32 cells. In the available post-warmup three-repetition long-context reports, timing CV is
# 0.019--0.170%, full spread reaches 0.329%, and the first measured value differs from the following
# pair by as much as 0.198%. Those results support the current 1+3 floor, not a two-sample or
# zero-warmup gate. Do not multiply repetitions by lane: one C-lane repetition already measures the
# complete startup-fixed concurrent product workload.
# Terminal base assembly consumes only spec-none ordinary capacity and whole rows. MTP acceptance
# accounting remains optional diagnostic evidence and cannot alter base selection.

# DFlash runs begin only after the terminal base decision selects the artifact, cache group, and
# ordinary Text-prefill profile. Convert and measure the companion for that selected base recipe.
# The selected binary applies XAttention only to ordinary Phase::Prefill (including DFlash feature
# capture); DFlash proposal attention and target/tree verification remain dense in every build.
## This preparation fails until the schema-v7 base winner exists. It resolves the exact base,
## companion, selected chunk, cache group, attention build, and conversion receipt. It runs only
## the selected companion and derives K/W from the physical shortlist; no K/W is entered here.
## Companion conversion is bound to `/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python`
## with `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib`; preparation fails if that
## interpreter cannot import ROCm Torch and safetensors.
bash profiles/bench/selected-dflash-prepare-20260905/prepare.sh

## Root schedules this future command after reviewing the generated plan. The staged runner uses
## C1 for the shortlist, C1..4 for every derived frontier capacity row, and the full Pareto matrix
## only for capacity-eligible rows before publishing one schema-v3 selection without overwrite.
bash profiles/bench/selected-dflash-20260905/commands.sh

# Optional diagnostic only: required-32K feasibility is not a Pareto capacity objective.
python3 tools/bench/run_ninfer_bench_matrix.py --preset dflash-feasibility \
  --prefill-chunk "${DFLASH_PREFILL_CHUNK}" \
  --dflash-draft-tokens 7 --dflash-verify-width 12 \
  --bench "${DFLASH_BENCH}" --no-build \
  --weights "${DFLASH_ARTIFACT}" \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-kv-value-group "${DFLASH_GROUP}" \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile "${DFLASH_ATTENTION}"

# Run only the MTP draft-window sweep.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full --suite mtp_sweep \
  --weights /absolute/path/to/selected.ninfer

# Native Engine concurrency decomposition at every supported fixed C.
python3 tools/bench/run_ninfer_bench_matrix.py --preset concurrency \
  --weights /absolute/path/to/selected.ninfer \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4
```

The `prefill-chunk` preset writes schema-v20 raw reports and a schema-v14 manifest, binding prompt,
chunk, artifact, executable, corpus, cache group, and attention profile. The selection owner first
normalizes each chunk's 8K throughput to that candidate's best result and advances the two chunks
with the greatest minimum ratio across all twelve candidates. Every candidate measures those same
two finalists at 32K. The final decision preserves each 8K objective's denominator from its complete
four-chunk screen, normalizes each 32K objective against its better finalist, and maximizes the
minimum of all twenty-four ratios. Exact ties minimize the maximum retained workspace peak, then select
the smaller chunk. The schema-v2 record reopens every raw report and binds every source hash.

These prefill-only cases request exactly one output token and run the spec-none ordinary route with
`--draft-tokens 0`. They measure Text prefill plus target sampling without MTP state/KV preparation,
proposal-head work, drafted tokens, or decode rounds. The selector requires that exact protocol and
records `base_chunk_profile=spec-none-ordinary`; DFlash selected-only evaluation later owns its own
feature/state performance.

Supply the selected value exactly once as `--prefill-chunk N` to fresh `pareto`, `pareto-whole`,
`pareto-feasibility`, and `pareto-capacity` runs. There is no implicit 4,096 fallback for these
campaign presets. Never resume a retained 4096 matrix with a different chunk. The isolated
structured-fixture operator speedup does not replace this real-model selection.

The ordinary-decode diagnostic uses the same all-Q4 G16 B128/S16/tau900 benchmark executable,
artifact, 4,096-token chunk, graph route, 8K prompt, and 256-token output extent as the retained
19.24 tok/s MTP3 row, changing only speculative execution. Its command explicitly uses
`--draft-tokens 0` with no speculative backend or draft-head option, and the report must record
`spec=none`, `draft_tokens=0`, `speculative_execution=false`, per-test
`speculative.enabled=false`, and the ordinary `device_graph` decode path. The fixed preset runs
only C=1 with three measured repetitions after one warmup, retains separate prefill/decode timings
and rates, rejects reuse of an existing output directory, and binds/rechecks the exact `auto` power
profile.

The retained run at the path below passed the strict validator with speculative execution disabled
in all three repetitions. It measured `239.5738442` prefill tok/s (`34.1947324 s`) and
`8.354688852` decode output tok/s (`30.64198944 s`), with `3.963623176` whole output tok/s over
`64.84022237 s`. Its manifest SHA-256 is
`a2ce50522910bd1442156383c85f359315c33bc5964578ba3ca28bf9685e104a`; the raw whole-report
SHA-256 is `049f3e4712a65ba618f28b47d830a96019cd16dea8dd70a920e1385aca4f1c96`.
The earlier `19.24458338` decode tok/s value belongs to the MTP3 speculative comparison row and
must not be described as ordinary decode performance.

```bash
OUT=profiles/bench/ordinary-none-xattention-s16-tau900-all-q4-g16-c1-8k-20260904
test "$(tr -d '\n' </sys/class/drm/card2/device/power_dpm_force_performance_level)" = auto
test ! -e "$OUT"
python3 tools/bench/run_ninfer_bench_matrix.py --preset ordinary-diagnostic \
  --bench build-r9700-xattention-model-s16-tau900/bench/ninfer_bench --no-build \
  --weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer --concurrency 1 \
  --expected-kv-value-group 16 --expected-q4-activation-bits 8 \
  --expected-w8-activation-bits 8 --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile b128-s16-tau900 --output-dir "$OUT"
python3 tools/bench/check_ordinary_decode_report.py "$OUT" \
  --executable build-r9700-xattention-model-s16-tau900/bench/ninfer_bench \
  --artifact out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer
```

Default outputs:

```text
profiles/bench/ninfer-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/c<N>/<case>.json
  logs/<suite>.<case>.c<N>.stderr.txt
  diagnostics/*.raw.json
  diagnostics/*.json
  greedy-token-parity.json
  dflash-proposal-determinism.json
  dflash-generated-quality.json
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check. `--no-build` uses the binary supplied by `--bench` without
building it. The benchmark sizes its pending queue to startup concurrency, sets the Engine fallback
timeout to `UINT32_MAX` milliseconds, and supplies an unbounded deadline on every concurrent
submission. Long queued C>1 prefill and whole-inference requests therefore cannot inherit the
30-second product default and expire. This is benchmark-only behavior; CLI and serving defaults
are unchanged.

Whole-inference profiling is conditional: do it only when the completed unprofiled base or DFlash
matrix leaves a named bottleneck question that could change the implementation. Prepare exactly
one already-measured whole point from its completed `pareto-whole` or `dflash-pareto` directory:

```bash
python3 -m tools.bench.prepare_whole_profile \
  --matrix-dir profiles/bench/SELECTED_COMPLETE_MATRIX \
  --out profiles/rocprof/SELECTED-C4-32K-trace \
  --concurrency 4 --prompt-tokens 32768 --generated-tokens 256 \
  --expected-weights-id SELECTED_WEIGHTS_ID --expected-kv-value-group SELECTED_GROUP \
  --expected-xattention-profile SELECTED_ATTENTION_PROFILE \
  --expected-prefill-chunk SELECTED_PREFILL_CHUNK \
  --kind trace --question "which dispatch family dominates selected C4 32K makespan?"
```

The CPU-only preparation validates the schema-v14 matrix, absence of `failures.json`, exact
artifact/executable bytes, schema-v20 source report, group, plane-layout, compile-bound XAttention
identity, selected prefill chunk, the source timing's pre/post `auto` power state, concurrency, and
measured geometry. It writes
one command rather than running it. The explicit selected weights identity and cache group prevent
a completed losing matrix from being profiled accidentally. A DFlash matrix additionally requires
the selected `--expected-dflash-draft-tokens` and `--expected-dflash-verify-width`. The generated
command uses rocprof selected-region collection and the
benchmark's `ninfer_bench_measured` boundary, so artifact load, graph priming, and warmup are not
captured. Nested runtime ranges are default-off in every product executable and are enabled only
inside this explicit one-repetition boundary, so unprofiled matrix timing performs no label
formatting or ROCTX push/pop calls. Trace mode records marker, kernel, and memory-copy activity and
its generated script fails closed unless the GPU is still in `auto`. Generate a separate
`--kind dispatch-pmc` plan only after the trace names a dispatch family, and pass that family with
`--kernel-include-regex`. Its script requires initial `auto`, uses `profile_standard` only during
collection, and restores and records `auto` afterward and from its EXIT trap. It
records the dispatch-scoped nonzero GL2C hit/miss, TCP request/miss, GL2C read/write request, and
SQ wave controls. Those counters support relative cache-hit and activity diagnosis, not absolute
cache/HBM bandwidth. Profiler timing is attribution-only and never replaces the matrix's
unprofiled whole-inference timing.

After the trace identifies a kernel family, prepare its counter capture without changing the
matrix timing authority:

```bash
python3 -m tools.bench.prepare_whole_profile \
  --matrix-dir profiles/bench/SELECTED_COMPLETE_MATRIX \
  --out profiles/rocprof/SELECTED-C4-32K-attention-pmc \
  --concurrency 4 --prompt-tokens 32768 --generated-tokens 256 \
  --expected-weights-id SELECTED_WEIGHTS_ID --expected-kv-value-group SELECTED_GROUP \
  --expected-xattention-profile SELECTED_ATTENTION_PROFILE \
  --expected-prefill-chunk SELECTED_PREFILL_CHUNK \
  --kind dispatch-pmc --kernel-include-regex 'EXACT_KERNEL_FAMILY_REGEX' \
  --question "what are the selected dispatch family cache-hit ratios and wave count?"
```

Run either generated script only while the sysfs profile is `auto`. The trace leaves it unchanged;
the dispatch-PMC script elevates only its two sysfs writes, temporarily selects `profile_standard`,
and restores and verifies `auto` after rocprof and from its EXIT trap.

Analyze a completed selected-region trace using explicit paths (never a newest-file glob):

```bash
python3 -m tools.bench.analyze_whole_profile \
  --database profiles/rocprof/EXPLICIT-TRACE/rocprof-trace/r2d2/EXPLICIT_results.db \
  --benchmark-report profiles/rocprof/EXPLICIT-TRACE/benchmark-report.json \
  --out-json profiles/rocprof/EXPLICIT-TRACE/prefill-attribution.json
```

This also accepts a one-test prefill-only (`pp`) schema-v20 report. Kernel and memory-copy stage
ownership comes from rocprof's captured ROCTX `region` association, not temporal containment in
asynchronous host ranges. The output separates base Text, MTP, prefill orchestration, unattributed
prefill work, and measured non-prefill work; compares marker and symbol families; and reports top
kernels, copy calls/bytes, and the union-derived kernel-inactive share of Text-prefill wall time.
The completeness fields make missing region associations visible. Marker/kernel/copy trace does
not collect HIP runtime/API activity, so its kernel-inactive wall gap cannot be split exactly into
host execution and true GPU idle; collect a new runtime/API trace only if that distinction becomes
the named profiling question.

Build logical roofline evidence only after an exact static target schedule has been reconciled with
the validated selected-region trace:

```bash
python3 -m tools.bench.prepare_qwen3_8_27b_dispatch_schedule \
  --trace-authority profiles/rocprof/EXPLICIT-TRACE/validated-trace.json \
  --artifact /absolute/path/to/SELECTED.ninfer \
  --executable /absolute/path/to/ninfer-bench \
  --kv-value-group SELECTED_GROUP --prefill-chunk SELECTED_PREFILL_CHUNK \
  --out profiles/rocprof/EXPLICIT-TRACE/static-dispatch-schedule.json

python3 -m tools.bench.reconcile_qwen3_8_27b_dispatches \
  --trace-authority profiles/rocprof/EXPLICIT-TRACE/validated-trace.json \
  --static-schedule profiles/rocprof/EXPLICIT-TRACE/static-dispatch-schedule.json \
  --out profiles/rocprof/EXPLICIT-TRACE/dispatch-reconciliation.json
```

The static schedule is `ninfer_qwen3_8_27b_static_dispatch_schedule` schema v1. It binds the exact
trace workload, selected artifact/executable route, and hashed target sources, declares
`trace_start_ns_then_dispatch_id` order, and
enumerates every dispatch with a contiguous schedule index, exact ROCTX marker, layer identity,
symbol, grid/workgroup, and call ID/ordinal/within-call multiplicity. A modeled row additionally
supplies role, operation, canonical format, and all shape/format parameters. An unsupported or
unmodeled row instead supplies the explicit nonsemantic `unassigned` role,
`unmodeled_dispatch` operation, and a nonempty reason; it remains in coverage with its exact
duration and is never silently discarded. An unmarked dispatch is retained explicitly as
unmodeled and cannot receive a semantic role. Ambiguous simultaneous identical marked dispatches, source drift,
omissions, extras, reordered calls, or any launch-field mismatch fail before output.
The one no-clobber reconciliation JSON atomically contains all traced rows, the exact
modeled/unmodeled/unsupported count and duration partition, unprofiled P2048 performance authority,
and embedded roofline timing/inventory objects. This explicit expanded schedule is required because
same symbols and grids repeat across layers and cannot establish model ownership by themselves.
The producer supports the terminal C1 P2048 route for either dense attention or the fixed
`b128-s16-tau900` XAttention profile. It validates the source-defined 64-layer hybrid order
independently for every selected prefill chunk, reads each modeled matrix's shape and
Q4G64/W8G32/F8E4M3-row format from the exact selected artifact, and accepts only the matching
integer quantizer/CTA pair or FP8 quantizer/hipBLASLt/nonfinite-poison sequence in source call
order. The four-role profile therefore assigns FP8 only to its exact 144 projection objects. A
sparse full-attention layer requires one exact key-pack, rank, and selected G16/G32 flash-consumer
sequence; a dense layer requires its single group-matched dense consumer. MTP,
orchestration, runtime copies/fills, BF16 control projections, and other kernels remain explicit
`unmodeled` rows. The GDN input RMSNorm, four-tap SiLU convolution, ordinary FP32-state recurrence,
gated RMSNorm, and output residual are modeled only when their unique symbol and exact source launch
geometry occur once in the GDN layer marker. Gated RMSNorm counts represented BF16 input/gate/
weight/output traffic, source-expression reloads, scalar sumsq/mean/SiLU/output arithmetic, and
separate exponential/rsqrt arguments; it makes no matrix-instruction claim. Each post-mixer
RMSNorm, strided SiLU-multiply, and residual is admitted by the same rule. GDN control remains
unmodeled because the exact softplus branch count is data-dependent. MTP bulk and copy
work remains unmodeled because its one chunk marker does not identify the final-chunk and
autoregressive subcalls precisely enough to partition repeated symbols.

Missing ROCTX associations are retained with null markers and exact symbol-count
collision evidence, but never receive inferred ownership. An incomplete or reordered marked-layer
sequence is an evidence failure; the producer does not infer ownership from symbols or timestamps.
If those currently uncovered families become material, the minimal follow-up is a profiling-only
leaf marker carrying fixed semantic role plus layer, chunk index, and call ordinal around GDN
control and each MTP stem, attention, post-mixer, final-head, and AR-step call.
It must remain disabled outside the existing explicit profiler range and needs a newly built
profiling executable/evidence chain; frozen finalist binaries and current timing evidence are not
retrofitted.
The source list binds the implementation closure used by the producer: the producer and
reconciler, artifact directory reader/layout registry, ROCTX/topology/configuration, target
Program/variant/binder, and selected Q4/W8/dense dispatch predicates and kernels. Reconciliation
rehashes every member before accepting the schedule.

### FP8 gate/up post-measurement decision

`decide_fp8_gate_up.py` owns only the all-64-layer `text.mlp.gate_up` decision after the fixed
P2048 FP8/Q4 qualifier schema v2 exists. It requires explicit SHA-256 values for the qualifier, hybrid and
capacity analyses, benchmark, kernel trace, and marker trace. Unknown schemas, the wrong
`[2048,34816,5120]` shape, a non-R9700 profile, incomplete axis-sensitive represented-format
oracle/status/no-clobber flags, a non-7-pair raw timing set, inconsistent even-sample medians, a
changed trace/capacity/source/executable hash, or a changed 64-object inventory fails before
output. It does not accept a theoretical FP8 timing estimate. Capacity budgets are recomputed from
the capacity authority and gate/up Q4 service is reconstructed again from the bound raw trace;
copied hybrid summary values are cross-checks, not independent authorities.

The retained physical input passed this contract at 4.129716 ms FP8 versus 6.868267 ms Q4
(1.663133x faster). Reproduce the retained decision with:

```bash
python3 -m tools.bench.decide_fp8_gate_up \
  --qualifier profiles/bench/r9700-fp8-vs-a8q4-gate-up-20260904.json \
  --qualifier-sha256 ba1434a773df7d7cf51ec41158ba7d1a2753f2ee589a945e150f4c606f3470ff \
  --hybrid profiles/bench/r9700-e4m3-q4-hybrid-selection-20260904.json \
  --hybrid-sha256 eae6e53c0dc9a7d9d38f92072124e0af70610d57b3d1f6c7ae526b6b1de6e04a \
  --capacity profiles/bench/r9700-e4m3-capacity-analysis-20260904.json \
  --capacity-sha256 18652c9355fa6e4b8321d7784214a5a349a7e2539dd0879be3337dc803b0b8b1 \
  --benchmark profiles/rocprof/diagnostic-p2048-production-pingpong-20260904/report.json \
  --benchmark-sha256 22589e68ed2926ae44b5a0e78f5a654164a61d3488b7b2e0047b2a7324b836f8 \
  --kernel-trace profiles/rocprof/diagnostic-p2048-production-pingpong-20260904/trace_kernel_trace.csv \
  --kernel-trace-sha256 29ba3816ba225053167a36bfd7e44903711c9d7cdb0768ba524ab7129e51ec82 \
  --marker-trace profiles/rocprof/diagnostic-p2048-production-pingpong-20260904/trace_marker_api_trace.csv \
  --marker-trace-sha256 089af1710a7baf5d8dfb29fc7a1d821db1b06ba719c470f7502bae6c3f4dbdb9 \
  --output profiles/bench/r9700-fp8-gate-up-decision-20260904.json
```

The decision replaces the fresh trace's measured Q4 gate/up service with the qualifier's measured
complete-path FP8/Q4 time ratio. It reports the exact 64-object resident-byte delta, the historical
P2048/P8192 capacity calculation for every G16/G32 C1..4 cell, and projected whole-P2048 time and
throughput. Its `proceed` verdict required a faster complete FP8 median, nonnegative historical
slack in all 16 cells, and an improving whole projection. That historical capacity calculation
predates the current N16 artifact, selected chunk, and exact ordinary physical-capacity contract;
only fresh selected-chunk physical capacity can admit the route.

### FP8 post-gate/up fixed-role decision

`decide_fp8_post_gate_up.py` owns the retained role-selection step after the gate/up decision. It
recomputed converter inventories and the then-used 16 planner capacity cells, then maximized
measured Q4 service under that envelope. The fixed role result is
`text.attention.query_key`, `text.attention.gate_value`, and `text.gdn.query_key`:
1,024,065,536 additional bytes and 87,724,946 ns of measured P2048 Q4 service. The two attention
roles share `[2048,7168,5120]`, so the set requires only that matched complete-path qualifier and
the `[2048,4096,5120]` GDN query/key qualifier. Its reported 2,004,481-byte tight-cell remainder is
invalid because its capacity input predates the current N16 artifact, selected chunk, and exact
ordinary physical-capacity contract; it is not current capacity evidence and admits no additional
role. Fresh selected-chunk physical capacity owns admission.

Each pending input uses schema `ninfer.r9700.fp8_projection_qualification.v1`, version 1, with its
fixed qualification ID, shape, live source/executable hashes, R9700/gfx1201/auto identity, the
nine-point axis oracle, seven balanced timing pairs, raw 14-sample lists, algorithm/workspace
identity, correctness/status gates, and no-clobber proof. Once both physical reports exist, run:

```bash
python3 -m tools.bench.decide_fp8_post_gate_up \
  --gate-decision profiles/bench/r9700-fp8-gate-up-decision-20260904.json \
  --gate-decision-sha256 b6826119284e18966390ebc3a78423f218ef74b39ac1fdb6f163b65d7f08ae1f \
  --hybrid profiles/bench/r9700-e4m3-q4-hybrid-selection-20260904.json \
  --hybrid-sha256 eae6e53c0dc9a7d9d38f92072124e0af70610d57b3d1f6c7ae526b6b1de6e04a \
  --capacity profiles/bench/r9700-e4m3-capacity-analysis-20260904.json \
  --capacity-sha256 18652c9355fa6e4b8321d7784214a5a349a7e2539dd0879be3337dc803b0b8b1 \
  --attention-qk-gate-value profiles/bench/EXACT-FP8-ATTENTION-QK-GATE-VALUE.json \
  --attention-qk-gate-value-sha256 EXACT_SHA256 \
  --gdn-query-key profiles/bench/EXACT-FP8-GDN-QUERY-KEY.json \
  --gdn-query-key-sha256 EXACT_SHA256 \
  --output profiles/bench/r9700-fp8-post-gate-up-decision-20260904.json
```

The tool emits nothing until both hash-bound physical inputs validate. A `proceed` verdict requires
both complete FP8 paths to beat Q4, its historical capacity calculation to remain nonnegative, and
the projection to improve on the gate/up projection. The capacity clause is now superseded by the
exact feature-materialization correction. The result remains speed/design evidence until a selected
hybrid artifact is converted and measured at whole-inference scope.

The roofline tool does not infer a model role, shape, weight format, or inner dimension from a
kernel symbol or launch grid:

```bash
python3 -m tools.bench.model_qwen3_8_27b_roofline \
  --dispatch-reconciliation profiles/rocprof/EXPLICIT-TRACE/dispatch-reconciliation.json \
  --memory-probe profiles/rocprof/EXPLICIT-TRACE/memory-probe.json \
  --out-json profiles/rocprof/EXPLICIT-TRACE/logical-roofline.json
```

The reconciliation is the only required dispatch input. The roofline consumer reopens and rehashes
it, its validated trace, static schedule and schedule sources, and the linked low-context evaluation
and unprofiled P2048 report. It verifies that the trace-selected artifact, executable, G16/G32
group, dense profile, and selected prefill chunk equal the reconciled workload. Every validated
trace dispatch must occur in the exact retained order as `modeled`, `unmodeled`, or `unsupported`;
the latter two require a reason and remain in `uncovered_dispatches` with exact count and duration.
The embedded timing and inventory arrays must equal precisely the modeled subset and agree with
the trace coverage row at symbol, marker, interval, operation, role, format, and parameters. A
missing dispatch, silent subset, invalid coverage partition, changed authority, or self-consistent
but differently bound embedded row fails before output.

Supported operation/format pairs are `q4_linear/a8q4g64_bf16_output`,
`w8_linear/a8w8g32_bf16_output`, `q4_quantize/bf16_to_a8g64`,
`w8_quantize/bf16_to_a8g32`,
`dense_attention/bf16q_fp8k_int4v_fp16scale_fp32_output`,
`sparse_pack/fp8k_to_bf16_packed_k`, `sparse_rank/bf16q_bf16packedk_fp32_rank`,
`sparse_consumer/bf16q_bf16packedk_int4v_fp16scale_fp32_output`,
`gdn_recurrence/bf16_io_fp32_state`, `gdn_control/bf16_input_fp32_output`,
`gated_rmsnorm/bf16_io`, and the other BF16-I/O GDN-convolution and elementwise operations. The
operation-specific parameter errors name every missing dimension. GDN recurrence additionally
records the executed row-tile multiplicity; gated RMSNorm reports countable FP32 scalar/VALU
arithmetic and SFU arguments separately and never assigns WMMA work;
sparse rank records its logical pair count, exact number of issued B16 WMMA tiles, tensor element
counts, and scratch bytes rather than asking the tool to guess estimator traversal.

The current selected-P2048 trace/reconciliation contract is deliberately dense; a sparse consumer
cannot be modeled without a future reconciliation schema that atomically binds its retained-key
telemetry. The optional memory probe is
`ninfer_r9700_memory_probe` schema v1 with positive `read_gbps`, `write_gbps`, and `copy_gbps`.
The tool binds its exact supplied bytes but has no shared session identifier with the timing
authority, so it labels the result as a supplied reference rather than claiming same-session
provenance. Record and inspect acquisition provenance separately before interpreting it as a
matched-session control.

The output separates semantic logical operations, actual padded WMMA/IU4/IU8 operations,
represented-minimum bytes, and modeled source-request bytes. Effective TOPS, logical GB/s,
arithmetic intensity, and theoretical issued-operation peak fractions use independent device
service time. Those are profiler attribution rates, never selection or end-to-end throughput.
Aggregation divides summed work by summed dispatch duration; a separate interval union preserves
overlapping-stream wall time. `unprofiled_whole_p2048` separately retains the validated low-context
3/1 authority's P2048 prefill seconds and tok/s and cannot be derived from profiler intervals. The
advertised 640-GB/s ratio and optional supplied
copy-probe ratio are explicitly logical reference ratios. They are not physical HBM utilization.
Physical HBM bytes/bandwidth/peak fraction and stall fraction remain `null`; GL2/TCP hit/activity
evidence remains a separate profiler result.

After the atomic reconciliation, roofline report, and profile-standard PMC authority exist, assemble
the selected-P2048 operation evidence with one explicit static report for every executed
stage/symbol family:

```bash
python3 -m tools.bench.assemble_selected_p2048_evidence \
  --dispatch-reconciliation profiles/rocprof/EXPLICIT-TRACE/dispatch-reconciliation.json \
  --roofline profiles/rocprof/EXPLICIT-TRACE/logical-roofline.json \
  --pmc profiles/rocprof/EXPLICIT-PMC/validated-pmc.json \
  --static-report profiles/rocprof/EXPLICIT-STATIC/q4-linear.json \
  --static-report profiles/rocprof/EXPLICIT-STATIC/dense-attention.json \
  --static-report profiles/rocprof/EXPLICIT-STATIC/gdn.json \
  --static-report profiles/rocprof/EXPLICIT-STATIC/orchestration.json \
  --out profiles/rocprof/EXPLICIT-TRACE/selected-p2048-evidence.json
```

Revalidate the retained final record with the same explicit reconciliation, roofline, PMC, and
ordered `--static-report` arguments by replacing `--out ...` with:

```bash
  --validate profiles/rocprof/EXPLICIT-TRACE/selected-p2048-evidence.json
```

Validation reconstructs the entire record from the reopened inputs and requires exact structural
equality; it does not trust copied summaries inside the final JSON.

Each static input must be `ninfer_r9700_operation_static_evidence` schema v1. Existing checker
console text is not this authority and is rejected. Produce each report from an explicit selected
code object, its disassembly/metadata, and exact sources; this scalar/VALU example intentionally
does not claim WMMA:

```bash
python3 -m tools.bench.produce_operation_static_evidence \
  --artifact /absolute/path/to/selected.ninfer \
  --executable /absolute/path/to/selected/ninfer_bench \
  --code-object /absolute/path/to/selected-code-object.co \
  --assembly /absolute/path/to/selected-disassembly.s \
  --metadata /absolute/path/to/selected-metadata.s \
  --source src/ops/r9700/gdn/gdn_ops.hip \
  --stage 'ninfer.gdn.prefill.gdn payload=0' \
  --dispatch-symbol 'EXACT_ROCPROF_DISPLAY_SYMBOL' \
  --code-symbol EXACT_MANGLED_CODE_OBJECT_SYMBOL \
  --operation-family gdn_recurrence --specialization gdn-selected-gfx1201 \
  --hardware-class scalar_valu --arithmetic fp32_valu \
  --opcode v_fma_f32=EXACT_POSITIVE_COUNT \
  --max-lds-bytes EXACT_ADMITTED_CEILING --max-vgpr-count EXACT_ADMITTED_CEILING \
  --memory-residency register_reuse \
  --memory-access 'exact source-derived access classification' \
  --overlap not_applicable --memory-evidence 'exact source/disassembly rationale' \
  --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" --kv-value-group "$NINFER_SELECTED_GROUP" \
  --xattention-profile dense --out profiles/rocprof/EXPLICIT-STATIC/gdn.json
```

Repeat `--opcode NAME=COUNT` and `--source PATH` as needed. Repeat `--stage EXACT_ROCTX_MARKER`
for every exact layer/chunk marker that executes this same dispatch display symbol and code
specialization; the report owns only those enumerated signatures. The code object must be the exact
nonempty gfx1201 offload object extracted from the selected executable; the producer and assembler
both require that byte sequence to occur exactly once in the executable. Use `--unmarked` instead
of `--stage` only for an explicitly unmodeled null-marker dispatch. `proven_dependency_safe`
overlap is currently admitted only for the promoted Q4 CTA and requires `--overlap-evidence` to
name the exact assembly also passed to `--assembly`; its delegated checker proves the
successor-group request, compute window, wait, and publication ordering. Other routes remain
`unproven` until an equally semantic checker is added. Q4/W8 promoted CTA
reports additionally pass `--cta-recipe q4` or `--cta-recipe w8`, their exact selected mangled
symbol/opcode count and fixed admitted LDS/VGPR ceilings. That mode delegates to
`check_prefill_cta_static.py` in challenger mode, so the two ordered signal/wait pairs, no
monolithic barrier, no `global_inv`, native IU4/IU8 opcode count, and zero scratch are all required.
The retained incumbent may still be inspected with the older checker's
`incumbent-diagnostic` mode, but it cannot produce terminal static evidence.

Extract the executable's embedded HIP fatbin only through the non-mutating owner before producing
static evidence:

```bash
python3 -m tools.bench.extract_embedded_code_object \
  --executable /absolute/path/to/selected/ninfer_bench \
  --out /absolute/new/path/selected-ninfer-bench.hip_fatbin
```

The extractor gives `llvm-objcopy` an explicit, distinct temporary output ELF; rejects symlink or
hard-link aliases among the input, dumped section, and discarded output; requires the extracted
bytes to occur exactly once in the executable; and publishes the fatbin without overwriting an
existing path. Symbol-selected extraction additionally requires one non-overlapping bundle member
whose exact offload target token is `gfx1201`; substring targets such as `gfx12010` do not match.
Its before/after check binds the source device, inode, mode, size, modification
time, and SHA-256, so an identical-byte path replacement is not accepted as an unchanged input.
Never run `llvm-objcopy --dump-section SECTION=FILE INPUT_ELF` without a distinct final ELF operand:
with no output ELF, `llvm-objcopy` rewrites `INPUT_ELF` in place even when the apparent intent is
only to dump a section.

MTP whole-route reports retain exact target-token parity, speculative counters, and measured
acceptance/throughput as regression evidence. DFlash is the only speculative backend with new
optimization and admission work; no standalone MTP shortlist-head trace or precision branch is
scheduled or accepted as terminal evidence.

The hardware-use evidence assembler rehashes every nested file snapshot again after all reads,
refuses an existing or dangling output path, and assigns every reconciled dispatch to exactly one
static report. Modeled, unmodeled, and unsupported dispatches all require intended-hardware
ownership; extras, ambiguity, and silent subsets fail. Profile-standard PMC evidence retains each
captured profiler display symbol and joins it through the exact same-capture ROCTX stage plus the selected trace's equal
`{stage,symbol}` multiplicity. A kernel-regex-filtered capture attributes counters only to the
captured operation/specialization set, not every operation sharing its stage. PMC dispatch IDs are
intentionally not joined to the separate auto trace. The output preserves auto profiler durations and roofline rates
as attribution-only evidence, while P2048 tok/s/seconds come only from the validated unprofiled 3/1
authority. Physical HBM bytes/bandwidth and hardware stall fields remain null; GL2/TCP are retained
only as relative stage ratios.

After the trace names a specific production kernel, inspect only that owner's retained ISA/resource
target: `make -C tools/r9700 q4g64-linear-isa q4g64-linear-resources` for Q4 Linear,
`make -C tools/r9700 isa kv-resources` for typed KV attention, `make -C tools/r9700 gdn-isa
gdn-recurrence-resources` for GDN, or `make -C tools/r9700 sampling-isa rope-resources` for the
named sampling/RoPE boundary. Do not run this list speculatively or use unavailable VALU/LDS/cache
counters.

Each raw report must be `ninfer_bench_report` schema v20. Its config records concurrency, the
compiled G16/G32 KV value group, exact K/V/V-scale plane layouts, compile-selected Q4 and W8
activation profiles, and the exact
FP8-Q/K WMMA classifier (`T1 >= 64`, `T2 >= 320`, `T >= 3` streaming). The W8 field names the
binary profile: an A8 profile still retains its measured represented-BF16 crossover and does not
claim that every W8 invocation quantizes. Resume validates these fields, every expected test row,
positive finite timings/throughput, internally coherent acceptance counters, and the complete point
command. It also records either the strict dense identity or the private B128/S16/tau900
XAttention identity; the validator rejects missing, stale, or mismatched sparse fields. The
schema-v14 manifest retains artifact and benchmark-executable path, byte size, identity
where applicable, and SHA-256; both are hashed again after the run. Flattened rows repeat the
artifact identity and SHA-256 beside every speed cell. A report
from another artifact, build, or matrix point cannot be silently reused;
`--expected-kv-value-group` additionally pins the intended compiled cache group. A8 is required by
default for Q4; `--expected-q4-activation-bits 4` is reserved for an explicit A4 evaluator binary.
The default 8 identifies the selected adaptive A8 profile; use
`--expected-w8-activation-bits 16` only with the represented-BF16 W8 control. The flattened summary and schema-v14 matrix
manifest carry native names from the report: selected target,
canonical `weights_id`, artifact,
GPU architecture and HIP runtime/driver identity, load/read/upload/staging values, Engine memory
arenas including request transient and Device Graph
allowance/observed allocation, per-test planned logical and allocator-observed workspace peaks, KV capacity and
payload plus automatic-sizing resolution/headroom and its binding constraint, configured proposal head and graph mode,
phase timings and throughput, and speculative
rounds/drafts/acceptance/fallbacks. The `pareto` preset is deliberately small and matched: one case
contains 8K/32K MTP3 prefill and one contains 8K/32K MTP3 graph decode with 256 requested decode
tokens at each context. Repeat `--concurrency` from 1 through 4 and run it separately for the
all-Q4, mixed Q4/W8, and authority-bound four-role FP8/Q4 artifacts.

The automatic resolver is bounded by `max_context * concurrency`. Consequently,
`pareto-feasibility` and `dflash-feasibility` retain the resolved usable capacity for the required
32K workload but classify it as `logical_context_ceiling`, `device_memory`, or both. The flattened
`memory_limited_capacity_tokens` is null when spare memory could hold another KV page but the
logical ceiling prevented allocation. Such a row proves feasibility only and is forbidden as a
resolved effective Pareto capacity value. At C=1 the addressable curve has no adjacent page point, so the
memory-limited capacity is also null. Do not extrapolate a reported byte stride beyond the target's
addressable capacity curve.

The corresponding `pareto-capacity` and `dflash-capacity` presets use the model-native 262,144
logical-token ceiling. A result below `C*262144` must pass the next-page memory infeasibility check
and is labeled `device_memory`; a result equal to the curve maximum is labeled `model_context`.
Both are exact `resolved_effective_maximum` measurements; the flattened
`resolved_effective_maximum_tokens` is the capacity objective accepted by the Pareto classifier.
Every Pareto campaign requires one explicit chunk from the retained schema-v2 global selection;
the schema-v14 manifest and each raw command bind that same value. Capacity validation also
requires the runtime's nonzero Device Graph allowance and its measured allocation, rejects an
observed allocation above the plan, and retains both byte values in the flattened row.

The `dflash-pareto` preset is separate because DFlash proposal quality is acceptance, not target
PPL. It requires an explicit startup-fixed K and records both the requested and resolved verify W.
Physical DFlash presets accept only the registered companion for the terminally selected base:
`qwen3.8-27b/r9700-q4g64-n16k16-dflash2-q4-eval` or
`qwen3.8-27b/r9700-q4-w8-mse-n16k16-dflash2-q4-eval`, or
`qwen3.8-27b/r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval`. The third companion is intentionally
absent until schema-v7 selects its authority-bound hybrid base; it may not reuse either existing
companion and requires a fresh conversion plus DFlash K/W quality, C=1..4 capacity, and whole
campaign. Dry runs remain path-only so future commands can be inspected before the artifact exists.
For each fixed C=1..4 it retains matched 8K/32K DFlash prefill and graph-decode throughput,
per-position acceptance, and an ordinary greedy decode control over identical prompts. It also
retains fresh-prompt DFlash whole inference and a separate exact spec-none ordinary `--whole-pg`
control. Both controls use `--draft-tokens 0`, with no `--spec` or draft-head flag. Whole-route
parity retains the complete generation including its first output token; isolated decode parity
retains all 257 post-seed outputs and inherits seed equality from the deterministic whole route.
The two DFlash Pareto `-pg` cases alone carry `--isolate-prompt-decode`, so C1 follows the same
one-token seed/discard protocol as C2..4; ordinary benchmark `-pg`, shortlist, and whole semantics
are unchanged. A four-role hybrid campaign additionally queries and binds the compile-matched planner inventory for every
actual resolved DFlash verification width: MTP-width coverage is not accepted as DFlash workspace
evidence. The planner authority additionally binds the same build root, `CMakeCache.txt`, and
`compile_commands.json` identities as the benchmark executable. Schema-v20
JSON can opt into per-repetition/per-lane token IDs; the matrix rejects the campaign unless every
ordinary and DFlash greedy output is exactly identical. Two isolated eager C=1 probes synchronize
proposal logits and selector outcomes into raw counters for top-16/64/256, tree/head membership,
first rejects, depth, and finite-logit health. Each also retains the ordered per-round proposal
IDs, parent topology, and aligned licensed target tokens. The runner binds both diagnostics to the
exact artifact bytes, benchmark executable bytes, report hash, activation/attention profile,
K/W/head, and concurrency, then requires exact proposal traces and final target outputs across the
two executions. `dflash-generated-quality.json` binds those diagnostics and the C=1..4 ordinary
output parity artifact into one generated-behavior gate. It records target-model NLL as not
applicable: DFlash changes proposal execution rather than the teacher-forced target distribution,
whose BF16-source comparison remains owned by the base artifact PPL campaign. Diagnostic timing is
explicitly ineligible for performance comparison.

`assemble_dflash_selection.py` reopens the schema-v20 reports and schema-v14 manifests. Capacity
inputs must cover the complete shortlist frontier, including exact retained failure provenance for
excluded K/W profiles; `dflash-pareto` inputs must cover exactly the profiles with complete C=1..4
effective maxima. Every eligible profile must retain all 22 phase/whole/control/diagnostic points,
exact C=1..4 ordinary-output parity, exact repeated proposal and licensed-target traces, and the
bound generated-quality gate. Capacity admission and generated-quality/acceptance remain separate
from speed admission. For every capacity-eligible K/W, the assembler matches its DFlash and exact
spec-none ordinary whole/decode rows by prompt, generated length, and C. It requires at least a
`1.02x` raw-mean speedup and a strictly positive two-standard-deviation conservative speedup in
every 8K/32K C=1..4 whole and decode cell; equality, noise-overlap, or one materially regressed cell
rejects that K/W from promotion. The schema-v3 output preserves the
speed-eligible objective frontier and chooses the static K/W by maximizing the worst normalized
whole-inference throughput over all 8K/32K and C=1..4 cells, then worst normalized capacity over
C=1..4, then worst normalized acceptance length over the matched decode cells. A complete numerical
tie selects the first numeric `(K, W)` tuple. No workload weights are introduced: exact quality is
mandatory, positive matched ordinary speed is mandatory, and acceptance breaks ties only after
end-to-end throughput and supported capacity.

The `concurrency` preset is a phase decomposition, not a whole-request latency measurement. Its
prefill cases submit `C` Engine requests concurrently. Pure and context decode cases exclude their
seed prefills and report aggregate batched-decode throughput. Use the serving benchmark below for
the end-to-end request makespan and request throughput claim.

`run_serve_corpus.py` runs the selected target and published MTP0/MTP3 suites. Pass the sole
Qwen3.8 R9700 `--artifact` and `--mode mtp0` or
`--mode mtp3` to run only that suite. `--mode dflash7` runs the same decode corpus with DFlash
block=8 (`k=7`) and the optimized proposal head on Qwen3.8-27B DFlash2. Add
`--sampling greedy` to force exact argmax while retaining the same fixtures and repetition count.
Its schema-v6 result and flattened summaries retain the canonical `weights_id` received from the
schema-v20 serving startup record. The stochastic route pins its complete
temperature/top-p/top-k/min-p/presence/frequency profile explicitly, so model-default changes do
not alter the measurement method.

The serving-corpus runner has no chunk default: pass the value from the validated schema-v2
selection explicitly. Schema-v6 request records retain it, and resume rejects records produced
with another chunk. This corpus-quality workflow explicitly launches and validates
`--max-concurrency 1`; concurrency evidence belongs to the separate C=1..4 runner below.

```bash
NINFER_SELECTED_BUILD=/path/to/selected-build
NINFER_SELECTED_ARTIFACT=/path/to/selected.ninfer
NINFER_SELECTED_PREFILL_CHUNK=1024 # replace with the validated selected value
NINFER_SELECTED_GROUP=16           # replace with the validated selected value
NINFER_SELECTED_ATTENTION=dense    # dense or b128-s16-tau900
python3 tools/bench/run_serve_corpus.py \
  --serve "$NINFER_SELECTED_BUILD/apps/ninfer-serve" \
  --artifact "$NINFER_SELECTED_ARTIFACT" \
  --mode mtp3 --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
  --expected-kv-value-group "$NINFER_SELECTED_GROUP" \
  --expected-xattention-profile "$NINFER_SELECTED_ATTENTION" \
  --output profiles/bench/serve-corpus-selected
```

## Concurrent serving benchmark

`run_serve_concurrency.py` measures two separate concurrency properties through real loopback
Chat Completions requests:

- `decode-saturation` submits one long-decode wave and uses only complete one-second intervals in
  which every decode round has exactly the configured batch size. Ramp-up, prefill, and drain
  intervals are excluded.
- `corpus-makespan` shuffles the existing mode-specific corpus once with the fixed seed `20260811`,
  then runs that same order with exactly `N` persistent client workers. A worker submits the next
  request only after its current response completes, and makespan ends when the final response has
  been read. Request bodies are sent in shuffled-order sequence while response waits remain fully
  concurrent, removing client-thread arrival races without serializing inference.

Each concurrency point starts a fresh server because its execution graphs and memory plan are
startup-fixed. Prefix reuse is disabled, startup and warmup are outside both measurements, and the
runner writes per-point JSON, raw serving JSONL, and combined JSON/CSV/Markdown summaries.
The point report records the shuffle seed, dispatch method, shuffled position, and canonical corpus
position for every request.

```bash
NINFER_SELECTED_PREFILL_CHUNK=1024 # replace with the validated selected value
NINFER_SELECTED_GROUP=16           # replace with the validated selected value
NINFER_SELECTED_ATTENTION=dense    # dense or b128-s16-tau900
python3 tools/bench/run_serve_concurrency.py \
  --artifact /path/to/qwen3_8_27b_r9700.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
  --expected-kv-value-group "$NINFER_SELECTED_GROUP" \
  --expected-xattention-profile "$NINFER_SELECTED_ATTENTION" \
  --decode-tokens 8192 \
  --output profiles/bench/concurrent-decode

python3 tools/bench/run_serve_concurrency.py \
  --artifact /path/to/qwen3_8_27b_r9700.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
  --expected-kv-value-group "$NINFER_SELECTED_GROUP" \
  --expected-xattention-profile "$NINFER_SELECTED_ATTENTION" \
  --output profiles/bench/concurrent-corpus
```

Dry runs accept a future explicit artifact path and print every server command without opening the
artifact or starting a process. Use `--kv-capacity auto` when the fixed corpus needs more shared KV than the default 262,144-token
pool. A point is intentionally not resumable: combining fragments from separate server processes
would not preserve either a steady interval or one continuous makespan.
For post-selection evidence, derive `NINFER_SELECTED_PREFILL_CHUNK` from the retained schema-v7
selection as in the NIAH workflow below; do not rely on this generic runner's 4096 default.

## Speed suite (live-serve scenario matrix)

`run_speed_suite.py` is the **fast-iteration** tool: it hits the *already-running* serve
(default `http://127.0.0.1:8081`) — no engine spawn, restart, or reconfiguration — and reports,
per scenario, prefill tok/s, decode tok/s, time-to-first-token, end-to-end wall, MTP draft
accept, and an output hash. Unlike the offline `ninfer_bench` matrix above, it measures the
production serve path and is the right instrument for watching a branch while you edit kernels.

Scenario matrix lives in [`speed_suite_cases.json`](speed_suite_cases.json); fixtures resolve
relative to the repo root (NIAH / chat-history from `examples/cli/messages/`, the OWUI capture and
room-puzzle prompt from `tools/bench/fixtures/`). Add your own realistic prompts there as inline
`prompt` text or committed JSON/`prompt_file` fixtures. Prefill is reported two ways:
`prefill_tok_s` (overall average: tokens / wall prefill time) and `prefill_tail_tok_s`
(steady-state rate over the trailing ≤1 s of prefill — the number your kernel work moves; for
long-context prefill the tail is slower than the average because attention cost grows with
depth). Both come from the server's `usage.prompt_tokens_details` block; the tail requires the
current engine build. Prefill rates there cover the computed (non-reused) suffix only — a cached
prefix is reported as `cached_tokens`, not counted into the rate.

```bash
# Full base set (small ± thinking, multi-turn, tools, 8k/64k NIAH, decode). ~2 min.
python3 tools/bench/run_speed_suite.py --label LABEL --runs 2 --warmup 1

# Fast subset while iterating:  --suite smoke,small,multiturn
# Long-context 128k point:      --include-slow
# Production prefix-reuse-on:   --no-cache-bust   (on by default so identical prompts refetch)
# Server-side TTFT only:        --no-stream
```

Outputs land under `profiles/bench/speed-suite/` (gitignored): a per-run JSON
(`speed-suite-LABEL.json`, per-case aggregates) and a publishable one-row-per-case CSV
(`speed-suite-LABEL.csv`) for the progress doc / cross-branch diff. Metrics read the server's
`usage.prompt_tokens_details` block, so no extra instrumentation is needed.

**Trials:** `--runs N` repeats each case N times *in one session* (mean/median/min/max reported).
For *independent* trials — cold vs warm start, thermal drift, minutes apart, or a different
branch/build — run the script N times with N `--label`s and aggregate the per-label JSONs; the
two are different claims, so keep them separate.

No cross-hardware baseline is admitted for the R9700 product. The 64k NIAH case remains in the base
set because long-context prefill behavior can be invisible at 8k; publish only results from the
selected R9700 artifact and current fixed cache profile.

### NIAH: position matrix (lengths x positions)

The NIAH (needle-in-a-haystack) fixtures form a **position matrix**: 6 lengths x 5 needle
positions = 30 cells, deterministic and pure, generated by
[`make_niah_positions.py`](make_niah_positions.py). A cell is a pure function of
`(length, position)` (re-running reproduces the identical fixture byte-for-byte, so committed
fixtures are reproducible). Every length is a prefix slice of one shared **master stream** —
the needle-free 256k master document (one continuous essay, no per-cell content drift) — with
a single `OFFICIAL RECORD` line spliced in as a standalone line at the requested depth.

- **Lengths**: 8k / 64k / 100k / 128k / 150k / 200k. **200k (≈200k tokens) is the new long
  end** — it fits the 262k context, unlike 256k (≈277k tokens) which overflows it. So **256k is
  the master stream** (a content source), not a ladder rung at the 262k context.
- **Positions**: start / q25 / mid / q75 / end (needle depth 0 / 0.25 / 0.5 / 0.75 / 1). The
  committed cells cover the full 8k and 200k ladders plus the mid spine for the middle lengths;
  the rest of the 30-cell matrix is on-demand through the generator.
- **Non-leaking question (defect fix)**: the question states the answer *form* (`ORCHID=<code>;
  COLOR=<color>`) but NOT the values, so a correct answer requires actually retrieving the
  needle. The old question echoed the exact answer, which made every NIAH gate vacuous (a model
  that ignored the haystack still passed); that is fixed in every committed fixture.

```bash
# Generate a cell / a whole ladder (deterministic, writes into examples/cli/messages/)
python3 tools/bench/make_niah_positions.py --length 200k
python3 tools/bench/make_niah_positions.py --length 8k --position all
# Optional broader NIAH coverage over three length boundaries.
python3 tools/bench/run_niah_check.py --lengths 8k,64k,200k --positions start,q25,mid,q75,end
```

Durable evidence binds the completed schema-v7 terminal artifact/cache/execution selection, exact schema-v20
server-start record, artifact bytes, and server executable bytes. Run this gate only after
capacity/whole evidence has selected the fixed cache and execution profile; NIAH is a
post-selection admission gate, not an input to that selection. Build `ninfer-serve` in the winning
candidate's exact matrix build and use a group/profile-qualified output directory. The binder
rejects a server whose compiled G16/G32 value group or dense/B128-S16-tau900 attention profile
differs from the selected candidate. It also rejects any server-start record other than the exact
262,144-token context, explicit 262,144-token KV capacity, one-request concurrency, the
schema-v7-selected prefill chunk, and disabled-prefix-reuse envelope below. Disabling reuse ensures shared document
prefixes cannot turn later cells into suffix prefills.

```bash
# This fails before creating a campaign until the terminal schema-v7 authority exists.
bash profiles/bench/post-terminal-niah-prepare-20260905/prepare.sh

# Future serialized physical-GPU admission run. It starts and stops the exact selected server.
bash profiles/bench/post-terminal-niah-20260905/commands.sh
```

The preparer resolves the terminal winner rather than mapping filenames: it reopens both selected
schema-v14 matrices and binds their artifact, benchmark/build, cache group, attention profile,
selected chunk, and hybrid width planner when applicable. The generated command builds only
`ninfer-serve` with parallelism four, uses one request lane, and publishes a separate admission
record only after all five responses and their fresh-prefill request-log records validate.

**Gate semantics (read before interpreting matrix results).** The default check is the
*exact-format* needle `ORCHID=493817; COLOR=COBALT` — it passes only if the model both
retrieves the needle AND emits the exact answer form. Two levers: `run_niah_check.py
--needle 493817` is a *recall-only* gate (the value must appear in the answer, format
ignored — the pure recall signal), and `--runs N` repeats each cell for a per-cell pass
rate (e.g. 5/5). The console prints a `[FAIL] <case>: n/N runs failed to retrieve the
needle` line per failing case plus a total failure count on the final verdict; per-cell
retrieved/total and answer snippets are also in the JSON record. The default gate is the
light **mid-spine** (8k + 64k, fast); the matrix runs on demand.

The checked NIAH cases are behavioral gates, not numerical attention oracles. The 8k and 200k
ladders and the complete 6-length x 5-position matrix are optional broader coverage for the current
XAttention admission. Run the full matrix as a post-change regression after a production attention
change and preserve the per-cell outputs with the corresponding hardware, artifact, and
server-build provenance.
