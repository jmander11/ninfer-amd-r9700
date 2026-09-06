# R9700 perplexity and Pareto-quality gate

This campaign measures the fixed Qwen3.8-27B FP8-K/INT4-V product against an
independent BF16 reference. It does not expose a cache-format or attention-mode
runtime switch. Before the persistent ABI is frozen, G16 and G32 are evaluated
as separately configured executables over the same explicit R9700 candidate
weight artifact; the losing executable profile and codec branch are deleted
after selection.

`ninfer-ppl` computes teacher-forced next-token NLL:

```
mean_nll = sum_t -log p(token[t + 1] | prefix) / tokens_scored
ppl      = exp(mean_nll)
```

The two schedules exercise different production shapes:

| schedule | execution |
|---|---|
| `prefill` | chunked full-sequence prompt path |
| `decode` | prefill the skipped prefix, then teacher-force the suffix at T=1 |

The default `--skip half` excludes the first half from the mean while retaining
those tokens in KV. `--skip 0` scores every next-token position. Device Graphs
are enabled by default. At both 8K and 32K the extras pair Device Graph with
eager execution and MTP with ordinary decode. The 8K controls also compare the
default MTP draft window with `k=4`, start decode mid-page, and cover a short
context.

Each cell emits JSON plus two index-aligned binary sidecars:

- `.nllf32`: little-endian float32 NLL for every scored position.
- `.argmaxi32`: little-endian signed int32 greedy prediction at each position.

Quality eligibility requires finite, complete, exactly aligned sidecars and an explicitly named
tier; there is no silent default. The `accuracy` tier caps paired mean-NLL delta at 0.02 and new
severe membership at `max(4, ceil(0.001 * scored_positions))`. The `capacity-speed` tier caps
mean-NLL delta at `ln(1.05) = 0.048790164` and new severe membership at
`ceil(0.0025 * scored_positions)`. Severe means NLL at least 10. This position-aware guardrail
cannot be replaced by aggregate count: repairing one BF16-severe position does not hide a new
severe position elsewhere. Sidecar lengths must match exactly. Prefill/decode compares two
qualified schedules that can legitimately select different private attention precision. It requires
finite, position-aligned sidecars and a measured maximum per-token NLL bound supplied explicitly as
`--schedule-parity-max-abs-nll`; greedy flips are reported but do not gate this comparison. There is
no universal default inferred from a short probe. Graph/eager, MTP/ordinary, and draft-window pairs default to
exact NLL equality; an intentional relaxation must be passed explicitly with
`--execution-parity-max-abs-nll`. Exact greedy-token equality remains mandatory for
those same-route execution pairs. The report also records paired delta standard
error, absolute-error percentiles, and the ten worst signed token deltas.
Each `worst_delta_tokens.position` is the zero-based index in the cell's aligned scored-position
sidecars; add the enclosing cell's `skip_tokens` to recover the corresponding scored prompt
position. `delta_nll` retains the candidate-minus-BF16 sign, while `abs_delta_nll` determines the
largest-magnitude ordering.

Exact-token comparison has two distinct roles. Each complete weight-and-cache
candidate is compared directly with the independent BF16-source authority; its
greedy mismatch count and rate are quality diagnostics, not admission gates. The fixed lossy FP8
E4M3FN-key/INT4-value cache alone changes 93/94 of 4,095 BF16 greedy positions for G16/G32, so zero
BF16 flips cannot be a coherent product requirement. Execution variants of one candidate (graph/eager,
MTP/ordinary, and draft-window controls) are compared directly with that
candidate's primary sidecars and must also be token-exact. G16 and G32 have no
separate candidate-to-candidate quality gate: they consume byte-identical weights and each is
numerically gated against BF16. A direct G16/G32 comparison remains useful diagnostic evidence.

Here “greedy flip” means disagreement with the BF16 scorer's argmax, not disagreement with the
corpus's teacher-forced next token. Corpus-token quality is represented by NLL/PPL. Flip rate is
retained for diagnosis and regression attribution, but it neither admits nor rejects a lossy
candidate.

After schema-v7 selects one evaluation winner, prepare its separate exact same-route token gate:

```bash
bash profiles/ppl/post-terminal-exact-token-prepare-20260905/prepare.sh
```

That CPU-only step resolves and binds the winner's artifact, G16/G32 scorer, dense or
B128/S16/tau900 profile, selected chunk, candidate-local quality authority, and validated 18-shard
BF16 source. It emits one future C1 command under
`profiles/ppl/post-terminal-selected-exact-token-20260905`. The physical campaign covers only 8K
and 32K decode: graph/eager and MTP3/ordinary must have exact I32 greedy tokens and zero maximum
absolute NLL delta at both lengths, and the existing MTP3/MTP4 control must do so at 8K. Its final
admission JSON is atomically published only after recomputing those five comparisons from the raw
sidecars. Candidate-to-BF16 greedy flips remain diagnostic; no zero-flip BF16 threshold is added.

## Required artifacts and executables

The reference scorer is the checkpoint-direct BF16 mathematical/model authority
at `tools/reference/qwen3_8_27b_bf16/ppl.py`. It opens the complete original
18-shard safetensors directory and never uses `.ninfer`, packed W8 arithmetic,
or the product Engine. The G16 and G32 inputs use the same explicit candidate
artifact because the cache is runtime state, not artifact payload. `run.py`
requires the two paths to be the same file or byte-identical files with the same
NInfer identity, hashes that candidate once, and rejects per-cell artifact path
or identity drift. Each product scorer reports its compiled `kv_value_group`, exact K/V/V-scale
plane layouts, Q4 and W8 activation profiles, and the enabled FP8-Q/K classifier with its T1/T2
minimum contexts. `run.py` rejects a scorer whose report disagrees with its requested
cache/plane-layout/activation/attention profile. The W8 value
describes the compile-time evaluator profile; an A8 profile retains its measured represented-BF16
crossover for smaller shapes rather than quantizing every invocation. The bound evaluator identities
are `r9700-int-candidate`, `r9700-w8g32-mse-eval`,
`r9700-q4g64-n16k16-eval`, `r9700-q4-w8-n16k16-eval`, `r9700-q4-w8-mse-n16k16-eval`,
`r9700-w8-bf16-embed-eval`,
`r9700-w8-bf16-attn-qk-eval`, `r9700-w8-bf16-attn-vo-eval`, and
`r9700-w8-bf16-gdn-qk-eval`. Each campaign still accepts exactly one explicit candidate artifact
for its selected cache-profile binary, preventing converter plans or identity drift from being
mistaken for a runnable comparison.
The same-format mixed calibration control keeps its result under
`profiles/ppl/q4-w8-mse-real/<profile>.json` with the adjacent complete NLL and argmax sidecars.
Its `r9700-q4-w8-mse-n16k16-eval` identity must be preserved in every cell and its completed 8K row is
added to `profiles/ppl/8k-candidate-comparison.json` and the paired Markdown report rather than
left as terminal-only output. Those reports retain PPL, NLL deltas/distribution, exact flips,
terrible-token membership, score time, gates, and disposition.
The selected adaptive-A8 W8 execution A/B for those identical artifact bytes is retained under
`profiles/ppl/w8a8-shape-dispatch-ab-20260903/`; the aggregate comparison preserves the earlier
represented-BF16 W8 row as its explicit control.
Activate the ROCm Python environment described in the scorer's adjacent README
before invoking `run.py`; it executes `ppl.py` directly through that environment.

Configure the temporary candidate identities explicitly:

```bash
cmake -S . -B build-r9700-g16 -G Ninja \
  -DNINFER_R9700_KV_VALUE_GROUP=16 -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-r9700-g16 --target ninfer-ppl ninfer_bench --parallel 8

cmake -S . -B build-r9700-g32 -G Ninja \
  -DNINFER_R9700_KV_VALUE_GROUP=32 -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-r9700-g32 --target ninfer-ppl ninfer_bench --parallel 8
```

Run both candidates:

```bash
python3 tools/ppl/run.py \
  --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
  --bf16-reference-weights /path/to/complete/Qwen3.8-27B-BF16 \
  --g16-ppl-bin build-r9700-g16/apps/ninfer-ppl \
  --g32-ppl-bin build-r9700-g32/apps/ninfer-ppl \
  --g16-weights out/qwen3.8-27b-r9700-w8g32-candidate.ninfer \
  --g32-weights out/qwen3.8-27b-r9700-w8g32-candidate.ninfer \
  --quality-tier accuracy \
  --gate r9700-g16=0.02 \
  --gate r9700-g32=0.02 \
  --schedule-parity-max-abs-nll MEASURED_8K_BOUND
```

The default matrix is 8K and 32K, prefill and decode, followed by the paired
execution and 8K position/window extras above. Execution-only candidate cells
reuse the same-length BF16 decode sidecars because the independent reference
formula has no graph or speculative path. Every selected candidate needs an
explicit `--gate`; use
`--allow-ungated` only for bring-up that cannot be used as selection evidence.
Replace `MEASURED_8K_BOUND` with the retained maximum from the complete 8K
schedule comparison; the 63-position probe is not sufficient to define it.
To inspect one candidate while bringing it up, use `--profiles r9700-g16`; the
BF16 reference is inserted automatically. Useful reductions are `--tokens N`,
`--long`, `--schedule prefill`, and `--no-extras`.

Decode cells use the production MTP configuration by default. Pass `--spec none`
for a spec-free decode campaign. Prefill is spec-free. A DFlash target may be
invoked directly through `ninfer-ppl --schedule prefill`; teacher-forced DFlash
decode is intentionally rejected. The BF16 scorer accepts the MTP setting only
as the paired-cell label: it always evaluates the ordinary target distribution,
which is also the target-logit column compared for a product MTP cell.

The corpus is encoded by the tokenizer embedded in the first candidate artifact.
The BF16 source authority implements scoring only, so a reference-only run needs
an already populated `--ids` file. If the checked corpus is too short, the paired
runner invokes `bake_corpus.py` with the first candidate. A direct bake requires
an explicit product artifact:

```bash
python3 tools/ppl/bake_corpus.py \
  --weights /path/to/qwen3_8_27b_r9700_g16.ninfer \
  --ppl-bin /path/to/ninfer-ppl-r9700-g16 \
  --long
```

The output directory contains `results.json`, `results.md`, and every cell's
sidecars. `results.json` is `ninfer_r9700_ppl_campaign` schema v6 and records the
candidate and scorer hashes, validated BF16 source identity/counts, validated
source config/index and all 18 shard SHA-256 values, corpus manifest/hash/token count, exact
commands, compiled group, both activation profiles, the
`t1-ge64-t2-ge320-t3plus-stream-v1` attention classifier, compile-bound dense or XAttention
prefill profile, explicit quality tier, execution labels, sidecar hashes, thresholds, and the
overall gate result. The runner
rejects corpus, model, weights, group,
schedule, spec, graph, chunk, prompt-length, or terrible-token-threshold drift.
Preserve all outputs when moving evidence: aggregate PPL without the paired
token records is not an admission result. Historical `argmax_exact` fields remain in reports but
are explicitly diagnostic for BF16 comparisons.

### Q4 activation-width evaluator

Q4 activation precision is an execution intermediate rather than artifact metadata. A8G64 is the
production-style default for every Q4-bearing artifact; it preserves native IU4 execution while
materially reducing activation error. The ordinary product build selects A8, so build the G16
scorer while leaving the Q4 artifact unchanged:

```bash
cmake -S . -B build-r9700 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_R9700_KV_VALUE_GROUP=16 \
  -DNINFER_BUILD_APPS=ON
cmake --build build-r9700 --target ninfer-ppl --parallel 8
```

Run the same command once for each existing Q4-bearing evaluator, changing only the explicit
artifact and output directory:

```bash
mkdir -p profiles/ppl/q4g64-a8-real
build-r9700/apps/ninfer-ppl \
  --weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --ids tools/ppl/corpus.ids --scheme r9700-g16-a8q4 \
  --schedule prefill --tokens 8192 --skip half --prefill-chunk 4096 \
  --no-device-graph \
  --out-json profiles/ppl/q4g64-a8-real/prefill-8192-g16.json

mkdir -p profiles/ppl/q4-w8-a8-real
build-r9700/apps/ninfer-ppl \
  --weights out/qwen3.8-27b-r9700-q4-w8-n16k16-eval.ninfer \
  --ids tools/ppl/corpus.ids --scheme r9700-g16-a8q4 \
  --schedule prefill --tokens 8192 --skip half --prefill-chunk 4096 \
  --no-device-graph \
  --out-json profiles/ppl/q4-w8-a8-real/prefill-8192-g16.json

mkdir -p profiles/ppl/q4-w8-mse-a8-real
build-r9700/apps/ninfer-ppl \
  --weights out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer \
  --ids tools/ppl/corpus.ids --scheme r9700-g16-a8q4 \
  --schedule prefill --tokens 8192 --skip half --prefill-chunk 4096 \
  --no-device-graph \
  --out-json profiles/ppl/q4-w8-mse-a8-real/prefill-8192-g16.json
```

Every JSON row reports `q4_activation_bits`, `w8_activation_bits`, FP8-Q/K enablement and exact
crossover classifier, the bound artifact identity, KV group, exact workload, and
timing; adjacent NLL and argmax sidecars retain the complete position-aligned result.
Reject a row before comparison if either the reported activation width or artifact identity is not
the requested profile. The campaign runner enforces A8 by default; pass
`--expected-q4-activation-bits 4` only with a scorer explicitly configured with
`-DNINFER_R9700_Q4_ACTIVATION_BITS=4` to reproduce the retained A4 evaluator. Q4 A8 changes no
persistent Q4 bytes. A mixed artifact's W8 matrices use the selected adaptive A8 profile by
default; request `--expected-w8-activation-bits 16` only for the represented-BF16 control.

Configure `build-r9700-a4q4` with `-DNINFER_R9700_Q4_ACTIVATION_BITS=4` only when reproducing
the retained A4 evidence. Leaving the option unset in any fresh build selects A8.

PPL verifies the target distribution under the MTP transaction, but it does not
measure proposal acceptance or product throughput. Record acceptance and whole-path latency with the
same candidate artifact in separately compiled schema-v20 benchmark runs:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py \
  --bench build-r9700-g16/bench/ninfer_bench \
  --weights out/qwen3.8-27b-r9700-w8g32-candidate.ninfer \
  --expected-kv-value-group 16 --preset pareto --no-build \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --output-dir profiles/bench/r9700-g16-qualification

python3 tools/bench/run_ninfer_bench_matrix.py \
  --bench build-r9700-g32/bench/ninfer_bench \
  --weights out/qwen3.8-27b-r9700-w8g32-candidate.ninfer \
  --expected-kv-value-group 32 --preset pareto --no-build \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --output-dir profiles/bench/r9700-g32-qualification
```

For all-Q4+A8 or mixed Q4/W8+A8, use the default `build-r9700/bench/ninfer_bench` and substitute
the exact `r9700-q4g64-n16k16-eval`, `r9700-q4-w8-n16k16-eval`, or `r9700-q4-w8-mse-n16k16-eval` artifact above. Keep
`--expected-q4-activation-bits 8`; the default expected W8 profile is the selected adaptive A8
route. Pass `--expected-w8-activation-bits 16` only for the represented-BF16 control. The runner
rejects an accidentally reused activation or attention-profile executable.

The raw benchmark report includes the compiled KV group plus speculative rounds,
drafts, accepted tokens, accepted-per-position, acceptance rate, acceptance
length, and fallbacks. The matrix runner rejects a resumed or newly produced
report from the wrong compiled group. DFlash2 requires its own companion
acceptance/performance campaign because teacher-forced DFlash decode is not a
defined PPL schedule.

### XAttention real-model gate

XAttention is a compile-isolated prefill route, not a runtime profile. Its quality gate therefore
uses fresh dense and S16/tau=.9 executables for both unresolved cache-group candidates, the exact
same all-Q4 artifact, corpus, prefill chunk, schedules, and independent BF16 source. Configure and
build the same four selection trees used by the matched benchmark campaigns. Rebuild immediately
before acquisition and let `run.py` retain the resulting executable hashes; an earlier named hash
is never an authority after any linked core or challenger source changes, even when that route is
still unpromoted:

```bash
cmake -S . -B build-r9700-dense-selection-g16 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_R9700_KV_VALUE_GROUP=16 -DNINFER_R9700_XATTENTION_QUALIFICATION=OFF \
  -DNINFER_R9700_Q4_ACTIVATION_BITS=8 -DNINFER_R9700_W8_ACTIVATION_BITS=8 \
  -DNINFER_R9700_FP8_QK_WMMA=1 -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON
cmake --build build-r9700-dense-selection-g16 --target ninfer-ppl --parallel 8

cmake -S . -B build-r9700-dense-selection-g32 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_R9700_KV_VALUE_GROUP=32 -DNINFER_R9700_XATTENTION_QUALIFICATION=OFF \
  -DNINFER_R9700_Q4_ACTIVATION_BITS=8 -DNINFER_R9700_W8_ACTIVATION_BITS=8 \
  -DNINFER_R9700_FP8_QK_WMMA=1 -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON
cmake --build build-r9700-dense-selection-g32 --target ninfer-ppl --parallel 8

cmake -S . -B build-r9700-xattention-model-s16-tau900 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DNINFER_R9700_KV_VALUE_GROUP=16 \
  -DNINFER_R9700_XATTENTION_QUALIFICATION=ON \
  -DNINFER_R9700_XATTENTION_STRIDE=16 \
  -DNINFER_R9700_XATTENTION_TAU_PERMILLE=900 \
  -DNINFER_R9700_Q4_ACTIVATION_BITS=8 -DNINFER_R9700_W8_ACTIVATION_BITS=8 \
  -DNINFER_R9700_FP8_QK_WMMA=1 -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON
cmake --build build-r9700-xattention-model-s16-tau900 \
  --target ninfer-ppl --parallel 8

cmake -S . -B build-r9700-xattention-model-s16-tau900-g32 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DNINFER_R9700_KV_VALUE_GROUP=32 \
  -DNINFER_R9700_XATTENTION_QUALIFICATION=ON \
  -DNINFER_R9700_XATTENTION_STRIDE=16 \
  -DNINFER_R9700_XATTENTION_TAU_PERMILLE=900 \
  -DNINFER_R9700_Q4_ACTIVATION_BITS=8 -DNINFER_R9700_W8_ACTIVATION_BITS=8 \
  -DNINFER_R9700_FP8_QK_WMMA=1 -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON
cmake --build build-r9700-xattention-model-s16-tau900-g32 \
  --target ninfer-ppl --parallel 8
```

Run the matched 8K/32K prefill campaigns. The all-Q4 artifact is a `capacity-speed` candidate, whose
mean-NLL ceiling is `ln(1.05) = 0.048790164169432`; every candidate is compared directly with the
independent BF16 authority and greedy differences are retained diagnostically. The dense campaign
is the route-attribution control, while the S16 campaign is the admission gate:

The abstract causal model is chunk-partition invariant: every row has the same visible prefix and
the GDN state advances in the same token order. The realized BF16 scorer is not proven bitwise
chunk-invariant. Changing the span changes batched matrix shapes and fused-recurrent GDN call
boundaries, which can change deterministic algorithm selection and floating-point association.
The fixed 8,192-source-row FP32 attention-PV accumulation is independent of the outer prefill
chunk, but it does not make projections, QK/softmax, or GDN execution partition invariant. Schema
v6 therefore binds `prefill_chunk` in the campaign and every raw command. Pairing a candidate at
the selected chunk with BF16 rows from another chunk could move NLL, severe-position, or argmax
diagnostics. The contingency below creates exactly two fresh realizations at each of 8K and 32K;
do not use a cross-chunk comparison as a reuse waiver.

```bash
set -euo pipefail
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin:${PATH}
if ! NINFER_SELECTED_PREFILL_CHUNK="$(python3 -c \
  'from pathlib import Path; from tools.bench.select_prefill_chunk import validate_selection_record; print(validate_selection_record(Path("profiles/bench/prefill-chunk-selection-20260904.json"))["selected_prefill_chunk"])')"; then
  echo "prefill-chunk selection validation failed" >&2
  exit 1
fi
case "$NINFER_SELECTED_PREFILL_CHUNK" in
  1024|2048|4096|8192) ;;
  *) echo "unsupported selected prefill chunk" >&2; exit 1 ;;
esac

for PPL_EXECUTABLE in \
  /ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  build-r9700-dense-selection-g16/apps/ninfer-ppl \
  build-r9700-dense-selection-g32/apps/ninfer-ppl \
  build-r9700-xattention-model-s16-tau900/apps/ninfer-ppl \
  build-r9700-xattention-model-s16-tau900-g32/apps/ninfer-ppl; do
  [[ -x "$PPL_EXECUTABLE" ]]
done
for PPL_INPUT in \
  tools/reference/qwen3_8_27b_bf16/ppl.py \
  tools/ppl/corpus.ids \
  out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer; do
  [[ -f "$PPL_INPUT" ]]
done
[[ -d /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 ]]

# All-Q4 sparse and both mixed routes are always fresh. Dense all-Q4 is fresh only when the
# selected chunk differs from the retained 4096-token authority. Check every candidate destination
# before a non-4096 branch starts the expensive BF16 pair.
for PPL_OUTPUT in \
  profiles/ppl/xattention-s16-tau900-q4g64-post-challenger-20260904 \
  profiles/ppl/xattention-dense-q4-w8-mse-post-challenger-20260904 \
  profiles/ppl/xattention-s16-tau900-q4-w8-mse-post-challenger-20260904 \
  profiles/ppl/xattention-q4g64-route-comparison-post-challenger-20260904.json; do
  [[ ! -e "$PPL_OUTPUT" ]]
done
if [[ "$NINFER_SELECTED_PREFILL_CHUNK" != 4096 ]]; then
  [[ ! -e profiles/ppl/xattention-dense-q4g64-post-challenger-20260904 ]]
fi

# The retained exact-v3 BF16 pair is reusable only at its recorded chunk 4096. If another chunk
# wins, run this contingency exactly twice. Each campaign process launches one fresh untraced 8K
# scorer process and one fresh untraced 32K scorer process.
if [[ "$NINFER_SELECTED_PREFILL_CHUNK" == 4096 ]]; then
  NINFER_BF16_AUTHORITY=profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json
  NINFER_BF16_REPEAT=profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json
  ALL_Q4_DENSE_QUALITY=profiles/ppl/xattention-dense-q4g64-v3-rebase-20260904/results.json
  [[ -f "$NINFER_BF16_AUTHORITY" ]]
  [[ -f "$NINFER_BF16_REPEAT" ]]
  [[ -f "$ALL_Q4_DENSE_QUALITY" ]]
else
  NINFER_BF16_A="profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-chunk-${NINFER_SELECTED_PREFILL_CHUNK}-a-20260904"
  NINFER_BF16_B="profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-chunk-${NINFER_SELECTED_PREFILL_CHUNK}-b-20260904"
  NINFER_BF16_REPEAT="profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-chunk-${NINFER_SELECTED_PREFILL_CHUNK}-repeat-comparison-20260904.json"
  ALL_Q4_DENSE_QUALITY=profiles/ppl/xattention-dense-q4g64-post-challenger-20260904/results.json
  [[ ! -e "$NINFER_BF16_A" ]]
  [[ ! -e "$NINFER_BF16_B" ]]
  [[ ! -e "$NINFER_BF16_REPEAT" ]]
  for NINFER_BF16_OUT in "$NINFER_BF16_A" "$NINFER_BF16_B"; do
    /ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python tools/ppl/run.py \
      --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
      --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
      --ids tools/ppl/corpus.ids --profiles bf16-reference \
      --schedule prefill --spec none --no-extras \
      --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
      --quality-tier accuracy --allow-ungated --out "$NINFER_BF16_OUT"
  done
  python3 tools/ppl/compare_bf16_repeats.py \
    --first "$NINFER_BF16_A/results.json" \
    --second "$NINFER_BF16_B/results.json" \
    --out "$NINFER_BF16_REPEAT"
  NINFER_BF16_AUTHORITY="$NINFER_BF16_A/results.json"
fi
export ALL_Q4_DENSE_QUALITY

if [[ "$NINFER_SELECTED_PREFILL_CHUNK" != 4096 ]]; then
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python tools/ppl/run.py \
  --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
  --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --g16-ppl-bin build-r9700-dense-selection-g16/apps/ninfer-ppl \
  --g32-ppl-bin build-r9700-dense-selection-g32/apps/ninfer-ppl \
  --g16-weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --g32-weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --ids tools/ppl/corpus.ids \
  --profiles bf16-reference,r9700-g16,r9700-g32 \
  --schedule prefill --spec none --no-extras \
  --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
  --quality-tier capacity-speed \
  --gate r9700-g16=0.048790164169432 --gate r9700-g32=0.048790164169432 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile dense \
  --reuse-bf16-campaign "$NINFER_BF16_AUTHORITY" \
  --bf16-repeat-comparison "$NINFER_BF16_REPEAT" \
  --out profiles/ppl/xattention-dense-q4g64-post-challenger-20260904
fi

/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python tools/ppl/run.py \
  --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
  --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --g16-ppl-bin build-r9700-xattention-model-s16-tau900/apps/ninfer-ppl \
  --g32-ppl-bin build-r9700-xattention-model-s16-tau900-g32/apps/ninfer-ppl \
  --g16-weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --g32-weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --ids tools/ppl/corpus.ids \
  --profiles bf16-reference,r9700-g16,r9700-g32 \
  --schedule prefill --spec none --no-extras \
  --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
  --quality-tier capacity-speed \
  --gate r9700-g16=0.048790164169432 --gate r9700-g32=0.048790164169432 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile b128-s16-tau900 \
  --reuse-bf16-campaign "$NINFER_BF16_AUTHORITY" \
  --bf16-repeat-comparison "$NINFER_BF16_REPEAT" \
  --out profiles/ppl/xattention-s16-tau900-q4g64-post-challenger-20260904

# The mixed Q4/W8-MSE recipe uses the accuracy tier. Its BF16 rows are identical to the retained
# all-Q4 authority, so both route campaigns import those rows and score only the mixed candidates.
run_mixed_xattention_ppl() {
  local route="$1" g16_bin="$2" g32_bin="$3" output="$4"
  /ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python tools/ppl/run.py \
    --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
    --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
    --g16-ppl-bin "$g16_bin" --g32-ppl-bin "$g32_bin" \
    --g16-weights out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer \
    --g32-weights out/qwen3.8-27b-r9700-q4-w8-mse-n16k16-eval.ninfer \
    --ids tools/ppl/corpus.ids \
    --profiles bf16-reference,r9700-g16,r9700-g32 \
    --schedule prefill --spec none --no-extras \
    --prefill-chunk "$NINFER_SELECTED_PREFILL_CHUNK" \
    --quality-tier accuracy \
    --gate r9700-g16=0.02 --gate r9700-g32=0.02 \
    --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
    --expected-fp8-qk-wmma 1 --expected-xattention-profile "$route" \
    --reuse-bf16-campaign "$NINFER_BF16_AUTHORITY" \
    --bf16-repeat-comparison "$NINFER_BF16_REPEAT" \
    --out "$output"
}
run_mixed_xattention_ppl dense \
  build-r9700-dense-selection-g16/apps/ninfer-ppl \
  build-r9700-dense-selection-g32/apps/ninfer-ppl \
  profiles/ppl/xattention-dense-q4-w8-mse-post-challenger-20260904
run_mixed_xattention_ppl b128-s16-tau900 \
  build-r9700-xattention-model-s16-tau900/apps/ninfer-ppl \
  build-r9700-xattention-model-s16-tau900-g32/apps/ninfer-ppl \
  profiles/ppl/xattention-s16-tau900-q4-w8-mse-post-challenger-20260904
```

These post-challenger output paths are intentionally unused before acquisition. At selected chunk
4,096, `ALL_Q4_DENSE_QUALITY` names the already admitted exact-v3 dense all-Q4 rebase and that one
candidate campaign is not rerun. At any other selected chunk it names the fresh dense all-Q4
output alongside the other three fresh candidates. The earlier dated all-Q4 and mixed-dense
campaign directories remain immutable historical evidence bound to their original scorer bytes;
never rerun a rebuilt scorer into those directories.

At preparation time the all-Q4 artifact SHA-256 is
`19d029a89c1ef1cf87420067555021a7c7b435c31a92bea7c64ccf42c03d80e9`, the mixed
Q4/W8-MSE artifact SHA-256 is
`8fbadf14e355b1943ef9386a91ebafff0d852295a9adc0054bdd430291505ebd`, and the
32,768-token corpus SHA-256 is
`aa1ba6d9932a4bcefdfd1c395bde73911daeba083c6b45d353a421dcd1eae077`. The runner
recomputes each applicable identity and retains it with the BF16 source and newly built scorer
identities.

Do not reuse the earlier `profiles/ppl/q4g64-a8-real/` dense cells for this gate. They predate the
compile-bound `xattention_qualification: false` report field and do not bind their scorer executable
hash in one campaign envelope. The newer direct
`profiles/ppl/xattention-s16-tau900-g16/dense-prefill-8192-g16` sidecars do report the dense
compile profile, but cover only G16/8K and likewise have no campaign-level scorer, artifact, corpus,
or BF16-source binding. Schema-v6 `results.json` binds the exact scorer binaries, artifact, corpus,
BF16 source, expected attention profile, cell commands, and sidecar hashes; preserve both campaign
directories together for matched attribution.
Each candidate command imports rather than recomputes the expensive BF16 rows. The option is available
to any prefill-only, non-speculative campaign whose requested lengths, skip, chunk, device, corpus,
BF16 source, and BF16 scorer match the retained schema-v6 authority. It reopens the imported BF16
raw reports and sidecars and checks their source hashes, complete deterministic execution
provenance, and completeness. Candidate route, recipe, quality tier, gates, candidate cells, and
campaign-wide pass do not affect BF16 reference math and are not reuse constraints.

When the BF16 execution authority changes without changing an already measured dense candidate,
`--reuse-candidate-campaign` performs the complementary offline operation. It imports only the
primary G16/G32 prefill cells from schema-v6 campaigns, verifies the exact artifact and scorer
bytes, corpus, workload, cache/activation/attention profile, raw scorer JSON, commands, and sidecar
hashes, removes every old BF16-derived field, and recomputes those fields against the selected BF16
campaign. It cannot import decode/execution extras, cannot cross a dense/XAttention profile boundary,
and must not be used for candidate code that changed. Repeat the option only to combine disjoint
profile/length subsets: overlapping cells fail, while requested cells absent from the retained
sources execute normally. An incomplete source is never itself promoted as complete. The following
retained invocation rebound the
unchanged historical all-Q4 dense cells to the v3 authority without either scorer running. Its
distinct output already exists and is immutable; a new rebase must name another unused output path:

```bash
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python tools/ppl/run.py \
  --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
  --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --g16-ppl-bin build-r9700-xattention-ppl-dense-g16/apps/ninfer-ppl \
  --g32-ppl-bin build-r9700-xattention-ppl-dense-g32/apps/ninfer-ppl \
  --g16-weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --g32-weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --ids tools/ppl/corpus.ids \
  --profiles bf16-reference,r9700-g16,r9700-g32 \
  --schedule prefill --spec none --no-extras --prefill-chunk 4096 \
  --quality-tier capacity-speed \
  --gate r9700-g16=0.048790164169432 --gate r9700-g32=0.048790164169432 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 --expected-xattention-profile dense \
  --reuse-bf16-campaign profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json \
  --bf16-repeat-comparison profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json \
  --reuse-candidate-campaign profiles/ppl/xattention-dense-q4g64-20260903/results.json \
  --out profiles/ppl/xattention-dense-q4g64-v3-rebase-20260904
```

Use the analogous original mixed-dense campaign and its exact retained dense scorers/artifact for
that recipe. Sparse sidecars measured before an XAttention consumer change are not unchanged and
therefore require fresh scoring.

Every BF16 reuse requires the exact schema-v1 A/B comparison. The runner reopens both comparison
inputs, verifies their hashes and exact 8K/32K rows, requires the selected reusable campaign to be
one of that pair, and binds the comparison and authority hashes into the new schema-v6 campaign.
The terminal Pareto assembler revalidates that binding and requires the same exact comparison for
all dense/sparse and G16/G32 candidates; a deterministic-profile label alone is insufficient.

After both campaigns complete, assemble their direct route comparison:

```bash
python3 tools/ppl/compare_xattention.py \
  --dense "$ALL_Q4_DENSE_QUALITY" \
  --xattention profiles/ppl/xattention-s16-tau900-q4g64-post-challenger-20260904/results.json \
  --out profiles/ppl/xattention-q4g64-route-comparison-post-challenger-20260904.json
```

The schema-v1 comparison requires the all-Q4 `capacity-speed` tier and exact `ln(1.05)` gate,
rejects scorer, artifact, corpus, BF16 source, cache layout, activation, threshold, or workload
drift; verifies the retained report and sidecar hashes; and requires the
two independently executed BF16 sidecar pairs to be byte-identical. For each G16/G32 8K/32K cell
it reports XAttention-minus-dense paired NLL statistics, severe-position membership changes,
diagnostic greedy flips, scorer seconds, and dense-over-XAttention speedup. This comparison is
route attribution, not a replacement for either campaign's independent BF16 admission gates or a
production-throughput measurement.

### Source-only Q4 group-size gate

Before adding a Q4G128 artifact format or runtime route, isolate its weight-codec effect with the
separate diagnostic owner. It decodes canonical signed Q4 with represented FP16 scales back to
BF16 one matrix at a time and then uses the unchanged deterministic source formula. Activations
remain BF16, so this is a go/no-go gate for implementing the format, never product A8Q4 PPL.
The CPU-only preflight validates all 18 shards, all 851 Text tensors, and the 498 raw matrix views
corresponding to the 322 logical matrices consumed by non-speculative Text scoring (the separate
draft head is not executed):

```bash
LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib \
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  tools/ppl/q4_group_source_diagnostic.py \
  --weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --ids tools/ppl/corpus.ids --group 128 --tokens 8192 --device 0 \
  --preflight-only \
  --out profiles/bench/r9700-q4g128-source-ppl-preflight-20260904.json
```

After the GPU queue is explicitly released, run the paired 8K diagnostics into fresh paths and
compare them with the retained exact v3 BF16 cell:

```bash
LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib \
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  tools/ppl/q4_group_source_diagnostic.py \
  --weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --ids tools/ppl/corpus.ids --group 64 --tokens 8192 --device 0 \
  --out profiles/ppl/q4-group-source-8k-20260904/q4g64.json

LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib \
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  tools/ppl/q4_group_source_diagnostic.py \
  --weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --ids tools/ppl/corpus.ids --group 128 --tokens 8192 --device 0 \
  --out profiles/ppl/q4-group-source-8k-20260904/q4g128.json

python3 tools/ppl/compare_q4_group_source.py \
  --q4g64 profiles/ppl/q4-group-source-8k-20260904/q4g64.json \
  --q4g128 profiles/ppl/q4-group-source-8k-20260904/q4g128.json \
  --bf16 profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/8192.prefill.bf16-reference.json \
  --out profiles/ppl/q4-group-source-8k-20260904/comparison.json
```

The two score processes each use the existing single-layer-resident scorer plus bounded 4,096-row
quantization scratch. The largest BF16 matrix is the 2,542,796,800-byte output head; source and
decoded copies coexist, and the largest FP32 scratch plane is 285,212,672 bytes. Use the 32 GiB
R9700 and budget roughly 3--4 minutes of established scorer time per 8K run plus codec overhead
(about 10 minutes for the pair conservatively). Escalate a passing pair to 32K with new paths;
retained v3 scorer times bound that pair at roughly 25--30 minutes plus codec overhead.

### Source-only FP8/Q4 hybrid gate

The canonical four-role hybrid has a separate weight-codec diagnostic before its product PPL
campaign. It reads the target-owned `fp8_hybrid_selection.inc` identity and 144 logical names,
then mechanically follows the existing source recipes to identify raw rows. This is important for
the head-interleaved attention Q projection and for the first 4,096 rows of each GDN QKV source.
The selected rows use canonical row-scaled E4M3FN with FP32 row scales; 178 logical Text matrices
use canonical Q4G64, and the 144 direct-BF16 rank-two Text matrices remain unchanged. The full
artifact inventory remains 144 E4M3 and 295 Q4 because MTP, Vision, and the draft head are outside
ordinary Text scoring.

After the GPU queue is explicitly released, produce the 8K codec-only cell and compare it with the
retained deterministic BF16 authority:

```bash
mkdir -p profiles/ppl/fp8-hybrid-source-8k-20260904
LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib \
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  tools/ppl/fp8_hybrid_source_diagnostic.py \
  --weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --ids tools/ppl/corpus.ids --device 0 \
  --out profiles/ppl/fp8-hybrid-source-8k-20260904/hybrid.json

LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib \
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  tools/ppl/compare_fp8_hybrid_source.py \
  --hybrid profiles/ppl/fp8-hybrid-source-8k-20260904/hybrid.json \
  --bf16 profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/8192.prefill.bf16-reference.json \
  --out profiles/ppl/fp8-hybrid-source-8k-20260904/comparison.json
```

The comparison rehashes both sidecars, requires exact source/corpus and 8K prefill-half alignment,
and applies the existing capacity-speed bounds: mean-NLL delta at most `ln(1.05)` and no more than
0.25% newly severe positions. Activations and matrix arithmetic remain BF16, so this result isolates
the persistent codec and cannot substitute for hybrid product PPL or whole-inference admission.

If that source-codec comparison passes, run the artifact-backed 8K product cell. The campaign
runner requires the decision-owned hybrid identity and its adjacent conversion receipt, including
recipe, selection, object-plan, source-index/ranking, byte-size, and artifact hashes. It reuses only
the 8K cell from the retained exact joint 8K/32K BF16 authority and recomputes candidate-versus-BF16
NLL, newly severe positions, and diagnostic greedy-token flips from the aligned sidecars:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib \
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python tools/ppl/run.py \
  --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
  --g16-weights out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer \
  --g16-ppl-bin build-r9700-dense-selection-g16/apps/ninfer-ppl \
  --profiles r9700-g16 --tokens 8192 --schedule prefill --skip half \
  --prefill-chunk 4096 --spec none --no-extras --device 0 \
  --quality-tier capacity-speed --gate r9700-g16=0.04879016416943205 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 --expected-xattention-profile dense \
  --require-fp8-hybrid \
  --reuse-bf16-campaign profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json \
  --bf16-repeat-comparison profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json \
  --out profiles/ppl/fp8-hybrid-product-8k-20260904
```

After the 8K product campaign passes, seal its greedy-token diagnostic without rerunning either
scorer. This validator reopens the live hybrid artifact and adjacent conversion receipt, the exact
BF16 repeat authority and retained scorer sidecars, and the candidate sidecars; it then recomputes
the position-aligned I32 argmax comparison rather than trusting the campaign's derived fields:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib \
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  tools/ppl/validate_fp8_hybrid_greedy.py \
  --campaign profiles/ppl/fp8-hybrid-product-8k-20260904/results.json \
  --artifact out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer \
  --bf16-campaign profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json \
  --bf16-repeat-comparison profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json \
  --tokens 8192 \
  --out profiles/ppl/fp8-hybrid-product-8k-20260904/greedy-token-diagnostic.json
```

The following CPU-only comparison localizes the residual between the source-codec formula and the
artifact-backed product. It first revalidates the same artifact receipt, BF16 repeat authority,
source/corpus identity, workload, and all six sidecars. Its product-minus-source row collectively
contains activation quantization, product matrix arithmetic/reduction, cache codecs, and fused
implementation differences; these effects cannot be separated further from final-token sidecars:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib \
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  tools/ppl/compare_fp8_hybrid_execution.py \
  --source-codec profiles/ppl/fp8-hybrid-source-8k-20260904/hybrid.json \
  --product-campaign profiles/ppl/fp8-hybrid-product-8k-20260904/results.json \
  --artifact out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer \
  --bf16-campaign profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json \
  --bf16-repeat-comparison profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json \
  --tokens 8192 \
  --out profiles/ppl/fp8-hybrid-execution-localization-8k-20260904.json
```

Only if the 8K product campaign exits successfully and its `results.json` reports `pass: true`,
run the separate 32K cell. It consumes the same artifact, receipt, scorer, corpus, BF16 authority,
and gate:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib \
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python tools/ppl/run.py \
  --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
  --g16-weights out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer \
  --g16-ppl-bin build-r9700-dense-selection-g16/apps/ninfer-ppl \
  --profiles r9700-g16 --tokens 32768 --schedule prefill --skip half \
  --prefill-chunk 4096 --spec none --no-extras --device 0 \
  --quality-tier capacity-speed --gate r9700-g16=0.04879016416943205 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 --expected-xattention-profile dense \
  --require-fp8-hybrid \
  --reuse-bf16-campaign profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json \
  --bf16-repeat-comparison profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json \
  --out profiles/ppl/fp8-hybrid-product-32k-20260904
```

If the conditional 32K product campaign passes, seal its independent greedy diagnostic with the
same authorities:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib \
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  tools/ppl/validate_fp8_hybrid_greedy.py \
  --campaign profiles/ppl/fp8-hybrid-product-32k-20260904/results.json \
  --artifact out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer \
  --bf16-campaign profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json \
  --bf16-repeat-comparison profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json \
  --tokens 32768 \
  --out profiles/ppl/fp8-hybrid-product-32k-20260904/greedy-token-diagnostic.json
```

Also validate the complete 16,383-position product/BF16 localization boundary. There is currently
no same-length 32K source-codec sidecar, so omitting `--source-codec` is intentional: the report
must say `source_localization_available: false` and makes no cross-length attribution. If a 32K
source-codec diagnostic is later acquired independently, pass its JSON with `--source-codec` to
enable the same three-way decomposition used at 8K.

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib \
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  tools/ppl/compare_fp8_hybrid_execution.py \
  --product-campaign profiles/ppl/fp8-hybrid-product-32k-20260904/results.json \
  --artifact out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer \
  --bf16-campaign profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json \
  --bf16-repeat-comparison profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json \
  --tokens 32768 \
  --out profiles/ppl/fp8-hybrid-execution-localization-32k-20260904.json
```

These bounded prefill cells establish artifact-backed quality only. Exact greedy disagreements
against BF16 are reported for diagnosis but are not a coherent gate for a lossy recipe. Decode,
execution-parity extras, full product PPL, capacity, and whole-inference selection remain separate.

After both product PPL cells pass, run the bounded C1 selected-artifact execution gate. Do not use
`--no-extras`: the runner's existing extras compare the MTP3 Device Graph cell with eager execution,
ordinary decoding, and an MTP4 draft-window control using complete aligned NLL sidecars and exact
argmax identity.

The runner probes its own interpreter for a ROCm PyTorch build before creating the campaign
directory. Keep the explicit environment below: Python scorer scripts are launched with this same
interpreter, and the complete argv is retained in each cell's provenance.

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin:${PATH}
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python -c \
  "import torch; assert torch.version.hip, 'the execution gate requires ROCm PyTorch'"
NINFER_EXECUTION_GATE_DIR=profiles/ppl/fp8-hybrid-product-execution-c1-20260904
test ! -e "$NINFER_EXECUTION_GATE_DIR" || \
  { test -d "$NINFER_EXECUTION_GATE_DIR" && \
    test -z "$(find "$NINFER_EXECUTION_GATE_DIR" -mindepth 1 -print -quit)"; }

/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python tools/ppl/run.py \
  --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
  --g16-weights out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer \
  --g16-ppl-bin build-r9700-dense-selection-g16/apps/ninfer-ppl \
  --profiles bf16-reference,r9700-g16 --tokens 8192 --schedule decode --skip half \
  --prefill-chunk 4096 --spec mtp --draft-tokens 3 --device 0 \
  --quality-tier capacity-speed --gate r9700-g16=0.04879016416943205 \
  --execution-parity-max-abs-nll 0 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 --expected-xattention-profile dense \
  --require-fp8-hybrid \
  --out "$NINFER_EXECUTION_GATE_DIR"

/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  tools/ppl/validate_fp8_hybrid_execution_gate.py \
  --results "$NINFER_EXECUTION_GATE_DIR/results.json" \
  --width-tool build-r9700/src/ninfer_r9700_runtime_planner_qual \
  --out "$NINFER_EXECUTION_GATE_DIR/execution-gate.json"
```

The validator reopens the exact hybrid artifact and adjacent conversion receipt, rehashes both
scorers, requires the complete MTP3 graph/eager, MTP3/ordinary, and MTP3/MTP4 exact comparison
matrix, and asks the compiled host authority for the C=1..4 prepared descriptor inventories.
For chunk 4096 those inventories must be ordinary `{1,2,3,4,4096}`, MTP3
`{1,2,3,4,8,12,16,4096}`, and MTP4 `{1,2,3,4,5,10,15,20,4096}`. This host inventory check does
not allocate a device and does not substitute for the physical C1 execution cells.

## Candidate lifecycle

The registry in `schemes.py` deliberately contains only the independent BF16
authority and the G16/G32 FP8-K/INT4-V candidates. A kernel implementation does
not become another PPL profile: it first passes its complete Op oracle and then
replaces the route inside the appropriate fixed candidate build. Do not add
format flags, sparse-attention selectors, or placeholder profiles.

Quality eligibility is not model selection. Pareto comparison requires complete matched 8K and
32K quality, resolved capacity, and whole-inference tokens/second for every required prefill/decode
and C=1..4 workload. A quality-eligible candidate A dominates candidate B only when A is no worse
in mean-NLL delta, new-severe-position rate, resolved capacity, and every required whole-inference
speed cell, and is strictly better in at least one. Otherwise both remain retained frontier
choices. Raw PPL `score_seconds` is never a speed objective. `pareto.py` applies this rule and
rejects records missing any required whole-inference or capacity cell.

Frontier retention is the evidence boundary, not permission to ship two cache ABIs. If G16 and
G32 builds of the same eligible weight recipe are both non-dominated, select the one static
production cache/execution profile by this ordered rule:

1. For every required whole-inference cell, divide the candidate's tokens/second by the best
   frontier value in that cell; maximize the minimum ratio.
2. On an exact tie, apply the same maximin normalization to resolved capacity over every required
   concurrency cell.
3. On an exact tie, minimize the worst fraction of the declared quality tier consumed across all
   required quality cells, considering both mean-NLL delta and severe-position budget.
4. If all measured objective vectors are exactly tied, choose the lexicographically first
   canonical static-profile tuple `(value_group, K layout, V layout, V-scale layout,
   Q4 activation bits, W8 activation bits, FP8-Q/K profile, XAttention profile)`.

This post-frontier rule uses no workload weights and cannot trade a severe slow cell for an
average win. It first collapses static cache/execution profiles for byte-identical weights, then
applies the same ordered rule globally to the retained per-recipe winners. Retain the complete
frontier, every per-recipe decision, normalized ratios, and the terminal decisive stage.

Schema v7 emits this under `same_recipe_static_profile_selection`: `selections` contains one
machine-readable winner, ordered rationale, and normalized objective map for each weight recipe
with multiple frontier static profiles. `not_collapsed` names frontier candidates that have no
same-recipe peer. Assembled inputs derive recipe identity from the provenance-bound artifact
identity and hash; candidate names are never parsed as recipe identities.

The `--require-xattention-dense-controls` admission mode is stricter: for every candidate artifact
it requires native schema-v6 dense and B128/S16/tau900 PPL campaigns for G16 and G32, all sharing
one BF16/corpus authority, plus each candidate's matched C=1..4 capacity and whole-inference
matrices. The all-Q4, mixed Q4/W8, and authority-bound four-role FP8/Q4 recipes require all twelve
Cartesian artifact/cache/attention candidates; dense-only recipe selection cannot establish how sparse
prefill changes cross-recipe speed, capacity, or combined quality. Schema v7 retains
each same-recipe winner and emits one `terminal_production_selection` across those winners by the
same ordered global maximin rule. The retained 1+3 means are compared exactly without a noise
tolerance, and the raw repetitions and spread remain evidence.
Each schema-v7 candidate carries an explicit storage-profile identity derived from its exact
artifact ID. The hybrid campaign must additionally retain the authority-bound conversion receipt;
it never inherits the all-Q4 storage profile merely because its unselected matrices are Q4G64.
The terminal record binds and reopens the complete schema-v4 Pareto input before downstream use.

Its input is an explicit retained comparison manifest:

```json
{
  "required_quality_cells": ["8k", "32k"],
  "required_capacity_cells": ["c1", "c4"],
  "required_speed_workloads": ["prefill_8k_c1", "decode_8k_c1", "decode_8k_c4"],
  "candidates": [
    {
      "name": "all-q4-a8",
      "cache_profile": {
        "value_group": 16,
        "plane_layouts": {
          "key": "token-fastest-head-major",
          "value": "feature-fastest-page-major",
          "value_scale": "feature-fastest-page-major"
        }
      },
      "execution_profile": {
        "q4_activation_bits": 8,
        "w8_activation_bits": 8,
        "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
        "xattention_profile": "dense"
      },
      "quality_cells": {
        "8k": {
          "eligible": true, "tier": "capacity-speed", "mean_nll_delta": 0.01,
          "complete_finite_aligned": true, "scored_positions": 4095,
          "new_severe_positions": 2
        },
        "32k": {
          "eligible": true, "tier": "capacity-speed", "mean_nll_delta": 0.02,
          "complete_finite_aligned": true, "scored_positions": 16383,
          "new_severe_positions": 4
        }
      },
      "whole_inference_tokens_per_second": {
        "prefill_8k_c1": 1.0,
        "decode_8k_c1": 1.0,
        "decode_8k_c4": 1.0
      },
      "capacity_by_cell": {
        "c1": {"measurement_kind": "resolved_effective_maximum",
               "binding_constraint": "model_context", "tokens": 262144},
        "c4": {"measurement_kind": "resolved_effective_maximum",
               "binding_constraint": "device_memory", "tokens": 260000}
      }
    }
  ]
}
```

Run `python3 tools/ppl/pareto.py --input comparison.json --out pareto.json`. The schema-v7
classifier rejects 32K workload-feasibility rows as capacity objectives; each capacity cell must
explicitly identify a `resolved_effective_maximum` and whether device memory or the model's native
context ceiling binds. The classifier
recomputes tier eligibility and severe rate from the counts, rejects a false declared eligibility,
keeps the full objective values and names every dominator. The retained result is then collapsed
to one required static cache and execution profile only by the post-frontier rule above.

Do not construct `comparison.json` by copying summary values. Assemble it from the retained raw
authorities so artifact, executable, compiled profile, command, case, and concurrency provenance
are revalidated:

```bash
python3 tools/ppl/assemble_pareto.py \
  --prefill-chunk-selection profiles/bench/prefill-chunk-selection-20260904.json \
  --candidate all-q4-g16 r9700-q4g64-n16k16-eval 16 QUALITY_JSON CAPACITY_DIR WHOLE_DIR \
  --candidate mixed-g16 r9700-q4-w8-mse-n16k16-eval 16 QUALITY_JSON CAPACITY_DIR WHOLE_DIR \
  --out profiles/bench/pareto-input.json

python3 tools/ppl/pareto.py \
  --input profiles/bench/pareto-input.json \
  --out profiles/bench/pareto-result.json
```

Each candidate's `QUALITY_JSON` accepts either the retained
`ninfer-r9700-q4-a8-final-quality-v2` comparison or a passing native
`ninfer_r9700_ppl_campaign` schema-v6 `results.json`. Candidate-local quality inputs allow dense
and B128/S16/tau900 static profiles to enter the same decision without merging or relabeling their
campaigns. The assembler reopens every selected native raw cell and sidecar, verifies its compiled
profile identity, and binds their hashes into the schema-v4 input.
`--prefill-chunk-selection` is mandatory. The assembler recomputes the schema-v2 global selection,
binds its hash, and requires every PPL cell plus every capacity and whole command to carry its
selected value. Each whole directory must contain one optimized-head MTP3 report and one matched
ordinary greedy control at every C=1..4 point. Assembly takes timing only from MTP3, rejects zero
draft activity, and requires exact retained target-token parity for every workload, repetition,
and lane. The raw repetition counter sums request lanes: for g256/MTP3 it must equal `64*C`, and
the test aggregate must equal the sum of all three repetitions. The theoretical minimum remains
diagnostic acceptance accounting; extra rounds neither trigger a proposal-head precision branch
nor alter base selection. Schema v7 carries no downstream-readiness or MTP-head status.
Selected-profile NIAH and DFlash admission remain separate gates. There is no manual or implicit
historical-4096 path.

After the dense and sparse G16/G32 capacity/whole matrix pairs finish, XAttention admission
assembles the complete four-profile Cartesian set for every capacity-eligible artifact:

```bash
python3 tools/ppl/assemble_pareto.py \
  --require-xattention-dense-controls \
  --prefill-chunk-selection profiles/bench/prefill-chunk-selection-20260904.json \
  --candidate dense-g16 r9700-q4g64-n16k16-eval 16 \
    "$ALL_Q4_DENSE_QUALITY" \
    profiles/bench/pareto-capacity-post-promotion-dense-all-q4-g16-20260904 \
    profiles/bench/pareto-whole-post-promotion-dense-all-q4-g16-20260904 \
  --candidate dense-g32 r9700-q4g64-n16k16-eval 32 \
    "$ALL_Q4_DENSE_QUALITY" \
    profiles/bench/pareto-capacity-post-promotion-dense-all-q4-g32-20260904 \
    profiles/bench/pareto-whole-post-promotion-dense-all-q4-g32-20260904 \
  --candidate xattention-g16 r9700-q4g64-n16k16-eval 16 \
    profiles/ppl/xattention-s16-tau900-q4g64-post-challenger-20260904/results.json \
    profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-all-q4-g16-20260904 \
    profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-all-q4-g16-20260904 \
  --candidate xattention-g32 r9700-q4g64-n16k16-eval 32 \
    profiles/ppl/xattention-s16-tau900-q4g64-post-challenger-20260904/results.json \
    profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-all-q4-g32-20260904 \
    profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-all-q4-g32-20260904 \
  --candidate mixed-dense-g16 r9700-q4-w8-mse-n16k16-eval 16 \
    profiles/ppl/xattention-dense-q4-w8-mse-post-challenger-20260904/results.json \
    profiles/bench/pareto-capacity-post-promotion-dense-mixed-g16-20260904 \
    profiles/bench/pareto-whole-post-promotion-dense-mixed-g16-20260904 \
  --candidate mixed-dense-g32 r9700-q4-w8-mse-n16k16-eval 32 \
    profiles/ppl/xattention-dense-q4-w8-mse-post-challenger-20260904/results.json \
    profiles/bench/pareto-capacity-post-promotion-dense-mixed-g32-20260904 \
    profiles/bench/pareto-whole-post-promotion-dense-mixed-g32-20260904 \
  --candidate mixed-xattention-g16 r9700-q4-w8-mse-n16k16-eval 16 \
    profiles/ppl/xattention-s16-tau900-q4-w8-mse-post-challenger-20260904/results.json \
    profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-mixed-g16-20260904 \
    profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-mixed-g16-20260904 \
  --candidate mixed-xattention-g32 r9700-q4-w8-mse-n16k16-eval 32 \
    profiles/ppl/xattention-s16-tau900-q4-w8-mse-post-challenger-20260904/results.json \
    profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-mixed-g32-20260904 \
    profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-mixed-g32-20260904 \
  --candidate four-role-dense-g16 r9700-q4g64-f8e4m3-four-role-n16k16-eval 16 \
    "$FOUR_ROLE_DENSE_QUALITY" \
    profiles/bench/pareto-capacity-post-promotion-dense-four-role-g16-20260904 \
    profiles/bench/pareto-whole-post-promotion-dense-four-role-g16-20260904 \
  --candidate four-role-dense-g32 r9700-q4g64-f8e4m3-four-role-n16k16-eval 32 \
    "$FOUR_ROLE_DENSE_QUALITY" \
    profiles/bench/pareto-capacity-post-promotion-dense-four-role-g32-20260904 \
    profiles/bench/pareto-whole-post-promotion-dense-four-role-g32-20260904 \
  --candidate four-role-xattention-g16 r9700-q4g64-f8e4m3-four-role-n16k16-eval 16 \
    "$FOUR_ROLE_XATTENTION_QUALITY" \
    profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-four-role-g16-20260904 \
    profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-four-role-g16-20260904 \
  --candidate four-role-xattention-g32 r9700-q4g64-f8e4m3-four-role-n16k16-eval 32 \
    "$FOUR_ROLE_XATTENTION_QUALITY" \
    profiles/bench/pareto-capacity-post-promotion-xattention-s16-tau900-four-role-g32-20260904 \
    profiles/bench/pareto-whole-post-promotion-xattention-s16-tau900-four-role-g32-20260904 \
  --out profiles/bench/pareto-input-post-promotion-20260904.json

python3 tools/ppl/pareto.py \
  --input profiles/bench/pareto-input-post-promotion-20260904.json \
  --out profiles/bench/pareto-result-post-promotion-20260904.json
```

`FOUR_ROLE_DENSE_QUALITY` and `FOUR_ROLE_XATTENTION_QUALITY` name the native schema-v6
four-role-hybrid campaigns carrying the exact authority-bound conversion receipt for their dense
and B128/S16/tau900 profiles, respectively.

The dense candidates are mandatory for XAttention admission: without them the result compares only
the sparse cache groups and cannot prove an end-to-end sparse speed improvement. A sparse campaign
or matrix must never be relabeled as dense evidence. `--reuse-bf16-campaign` may import BF16-only
cells across candidate recipes, dense/sparse routes, and quality tiers because none of those
choices changes the independent BF16 reference. It revalidates schema/model identity, BF16
  source/scorer, corpus, lengths, prefill schedule, skip/chunk/device settings, raw reports, and
  sidecars. The raw BF16 cells must also carry the mandatory hipBLAS/no-atomics/strict-deterministic
execution profile, and every cell must match the campaign's exact `reference_execution` object;
that object binds the complete scorer/FLA implementation and interpreter/package/runtime/device
environment. The required v3 profile additionally binds the structured full-attention PV identity:
explicit FP32 `torch.mm`, fixed 8,192-row absolute source chunks in ascending order, final partial
chunk inclusion, and ordered FP32 accumulation. It also binds the GDN recurrence dispatch as the
  project’s explicit FP32 route at T=1 and FLA fused-recurrent at every T>1 span, including the
  fixed `FLA_USE_FAST_OPS=0` exponent route, operand,
output, final-state, scale, and state-format identity. The same object now requires disabled
PyTorch TunableOp, highest matmul precision, canonical TF32/Triton/rocBLAS/launch/allocator
environment controls, absence of architecture, Tensile-library, and custom-allocator overrides,
actual gfx1201 identity, and the resolved Triton/AMD buffer, atomic, prefetch, async-copy,
ping-pong, transpose, and packed-float lowering controls. Older v2 chunk-GDN results and earlier
v3 results lacking this complete structured boundary are rejected.
Candidate artifact identity, candidate cells,
route, tier, gates, and campaign-wide pass are deliberately not BF16 reuse constraints.

Each `--candidate` takes exactly six values. The assembler requires schema-v14 complete matrices,
all C=1..4 points, the same exact benchmark bytes across that candidate's two matrices, and
artifact bytes matching its 8K/32K quality rows. It reopens and validates every schema-v20 raw
report before emitting 24 matched speed cells and exactly four resolved effective-capacity cells.
Each fresh-request whole row owns prefill throughput, decode throughput and acceptance, and
end-to-end throughput from the same repetitions; separate pure-prefill and prefix-reuse-seeded
decode matrices are diagnostic duplicates and are not selection inputs.
The assembler emits `ninfer_r9700_pareto_input` schema v4; the classifier emits
`ninfer_r9700_pareto_comparison` schema v7, retaining the full frontier, deterministic per-recipe
static-profile selections, and the terminal artifact/cache/execution winner with normalized values
and decisive stages.
If native-context startup cannot satisfy its minimum reservation plus headroom, the missing
capacity cell makes that candidate non-comparable. A missing report is accepted only when it has a
unique matching entry in the matrix `failures.json`; the assembler binds that entry, exact command,
stdout/stderr hashes, and stderr text as an unresolved capacity-measurement failure instead of
inferring an OOM or inventing a capacity value. Any unrelated failure rejects the campaign.
For a candidate already excluded by such a capacity failure, pass `-` for `WHOLE_DIR`;
the assembler retains it as explicitly non-comparable without requiring wasteful speed campaigns.
