# R9700 Softmax Attention ownership

This document records the current Softmax Attention ownership for the sole Qwen3.8-27B R9700
product. It specializes the numerical and ownership rules in `op-development.md`; it is not a
proposal for a generic Attention framework.

## Semantic vocabulary

For head dimension `D`, query-head count `Hq`, KV-head count `Hkv`, query row `i`, and the
entry-defined visible key set `A(i)`:

```text
group          = Hq / Hkv
kv_head(h)     = floor(h / group)
score(i,h,j)   = scale * dot(Q[:,h,i], K[:,kv_head(h),j]), j in A(i)
probability    = stable_softmax(score)
ideal[:,h,i]   = sum(j in A(i)) probability(i,h,j) * V[:,kv_head(h),j]
```

`Hq == Hkv`, `Hkv == 1`, and `1 < Hkv < Hq` are MHA, MQA, and GQA geometries respectively.
Those terms describe head mapping; they do not create separate public Op families or storage
owners. Kernel tiling, online Softmax, paged addressing, and WMMA are private implementation
profiles and may not change `A(i)`, represented cache values, state publication, or outputs.

Every floating-point route is checked directly against an independent naive FP64 oracle built
from represented public inputs. A production route may use its natural private operand precision
and reduction association. Exact codec bytes and state transitions use exact comparison.

## Current native entries

| Consumer | Geometry and visible set | Persistent representation | Current owner |
|---|---|---|---|
| Text and MTP | `D256/Hq24/Hkv4`; causal prefix or explicit packed-tree ancestry | typed FP8-K/INT4-V growing cache | Qwen3.8 R9700 full-attention leaf |
| DFlash Full | `D128/Hq32/Hkv8`; read-only context plus the complete live query segment | independent BF16 paged context | native bidirectional GQA Op |

Vision packed attention and DFlash Local sliding-window attention have different represented
inputs and visibility rules. They do not borrow the Text/MTP cache codec or publication state.

## Text and MTP: typed FP8-K/INT4-V cache

Text and MTP have no homogeneous cache-dtype selector. Their only growing-cache representation is:

- one OCP E4M3FN byte per K feature;
- two canonical signed INT4 V codes per byte;
- one IEEE binary16 V scale per feature group (provisionally 16 features; G16 versus G32 remains a
  real-model gate);
- a 64-token paged block table and an explicit physical plane-layout identity.

`src/core/fp8_int4_paged_kv_cache.{h,cpp}` owns the physical storage plan, semantic fingerprint,
typed per-layer view, and allocation validation. `qwen3::PagedKVCache` owns per-sequence
allocation and publication state. A cache view cannot be reconstructed from arbitrary homogeneous
planes or selected by dtype at runtime.

### Mutation and publication

`PagedKVTransaction` is the sole Text/MTP mutation authority. An append transaction validates the
logical suffix, clears one caller-owned status word, requires each layer exactly once on one HIP
stream, and publishes the new frontier only after the codec status succeeds. A pending
`PagedKVLayerRead` authorizes the same stream to attend a newly appended layer before the all-layer
transaction commits; it does not expose that frontier to another consumer.

The raw native mutations are:

- `src/ops/r9700/kv/fp8_int4_kv_append.{h,hip}` for represented-BF16 K/V to stored FP8-K/INT4-V;
- `src/ops/r9700/kv/fp8_int4_kv_compact.{h,hip}` for in-place monotone speculative-path gather.

Accepted speculative paths use `PagedKVCache::begin_compact`, launch every layer through the same
transaction, and commit the checked publication frontier only after the shared device status stays
zero. When retained tokens are already in place, `truncate_publication` closes the speculative
suffix without copying bytes. Neither route permits direct frontier repair by the schedule.

### Attention leaf

`src/targets/qwen3_8_27b/impl/r9700_full_attention.{h,hip}` is the target-private semantic bridge.
It accepts represented BF16 queries, a generation-bound typed layer read, optional device I32
causal positions, optional paired packed-tree ancestry metadata, and an optional device-resident
active-row count. It delegates to
`src/ops/r9700/kv/fp8_int4_kv_attention.{h,hip}` at the fixed Qwen3.8 geometry.

The selected production implementation has two finite crossovers. At context 8,192 and above,
ordinary T=1 and fixed-width T=4 use the admitted split-512 score, partial Softmax/PV, and merge
stages; T=4 supports causal, packed-tree, and device-active-row metadata. T=1 with tree or
device-active-row metadata retains the fused leaf at every context; workspace sizing and launch
share this metadata-aware decision. Below the split boundary, ordinary T=1
at context 64 and above and T=2 at context 320 and above privately cast represented BF16 Q to
E4M3FN and use wave32 FP8-Q/FP8-K WMMA, a caller-owned FP32 score panel, stable FP32 Softmax, and
exact vector FP32-probability INT4-V accumulation. Remaining shapes stream FP8 K and signed-
INT4-times-FP16-scale V through online FP32 Softmax without materializing scores. The query cast is
an implementation profile, not a cache-format or public-semantic change, and the compile-isolated
score-streaming build remains the PPL control. Output is FP32; the family schedule performs the
explicit BF16 cast where the next semantic boundary requires it. Inactive fixed-width rows are
exact positive zero. Invalid represented position/tree/count/table-row metadata remains
conspicuous as NaN in the affected contract-defined rows.

### Qualification-only XAttention prefill evaluator

The sparse-prefill candidate follows Algorithm 1 of Xu et al., *XAttention: Block Sparse
Attention with Antidiagonal Scoring* (ICML 2025, PMLR 267). For stride `S`, estimator plane `p`
concatenates query slices in the inverse order `Q_slice[S-1::S]` through `Q_slice[0::S]`, while
the key groups retain forward order `K[0::S]` through `K[S-1::S]`. Each represented score is
therefore the explicit antidiagonal sum

```text
logit(p,g) = scale / S * sum(s=0..S-1) dot(Q[p*S + S-1-s], K[g*S+s])
```

where `scale` is `1/sqrt(D)` for Qwen3.8. The implementation applies a separate stable Softmax
over the causal key groups of every plane, sums those normalized probabilities into paper B128
key blocks, and retains the minimum descending-mass set reaching the compile-fixed `tau`. Each
retained block expands to two ordered B64 cache pages. Equal masses choose the lower logical block;
the consumer then visits retained pages in
ascending logical order. This normalization is per plane: accumulating only
`exp(logit - plane_max)` is not the represented algorithm because unrelated plane denominators
would change page rank and threshold.

NInfer's causal/ragged adaptation uses one estimator per B128 prefill block. Complete query planes
use the paper's concatenation literally. Ragged final Q and K groups are zero padded by omitting
out-of-range terms while retaining the `/S` scale; neither is discarded. These choices keep the
estimator causal and prevent reading an unpublished key tail.

Q and K use the same global block origin. An ordinary prefill chunk whose first absolute query
position is not divisible by B128 emits the complete causal page list and executes exact dense
attention. It is never locally reanchored, because doing so would change the keep set solely when
prefix reuse or suffix rewriting moves a chunk boundary. Aligned B128 blocks preserve the internal
stride origin.

The causal selector makes the sink B128 block and current B128 query block mandatory and counts
their estimated mass toward `tau` before greedily adding other blocks. This matches upstream
`find_blocks_chunked` and avoids the over-retention caused by unioning mandatory blocks only after
selection. The candidate consumes the same FP8-E4M3FN K,
signed-INT4 V, and FP16 V-scale planes as dense attention; it introduces no cache representation or
runtime selector. `tau=1` bypasses all estimator arithmetic and emits every causal page in dense
order.

Dense P2048 uses the physically selected production page-tiled route, not the sparse consumer and
not the decode split-512 path. Its fixed Bk64 tile matches one cache page and Bq16 is the sole
retained query tile. A 256-thread query-head CTA cooperatively decodes the
token-fastest FP8 K page to BF16 LDS once, reuses that represented tile across its query rows, and
uses native BF16 WMMA for QK. It retains increasing logical-key order, per-row causal positions,
FP32 online softmax, and direct feature-fastest signed-INT4/FP16-scale V accumulation, with no
global score or probability matrix. Fragmented physical page IDs affect only the cooperative page
resolution. Production selects it only for initial-prefix P=128..4096 with the selected physical
plane layouts; shorter contexts, later chunks, tree masks, and other layouts retain their existing
routes.

The static resource gate uses the combined gfx1201 residency envelope rather than an isolated
register target: next-free VGPR must be at most 240, allocation-rounded LDS at most 43,520 bytes,
reported occupancy at least 6, with wave32, a 256-thread maximum workgroup, WGP mode, and no
private, scratch, or register-spill storage. Both G16/G32 Bq16 specializations report 217 next-free
VGPR and 37,160 bytes of LDS. That LDS footprint already bounds the CTA residency,
so reducing register allocation alone is not a valid optimization claim.

The retained pre-promotion physical A/B report
`profiles/bench/r9700-dense-prefill-attention-ab.json` has SHA-256
`d7a0f50a6ae91448583b07d7477d2926b950b65eeadd3ac924f28a8354ed95a6`. Bq16 won every G16/G32
P128/512/1024/2048/4096 cell. Its selected/incumbent median ratios are respectively
0.561224173/0.566735009/0.504308119/0.437316979/0.416703457 for G16 and
0.546212923/0.554317816/0.495417690/0.429230227/0.417418378 for G32. Bq4/Bq8 were removed after
selection. Post-promotion operator revalidation and the complete low-context whole ladder remain
required; the operator A/B alone does not prove the 2,000 tok/s whole-prefill floor.

The evaluator is scoped only to ordinary Text prefill. A DFlash-enabled request uses that selected
ordinary prefill route while capturing its companion features; DFlash proposal attention and
target/tree verification, plus decode, MTP, and GDN, remain dense throughout qualification. The default build does not compile or link
the sparse leaf and exposes no Engine, CLI, serving, PPL, or benchmark selector. A separate
`NINFER_R9700_XATTENTION_QUALIFICATION=ON` build compiles the S16/tau900/B128 profile through the
target-private Text-prefill leaf and its exact workspace planner, allowing matched PPL and whole
prefill measurement without changing the public runtime contract. Its PPL sidecar reports
`xattention_qualification=true`, `xattention_profile=b128-s16-tau900`, B128, S16, and tau 900;
the ordinary scorer reports `xattention_qualification=false`. `tools/ppl/run.py` rejects either
binary when it does not match the explicit `--expected-xattention-profile` campaign identity.

`tools/r9700/xattention_prefill_qual.hip` independently derives keep sets and retained-page FP64
attention from represented logical Q/K/V values, and then checks the raw physical execution at
`D256/Hq24/Hkv4`. Its fixture writes the chosen physical planes with standalone handwritten
offsets; it does not by itself prove the generic plane layout. Physical-format evidence is
compositional with `tools/r9700/kv_op_qual.hip`, which independently qualifies the selected
token-fastest K and feature-fastest V/V-scale codec and addressing contract.

The September 3, 2026 schema-v1 reports are historical diagnostics, not rejection or cross-profile
evidence. Their zero-filled Q/K corpus makes every estimator block equiprobable, so `tau=0.900`
necessarily keeps about 90% of the cache. They also predate B128 selection and mandatory-mass
accounting and retain no executable/source identity; their S16/S8 ranker ratio is not attributable
across binaries. Earlier RTX 5090 work remains useful architecture-direction evidence, not R9700
admission: at `tau=0.9` its tensor-core B128 implementation improved 32K prefill by about
6.5--6.9% and 64K by 14.6--15.7%, while its 32K NLL and 64K needle gates passed.

The corrected evaluator decodes each logical FP8 K once into caller-owned BF16 packed storage,
then performs two skinny gfx1201 BF16-WMMA passes: one for per-plane maxima and one that accumulates
B128 exp mass without materializing a full logit matrix. The one causal diagonal group per plane
uses a wave-reduced BF16-input FP32 correction so terms beyond that plane's real query frontier are
zero padded rather than entering the WMMA score. FP8 K is exactly representable in BF16 and Q
remains represented BF16, so this is the same estimator oracle rather than an FP8-Q profile.
For a 4,096-row planning chunk, workspace is 17.925/71.690/573.503 MiB at 8K/32K/262K for S16
and 19.425/77.690/621.503 MiB for S8; the 128-row timing fixture uses 513.922 MiB at 262K for
S16. Schema v3 uses a fixed concentrated represented Q/K corpus and reports achieved keep fraction,
independent total/ranker/consumer times, and the measured physical-oracle maximum absolute error
with its absolute-or-relative acceptance tolerances. Both profiles have `2.221e-8` maximum
absolute error under the recorded `3e-4` tolerance. S16 measures `6.74x` dense-over-sparse at 8K
and `7.80x` at 32K, with 14.06% and 11.72% page retention; its rank stage is
`1.162/4.744 ms`. S8 measures `6.61x/7.60x` with the same fixture retention but a slower
`1.416/5.581 ms` rank stage and larger
workspace. S16 therefore advances to model gates. These are structured-fixture Op measurements,
not model-quality or whole-inference evidence. Dense remains the sole production route until the
remaining gates pass.

The compile-isolated G16 S16/tau900 runtime passed the physical planner. A fresh 128-token
dense/XAttention diagnostic produced byte-identical FP32-NLL and I32-argmax sidecars. A standalone
8K XAttention scorer also completed in 50.47 seconds. A numerical comparison with the older
72.09-second dense sidecar gives +0.00048794 mean NLL, two new and six resolved NLL-at-least-10
positions, and 152 diagnostic greedy flips, but that dense sidecar predates the compile-bound
profile and scorer-hash campaign envelope. The comparison is directional smoke evidence, not a
matched model-distribution admission gate. The fresh schema-v6 dense campaign now covers G16/G32;
the matched XAttention campaign at 8K and 32K, schema-v20/v13 benchmark matrices, and long-context
needle retrieval remain required.

Capacity evidence must also match the compile-isolated route. At the 4,096-row prefill chunk and
262,144-token context envelope, the S16 leaf requires 601,361,408 bytes; after the live Text
prefill roots and all-Q4 A8 linear reserve, the current planner's global workspace is
1,085,967,619 bytes. The retained dense G16/G32 capacity executables reserve 603,619,587 bytes, so
their historical resolved C=1..8 curves cannot be reused for the C=1..4 XAttention admission.
Fresh ON-profile G16/G32
capacity curves must bind the same executable/profile as the matched whole-inference evidence.
Static projections from the old free-memory snapshots are diagnostic only and are not resolved
capacity results.

Whole-inference admission is a dense-versus-sparse decision, not merely a choice between sparse
G16 and G32. The all-Q4 route comparison therefore contains four same-artifact candidates: dense
G16, dense G32, XAttention G16, and XAttention G32. Under the C=1..4 product cap the mixed recipe
is also capacity-eligible, so the final product comparison must add the equivalent four
mixed-recipe candidates, with matched dense/XAttention schema-v6 PPL and fresh C=1..4
capacity/whole evidence. Each dense candidate needs its own fresh
schema-v14 capacity/whole pair from one current dense executable. Each whole row retains its
separately timed prefill and decode phases, acceptance, and fresh-request makespan. The retained
schema-v12/v19 dense capacity and standalone phase directories, plus the absent dense whole
matrix, do not match the current assembler or provide a complete speed control. The fixed dense
candidate sidecars remain reusable, but the former BF16 rows are not: the replacement schema-v6
campaign must bind the mandatory deterministic BF16 `reference_execution` provenance alongside
both cache groups, artifact bytes, scorer identity, and the dense execution profile. PPL scorer
seconds and standalone Op timing remain attribution, never whole-inference speed objectives. The retained
all-Q4 four-candidate evidence cannot by itself exclude or select the restored mixed recipe.
The full classifier retains one cache/execution winner per recipe and then applies the same
globally normalized maximin throughput, capacity, quality-budget, and canonical-identity ordering
to emit one schema-v7 `terminal_production_selection`. Quality remains admission rather than the
leading rank.
The decision compares retained per-cell means exactly and retains raw repetition spread only as
evidence; NIAH and DFlash consume the resulting terminal winner.

The retained reproduction command for each candidate is:

```bash
cmake -S . -B build-r9700-xattention-s16-tau900 -G Ninja \
  -DNINFER_BUILD_APPS=OFF -DBUILD_TESTING=ON \
  -DNINFER_R9700_XATTENTION_STRIDE=16 \
  -DNINFER_R9700_XATTENTION_TAU_PERMILLE=900
cmake --build build-r9700-xattention-s16-tau900 --parallel 8 \
  --target ninfer_r9700_xattention_qual
mkdir -p profiles/bench/r9700-xattention-requal-s16-tau900
build-r9700-xattention-s16-tau900/src/ninfer_r9700_xattention_qual \
  --benchmark --iterations 7 \
  --out-json profiles/bench/r9700-xattention-requal-s16-tau900/timing.json
sha256sum build-r9700-xattention-s16-tau900/src/ninfer_r9700_xattention_qual \
  > profiles/bench/r9700-xattention-requal-s16-tau900/executable.sha256
sha256sum src/ops/r9700/kv/fp8_int4_kv_xattention.{h,hip} \
  src/ops/r9700/kv/r9700_xattention_profile.h tools/r9700/xattention_prefill_qual.hip \
  > profiles/bench/r9700-xattention-requal-s16-tau900/sources.sha256
make -C tools/r9700 XATTENTION_STRIDE=16 XATTENTION_TAU_PERMILLE=900 \
  xattention-reports
```

Repeat with `16` replaced by `8` in the build directory, configure value, evidence directory, and
Make variables. The assembly report does not execute the GPU and may be generated independently;
the qualification executable performs the physical correctness and timing run.

The model-distribution build and prefill campaign use the same compile-bound S16 implementation:

```bash
cmake -S . -B build-r9700-xattention-model-s16-tau900 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON \
  -DNINFER_R9700_KV_VALUE_GROUP=16 \
  -DNINFER_R9700_XATTENTION_QUALIFICATION=ON \
  -DNINFER_R9700_XATTENTION_STRIDE=16 \
  -DNINFER_R9700_XATTENTION_TAU_PERMILLE=900
cmake --build build-r9700-xattention-model-s16-tau900 --parallel 8 \
  --target ninfer-ppl ninfer-serve ninfer_bench ninfer_r9700_runtime_planner_qual

/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python tools/ppl/run.py \
  --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
  --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --g16-ppl-bin build-r9700-xattention-model-s16-tau900/apps/ninfer-ppl \
  --g16-weights "$NINFER_XATTENTION_ARTIFACT" \
  --profiles bf16-reference,r9700-g16 \
  --quality-tier capacity-speed --gate r9700-g16=0.048790164169432 \
  --schedule prefill --no-extras \
  --expected-xattention-profile b128-s16-tau900 \
  --out profiles/ppl/r9700-xattention-s16-tau900
```

Use the selected cache-group profile and its matching `--g16-*` or `--g32-*` flags. A retained
dense sidecar may be reused only when corpus identity, scoring schedule, prefill chunk, cache and
artifact identity all match; executable hashes and the explicit XAttention fields distinguish the
two candidate routes. The same isolated build's `ninfer_bench` is the whole-prefill challenger;
decode rows in that binary remain dense and are not sparse-speed evidence.

The corresponding capacity and whole-inference campaigns must use the isolated benchmark bytes
and require the compile-bound profile explicitly. Re-run the G16 configure/build command above
before the campaign: an older cache with `NINFER_BUILD_BENCHMARKS=OFF` is not a usable benchmark
build. Build G32 separately so its compile-time cache layout cannot be confused with G16:

```bash
cmake -S . -B build-r9700-xattention-model-s16-tau900-g32 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON \
  -DNINFER_R9700_KV_VALUE_GROUP=32 \
  -DNINFER_R9700_XATTENTION_QUALIFICATION=ON \
  -DNINFER_R9700_XATTENTION_STRIDE=16 \
  -DNINFER_R9700_XATTENTION_TAU_PERMILLE=900
cmake --build build-r9700-xattention-model-s16-tau900-g32 --parallel 8 \
  --target ninfer_bench ninfer_r9700_runtime_planner_qual
```

Run both capacity groups into new sparse directories rather than the completed dense directories:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-capacity \
  --bench build-r9700-xattention-model-s16-tau900/bench/ninfer_bench --no-build \
  --weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-kv-value-group 16 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile b128-s16-tau900 \
  --output-dir profiles/bench/pareto-capacity-xattention-s16-tau900-all-q4-g16-20260903

python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-capacity \
  --bench build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench --no-build \
  --weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-kv-value-group 32 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile b128-s16-tau900 \
  --output-dir profiles/bench/pareto-capacity-xattention-s16-tau900-all-q4-g32-20260903

python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-whole \
  --bench build-r9700-xattention-model-s16-tau900/bench/ninfer_bench --no-build \
  --weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-kv-value-group 16 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile b128-s16-tau900 \
  --output-dir profiles/bench/pareto-whole-xattention-s16-tau900-all-q4-g16-20260903

python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-whole \
  --bench build-r9700-xattention-model-s16-tau900-g32/bench/ninfer_bench --no-build \
  --weights out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-kv-value-group 32 \
  --expected-q4-activation-bits 8 --expected-w8-activation-bits 8 \
  --expected-fp8-qk-wmma 1 \
  --expected-xattention-profile b128-s16-tau900 \
  --output-dir profiles/bench/pareto-whole-xattention-s16-tau900-all-q4-g32-20260903
```

Each capacity command produces exactly four cells, one native-262,144-context MTP3 workspace
result for each C=1..4. On interruption, repeat the byte-identical command with `--resume`; the
schema-v14 manifest must match the artifact, executable bytes, cache group, sparse profile, preset,
and concurrency matrix before any valid cell is skipped. Raw schema-v20 reports bind
qualification, B128, S16, and tau900; schema-v14 manifests and flattened rows carry the expected
profile. The runner rejects dense/sparse mismatches on initial validation and resume. Never resume
these commands into the retained schema-v12 dense capacity directories.
The same no-overwrite rule applies to whole evidence. Keep each group's capacity and whole matrices
on byte-identical benchmark and artifact files. A whole matrix contains four optimized-head MTP3
timing commands and four matched ordinary greedy controls across C=1..4. The eight MTP3
fresh-request rows retain separately timed prefill/decode phases, speculative acceptance, and
whole-request makespan; all sixteen rows retain target tokens so assembly can require exact
MTP/ordinary parity per repetition and lane. The assembler uses only MTP3 timings for all 48 speed
objectives, so the older standalone phase matrices are diagnostic history and must not be scheduled
or supplied as selection evidence. Only after both manifests are complete and have no
`failures.json` may they enter the Pareto assembler.

`prepare_whole_profile.py` is optional attribution after an unprofiled whole matrix completes; it
is not a prerequisite or replacement for that timing. If a named bottleneck question remains,
prepare one measured point into a new profiler directory, for example G16 C4/32K:

```bash
python3 -m tools.bench.prepare_whole_profile \
  --matrix-dir profiles/bench/pareto-whole-xattention-s16-tau900-all-q4-g16-20260903 \
  --out profiles/rocprof/xattention-s16-tau900-all-q4-g16-c4-32k-trace-20260903 \
  --concurrency 4 --prompt-tokens 32768 --generated-tokens 256 \
  --expected-weights-id r9700-q4g64-n16k16-eval --expected-kv-value-group 16 \
  --expected-xattention-profile b128-s16-tau900 --expected-prefill-chunk 4096 \
  --kind trace --question "which dispatch family dominates XAttention C4 32K makespan?"
```

Preparation reopens the v13 manifest and v20 raw report, verifies the explicitly selected
XAttention profile and prefill chunk, artifact and executable bytes, group, plane layout,
concurrency, measured geometry, and `auto` timing provenance, then only writes a command plan.
Execute that profiler command later under the serialized GPU protocol.

For the long-context gate, start `ninfer-serve` from that same isolated build with
`--request-log-jsonl`. Its schema-v20 `server_start` record carries the selected cache group and identical compile-bound
profile. Durable NIAH must consume the one `terminal_production_selection` and pass its exact
server executable and artifact; validation derives the selected dense or B128-S16-tau900 profile
from that authority and rejects a mismatch before
issuing any request and then requires a fresh full-prefill `request_done` for every response.
The admission gate is exactly one run of the 64K start/q25/mid/q75/end ladder (five requests), with
the exact-format needle, `--model qwen3.8-27b`, and `--max-tokens 64`. The server must use
`--no-prefix-reuse`; otherwise the shared fixture stream can turn later cells into suffix prefills
that the durable validator correctly rejects. The 8K/200K ladders and complete 6x5 matrix are
optional broader or post-production-change coverage, not additional XAttention admission gates.

There is deliberately no public append-and-attend function, homogeneous cache view, cache-dtype
branch, or execution-envelope type for Text/MTP. The target schedule composes the checked cache
transaction with the target leaf.

## DFlash Full: native BF16 context attention

`include/ninfer/ops/bidirectional_gqa_attention.h` and
`src/ops/r9700/dflash/bidirectional_gqa_attention.{cpp,hip}` remain the native DFlash Full Op.
`GqaContextExecutionEnvelope` bounds launch resources for graph replay; device context lengths and
valid widths define the mathematical visible set. `BidirectionalGqaBF16ContextView` is a read-only
BF16 paged state that is intentionally independent of the asymmetric Text/MTP cache.

Every live query row sees the complete persistent context followed by every live query K/V row in
its batch row; there is no causal triangle. The Op does not mutate context or query inputs, writes
exact zero to inactive output tails, and owns no Text/MTP publication authority. Its retained GQA
name denotes the fixed `Hq32/Hkv8` head geometry and must not be confused with the deleted
homogeneous Text/MTP boundary.

## Qualification and performance admission

The maintained physical gates are:

- `tools/r9700/kv_op_qual.hip`: exact codec, address, append, compact, QK/PV, malformed-input, and
  layout cases;
- `tools/r9700/hip_persistent_state_qual.hip`: allocation, all-layer transaction, publication,
  compaction, and restore behavior;
- `tools/r9700/full_attention_leaf_qual.hip`: causal, packed-tree, device-active-row, graph replay,
  pending/committed-read, input-immutability, and FP64-oracle checks at `D256/Hq24/Hkv4`;
- `tools/r9700/bidirectional_gqa_attention_qual.hip`: DFlash Full BF16 context/query semantics at
  `D128/Hq32/Hkv8`.

A performance change is admitted at the smallest level that supports its claim. Raw codec or
attention candidates need their independent oracle and physical gfx1201 timing; a production
selection additionally needs target-leaf timing and, when its private arithmetic changes, paired
real-model quality evidence. Candidate timing alone cannot change the cache ABI or
production dispatch.

## Ownership constraints

- Do not restore the removed homogeneous Text/MTP Attention API, its retired launchers/kernels, or
  alternate cache-format branches.
- Do not expose raw R9700 KV pointers through the Engine or family public state.
- Do not let DFlash Full or Local state alias the Text/MTP cache or publication transaction.
- Do not create MHA/MQA/GQA class hierarchies, backend registries, arbitrary-mask interfaces, or
  model-key dispatch inside an Op.
- A new visibility rule or persistent representation requires its own closed semantic contract and
  direct qualification; a new tile or instruction schedule does not.
