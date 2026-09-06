# R9700 performance status

This document reports only measurements produced by the current native HIP/gfx1201 migration.
Measurements from the retired backend are not comparable and are not product evidence.

## Evidence levels

NInfer distinguishes four scopes:

1. raw kernel candidates;
2. complete semantic Ops;
3. runtime phases/rounds;
4. complete Engine inference at fixed concurrency.

A result is stated only at the scope directly measured. Kernel or Op timings do not establish
tokens per second. Final production selection requires a complete model artifact, numerical
quality guardrails, same-candidate graph/eager parity, resolved capacity, and end-to-end C=1..4
measurements on an otherwise idle R9700.

## Platform

The latest accepted measurements use:

| Component | Value |
|---|---|
| GPU | AMD Radeon AI PRO R9700 |
| architecture | `gfx1201`, wave32 |
| driver | `7.1.3.31500000` |
| kernel | `7.0.0-30-generic` |
| HIP | `7.15.26333` |
| compiler | AMD Clang 23 under `/opt/rocm/core-10.0` |
| build | Release, exact `gfx1201` code object |

The GPU must otherwise be idle. Timings use HIP events around the claimed physical work after
warmup. Independent exact/FP64 oracles run before performance values are admitted.

### Sustained memory bound

The native exact-device probe uses two 4 GiB buffers, 512 times larger than L2, and reports five
trials of at least 0.25 seconds. Its aggregate read checksum covers every workgroup and matched the
expected value exactly.

| Method | Best bus rate | Median bus rate | Advertised-peak fraction |
|---|---:|---:|---:|
| kernel uint4 read | `636.2 GB/s` | `636.0 GB/s` | `99.4%` |
| kernel uint4 write | `588.0 GB/s` | `587.9 GB/s` | `91.9%` |
| kernel uint4 copy | `549.0 GB/s` | `548.6 GB/s` | `85.8%` |
| HIP D2D copy | `543.8 GB/s` | `543.4 GB/s` | `85.0%` |

Copy bus rate counts both the read and write traffic. This is the hardware bandwidth bound, not a
model or individual-Op throughput claim.

## Typed Text/MTP cache and attention

The growing cache stores FP8 E4M3FN K, signed INT4 V, and FP16 V scales. A 32-point sweep covering
G16/G32, every K/V/scale plane order, T=1..8, and 1K/4K/8K/32K contexts passed the independent
layout and attention oracles after the coherent ROCm update.

The provisional latency leader is G16 with token-fastest K, feature-fastest V, and feature-fastest
V scales: mean normalized latency `1.004041`, mean rank `2.188`, and 14 wins. This is not yet the
permanent ABI: the retained real 8K comparison measured both G16 and G32 within their quality tier
and favored G16 on NLL error against one historical BF16 realization. The exact v3 BF16 authority
is now closed and the dense all-Q4 sidecars pass their offline rebase; current mixed and sparse
quality evidence, matched phase, and whole-inference speed evidence remain incomplete. The
pre-promotion C=1..4 capacity matrices below are retained, but split-512 promotion and final
prefill-chunk selection require a fresh 32-cell capacity rerun before selection.

The production leaf uses two measured crossovers. At context 8,192 and above, ordinary T=1 and
fixed-width T=4 select the three-stage split-512 leaf; T=4 includes causal, packed-tree, and
device-active-row panels. At shorter contexts, T=1 selects wave32 FP8-Q/FP8-K WMMA from context 64,
T=2 selects it from context 320, and remaining shapes retain represented-BF16-Q score streaming.
The short-context WMMA path writes one caller-owned FP32 score panel, applies stable FP32 softmax,
and keeps exact
FP32-probability INT4-V accumulation vectorized because the tested WMMA PV formulation rounded
information required by the represented contract. A separately compiled score-streaming control
isolates the private FP8-Q profile in PPL and other model-level comparisons. The complete 8K
decode comparison scores 4,095 aligned positions: WMMA PPL is `6.538098` versus `6.541706` for
streaming, mean signed ΔNLL is `-0.000552`, maximum absolute ΔNLL is `1.017672`, and 72 greedy
choices differ. Neither route creates a new NLL-at-least-10 position relative to the other. WMMA
reduces the complete scorer time from `882.169` to `718.316` seconds (`1.228x`) and measured within
the accuracy tier against that retained source-BF16 realization at mean ΔNLL `+0.011989` with three new
severe positions. This is private arithmetic, not an observable semantic cast: schedule comparison
therefore gates finite aligned NLL under an explicit campaign bound and retains greedy flips
diagnostically. The short-context crossover is retained; scorer wall time supports this isolated A/B
but does not replace the pending whole-inference benchmark objective. A
paired pre-promotion MTP3 probe over the same 63 positions was bit-identical in both NLL and
argmax sidecars (PPL `4.313991` in each build), when T=4 target verification still used score
streaming. The cell JSON, raw sidecars, and machine-readable comparison
are retained under `profiles/ppl/fp8-qk-wmma-crossover/` and
`profiles/ppl/fp8-qk-wmma-crossover-8k/`.

The retained split-512 schema-v2 admission report passed the independent stored-byte FP64 oracle,
invalid-input and workspace-boundary suites, fixed-address Device Graph replay, and an interleaved
20-sample timing matrix on the R9700 under `auto`. Across G16/G32, 8K/32K, ordinary T=1, and every
active prefix of fixed-width T=4, the complete incumbent/split ratio ranges from `6.323x` to
`17.646x`. The report SHA-256 is
`bb4f0f2fba4773ab7c16e1f9c2e746fa9105f19ab4073fdffaa810108f2e4f1e`. The native-context
caller-owned scratch is 37,847,040 bytes for T=1 and 151,388,160 bytes for T=4; it aliases across
sequential layers and request slots. Promotion introduces distinct captured graph topologies; an
MTP3 graph tracks its max+6 MTP-cache and max+4 Text T=4 leaves independently as each crosses 8K.
The 32 C=1..4 capacity cells therefore remain pending fresh post-promotion measurement even though
modeled Text-prefill scratch still dominates the global arena.

The production P128..4096 initial-prefix leaf now uses the physically selected three-stage
full-score GQA6 route. After the gated-RMSNorm, split-view SiLU, and K256 token8 RMSNorm
promotions, a matched all-Q4/G16 P2048 C1 run under `auto`, with speculative execution disabled,
measures `1,214.498278` prefill tok/s and `1.686293799 s` mean prefill with `0.001380049 s`
standard deviation across three repetitions. The retained report is
`profiles/bench/prefill-p2048-post-k256-rmsnorm-20260904.json`, SHA-256
`fe0598f603b9a3d3e5d25475b27c39da897ed0b4069566495053f04c1246b6b4`. This is retained as a
historical low-context baseline and is superseded by the scalar-base production result below; it
did not satisfy the explicit P2048 floor of
`2,000` prefill tok/s (`1.024 s`). A matched current mixed-Q4/W8 diagnostic, using A8 activation
coding, C1, chunk 4,096, and no speculative execution, is slower at `920.8914604` prefill tok/s
and `2.223935366 s`. Its report is
`profiles/bench/prefill-p2048-mixed-q4-w8-current-20260904.json`, SHA-256
`590f3e226d01a11269b79b959f7daf3800b3d392f7c6c8cea76c974bc1ec76df`. It is evidence that
the current mixed recipe is not a shortcut to the dense P2048 gate, not a replacement for the
all-Q4 authority.

The retained post-full-score trace at
`profiles/rocprof/diagnostic-p2048-post-fullscore-causal-20260904/trace_kernel_stats.csv`
(SHA-256 `afba65658aea8dd344b8c81b17730c8e35402b61868d7a1f39c5733e0a3e0c88`) captured both
one warmup and one measured pass, so the production Q4 CTA calls and durations are two-pass totals
rather than the decision attribution. The unique `ninfer_bench_measured` marker instead retains
2,249 wholly contained kernels with no boundary crossing. Its selected-region report is
`profiles/rocprof/diagnostic-p2048-post-fullscore-causal-20260904/measured-selected-region-attribution.json`,
SHA-256 `3993cb56bdc620f13e9ba223463f041120104d28cf91293dbe491d3bd3d976c7`.
Independent kernel service is 1,729.283668 ms, with a 1,719.161926 ms timestamp union inside
1,753.456094 ms of marker wall. The remaining 34.294168 ms is kernel-inactive marker wall, not
proven GPU idle because runtime/API activity was not captured.

On that measured-only basis, the production Q4 M64xN128 CTA accounts for 320 calls and 64.5355% of
kernel service, ordinary GDN recurrence 48 calls and 8.4243%, dense full-score PV 16 calls and
5.0115%, dense full-score QK 16 calls and 4.8159%, the then-generic split-view SiLU 64 calls and
4.1999%, the then-incumbent gated RMSNorm 48 calls and 3.7309%, activation quantization 321 calls
and 2.0451%, and the then-incumbent K256 RMSNorm 32 calls and 1.4637%. The global trace's 644 Q4
WMMA32 calls are not ordinary two-pass P2048 work: 640 layer calls belong to the startup T=1
ordinary code-warm execution and Device Graph capture, while four T=1 vocabulary-head calls cover
those two startup executions plus warmup and measured prefill. Only the measured vocabulary head,
at `2.585003 ms` or 0.1495% of measured kernel service, belongs to the measured P2048 pass; startup
service is excluded from steady
prefill attribution. Because the trace predates the gated-RMSNorm, split-view SiLU, and K256
RMSNorm promotions visible in the current executable, those rows explain the completed changes but
do not establish current residual timings.

The initial prefill-chunk screen then found a metadata boundary missed by the admission fixture. A
T=1 attention call at visible frontier 8,192 carried a device active-row pointer; the former
row/context-only decision selected split-512, whose admitted contract rejects active-row metadata
for T=1, and returned `hipErrorInvalidValue`. Workspace sizing and launch now use the same
metadata-aware selector. Metadata-bearing T=1 retains the fused leaf, ordinary T=1 remains split at
8K and above, and fixed-width T=4 retains its admitted causal/tree/device-active-row split forms.
The host routing test passes the exact 8,191/8,192 selector and workspace boundaries, and the direct
8K MTP reproduction completes in
`profiles/bench/prefill-split512-failure-diagnostic-20260904.json`. This is regression evidence, not
a prefill-chunk selection or a terminal throughput result.

The September 3, 2026 XAttention schema-v1 measurements are superseded diagnostics, not production
rejection evidence. Their zero Q/K corpus made the estimator uniform and therefore forced
`tau=0.900` to retain about 90% of pages; those builds also lacked retained executable/source
identity and had regressed from paper B128 selection to B64. The corrected candidate uses B128
query/keep blocks expanded to the B64 cache, counts mandatory causal blocks inside the threshold
budget, includes ragged estimator planes/groups, and uses a concentrated represented timing
fixture with explicit achieved keep fraction. The current S16 `tau=0.900` candidate uses a
B16-query sparse FlashAttention-style consumer: native BF16 WMMA forms B16xB16 QK tiles over the
ranker's packed logical keys, FP32 online softmax preserves increasing retained-key order, and
each direct INT4-V/FP16-scale load is reused across 16 queries without a global score workspace.
Separate G16/G32 physical qualifiers passed the D256/Hq24/Hkv4 independent FP64 oracle at maximum
absolute errors `1.6023e-8` and `1.6986e-8`, respectively, under the recorded `3e-4`
absolute-or-relative tolerance. G16 measured `2.4200/8.8416 ms` sparse versus
`69.1062/286.5128 ms` dense at 8K/32K (`28.56x/32.40x`), with `1.2393/4.0885 ms` consumer time.
G32 measured `2.4008/8.7894 ms` versus `69.1753/287.6155 ms` (`28.81x/32.72x`), with
`1.2454/4.0854 ms` consumer time. Both retained 14.06%/11.72% of pages. Schema-v4 reports have
SHA-256 `697d26c1cb24a0cd059e92336a1f3494dac6cdc65a4c74aa80fef7af8d290519` (G16) and
`dfcf0cd17801a83c97d978203e4763c089afc47e5bd15afe371ccceda1d6850e` (G32), with executable and
source hashes beside them under
`profiles/bench/r9700-xattention-b16-requal-s16-tau900-g{16,32}/`. These concentrated-fixture
results qualify the redesigned operator, not whole-model performance. Static-profile selection,
the selected-profile retrieval gate, production-prefill chunk selection, and fresh whole-inference
evidence remain open; no end-to-end admission follows from this isolated speedup.

A diagnostic selected-region trace of the compile-isolated S16/`tau=0.900` route at C=1,
32K context, and a 4,096-token prefill chunk is retained under
`profiles/rocprof/xattention-s16-tau900-all-q4-g16-c1-32k-trace-20260904/`. The benchmark
report SHA-256 is
`a22837755d61d295ed00e160a0ee272b96c6bff87e90121ef9d11803f21a02d9`; the rocprof database
SHA-256 is `83a42f5ce11d73f03354cf5ec2872b29fa8e1d53a45ae537d96f519be40357ae`.
The derived schema-v1 attribution report is
`profiles/rocprof/xattention-s16-tau900-all-q4-g16-c1-32k-trace-20260904/prefill-attribution.json`
with SHA-256 `81a9efbb0115775d102203441849dabc8aebe699d8368b5f124fb8bf3e6cd6b4`.
Its eight Text-prefill chunk ranges total `444.416 s`, matching the benchmark's `444.418 s`
prefill measurement within `2.312 ms`. Kernel activity covers `444.068 s` of their union,
leaving only `0.348 s` (`0.078%`) as host/idle gap. Independently summed kernel time is
`443.849 s` in base Text prefill versus `0.318 s` in MTP-prefill; concurrent streams make
those category totals non-additive. Base full attention accounts for `325.124 s`, post-mixer
for `92.908 s`, and GDN for `25.817 s`. Within attention, the former serial sparse consumer is the
dominant kernel at `302.770 s` over 128 dispatches (`68.13%` of prefill); the prefill-only
A8Q4 linear totals `116.717 s`, and the ranker totals `15.353 s`. This rules out the
MTP-prefill or host-gap explanation.

The runtime/qualifier comparison must normalize query work. The 32K qualifier's
`32.450 ms` consumer result covers one 128-row query block, while the runtime chunk contains
32 such blocks: its comparable qualifier baseline is therefore `1.038 s` per layer. The final
runtime chunk measures `4.146 s` per full-attention layer, about `3.99x` that baseline. If
consumer time scaled only with kept pages, applying that ratio to the qualifier's `11.71875%`
keep fraction would suggest roughly `46.8%` retention on this model input. That value is an
inference, not an observed retention measurement, because fixed costs and the real keep
distribution can also change scaling. The existing C=1 whole row and this trace remain
diagnostic that motivated the B16-query redesign above. It does not predict that redesign's real
model keep distribution or end-to-end speed. The production prefill chunk and fresh whole matrices
must still be selected and measured before admission or whole-inference performance claims.

Selected complete target-leaf timings at context 257 are:

| Query rows | Time |
|---:|---:|
| 1, selected WMMA | about `0.104 ms` (`0.191 ms` streaming control) |
| 2, selected streaming | about `0.195 ms` (`0.196 ms` streaming control) |
| 9 | about `0.204 ms` |
| 17 | about `0.442 ms` |
| 128 | about `2.504 ms` |

At 4K context the focused raw-candidate trace attributes roughly `7.7 us` to parallel softmax and
`818.8 us` to exact PV. QK/softmax/PV metadata is 24/23/15 VGPR, 0/76/0 bytes
LDS, zero private scratch, wave32, occupancy 16.

The serialized boundary sweep measured T1 streaming/WMMA medians of `0.0376/0.0431 ms` at
context 48, `0.0427/0.0443 ms` at 56, and `0.0479/0.0460 ms` at 64. T2 medians were
`0.1696/0.1724 ms` at context 256, `0.1897/0.1851 ms` at 288, and `0.2099/0.1980 ms` at 320;
the conservative T2 classifier starts at 320. At 4K, WMMA is `2.998x` and `1.499x` faster for
T1/T2, while T3 remains a measurement tie and stays on streaming. The represented-input FP64
oracle, classifier, and timing record is retained in
`profiles/bench/r9700-fp8-qk-wmma-crossover.json`.

Production-shaped suffix append plus selected attention was measured independently at T=1..8 and
T=9/17/128 over 1K/4K/8K/32K. Append costs about 5--8 microseconds at narrow decode and 37--46
microseconds at T=128. Even an impossible zero-cost append bounds improvement to 0.80/0.20/0.10/
0.03 percent at 1K/4K/8K/32K for decode and 0.40/0.10/0.05/0.01 percent at T=128. Ordered-pair
timings showed no repeatable gain beyond clock variance.

One ordinary HIP kernel cannot globally order the independent T x Hkv append writers before the
T x Hq attention readers. A cooperative launch cannot keep the T=128 grid resident, while
reader-local encoding duplicates the GQA codec work sixfold and violates the transaction's
single-writer/status ownership. Production therefore retains separate ordered append followed by
the already fused QK, online FP32 softmax, and exact PV attention Op. Append/attention metadata is
13/23 VGPR, zero/52 bytes LDS, zero scratch, and occupancy 16.

## Linear

Standalone real-shape `[N,K]=[7168,5120]` prefill results:

| Represented weights | T16 | T32 | T64 | T128 |
|---|---:|---:|---:|---:|
| BF16 wave32 WMMA | `0.203 ms` | `0.267 ms` | `0.698 ms` | `2.418 ms` |
| provisional W8G32 | `1.119 ms` | `2.225 ms` | `4.371 ms` | `8.751 ms` |

Both routes pass direct independent FP64 formulas. W8G32 uses exact signed codes and FP16 scales
with FP32 FMA because no available WMMA form preserves that codec. These measurements qualify the
Op; they do not select the final model weight recipe.

The compile-time W8-A8 evaluator adds a native signed-INT8 WMMA route while preserving exact
represented-BF16 execution below measured per-shape crossovers. Across the mixed artifact's 13
unique W8 shapes, A8 begins at T3 or T4 for nine shapes, T32 for `[34816,5120]`, and T64 for the
three 1152-row Vision shapes; unknown shapes remain exact. At `[7168,5120]`, interleaved medians
show 1.32x at T3, 1.53--1.75x at T4--8, and 3.37--5.55x at T16--128. These are operator results;
matched real-model quality and whole-inference results are still required before selection.

The persistent Q4G64 evaluator additionally has separately compiled A4G64 and A8G64 activation
profiles over identical artifact bytes. A8 is the production-style compile default; A4 requires
an explicit evaluator build. A8 encodes a signed code into unsigned-low and signed-high
nibble planes and uses two native IU4 WMMA sign modes with exact I32 recombination. Both profiles
pass exact activation images and every independent FP64-formula output at `[7168,5120]`,
T=1..8/16/32/64/128; four active lanes rotate across groups to cover all 64 packed K-lane/WMMA
fragment positions. The timing harness measures five alternating forward/reverse route rounds,
retains every sample, and reports the per-route median. The interleaved complete-Op medians are:

| Q4 activation profile | T1 | T8 | T16 | T32 | T64 | T128 |
|---|---:|---:|---:|---:|---:|---:|
| A4G64 | `0.05356 ms` | `0.07826 ms` | `0.09961 ms` | `0.11374 ms` | `0.13436 ms` | `0.24254 ms` |
| A8G64 two-plane | `0.05874 ms` | `0.08699 ms` | `0.11070 ms` | `0.12538 ms` | `0.15366 ms` | `0.30146 ms` |

A8 reduces the represented-BF16-times-decoded-Q4 maximum error from 0.212891 to 0.015625. Its
WMMA kernel uses 64 VGPR, zero LDS/scratch, occupancy 16; activation quantization uses 10 VGPR,
zero LDS/scratch. The machine-readable result is
`profiles/bench/r9700-a8q4g64-linear-qualification.json`; direct compile-selected A4/A8
Tensor/WorkspaceArena results are retained beside it as
`r9700-q4-tensor-dispatch-a4.json` and `r9700-q4-tensor-dispatch-a8.json`. Nonzero quantizer status
is consumed asynchronously by both routes and poisons every output with exact BF16 NaN.

The selected wave32 quantizer was also measured across all 17 distinct all-Q4 artifact shapes,
all six mixed-artifact Q4 shapes, and all 32 Q4 DFlash2 matrices at T=1..8/32/128. Five additional
DFlash shapes extend the inventory to 22 unique shapes and 220 extents, including the K=25600
feature projection. Every extent matched the exact codec and independent sampled FP64 formula
with zero BF16-step error. Count-weighted timing is effectively tied at the narrowest points:
the selected route ranges from a 0.007% regression to a 2.034% gain for all-Q4, a 0.506%
regression to a 1.401% gain for the mixed Q4 subset, and a 0.048% regression to a 1.436% gain for
the DFlash matrices. The complete inventory, seven-trial interleaved samples, and medians are
retained in `profiles/bench/r9700-a8q4g64-artifact-shape-sweep.json`. Physical public-wrapper
checks also pass Q4 through grouped dynamic-convolution preparation and both selector chain/tree
routes using their caller-owned nested Linear arena.

The earlier matched real-model 8K prefill scoring isolates activation width over identical artifact bytes.
Every row scores the same 4,095 index-aligned tokens after a 4,096-token skip against the
source-BF16 reference (PPL 6.460181); raw FP32 NLL and I32 argmax sidecars are retained.

| Stored matrix recipe | Q4 activation | PPL | mean-NLL delta | BF16-greedy flips | new NLL>=10 positions | score seconds |
|---|---:|---:|---:|---:|---:|---:|
| all Q4G64 | A4G64 | 7.397805 | +0.135526 | 800 | not separately attributed | 69.394 |
| all Q4G64 | A8G64 | 6.720524 | +0.039509 | 450 | 9 | 71.601 |
| 183 Q4G64 / 256 W8G32 | A4G64 | 6.765289 | +0.046148 | 420 | not separately attributed | 263.145 |
| 183 Q4G64 / 256 W8G32 | A8G64 | 6.550048 | +0.013815 | 237 | 2 | 264.720 |
| source-MSE 183 Q4G64 / 256 W8G32 | A8G64 | 6.544746 | +0.013005 | 234 | 3 | 264.980 |
| source-MSE 183 Q4G64 / 256 W8G32 | A8G64 Q4 + adaptive A8G32 W8 | 6.538677 | +0.012077 | 234 | 3 | 133.504 |

A8 is decisively useful: it reduces PPL by 0.677281 for all-Q4 and 0.215241 for the mixed artifact
at only 2.207 and 1.575 seconds additional score time in these single runs. The source-MSE mixed
adaptive-W8 profile is the Q4-containing accuracy leader in that retained comparison: +0.012077
mean NLL and three new severe positions measured within the accuracy tier's 0.02/five-position
limits. Its represented-BF16 W8 control remains retained at +0.013005. The all-Q4+A8 profile also
measured within the declared capacity-speed tier: +0.039509 is below `ln(1.05)` and nine new severe
positions are below its eleven-position limit in that realization. At that point both remained
capacity candidates with BF16-derived quality admission open; BF16-greedy choices are diagnostic.
The adaptive W8 route is selected within the mixed profile: versus the identical stored artifact
with represented-BF16 W8 activations, it improves mean NLL by 0.000928 and reduces this scorer's
wall time by 1.98x. Scorer wall time is useful for this isolated A/B but is not substituted for the
pending whole-inference Pareto matrix.
Complete comparison data is retained at `profiles/ppl/8k-candidate-comparison.json` and
`profiles/ppl/8k-candidate-comparison.md`; raw A8 results are under `profiles/ppl/q4g64-a8-real/`,
`profiles/ppl/q4-w8-a8-real/`, and `profiles/ppl/q4-w8-mse-a8-real/`.

The final compile profile (A8 for Q4, adaptive A8 for W8, and the selected finite FP8-Q/K WMMA
classifier) now has matched G16/G32 results at both 8K and 32K. Every FP32-NLL and I32-argmax
sidecar is finite, complete, index-aligned with the independent BF16 source, and retained with its
SHA-256:

| matrix recipe | cache | length | scored | PPL | mean-NLL delta | BF16-greedy flips | new severe / tier budget | quality tier |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| all Q4G64 + A8G64 | G16 | 8K | 4,095 | 6.720524 | +0.039509 | 450 | 9 / 11 | capacity-speed, one realization |
| all Q4G64 + A8G64 | G32 | 8K | 4,095 | 6.730933 | +0.041056 | 445 | 10 / 11 | capacity-speed, one realization |
| all Q4G64 + A8G64 | G16 | 32K | 16,383 | 5.892102 | +0.044561 | 1,673 | 41 / 41 | capacity-speed, one realization |
| all Q4G64 + A8G64 | G32 | 32K | 16,383 | 5.900777 | +0.046032 | 1,682 | 39 / 41 | capacity-speed, one realization |
| source-MSE Q4G64/W8G32 + adaptive A8 | G16 | 8K | 4,095 | 6.538677 | +0.012077 | 234 | 3 / 5 | accuracy, one realization |
| source-MSE Q4G64/W8G32 + adaptive A8 | G32 | 8K | 4,095 | 6.542475 | +0.012658 | 232 | 2 / 5 | accuracy, one realization |
| source-MSE Q4G64/W8G32 + adaptive A8 | G16 | 32K | 16,383 | 5.724998 | +0.015790 | 956 | 16 / 17 | accuracy, one realization |
| source-MSE Q4G64/W8G32 + adaptive A8 | G32 | 32K | 16,383 | 5.723254 | +0.015485 | 942 | 16 / 17 | accuracy, one realization |

Against that retained BF16 realization, all-Q4 measured outside the stricter accuracy tier but
within its declared capacity-speed tier in every cell; 32K G16 is exactly at that realization's
new-severe budget. The mixed source-MSE recipe measured within the accuracy tier in every cell,
with one new-severe position of margin at 32K. Direct G16/G32 NLL
differences are diagnostic and do not choose a group: three of four are within two paired standard
errors, while all-Q4 at 32K favors G16 by 0.001471 mean NLL (2.18 standard errors). Under the
current C=1..4 product cap, both recipes and both cache groups remain capacity candidates.

The table above is retained historical one-realization evidence, not current admission. The old
source scorer was nondeterministic at both lengths; long-context tracing localized its first
divergence to attention PV, and a later allocator-sensitive GDN route produced two stable 8K
branches. The replacement v3 authority uses fixed-order PV and a single fused-recurrent GDN route,
and binds the complete scorer/backend/environment boundary. Two fresh non-trace campaigns are
byte-exact at both lengths. Their comparison is retained at
`profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json` (SHA-256
`9b13916a2a3ca8b8e5d01352f98a8306414f4e13a79989c19b1d1248d5eb00de`): 8K NLL/argmax SHA-256
are `6ba4009ac23da9c831b475ce38fa98c377fd504ade81733de711be95e4092c0e` /
`c9904bd5b09d00666aee8500023cb52cc6f22d5c369cc00ceba57ea04a6d80c1`, and 32K are
`f5da949265ff566ebc368e3521cadc3f2ea7c687f1072ec993dbaec101ab1ab4` /
`ec4e4d6b731d837f98b5f8bac7161ddacb9e55f0cf1842d68b58b9a2957a314b`.

The unchanged dense all-Q4 candidate sidecars were rebound offline to that exact authority in
`profiles/ppl/xattention-dense-q4g64-v3-rebase-20260904/results.json` (SHA-256
`fc05174d70421f60351c0cd87c587d04f74eabd205e0d9431887a60c7c7bc4fa`). G16/G32 pass the
capacity-speed tier: 8K mean-NLL deltas are `+0.0397932579` / `+0.0413409097` with 10/11 new
severe positions of an 11-position budget; 32K deltas are `+0.0447215401` / `+0.0461926079` with
40/38 of 41. This closes only dense all-Q4 quality admission. The retained sparse sidecars require
fresh scoring after the B16 consumer rebuild. The mixed-dense historical scorer binaries were overwritten,
so its candidate provenance cannot be reopened for a valid offline rebase; mixed dense and sparse
both require fresh acquisition using the selected prefill chunk.

The historical mixed dense schema-v6 campaign is fixed candidate-sidecar evidence at
`profiles/ppl/xattention-dense-q4-w8-mse-20260903/results.json` (SHA-256
`7530b27e503baf8bf29c2fd83135c0e2395a9a04bd050a2872dd2a7b341b9220`). It binds mixed artifact
SHA-256 `8fbadf14e355b1943ef9386a91ebafff0d852295a9adc0054bdd430291505ebd` and dense G16/G32 scorer
SHA-256 `fb36b2fca7871fbcbdbbce32cb1da8d05965e2164fea0d2773dd8259d686b6d4` and
`552e73200a351691d5451adccfa7dd91e9381631388e366d3c03c051338a967d`. Against its shared old BF16
realization, G16/G32 measured +0.011503/+0.012084 mean NLL with 4/3 new severe positions of a
five-position budget at 8K and +0.015520/+0.015216 with 14/14 of 17 at 32K. These provisional
pass flags are diagnostic only: the exact retained scorer bytes are no longer present, so the
campaign cannot be rebound to the v3 authority and does not close current mixed quality admission.

The retained pre-split-promotion all-Q4 schema-v20 reports under schema-v13 manifests record exact
C=1..4 capacity for dense and B128/S16/tau900 execution at the 4,096-token campaign-control chunk:

| execution | group | C1 | C2 | C3 | C4 | binding at C1/C2; C3/C4 |
|---|---:|---:|---:|---:|---:|---|
| dense | G16 | 262,144 | 524,288 | 570,304 | 558,080 | model context; device memory |
| dense | G32 | 262,144 | 524,288 | 593,152 | 580,416 | model context; device memory |
| B128/S16/tau900 | G16 | 262,144 | 524,288 | 553,280 | 541,056 | model context; device memory |
| B128/S16/tau900 | G32 | 262,144 | 524,288 | 575,424 | 562,688 | model context; device memory |

All sixteen historical cells are uncensored resolved effective maxima and exactly reproduce the
corresponding C1..4 rows of their superseded C1..8 campaigns. The four manifests bind all-Q4 artifact
SHA-256 `19d029a89c1ef1cf87420067555021a7c7b435c31a92bea7c64ccf42c03d80e9` and benchmark
SHA-256 values `78b4f3ba85436badaaccb4695848406d4d8f5d4f4f1439235d8d3200758490fb` (dense G16),
`5351c7c8fc4c7cd8fc136c2e6ab0107c65201d3fc0c4ab9fa4000f07e7587729` (dense G32),
`ec85dfe59aa9fc8ca369f4ffb785bd33fda5959d2ac5889010643a8a01e4e775` (sparse G16), and
`5780de5e08b5317c54e2212f4d3088dfeaad8bde290079c8adb42565b5e28e02` (sparse G32).
The corresponding manifest SHA-256 values are
`03265de925aa4be0b7a0584b60d819b80b3a1d8053e2cd0fa70eed37d104d63c`,
`cb38eea0a67ceb9875aa04df5ea5c5be0c0f13881a8adf022c05ae86ee13460f`,
`b75f8f4cf08c1ae28e7c51cef8b3ce11ad6eedf7605901a3b11798d50697a856`, and
`74d448c2b2869dd3c37a33a26d7ce8e3f294acf221a7e934a6800b19437ee459`
in the same dense-G16, dense-G32, sparse-G16, sparse-G32 order.
The corresponding retained pre-promotion mixed schema-v20/schema-v13 matrices record dense and
B128/S16/tau900 G16/G32 capacity at the same 4,096-token campaign-control chunk:

| execution | group | C1 | C2 | C3 | C4 | binding at C1; C2--C4 |
|---|---:|---:|---:|---:|---:|---|
| dense | G16 | 262,144 | 314,112 | 301,888 | 289,664 | model context; device memory |
| dense | G32 | 262,144 | 326,656 | 313,984 | 301,248 | model context; device memory |
| B128/S16/tau900 | G16 | 262,144 | 297,088 | 284,800 | 272,576 | model context; device memory |
| B128/S16/tau900 | G32 | 262,144 | 308,928 | 296,192 | 283,520 | model context; device memory |

Both dense manifests bind `r9700-q4-w8-mse-n16k16-eval` artifact SHA-256
`8fbadf14e355b1943ef9386a91ebafff0d852295a9adc0054bdd430291505ebd`.
G16 binds benchmark SHA-256
`78b4f3ba85436badaaccb4695848406d4d8f5d4f4f1439235d8d3200758490fb` and manifest
SHA-256 `cf87ab0c3cfd2e574c0c114652658973b53785af8b34026a077ad8bf5cbbed58`;
G32 binds benchmark SHA-256
`5351c7c8fc4c7cd8fc136c2e6ab0107c65201d3fc0c4ab9fa4000f07e7587729` and manifest
SHA-256 `93bcb29aaf2aceef9558cfd5221cb49b32688e691dd194c11f4bcf990a24995c`.
Sparse G16 binds B128/S16/tau900 benchmark SHA-256
`ec85dfe59aa9fc8ca369f4ffb785bd33fda5959d2ac5889010643a8a01e4e775` and manifest SHA-256
`c4229e99a882c51158fe433a43cb44bc17b0c20c88cbaf352ecae178aec160b0`.
Sparse G32 binds B128/S16/tau900 benchmark SHA-256
`5780de5e08b5317c54e2212f4d3088dfeaad8bde290079c8adb42565b5e28e02` and manifest SHA-256
`f12dfac84dc0d005e51e95142862dce1cc5abd020aeb7f98e1e354c76f937b03`.
All sixteen mixed cells, and all 32 retained capacity cells across both recipes, were uncensored
resolved effective maxima for their bound pre-promotion executables. The mixed dense cells exactly
reproduce the corresponding historical C1..4 rows. The post-promotion capacity rerun, every
whole-inference matrix, and terminal selection remain open.

The superseded mixed-recipe campaign's C=1..4 rows likewise resolved G16 to 262,144, 314,112,
301,888, and 289,664 tokens and G32 to 262,144, 326,656, 313,984, and 301,248 tokens.
These values establish why the old C7/C8 exclusion no longer applies. Those old mixed manifests
remain superseded history; the exact C=1..4 matrices above are retained controls and cannot become
current selection inputs until rerun against the promoted executable and selected chunk.

The superseded all-Q4 schema-v12 manifests are
`profiles/bench/pareto-capacity-markerfree-layout-all-q4-g16-20260903/manifest.json` (SHA-256
`be1d5727948512560f28a7e149547714f9b80dca96169646a2eef0310a883feb`) and
`profiles/bench/pareto-capacity-markerfree-layout-all-q4-g32-20260903/manifest.json` (SHA-256
`c7e5020b03cd104440011fcc17095e3166defdc59045274192e6b930b027d922`). Both bind artifact
`r9700-q4g64-n16k16-eval`, 15,172,829,184 bytes, SHA-256
`19d029a89c1ef1cf87420067555021a7c7b435c31a92bea7c64ccf42c03d80e9`. G16 binds benchmark
SHA-256 `86763d6d3ac2ff8fc27a3815f3816aaba44a11acfaf4ebf54ca5d8991ddc7d65`; G32 binds
`2c9bb5a64f0e25a307f3f2b783a527ed6b42d1671c6e826d7ca98545052e9b31`. These executables leave
nested ROCTX ranges disabled in ordinary measurement. The prior `pareto-capacity-layout-*`
campaigns, the completed G16 and interrupted G32 `pareto-capacity-unprofiled-layout-*`
intermediate campaigns, and the schema-v11/schema-v18 `pareto-capacity-max-*` campaigns are all
superseded for active selection. Their raw files remain unchanged as historical evidence. The durable
exact comparison is `profiles/ppl/q4-a8-final-8k-32k-quality-comparison.json` with the concise
companion `profiles/ppl/q4-a8-final-8k-32k-quality-comparison.md`. Matched whole-inference
performance remains required to select a recipe, cache group, and execution profile.

Cache group and plane order are runtime-state build profiles rather than artifact metadata. G16
and G32 comparisons therefore use separately compiled evaluator/Engine binaries; the same explicit
artifact is used for both when the experiment holds the weight recipe constant.

## DFlash2 and Vision

DFlash recipe-independent scheduling work and synthetic exact-shape operator qualification are now
active after the terminal dense C1/P2048 prefill rejection; they do not wait for the terminal base
or chunk. Physical companion, acceptance, capacity,
and whole-inference campaigns begin only after schema-v7 `terminal_production_selection` fixes the
base weight recipe, cache group, and execution profile and the receipt-bound shared chunk is
selected. The companion is then produced directly from the real BF16 DFlash2 checkpoint; it does
not automatically inherit the currently implemented all-Q4 evaluator recipe. Recipe selection
first compares canonical Q4G64,
source-MSE-refined Q4G64, and source-MSE-refined W8G32. Row-scaled E4M3 FP8 is conditional on
exact-shape R9700 speed and DFlash quality evidence because it adds storage and currently lacks
DFlash-owned prepared Linear execution. Every recipe keeps both selector codebooks and the private
DFlash state in model-specified BF16.

The active proposal set contains exactly K4/W5 and K5/W6. K is the number of predicted drafts and
W includes the target anchor, so these are respectively four predictions verified in a width-five
one-block chain and five predictions verified in a width-six one-block chain. The former K1..11
C=1 shortlist and its derived broad frontier/capacity campaign are superseded; do not run or
regenerate them. After a short recipe screen, each surviving recipe/K pair receives exact
ordinary-output parity, proposal determinism, generated-quality, acceptance, whole-throughput, and
C=1..4 capacity evidence. Missing capacity excludes only the exact recipe/K pair when its failed
command and logs are retained. Matched 8K/32K DFlash decode and retained-token fresh-prompt whole
inference use exact spec-none ordinary controls at C=1..4.

The replacement decision authority must bind the base decision through each conversion report to
the exact terminal winner, BF16 DFlash source, DFlash matrix recipe, companion artifact,
executable, cache group, and K/W. It must reopen every retained report and recompute recipe
eligibility, exact ordinary-output parity, exact repeated proposal/target determinism, and
generated-quality evidence. The current schema-v3 assembler and selected-DFlash preparation are
fixed-Q4/K1..11 implementations and are not valid authorities for this replacement campaign until
regenerated consistently. Whole parity covers the complete requested generation
including its first output token; isolated decode parity covers all 257 post-seed outputs, with seed
equality inherited from deterministic whole-route parity. Before frontier ranking, every K/W must provide at least
`1.02x` raw-mean speedup and a strictly positive two-standard-deviation conservative speedup over
its matching ordinary route in every 8K/32K C=1..4 whole and decode cell. Capacity, acceptance,
quality, and speed remain distinct gates. Its retained frontier treats every whole-throughput,
capacity, and acceptance cell as a separate maximize objective. The static winner maximizes the
worst normalized 8K/32K C=1..4 whole-throughput ratio, then worst normalized C=1..4 capacity,
then worst normalized matched acceptance length. Numeric `(K, W)` resolves only a complete tie.
No average or workload weighting may replace this rule. The DFlash gate closes only when this
record passes; a shortlist row, capacity summary, or manually chosen frontier member is not a
selection result.

The first kernel hypothesis after prefill closes is a packed-W4 small-T route. Current A8Q4 WMMA
uses a 16-token tile, so C1 K4/K5 proposal/head calls and W5/W6 DFlash/target calls leave most token
lanes inactive. The retained complete 39-cell C1 screen admits only N34816/K5120 at T4/T5/T6:
balanced-median candidate/incumbent ratios are `0.839747/0.879463/0.788788`, with conservative
paired-ratio uppers `0.855786/0.895273/0.801884`; all other 36 shape/width cells remain on WMMA.
The summary SHA-256 is `27cba659...b122`, independent review passed, and routing remains false.
The SHA-bound flattened screen then admits T8/T10/T12/T18/T20 at balanced-median ratios
`0.821087/0.889587/0.884984/0.883982/0.926681` and rejects T15/T16/T24 at
`1.139511/1.143556/1.041482`; its independently reproduced summary SHA-256 is
`bc8f72b5...4feb`. Combined with C1, only T=`4,5,6,8,10,12,18,20` at N34816/K5120 are eligible.
An off-by-default compile-selected candidate profile routes exactly those cells through packed dot8
for matched whole-DFlash A/B while the control build and all rejected/unlisted cells retain WMMA.
The selector changes no caller-owned workspace or graph-stable addresses and is not production
promotion. The current N16 four-role canonical-Q4 control companion is materialized at
`out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer`, 22,763,026,944
bytes with SHA-256 `d8fc77c36cf17c92e96d67b9a6b5a1826a1ade4f59d59c003b2368fe981fc512`;
its conversion report has SHA-256
`fb657164b9a9dc2987542ca4578b2bf75d79092a3efa8b7b1776520b4959f61f`. The matched control and
candidate benchmark receipt is
`profiles/bench/r9700-dflash-small-t-n16-matched-builds-20260906/build-receipt.json` (SHA-256
`cedb40ad2a995dceb339d81f530eddda17629bed211ae83033252e923a81b8ec`); it binds source commit
`cd966d72ed18e1b5b7b57b664572b3c8aa1e02ce`, including the DFlash fixed-family Device Graph
allowance correction from 40 MiB to 42 MiB (68 MiB total for the observed C1/K4/W5 topology),
control binary SHA-256
`04122c7da1e262f0fc0ba5a3a2065b1e1750dadd64df3337038b54c3d85a1f75`, and candidate binary
SHA-256 `6dbb09fcc75185ee1a09c0261c1a2dbbccbaa29021eb6f303f883b70350f469f`; normalized compile
commands differ only in the compile-selected candidate. These are evaluation inputs, not a
production recipe or routing decision, and the matched whole-DFlash A/B remains open.

The third bounded whole-A/B attempt cleared the graph-residency startup failure but stopped at
functional qualification. Within each K4/W5 and K5/W6 role, both candidate runs are token-exact
with both control runs, and every report is exact across its three repetitions. The fresh-prompt
whole output is also token-exact with the fresh exact ordinary control. The isolated post-seed
decode instead first differs from ordinary at output index 55 and then follows a different tail.
This does not identify the small-T selector as the cause: the same isolated sequence is produced
by candidate and control. Candidate timing from this attempt is inadmissible because the required
isolated ordinary-parity gate did not pass. There is no speed, winner, or routing conclusion from
attempt 3. The follow-up retained under
`profiles/bench/r9700-dflash-p129-isolation-discriminator-20260906/results`, with result closure
SHA-256 `fa8a7b8d7def4c59fc17370947373979ce920ef35fd1ca68f5315ad125267c56`, makes all four
eager/graph ordinary/DFlash P128 seed authorities exact at token 24178 and reproduces each eager
sequence exactly under Device Graph. Ordinary and DFlash append-versus-fresh execution both first
differ at generated index 21 (128415 versus 96723). Fresh-P129 ordinary versus DFlash separately
first differs at index 27 (95946 versus 98003); the isolated cross-mode pair differs at index 55
(112522 versus 100730). The result is functional classification only: graph capture is excluded,
while shared append execution and DFlash target/accept/commit behavior remain separate localization
owners. It provides no admissible timing or routing claim.

The follow-up exact layer-boundary capture is retained under
`profiles/bench/r9700-qwen3-layer-boundary-traces-43e5e4cc-20260906/results`. Its unchanged raw
closure is `bab590a7a40307a6f440a2bb41bd4c4135987eb7e67d8bac4ca5b58a38aa0d00`; the analysis-only
zero-draft-report repair is bound by closure
`b8096ffe54b6b51ead578c51a9357d96346073c64a37f27ef6fea91799569ec2`, and the summary SHA-256 is
`ffaf7dc9a0ea43be5c81e617680d0b5b842c2323778b38e350da45d40990c3c0`. Both comparisons are exact
through input, layer-zero mixer, and layer-zero MLP, then first differ after the layer-one GDN
mixer. Ordinary W1 versus DFlash W5 differs in 1,219/5,120 represented BF16 residual values; fresh
T129 column128 versus append T1 column0 differs in 142/5,120. The capture localizes the first
visible divergence to one mixer but does not yet identify its primitive cause, establish a
semantic failure threshold, or provide timing evidence.

A second packed-W4 candidate has passed standalone qualification through the production entry symbol
for the exact DFlash MLP-down N5120/K17408, T5 cell. The retained summary is
`profiles/bench/r9700-dflash-mlp-down-t5-production-symbol-qualification-20260906/summary.json`
(SHA-256 `a10d6789b6df7914d32774bf5492e50be58645d6f6483a5904f28cab58a9381a`), with result closure
`profiles/bench/r9700-dflash-mlp-down-t5-production-symbol-qualification-20260906/result.sha256`
(SHA-256 `50e39c0ed6e5bf70c162a7e69d7fc819bddc2f458f06d090869f693f2ee555d2`).
Its independent represented-input oracle reports zero BF16 steps for incumbent and candidate.
Balanced medians are 0.207386 ms and 0.057680 ms respectively, a candidate/incumbent ratio of
0.278129 (about 3.60x faster). Both launch orders win; the paired-ratio two-standard-error upper
bound is 0.276878 and the order-ratio delta is 0.006449. Independent review passed the exact
source/binary/assembly/static bindings, hardware and power-state checks, raw-decision recomputation,
and closure. The corresponding matched Engine builds are bound by
`profiles/bench/r9700-dflash-mlp-down-t5-matched-builds-20260906/build-receipt.json` (SHA-256
`aa20148816786691eac0d7916201e13e3f489b8a9c78a9a24e1a309a210cf574`) at source commit
`d1a9b6fb33a843363fba3ad46d6569071df1ab66`. Its control and candidate benchmark SHA-256 values are
`db248587690acc76a2d4dd2e74a3a5fccd4d8558d3cb66a80082a36ff8c75949` and
`1a6977420a4ddb460d0acea901ea10824ec90c798912d823f0b4c01fb24dc0f2`; the separate gate/up selector
is off in both arms. The MLP-down selector remains off by default. This is standalone
represented-input evidence only: it establishes neither generated-token parity nor whole-DFlash
speed, recipe selection, or production routing. The corresponding T6 candidate failed its static
resource gate before timing and remains on incumbent WMMA.

The owner trace under `profiles/rocprof/r9700-dflash-q4-c1-k4k5-owner-trace-plan-20260906`
attributes 84.95%/85.27% of summed K4/W5 and K5/W6 decode service to target verification. Target
Q4 Linear consumes 2707.578/2647.628 ms across 40/39 verification rounds, versus
497.823/472.893 ms for proposal/service Q4 Linear. K4/W5 and K5/W6 evidence SHA-256 are
`dd8ffd1eb031b8c7c5004c0d45690d2d08aaa4c05eab5d82845c72a9cc397ad8` and
`094139b10b607db827b6782dbe6b1a02dd2b5bee16771a35aa4211df19e11fe1`; asynchronous-drain repair
SHA-256 is `95d652a9c0a2bce16d6e3e24d488b88dfe8b4b0735a2471347765118f9fc6fbc`.
This identifies target N34816/K5120 gate/up only for that legacy all-Q4 execution, not for the
current four-role companion. In the current artifact all 64 Text gate/up matrices are FP8 and
bypass A8Q4; only the five Q4 DFlash proposal gate/up matrices use the selected small-T route per
round. Direct-cell medians therefore predict only about 7.1 ms per K4 run and 5.3 ms per K5 run
(roughly 0.15--0.19%), consistent with the near-neutral whole screen. The trace used a legacy
RowSplit all-Q4/G32 artifact, eager P128/G64 execution, and profiler interception; its durations are
attribution only, with no current-N16 performance, bandwidth/cache/stall, quality, acceptance, or
whole-inference claim.

Independent represented-input oracle, exact gfx1201 ISA/resources, and current-companion
whole-DFlash A/B remain required. The proposed `prepare_ragged_prefix` Wceil=12
compaction is not a live
optimization: startup planning already sizes persistent features, round tensors, append positions,
and workspace to the one resolved W5 or W6, and the Op writes that width directly. Its unreachable
Wceil/copy branch is removed rather than generalized. Physical profiling, not static traffic alone,
selects the next kernel.

Selected qualified Op results:

| Op/workload | Time | Numerical result |
|---|---:|---|
| SWA W4096 T3/B2 | `0.173 ms` | max abs `0.001246`, rel L2 `0.001619` |
| SWA W2048 T1 | `0.063 ms` | same complete oracle suite |
| grouped convolution BF16 T1/B1, D5120/G320 | `0.061 ms` | max abs `1.58101e-5` |
| grouped convolution W8 T2/B2 | `0.091 ms` | complete represented-input formula |
| Vision attention P194/S3 | about `0.087 ms` | rel L2 `0.001663` |
| Vision positional embedding P1024 | about `0.021 ms` | rel L2 `0.001633` |

The selected Vision attention route is resource-heavy (256 VGPR, 4 KiB LDS, 368 bytes private
scratch) but measured faster than both its spill-free recompute candidate and scalar baseline.
Resource counts alone are not a reason to replace a physically faster qualified implementation.

After the product request-lane cap moved to C/B=4, a focused physical gfx1201 rerun passed the
runtime planner, GDN recurrence, target GDN composition, scalar schedule, sampling, DFlash KV
append-prefix, SWA, MTP-round, and speculative-round qualifiers. The first SWA attempt exposed a
stale qualifier-only B8 workspace-capacity probe; the production Op correctly rejected that
out-of-contract request before launch. Changing the probe to the shared B4 maximum restored the
intended active-domain coverage, and the complete rerun passed. The retained direct W2048 T16/B4
SWA point is `0.053 ms`, with maximum absolute oracle error `0.00125`. This is support-boundary
correctness evidence, not model quality, capacity, whole-inference, or selection evidence.

## Speculative transitions

The retained pre-cap speculative-round measurements were approximately `2.01 ms` for chain K8/B8
and `0.59 ms` for mixed product-tree W12/B8 over 20 events at the 248320-token vocabulary. Those
B=8 values are historical, not supported-product evidence. The active exact qualifier covers
B=1..4; MTP next-round transformation passes exactly for K=1..5 and B=1..4. These measurements
cover the complete state transition but not a model round.

## Reproduction

Build the current graph:

```sh
cmake -S . -B build-r9700 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-r9700 --parallel
```

Focused physical targets include:

```text
ninfer_r9700_full_attention_qual
ninfer_r9700_swa_qual
ninfer_r9700_grouped_dynamic_conv_qual
ninfer_r9700_vision_attention_qual
ninfer_r9700_vision_pos_embed_qual
ninfer_r9700_speculative_round_qual
ninfer_r9700_runtime_planner_qual
```

The standalone Linear sweep/ISA recipe is `make -C tools/r9700 build/linear_op_qual linear-isa
linear-resources`. The Q4 A4/A8 oracle, timing, ISA, and durable report recipe is
`make -C tools/r9700 q4g64-linear q4g64-linear-isa q4g64-linear-resources`.

The fixed K6144 gated-RMSNorm prefill route assigns one token to each of eight wave32 waves while
retaining the incumbent feature-ordered FP32 reduction and BF16 result. Its immutable
pre-promotion report `profiles/bench/r9700-gated-rmsnorm-k6144-token8-ab-20260904-r3.json`
(SHA-256 `31c50ee9f42edd61dae21266b64b099af0e5b3a94051eecb4f55fa783f5ed77f`)
passed bit-exact incumbent and independent FP64 checks. Median P2048 time fell from 1.150442 to
0.155001 ms (7.4222x); every measured T>=128 point was non-regressing, and T64 won by 1.7874x.
Production therefore selects token8 for K6144/T>=64 and retains the incumbent elsewhere. LLVM
reports 22 VGPR, occupancy 16, four 128-bit input loads, and zero LDS/private/scratch.

The fixed K128 ordinary-GDN gated-RMSNorm route assigns one flattened value-head row to each
wave32 wave and eight rows to each CTA only at `48*T` rows for T=1024/2048/4096/8192. Its
immutable pre-promotion report `profiles/bench/r9700-gated-rmsnorm-k128-rows8-ab-20260904.json`
(SHA-256 `bdada371d626a6f4398ac350f5aaebdf3318d1743fb507e32f626b3c062e2e5c`) passed incumbent
BF16-bit parity, an independent FP64 complete-formula oracle, poison rewrite, input/output alias,
and malformed-input checks and won every admitted extent. At P2048 it fell from 1.16636395 to
0.143720999 ms, a 0.123221397 candidate/incumbent ratio. LLVM reports 24 VGPR, occupancy 16,
128-bit reduction loads, and zero LDS/private/scratch. Other K128 row counts and feature widths
retain the general route.

On the same four-role hybrid artifact and P2048/C1/chunk4096/no-spec workload, the production
promotion improved the matched whole mean from 1.251199252 s / 1,636.830337 tok/s to
1.202762421 s / 1,702.749429 tok/s, a 1.040272x throughput gain. The post-promotion report is
`profiles/bench/r9700-gated-rmsnorm-k128-rows8-production-p2048-c1-20260904.json`, SHA-256
`d1f67fb21427c5378e87afec4f4a273a34425223dd4c941d58e01fa3bc45a5aa`. This admits the eager
kernel, but it does not promote the evaluation-only hybrid artifact or satisfy the 2,000 tok/s
product floor.

The subsequent ordinary-GDN LDS-scope candidate narrowed synchronization only for CTA-private
LDS and passed direct exact-parity and independent-oracle checks. Although its direct P2048 Op
median improved from 3.113205 to 2.761526 ms, the matched whole run improved the accepted K128
baseline from 1.202762421 s / 1,702.749429 tok/s to 1.193522512 s / 1,715.930810 tok/s. The
9.239909 ms saving missed the predeclared 10 ms whole gate by 0.760091 ms, so the route is rejected
and production retains the incumbent ordinary recurrence. The immutable terminal report is
`profiles/bench/r9700-gdn-ordinary-lds-production-p2048-c1-20260904.json`, SHA-256
`0af07a8340f310b7f89a800bc90864377fb76ac5e45c2e22f8dd74d580a6a69f`.

The production Text-MLP boundary fuses SiLU-multiply with signed-A8G64 preparation only when the
gate/up weight is row-scaled E4M3, the down weight is Q4G64, T=2,048, and the caller is one of the
64 main Text layers. It consumes the concatenated BF16 gate/up result, computes the FP32
SiLU-multiply, explicitly rounds that internal result to BF16, preserves the existing A8
scale/code/status semantics, and invokes the unchanged M64N128 Q4 down matrix. This removes one
142,606,336-byte BF16 write/read handoff and one launch per layer (9,126,805,504 logical bytes and
64 launches for P2048), without changing the graph-stable 608,387,072-byte workspace. Every other
weight profile, token width, and layer retains the ordinary SiLU plus linear path.

The immutable direct report
`profiles/bench/r9700-fused-silu-a8q4-down-p2048-ab-20260904.json` (SHA-256
`4d4143bc096d759c71c7d12cca2e3d68aa2a5c2bf0a4c74d2e972dfbc195f2b3`) passed full workspace/output
bit parity, independent represented-input formula/codec probes, poison, tail/alias, and resource
gates. Its stage median fell from 0.659238994 to 0.342199489 ms (ratio 0.519082598), and the complete
MLP median fell from 3.926753998 to 3.644394517 ms (ratio 0.928093412). The matched hybrid whole
report `profiles/bench/r9700-fused-silu-a8q4-down-production-p2048-c1-20260904.json` (SHA-256
`5daa98aafa34efcc5a55ee2eeca464ae94575c95ed60e6303cd9216ae253ab76`) improved the accepted K128
baseline from 1.202762421 s / 1,702.749429 tok/s to 1.178537057 s / 1,737.747715 tok/s: a
24.225364 ms saving, 0.97985855 elapsed ratio, and 1.02055953 throughput speedup. This clears the
predeclared 10 ms whole gate and admits the exact fused boundary; it does not promote the
evaluation-only hybrid artifact or satisfy the 2,000 tok/s product floor.
The production static report `profiles/bench/r9700-fused-silu-a8q4-down-static-20260904.json`
(SHA-256 `aea9061d6bb27fb9e0fec9508b242816cf5cf69118fe1f36b5616a3fcdbf5336`) binds the exact emitted
symbol and confirms 12 VGPR, zero LDS/private/scratch, the required load/store and native-exp
inventory, and no intermediate BF16 global traffic.

The exact ordinary Text P2048/C1 GDN scale-sidecar candidate passed its direct gate but failed its
fixed whole-prefill gate. It combined the two represented-BF16 Q/K extracts with one
incumbent-order FP32 inverse-norm sidecar per token and Q/K head, then retained the production
192-CTA/four-row-tile FP32-state recurrence. The provenance-complete direct report measured
2.252036095 ms versus 3.259594917 ms per call, or 108.0977325 ms over 48 calls, while remaining
bit-exact with the incumbent and passing the complete FP64 state/output oracle. Preparation was
0.03379999846 versus 0.07719899714 ms and recurrence was 2.164277077 versus 2.893904924 ms. That
report is
`profiles/bench/r9700-gdn-scale-sidecar-p2048-ab-v2-20260904.json`, SHA-256
`20288276c005bf469f99d6a7dccddc497c5d49c6a2301b65dac746f71fc38304`. The clean isolated whole
run measured 1.153886920 s / 1774.872034 tok/s, saving 24.650137 ms from the accepted 1.178537057 s
baseline but missing the fixed 1.148537057 s / 30 ms gate by 5.349863 ms, with workspace unchanged
at 608,387,072 bytes. Its report is
`profiles/bench/r9700-gdn-scale-sidecar-production-p2048-c1-v2-20260904.json`, SHA-256
`110dbb4090dfa4a92ac993bd70ac4c21666b7a259abd17f63a4dbb913d9f6e95`. The candidate and its
qualification surfaces were removed; production retains the general recurrence with no adjacent
variant.

The exact P2048 MLP gate/up hipBLASLt algorithm check is terminal. All eight algorithms returned
under the production zero-workspace preference passed the represented FP64 and eager/captured
correctness gates. The best non-default result, rank 1 fingerprint
`dde00100000000000000000000000000`, was effectively tied with production rank 0: matrix medians
were 4.064851046 versus 4.061550617 ms (ratio 1.000812603), and complete medians were 4.186669827
versus 4.189990520 ms (ratio 0.999207470). Its projected 64-call saving was only 0.212524414 ms,
far below the required 10 ms, while ranks 2 through 7 were slower. Production therefore retains
rank 0 fingerprint `e0e00100000000000000000000000000`; no alternate fingerprint, workspace,
capacity change, or whole run is admitted. The immutable terminal report is
`profiles/bench/r9700-fp8-gate-up-algorithm-p2048-ab-20260904.json`, SHA-256
`5dd95d8f01d911e34e437175e028ed973ce025361ba61e9a269239f97c32de31`.

The ordinary-GDN output gated-RMSNorm-to-A8G64 fusion passed exact incumbent parity, an independent
represented-FP64 formula/codec, poison and alias checks, and the gfx1201 static gate at 25 VGPR,
256 threads/eight waves, and zero LDS/private/scratch. Under `auto` at P2048, preparation improved
from `0.239998996` to `0.196279004 ms`, but the candidate's 48-call aggregate was
`9.421392202 ms`, above the predeclared `5.699527 ms` ceiling. Complete output improved from
`1.421954989` to `1.358394980 ms`, only `3.050880432 ms` across 48 calls versus the required
`5 ms`. The candidate is rejected and removed; production retains separate K128 gated-RMSNorm,
A8G64 preparation, and Q4 output projection. The immutable report is
`profiles/bench/r9700-gdn-output-gated-rmsnorm-a8-fusion-p2048-ab-20260904.json`, SHA-256
`8396c5e76b4166405acab70abce11f7f7137e264a1eb61eec3b25d0a098ba962`.

The subsequent fixed-P2048 ordinary-GDN projection/convolution direct-scatter candidate passed
its direct gate: its per-layer median was 0.288159013 versus 0.559597015 ms (ratio 0.514940202),
projecting 13.0290241 ms over the 48 GDN layers. The immutable direct report is
`profiles/bench/r9700-gdn-prefill-projection-conv-direct-scatter-ab-20260904.json`, SHA-256
`fe852ad70f33f2e654cf31f2ef0da874c5355cdd491f9f97e027cb68142cf310`. At the isolated whole gate
after the accepted MLP fusion, however, prefill improved only from 1.178537057 s /
1,737.747715 tok/s to 1.168570885 s / 1,752.573288 tok/s: 9.966172 ms and 1.00852852x, missing the
predeclared 12 ms and 1.01x gates by 2.033828 ms and 0.00147148x. Workspace remained exactly
608,387,072 bytes. The immutable whole report is
`profiles/bench/r9700-gdn-prefill-projection-conv-direct-scatter-production-p2048-c1-20260904.json`,
SHA-256 `8ceaa1d948e0a76d29765a4255dd806a2bc3ffeba4b5ec009fc39f8b893e7c2f`. The candidate is rejected;
production again uses the ordinary projection copies, causal convolution, and three extracts, and
all candidate-only implementation and tooling surfaces are removed.

The dense full-score P2048/G16 16-wave PV head-partition candidate passed its represented FP64,
incumbent-bit, poison/error, and static checks (70 VGPR, occupancy 16, 9,208-byte LDS, 512 threads,
zero private/scratch), but missed both direct admission gates. Candidate versus incumbent medians
were 6.84690714 versus 7.28727818 ms for PV (ratio 0.93956989) and 10.1469564 versus 10.8013163 ms
for the complete Op (ratio 0.939418495), above the required 0.80 and 0.90 ratios. No whole run was
admitted, production retains the eight-wave PV kernel, and this result does not authorize an
adjacent mapping sweep. The immutable terminal report is
`profiles/bench/r9700-dense-full-score-pv-w16-head-partition-ab-20260904.json`, SHA-256
`38ebd0c64352ccb0b27c1b60f92fa537865a92827901879f89b2c189c212feee`.

A capacity-preserving FP32 hipBLASLt PV replacement was also terminally rejected at P2048/G16.
Four zero-workspace strided-batched calls (`batch_count=6`, `[M,N,K]=[256,2048,2048]`) preserved
the existing interleaved query-head score/output planes and used one 8,388,608-byte decoded-V image.
The in-process selected solution 140189 was GSU1/SK0 and passed the complete FP64 causal oracle,
bitwise repeat, graph replay, invalid-frontier, fixed-domain, and no-clobber gates. Candidate versus
incumbent medians were 4.524150848 versus 5.141388893 ms, only 9.875808716 ms direct saving over
16 calls; its old-bucket projection was 80.216934204 ms versus the required 61.161073 ms. Production
is unchanged and no adjacent layout or algorithm sweep is admitted. The immutable terminal report
is `profiles/bench/r9700-dense-fp32-gemm-pv-p2048-ab-20260904.json`, SHA-256
`b231a114f4b50684f2e2fdc3eb8a8ce08a9702053dc32e5ff9346a80132a8538`.

The dense full-score P2048 FP8-Q/Bk32 candidate also passed every numerical, liveness, and static
gate (32 native FP8 and zero BF16 WMMAs, 63 VGPR, 8,192-byte LDS, occupancy 16, no spills), and was
faster directly at 2.983591080 versus 3.442310095 ms per call (`0.8667409378x`). Across the 16
dense layers this is 47.73745728 ms and only 7.33950424 ms matched projected saving, short of the
predeclared 32.439611 ms / 15 ms gate. No whole run was admitted. Production therefore retains the
BF16-WMMA Bq16/Bk32 QK route, and the candidate-only implementation and tooling are removed without
an adjacent sweep. The immutable design and terminal reports are
`profiles/bench/r9700-dense-full-score-fp8-q-bk32-static-design-20260904.json` (SHA-256
`aed480c787313c280aefc83d8cdf943a6b81711872786677f905f3d4036cd521`) and
`profiles/bench/r9700-dense-full-score-fp8-q-bk32-ab-20260904.json` (SHA-256
`ef0d6557a59145c3f632b96b24f41c922369d9f42fa48ab2c7b24d5b1c7e7f30`).

The fixed K256 query/key RMSNorm route likewise assigns one row to each of eight wave32 waves at
T>=128 while retaining the exact feature-order FP32 reduction. Its immutable pre-promotion report
`profiles/bench/r9700-rmsnorm-k256-token8-ab-20260904.json` (SHA-256
`87e4be62c5c7574273f7ee22eaf4858fa8442a496d738fa68a42d8b4452edbce`) passed exact-incumbent
and independent FP64 checks and won every measured row. At 2048 rows it fell from 0.093921 to
0.041240 ms (2.2774x); at the real 49152-row query extent it fell from 1.101083 to 0.095321 ms
(11.5513x). Smaller K256 rows and other widths retain the incumbent. LLVM reports 23 VGPR,
occupancy 16, four 128-bit loads, and zero LDS/private/scratch.

The exact Text-MLP split view `[17408,T]` with element strides
`[1,34816,34816*T,34816*T]` uses a two-dimensional feature-by-token SiLU-multiply grid at T>=128,
removing generic coordinate decomposition without changing the FP32 formula or BF16 result. The
immutable pre-promotion report `profiles/bench/r9700-silu-mul-split17408-2d-ab-20260904.json`
(SHA-256 `a561f8b7981b836b2a5a582d093d67d087f2841efede95791aa5969a13b7bc10`)
passed exact incumbent and independent FP64 checks and won every measured row. P2048 fell from
1.009159 to 0.374680 ms (2.6934x). Other layouts and T<128 retain the generic strided route; the
selected gfx1201 kernel uses 9 VGPR, occupancy 16, and zero LDS/private/scratch.

The whole-product benchmark builds as `build-r9700/bench/ninfer_bench`; it requires a real accepted
artifact. Run its `--help` output for the exact workload options.

## Missing final evidence

The complete 18-shard Qwen3.8-27B BF16 source and matched 8K/32K dual-A8 candidate sidecars are
present. The localized deterministic scorer route is now mandatory in the implementation:
hipBLAS/no rocBLAS atomics are fixed before PyTorch import, strict deterministic algorithms are
enabled before backend construction, conflicts fail, and ordinary results bind the complete
implementation/runtime provenance. Two fresh-process campaigns are byte-exact at 8K and 32K and
therefore close the BF16 numerical authority; the unchanged dense all-Q4 sidecars have also passed
offline gate recomputation against it. Its greedy argmax is diagnostic rather than an admission
condition. The following remain explicitly unproven before Pareto classification:

- fresh current sparse all-Q4 and mixed dense/sparse quality acquisition against that authority;
- the selected integer weight recipe;
- G16 versus G32 under the qualified fixed plane orders;
- graph/eager model parity;
- DFlash2 acceptance and speed on the final companion artifact; existing MTP requires only
  regression preservation of its exact execution, cache, and state contracts, not further tuning;
- complete prefill/decode and C=1..4 throughput;
- whole-inference profiler attribution.

After the remaining candidate quality acquisition closes, the schema-v7 Pareto classifier retains
every non-dominated candidate. A quality-eligible
candidate is dominated only if another eligible candidate is no worse in
mean-NLL delta, new-severe-position rate, resolved capacity, and every matched whole-inference
speed cell, and is strictly better in at least one. Raw scorer seconds are not a throughput or
dominance objective.

The Pareto frontier remains the audit record, but the product ships one artifact, cache, and
execution profile. Quality is the admission gate. The classifier first retains one static-profile
winner for each recipe, then applies the same global ordering to those recipe winners: normalize
each required whole-inference cell against the best frontier result and maximize the candidate's
minimum ratio; on an exact tie maximize minimum normalized resolved capacity, then minimize the
worst fraction of the declared mean-NLL/severe-position quality budgets consumed. Canonical
artifact and static-profile identity resolves only a complete measured tie. The deterministic
decision uses retained per-cell means; raw repetition spread remains evidence rather than a noise
tolerance. Schema v7 retains this as `terminal_production_selection` under
`global_maximin_whole_then_capacity_then_quality_then_canonical_v1`, including the complete
frontier, every recipe winner, the one terminal winner,
and its compile-bound dense or B128/S16/tau900 attention identity.
This maximin decision reflects whole-product performance without inventing workload weights or
allowing an average to conceal a material regression.

`rocprofv3` runtime/kernel/memory traces and SQ busy/wave events are usable. On gfx1201, dispatch PMC
cache collection additionally requires a stable power state. A focused real-model 8K G16
XAttention-consumer pass under `profile_standard` produced nonzero `GL2C_HIT`, `GL2C_MISS`,
`TCP_REQ`, `TCP_REQ_MISS`, GL2 external read/write activity, and `SQ_WAVES`. Summing hardware
dimensions gives a 98.371% GL2 hit ratio and 77.692% TCP/vector-GL0 hit ratio. The raw CSV is
`profiles/rocprof/xattention-b16-all-q4-g16-c1-8k-cache-pmc-20260904/cache-pmc_counter_collection.csv`
(SHA-256 `7bc45f6fc1342ec4d49a42e3a195ecbf9e02e3e59c219c1682eae730989a9cae`).
Its 32 selected consumer dispatches use the expected 1,572,864-thread grid and total 4.5698 seconds
of device time, with a 126.027 ms median. That is 2.76% of this pass's prefill duration, so the new
consumer does not own the stable-profile whole duration. It is nevertheless about 3.16x slower at
4,096 real query rows than a linear extrapolation of the hot 128-row qualifier. The completed
production-scale control below resolves that operator-scaling question; final matched `auto`
whole-model attribution remains required.

The schema-v6 XAttention qualifier supplies the production-scale control at
context 8,192/T=4,096/Hq24/Hkv4/D256. It rotates four address-distinct operand and workspace sets,
reports rank and consumer stages separately, and uses signed nonzero INT4 values with exact FP16
scales varying by physical token, page, KV head, and value group behind a nonidentity page table.
Its all-element FP64 softmax/PV check reconstructs the selected logical pages and therefore rejects
payload or scale addressing failures that an all-zero timing fixture would conceal. Physical timing
under `auto` passed exactly for both value groups. The current-tree refresh binds the updated dense
control as well as the unchanged sparse implementation and independent oracles. At a measured
16.2502% keep fraction, G16 took `53.4330635 ms` (`26.5418129 ms` rank and `26.3109531 ms`
consumer) and G32 took `56.6925545 ms` (`26.3639278 ms` rank and `30.2011223 ms` consumer).
Both all-element production oracles reported exact zero relative-L2 and maximum-absolute error.
The retained reports under
`profiles/bench/r9700-xattention-current-tree-refresh-20260905/` have SHA-256
`205bac511ff61511a5e61233942beeead49829a417689a0ceaeafd4ef5427d11` (G16) and
`ab5df29c798cdab9aae233da8433f84e2da8e36f8d3fde27a85f6e56d9bdcaaf` (G32). Schema v6
binds the executable, exact current source inventory, device/runtime identity, and power state,
rechecks `auto` after numerical qualification and timing, and publishes only to a fresh path. This
closes the stale dense-control source-hash gap and operator-scale qualification; it does not select
G16/G32, a weight recipe, a prefill chunk, or a whole-model route.

The production-extent linear-Op gates also passed under `auto`. The A8Q4G64 cooperative CTA won
all 32 qualified shape-by-token rows by `2.5261x` through `8.1587x`; the A8W8G32 CTA won all 16
qualified rows by `14.8205x` through `18.4695x`. Both independent BF16-step oracles reported zero
maximum steps, including their exhaustive padded-tail cases. Their schema-v3 report SHA-256 values
are `b3c78cab21dfc0f97521269245b8442009048aadcd9db5ffd947a4d7d3baf889` (Q4) and
`1547f7d62fae4077f32f203b61831fc97b4f98bffcc930c6a29de7284f5c40c0` (W8). These results admit
only the exact measured T=1,024/2,048/4,096/8,192 tuple predicates, which are now the production
linear-Op dispatch boundary. Decode, partial chunks, Vision, DFlash, and unmeasured matrix shapes
retain the incumbent one-wave path. The reports are immutable pre-promotion admission provenance;
future decisions use rebuilt whole-model evidence rather than regenerating an admission report
against changed production source. Matched model-prefill and whole-inference selection remain open.

The follow-up Q4 M64xN128 ping/pong staging challenger passed its exact oracle and had no
regressing tuple, but it did not pass its predeclared `>=1.5x` admission gate. Weighted P2048 fell
from `1126.778368` to `914.413888 ms`, a `0.8115295` challenger/incumbent ratio (`1.23224x`,
`212.364480 ms` saved). The terminal report is
`profiles/bench/r9700-a8q4-prefill-cta-pingpong-ab-20260904.json`, SHA-256
`06a2846e1ae5ba90989dcb401e578a1b96479a8e472b7aa0d5befdcbcf6281f7`. Production remains
the selected M64xN128 persistent-N2 CTA at that operator-only decision point; this rejected
staging-only result is not by itself a production performance claim.

The later matched whole-P2048 gate overrides that operator-only promotion decision without
rewriting its evidence. The control report
`profiles/bench/prefill-p2048-q4-pingpong-matched-control-20260904.json` (SHA-256
`66a4c832b67156e713fbe5d3de57598086ec4a264acf513b212d149e70ec1254`) identifies
`m64n128-production` and measured `1219.355187 tok/s` in `1.679577815 s`. The challenger report
`profiles/bench/prefill-p2048-q4-pingpong-whole-20260904.json` (SHA-256
`a187d24e1ed78154f40a71ec1243f1d64ef46fbb25d3f0a2b2af3937667bb244`) identifies ping/pong and
measured `1348.188927 tok/s` in `1.519077416 s`. That is a `1.105657x` throughput improvement and
`0.9044400` elapsed-time ratio, saving `0.160500399 s` at the whole-prefill scope. Ping/pong is now
the sole production Q4 CTA for the exact eight-shape by P=1,024/2,048/4,096/8,192 predicate; the
single-bank M64xN128 implementation remains only as an explicit regression/tail control.

The later source/emitted-compile-matched scalar-base/U32-voffset promotion preserves the same
Q4G64/A8G64 representation, arithmetic, LDS topology, grid, ABI, and exact production predicate.
Its actual-wrapper operator gate passed with a `21.057251555 ms` exact-call-weighted robust saving
lower; the whole P2048 C1/chunk4096/spec-none gate reduced median total from `1097.650069` to
`1075.475953 ms`, with a `20.306733144 ms` robust saving lower and exact token identity. The final
production profile is `m64n128-pingpong-n16-k16-scalar-base-production`, with no build/runtime
selector or duplicate full-tile path. Its `1075.438603 ms` prefill median is `1904.339303 tok/s`,
which is the current dense authority but remains below the 2,000 tok/s floor. The immutable
operator and whole reports are
`profiles/bench/r9700-a8q4-n16k16-scalar-base-product-p2048-ab-20260905.json` and
`profiles/bench/r9700-scalar-base-production-p2048-c1-ab-20260905.json`.

The next exact LDS-index-remap candidate was rejected statically without GPU execution. The
selector-free canonical scalar-base object emits `88` VGPR; both algebraically equivalent remap
forms emitted `92` VGPR. All three retained the same 17,152-byte LDS allocation, occupancy 16,
zero private/scratch/spill storage, eight native IU4 instructions, and ten
scalar-base/single-U32-voffset loads. The candidate therefore failed its explicit no-VGPR-regression
gate before numerical or timing work. Both private remap forms and their temporary qualification
surfaces were removed; the restored canonical `r9700_linear.hip` SHA-256 is
`7d204bb10feeeb5988c26960ce992a6f487fd53afc162e89508ee3524b90b576`.

Complete dual-FMAC pairing preserved the canonical static resource contract (88 VGPR, 17,152-byte
LDS, occupancy 16, zero scratch/spills, eight IU4 sites, and ten protected scalar-base/U32 loads)
while changing the eligible instruction mix to eight dual FMACs and zero singles. The v1 and v2
operator reports were retained as inconclusive startup-transient evidence:
`profiles/bench/r9700-q4-prefill-dual-fmac-p2048-ab-20260906.json` (SHA-256
`7c440f0b318c0886ad5e47a7629e91e992234637a96125b026943ad00edf9145`) and
`profiles/bench/r9700-q4-prefill-dual-fmac-p2048-ab-v2-20260906.json` (SHA-256
`a60b1a7ae2795b6be0c75ed9da657c88e782092d84417a3dcd71d5884ead64d6`). After adding two
balanced unscored intervals per server, v3 accepted every operator cell (robust upper ratios at
most `0.9655278292`) and achieved a `12.6098960755 ms` call-weighted robust saving lower; its report
is `profiles/bench/r9700-q4-prefill-dual-fmac-p2048-ab-v3-20260906.json` (SHA-256
`42c3847f7b52a8d9c436dcc957175439191decc057f51d2234c8d729676f6e1d`).

The terminal whole-P2048 gate rejected the candidate despite exact output parity and token
identities passing. Control/candidate medians were `1072.2601595`/`1059.691983 ms`; ratio upper
`0.9924224143` passed, but the robust saving lower was `8.1438331535 ms`, below the required
`10 ms`. The authoritative report is
`profiles/bench/r9700-q4-prefill-dual-fmac-whole-p2048-c1-full-v2-20260906/report.json` (SHA-256
`88da09557579e3d2e729ef1d7858c69fca5a29718c246ad9dc2403749cedfce1`). The selector,
challenger, and timing tooling were removed, leaving canonical scalar-base production unchanged.
The final explicitly scheduled two-entry register-load FIFO was rejected at its loaded-object
resource gate. Its fixed G+1/G+2 payload slots retained both 8,576-byte LDS banks, 17,152-byte total
LDS, a 512-thread maximum workgroup, zero private storage/spills, eight IU4 sites, and the canonical
six dual plus four single FMAC sites, but the selected kernel used `97` VGPR: nine above the
canonical `88`-VGPR gate and one above the absolute `96`-VGPR ceiling. No numerical or GPU timing
work was admitted. The immutable report is
`profiles/bench/r9700-q4-prefill-register-fifo-static-rejection-20260906.json` (SHA-256
`15b9369772015ae1cb079d17272f3c027d29fb2a7148236968e2e014bf4cd133`), with extracted inner gfx1201
ELF SHA-256 `24da41789f510c0f848b62ca2020b890ce0a33be96cea05ffbaeea12ea73aef7`. The selector and candidate
were removed. Canonical P2048 therefore remains `1075.438603 ms` / `1904.339303 tok/s`.

The focused retained-production trace now refreshes attribution for that exact dense
C1/P2048/G0/chunk4096/spec-none route. The validated evidence is
`profiles/rocprof/r9700-retained-production-p2048-trace-plan-20260906/evidence.json` (SHA-256
`73b1f5ff024ec2b761411ab5f3e64eb7c9accdbd9fb0c93720a95ceb4c4857f0`). It binds the
selector-free executable SHA-256
`a7c9303bd213fa3dbdb29ca0cee73addef1de8ab6a6e6b231509e25239776425`, four-role N16/K16
artifact SHA-256 `040c6e7ed29c856718a638c00181975710d987b7d5f49f4cafbdf68911f7e7d2`, and corpus
SHA-256 `27e4f63c17efe3f89b5cf278b3b1a42a737316ed4044d7d0d1d52437059d1002`. The separate
unprofiled authority remains `1075.438603 ms` / `1904.339303 tok/s`; trace timing cannot replace
it. The Text range was `1076.624083 ms`, with `1043.339444 ms` active-kernel wall union and
`33.284639 ms` kernel-inactive wall. Independent service attribution places Q4 projection at
`334.762518 ms` across 176 calls (`65.6891` useful TMAC/s), FP8 projection at `307.310638 ms`
across 144 calls, dense attention at `147.247476 ms` across 48 QK/maximum/PV calls, and GDN
recurrence at `139.397112 ms` across 96 calls. Ninety-three known short runtime fill/copy
dispatches overlap Text without a stage marker, totaling only `0.357956 ms` (maximum
`0.006760 ms`, `0.033248%` of Text wall); they are retained as unattributed and make stage attribution
explicitly incomplete. The trace has no hardware counters and establishes neither profiled
throughput, physical bandwidth/cache utilization, causal stalls or stall freedom, nor practical-
ceiling closure. The completed FP8 audit splits the `307.310638 ms` matrix service exactly:
64 MLP gate/up calls at `[T,N,K]=[2048,34816,5120]` take `255.024909 ms` (`183.234` useful
TFLOP/s), 48 GDN query/key calls at `[2048,4096,5120]` take `23.726265 ms` (`173.781` useful
TFLOP/s), and the 32 query/key plus gate/value calls across 16 full-attention layers at
`[2048,7168,5120]` take `28.559464 ms` (`168.433` useful TFLOP/s). The first two roles share
hipBLASLt solution `123104`/fingerprint `e0e001...` and the attention pair uses solution
`123100`/fingerprint `dce001...`. Their fresh runtime symbols and resource envelopes exactly match
the prior URI-bound loaded-ELF proofs in
`profiles/rocprof/r9700-fp8-gate-up-linear-execution-proof-20260904/proof/proof.json` and
`profiles/rocprof/r9700-fp8-attention-qk-gate-value-linear-execution-proof-20260904/proof/proof.json`:
the gfx1201 kernels contain respectively 112 and 160 static
`v_wmma_f32_16x16x16_fp8_fp8` sites, use 192 architectural VGPR and zero private/scratch, and use
25,088/12,544 bytes LDS. These are E4M3-by-E4M3 FP8 matrices with FP32 accumulation and BF16
publication; they are not the separate A8W8/IU8 CTA route.

Closing the unchanged 2,000 tok/s floor from `1075.438603 ms` requires at least `51.438603 ms` of
whole-P2048 saving. Removing that amount from the FP8 matrices alone would require aggregate
service no greater than `255.871638 ms`, or `217.542` useful TFLOP/s (`1.20103x` current). If the
dominant gate/up family supplied it, that family would need `229.531` useful TFLOP/s
(`1.25266x`). The two smaller families total only `52.285729 ms`; even at the retained
`400.835`-TFLOP/s issue reference they can save at most `29.998424 ms`. The installed gate/up
catalog has already reduced 791 entries to ten supported zero-workspace solutions and timed all
ten; the nearest alternate projects only `0.212524 ms` saving over 64 calls. The exact custom
M128xN128 route regressed from `3.605496` to `7.640492 ms` per call. The retained M128xN256 route
was reopened after separating static logical (`97`), HIP runtime-reported (`101`), and descriptor
allocation-rounded (`104`) VGPR. It passed the complete-FP64 oracle at all 2,192 probes,
status/alias/canary and graph checks, exact FP8-ISA/resource checks, and launch-order stability,
but terminally regressed from the solution-123104 incumbent median `3.499504089 ms` to
`49.380744934 ms` (`14.7860151` useful TFLOP/s). Projected complete time is
`49.48067909025 ms`, and projected 64-call saving is `-2936.39941408 ms`. The immutable decision
is `profiles/bench/r9700-fp8-gate-up-m128n256-retained-reopen-20260906/attempt-4/decision.json`
(SHA-256 `f2510b2587bb56a18c175e8d1fcc456c3cb7ad38890b9805acfd7b159c913e9b`). Complete removal
of the downstream SiLU/A8 stage is bounded near `20.76 ms`. Therefore
no FP8 catalog, custom-matrix, adjacent-fusion, or counter-only experiment is admitted as a
credible `>=51.439 ms` mechanism. Canonical dense P2048 remains `1904.339303 tok/s`; the floor has
not passed, practical-ceiling closure has not been proved, and choosing whether to retain or revise
that floor is now a product-contract decision rather than authorization for another unbounded
optimization sweep. The bounded prefill phase is closed and work transitions to recipe-independent
DFlash2 scheduling and exact-shape operators. Do not rerun this object or an adjacent tile sweep.
M128N256 may return only for a materially different source-level mechanism with independent
numerical/static evidence and a concrete bound for at most `3.18103603125 ms` per matrix and at
least `51.438603 ms` projected whole-P2048 saving.

This exhausts the explicit bounded candidate list. A targeted audit of the canonical loaded ISA and
retained P2048 trace found no new mechanism outside the rejected topology and representation
families with a credible `>=35.9 ms` whole-P2048 saving. That threshold is a `10.08%` reduction of
the retained `356.276973 ms` Q4 service. Canonical ISA already issues successor loads ahead of its
eight IU4 WMMAs at occupancy 16; the independent optimistic bounds for activation/scale-side cleanup
(`19.045 ms`) and arithmetic-encoding cleanup (`<14.3 ms`) are each too small, and the concrete
second-payload overlap schedule costs the nine VGPRs that failed this gate. Those overlapping bounds
cannot be summed. Candidate exhaustion is not practical-ceiling evidence, so that closure remains
open. Ordinary decode has separately reached its bounded roof. The P2048 floor/product-contract
decision is now next; chunk/base selection and the selected companion remain blocked prerequisites
for DFlash2.

The structural M64xN256 plus ping/pong follow-up also passed its exact/FP64 oracle and every tuple
was nonregressing, but it likewise failed the fixed `>=1.5x` gate. Weighted P2048 fell from
`1109.850230` to `920.208770 ms`, a `0.8291288` challenger/incumbent ratio (`1.20609x`,
`189.641460 ms` saved). Its terminal report is
`profiles/bench/r9700-a8q4-prefill-cta-m64n256-ab-20260904.json`, SHA-256
`e1d611ff2a74ebad9c300297c93ea1722a74c57d01c71a75836987fe7fe3ba3b`.
At that operator-only decision point, production remained the single-bank M64xN128 route; the
later matched whole-P2048 selection above supersedes that route with ping/pong.

The subsequent trace-selected M64xN256 32-wave experiment narrowed the candidate to the dominant
Q4 MLP-down tuple `[N,K]=[5120,17408]`. Its 1,024-thread kernel retained the eight IU4 instructions
and compiled to 87 VGPR, 25,856 bytes LDS, and zero private/scratch storage, but was slower than
production at every qualified token extent. At T=1,024/2,048/4,096/8,192 the challenger/control
ratios were `1.0146974/1.0120804/1.0203344/1.0149652`; T2048 measured `3.419854/3.379034 ms`.
The terminal report is
`profiles/bench/r9700-a8q4-m64n256-w32-mlp-down-ab-20260904.json`, SHA-256
`a41e2a2ad65809ade16b629e698244834d2c2da065335f8fab62869813aaef5b`. Both rejected M64xN256
implementations have been removed, production remains M64xN128 ping/pong, and the bounded stop rule
forbids an adjacent N256 or cache-hint sweep.

The subsequent production-geometry next-G64 scale-prefetch experiment retained the M64xN128
ping/pong tile, eight native IU4 WMMAs, exact arithmetic, and 17,152-byte LDS while moving two
scale halfword loads into the existing code-prefetch window. It passed static and numerical gates
at 89 VGPR, occupancy 16, and zero scratch, but lost every MLP-down/GDN value-Z/GDN output cell at
T=1,024/2,048/4,096/8,192 (`1.016370x`--`1.048674x`). Trace-call-weighted P2048 regressed from
`375.835487` to `391.499994 ms` (`1.041679x`, `-15.664507 ms`). The immutable terminal report is
`profiles/bench/r9700-a8q4-prefill-cta-scale-prefetch-ab-20260904.json`, SHA-256
`aaeeaf16a0040d5a23046d40fff532f34805124cdd3e5bbeb5d0a721962ad8ef`. The candidate was removed;
production retains its post-compute scale loads.

The bounded residual-add to K5120 token8 RMSNorm prefill fusion likewise passed exact residual and
output publication, complete independent-oracle, alias, rewrite, and static resource gates at
18 VGPR, occupancy 16, and zero LDS/private/scratch. Physical `auto` timing rejected it at every
T=1,024/2,048/4,096/8,192 extent: challenger/control ratios were
`2.98276377/2.94057178/2.26832342/1.89297712`. P2048 regressed from `0.188480005` to
`0.554238975 ms` per pair, a `-0.365758955 ms` saving. The immutable terminal report is
`profiles/bench/r9700-residual-rmsnorm-k5120-prefill-ab-20260904.json`, SHA-256
`6ed367d0438f0f7f8dbfe4bd01f7da126a6857c1953da8f1ed6e294eec20c134`; the retained static design
is `profiles/bench/r9700-residual-rmsnorm-k5120-prefill-static-design-20260904.json`, SHA-256
`40dcc021e59e8c80272ed6c36c9ab4ef36e1bd6c2e9939c439671426a44eca73`. The candidate and its
qualification tooling were removed. Production remains the separate residual-add plus selected
K5120 RMSNorm pair, with no adjacent fusion variant authorized.

A separate exact-device ceiling probe under `auto` measured `801.669240` median issued-IU4 TOPS
from an eight-chain register-resident native `v_wmma_i32_16x16x32_iu4` kernel and
`633.264977 GB/s` from a 272-times-L2 packed-Q4-code/FP16-scale stream with the production G64
16:1 byte ratio. The streaming result is `98.9477%` of the nominal 640 GB/s bus rate. The retained
report is `profiles/bench/r9700-q4-hardware-peak-20260904.json`, SHA-256
`72e4c91eb9177ff5969cce6c06febde251a4852e1e69bd317b77646ddcb7baa4`. Its raw HIP
`multiProcessorCount` value is 32 scheduler units and does not supersede the architectural 64-CU
hardware description.

For scale only, the production M64xN128 weighted P2048 aggregate's `101.082` tera-issued-operation-
equivalents over `1.109850230 s` is `91.077154` issued TOPS, `11.3609%` of the isolated native-IU4
ceiling. The traffic model's `842.961535 GB` over that duration is `759.527288 GB/s`, or `119.938%`
of the isolated stream rate. This apparent excess is not a physical bandwidth measurement: the
model counts represented source requests, including requests that may be served or combined by
cache, coalescing, and LDS reuse, while the physical transactions and their overlap are unknown.
Consequently these ceilings show substantial unused native-IU4 issue capacity and a healthy
near-nominal isolated memory path, but do not establish production HBM utilization, cache reuse,
or stall freedom. Those production claims remain gated on a matched GL2C/TCP/SQ profile.

The corresponding complete direct-A8/packed-W4-to-IU8 P2048 pipeline was rejected statically
before timing. Its fair M64xN128 expand-once ping-pong kernel used 25,344 LDS bytes, exact eight
signed IU8 sites, and zero private/scratch bytes, but emitted 115 VGPR/occupancy 12. Preventing
four-K16 operand hoisting and replacing tail-aware admitted-shape W loads reduced it only to 111
VGPR with occupancy still 12, missing the fixed <=96 VGPR/occupancy-16 gate. The best compact legal
expansion idiom projected roughly 100 VGPR even with sequential publication, so the route is closed
without GPU or whole-inference timing. The retained evidence is
`profiles/bench/r9700-a8q4-direct-iu8-pingpong-static-rejection-20260905.json`, SHA-256
`d4f2c610b973935aa60b732c3ee39e9dc84a512781839458a5caadcca78f2a2c`; all qualifier
source, checker, test, and build surfaces were removed.

The signed-A4G64 activation challenger halved its matrix instruction count and activation-code
plane, but its complete quantize-plus-M64xN128 path failed operator admission. Weighted P2048
improved from `1121.674698` to `1006.050599 ms` (`1.11493x`), while the N1024 shape regressed at
every measured extent by `1.21819x` through `1.35600x`. The retained report is
`profiles/bench/r9700-a4q4-prefill-m64n128-vs-a8-production-ab-20260904.json`, SHA-256
`2ac938b230197384a476bd57ccfd39d72a1b73c03bee82c58bf3be9df7de5b1f`. Production remains A8;
model-quality gates were not rerun because the operator gate failed and existing A4 quality
evidence already rejects that private activation profile.

The matched `auto` ordinary C1 diagnostic used the all-Q4 G16 B128/S16/tau900 build, an 8,192-token
prompt, a 4,096-token chunk, the Device Graph path, and 256 decoded tokens with speculative execution
disabled. Across three repetitions it measured `239.5738442` prefill tok/s (`34.1947324 s`) and
`8.354688852` decode output tok/s (`30.64198944 s`); whole output throughput was `3.963623176`
tok/s over `64.84022237 s`. The raw report SHA-256 is
`049f3e4712a65ba618f28b47d830a96019cd16dea8dd70a920e1385aca4f1c96` and its manifest SHA-256 is
`a2ce50522910bd1442156383c85f359315c33bc5964578ba3ca28bf9685e104a`. The previously retained
`19.24458338` tok/s result is the MTP3 speculative cell, not ordinary decode.

The current dense all-Q4 G16 ordinary baseline supersedes that `8.354688852` tok/s row for active
optimization decisions. At C1/P8192+G256, spec-none, Device Graph, and one measured repetition, it
reaches `15.13345066` decode output tok/s in `16.91616841 s`. This is a diagnostic baseline until a
source-matched whole A/B supplies repeated candidate/control evidence. The report is
`profiles/bench/ordinary-none-dense-all-q4-g16-c1-8k-current-20260905.json`, SHA-256
`829d19d4eff2364e0d782f0da3cfe7dfa7c3bee313614799f5aa53a333e17f7f`.

The corresponding one-round ordinary trace has a `78.193701 ms` ordinary host marker and 1,806
dispatches with the exact ordinary-range association. Their independent durations sum to
`64.388840 ms`: Q4 WMMA contributes `42.301380 ms` across 321 calls, RMSNorm `13.642834 ms` across
161, split512 PV `4.276113 ms` across 16, and QK `1.030131 ms` across 16. The benchmark report is
`profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/benchmark-report.json` (SHA-256
`7d9da725902c04e208525ae2d02dd33fc696ab452ad6673579a730f6e8f7eee6`), the ROCPD database is
`profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/raw/ordinary-decode-c1-one-round_results.db`
(SHA-256 `2cae81419c0cf360eb032537ca95a4030a1c80df1e9221dc5d6dfcd06f3aa0ff`), and the corrected analysis
is `profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/analysis-v2.json` (SHA-256
`ca8937876bc16259aeaf793a1c6070c5b99dac74d71f2eca4b348ff0b2f7a9d1`). A dispatch-interval union
means only the union of rocprofiler kernel start/end records. Graph dispatch intervals overlap, and
the union is not GPU-active time, GPU wall time, utilization, or CU occupancy; it must not be
subtracted from host-marker time to assign a physical idle fraction. It directs attention first to
the source-matched Q4 Linear candidate and then RMSNorm, with whole A/B as the decision authority.

The native-dot8 T=1 Q4 Linear operator gate passed every selected tuple and reported a robust
ordinary-call-weighted saving lower bound of `14.1025415618 ms` per token. Its report is
`profiles/bench/r9700-a8q4-t1-native-dot8-all-text-20260905.json`, SHA-256
`300626f0d45b5b9bb8f6b420f44b7e5652e3a848738c636959f02f3448d2b5b0`. The seven full-K tuples
were selected from ordinary decode; `[5120,17408]` and `[34816,5120]` also occur in DFlash2, so the
canonical route is shape-owned by Linear rather than caller-specific. Its post-gate production
regression uses nonuniform signed W4 codes in exact N16/K16 storage, nonuniform FP16 scales, dense
varied activations, full-output WMMA parity, spread-row complete-K FP64 checks, status poisoning,
full rewrites, and output canaries.

The source-matched four-pair C1/P8192+G256 ordinary Device Graph gate promoted native dot8. Median
decode fell from `16.9262812925 s` (`15.1244089340 tok/s`) to `12.5301623450 s`
(`20.4307009719 tok/s`). The robust candidate/control upper ratio was `0.7427031671`, the robust
saving lower was `17.0000856399 ms/token`, and the prefill upper ratio was `1.0016298424`; exact
generated tokens and configuration, environment, artifact, and workspace identities matched. The
immutable report is `profiles/bench/r9700-a8q4-t1-dot8-whole-p8192-g256-full-20260905.json`, SHA-256
`19278ca8c8df5d417bbd5d36ce1689760e3cb6dbe606c3a597a6b2f0b847fee1`. There is no build or runtime
selector. T=1 calls outside the seven exact full-K tuples retain WMMA. That promotion left ordinary
decode open for the trace-selected RMSNorm candidate. MTP remains supported regression behavior
and is not an optimization target.

A fresh selector-free production build retained exact 257 generated IDs for P8192+G256 and measured
`20.45440879 tok/s` (`12.51563918 s` decode) in
`profiles/bench/r9700-dot8-production-final-p8192-g256-c1-20260906.json`, SHA-256
`c341f1eeb2f5d5597272dbecc532982b008701c7ff6a22d946541f98c9c94b2c`. The extracted loaded gfx1201
code object has SHA-256 `6a4e9eed7804da2321e112a3c520f65eb3d6a1b88e8352c39f9b364a30033312`
and contains exactly 16 `v_dot8_i32_iu4` instructions and no WMMA in the selected kernel, with 18
VGPR, zero LDS/private/spills, wave32, and a 256-thread maximum workgroup.

The subsequent ordinary-decode promotion is the parallel K5120 RMSNorm CTA at exactly rows 1
through 4.
The immutable operator report
`profiles/bench/r9700-rmsnorm-k5120-rows4-qualification-8192i-20260906.json` (SHA-256
`e58b2e56980fb083548f1c357c26fc98a8a5bae27b1723ccf712451eaf4b8103`) passed the independent
complete-formula FP64 oracle, both gain modes, three epsilon values, ordinary/zero/mixed-magnitude
inputs, Device Graph replay, invalid-input no-write checks, and every row-count timing gate. Its
minimum robust ordinary-round saving lower bound was `11.7782198046 ms`. Production selects this
route only for K5120 rows 1..4; rows 5..127 retain generic RMSNorm, rows >=128 retain the K5120
token8 prefill route, and other feature widths retain their existing routes.

The four-pair source-matched C1/P8192+G256 Device Graph gate reduced median decode from
`12.521242570 s` (`20.44525522 tok/s`) to `9.467485694 s` (`27.03991411 tok/s`). The robust
candidate/control upper ratio was `0.7598847464`, the robust saving lower was
`11.72510638 ms/token`, and the robust prefill upper ratio was `1.0029548902`; generated tokens
were exact in every pair. The immutable report is
`profiles/bench/r9700-rmsnorm-rows4-whole-p8192-g256-full-20260906.json`, SHA-256
`3f5c7a678f29b09537b46ebf7692e6f8d9b422e08f9ffe656367815932b2d0f6`.

The next bounded grouped-PV experiment did not promote. Its reduced fixed-fixture mechanism report,
`profiles/bench/r9700-split512-grouped-pv-qualification-v2-20260906.json` (SHA-256
`f690accdc47bd85096aa412aa415e3f55446e8d9b737b9928cef671b1054ea6d`), is screen-only. The
product-integrated route nevertheless passed exact numerical, rejection, and fixed-address Device
Graph qualification in `profiles/bench/r9700-split512-grouped-t1-direct-20260906.json` (SHA-256
`958b38efaced4b671222c32de93ce72264df6a58aa6fbb5797f0738e965ba65b`). The source-matched
C1/P8192+G32 whole screen then measured a `1.0038023342` candidate/control decode ratio,
`-0.141697625 ms/token` saving, and `1.0033795393` prefill ratio with exact generated tokens. The
rejected report is `profiles/bench/r9700-split512-grouped-t1-whole-p8192-g32-screen-v2-20260906.json`
(SHA-256 `bcd24621462dbb404ecc77b79a6b324f0c4d4b67a7e0afdcbacc69281cd1d9f2`). No full gate was run;
the selector and challenger were removed. The subsequent post-RMSNorm proxy analysis is
`profiles/rocprof/r9700-post-rmsnorm-ordinary-c1-p8192-g256-proxy-plan-20260906/analysis.json`
(SHA-256 `8784fe63db40d13163999077da8b62f049242807cfa3045a88d0412e7a0534ae`).
It validates 256 exact ordinary regions, each with the terminal 37-tuple/1,806-dispatch Device
Graph inventory. Whole-round ratio-of-sums proxies are 56.7040% GL2 hit, 52.0346% TCP hit,
97.5089% wait-any/wave cycles, 1.02066% issue-wait/wave cycles, 25.6970% occupancy, and 13.1582%
VALU busy. More decisively, the 82,176 dot8 dispatches account for 96.6624% of GL2 read requests,
96.4485% of GL2 misses, 83.2272% of TA activity, and 72.0015% of wave cycles. Their symbol-local
proxies are 51.0133% GL2 hit, 21.5963% TCP hit, 99.3061% wait-any/wave cycles, 0.03400%
issue-wait/wave cycles, 28.7011% occupancy, and 6.08721% VALU busy. These overlapping counters do
not establish physical HBM bandwidth, causal stalls, or stall freedom; the separate same-session
4-GiB stream probe reaches 635.9 GB/s but is not inference traffic. The admitted weight-only
non-temporal dot8 challenger preserved exact numerics and passed its static/resource gate, but its
current-identity repeated-buffer direct report rejected it at `-8.26204 ms/token` weighted point
saving and `-8.3750512166 ms/token` conservative lower saving. That report is
`profiles/bench/r9700-dot8-weight-nt-direct-v2-20260906/report.json`, SHA-256
`63ed87fef150a885ed3375548635ba529d8e08cbcb9df0e932b24d248e65034e`. Since repeated buffers do
not exercise the proposed downstream cache-pollution benefit, the terminal decision used a
source-matched three-pair, 12-process C1/P8192+G32 ordinary Device Graph screen. All three paired
decode savings were negative (`-2.683606`, `-2.5499134531`, and `-2.6231869063 ms/token`); the
robust decode ratio upper was `1.0773990082`, robust saving lower was
`-2.8873396633 ms/token`, and robust prefill ratio upper was `1.0186573578`. The rejected report
is `profiles/bench/r9700-dot8-weight-nt-whole-p8192-g32-screen-20260906.json`, SHA-256
`c6a1dea0f0e94d9dfcd461145c83d9161e35313e17e2fe9b0ca041a9b405e31c`. No G256 gate was run. The
private selector, challenger, and temporary qualification tooling were removed while both reports
and raw whole-screen evidence remain immutable. This terminal rejection does not infer physical
bandwidth saturation or stall freedom.

The fresh selector-free build then measured `9.461402437 s` for 256 decode tokens
(`27.05729956 tok/s`) with all 257 generated token IDs retained. The final report is
`profiles/bench/r9700-rmsnorm-production-final-p8192-g256-c1-20260906.json` (SHA-256
`b05db0068a4f1c73ce9c2092443b42f9f48b0fdb80ff8b3335db60cd5bdca74b`) and its executable SHA-256
is `a7c9303bd213fa3dbdb29ca0cee73addef1de8ab6a6e6b231509e25239776425`. A selected-region trace of
that executable records exactly 129 rows1..4 CTA dispatches and 32 legitimate generic RMSNorm
dispatches, so none of the 129 K5120 ordinary-decode calls fell back. The trace database is
`profiles/rocprof/r9700-rmsnorm-production-selected-trace-20260906/raw/rmsnorm-production-selected_results.db`
(SHA-256 `8fe71be97e77c2651cb0c75fe203cedb13f200f8ac76082e4310bbfb855e5370`). The extracted loaded
gfx1201 code object has SHA-256 `c3dcad45559a112f42f07b1d7e87fbd1d494d1d024083efc0498500ee678cd72`;
the selected kernel uses 17 VGPR, 32 bytes LDS, wave32, occupancy 16, and zero scratch/spills.
This selector-free `27.05729956 tok/s` result is the bounded practical ceiling used to close
ordinary base-decode work. It is explicitly not an absolute hardware or physical ceiling and does
not claim the maximum achievable throughput of the R9700.

The existing MTP shortlist head remains Q4G64 with A8G64 activations. MTP stays in exact-output,
state, cache, row-view, and whole-route regression coverage, but a new shortlist-head trace,
alternate head precision, acceptance campaign, or MTP performance optimization is not a final
admission requirement. DFlash/DFlash2 is the preferred speculative path and the only speculative
backend with remaining support and performance work.
Historical MTP diagnostics can retain the theoretical `64*C` round minimum because their counter
sums request lanes, but schema-v7 terminal tooling does not consume or validate those rows.
Extra rounds remain diagnostic acceptance evidence; they do not trigger an MTP head-precision
branch or alter base selection. Schema v7 carries no downstream-readiness status;
selected-profile NIAH and DFlash admission remain separate final gates.
The fail-closed NIAH entry point is
`profiles/bench/post-terminal-niah-prepare-20260905/prepare.sh`; it remains blocked until the
schema-v7 terminal selection exists and then schedules only the required 64K five-position ladder.

The `[131072,5120]` head payload is exactly 356,515,840 bytes (340 MiB) in Q4G64,
713,031,680 bytes (680 MiB) in W8G32, and 1,342,177,280 bytes (1,280 MiB) in BF16. W8 therefore
adds 340 MiB and BF16 adds 940 MiB over Q4. At the complete Text+MTP cache costs of 28,288 bytes per
token for G16 and 27,200 for G32, those deltas correspond before page/allocator rounding to about
12,603/13,107 aggregate cached tokens for W8 and 34,844/36,238 for BF16; divided evenly at C4,
about 3,151/3,277 and 8,711/9,059 tokens per lane, respectively. The retained Q4/A8 shape sweep
measured 2.337/2.265 ms at T=1/3, while the older W8 public-Linear bring-up observed
3.153/9.428 ms. These results use different qualifier harnesses and are only a reason to keep the
comparison conditional; they are not a same-base direct A/B or a precision-selection result. No
source-BF16 MTP acceptance authority currently exists: the checkpoint-direct BF16 scorer treats
MTP as a non-executed comparison label, and the Python artifact reference consumes the artifact's
encoded shortlist head.

This pass resolves the former known-zero cache-counter result, but it does not establish absolute
bandwidth or final-route cache efficiency. The gfx1201 request-size buckets remain known-zero and
base absolute counts may undercount, so no physical GL2/HBM byte rate is derived. The stable profile
also pinned substantially lower clocks; its 49.43 tok/s benchmark timing is profiler control only,
not production evidence. Ordinary performance timing stays under `auto`, and the same focused PMC
pass must be repeated on the selected chunk/profile. ROCm Compute Profiler 3.8.0 still has no
gfx1200/gfx1201 analysis configuration, and complete VALU/LDS/stall counters remain unavailable.
