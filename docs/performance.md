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

Benchmark schema 21 sums each request's active prepare, Vision and prefill service time across
lanes. Aggregate prefill throughput divides all prompt tokens by that summed prefill service
time. Batched decode retains the maximum shared lane duration; complete inference separately
measures elapsed wall time. Service costs are not an additive wall-time decomposition.
Schema 20 instead used the maximum lane prefill duration and overstated aggregate C>1 prefill
throughput. Those concurrent prefill fields are ineligible for selection; retained C1, capacity,
decode and independently measured whole-wall evidence remain valid in their respective scopes.
The stopped attempt and valid raw results are retained in
`profiles/bench/r9700-terminal-base-fp8-context-recovery-20260921/whole/closure.json`.

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

## Remaining projections and adaptive correctness (2026-09-23)

The selected cap26 Q4-head/gate-up-A4 model and Q4 DFlash companion are unchanged.
Eight T5/T6 Linear cells now use the existing successor pipeline: N7168/K5120,
N6144/K5120, N1280/K5120 and N5120/K4096. Independent original-input and
represented-A8 FP64 bounds, exact generic/codec/graph checks and guards pass;
the prior36 instruction/resource streams are unchanged. Cold complete-Op savings
are17–50%; final whole C1 K5 rises98.0082→101.5431tok/s, all repetitions exact
ordinary tokens. These are separate operator and whole-inference claims.

Final unprofiled P4096/G128, chunk2048, auto, one warmup/three repetitions:

| Decode mode | Aggregate tok/s | Per-request tok/s |
|---|---:|---:|
| C1 fixed K5 | 101.54 | 101.54 |
| C1 fixed K4 | 87.64 | 87.64 |
| C4 fixed K4 | 162.22 | 40.56 |
| C4 fixed K5 | 153.98 | 38.50 |
| C4 adaptive, maximum K5 | 160.90 | 40.22 |

All15 repetitions match same-C ordinary tokens. Fresh control C4 fixedK4 is
161.84: that fixed route is essentially unchanged. Prior-pass C1 K4/C4 K5 were
84.71/149.08; these are historical, not fresh paired controls. Ordinary decode
was not changed or remeasured in this pass (previous C1 approximately30.16tok/s).

Adaptive DFlash distinguishes physical captured K from logical output allowance.
Additional padded candidates require measured same-C cost and enough physical
context. Logical ingress/publication masks and the existing minimum-width masked
fallback remain; K3 is not banned. Selection occurs at the actual compact round
boundary. Fresh C4 adaptive rises156.6669→160.8980 aggregate tok/s
(40.2245 per request), mostly selecting K4 on this workload.

Expanded cold-start tests exposed and fixed a preexisting K3/W4 attention seam:
W4 used BF16-Q fused arithmetic where ordinary W1 and W5/W6 used FP8-Q WMMA.
Warmup could train adaptive selection away from K3 and conceal the mismatch.
W4 now shares the same batched attention profile at context64..8191, with proper
workspace/topology planning. The public BF16-Q FP64 oracle is unchanged;
the existing explicit FP8-Q quantization/profile bound and separate arithmetic
check pass at short/4K contexts, C1–4 metadata, graph replay and poisoned rows.
All three renamed device instruction/resource streams are unchanged.

Forty-four real Engine cases now match ordinary tokens exactly: cold fixedK3,
adaptive graph with two prompt-offset sets, and adaptive eager; cases cover short
C1 output limits, near-context C4, and long unequal request budgets. The trace
observes9 padded and4 target-only rows. A six-pending-row K5→K4 transition was
not observed; append/storage mathematics are unchanged and the maximum-append
workspace regression passes. Existing routes outside the admitted W4 domain,
weights, MLP activation policy, fixed cache and prefill/PPL routes are unchanged.

Gate/up remains~25% of C1 traced kernel service, at~558GB/s useful-weight rate
versus the636GB/s stream control. No new scheduler/issue mechanism justified
another challenger; prior depth2 failures remain excluded. This is not proof
of an absolute hardware ceiling. Prefill remains paused; chunk2048 is retained.
Evidence: `profiles/bench/r9700-dflash-projections-tail-20260923/`.

## Compact prefill and concurrent decode optimization (2026-09-23)

Same selected cap26 Q4-head/gate-up-A4 weights and precision as the delivery below.
The existing SiLU→BF16-round→A8-prepare→Q4-down fusion now admits BF16 gate/up output
independently of its producer format, retaining exact T2048/down-A8 guards.
All six mixed-profile NLL sidecars remain byte-exact. C1 codeP4096/G128 ordinary
prefill1494.30→1519.18tok/s; K5 prefill1450.71→1472.19tok/s. Ordinary decode
stays30.15tok/s; this is a materialization/launch saving, not lower precision.

Ordinary T2–4 Q4 MLP gate/up, down and N5120/K6144 output use successor loading through the semantic
Linear owner. Independent FP64/public-input, exact codec, graph/poison/guard and
exact generic-output checks pass; native IU4 WMMA,59–61VGPR, no LDS/private scratch.
Cold complete-Op A/B improves gate/up10–14%, down39–43%, output22–29%.
Concurrent K5 T12/18/24 MLP uses a16-row tiled pipeline: gate/up7–18% and
down41–50% lower complete-Op latency. Full public BF16-input FP64 checks pass
for all14 selected MLP cells, two dedicated down cells and five output cells;
exact generic/eager/graph outputs and codec/poison/guards pass. Tiled kernels
use70VGPR,22SGPR, native IU4, no LDS/private scratch; retained small-width
instruction streams are identical. Whole medians,
auto/G16/dense/chunk2048, warm1/reps3:

| C | Ordinary before, aggregate tok/s | Ordinary after, aggregate tok/s | Ordinary per request | K5 after, aggregate tok/s | K5 per request |
|---|---:|---:|---:|---:|---:|
| 1 | 30.15 | 30.15 | 30.15 | 96.28 | 96.28 |
| 2 | 38.15 | 49.98 | 24.99 | 119.66 | 59.83 |
| 3 | 53.90 | 71.04 | 23.68 | 133.66 | 44.55 |
| 4 | 65.79 | 86.03 | 21.51 | 140.39 | 35.10 |

All18 final C2–4 repetitions exactly match prior same-C greedy tokens; K5
matches ordinary at each C. Lanes use corpus offsets, not identical prompts.
Per-request rates are aggregate throughput divided by C, not individual request latency.
The tiled K5 step improves prior101.56/120.11/125.35 aggregate tok/s by17.8/11.3/12.0%.
C1 rates reuse the unchanged fusion checkpoint; later changes select only C>1
extents and preserve C1 instructions. Final C2–4 ordinary prefill1506–1520tok/s,
K5 prefill1467–1470tok/s; no new prefill arithmetic or precision change.
Candidate safety limits are nonbinding with no CPU throttling. CLI/server/PPL/bench
are rebuilt. This bounded pass does not establish a physical throughput ceiling.
Evidence: `profiles/rocprof/r9700-compact-mixed-speed-20260923/`. Its
`final-verified-summary.json` retains rates and exact prior-token checks;
`baseline-corrected.json` supersedes the explicitly marked first helper summaries.

Numerical limitation found during the subsequent larger-batch experiment:
the full-output public BF16-input oracle (rather than the prior sampled-row norm)
rejects the unchanged generic N4096/K5120/T6 control at 2.0341% relative RMS
against the 2% criterion. This is not a measured model-quality regression, but
the historical complete small-batch qualification failed; do not infer universal
A8 accuracy from the passing model sidecars or narrower MLP checks. The failing
test and receipt are retained. The subsequent principled implementation-profile
qualification below resolves the unexplained operator cutoff, not model quality.

### Four-token concurrent DFlash follow-up

The same successor pipeline now covers K4 MLP widths T10/15/20, corresponding to
C2/3/4. Six new cells pass the unchanged full-output public BF16-input FP64 gate
(maximum relative RMS1.96781%), represented-A8 oracle, exact codec/generic/graph
output, poison and guards. All23 prior kernel instruction streams are unchanged;
new cells use native IU4,70VGPR/22SGPR and no LDS/private scratch. Cold complete-Op
gate/up latency falls7–18%, down44–50%. Same codeP4096/G128/chunk2048, auto,
warm1/reps3: K4 aggregate decode98.32/132.17/142.61→122.14/160.87/161.66tok/s
at C2/3/4 (per-request61.07/53.62/40.41). All nine selected repetitions exactly
match prior same-C ordinary greedy tokens. K4 now beats the retained K5 rates on
this concurrent sample; C1 routing is unchanged and its sample still favors K5.
This is workload-specific selection, not a universal best draft length.
Evidence: `profiles/bench/r9700-compact-followup-20260923/`.

The inherited N4096/T6 failure was localized separately: exact represented-A8
FP64 output itself differs2.01625% from the original BF16-input oracle, versus
2.03407% GPU error. Input quantization error is0.26831%; cancellation amplifies
it at the output. Exact codec/generic checks pass. Arithmetic scheduling alone
cannot repair the2% failure while retaining those codes/scales. One lower-input-MSE
scale refinement worsened output error to2.18842% and was rejected. No numerical
codec was changed. The subsequent numerical review keeps this result explicitly.

The v3 small-batch/down qualification retains the original full-output BF16-input
FP64 oracle and adds an analytically derived A8-profile envelope: independently
measured quantization error plus exact-integer/group-FP32-FMA/BF16-cast error.
A separate per-output arithmetic bound prevents quantization allowance from
concealing kernel defects; both bounds also apply per-token in L2. The derivation
is in `tools/r9700/README.md`. All32 small-batch and two down cells pass, including
exact codec/generic/eager/graph, poison recovery and guards; all102 deliberately
corrupted outputs fail. Maximum public/arithmetic norm-budget fractions are
0.88150/0.45584. The historical2.03407% RMS remains recorded and still fails its
old diagnostic screen. This changes no production arithmetic, weights, precision,
PPL threshold or previously measured model quality. Evidence:
`profiles/bench/r9700-a8-bound-dflash-20260923/`.

Matched chunk check on the frozen preceding build: C1 codeP4096 prefill
1473.39/1523.68/1459.01tok/s at chunks1024/2048/4096; WikiTextP8192 gives
1241.86/1197.63 at2048/4096. Each uses warm1/reps3, same weights, auto and G16.
The2048/4096 runs generate identical tokens on both samples. Keep2048;4096 is
3.6–4.2% slower here, not a theoretically preferred size. No model precision changed.

A bounded high/low-BF16 WMMA PV prototype passed two independent complete-attention
oracle cases and exhaustively reconstructed finite INT4×FP16 values, but complete
attention regressed10.09→11.01ms for P2048/context2048 and42.89→43.98ms for
P2048/context4096. Despite fewer arithmetic instructions and zero scratch spills,
the added decomposition/data/synchronization cost defeated the mechanism.
Rejected; no candidate remains in production and no model-quality claim follows.

Concurrent adaptive K5 initially failed before inference because smaller captured
FP8 verification widths were absent from load-time preparation. Load and Program
binding now share the full captured-width inventory, without changing workspace
maxima or adding preparation during capture. Host contracts pass; repaired C4
adaptive measures154.82 aggregate tok/s and all three repetitions exactly match
ordinary tokens. FixedK4 remains faster on this sample. The first adaptive
repetition records130 rounds,517 drafted tokens and381 accepted (73.69%).

The remaining N5120/K6144 K5 output projections now use the same tiled successor
pipeline at T12/18/24. All eight output cells pass unchanged full-output numerical,
codec, graph/poison and guard criteria; all29 previous instruction streams are
unchanged. New cells use70VGPR/22SGPR, nativeIU4, no LDS/private scratch. Matched
cold complete-Op latency falls26–36%. Fresh C4 whole comparison improves fixedK5
140.488→144.777 aggregate tok/s (+3.05%,36.19 per request), with all six control/
candidate repetitions exact ordinary tokens and identical121 verification rounds.
Candidate repetitions144.660–144.786 versus control140.172–140.533tok/s.
The repaired-adaptive154.82 measurement precedes this last output extension;
its result is not a fresh final-build adaptive timing. FixedK4's measured161.66
still leads; K4/C1 routes retain identical instructions. Final delivered binaries
were rebuilt and host contracts passed. The subsequent A8 bound qualification
above resolves the operator-criterion question without changing model quality.

### Draft projection and adaptive-tail follow-up

The compact C1 draft still used generic Linear for N5120/K17408 down and
N5120/K25600 feature projection at T5/6. Transferring the existing successor
pipeline to these four shape-owned public routes passes complete independent
FP64/A8-profile qualification, exact generic/codec/eager/graph, poison and guards.
The prior32 kernel instruction/resource streams remain exact. New kernels use
nativeIU4,62/63VGPR,22SGPR and no LDS/private scratch. Cold complete-Op latency
falls44.6–47.4%. Matched C1 codeP4096/G128 K5 improves95.9727→97.9506tok/s
(2.06%); all three repetitions on each build match ordinary tokens and use25
request-lane speculative rounds. K4 improves83.1713→84.5524tok/s (1.66%),
again exact ordinary tokens in all six repetitions and30 rounds each.
Same weights, precision, auto, G16, chunk2048,
warm1/reps3; no PPL arithmetic changes.

Fresh fixedK4/adaptive C4 baseline measures161.7119/156.5149 aggregate tok/s
(40.4280/39.1287 per request), with exact ordinary tokens. Adaptive actual
K3/K4/K5 request-lane histograms are3/123/4,2/129/1,2/129/1; these are not graph
launch counts. Matched traces show31 steady C4/K4 rounds differ only0.41% in
kernel service. Most of the difference is at the tail: adaptive uses slow K3/W4
routes as requests drain, while fixedK4 keeps masked/padded W5 execution.
The current budget-clamp policy is intentional and tested; no numerical or
steady-state chooser bug is demonstrated. Five redundant compact copies cost
about8us/round, too small to explain the gap. Keep fixedK4 for this measured
concurrent workload. Do not banK3 or change adaptive semantics from this one
sample; physical-width/logical-budget separation is a future policy experiment.
Evidence: `profiles/bench/r9700-a8-bound-dflash-20260923/`.

After removing the duplicate verify-only Linear API/kernel and rebuilding all
deliverables, the final public four-cell qualification passes again. Its linked
projection code object matches the qualified candidate exactly, and the previous
target-down instruction/resource streams match the consolidated route. Final
matched confirmation (same workload, median of three repetitions):

| Mode | Before, aggregate tok/s | Final, aggregate tok/s | Final per request |
|---|---:|---:|---:|
| C1 K5 |95.97|98.03|98.03|
| C1 K4 |83.17|84.71|84.71|
| C4 K5 |144.82|149.08|37.27|
| C4 K4 |161.71|161.74|40.43|
| C4 adaptive maxK5 |156.51|156.28|39.07|

C1 K5/K4 gains are2.15%/1.85%, C4 K5 gains2.94%; fixedC4 K4 is unchanged.
Adaptive has no demonstrated gain (repetition ranges overlap); its tail limitation
remains. All15 final repetitions exactly match same-C ordinary tokens. Final
scope limits are nonbinding with zero recorded throttling. Focused execution-state
capacity and benchmark host tests, plus two Python matrix tests, pass. Ordinary
T1–4 decode and large-prefill routes are unchanged; no new speed or ceiling claim
is made for them. `final-summary.json` binds the final confirmation.

## Selected compact mixed-profile delivery (2026-09-23)

The user selected the compact cap26 Q4 embedding/head model with gate/up-only A4 large
prefill. Fresh builds now default to `NINFER_R9700_Q4_PREFILL_A4_FAMILIES=1`, global Q4
bits8, W8 bits8; delivered `build-r9700` was explicitly reconfigured and CLI/server/PPL/bench
rebuilt. T>128 full-K N34816/K5120 Q4 uses the already-qualified fast A4 route; other Q4
calls, including ordinary decode and small DFlash verify, stay A8. No new kernel arithmetic
or weight conversion was needed. Broader A4 profiles remain evaluators, not defaults.

Both artifacts and their creation receipts remain installed in
`/ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-q4-fp8-selective-cap/`:
base15,793,065,984 bytes; canonical-Q4/BF16-codebook DFlash companion17,002,543,616 bytes.
The adjacent README logs exact creation/build/run commands. These evaluation identities are
not renamed, and the independent BF16-source production-admission status is unchanged.

All six final-build PPL NLL sidecars are byte-identical to the retained mixed profile:
worst per-text prefill regression+1.880% versus NVFP4, decode-0.642%. The10-versus6 newly
severe technical-prefill tradeoff remains; this is not a universal quality-equivalence claim.

Delivered companion, codeP4096/G128, C1, chunk2048, context4240, dense/G16, power`auto`,
Device Graphs, one warmup/three measured repetitions (median rates):

| Mode | Prefill tok/s | Output decode tok/s |
|---|---:|---:|
| Ordinary |1494.30|30.15|
| DFlash K4 |1448.40|83.20|
| DFlash K5 |1450.71|96.05|
| DFlash adaptive, maxK5 |1459.47|92.53|

Every repetition in all modes produces exactly the same128 greedy tokens as ordinary decode.
K5 was fastest on this sample; the prior short-chat uniform-A8 result favoredK4. Do not treat
either fixed K or this small adaptive comparison as a universal optimum. The selected mixed
profile changes prefilled state, so old uniform-A8 acceptance/speed numbers are not guarantees.
The memory safety scopes recorded no high/max/OOM events and no CPU throttling; benchmark
limits were nonbinding. All heavyweight work ran serially, with four build jobs (maximum14).

Checks: selected dispatch thresholds/padding/exclusions, FP8 execution-state contract, six exact
quality sidecars, real companion ordinary/K4/K5/adaptive execution and repeated-token parity.
Independent review passed. Uniform-control preparation now rejects the mixed-default build
before snapshotting; configure family0 and complete the build to reproduce uniform controls.
Evidence and replay script: `profiles/bench/r9700-compact-mixed-delivery-20260923/`.
This delivers the selected recipe; it does not reopen paused XAttention or broad kernel sweeps.

## Endpoint precision and mixed-activation search (2026-09-23)

Completed 27 configurations / 162 finite, aligned scoring cells on the R9700, using the
unchanged three-text 5090 reference and the same six scoring spans below. This is measured
quality plus theoretical byte/work accounting: **no new speed measurements or promotion**.
Six exact-copy artifacts replace embedding, output head, or both with W8 on the 26-protection
and smaller 22-protection FP8 bases. The 26-protection variants were crossed with all six
activation profiles; smaller-base A8 controls led to focused gate/up and gate/up+attention
tests with Q4 or W8 head. Two gate/up W8-head controls used BF16 rather than A8 head activations.

Key comparisons (decimal GB, base-only artifacts without DFlash; worst per-text PPL change
versus NVFP4, not a comparison between the different prefill/decode scoring spans):

| Base / endpoints / activation | File GB | Worst prefill | Worst decode | Logical weight stream GB/step |
|---|---:|---:|---:|---:|
| Retained cap26 / Q4 / A8 default |15.793|+0.126%|+1.814%|14.272|
| Retained cap26 / Q4 / gate-up A4 |15.793|+1.880%|-0.642%|14.272|
| cap26 / W8 head / gate-up A4 |16.468|+1.116%|-0.879%|14.947|
| cap26 / W8 head / gate-up+attention A4 |16.468|+1.670%|+2.090%|14.947|
| cap22 / Q4 / gate-up A4 |15.542|+2.679%|-0.098%|14.021|
| cap22 / W8 head / gate-up A4 |16.218|+2.156%|+0.197%|14.696|

Only cap26 W8-head/gate-up passes the preferred 2% screen among the new weight/profile
combinations. Its BF16-head control also passes (+1.184% worst prefill); cap22's remains a
near miss (+2.138%). All six decode NLL files are byte-identical between head A8/BF16 builds:
ordinary T1 already uses BF16 activations. Batched PPL scores many head rows, unlike ordinary
prefill's final logits. This sensitivity test is not evidence for faster A16 execution.

No embedding-only or both-endpoint cap26 profile passes. MLP-wide A4's best endpoint variant
still has a 3.490% worst regression; all-projection A4's best is 5.294%. These results do not
support extending broad A4 to the less-protected cap22 variants. Near misses remain evidence,
not statistical proof of a quality cliff. The 2% screen is not the BF16-source production gate:
cap26 W8-head/gate-up has new severe counts 5/10/5 prefill and 1/1/0 decode (Wiki/technical/code),
versus retained Q4-endpoint/gate-up's 5/10/3 and 1/2/0. Average PPL does not erase that tradeoff.

Theoretical ranking and next optimization priorities, not additional authorized experiments:

1. Keep cap26 Q4 endpoints + gate/up A4 as the compact speed-oriented shortlist leader.
   Its modeled Q4 integer-product work is 75.98% of same-weight A8; gate/up+attention is
   74.86%, MLP-wide 63.97%, and all eligible projections 50%, but the broader routes fail
   this screen. Optimize the qualified gate/up route before broadening A4 coverage.
2. Retain cap26 W8-head/gate-up as a PPL-oriented alternative: it improves all six aggregate
   PPLs over Q4-endpoint/gate-up, but costs 675,430,400 extra file/head bytes (644.14 MiB)
   and 4.73% more logical ordinary weight-stream bytes. It has the same body-work proxy,
   not a demonstrated speed advantage. Any future performance work must include the tiled
   W8 head's bulk A8 and small-T BF16 routes, and qualify DFlash verify/acceptance separately.
3. Retain cap22 W8-head/gate-up as a near-miss capacity/quality option, not a promoted winner.
   Revisit with broader held-out quality evidence if accepting its roughly 2.15% worst span
   would change the decision; do not rerun unchanged cells or relax the screen implicitly.

Embedding alone adds the same 644.14 MiB but just 2,720 extra logical bytes per token; it did
not buy the hoped-for quality headroom. Stream accounting includes all Text layer weights,
the complete head, final norm, and one embedding row, excluding KV, activations, cache effects,
rereads and launch/reduction costs. Q4 product work excludes FP8 and head arithmetic. It cannot
predict tok/s or establish a global optimum. At this search's closure the default was unchanged;
the subsequent user-selected delivery above promotes only gate/up A4, not the W8-head alternative.

Evidence/commands: `profiles/ppl/r9700-endpoint-precision-20260923/`, `comparison.md/json`,
and `tools/ppl/endpoint_precision.py`. Original compact controls remain in
`profiles/ppl/r9700-fp8-capped-selection-20260923/`; immutable NVIDIA fixtures are unchanged.
Verification includes exact payload readback, all six real-artifact bindings, five focused
converter tests, the FP8 execution-state contract, and per-cell profile/identity/NLL checks.
After a shared-host memory/I/O stall, resumed conversion/build/GPU phases were strictly serial;
builds used four jobs, with 24 GiB job memory and zero job-swap limits. These constrained runs
must not be used as speed evidence. The larger NVIDIA +328 MiB artifact remains unmeasured.

## NVIDIA-aligned precision follow-up (2026-09-23)

The bounded follow-up completed42 finite/aligned quality cells: four broader A4 activation
profiles on selective-cap, and three smaller weight recipes with uniformA8. Same immutable
5090 reference,4096-token texts, chunk2048, prefill2047 and decode128 scored positions as below.
The local admission criterion remains at most2% worse PPL on every text/schedule, with newly
severe positions reported separately. No new candidate passed; none was timed or promoted.
Default weights/execution and the frozen NVIDIA reference remain unchanged. This does not
establish a global precision optimum or make2% a statistical quality cliff.

| Candidate | File GB | Worst prefill PPL regression | Worst decode PPL regression | New severe prefill / decode (Wiki, technical, code) |
|---|---:|---:|---:|---|
| Selective-cap, A4 all eligible projection shapes |15.793|6.418%|5.197%|4,9,2 /1,2,0|
| Selective-cap, A4 MLP gate/up+down |15.793|4.180%|5.256%|4,9,2 /1,2,0|
| Selective-cap, A4 attention input only |15.793|0.739%|3.719%|5,3,7 /1,2,0|
| Selective-cap, A4 gate/up+attention input |15.793|2.300%|2.419%|5,6,5 /1,4,0|
| Default-NVIDIA15 protection locations, uniformA8 |15.424|0.526%|5.911%|3,5,3 /1,3,0|
| Three output protections only, uniformA8 |15.217|0.833%|4.869%|4,6,5 /1,2,0|
| Selective-cap without late MLP protections, uniformA8 |15.542|0.336%|2.232%|5,4,5 /1,4,0|

All A4 changes apply only to full-K public Q4 Linear at T>128; decode/verify remainsA8.
Decode quality nevertheless changes through prefilled KV/recurrent state. This is why prefill
PPL alone cannot select the profile. Attention input is N7168/K5120, excluding GDN inputs.
The existing gate/up-only mixed profile remains within2% with its previously measured prefill
win, but retains its severe-position tradeoff; uniformA8 remains default.

Selective-cap already protects the same large Text projection locations as NVIDIA's+328MiB
artifact:15 split counterparts of the default's9 BF16 projections, plus11 split counterparts
of the larger variant's8 added FP8 projections. AMD stores all26 at FP8 and keeps endpointsQ4;
NVIDIA retains the original BF16 protections and W8 endpoints. Identical protection locations
do not imply equivalent integer-A4 and NVFP4 activation errors.

Scale payloads, excluding alignment/runtime KV/activation images:

| Artifact | Group scale MiB | Row scale MiB | BF16 DFlash codebooks MiB |
|---|---:|---:|---:|
| AMD selective-cap base |807.747 FP16|0.875 FP32|none|
| AMD selective-cap+Q4 DFlash |861.301 FP16|0.875 FP32|242.5|
| NVIDIA default+NVFP4 DFlash |1525.234 FP8 +211.895 FP16|none|242.5|
| NVIDIA+328MiB |1478.359 FP8 +211.895 FP16|0.244 BF16|242.5|

NVIDIA additionally has about2KiB of FP32 divisors. Codebooks are not scales; the larger
artifact's344,311,808 extra bytes are overwhelmingly increased weight precision, not BF16
row scales. Its exact matched PPL remains unmeasured here because no NVIDIA GPU is visible.

Verification: exact converted payload readback; real-artifact binding/PPL for all three new
recipes; host selector0..5 and FP8 workspace contracts; public-input FP64/independent codec,
finite/poison and arena checks for all six A4 projection shapes atT2048 and threshold cases.
The explicitlyA8 fused-down route now respects the selected down activation width. No kernel
arithmetic was rewritten. Evidence/commands: `profiles/ppl/r9700-nv-aligned-precision-20260923/`;
runner `tools/ppl/select_nv_aligned.py`. Rejected candidates and sidecars are retained.

## FP8-capped compact base selection (2026-09-23)

### Subsequent decode optimization checkpoint

On the new17.0025GB selective-cap canonical-Q4 DFlash companion, matched C1 short-chat
P89/G128/chunk2048, maxctx1024, auto, Device Graph, warm1/reps3 measured **30.7514 ordinary,
73.7744 K4, and69.1459 K5 output tok/s**. All three repetitions in both speculative modes
match the same-artifact ordinary generated tokens exactly. K4 has37 rounds/90 accepted/one
fallback; K5 has39 rounds/88 accepted/one fallback. These rates exceed the requested30/60
targets for this workload, not every context or concurrency. Evidence:
`profiles/bench/r9700-compact-decode-20260923/dflash-loadstage4-chat/`.

The matched base-only code P4096/G128 baseline20.2346 ordinary tok/s improves to29.7623 with local-Q4
normalization/residual fusion and ordered attention-PV load staging; all repeated output tokens
remain exact. Further staging experiments are active, so this is a checkpoint, not a ceiling.
Mixed recipes now select the existing fused Q4 operations per local format/shape; explicit
ordinary-decode intent excludes speculativeT1 rather than testing unrelated matrix inventories.
PV retains its exact serial FP32 FMA order and decoded INT4/FP16 values; page lookup is hoisted
and independent reads staged before accumulation. Full represented-input FP64 plus supplementary
exact serial checks pass at4096 and64–67, with final32-load staging checked across all16layouts
at65; context3 fails an unrelated FP8-query attention
comparison afterPVpasses and is retained without relaxing that criterion.

The matched **companion** at P4096/G128/chunk2048/context4240 subsequently measures
28.6544 ordinary,68.8281 K4 and83.2839 K5 outputtok/s, warm1/reps3 and exactordinarytokens
throughout. K4 has29rounds/99acceptedtokens; K5 has24rounds/104acceptedtokens, nofallbacks.
This is not the same resident artifact/capacity as the base-only29.7623 row. Additional graph
frontier-update accounting fixes the reproduced75MiB-versus74MiB startup deficit; observed
allocation checks remain unchanged. Evidence: `dflash-graph-updates-code/` under the package above.
A measured-region graph-only trace attributes27.77% of K5 decode-kernel time to batchedPV.
Sharing the ordinary ordered-load-staging body improves complete attention W5/W6 at4101/4102
from0.9007/0.9436ms to0.2818/0.3450ms. Independent completeFP64, all-row serial-bit parity,
Device Graph exactness and guard checks pass at short and long context. ISA:111VGPR,
48SGPR, noLDS/scratch, occupancy12. Matched wholecompanion ordinary/K4/K5 now measures
29.8313/89.6352/104.8799 outputtok/s with identical tokens and acceptance in all3repetitions.
Evidence: `dflash-attention-{long,short}-staged32/` and `dflash-staged32-code/` in the package.
These are workload-specific C1 results, not a proven ceiling or a C2–4 admission.

Mixed-A4 cooperative single-bank gate/up preserves all six prior mixed-precision NLL sidecars
byte-for-byte, but wholeprefill1376–1389tok/s remains belowuniformA8. ExactN34816/K5120/T2048
completequantize+matrix timing gives A4 7.4804ms versus A8 5.6684ms.
The selected A4 ping/pong consumer reduces this to5.6628ms (paired A8 5.8518ms);
wholeprefill wiki/technical/code=1498.1668/1492.7539/1488.6179tok/s. Public-input FP64 at129/2048,
exactcodec/represented-code FP64, and all six retained mixed-precision NLL sidecars pass.
ISA uses4signedIU4 instructions/group,73VGPR,13056LDS,no scratch,occupancy16. A8's original
owning kernel and emitted instruction schedule are preserved. Wholeevidence:
`speed-selective-cap-pingpong-original-a8_a8-{wiki,technical,code}/` in the package above.
Retain uniformA8 as the default precision choice; the mixedrecipe still adds10versus6 newly
severe technical-prefill positions despite passing the same2%NVFP4PPL screen. The fast A4
kernel is selected within the explicitly enabled mixed evaluator, not by a runtime flag.

Further DFlash row-sharing PV passes completeFP64/serial/graph checks and improves W5/W6
Op latency to0.2440/0.2831ms, but wholeK4/K5=89.4565/105.4945tok/s shows no material consistent
gain over staged loads. It is rejected and removed; microbenchmark reuse alone is insufficient.

The final base-decode change stages two raw G64 groups in the N5120/K17408 projected-residual
operation. Integer dot order, serial FP32 group FMAs, explicit BF16 delta and residual addition
remain unchanged; K6144 keeps its incumbent. Full5120-output independent FP64, exact unfused
and graph parity, guards and poison checks pass. Three cold weight copies with80MiB scrub
measure0.1336815→0.1277205ms. ISA has32nativeIU4dot8 instructions,58VGPR,18SGPR, noLDS/scratch,
occupancy16, and both groups' raw loads precede computation.
Balanced C1/P4096/G128/chunk2048/maxctx4224 base-only ABBA whole measurements give controls
29.8817/29.8317 versus selected30.1973/30.1730 outputtok/s; all12 measured runs retain every
baseline token. Evidence: `residual-stage2-whole/` and `residual-down-{before,stage2}.json`.
Depth4 remains numerically exact but only reaches0.1264000ms, below the0.2ms/token material
saving bound over depth2; it is removed. Prior losing CTA and gate/up-prefetch sweeps are not
repeated. These results establish the requested C1 targets on the stated workloads, not memory
bandwidth saturation, a universal throughput guarantee, or an optimization ceiling.
The first final-linked confirmation measured29.4460tok/s with exact tokens. A follow-up balanced
old/final/final/old comparison measured30.1642/30.1673/29.5064/29.6305. Both binaries show
run-to-run variation, not a consistent final-build regression: their entire KV/GDN objects and
named dot8/residual/GDN-pair/A8-prefill machine code and resource metadata are identical.
Report approximately30tok/s at4K, with observed29.45–30.20 range, not a guaranteed30tok/s
minimum on every run. All output-token checks still pass. Evidence: `ordinary-final-code/`
and `final-linked-comparison/`; no failed or slow result was discarded.
Final-linked shortchat confirmation measures30.7130 ordinary,75.5710 K4 and70.7222 K5
outputtok/s, all three repetitions exactordinarytokens (`final-linked-chat/`). Both C1 targets
are satisfied on this delivered build and workload. CLI, serving, PPL and benchmark rebuilds
pass; uniformA8 and mixedA4 public dispatch, standaloneA4 regression and selected down Op
qualification pass. No further material winner remains in the bounded candidates above.

The companion is produced with `compose_fp8_capped_dflash`, preserving every selected base
payload and copying only the donor's66 DFlash objects (32Q4/34BF16). Both selector codebooks
stay BF16. Payload readback, real1190-object strict binder, missing/wrong companion rejection,
workspace and host routing checks pass. This changes neither the retained base quality selection
nor the separate final BF16-source admission requirement. The original recipe-comparison rates
below predate these kernel changes and remain historical comparison evidence.

### Recipe and quality selection

Original selection for subsequent optimization: **Q4/FP8 selective-cap, uniform Q4 A8 execution**.
The selected local delivery above now uses gate/up-only A4 prefill with these same weights.
The15,793,065,984-byte artifact is saved at
`/ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-q4-fp8-selective-cap/qwen3.8-27b-r9700-q4-fp8-selective-cap-n16k16-eval.ninfer`.
It is10.7% smaller than the17.68 GB selective base and26.7% smaller than four-role.
It does not contain a DFlash companion and does not replace final BF16-source production
admission. Existing model files remain unchanged.

The recipe retains selective's26 protected projection positions, replaces its15 BF16 matrices
with row-scaled FP8, and returns embedding/output head to Q4. Its other11 FP8 projections stay
FP8. Everything else retains all-Q4 bytes or existing direct norm/control precision. This caps
large promoted weight matrices, not norms, cache, GDN state or DFlash selector codebooks.
The exact fixed inventory and converter are in `qwen3.8-27b-artifact.md` under maintainer docs.

The immutable5090 reference is Git-tracked in `tools/ppl/fixtures/nvfp4-5090-20260922/`
(`0b00efa1`): three exact4096-token inputs and six NLL sidecars/reports. These are the same
prefill2047-position and short-decode128-position spans as the comparison below, with identical
cache and measurement settings. All24 new recipe quality cells are finite and aligned.

| FP8 promotions over all-Q4/A8 | File GB | Worst prefill PPL change vs NVFP4 | Worst short-decode PPL change | Prefill tok/s | Ordinary decode tok/s |
|---|---:|---:|---:|---:|---:|
| Early6 attention QK/GV pairs | 15.38 | +0.63% | +5.79% | not timed | not timed |
| All attention QK/GV pairs | 15.72 | -0.01% | +4.33% | 1442–1456 | 20.36–20.45 |
| All attention + GDN QK | 16.20 | +0.11% | +4.59% | 1423–1455 | 19.45–19.50 |
| Selective-cap26 projections | 15.79 | +0.13% | +1.81% | 1431–1446 | 20.18–20.20 |

Rates span per-text medians, C1/P4096/G128/chunk2048, ordinary graph, auto, warm1/r3.
All nine admitted speed cells have exact repeated tokens. The early-attention candidate exceeds
the5% secondary PPL screen and was not timed. Selective-cap alone meets the preferred2% screen
in both schedules among the four new recipes. Its PPLs (WikiText/technical/code) are
6.455954/9.389574/2.255864 for prefill and6.680329/22.708779/3.239396 for short decode.
Its newly severe positions versus NVFP4 are5/6/6 and1/3/0 respectively: aggregate PPL agreement
is not the separate BF16-source severe-position production gate.

Relative to the old tiled selective companion's ordinary mode, selective-cap improves prefill
about54% and decode about4%. It is not fastest on every axis: all-Q4/A8 still leads decode
(~21.9 tok/s) and four-role leads prefill (~1627 tok/s). The benefit is a compact candidate with
tighter measured PPL agreement, not proof of a global Pareto optimum or hardware ceiling.

The bounded mixed evaluator applies A4 only to Q4 MLP gate/up N34816/K5120 at T>128;
decode/verify and other Q4 operations stay A8. Its same-artifact PPLs are
6.602872/9.655792/2.282552 prefill and6.705888/22.160924/3.216515 short decode: all within2%
of NVFP4. Technical-prefill newly severe positions rise from6 to10, so this is not numerical
equivalence. Mixed A4 measures900–901 prefill tok/s and20.18–20.19 decode tok/s with exact
repeat tokens in all three speed cells. Uniform A8 remains the selected default; the generic
A4 route needs a separately qualified cooperative prefill implementation before it can earn
promotion on speed. This identifies an accuracy-tolerable optimization target, not a speed win.

Verification: complete artifact payload readback and strict binder rejection checks for all four
recipes; FP8 execution-state and host planning contracts; unchanged all-Q4 NLL control; mixed
real-shape public-input FP64 oracle within its A4 error bound plus exact quantized-reference
checks at768 tile/shape-boundary outputs, full finite/poison and arena checks; both-build
selector/report tests. The installed artifact/current default build reproduces every frozen
code-sample NLL exactly. The mixed evaluator corrected an exact-span A4 workspace binding
failure before any model evaluation. No kernel arithmetic was rewritten.

Evidence and exact commands: `profiles/ppl/r9700-fp8-capped-selection-20260923/`, including
`quality-summary.json`, `comparison.md/json`, `mixed-quality-summary.json`, and `selection.json`.
Drivers: `tools/ppl/select_fp8_capped.py`, `tools/convert/qwen3_8_27b_r9700/convert_fp8_capped.py`.
The four-recipe integration is committed as `03240dd5`; the mixed evaluator is `0a439704`.
Further kernel tuning and DFlash integration are separate follow-ups, not completed claims.

## Matched NVFP4 quality and AMD recipe comparison (2026-09-23)

The bounded comparison uses the user's standard 5090 artifact
`/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`
and a frozen scorer from `ninfer-dylan2` at `dfc818ae32c4e0e1ded7b9252b87930338fb8455`.
Its source, build, and model bytes were not changed. Three identical4096-token inputs cover
WikiText, technical prose, and C++ source. Both embedded tokenizers and native tokenization
match. Prefill PPL scores2047 next-token positions after2048 warmup tokens; a separate
teacher-forced decode check scores the last128 positions. Compare recipes within each span,
not prefill PPL against decode PPL. No speculation participates in either quality schedule.

AMD uses the fixed FP8-K/INT4-V/FP16-scale G16 dense cache; NVIDIA uses its standard NVFP4
cache without Sage/sparse attention. This is an end-to-end product comparison, not isolated
activation quantization. A4/A8 refers to Q4 activation width; W8 activation configuration stays8.
The A4 evaluator was corrected to exclude fused routes hardcoded to A8 (`3540705b`), and both
activation profiles pass the owning N16/K16 public Q4 dispatch oracle. A4 has fewer optimized
routes; its timings do not establish the hardware limit of four-bit arithmetic.

AMD speed uses C1, fresh P4096/G128, chunk2048, context4224, ordinary Device Graph execution,
auto power, one warmup and three measured repeats per text. Model loading is excluded. Rates
below span the three per-text medians, not pooled scorer runtime or short-context decode.
The host-only graph profile-update allowance fix (`44013616`) enabled affected startup cells;
original failures are retained and explicit retry receipts select corrected runs. Kernel math
and successful earlier measurements are unchanged.

| Recipe / Q4 activation | File GB | Worst prefill PPL change vs NVFP4 | Worst short-decode PPL change | Prefill tok/s | Ordinary decode tok/s |
|---|---:|---:|---:|---:|---:|
| All-Q4 / A4 | 15.17 | +12.13% | +22.15% | 674–676 | 15.85–15.87 |
| All-Q4 / A8 | 15.17 | +0.74% | +2.81% | 1437–1440 | 21.90–21.92 |
| Source-MSE mixed Q4/W8 / A4 | 22.88 | +1.29% | +0.70% | 656–657 | 10.62 |
| Source-MSE mixed Q4/W8 / A8 | 22.88 | -0.99% | +0.83% | 951 | 11.15–11.18 |
| Selective / A8, old head | 17.68 | -0.48% | +4.67% | 929–941 | 18.07–18.09 |
| Four-role FP8/Q4 / A8 | 21.55 | -0.77% | +1.57% | 1625–1628 | 15.12 |
| Selective / A8, current tiled head | 18.89 | -0.48% | +4.67% | 928–939 | 19.41–19.42 |

Sizes are decimal file GB, not resident VRAM. Current selective includes its unused DFlash
companion; the base portion is approximately17.68 GB. Selective changes28 all-Q4 matrices:
two W8 embedding/head, fifteen BF16, eleven FP8. Mixed retains its available older RowSplit
W8 head; four-role has a Q4 head. This compares available complete implementations, not
intrinsic format performance. The tiled selective head preserves measured PPL exactly and
improves ordinary decode about7.4% over its older counterpart.

Prefill PPL, lower is better:

| Recipe / activation | WikiText | Technical | C++ |
|---|---:|---:|---:|
| 5090 NVFP4 reference | 6.48105 | 9.57639 | 2.25303 |
| All-Q4 / A4 | 7.14944 | 10.73758 | 2.41774 |
| All-Q4 / A8 | 6.45516 | 9.59965 | 2.26963 |
| Mixed / A4 | 6.56223 | 9.53078 | 2.28199 |
| Mixed / A8 | 6.34720 | 9.16747 | 2.23071 |
| Selective / A8, either head | 6.41437 | 9.35286 | 2.24231 |
| Four-role / A8 | 6.41363 | 9.33412 | 2.23561 |

For a5% per-text PPL screen across both schedules, all-Q4/A8 is the practical light/fast
ordinary-decode choice. With a2% screen across both, four-role has the best measured prefill
and decode rates among eligible rows; mixed/A8 has the lowest prefill PPL but costs capacity
and speed. These are diagnostic PPL screens, not the existing BF16-source severe-position
admission gate. All-Q4/A4 is unattractive: worse PPL and slower than all-Q4/A8. No production
artifact was replaced, and this does not establish a DFlash winner for a different base recipe.

Strict per-text prefill-quality/severe-position/speed dominance retains all five A8 rows;
including short-decode severe positions retains all seven rows. Small sample-specific severe
counts and timing differences therefore make the mathematical frontier less selective than
the practical screens above. Neither frontier asserts statistical significance. Raw per-text
newly severe positions and timing ranges are retained rather than hidden by an aggregate.

Exact commands, matched inputs, aligned NLLs, per-repeat speeds, binary identities, and reports:
`profiles/ppl/r9700-nvfp4-multitext-pareto-20260922/`. The reproducible driver is
`tools/ppl/compare_nvfp4.py`; `comparison.md` and `comparison.json` contain per-text matrices
and strict measured frontiers. These three local samples do not replace BF16-source production
admission, general capability testing, or DFlash acceptance/throughput qualification.

## Corrected concurrent decode qualification (2026-09-22)

The unchanged selective-protected tiled-head artifact named below passes the finite P89/G128
greedy workload at C1–4 with DFlash K4/K5 exactly matching fresh same-C ordinary tokens.
All12 graph cells use warm1/r3; all9 C2–4 eager cells use warm0/r1. Repeated graph outputs
and eager/graph tokens and speculative accounting agree exactly. R9700/gfx1201, ROCm10,
fixed G16 cache, context1024, chunk4096 and power auto are unchanged.

| Concurrency | Ordinary aggregate tok/s | DFlash K4 aggregate tok/s | DFlash K5 aggregate tok/s |
|---|---:|---:|---:|
| 1 | 27.51 | 58.75 | 57.29 |
| 2 | 34.04 | 69.31 | 64.42 |
| 3 | 45.34 | 87.94 | 77.15 |
| 4 | 57.67 | 97.99 | 86.73 |

These are medians of three unprofiled graph repetitions; divide aggregate throughput by C for
normalized per-request throughput, not measured request latency. Added lanes rotate the corpus.
K4 wins at every measured C; C2 K4 now exceeds C1 aggregate throughput by18%.
The correctness changes cost about8% C1 speculative speed versus the earlier measurement below.
C4 speculative rates also decrease; the old faster numbers failed exact-greedy qualification.

The corrections canonicalize small-token K5120 RMSNorm, protected BF16 Linear, projected GDN
controls, and accepted-token replay-fold normalization/dot association. Independent FP64 Op
oracles pass; same-input checks establish cross-width equality and replay state matches ordinary
snapshot state exactly. The artifact, output-head math and tolerances are unchanged, and target
verification is still batched. C4 ordinary output changes from its old width-dependent arithmetic;
shared ordinary lanes now match across C1–4. This finite workload is not universal context parity,
terminal artifact/PPL selection, or proof of bandwidth saturation or exhausted speed headroom.

Evidence, exact launch commands and retained failed localization runs:
`profiles/bench/r9700-dflash-concurrent-correctness-20260922/` (`run_final.py`,
`summarize_final.py`, `final-summary.json`, per-cell reports). Output directories are create-only.

## Historical pre-correction bandwidth and concurrency follow-up (2026-09-22)

The earlier selected build was measured without kernel or recipe changes. Its numerical failures
are resolved only by the subsequent corrected qualification above. The fresh native
4 GiB read-stream probe sustained **636.0 GB/s median** under `auto`, with exact checksum;
write/copy medians were 588.0/548.6 GB/s. This is a streaming reference, not measured inference
traffic. The installed gfx1201 counters do not reliably establish absolute GDDR6 byte rates.

Descriptor- and schedule-based C1 weight accounting gives the following conditional comparison.
It excludes inactive tensors and full-table embedding/codebook scans, counts selected rows and
known weight replays, and uses observed useful decode outputs per speculative round (including
correction/bonus tokens), rather than nominal draft width.

| C1 mode | Measured decode tok/s | Modeled weight GB/round | Useful tokens/round | Weight-only stream reference tok/s |
|---|---:|---:|---:|---:|
| Ordinary | 27.53 | 15.482 | 1.000 | 41.08 |
| DFlash K4 | 63.76 | 17.066 | 3.368 | 125.53 |
| DFlash K5 | 62.15 | 17.113 | 3.459 | 128.57 |

These are **not achievable-speed predictions or proven ceilings**. The model assumes reuse
within weight calls and omits some rereads, activation/intermediate traffic, state work, compute,
synchronization and scheduling. Its logical weight rates of 426/323/307 GB/s must not be called
physical bandwidth utilization. Neither saturation nor exhaustion of optimization headroom is
established. The complete source-accounted assumptions and fresh C1 reports are retained in
`profiles/bench/r9700-decode-bandwidth-concurrency-20260922/bandwidth/fresh-c1-accounting.json`;
`run_stream.py` and `account.py` in that directory reproduce the respective measurements/model.

### C1–4 measurements

The admitted saved binary `r9700-bf16-controls-wave32-20260922/whole-inference/candidate-ninfer_bench`
and the selected tiled-head artifact below were held fixed: greedy, Device Graph, G16 fixed cache,
P89/G128, context1024, chunk4096, one warmup and three repetitions per cell under `auto`.
Each cell ran in a separate process, serialized on the GPU. Values are mean aggregate decode
tok/s, with normalized per-request throughput (`aggregate / C`) in parentheses, not measured
individual request latency.

| Concurrency | Ordinary | DFlash K4 | DFlash K5 |
|---|---:|---:|---:|
| 1 | 27.53 (27.53) | 63.76 (63.76) | 62.15 (62.15) |
| 2 | 33.79 (16.90) | 62.25 (31.12) | 61.66 (30.83) |
| 3 | 44.67 (14.89) | 89.11 (29.70) | 82.08 (27.36) |
| 4 | 56.82 (14.21) | 108.92 (27.23) | 98.77 (24.69) |

Relative to each mode's fresh C1, aggregate ratios at C2/C3/C4 are ordinary
1.228/1.623/2.064, K4 0.976/1.398/1.708, and K5 0.992/1.321/1.589.
C4 whole-output rates including prefill are 51.89/91.15/83.67 tok/s respectively.
The stock benchmark rotates the corpus by lane index: added lanes have different P89 prompts,
so these ratios describe this concurrent workload, not an isolated identical-request scaling test.
Same-C modes use the same per-lane inputs. K4 acceptance rates for C1–4 are
60.40/56.91/60.76/64.16%; K5 rates are 50.28/47.87/51.59/56.04%.
K4 has one summed fallback step at each C2–4 cell; K5 has none.

All three repetitions within every cell reproduce exact tokens and accounting. C1 K4/K5 also
match ordinary greedy output. **C2–4 DFlash does not match same-C ordinary output on all lanes**:
matching lanes are 0/2, 1/3, 1/4 for K4 and 0/2, 0/3, 0/4 for K5. These are throughput
observations, not exact-greedy production admission. No numerical gate was relaxed and no kernel
was changed in this measurement task.

Source dispatch establishes a plausible follow-up, not a measured causal attribution: several
Q4 pipeline, BF16 staging/control and RMSNorm optimizations select only T5/T6, whereas batched
verification uses T10/12, T15/18 and T20/24. K4 proposal batches requests; K5 proposal executes
per sequence. These route differences prevent extrapolating C1 optimization coverage or ideal
weight reuse to C2–4. Any follow-up must resolve the relevant exact-greedy discrepancy and measure
the actual larger-batch owner before optimization or promotion.

Evidence and exact launch commands: `profiles/bench/r9700-decode-bandwidth-concurrency-20260922/`
(`concurrency/run.py`, `concurrency/summary.json`, per-cell reports/receipts under `concurrency/results/`).
The first ordinary process succeeded; a checker incorrectly expected `spec=mtp` instead of the
reported draft-disabled `spec=none`. The corrected checker validated and reused that retained run;
no failed GPU measurement was overwritten or repeated. C1 traffic accounting is not extended to
the fallback-bearing C2–4 schedule as a measured physical traffic estimate.

### Measured C2 verification cost

Pre-correction C1/C2 K4 traces reproduce each mode's saved tokens and accounting. These are not
timings of the corrected build above. Profiling measures
kernel attribution, not unprofiled speed or physical memory utilization:

| Active batch | Physical graph launches | Kernel ms/graph | Target-layer ms/graph |
|---|---:|---:|---:|
| C1 | 38 | 47.122 | 38.585 |
| C2, two active lanes | 37 | 97.565 | 87.180 |
| C2, one-lane tail | 5 | 47.256 | 38.731 |

The C2 target layers take 2.26 times C1's cost, overwhelming the reuse elsewhere: target
norm/head is only 2.328 versus 2.221 ms/graph, and drafting is 5.485 versus 4.218 ms.
The tiny N48 BF16 control projections are a major regression: their T10 WMMA route totals
15.660 ms/full C2 graph, versus 0.761 ms for the C1 paired-wave32 controls. C2 also leaves the
T5 Q4/down and normalization specializations. Generic N5120 Q4 trace rows include both down
and another projection; they cannot all be attributed to down alone.

Across the complete run, graph cost per useful output increases from 13.989 to 15.024 ms.
Lower acceptance (56.91% versus 60.40%) and five one-lane tails also limit sharing. The actual
79 lane evaluations are 37*2+5, consistent with 78 speculative rounds plus one fallback;
lane-summed rounds are not graph counts. Prompt rotation confounds an isolated acceptance
comparison. These findings explain the measured scaling, but do not resolve the separately
observed numerical mismatch or authorize a route change.

Explicit databases, attribution, reconstruction scripts and independent topology checks are
retained under `profiles/rocprof/r9700-dflash-c1-c2-k4-20260922/` (`analysis.md`, `comparison.json`).

## Selective-protected DFlash decode (2026-09-22)

The currently selected evaluation artifact is
`/ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-q4-selective-protected-dflash2-q4/qwen3.8-27b-r9700-q4-selective-protected-n16k16-dflash2-q4-head-n16k16-eval.ninfer`
(18,887,772,672 bytes). It losslessly reorders only the W8 output-head codes and scales;
1,189 other payloads remain byte-identical to the original combined artifact. The original
`qwen3.8-27b-r9700-q4-selective-protected-n16k16-dflash2-q4-eval.ninfer` is retained with its
matched control binary. That companion preserved all 1,124 selective-protected base objects and
added 32 Q4G64 matrices and 34 BF16 objects, retaining both selector codebooks.
This is AMD Q4, not NVIDIA NVFP4, and does not establish cross-platform quality equivalence.

Matched ordinary/K4/K5 baselines use C1, context 1024, chunk 4096, G16, Device Graph, greedy
sampling and one warmup. The 89-token chat prompt generates 128 decode tokens; ordinary decode
measured 24.77555 tok/s. Raw corpus P128+G64 has much lower acceptance: 1.65789 output tokens
per speculative round versus chat K4/K5's 3.36842/3.45946. It is not evidence by itself that
companion quantization is defective.

The selected W8 wave-kernel change packs identical operand bytes into wide loads without changing
scales, arithmetic, reduction order, or dispatch. Exact embedded gfx1201 ISA changes 32 byte-load
sites to four B64 loads, 41 load waits to 11, and 83 VGPRs to 61; two signed IU8 WMMA sites and
zero LDS/private storage remain. Independent original-BF16-input FP64 checks, exact A8 codec,
tails, four-mod-eight code alignment and poisoned graph replay pass. All six complete output
tensors are byte-identical to the retained incumbent. An initial qualifier initialization race
was repaired before admission; numerical thresholds were unchanged.

Balanced whole-Engine A/B uses three fresh-process pairs per cell, both launch orders, one warmup
and one measured repetition each, under `auto`. Every pair improves, each cell exceeds the
predeclared 2% mean paired decode-throughput gain, and all 24 runs preserve exact tokens and
speculative accounting:

| Workload | Draft width | Control decode tok/s | Selected decode tok/s | Mean paired gain |
|---|---|---:|---:|---:|
| Raw text | K4 | 17.50045 | 18.52400 | 5.85% |
| Raw text | K5 | 17.24277 | 18.38723 | 6.64% |
| Chat | K4 | 35.02727 | 37.96226 | 8.41% |
| Chat | K5 | 36.35116 | 39.00057 | 7.29% |

Chat K5 whole-output throughput, including prefill, is 36.06263 tok/s. These are workload-specific
C1 results, not a 60 tok/s claim, C2–4 admission, or terminal artifact selection. Evidence and
reproduction runners are under `profiles/bench/r9700-w8-packed-loads-20260922/`; matched baseline
packages are `r9700-dflash-selective-baseline-20260922` and `r9700-dflash-selective-chat-20260922`.

The pre-change graph trace attributes 68.616 ms to target layers, 17.394 ms to the target final
norm/head, and 5.155 ms to draft layers per graph. Its largest remaining target-layer families
are Q4 gate/up, normalization, Q4 down and protected BF16 projections. These intercepted durations
are attribution only; they must not be reported as post-change timing. Reattribute before choosing
a subsequent kernel mechanism. Raw-text acceptance remains a separate limitation.

### Exact-order normalization admission and rejected refinements

The subsequent K5120 RMSNorm change selects the existing exact-order token8 kernel at exactly
T5/T6, preserving the serial FP32 reduction order and all other shape predicates. Independent
original-BF16 FP64 checks, exact incumbent outputs and poisoned graph replay pass. Balanced
three-pair chat A/B under the same C1 workload admits this change:

| Width | Control decode tok/s | Selected decode tok/s | Mean paired gain | Selected whole-output tok/s |
|---|---:|---:|---:|---:|
| K4/W5 | 38.02294 | 41.27905 | 8.57% | 38.01150 |
| K5/W6 | 38.54031 | 42.17567 | 9.48% | 38.72952 |

Every chat pair wins and preserves exact token IDs and speculative accounting. Raw-text K4/K5
one-pair checks also pass, but are regression checks, not additional balanced performance
admission. Evidence: `profiles/bench/r9700-dflash-rmsnorm-exact-20260922/summary.json`.

Two follow-up mechanisms are rejected. All three W8 scale-gather variants preserve exact outputs,
but generic half-wave gathering loses its direct screen, and sustained fixed-T full-wave shuffle
and explicit-readlane variants both lose at T5/T6. The packed-load baseline remains selected;
lower ISA load/wait counts do not establish speed. Timings, protocol distinctions and retained
reports are summarized in `profiles/bench/r9700-dflash-scale-gather-20260922/README.md`.
Mapping the same RMSNorm row arithmetic to one 32-thread CTA per row also preserves exact outputs
and stays within one BF16 step of the independent oracle, but takes approximately 0.070 ms versus
token8's 0.050 ms; it is not promoted. Evidence:
`profiles/bench/r9700-dflash-rmsnorm-exact-20260922/row-cta-screen.json`.

The Q4 gate/up scale-gather candidate subsequently retained exact parity but failed whole
admission: chat K4/K5 mean paired gains were only 1.29%/1.20%, each with a losing pair.
Its selector was removed. Evidence:
`profiles/bench/r9700-dflash-gate-up-scale-gather-20260922/planner-retry/summary.json`.

The packed normalization epilogue has narrower admission. Chat K4/W5 improves
41.14269 → 42.60744 decode tok/s, a 3.56% mean paired gain with all three pairs winning.
K5/W6's 1.56% mean gain includes a losing pair and is not admitted. Selection is therefore
restricted to K5120 **rows5 only**; rows6 retain the exact scalar token8 epilogue.
The combined candidate's top-level `NO_WIN` result is preserved rather than presented as an
all-width win. Raw one-pair results remain regression-only evidence. See
`profiles/bench/r9700-dflash-rmsnorm-packed-20260922/summary.json`.

Protected BF16 pair-load/K512 staging passes its original diagnostic at both qualified shapes,
T5/T6 and aligned/two-byte-offset operands, with exact incumbent outputs and full-K independent
FP64 checks. Aligned N7168/K5120 calls improve approximately 0.655/0.670 → 0.210/0.221 ms;
N5120/K6144 improves 0.498/0.507 → 0.157/0.175 ms. Evidence:
`profiles/bench/r9700-bf16-staging-20260922/screen/report.json`. Qualification through the
actual selected generic dispatch now passes. The combined rows5-packed-plus-BF16 whole A/B
passes all three balanced chat pairs at both widths, preserving exact tokens and speculative
accounting: K4 40.78243 → 45.45008 tok/s (+11.50% mean paired gain), K5
42.15744 → 45.65818 tok/s (+8.30%). Whole-request rates are 41.36606 and 41.54923 tok/s.
Raw-text one-pair regression checks improve to 22.30746/21.69262 tok/s; these are not repeated
admission measurements. Evidence: `profiles/bench/r9700-bf16-staging-20260922/summary.json`.

### Lossless tiled head, coalesced ordinary decode and Q4 projections

An additional exact-order normalization next-block prefetch passes full numerical/graph/guard
checks and emits real overlap at38VGPR with no spills. It improves T5/T6 direct medians
0.035264/0.050209→0.032304/0.046477ms, but129-call savings0.382/0.481ms miss the predeclared
2%round screen. It is not selected; its temporary code is removed. Retained evidence:
`profiles/bench/r9700-rmsnorm-block-ahead-20260922/README.md`.

The selected bundle adds `r9700-w8g32-n16-k16-v1` output-head storage, a coalesced exact-reduction
BF16 consumer for ordinary T1–3, and Q4 projection scale gathering at T5/T6 for
N5120/K6144, N12288/K5120 and N4096/K5120. It retains the admitted BF16 staging and rows5-only
packed normalization. The head permutation is inverse-byte-exact, with no requantization,
duplicate resident head or runtime packing. Same recipe identity, BF16 codebooks and fixed cache
are preserved. Real binding/old-layout rejection and public-consumer exact/FP64, graph and guard
checks pass. The initial tiled ordinary consumer regressed C3 by 3.89% and was superseded; its
failed evidence remains intact. The coalesced replacement preserves all 256 logical partial
chains and the reduction tree; direct T1/T2/T3 warm medians improve from
5.826/11.626/17.448 ms to 2.143/4.258/6.369 ms, respectively.

Receipt-bound whole-Engine A/B against the retained BF16-staging control passes all 24 runs,
with exact token streams, speculative accounting and matched configuration. Chat uses P89/G128,
context 1024, chunk 4096, greedy sampling, Device Graph and `auto` power. C1 ordinary and K4/K5
each have three balanced fresh-process pairs; C3 ordinary and raw P128/G64 K4/K5 each have one
regression pair. Gains below are for the complete bundle, not an isolated projection claim.

| Workload | Control decode tok/s | Selected decode tok/s | Mean paired decode gain | Selected whole-output tok/s |
|---|---:|---:|---:|---:|
| Chat ordinary C1 | 24.87909 | 27.46441 | 10.39% | 26.18621 |
| Chat ordinary C3 aggregate | 38.56344 | 45.08326 | 16.91% | 41.98362 |
| Chat K4 C1 | 45.89039 | 53.56033 | 16.72% | 48.09161 |
| Chat K5 C1 | 45.68180 | 52.43020 | 14.77% | 47.19157 |
| Raw K4 C1 (one pair) | 22.31616 | 26.08567 | 16.89% | 23.77438 |
| Raw K5 C1 (one pair) | 21.53918 | 24.87419 | 15.48% | 22.78313 |

Every chat speculative pair wins and each width exceeds the predeclared 2% mean gain; no ordinary
or raw cell regresses by 2% in decode or whole throughput. Evidence:
`profiles/bench/r9700-w8-tiled-head-20260922/whole-inference-combined/summary.json`.
Original startup/ordinary failures remain in `whole-inference` and `whole-inference-retry`.
The resumed optimization goal remains active: this is neither a 60 tok/s result nor terminal
recipe/quality or full C1–4 performance admission. Separate C2/K4 and C4/K5 startup/correctness
smoke passes exact lane tokens/accounting and graph allocation bounds in the same package's
`concurrency-smoke/summary.json`; those short checks make no throughput-admission claim.
Fresh matched K5 attribution in
`profiles/rocprof/r9700-dflash-tiled-projections-chat-k5-20260922/attribution.json` measures
51.679ms target layers, 2.238ms target head and 4.688ms draft layers per graph. Remaining major
owners are gate/up13.509ms, down9.740ms and scalar normalization6.141ms. These intercepted
durations identify optimization targets, not unprofiled speedups.

### Down next-group pipeline admission

The subsequent down-only next-group pipeline is selected at N5120/K17408, T5/T6. Public-Op
qualification passes exact outputs, the original represented-input FP64 oracle, graphs and
guards. The matched whole A/B uses the admitted tiled-head/projection binary as control and
the identical tiled artifact on both arms; only the down implementation changes. All 16 runs
preserve exact streams, speculative accounting, artifact and configuration. Three balanced chat
pairs per width exceed the 2% mean-gain gate with every pair winning:

| Width | Control decode tok/s | Selected decode tok/s | Mean paired decode gain | Selected whole-output tok/s |
|---|---:|---:|---:|---:|
| K4/W5 | 53.70092 | 57.31582 | 6.73% | 51.08892 |
| K5/W6 | 52.48043 | 55.87248 | 6.46% | 49.88149 |

Raw K4/K5 one-pair regression checks reach 27.86485/26.48399 decode tok/s and pass both
decode/whole regression gates. Ordinary throughput is not remeasured: the exact T5/T6 predicate
does not change its route. Earlier C2/C4 smoke predates this pipeline and is not new pipeline
admission. Evidence: `profiles/bench/r9700-dflash-down-pipeline-20260922/whole-inference/summary.json`.
The artifact/recipe is unchanged; these results do not establish 60 tok/s or terminal quality.

The pre-pipeline trace also bounds the draft-side frontier. Context append totals 0.908 ms;
omitting unused Q rows in its five QKV projections could save at most about 0.262 ms.
Proposal head plus selector totals 1.044 ms, and the already split SWA draft attention totals
0.139 ms. Device round-service work including postgraph folding totals about 0.672 ms.
None identifies an independent roughly 1.3 ms/round (2%) mechanism; these are attribution bounds,
not measured savings or proof that host overhead is absent. Do not reopen the historical serial
draft-attention split-KV proposal against the current split SWA route. Normalization block-ahead
qualification remains separate and unselected; the optimization frontier is still active.

### Verify projection pipeline-family admission

The next bundle extends one-group-ahead loading to Q4 gate/up N34816/K5120 and projections
N5120/K6144 and N4096/K5120 at exactly T5/T6. N12288/K5120 retains its scale-gather route:
its pipeline screen did not establish a useful gain. All selected routes pass the public
represented-input FP64 oracle, exact incumbent outputs, poisoned graphs and guards. Linked
gfx1201 ISA confirms successor-load overlap, ordered reductions and bounded final drains,
with 62/63 VGPR, 22 SGPR and no LDS/private storage/spills. Public operator screens estimate
3.710/3.391 ms per round for gate/up and 1.569/1.666 ms for the narrowed projection family;
these estimates are not separate whole-model claims.

The same-artifact whole comparison against the admitted down-pipeline binary passes all 16
runs with exact tokens and speculative accounting. Every balanced chat pair wins:

| Width | Control decode tok/s | Selected decode tok/s | Mean paired gain | Selected whole-output tok/s |
|---|---:|---:|---:|---:|
| K4/W5 | 57.42168 | 63.02158 | 9.75% | 55.59136 |
| K5/W6 | 55.93994 | 61.24468 | 9.48% | 54.10987 |

Raw-prompt single-pair regression checks reach 30.58844/29.02133 decode tok/s, with no decode
or whole-output regression. These are C1 measurements; ordinary T1–4 and concurrent admission
are not repeated for this T5/T6-only change. This is a bundle gain, not an isolated gate/up
or projection whole-model gain. No quantization or artifact change is involved.
Evidence: `profiles/bench/r9700-verify-pipeline-family-20260922/whole-inference/summary.json`.
The subsequent paired BF16 control optimization is not included in these numbers.

That paired BF16 split-K16 candidate is rejected: its complete-Op FP64 oracle, native BF16 ISA,
guards, own-graph checks and direct timing screen pass, but the first real-model chat K4 run
changes output index 52 from token 413 to 3470. Overall 76/129 output positions differ, with
40 rather than 38 rounds and 88 rather than 90 accepted draft tokens. The campaign stops at this
exact-token failure; no candidate speed is admitted and the unchanged T1-only control route
is retained. Evidence:
`profiles/bench/r9700-bf16-projected-control-splitk-20260922/whole-inference/rejection.json`.

### Exact-order paired GDN controls admission

The selected follow-up uses one wave32 per (head,token), computing the two BF16 projections
with the incumbent's exact 160-FMA lane chains, shuffle reduction and BF16 rounding seams,
followed by unchanged FP32 gating. Four-step blocked loads remove staged CTA barriers without
reassociating either dot. Only T5/T6 changes; T1 retains its kernel, other widths retain their
composition. The owning public Op requires no scratch, and the existing GDN device module is
reused. Native code uses 25 VGPR/19 SGPR, with no LDS/private storage/spills. Public FP64,
exact-output, graph, alignment and guard checks pass. The first scalar-load version missed
three cold-cache material gates; the four-step version clears all eight warm/cold cells.

The strict same-artifact comparison against the admitted Q4 pipeline bundle passes all 16 runs
with exact tokens and speculative accounting:

| Width | Matched control decode tok/s | Selected decode tok/s | Mean paired gain | Selected whole-output tok/s |
|---|---:|---:|---:|---:|
| K4/W5 | 62.43569 | 64.81318 | 3.83% | 57.02881 |
| K5/W6 | 60.68160 | 62.92418 | 3.71% | 55.64922 |

Every chat pair wins: K4 gains range 2.88–5.72%, K5 2.67–5.54%; the final control pair is
slower at each width, so the mean is not a claim of a constant gain. Raw-prompt one-pair
regression checks reach 31.45723/29.82590 decode tok/s and pass both decode/whole gates.
Ordinary throughput is not retimed because T1 is unchanged; its latest measured C1 result
remains 27.46441 tok/s. These DFlash results are C1, P89/G128 chat, context1024, chunk4096,
greedy Device Graph, fixed selective-protected/Q4/BF16 artifact and typed production cache.
Evidence: `profiles/bench/r9700-bf16-controls-wave32-20260922/whole-inference/summary.json`.

The bounded optimization pass has no remaining evidenced, untested material mechanism above
its 2% round gate. This is not a claim of globally optimal kernels, saturated bandwidth, or
optimality across other prompts, contexts, concurrency or recipes. Remaining costs include
gate/up, exact-order normalization and down projections; their concrete challengers and
rejections are retained. Two-group gate/up prefetch and its fixed-slot repair both fail the
native overlap gate, so neither receives a GPU timing campaign. Further work needs a distinct
mechanism that resolves an observed limit, not another unchanged sweep.

## Typed Text/MTP cache and attention

### Bounded-panel dense prefill (2026-09-21)

The all-Q4 N16/K16, G16, C1, spec-none P8192/chunk1024 benchmark improved from
`184.8640144` to `1184.242683` prefill tok/s (`6.406x`). Three measured candidate durations
were `6.870248486`, `6.901032133`, and `6.982186078 s`; median duration was `0.155599830`
of the retained matched baseline. Each unprofiled cell used one warmup and three repetitions
on the R9700 under `auto`. A fresh P2048/chunk4096 regression pair measured
`1664.303599` versus `1659.247929 tok/s`, with candidate/baseline median duration `1.003513977`.
This pair passes the 2% regression bound but does not reproduce the older `1904.339303 tok/s`
measurement. Do not mix the historical value with this matched comparison.

The old route used tiled attention only for initial prefixes, falling back to serial causal
attention on appended chunks. A selected-region 8K trace attributed `39087.580 ms` of
`43838.812 ms` Text-prefill wall time to that fallback. The new route reuses the existing
BF16-WMMA QK and FP32 softmax/PV arithmetic in bounded query panels; score/max storage stays
at most 384.1875 MiB. G16/G32 independent FP64 checks cover initial/appended contexts,
partial panels, all KV heads, device-active rows, fragmented pages, graph replay, and the
262144-token boundary. The typed leaf and all five ISA/resource checks pass; emitted resource
counts and zero-spill status are unchanged.

Admission and complete input bindings are retained in
`profiles/bench/r9700-chunked-attention-candidate-leaf-retry1-20260921/whole/result.json`.
The original package retains passed raw qualification and the subsequently repaired public-leaf
fixture failure. Deferred driver VRAM teardown interrupted the first whole sequence; its completed
8K result was retained, and only the missing P2048 pair was measured in `whole-completion`.
These results admit the bounded-panel mechanism, not a terminal weight recipe, shared chunk,
model-quality gate, or DFlash performance claim.

The subsequent C1 ordinary-prefill campaign completed 48 8K screen cells (chunks 1024/2048/4096/8192)
and 24 32K finalist cells (2048/4096), crossing all-Q4, mixed Q4/W8, and four-role FP8/Q4 with
G16/G32 and dense/XAttention B128/S16/tau900. Each cell used one warmup and three unprofiled
repetitions under `auto`. Shared chunk **2048** wins the complete maximin-normalized objective
(`0.9959696531` versus `0.8884210782` for 4096), with maximum workspace 608,387,072 versus
813,924,352 bytes. It wins eleven of twelve 8K profiles and all twelve 32K finalist pairs.
Authority: `profiles/bench/prefill-chunk-selection-panel-attention-20260921.json`.

Representative matched C1/G16/chunk2048 prefill rates (tok/s):

| Recipe | 8K dense | 8K XAttention | 32K dense | 32K XAttention |
| --- | ---: | ---: | ---: | ---: |
| All-Q4 | 1189.0 | 1138.7 | 402.2 | 572.0 |
| Four-role FP8/Q4 | 1302.2 | 1260.5 | 414.6 | 608.8 |

XAttention is slower here at 8K but faster at 32K; these timing results do not admit its numerical
quality or select a production recipe/group. No decode rounds or draft head execute in these
measurements. Fresh BF16 references and the six matched quality campaigns have now completed
at chunk2048; their results follow.

### Selected-chunk numerical quality (2026-09-21)

The independent BF16 source run and repeat match exactly in NLL/token sidecars at both lengths.
BF16 PPL is `6.463887635` at8K and `5.632510588` at32K, scoring4095 and16383 positions.
The same Python3.11/ROCm reference environment first passed the sampled FP64 full-span GDN
oracle at4095/4096 rows. The selected artifacts and frozen dense/sparse G16/G32 scorers then
completed all24 candidate cells against that reference.

| Recipe/profile | Assigned tier | G16 | G32 |
| --- | --- | --- | --- |
| All-Q4 dense | capacity-speed | pass | pass |
| All-Q4 XAttention | capacity-speed | pass | pass |
| Mixed Q4/W8 dense | accuracy | pass | pass |
| Mixed Q4/W8 XAttention | accuracy | fail at32K | fail at32K |
| Four-role FP8/Q4 dense | capacity-speed | pass | pass |
| Four-role FP8/Q4 XAttention | capacity-speed | pass | pass |

Mixed XAttention passes the mean-NLL limit but introduces19/18 new severe positions at32K,
above its accuracy budget17. These are genuine measured exclusions, not missing evidence.
Its passing dense controls remain eligible. No threshold or recipe tier is changed, and
BF16 greedy-token differences remain diagnostic rather than an exact-quantized-model gate.
Ten profiles proceed to capacity/whole eligibility; this is not final production selection.
Evidence: `profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921`, including the
retained failed mixed-XAttention report and all aligned sidecars.

### Retained cache and attention evidence

The growing cache stores FP8 E4M3FN K, signed INT4 V, and FP16 V scales. A 32-point sweep covering
G16/G32, every K/V/scale plane order, T=1..8, and 1K/4K/8K/32K contexts passed the independent
layout and attention oracles after the coherent ROCm update.

The provisional latency leader is G16 with token-fastest K, feature-fastest V, and feature-fastest
V scales: mean normalized latency `1.004041`, mean rank `2.188`, and 14 wins. This is not yet the
permanent ABI: the retained real 8K comparison measured both G16 and G32 within their quality tier
and favored G16 on NLL error against one historical BF16 realization. Current chunk2048 numerical
eligibility is recorded above; matched capacity, phase, graph/eager, and whole-inference evidence
still precede terminal selection. The older C=1..4 capacity matrices below remain historical.
The current continuation retains all twelve profile capacity outcomes at C=1..4 and times only
quality- and capacity-eligible profiles.

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

The earlier production P128..4096 initial-prefix leaf used the physically selected three-stage
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
C=1..4 capacity evidence. Missing capacity excludes only the exact recipe/K/concurrency cell when
its failed command and logs are retained. Capacity decisions declare each candidate's eligible
concurrency subset; it must include C1 to advance. Matched 8K/32K DFlash decode and retained-token
fresh-prompt whole inference use same-build exact spec-none ordinary controls at every declared C.

The replacement decision authority must bind the base decision through each conversion report to
the exact terminal winner, BF16 DFlash source, DFlash matrix recipe, companion artifact,
executable, cache group, and K/W. It must reopen every retained report and recompute recipe
eligibility, exact ordinary-output parity, exact repeated proposal/target determinism, and
generated-quality evidence. `prepare_selected_dflash.py` now prepares three recipe-separated,
create-only CPU conversions using an explicit Python 3.11 environment and a fresh matching
benchmark/planner build. Its schema-v2 plan validates the real terminal base and chunk before
conversion. The schema-v5 `assemble_dflash_selection.py` recomputes the resulting evidence;
historical fixed-Q4/K1..11 packages remain superseded. Whole parity covers the complete requested generation
including its first output token; isolated decode parity covers all 257 post-seed outputs, with seed
equality inherited from deterministic whole-route parity. Before frontier ranking, each admitted cell must provide at least
`1.02x` raw-mean speedup and a strictly positive two-standard-deviation conservative speedup over
its matching ordinary route in both 8K/32K whole and decode measurements. C1 material screens run
first; no full followup is licensed without a material K4/W5 C1 route. Missing measurements for a
declared eligible C are failures, never inferred capacity exclusions. Capacity, acceptance,
quality, and speed remain distinct gates. The primary evaluation winner is the C1 frontier member
maximizing worst normalized 8K/32K whole throughput, then capacity, then matched acceptance;
canonical recipe/K/W order breaks only a complete tie. Separate per-concurrency frontiers retain
all exclusions, including a primary winner that is unsupported or slower at C2..4. This is not
dynamic recipe switching, an all-C production claim, or final production promotion.

The assembler's separate `admit` action requires an explicit recipe and K/W pair. It reopens
schema-v5 evaluation and reconstructs all conversion, shortlist, capacity, C1-screen and followup
inputs, then recomputes the raw gates. One chosen companion must qualify for every supported
startup concurrency C1..4: complete capacity, exact ordinary-output parity, proposal determinism,
generated-quality evidence, a material C1 win, and the existing matched 8K/32K whole/decode speed
gates at all four C values. A candidate need not win every per-C frontier, but cells from different
recipes or K/W pairs cannot be combined and the supported concurrency contract cannot be reduced.
The distinct schema-v1 single-resident admission binds that explicit resident to the terminal base
and all qualified evidence; schema-v5 stays evaluation-only. Its C1 screen is reused unchanged;
only remaining capacity-eligible C2..4 run followups. Both original manifests are retained,
with screen-owned proposal/generated-quality proof and per-stage raw ordinary-token parity.
Final cutover revalidates this admission instead of accepting a bare evaluation. Admission
does not rename/materialize a production artifact or satisfy the other final-cutover gates.

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

The off-by-default K5120 rows5/6 RMSNorm selector passes its matched public numerical and graph
gate in both selector states, but the focused P129 token experiment under
`profiles/bench/r9700-dflash-rmsnorm-rows56-token-parity-20260906/results` rejects parity
restoration. Its result closure SHA-256 is
`03e10a9f92b64ac8f49de81796750fb28b3eb138399508fb0fec14444eb34337` and summary SHA-256 is
`7fcd118d8a010c6162671a47700e27b743a2baf4f67b0af1d1b1ad2f6a2b6144`. Fresh ordinary and
selector-off DFlash exactly reproduce their prior 28-token authorities and differ only at index 27
(95946 versus 98003). Selector-on first differs from both at index 21 (128415 versus 96723), so its
index-27 token 96843 is on a different generated history and cannot establish repair of the prior
index-27 mismatch. The selector remains off; these functional runs contain no admissible timing.

The selector-on internal follow-up under
`profiles/bench/r9700-qwen3-layer1-gdn-detail-traces-e915a5e4-20260906/results` has result-closure
SHA-256 `2a3a95672f323ac9fbcbf16a3914c4496baa8eb0e0cf8cb92da3ccdd58126907` and summary SHA-256
`93518da3aef2845594512d1800d8c08ad39352cdd15189257966f62f7728749b`. Target ordinary W1 and
DFlash W5 are byte-exact for every captured layer-one GDN field from normalized h through final
mixer residual x; their first visible residual difference moves to layer three, the first
full-attention mixer, at 2,796/5,120 BF16 values. Text fresh T129 and append T1 are exact through
h, controls, and q/k/v/z, then first differ in recurrence output o at 106/6,144 BF16 values with
maximum absolute difference 0.000003814697265625. Gated normalization and the mixer residual inherit
that difference. This excludes layer-one RMSNorm, control, and input projection/convolution for the
selected Text column, but the trace does not distinguish an FP32 recurrence-state input mismatch
from different wide-prefill/append recurrence arithmetic. It is functional evidence only.

The next exact target trace is retained under
`profiles/bench/r9700-qwen3-layer3-attention-trace-617672eb-20260906/results`. Its immutable raw
capture closure is `815307ab0a04a6e834eb19cc670c75e6d1f2c4c60c693f8861d0e7fb0b8eb79f`;
an offline tuple-order repair changed no captured bytes, and the recovered summary and final result
closure are `37fd48842066ce1150cccdad6ed00ad49e383759620f7ff96b0e1ee2c6b3013a` and
`68ae75bc83ef4044148e594ca36cbc3eaa8a605c95be2aca3424eaad959ab3c2`. With small-T gate/up off
and the qualified MLP-down, RMSNorm rows5/6, and GDN wave-normalization selectors on, target ordinary
W1 and DFlash K4/W5 are byte-exact through the layer3 input, input RMSNorm, q/gate/k/v projections,
q/k normalization, RoPE, causal visibility, and canonical cache positions 0--129. Their first
difference is full-attention FP32 output element 0: bits 1052023983 versus 1052080823, with maximum
absolute difference 0.02367246150970459. This localizes the first owner to full-attention route
arithmetic; it is not an independent numerical oracle, timing evidence, or a routing decision.

The exact selector-one Text follow-up under
`profiles/bench/r9700-fp8-e2-t129-text-parity-gate-20260906/results` retains four C1 eager arms for
the same P129 history. All four emit `[96558,96917]`; fresh and append are byte-exact for both the
layer-zero and layer-one FP32 recurrent states and for every captured layer-one GDN field. The first
visible residual difference moves to layer-three post-mixer: 3,467/5,120 BF16 elements differ, first
at hidden 0 (bits 48413 versus 48414), with the preceding boundary exact. This combined-build result
is consistent with the e2 T129 algorithm removing the earlier layer-zero/layer-one difference, but
does not isolate e2 causality from the other bound selectors. It is functional and timing-ineligible;
whole Text parity remains blocked at the same first full-attention layer implicated by the target
trace above.

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

Base-decode bandwidth work was subsequently reopened and admitted three exact projection fusions.
At C1, the BF16 GDN control fusion's whole-model candidate/control decode-time ratios were
`0.9876921`, `0.9870972`, and `0.9868352`; the all-Q4 attention input-projection fusion's ratios
were `0.9887575`, `0.9873370`, and `0.9869812`. At C2, C3, and C4, the shared-quantize paired-Q4
route passed three whole-model pairs per concurrency with median ratios `0.9246617`, `0.9272089`,
and `0.9341644`, and upper-two-standard-error bounds `0.9250822`, `0.9283767`, and `0.9342851`.
Every process retained exact public tokens. The C2--C4 result closure is
`profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919/results`, whose
`result.sha256` digest is `26c4ae49e7a24ce97d1f56c891e841942507678aaab8a0df0ada315996e93cda`.
These gates authorize selector-free production routing only for singleton BF16 GDN controls,
singleton all-Q4 attention projection pairs, and all-Q4 projection pairs at exactly two through
four tokens. Mixed weights and other token widths retain their prior routes. These unprofiled
whole timings establish useful decode-speed gains, not physical HBM saturation or stall freedom,
and the two separately measured C1 gains are not added to make a combined claim.
The fresh selector-free composition run then reproduced exact retained tokens at every supported
concurrency. Its C1 through C4 decode times were `9.0186748`, `14.7080029`, `16.4007540`, and
`18.8960216 s`, corresponding to aggregate rates `28.3855`, `34.8110`, `46.8271`, and
`54.1913 tok/s`. The retained-candidate timing ratios were `0.9865955`, `0.9997177`, `1.0002975`,
and `1.0000448`. Evidence is
`profiles/bench/r9700-three-route-production-confirmation-20260920/results`; the SHA-256 of its
`result.sha256` closure is `9be2f67ff0f74f92e509bbfa6bb0909e7feb5e38993aed184d71ce1cd7033148`.

The subsequent projected-residual fusion passed the reviewed C1/P8192+G256 ordinary Device Graph
A/B under `auto`, using the same Q4G64 artifact and fixed cache. In balanced execution order,
control/candidate/candidate/control/control/candidate decode rates were `28.3689630`,
`29.0009861`, `29.0069622`, `28.3912167`, `28.3819193`, and `28.9849878 tok/s`. Every process
preserved the retained 257 public token IDs. The paired candidate/control decode-time ratios were
`0.9782068408`, `0.9787724938`, and `0.9791937627`, with mean `0.9787243658`, upper two-standard-error
bound `0.9792961944`, and median `0.9787724938`. Evidence is
`profiles/bench/r9700-projected-residual-t1-whole-ab-20260920/attempt-1`; its `result.sha256`
digest is `e7337ec8e9bc19d8519f52f06ffcb50359270fa47a4416ff859a4aee8afa9b79`.
The independent result audit admitted promotion only for all-Q4/A8 base Text T1 at N5120/K6144 or
K17408; mixed inventory, MTP, prefill, T>1, and A4 retain the incumbent composition. Selector-free
implementation review and fresh linked public-Op qualification passed. The subsequent selector-free
C1 confirmation retained all 257 public tokens and measured `28.9947`, `29.0153`, and
`28.9945 tok/s`, a `29.0015 tok/s` mean. Its median decode time was `0.9788658` of the retained
control and every run stayed within `1.01` of the admitted candidate median. Evidence is
`profiles/bench/r9700-projected-residual-t1-production-confirmation-retry1-20260920/attempt-1`;
its `result.sha256` digest is
`f629374d5dba788ba93837ee06b74eab6be1650f95d05772694af848d37c9d5a`. Independent audit reported
`SHIP`, closing this exact promotion. The result does not establish physical memory-bandwidth
saturation or stall freedom.

The next normalized-linear direct screen fused the cooperative K5120 RMSNorm and exact A8G64
preparation while preserving the explicit BF16 normalization seam and unchanged native-IU4
N34816/K5120 consumer. Its complete public boundary improved `0.2406990→0.2347595 ms`; every one of
three address-distinct allocations won, the paired-ratio mean plus two standard errors was
`0.9790384`, and the 64-MLP-boundary estimate saved `0.3801284 ms/token`. The independent FP64
normalization-to-BF16-to-exact-A8-to-signed-Q4 oracle passed with maximum relative L2
`0.00187542`, zero BF16-step error, exact public-control parity, and captured graph recovery.
Evidence is `profiles/bench/r9700-normalized-linear-t1-qualification-20260920/attempt-1`; the
`result.sha256` digest is `01e1e0571d92541311195202df919d3750759f3bcedf7033944f604bcf2baf2c`.
Independent audit admitted matched whole C1 A/B preparation; this direct result alone does not
establish a whole-decode gain or bandwidth saturation.

The matched ordinary Device Graph C1/P8192+G256 whole A/B then retained all 257 public tokens in
six balanced runs. Candidate/control decode-time ratios were `0.9895615`, `0.9896042`, and
`0.9897497`; median `0.9896042` and mean plus two standard errors `0.9897524` passed the admission
gate. Mean decode improved `29.0050→29.3087 tok/s`, about `1.05%`. Evidence is
`profiles/bench/r9700-normalized-linear-t1-whole-ab-20260920/attempt-1`; its `result.sha256` digest
is `2484b71a86314ec8bf4b116fe0b5caecb707d77747666a41dec5f2ee95ee32ee`. Independent audit admitted
promotion review only for all-Q4/A8 ordinary base Text C1 MLP gate/up boundaries. Selector-free
implementation review then passed, and fresh linked public-Op qualification at
`profiles/bench/r9700-normalized-linear-t1-promoted-qualification-20260920/attempt-1` passed with
`0.3647995 ms/token` direct saving. The selector-free production confirmation retained all 257
tokens and measured `29.3174`, `29.3157`, and `29.3116 tok/s` (mean `29.3149 tok/s`). Its median
decode-time ratio to the retained control was `0.9893966`; every run was faster than the admitted
candidate median. Evidence is
`profiles/bench/r9700-normalized-linear-t1-production-confirmation-20260920/attempt-1`; its
`result.sha256` digest is `f28d5a0e89bc8cfaa438d7c7ef86405c5eb486c650838c57bf086d9b40c2aebb`.
Independent confirmation result audit reported `SHIP`, closing this scoped promotion. These
timings do not establish physical bandwidth saturation or stall freedom.

The explicit next-group prefetch challenger for T1 N34816/K5120 gate/up was numerically and
statically qualified, then rejected on complete normalized-linear boundary timing. On the R9700
in `auto`, three disjoint weight allocations and 24 balanced graph timing pairs measured
`0.2361795→0.2369395 ms` (control→candidate), or `-0.0486398 ms/token` weighted saving over 64
calls. Every allocation was slower; paired ratio mean plus two standard errors was `1.0094922`.
This fails the `0.2 ms/token` gate and does not justify a whole C1 A/B. The independent FP64
oracle passed with zero BF16-step error and maximum relative L2 `0.001801662`; codec and public
control parity were exact. Embedded native-IU4 ISA confirmed four successor B64 loads ahead of
current compute, at 26 VGPR, 36 SGPR, occupancy 16 and zero LDS/scratch/spills. Correct instruction
overlap did not produce a useful gain at the complete boundary. Independent review accepted the
rejection; production routing was never changed, and temporary qualification ownership is removed.
Sealed evidence is `profiles/bench/r9700-gate-up-prefetch-qualification-20260920/attempt-1`;
`result.sha256` digest is `5ea1c61d7d02fdeb971c151ded538872cf46121c31d299686bf0b42390b68e15`,
closure SHA-256 is `8e08b7a134601a6dddb5c16c5a74b7b19ed6f635bf435f50ee50b742c75ac4db`.
Its scripts are retained provenance, not rerunnable commands. A GDN projection/control
heterogeneous grid has independently reviewed CPU/static feasibility (`SHIP`), with the complete
contract and next qualification gate in `plans/r9700-autonomous-todos.md`. The `112x256` grid
assigns CTAs `0..63` to paired Q4 and `64..111` to control heads `0..47`; CTA-uniform branches
confine the nine control barriers. Emitted gfx1201 code preserves native IU4 and the incumbent
control reduction/BF16 seams, using 34 VGPR, 52 SGPR, 2048 bytes LDS, compiler occupancy 16, and
zero scratch/spills. Four vector B128 activation loads replace two scalar B256 loads, leaving
an unresolved issue-cost risk. These static results establish neither numerical correctness nor
performance. Retained Q4-pair/control timings are `0.102400/0.020920 ms`
per layer; overlapping controls with the pair has an ideal `0.896 ms/token` saving after an
estimated `0.002259 ms/layer` extra streaming cost across 48 layers. This is an unmeasured bound;
the admission margin needs `4.167 us/layer` complete-boundary saving. A complete
control+quantize+pair numerical and timing package must receive independent review before GPU
work; no new physical command is admitted. Finish this bounded decision, then switch to
recipe-independent DFlash optimization before another base-decode mechanism; normalization
fusion is deferred. Final DFlash recipe/quality binding still follows numerical-accuracy/artifact
selection.

The existing MTP shortlist head remains Q4G64 with A8G64 activations. MTP stays in exact-output,
state, cache, row-view, and whole-route regression coverage, but a new shortlist-head trace,
alternate head precision, acceptance campaign, or MTP performance optimization is not a final
admission requirement. DFlash/DFlash2 is the preferred speculative path and the only speculative
backend with remaining support and performance work.
After final base/artifact dependencies, the C1 DFlash optimization target is at least 60
decode-output tok/s in matched whole inference. Admission first requires a material win over the
current production base with exact public greedy-token parity; the target does not relax either
gate.
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
