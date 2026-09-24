# Compact follow-up: accuracy, attribution and chunk4096

User scope: resolve the inherited accuracy question, profile the final selected
build, pursue material DFlash/prefill opportunities and compare chunks2048/4096
(1024 only as a useful control). Installed cap26 Q4-head model and Q4 DFlash
companion unchanged. One R9700 gfx1201/device0/PCI0000:13:00.0, auto power,
ROCm10; heavyweight jobs serial, memory20Ghigh/24Gmax/no swap, builds<=14.
Prior selected source checkpoint `a6f9a738`; explicit frozen benchmark:
`profiles/rocprof/r9700-compact-mixed-speed-20260923/final-bin/bench`.

## Accuracy diagnosis

`accuracy_diagnostic.hip` separates original BF16-input FP64 W*x, exact
represented-A8 FP64 W*xhat, and GPU output. Same stored signedQ4/FP16 weights,
full4096 output rows, K5120, T6, both original and scaled(-0.75) fixtures.
No criterion change. Exact GPU generic/selected output and host codec agree;
represented-A8 FP64 arithmetic criterion passes. Retained command/receipt:
`accuracy-diagnostic-run/`; per-token results `accuracy-diagnostic.json`.

Failing scaled fixture/token5:
- Activation relative L2 error0.2683108%.
- Ideal quantized FP64 output versus BF16-input oracle2.0162495%.
- GPU output versus BF16-input oracle2.0340653% (original2% gate still FAIL).
- GPU versus ideal quantized FP64 output0.1775203%.
- Norm of abs(W)*abs(x) divided by norm(W*x):306.1956.

This demonstrates quantization-error amplification through cancellation, not a
pipeline/WMMA indexing or accumulation defect. Keeping identical codes/scales
cannot pass this fixture by fixing arithmetic alone. It is not evidence of a
model PPL regression or a universal quality statement.

Host-only `scale_diagnostic.hip` attempts one least-squares scale update perG64,
rounds it to FP16, requantizes codes and accepts only lower group input MSE.
`scale-diagnostic-run/` retains results: failing output error worsens to2.18842%.
Reject this as a resolution; input MSE does not control cancellation-sensitive
output error. No codec or production source changed. User decision requested:
preserve A8 with an explicitly justified conditioning-aware qualification
contract, or retain strict2% and investigate more accurate activation arithmetic.
Original failure remains recorded; no threshold waiver or passing claim.

## K4 MLP coverage: Layer0

K4 flattens five verification rows per active request: C2/3/4 gives T10/15/20.
These widths miss the already-qualified tiled successor pipeline. Same public
BF16->A8G64->Q4N16K16/G64/FP16-scales->BF16 formula, existing workspace and
16-row mapping. Cold complete-Op medians (three disjoint weights,80MiB scrub,
24alternating samples, original-public/represented FP64 and exact output first):
gate/up T10/15/20=0.371300/0.374419/0.501520ms;
down=0.352680/0.356000/0.324340ms (`k4-bound.json`).
Streams94.70/47.35MB per matrix. The retained T12/18/24 same-body mechanism
reduced gate/up7–18%, down41–50%; even10% on both projects4.5ms per62-layer
round, material against C2 K4 baseline98.32 aggregate tok/s. Instantiate only
T10/15/20, no arithmetic/layout/codec change; test masked first/second tile,
unchanged full-output2%/10% public oracle, graph/poison/codec/guards, exactgeneric,
native ISA/resources and whole same-C K4 tokens/timing before promotion.
The non-MLP N4096 accuracy decision is independent and remains pending.

Qualification and whole-inference admission: six new MLP cells pass the original
full-output FP64 criteria (maximum public RMS1.96781%), exact codec/generic/graph
output, poison and guards. Native IU4,70VGPR/22SGPR, zero LDS/private scratch;
all23 prior kernel instruction streams/resources unchanged. Cold complete-Op
gate/up saves17.62/14.93/6.61%, down49.58/49.51/44.29% at T10/15/20.
`k4-qualification.json`, `k4-isa.json`, `k4-screen.json` retain direct evidence.

Matched codeP4096/G128,chunk2048,G16,dense,auto,warm1/reps3 whole decode:
| C | K4 before aggregate tok/s | K4 selected aggregate tok/s | Selected per request |
|---|---:|---:|---:|
|2|98.3202|122.1377|61.0689|
|3|132.1731|160.8733|53.6244|
|4|142.6092|161.6563|40.4141|

All nine selected repetitions and all nine controls exactly match prior same-C
ordinary greedy tokens. Per-request is aggregate/C, not latency. Whole receipts
are `k4-baseline-c{2,3,4}-k4/` and `k4-candidate-c{2,3,4}-k4/`.
No A8 codec, weights, workspace or arithmetic change; only these six MLP extents
are newly selected. The inherited non-MLP failure is not waived by this admission.

## Chunk selection

Same frozen prior-final binary, base, dense/G16, C1/G128, warm1/reps3:
| Prompt | Chunk1024 prefill tok/s | Chunk2048 | Chunk4096 |
|---|---:|---:|---:|
|code4096|1473.3933|1523.6843|1459.0137|
|WikiText8192|not needed|1241.8632|1197.6315|

2048/4096 generated tokens are exact for both prompts. Keep2048:4096 is4.24%
slower on code4K and3.56% slower on WikiText8K. The explicit8K corpus is
`tools/ppl/corpus.ids` (32768 IDs), not a cyclic repetition of the code4K fixture.
This is a measured preference, not a theoretical optimum. Reports/commands are in
`baseline-c1-k0/`, `chunk4096-baseline-c1-k0/`, `code4k-chunk1024-c1-k0/`,
`wiki8k-chunk2048-c1-k0/` and `wiki8k-chunk4096-c1-k0/`.

## Remaining prefill bound

Fresh C1/C4 traces place dense PV at13.6% of prefill kernel service. Existing
single-piece BF16, scalar head-remap and FP32-library exclusions stand. One new
prototype tests high/low BF16 decomposition of both probabilities and represented
INT4*FP16 values: four native WMMA products, FP32 accumulation and original FP32
denominator. Public input remains BF16 Q plus exact stored FP8K/INT4V/FP16scale;
complete FP64 attention oracle retains its existing2e-3 absolute/relative criterion.
One query head/CTA,16query rows,128features,8waves reduces approximately12288
wave scalar-FMA instructions to384WMMA instructions per original16-key work tile,
at the cost of2x probability and6x value traffic. This attacks arithmetic issue
and accumulator pressure, not physical HBM bandwidth. Full value decomposition
is exact for valid finite FP16 scales; probability decomposition is approximate.
Complete-Op cold baselines:10.0872ms at2048queries/context2048 and42.8940ms at
2048queries/context4096 (`prefill-attention-baseline.json`). One prototype only:
reject on spills, numerical failure, <20%PV saving or <10%complete-Op saving.
No promotion without whole-prefill benefit and model-quality evidence.

Outcome: REJECT. Two complete FP64 attention cells pass, and independent exhaustive
host decomposition checks all1,015,808 finite FP16-scale/signed-INT4 pairs exactly.
Candidate uses53VGPR/40SGPR,10408B LDS, zero private scratch, wave32. Complete-Op
medians regress10.0872→11.0122ms and42.8940→43.9773ms; the arithmetic-issue
reduction does not overcome this mapping's added data/decomposition/synchronization
cost. No further sweep or full-model quality campaign is justified. Active source
and routing removed; rejected header and direct reports retained here only.

## Remaining K5 output projection bound

Fresh C4 K5 graph attributes177.34ms/2748.9ms (6.45%) to N5120 output-shaped
verify dispatches. The existing T2–6 N5120/K6144 pipeline is qualified and saves
22–29% at small widths; T12/18/24 still use generic WMMA. New complete-Op baseline
passes unchanged public/represented FP64 and exact codec checks:
0.15362/0.13406/0.15656ms (`k5-output-bound.json`). Streams16.71168MB packed
weights/scales per tile plus activation/output; T18/24 use two16-row tiles.
Hypothesis: successor-load overlap lowers exposed memory latency at320waves/tile,
without new arithmetic/layout/codec or workspace. Applying25% saving to the
observed6.45% owner projects1.6% whole graph service improvement, worth one bounded
extension test. Same2%RMS/10%gross public oracle; reject any failing width or
non-material whole K5 gain. No speculative gain claimed from raw timing.

Direct candidate: all eight output cells (retainedT2–6 plus newT12/18/24) pass
unchanged independent original-BF16/represented-A8 FP64 criteria, exact generic/
codec/eager/graph, poison and guards. Matched cold medians at T12/18/24:
0.15134/0.13478/0.15660→0.09840/0.09990/0.10048ms (35.0/25.9/35.8%savings).
New kernels70VGPR/22SGPR, no LDS/private scratch, eight nativeIU4 sites, wave32;
all29 preceding instruction/resource streams unchanged. Reports:
`k5-output-qualification.json`, `k5-output-screen.json`. The first qualification
invocation had incorrect argument ordering and failed usage before any GPU work;
`k5-output-qualification-corrected-run/` is the valid execution receipt.
Whole fresh control/candidate comparison admits the extension: fixedK5 C4
140.4883→144.7768 aggregate tok/s (+3.05%,36.1942 per request). All six control/
candidate repetitions match prior ordinary tokens and each records121 verification
rounds. Candidate144.660–144.786 versus control140.172–140.533tok/s, so the saving
exceeds observed repetition spread. `output-control-c4-k5/` and
`output-candidate-c4-k5/` retain commands/reports. Final CLI/server/PPL/bench rebuilt.

## Adaptive startup correction

`k4-adaptive-c4-k5/` retains the pre-inference failure: FP8 projection width not
prepared before graph capture. Only maximum K5 verification width had been
prepared; adaptive K3/K4 graphs also require smaller flattened widths. Shared
family captured-width helpers now supply the exact same width inventory to
artifact loading and Program construction. No preparation/allocation is added
inside graph capture; workspace maximum is unchanged. Existing host execution
contract/adaptive-policy tests pass, including C1–4 width sets. CLI/server/PPL/
bench rebuilt. `adaptive-fixed-c4-k5/` passes all three repetitions with exact
same-C ordinary tokens,154.8209 aggregate decode tok/s, versus fixedK4's161.6563.
First repetition:130 rounds,517 drafted,381 accepted,73.694% acceptance.
Adaptive remains workload-dependent; no claim of universally matching best fixedK.
This repaired-adaptive measurement precedes the final K5 output extension and is
not presented as a fresh final-build adaptive timing.

## Reproduction

From the repository root, use the explicit Python3.11 interpreter and serialize
every build, qualifier, benchmark and profiler job. Build at most14 jobs (normally4).
Each completed cell retains its exact executable command and receipt. Heavy jobs
use a user systemd scope with MemoryHigh20G,MemoryMax24G,MemorySwapMax0; builds/
CPU oracle jobs also use CPUQuota1400%, timing jobs have no CPU quota.

Selected whole cells can be reproduced to a fresh label with:
```
PYTHONPATH=. /home/battlefront/.local/bin/python3.11 \
  profiles/bench/r9700-compact-followup-20260923/benchmark.py \
  --binary build-r9700/bench/ninfer_bench --label FRESH \
  --concurrency 2 3 4 --draft-tokens 4
```
Use `--concurrency 1 --draft-tokens 0 --chunk 4096` for the code4K chunk check;
add `--prompt 8192 --corpus tools/ppl/corpus.ids` for the matched WikiText8K check.
Use `--concurrency 4 --draft-tokens 5 --adaptive` for repaired adaptive mode.
The runner checks repeated tokens; `verify_k4.py` separately checks all selected
K4 repetitions against retained same-C ordinary output. Numerical qualification:
```
build-r9700/src/ninfer_r9700_a8q4_small_batch_projection_qual --out-json FRESH.json --mlp-only
build-r9700/src/ninfer_r9700_a8q4_small_batch_projection_qual --out-json FRESH.json --output-only
```
The unscoped owner still retains the inherited N4096 failure; these scoped
commands must not be presented as its replacement or global accuracy admission.
Raw profiler databases, binaries and code objects remain local prerequisites;
tracked summaries/receipts are sufficient to interpret this bounded decision.
