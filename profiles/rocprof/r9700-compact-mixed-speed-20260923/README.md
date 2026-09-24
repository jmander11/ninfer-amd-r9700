# Selected compact mixed-profile attribution

Question: after mixed gate/up-A4 delivery and prior staged PV/local-fusion work,
which current prefill, ordinary decode, and K5 DFlash kernels still dominate, and
does a new concrete mechanism have material whole-phase benefit?

Source cells: `profiles/bench/r9700-compact-mixed-delivery-20260923/whole-k0` and
`whole-k5`, frozen delivered benchmark, installed17.0025GB compact companion.
C1 codeP4096/G128/chunk2048/context4240, denseG16, auto, one measured region.
Reuse `profiles/bench/r9700-compact-decode-20260923/profile_dflash.py` with explicit
`--cell` and `--out`; it also accepts ordinary source cells and checks their tokens.
Traces are attribution only; original unprofiled reports remain timing authority.
Run captures serially, no build/conversion overlap, in memory-limited systemd scopes.

## Layer 0: compact prefill SiLU→A8 fusion

Fresh ordinary trace: graph0 kernel sum2694.06ms,128 separate SiLU launches47.95ms,
124 Q4 gate/up launches. Two2048-token chunks,62 Q4-down layers per chunk currently
miss the already-admitted fused SiLU/BF16-round/A8-prepare→Q4-down operation because
the target selector unnecessarily requires an FP8 producer. Public inputs to that
operation are BF16 gate/up values and represented Q4 weights, not producer metadata.
The down precision, matrix kernel, T2048 extent, explicit BF16 seam and workspace
remain unchanged. Eliminate124 intermediate BF16 writes/reads and launches; prior
four-role complete-operation evidence motivates a roughly1–2% phase opportunity,
not a measured compact speed claim. Fresh matched whole timing decides admission.

Independent reviewer: SHIP; actual down-A8 guard must remain. Activation and
scratch capacities already suffice. Historical independent oracle evidence:
`profiles/bench/r9700-fused-silu-a8q4-down-p2048-ab-20260904.json` (80 codec probes,
16 complete FP64 probes, exact full workspace/output). The original qualifier
source was removed; do not present its old binary as qualification of current code.
This change only admits an existing operation at a producer-independent BF16 input
boundary. Verify the changed composition with exact mixed-profile NLL sidecars
and ordinary/speculative output tokens, plus current host dispatch tests.

Ordinary graph1:128 launches,4112.99ms kernel sum; Q4 dot8 (primarily gate/up and
head)33.3%, down15.0%, GDN pair10.9%, protected FP8 and remaining projections next.
DFlash graph1:26 launches,1209.09ms kernel sum; gate/up verify24.5%, generic Q4
WMMA15.0%, pipelined down12.5%, batched PV7.1%. These are profiled attribution
fractions, not unprofiled request timings. Prior failed mechanisms remain excluded.

User extended scope to C1–C4. `concurrency.py` takes an explicit frozen binary and
fresh label; runs ordinary/K5 serially, retains tokens, checks speculative parity,
records aggregate and per-request decode rates and nonbinding safety limits.

Baseline timing authority: `baseline-corrected.json`, recomputed directly from
the eight raw reports by `summarize.py`. Earlier `baseline-summary.json`,
`baseline-complete-summary.json` and per-cell summaries are superseded: the first
helper omitted the C multiplier for aggregate decode and divided again for
per-request. Raw benchmark reports and token comparisons are unaffected. A first
summary write also encountered create-only output semantics after C2K5; both C2
cells completed, so neither was repeated. The resumed C3/C4 scope has zero
memory high/max/OOM events and no CPU throttling; C2 scope counters were not
retained after that helper failure.

## Layer 0: ordinary C2–C4 MLP successor pipeline

C2 graph2 Q4 generic kernels own4175.38/6284.25ms profiled kernel service;
gate/up alone1635.97ms (26%). Fresh complete BF16→A8G64→Q4N16K16→BF16
unprofiled cold graph medians (`mlp-bound.json`,3 disjoint allocations,80MiB scrub,
24 alternating samples) for T2/3/4: gate/up0.341823/0.351062/0.354622ms;
down0.290722/0.300602/0.310243ms. Oracle checks original represented BF16
FP64 probes and full decoded represented-A8 FP64, plus exact generic output.
Each gate/up streams94,699,520 weight+scale bytes; down47,349,760, plus activation
codec/output/workspace. Existing T5/6 successor loading reduces the same operation's
load/wait cost; safely specialize its eight-result map at T2/3/4 (not T12+).
Even10% of these62-per-round operations projects3.9ms against roughly52ms/C2 round.
Hypothesis: memory-latency/issue hiding, not a precision change. Preserve ascending
group FMAs, codec, ordinary ownership, T1 fusions and explicit DFlash5/6 route.
Full oracle, exact generic/graph parity, emitted ISA and whole C2–C4 A/B decide.

## Layer 0: K5 concurrent MLP pipeline

Fresh C2 K5 trace attributes1513.73/2340.08ms kernel time to generic Q4 WMMA;
gate/up alone365.17ms. Independent-oracle-qualified cold complete-Op baseline
`verify-bound.json`: gate/up T12/18/24=0.371000/0.496981/0.520261ms;
down=0.354720/0.303580/0.365960ms. Same BF16 input and signed Q4G64/FP16-scale
mathematical contract. Unique weight streams remain94.70/47.35MB respectively;
T18/24 currently use two16-row tiles, potentially rereading those planes.
Test successor loading plus one gathered activation-scale load per lane at these
exact widths, retaining the existing16-row tile count. Extend result mapping to
eight outputs per lane and both lane halves; mask the second-tile tails without
early exits. This attacks load/wait and repeated scale-load issue cost, not state,
quantization or reduction order. A10% MLP win would save about4.5ms per C2 round,
material against approximately2.52s/26 batched rounds. Reviewer approved the design;
public/represented oracles, exact generic outputs, poison/graph/tails and actual
unprofiled same-C whole timings remain mandatory. T<=8 must stay unchanged.

### Numerical gate correction (not a tolerance waiver)

The first tiled qualifier stopped at scaled(-0.75) gate/up T18/token14: sampled
public-input relative RMS2.2077% exceeded2%. Numerical-only diagnosis confirms
every output bit matches the old generic route across both finite fixtures/all
six new shapes; both routes fail the same sampled check at T18/24. Full public
BF16-input FP64 evaluation over all34816 output rows gives worst per-token
RMS1.96781% and gross error4.06539%-of-reference-RMS, within2%/10% respectively.
The30-row estimate was not the complete output norm. Changing the norm population
is explicitly a criterion correction, not automatically a stronger gate. Retain
`tiled-numerical-diagnostic/` and `full-oracle-diagnostic-report/` as failed-sample
and full-output diagnostic evidence. No candidate timing was run after the failure.
Requalify the entire affected suite, including old T2–6 and DFlash5/6, against
the complete public-input oracle for BOTH generic and selected routes. Keep2%
normwise/10% gross constants, nonfinite rejection and exact-zero handling; keep
the independent represented-A8 oracle, exact codec/generic parity, graph and
guard checks.

The full-owner run failed in the unchanged generic control at N4096/K5120/T6,
token5: full-output relative RMS2.0341%, gross3.7310%, before candidate comparison.
Retain `full-public-qualification-run/`; this is a known fixed-A8 public-error
limitation and the full-owner qualification is **FAILED**, not waived. The
new mapping affects only MLP shapes. Independent review permits a separately
labelled `--mlp-only` qualification of the complete changed domain (gate/up
T2/3/4/5/6/12/18/24; down T2/3/4/12/18/24), plus dedicated down T5/6 regression.
Both generic and selected routes must pass the same full-output2%/10% criterion;
retained T<=8 emitted instruction streams are identical. This bounded scope is
not global A8 qualification, and does not remove the default full-owner failing
test. Any failure within this MLP domain excludes the tiled candidate.

Scoped result: `mlp-full-public-qualification.json` passes all14 selected MLP
cells; worst full public relative RMS1.9678088%. Dedicated down5/6 passes
(`down-full-public-qualification.json`, worst0.7267701%). All finite cases retain
exact generic/eager/graph output, independent represented-A8 FP64, codec/status,
poison recovery, guards and immutability checks. These results reopen only the
six tiled MLP timing cells, not global A8 admission.

## Layer 0: ordinary output projection

C3/C4 traces leave generic Q4 at15.6%/14.6% of ordinary kernel service after MLP
pipelining. Static exact selected schedule has60 N5120/K6144 calls per round;
the61st generic call is the head and is excluded. Fresh complete-Op cold control
`output-projection-bound.json` gives T2/3/4 medians0.12892/0.13186/0.13684ms;
three disjoint allocations,80MiB scrub,24 alternating samples, full BF16/public
and represented-A8 FP64 plus exact output. Weight+scale traffic is16,711,680 bytes.
The same shape already uses successor loading at T5/6. Admit that unchanged
eight-result mapping at T2/3/4 to hide weight load/wait latency. A10% saving
projects0.77ms per ordinary C2 round (~2%); plausible larger direct gains warrant
this one bounded extension. Do not extend its predicate to tiled12/18/24.
Require full-output2%/10% oracle at every selected2..6 width, exact generic and
graph/status/guard checks, resource inspection, cold timing, then same-C whole A/B.

All five output-projection widths pass (`output-full-public-qualification.json`,
worst public RMS1.6727451%). Cold complete-Op control→pipeline medians T2/3/4:
0.12620→0.09786 /0.13290→0.09772 /0.13798→0.09810ms (22–29% reduction).
Final linked resources:59/60/61VGPR,22SGPR, no LDS/private scratch, wave32 and
native IU4 WMMA. `final-isa.json` proves all20 prior projection instruction streams
and resources unchanged, including the six newly qualified tiled MLP kernels.
Whole same-C ordinary/K5 results remain the promotion decision.

## Closure

Both extensions win at the whole-inference scope and are selected in the public
Linear dispatch. `final-verified-summary.json` compares all18 C2–4 repetitions
with the prior same-C ordinary tokens: all exact, including K5. Final aggregate
ordinary49.9756/71.0431/86.0301tok/s and K5 119.6639/133.6596/140.3863tok/s.
Against the preceding pipeline checkpoint, ordinary improves5.4/5.7/5.6%,
K5 improves17.8/11.3/12.0%. C1 reuses the unchanged fusion checkpoint:30.1515
ordinary/96.2832 K5; prefill1519.18/1472.19tok/s. No weight/activation changes.
Final CLI/server/PPL/bench link the measured source. Resource limits were
nonbinding, memory high/max/OOM zero and no CPU throttling (`final-resources.json`).

Exact reproduction after building the selected targets (fresh labels/outputs):

```sh
build-r9700/src/ninfer_r9700_a8q4_small_batch_projection_qual --out-json FRESH-mlp.json --mlp-only
build-r9700/src/ninfer_r9700_a8q4_small_batch_projection_qual --out-json FRESH-output.json --output-only
build-r9700/src/ninfer_r9700_dflash_verify_down_qual --out-json FRESH-down.json
systemd-run --user --scope -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=0 \
  env PYTHONPATH=. /home/battlefront/.local/bin/python3.11 \
  profiles/rocprof/r9700-compact-mixed-speed-20260923/concurrency.py \
  --binary build-r9700/bench/ninfer_bench --label FRESH --concurrency 2 3 4
```

Do not repeat the default full-owner qualifier expecting a pass: its inherited
non-MLP failure above remains active and recorded, not suppressed by scoped
commands. Final summary was produced with `summarize.py --label final --c1-label
fusion --reference-label pipeline --output .../final-verified-summary.json`.

Bounded exclusions: no new C1 kernel was justified after prior dot8/down/PV and
staging rejections. Sharing remaining protected-FP8 quantization has at most0.29%
of measured kernel-service opportunity (poison work0.13%); no catalogue/custom
GEMV rewrite was justified. Prefill retained the qualified large matrix kernels
and only removed the demonstrated redundant SiLU materialization. No broad chunk,
precision, power, or K sweep was reopened, and no hardware-ceiling claim is made.
