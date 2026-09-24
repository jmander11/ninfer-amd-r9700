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
