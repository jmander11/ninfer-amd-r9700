# DFlash remaining projections, verify gate/up and adaptive tails

User-authorized continuation from c63c893f: current generic Q4 call attribution,
verify gate/up physical bottleneck and a new mechanism if justified, adaptive
tail physical-width/logical-budget selection. Prefill and precision changes paused.
Model is the unchanged installed cap26 Q4-head/gate-up-A4 companion, Q4 DFlash
with BF16 selector codebooks; G16 FP8K/INT4V/FP16scales, dense, chunk2048.
One R9700/gfx1201/wave32/device0/PCI0000:13:00.0, ROCm10, auto. Heavy jobs serial,
memory20Ghigh/24Gmax/no swap, no CPU quota on timings, builds4 jobs (hard cap14).

Matched existing control: `r9700-a8-bound-dflash-20260923/final-*` reports and
its frozen `final-bin/bench`: C1 K5/K4 98.0343/84.7129, C4 K5/K4/adaptive
149.0754/161.7356/156.2805 aggregate tok/s, exact ordinary tokens. Reuse these
for attribution; fresh paired timing is required for any new promotion.
Do not rerun rejected split-K/down/PV/staging/gate-up candidates unchanged.

Numerical contract: unchanged original BF16-input/stored-Q4 FP64 oracle with
the documented finite A8 exact-integer/group-FP32-FMA/BF16 error envelope,
separate represented-arithmetic bound, exact codec and eager/graph checks.
Any materially different arithmetic profile needs its own justified criterion,
not relaxed fixture-specific thresholds. Whole greedy-token parity is mandatory.

Adaptive change must retain masked logical proposal/output limits, legal graph
and cache capacity, pending context after K5→K4, and exact public publication.
Never replace measured cost selection with unconditional K3 suppression.

## Current generic projection decision

Fresh C1 K5 trace `c1-current-trace` passes exact ordinary-output comparison.
Generic Q4 accounts for11.59% of graph kernel service. Source/node reconciliation:
N7168/K5120 target full-attention QK/GV14calls; N6144/K5120 draftQKV10calls;
N1280/K5120 dynamic-convolution projections10calls; N5120/K4096 draftoutput5calls,
allT6 (K4 usesT5). The N248320/K5120 vocabulary heads useT5/T6, and the
N256/K5120 selector usesT5; their different bounds do not justify blind admission.

Complete-public-Op cold baseline `projection-baseline.json` uses3 weight copies,
80MiB scrub,24 alternating event pairs and the unchanged FP64/A8 bound. AtT5/T6:
N7168 0.12680/0.12896ms; N6144 0.12326/0.12554ms;
N1280 0.11006/0.11302ms; N5120/K4096 0.10672/0.10714ms.
Minimum packed weight+scale bytes are19.50/16.71/3.48/11.14MB respectively;
activation preparation and output also belong to this measured boundary.
Hypothesis: the generic serial group-load dependency exposes memory latency;
the already-qualified single-successor pipeline overlaps it without changing
arithmetic or storage. These four new shapes affect~5.9% of traced graph service;
25% reduction would save~1.5% of that service, sufficient for a bounded screen.
Each shape/width still needs its own oracle, ISA and whole-inference admission.

Gate/up N34816/K5120 coldT5/T6 baseline0.30744/0.30726ms (94.70MB minimum
weights/scales), while steady traced service is0.1697ms. Cold effective byte rate
is not the steady whole-model bandwidth ceiling. Do not substitute one for the other.

## Gate/up bound review

No new challenger admitted. Steady useful bandwidth is~558GB/s versus the retained
~636GB/s stream control: even eliminating that gap saves~12.3% of kernel service,
~3.1% of this round. A2% round gain would require~8% kernel reduction. Independent
waves grouped into2/4-wave CTAs reduce workgroup count, not traffic or per-wave
instructions; no scheduler/workgroup-slot limitation is established. Loop coarsening
alone retains80 reductions and has no instruction-cost bound supporting that gain.
Previously rejected depth2 payloads force premature waits and higher VGPR pressure
(`r9700-gate-up-depth2-20260922`); do not repeat them. Reopen only for measured
scheduler limitation or a genuinely different emitted-overlap mechanism. This is
a bounded stop decision, not proof of an absolute hardware ceiling.

## Qualified projection extension

`projection-qualification.json`: eight cells pass full original-input/represented
FP64 criteria, exact generic/codec/eager/graph, poisoned-workspace, guards and
mutation checks. `projection-isa.json`: old36 instruction/resource streams exact;
new kernels62–65VGPR/22SGPR, wave32, nativeIU4, no LDS/private scratch.
Matched complete-Op medians (`projection-candidate.json`) save17–18% N7168,
25–26% N6144,49–50% N1280 and26–27% N5120/K4096, acrossT5/T6.
No change to accumulation order, quantization, persistent weights or workspace.

Vocabulary heads remain generic: minimum675.43MB versus traced1.095ms is~617GB/s;
the remaining gap to636GB/s offers only~0.15% round service saving. N256 selector
has only16waves, but its entire traced cost is~0.16% of the round. Neither is a
material independent target in this bounded pass.

## Adaptive transition evidence

Selection uses same-C measured physical cost and clipped logical expected yield,
including one output for an active target-only row. Extra padded routes require
full physical context fit. K3 and existing context-boundary masked fallback stay
eligible; MTP policy is unchanged. Next route is selected at the next real round
boundary after publication/compaction, not from stale post-round membership.
The eager-only decision trace schema2 records `physical_verify_width` separately
from storage `verify_width`, alongside existing logical extents and frontiers.
It is never a timing source and still rejects graph execution.

The first local Engine harness failed before GPU construction because its page
rounding incorrectly used256 instead of the product's64tokens. Failure retained
in `tail-ordinary-run` and `tail-ordinary.json`; corrected helper uses64 and writes
fresh `tail-transitions` evidence. No product defect or timing result inferred.

Cold-start expanded tests expose an inherited width seam: stride17 P4096 unequal
budgets diverge from ordinary with both candidate and c63c893f adaptive policy at
the same first positions66/64/38. The control uses archived policy/runtime headers
with the same exact-generic-qualified core; `parity-limitation.json` retains the
causal scope, not a claim of whole-old-binary identity. Its first link omitted
the runtime archive's non-Program frontend members; corrected `control-tail-link`
adds that archive after the explicit old runtime object. Failed link is retained.
Stride1 (the established benchmark prompts), without graph-prime/warmup, also
exposes failures. FixedK4 andK5 pass all11 stride1 cases exactly, including
short C1, near-context C4, and long unequal limits129/125/119/113.
Attention W4 uses represented-BF16-Q fused arithmetic at4K, while ordinaryW1
and W5/W6 use FP8-Q WMMA. Matching W4's route requires fresh public/FP8-profile
oracle, serial-WMMA parity, graph/guard and whole-token qualification; no predicate
expansion or quality waiver is admitted merely from this diagnosis.

`fixed-cold-comparison.json` isolates the seam: fixedK4 andK5 pass11/11 cases;
fixedK3 fails shortcase5 atoutput6 and longcases7/9 atoutputs36/12. These are
diagnostic runs, never performance-admission data. W4's extension to the existing
batched FP8-Q attention profile is therefore correctness work required for the
adaptive deliverable, not a new speculative precision experiment. Fused W4
and batched W5/W6 can each pass their own oracle yet disagree on greedy tokens
because their private Q precision differs. Maintain the public BF16-Q oracle
plus explicit FP8-Q profile allowance; PPL/prefill/weights remain unchanged.

## Closure

The historical parity limitation above is resolved, not waived. DFlash W4 now
shares the existing W5/W6 batched FP8-Q profile for G16/context64..8191; tree and
other domains retain their routes. Workspace and topology use the same predicate.
`w4-short/stdout.log` and `w4-long/stdout.log` contain16 passing oracle cells:
public BF16-Q FP64 plus existing FP8-Q profile allowance, separate arithmetic
check, serial-bit-exact, graph, guards, C1..4 metadata and poisoned invalid rows.
`attention-isa.json` proves all three renamed device instruction/resource streams
unchanged. No kernel body or profile tolerance was relaxed.

`verify_attention_fix.py` and `tail-w4-summary.json` retain44 exact-ordinary Engine
cases: cold fixedK3, adaptive graph stride1/17, adaptive eager stride1; short limits,
near-context and long unequal budgets. The140-row eager trace observes9 padded
and4 target-only rows. Six pending rows at K4 were not observed; append/storage
are unchanged and the existing maximum-append workspace regression passes.
The old-header causal helper requires the then-current pre-W4 core to reproduce
the historical failure; linking it to today's fixed core is not an old control.

Final unprofiled P4096/G128, auto, chunk2048, warmup1/repetitions3:

| Mode | Final aggregate tok/s | Comparison |
|---|---:|---|
| C1 K5 | 101.5431 | fresh control98.0082 (+3.61%) |
| C1 K4 | 87.6374 | prior-pass84.7129 |
| C4 K4 | 162.2249 | fresh control161.8388; essentially unchanged |
| C4 K5 | 153.9807 | prior-pass149.0754 |
| C4 adaptive maxK5 | 160.8980 | fresh control156.6669 (+2.70%) |

All15 final repetitions match same-C ordinary tokens. Each `final-*` directory
retains command, receipt, report and summary. Reproduce with Python3.11
`run_decode.py --binary build-r9700/bench/ninfer_bench --label <new-label>
--concurrency <1|4> --draft <4|5>`; add `--adaptive` for adaptive maxK5.
Use a new label and the serial resource scope documented above. Scripts use the
explicit installed model paths and ordinary-token references, not artifact globs.
Ordinary decode and prefill were not remeasured; weights/recipe/chunk are unchanged.
Gate/up and vocabulary/selector exclusions above close the bounded investigation,
not an absolute hardware-ceiling claim.
