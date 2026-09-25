# R9700 autonomous execution ledger

Status: live work ledger. Every unchecked item is assigned work. This temporary ledger supplements
the active authorities in `docs/README.md`; completed implementation and experiment history belongs
in `docs/performance.md` and `docs/maintainer/r9700-overhaul-plan.md`, not here.

## Fixed execution and product constraints

- Shared-machine resource safety: run heavyweight jobs strictly serially across the primary
  agent and all subagents. Never overlap model conversion/copy/readback, compilation, or
  PPL/inference/benchmark/profiler jobs; wait for the complete job and its children to exit.
  Lightweight editing/read-only review may continue. Default to 12 compile jobs, maximum 14:
  `cmake --build <build-dir> --parallel 12`; never use bare `--parallel`, bare `-j`,
  or uncapped native builds. This is a job cap, not CPU affinity or a memory limit.
  Reduce concurrency further or stop on sustained memory/swap/I/O pressure; 14 is not a target.
  The 2026-09-23 stall coincided with conversion/readback + PPL + an eight-job build;
  swap reached 99.83% and load exceeded 400 with blocked tasks. A build cap alone is insufficient.
- Conserve agent usage: normally use at most one independent implementation agent alongside
  the primary agent. Batch independent, read-only CPU review at material kernel/correctness
  checkpoints; the primary agent handles routine checks and small harness/document edits.
  Do not create jobs to fill slots or restart a reviewer for each mechanical change.
  Give each bounded implementation one owner. Review the material semantics, scope,
  graph/workspace invariants, oracle,
  ISA/resources, command safety, provenance, and decision logic; repair `NO-SHIP` findings and
  require the same reviewer to report `SHIP` before GPU qualification.
- The primary agent alone serializes R9700 work, reproduces preflight and result decisions, and
  pushes coherent WIP checkpoints. Never launch while its owner is editing or review is unresolved.
  At handoff, inspect this ledger, `git status`, pushed HEAD, and retained package summaries,
  receipts, and closure. Never overwrite a failed package; its persisted result is evidence.
- Resolve implementation and promotion questions from the product contract, semantic ownership,
  and retained evidence without asking the user when they determine one strongest route. When one
  compile flag controls independently owned routes, split it at the ownership boundary: promote
  only the route with complete correctness and performance admission, leave unrelated candidates
  unpromoted, and rename or remove the residual flag so its scope is exact. Never extend admission
  from a shared flag to an unqualified shape, layout, group, phase, or caller; preserve its fallback.
  Make routing, workspace planning, reports, tests, and active documentation agree. Record the
  selected predicate, evidence, excluded behavior, and final closure gate in the owning ledger item.
  Ask the user only when the choice changes the product contract or qualified alternatives retain a
  material unresolved tradeoff.
- The product workload is exactly one R9700 and startup-fixed `C=1..4`. Never schedule or require an
  active `C>4` cell. Retained `C=5..8` rows are historical only.
- Performance-admission timing requires device 0, Radeon AI PRO R9700, `gfx1201`, wave32, and power
  profile `auto`. Profiled timing is attribution-only. Do not monitor temperature or fan speed
  unless observed evidence first indicates throttling. Do not change clocks, power, fan, or
  temperature policy.
- The only growing cache is FP8 E4M3FN keys, signed INT4 values, and FP16 value scales. Candidate
  selection remains all-Q4, mixed Q4/W8, and four-role FP8/Q4, crossed with G16/G32 and dense or
  B128/S16/tau900 Text prefill. There is no runtime cache or artifact-recipe selector.
- Quality eligibility requires finite aligned sidecars, delta mean NLL at most `0.02` and at most
  `max(4, ceil(0.001 * scored_positions))` new severe positions for accuracy, or at most `ln(1.05)`
  and `ceil(0.0025 * scored_positions)` for capacity-speed. Schema-v7 terminal selection uses
  `global_maximin_whole_then_capacity_then_quality_then_canonical_v1` over the complete matched
  candidate set; quality is an admission gate, not the speed objective.
  A complete finite measured quality failure is a retained candidate exclusion, not a reason to
  loosen its tier, rerun unchanged measurements, or reject the complete evidence set. Preserve
  all twelve identities and failing sidecars; do not time a quality-ineligible candidate. Dense/
  sparse capacity symmetry does not require identical quality eligibility: a failed sparse route
  must not discard its passing dense control. Malformed or incomplete evidence still blocks selection.
- Canonical dense Q4 uses scalar-base/U32-voffset N16/K16 ping/pong. Its source-matched
  C1/P2048/G0/chunk4096/spec-none result is `1904.339303 tok/s`. On 2026-09-21 the user accepted
  this speed and authorized one bounded review for overlooked large prefill gains, followed by
  numerical accuracy and DFlash optimization. The 2,000+ tok/s target and unproved practical
  ceiling no longer block chunk, accuracy, or artifact selection. Implement another prefill
  candidate only for a concrete new mechanism with credible whole-prefill benefit; unexplained
  theoretical headroom alone does not justify an extended search.
- Ordinary non-speculative decode retains the selector-free `27.05729956 tok/s` result, but the
  user's 2026-09-19 direction reopens only its memory-throughput optimization as
  `BASE-DECODE-BW`. This baseline is not proof of bandwidth saturation or stall freedom.
- DFlash2 is the required speculative direction. Test production choices only at K4/W5 and K5/W6;
  do not resume the K1..11 shortlist or optimize MTP. Preserve both selector codebooks and private
  DFlash state in BF16. Exact public greedy-token parity is mandatory and cannot be waived by logit
  margins. Recipe-independent DFlash work may proceed now; companion selection and physical
  C1..4 admission wait for the schema-v7 base and receipt-bound shared chunk.
- Dense remains the product Text-prefill route until XAttention passes matched quality, capacity,
  whole-inference, selected-profile NIAH, and cutover gates.

Dependency notation is `[depends: ...]`. `DENSE-FLOOR-DECISION` is satisfied by the user's
2026-09-21 acceptance above. Conditional tasks retain their explicit `if:` clause.

## Mechanical continuation protocol

Run from `/ssdpool2nvme/local_llm/ninfer-amd-r9700`. Stop at the first failed prerequisite and
retain it; never continue a failed package to timing. The fixed GPU, delegation, and power rules
above apply to every command.

### Current checkpoint

COMPLETED USER REQUEST (2026-09-25, base-prefill compute campaign; XAttention/Sage deferred):
- [x] Gate/up back to A8 for no prefill speed loss: new M128xN128 A8Q4G64 prefill GEMM
  (token-fastest raster, exact magic-number I32->FP32, bit-exact to the prior A8 kernel,
  FP64 qualifier PASS) and default `NINFER_R9700_Q4_PREFILL_A4_FAMILIES=0`. 4K prefill
  1687.6->1725.8 tok/s, 8K 1407->1512, 32K 951->970; worst 4K prefill PPL vs NVFP4
  +1.56%->+0.008%. Evidence: `profiles/bench/r9700-a8-gateup-20260925/`.
- [x] Fused streaming dense prefill attention (no FP32 score plane or caller workspace; same
  BF16-QK / FP16-PV operand precisions, online FP32 Softmax). Dense qualifier FP64 PASS through
  262144; 32K WikiText mean-NLL +0.0031 (final 512) / +0.0006 (16383 positions, 8 new/8
  resolved severe). Prefill 4K 1726->1972, 8K 1512->1853, 32K 970->1591, 64K 678->1349 tok/s.
  Evidence: `profiles/bench/r9700-fused-attention-20260925/`.
- [x] Re-profile whole prefill; iterate on the dominant kernel until each is near its
  measured limit. Done: fused attention v2 (32 rows/CTA, full-D waves, spill-free,
  32K 1591->1643, 64K 1349->1428); A8 GEMM scale-first loads, U32 scalar-base offsets, peeled
  loop and WMMA/VALU interleave (bit-exact, 2-5% per shape); barrier-free staged GDN prefill
  recurrence for normalized widths 64..8192 (P2048 bit-exact, 2.85->1.39 ms/call). Whole
  prefill now 4K 2174, 8K 1976, 32K 1765 tok/s (`profiles/bench/r9700-gdn-gemm-20260925/`).
  GEMM remains bounded by its G64 two-sided scale epilogue under the 300 W cap.
- [x] Evaluate FP8 for remaining BF16/FP16 components: FP8-Q/FP8-K QK in the fused kernel gives
  32K +5.6%, 64K +10.4%, no 8K/4K gain, PPL unchanged within noise; not selected pending the
  user's decision (evidence `profiles/bench/r9700-fp8qk-eval-20260925/`). FP8 PV (Sage-style) not
  evaluated per the user's deferral. README, performance, softmax/model/artifact docs, qualifier
  README and static gates updated; all 86 registered tests pass.

COMPLETED USER REQUEST (2026-09-25, further long-context optimization):
- [x] Investigate serially and commit each admitted speedup before the next:
  PV tile/split scheduling (`e3e033d2`), matrix PV (`2ac730d2`), wider QK reuse
  (`4c7a763a`), and earlier matrix-PV crossover (`ee233cb9`).
- [x] Retain exclusions: probability materialization and register-owned probability
  mapping regress; FP16-operand denominator staging has no demonstrated material
  whole-model gain. Further split counts do not justify another production schedule.
- [x] G16/G32 FP64/graph/metadata/workspace/ISA, C1–4 planner and matched32K
  model-quality gates pass. Final C1/chunk2048 prefill:8K1490,32K967,64K672 tok/s.
  Final whole-phase profile and precise evidence/limits are in `docs/performance.md`.
- [x] Refresh the incremental runtime image and verify its32K GPU NLL/argmax sidecars
  are byte-identical to the native build. Server remains stopped; heavy jobs stayed
  serial, no128K whole-model rerun, no numerical-tolerance changes.
  Older campaigns remain paused. A fused streaming attention algorithm remains an
  unproved research possibility, not a measured speedup or a claim of hardware saturation.

COMPLETED USER REQUEST (2026-09-25, fix extreme long-context dense prefill slowdown):
- [x] Qualify and time PV query-tile and split-KV challengers against the existing dense route.
  Layer0: at32K PV owns63.47% of prefill GPU service; the bounded384MiB score
  buffer reduces full panels to128 query rows/32 PV blocks (32 rows/8 blocks
  at128K). Smaller PV query tiles increase independent blocks and reduce per-thread
  accumulators, at the cost of more repeated V reads. The tile-only candidate is
  bit-exact; adding16 key splits changes FP32 association, but not represented inputs,
  causal/page semantics, FP32 intermediates or the independent FP64 oracle criterion.
  G16 complete attention at32K:776.43→358.72ms; at128K:7478.94→993.55ms.
  G32 at32K:770.85→358.61ms. New nonperiodic FP64 conformance passes both groups,
  including graph replay, active rows, invalid metadata and context boundaries.
  Native FP32 ISA, wave32 and zero-spill checks pass. Timings include
  unchanged QK/max and all panels, under auto. No unchanged128K Engine rerun.
- [x] Promote only a materially faster qualified route, verify the public boundary
  and matched8K/32K whole-prefill results, then test larger contexts with bounded
  runtime if the improvement supports it. Keep container stopped and heavy jobs serial.
  Selected: four query rows/16 key splits from context12288, FP32 partial merge,
  and evenly balanced16-row query tiles across the minimum panel count. No candidate flag.
  Final G16/G32 FP64, graph/metadata/boundary checks and C1–4 workspace checks pass.
  Whole prefill8K/16K/32K/64K:1240/889/629/393 tok/s;32K and64K improve1.54×/2.61×.
  Matched32K WikiText final512-position PPL6.543→6.438, zero new severe positions.
  Production build and reusable-harness tests pass. Detailed evidence, numerical limits,
  workspace cost and remaining dense quadratic work are in `docs/performance.md`.
  The128K whole-model point remains canceled; the retained128K result is Op-only.
  Incremental Compose image refreshed with `scripts/hot-patch.sh --image-only`;
  server remains stopped. Start the updated image with `docker compose up -d --no-build`.

COMPLETED USER REQUEST (2026-09-25, prefill context ladder and precision comparison):
- [x] Stop Compose; expose existing per-request trailing prefill timing in benchmark
  JSON, without changing inference arithmetic. Focused benchmark contract test passes.
- [x] Measure C1 dense prefill at8192/16384/32768/65536 with the installed
  selected model and chunk2048; report average and trailing rate separately.
  User canceled131072 after >46 minutes on2026-09-25. Retain its incomplete
  command/log; do not repeat this point unchanged or report an inferred speed.
- [x] Profile a representative longer-context prefill, attribute its bottleneck,
  and compare actual activation routing with retained matched5090 PPL evidence.
  Leave the container stopped. No conversion or new long-context PPL campaign.
  At32K, PV63.47%, QK13.28%, maximum2.32% of prefill/setup GPU service;
  final-chunk PV launches32 blocks. Details and retained5090 comparison in
  `docs/performance.md`; no production kernel change or ceiling claim.
- [x] Retain a reusable context-speed and matched-token PPL setup for future XAttention
  testing; validate matching inputs/profiles, resumable evidence and offline comparison.
  Do not resume XAttention implementation or the older admission campaigns here.
  `tools/bench/context_ladder.md` owns commands; seven focused tests, retained-speed
  collection, real256-token PPL smoke/resume and retained/fresh trace analysis pass.

COMPLETED USER REQUEST (2026-09-25, post-fix optimization investigation):
- [x] Verify Compose healthy on8001; isolate one fresh long-context prefill/decode
  trace and chunk4096 screen, then restore the server. No product changes.
- [x] Attribute long-context prefill slowdown and record bounded next targets in
  `docs/performance.md`: dense PV37.23% of prefill GPU service, verify PV23.05%
  of decode graph service; retain chunk2048. Adaptive-versus-fixed policy gap
  remains a candidate investigation, not a proven bug or authorized policy change.
  Evidence: `profiles/rocprof/r9700-long-context-followup-20260925/`.

COMPLETED USER REQUEST (2026-09-25, production long-context DFlash): integrate the
measured batched-WMMA route, qualify it, and build/validate the Compose deployment.
- [x] Align dispatch, full score-workspace capacity, and graph topology through262144.
- [x] Public target numerical/workspace/graph qualification; matched long-context
  ordinary/fixed/adaptive whole-Engine accuracy and speed, including C1–4 coverage.
  Public FP64/profile, guards, pending publication and graph replay pass; whole
  exact-token checks pass for K3/K4/K5/adaptive, both4096/8192 transitions,
  C2–4 fixed/adaptive, eager, and a second long technical-text corpus.
- [x] Incrementally deploy and validate the real Compose server; record results and limits.
  Healthy on8001; Chat/Responses/Anthropic/SSE and cache restoration pass.
  A fresh15424-token chat attemperature1.5/adaptive measured51.5 decode tok/s
  (59.1% acceptance). Matched greedy code P15200 K5 improves24.29→83.75 tok/s;
  adaptive56.06→65.63. Details/limits in `docs/performance.md`, commands and
  logs in `profiles/bench/r9700-long-context-production-20260925/`.

COMPLETED USER REQUEST (2026-09-25, long-context DFlash attention investigation):
- [x] Compare W4/W5/W6 through the native 262,144-token limit: all 18 long-context
  cells pass existing independent-oracle checks and favor batched WMMA; tiny
  6/16/32-token controls favor fused. Retain the current lower cutoff.
- [x] Poisoned graph replay and extra nonperiodic V checks pass at 15,200/262,144
  for all three widths. Reviewer SHIP for investigation, not production admission.
- [x] Record timings and required workspace/graph/model-quality qualification in
  `docs/performance.md`; commands and raw evidence are under
  `profiles/bench/r9700-long-context-wmma-20260925/`.
Production integration and whole-model results supersede the investigation-only
status above; see the completed request and `docs/performance.md`. Unrelated campaigns
remain paused.

COMPLETED USER REQUEST (2026-09-24, Docker Compose deployment):
- [x] Add AMD Compose runtime, explicit artifact/build contexts, temperature1.5,
  4GiB RAM/32GiB persistent disk prefix cache, bounded host memory and build jobs.
- [x] Verify resolved Compose config and native-equivalent server settings:
  Chat/Responses/Anthropic/SSE, resolved temperature, RAM hit and disk restart hit.
- [x] Rebuild and run all86 registered host/GPU tests; correct stale qualifier
  expectations. Real Engine cancellation/RAM and exact-output disk restart pass.
- [x] Build and run the actual Compose image with Buildx. All three apps compile;
  healthy container passes four HTTP routes, temperature1.5, RAM restoration and
  restart disk restoration (12 prompt tokens reused,22 output tokens). Dedicated
  builder capped24GiB/no swap; exclude profiler dumps from Docker context.
- [x] Move Compose disk storage to an explicit NVMe bind at
  `/ssdpool2nvme/local_llm/cache_r9700` (entries in `prefix/`), separate from5090.
  Verify actual mount, four HTTP routes/RAM reuse and restart disk hit12tokens.
  Document existing AMD incremental `scripts/hot-patch.sh` workflow for Compose.
- [x] User adjustment: default 12 compile jobs across image/dev/test builds; remove
  dedicated Buildx memory/swap limits, retaining its 14-CPU quota and cached state.
  Server 24 GiB memory limit is unchanged. This supersedes earlier four-job defaults.
Image/service/cache retained; test server and builder stopped to release resources.
Local `.env` is configured and ignored. Old Docker named cache volume is retained
unused; see `docs/containers.md` for validation and incremental-build commands.
Older performance/quality campaigns stay paused.

COMPLETED USER REQUEST (2026-09-24, test and retain beneficial candidates):
- [x] Qualify and time gate/up cross-tile weight reuse at T20; keep only a measured win.
- [x] Qualify and time adjacent-feature PV reuse; keep only a measured win.
- [x] Qualify and time exact-tree GDN normalization, preserving replay state.
- [x] Remove losing candidates, verify retained production routes and whole decode,
  update evidence/delivery and commit natural checkpoints.
Unchanged installed weights/precision/cache/chunk2048, C<=4, auto power;
heavy jobs serial, builds capped at4 normally (never above14). Older campaigns paused.
Closure: gate/up T18/20/24 cold complete-Op savings28–31%; paired PV only W6/G16/
feature-fast at4096<=context<8192 (cold savings9–12%); exact-tree GDN only actual
batch1/4 after batch2/3 showed no benefit. Independent oracles, exact graph/state,
ISA/resources and replay-fold pass. Final C1K5/C2K5/C3K4/C4K4 rates are
105.15/149.40/191.05/203.26 aggregate tok/s, P4096/G128; C4K5/adaptive maxK5
186.88/201.04. All24 final repetitions and44 cold transitions match ordinary
tokens. CLI/server/PPL/bench rebuilt; four host and two static tests pass.
C1 reuses its unchanged qualified body measurement; concurrent modes are fresh.
Evidence/reproduction: `profiles/bench/r9700-remaining-candidates-20260924/`;
`docs/performance.md`. No renewed prefill/quality campaign or ceiling claim.

COMPLETED USER REQUEST (2026-09-24, investigation only):
- [x] Compare C3/K4 and C4/K4 projection tile costs and identify a distinct reuse mechanism.
- [x] Investigate attention PV and GDN recurrence/record costs against prior exclusions.
- [x] Record ranked next experiments, bounds and limitations; no production kernel change
  or renewed prefill/quality campaign is implied by this investigation.
Start from a24d51f9, unchanged installed model/precision and auto power. Heavy work
is serial; one CPU-only independent review. Reuse completed evidence where valid.
Closure: fresh exact-token C3 trace versus retained C4,31 steady rounds each,
shows gate/up180.50→324.83us/call but down93.76→94.10us. Prioritize T20
cross-tile weight reuse, not every projection. Next are adjacent-feature INT4 PV
reuse and exact-tree wave GDN normalization (not the different-order wave-QK
candidate). GDN record computes recurrence; there are no per-token full snapshots
to delete. C4 acceptance also differs, so tiling is not the sole scaling cause.
No production change or new speedup claim. Ranked mechanisms, bounds, risks and
reproduction: `profiles/bench/r9700-concurrent-overhead-20260923/remaining-investigation-20260924.md`.
Implementation is a follow-up choice, not an unchecked assignment from this investigation.

COMPLETED USER REQUEST (concurrent projections, round overhead, attention/FP8):
- [x] Profile current C2..4 Q4 projection gaps, qualify and promote material
  same-contract shape extensions; compare affected whole decode workloads.
- [x] Attribute DFlash GPU idle gaps, synchronization and host round overhead;
  optimize only an identified material mechanism.
- [x] Reassess decode attention and protected FP8 projections from current traces;
  pursue a new bounded mechanism or record evidence-backed exclusions.
- [x] Verify/rebuild affected delivery, record results and natural commits.
Continuation from8562efdd; unchanged installed weights/precision/cache/chunk2048,
C<=4, prefill and older campaigns paused. Heavy jobs strictly serial and resource
bounded; one independent CPU review agent. No repeat of rejected mechanisms
without new evidence. No absolute-ceiling claim from kernel bandwidth alone.

Closure: 45 additional concurrent Linear cells promoted after all48 affected
FP64/exact/graph cells passed and144 deliberate corruptions were rejected;
prior44 ISA/resource streams remain exact. Paired-row QK passes nine complete-Op
cells and canonical short/long/invalid-metadata checks; softmax/PV are unchanged.
Final P4096/G128 C1K5 is102.73tok/s; C2/C3/C4 K4 aggregate is
146.20/190.96/181.89, gains19.3/19.2/12.1% against fresh controls.
C4K5 is166.90 (+8.3%); adaptive maxK5 is180.21. All18 final repetitions and
44 cold transition cases match ordinary tokens; fixed acceptance/round counts
are unchanged. CLI/server/PPL/bench rebuilt, four focused host and two static
tests pass. Host overhead and FP8 rewrites are excluded by the measured bound,
not by an absolute-ceiling claim. Six pending rows at K4 remain unobserved;
append/storage are unchanged. Evidence and reproduction:
`profiles/bench/r9700-concurrent-overhead-20260923/`; `docs/performance.md`.
Older campaigns and prefill remain paused; no new campaign is required by closure.

COMPLETED USER REQUEST (remaining projections, verify gate/up and adaptive tails):
- [x] Attribute remaining generic Q4 calls on the current delivered build; qualify
  and promote a new specialized route only where its whole-inference bound is material.
- [x] Profile verify gate/up's physical bottleneck and evaluate a genuinely new
  latency/issue/traffic mechanism; preserve prior rejected alternatives.
- [x] Improve adaptive tail selection by separating logical output allowance from
  legal physical captured width; qualify budget/context transitions and exact tokens,
  measure against fixed/adaptive controls, and retain only demonstrated improvements.
  Fixed the inherited cold K3/W4 attention seam through the qualified matching
  FP8-Q batched route, not K3 suppression or a token-parity waiver.
- [x] Rebuild affected deliverables, record wins/exclusions and natural commits.
User authorized all three targets after checkpoint c63c893f. Installed cap26
Q4-head/gate-up-A4 weights, BF16-codebook Q4 DFlash, fixed KV and chunk2048 remain
unchanged. Prefill remains paused. Serial heavy jobs, C<=4, auto power, build cap14
(normally4), one independent agent at most. No arbitrary precision/policy changes,
unconditional K3 removal or repetition of rejected mechanisms without new evidence.
Closure: eight new T5/6 Linear cells qualify; old36 ISA streams remain exact.
Final C1 K5/K4 is101.54/87.64tok/s; C4 K4/K5/adaptive is162.22/153.98/160.90
aggregate. Fresh C1K5/adaptiveC4 controls98.01/156.67 yield3.61%/2.70% gains;
fixedC4K4 is essentially unchanged. All15 repetitions match ordinary tokens.
Sixteen W4 attention oracle cells and44 cold Engine transition cases pass;
the six-pending-at-K4 event was not observed, but unchanged append/storage's
maximum-capacity regression passes. CLI/server/PPL/bench rebuilt; four focused
host/state tests pass. Gate/up has no justified new mechanism; this is not an
absolute hardware-ceiling claim. No open task remains in this bounded request.
Evidence and reproduction: `profiles/bench/r9700-dflash-projections-tail-20260923/`;
current results: `docs/performance.md`. Older paused campaigns below stay paused.

COMPLETED USER REQUEST (numerical qualification and decode follow-up):
- [x] Derive and implement an A8 implementation-profile numerical criterion from
  represented-input quantization and floating-point error bounds; retain the
  original BF16-input FP64 oracle and failing fixture, requalify the entire
  affected small-batch/down domain, and keep model PPL admission separate.
- [x] Inspect current-compatible C1 DFlash attribution and pursue only a genuinely
  new, bounded material mechanism; qualify and measure any proposed improvement.
- [x] Diagnose adaptive C4 versus fixedK4, including actual captured-K residence
  and timing-policy overhead; fix a demonstrated issue and verify exact tokens.
- [x] Rebuild affected deliverables, document conclusions and commit natural checkpoints.
User authorized this follow-up after clarifying that2.034% is operator RMS error,
not PPL degradation. No user choice remains pending. Preserve weights/precision
unless evidence establishes a necessary change; prefill and unrelated campaigns
remain paused. Serial heavyweight jobs, explicit build cap14 (normally4), C<=4.
Numerical checkpoint: all32 small-batch plus two down cells pass the analytically
derived quantization+FP32-FMA/BF16 envelope and separate arithmetic bound;102
output corruptions rejected. Original public FP64 oracle, historical2.03407%
RMS and model PPL gates are unchanged. Evidence:
`profiles/bench/r9700-a8-bound-dflash-20260923/`.
Adaptive checkpoint: fixedK4 161.71 versus adaptive156.51 aggregate tok/s, all
exact ordinary tokens. Matched traces isolate most of the gap to K3/W4 tail
execution as lanes drain; steady C4/K4 differs only0.41%. Current budget clamp
is intentional and tested, not an implementation bug. No unconditional K3 removal
or policy rewrite; fixedK4 remains the measured concurrent recommendation.
Closure: ordinary Linear now owns draft down/feature T5/6 successor kernels;
the duplicate target-verify API/kernel is removed. Final four-cell public oracle,
exact codec/graph/poison checks pass; old32 projection and old target-down ISA
are unchanged. Final C1 K5/K4 98.03/84.71 versus95.97/83.17tok/s; C4 K5
149.08 versus144.82 aggregate. Final C4 K4/adaptive161.74/156.28 are essentially
unchanged. All15 final repetitions exact ordinary tokens. Delivered binaries
rebuilt, focused host/Python checks pass, model guidance updated. No weights,
precision or PPL changed; prefill remains paused. Bounded request complete, no
hardware-ceiling claim or pending user decision. Future adaptive policy work is
only a recorded experiment direction, not an unfinished item in this pass.

COMPLETED USER REQUEST (2026-09-23, follow-up): resolve the inherited A8 public-oracle
failure first, then work through final-build attribution, useful DFlash coverage,
remaining material prefill/decode mechanisms, and chunk selection including4096.
- [x] Diagnose N4096/K5120/T6 failure against represented public inputs; repair the
  actual cause with justified numerical criteria and affected-domain qualification,
  not a threshold increase to make a fixture pass.
- [x] Reprofile final C1/C4 ordinary/K5 paths; inspect intermediate C only where
  evidence can change the decision. Retain prior exclusions.
- [x] Evaluate K4/adaptive-width coverage and remaining material decode mechanisms;
  promote only independently qualified, whole-inference wins.
- [x] Evaluate material prefill mechanisms from current attribution, then compare
  chunks2048/4096 (1024 control where useful) on matched prompts and quality.
- [x] Rebuild/verify selected routes, record outcomes/exclusions and selected chunk,
  update installed model guidance and commit coherent checkpoints.
Preserve installed selected weights, fixed KV format, C<=4, auto power, serial
heavyweight work and build cap14 (normally4). No unrelated paused campaign resumes.
Follow-up checkpoint: inherited failure is cancellation-amplified A8 quantization
(ideal represented-A8 FP64 error2.01625%, GPU2.03407%); arithmetic/codec checks
pass. Scale-refinement experiment worsens output error and is rejected. The user
authorized principled numerical qualification in the active request above; no waiver.
K4 MLP T10/15/20 extension passes six full-oracle/graph cells and exact old ISA.
C2/3/4 aggregate decode98.32/132.17/142.61→122.14/160.87/161.66tok/s; all nine
repetitions exact same-C ordinary tokens. Chunk4096 initial code4K prefill1459
versus1524 at2048; WikiText8K1198 versus1242 also favors2048. Matched2048/4096
tokens exact;1024 control1473 on code4K. High/low-BF16 WMMA PV passes two
oracle cases but loses complete-Op timing, so removed. Adaptive C4 exposed missing
smaller FP8 graph widths; shared captured-width preparation fix passes host tests
and three C4 repetitions exact ordinary tokens at154.82 aggregate tok/s.
K5 output T12/18/24 passes eight output-oracle cells; direct savings26–36%,
fresh C4 whole140.49→144.78 aggregate tok/s, all repetitions exact ordinary
tokens. Retained instructions unchanged. Delivered binaries rebuilt and host
contracts pass; installed model guidance updated. Bounded optimization pass done.
The numerical item is resolved by the principled criterion review above, not a
fixture-specific threshold increase. All independent priorities of this preceding
pass are complete. Preserve failed evidence.
Evidence:
`profiles/bench/r9700-compact-followup-20260923/`.

COMPLETED BOUNDED USER REQUEST (2026-09-23): profiler-guided prefill and decode optimization on the
delivered cap26 Q4-head mixed gate/up-A4 model and Q4 DFlash companion. Preserve selected
precision, quality and exact ordinary/speculative token semantics; do not reopen recipe search.
- [x] Profile measured C1–C4 ordinary/K5 runs; identify current phase owners and retain prior rejections. Qualify concurrent output correctness and report aggregate and per-request throughput separately.
- [x] Pursue bounded new mechanisms with a material whole-phase bound; qualify with independent oracles,
  review, ISA/resources as needed, and unprofiled A/B before promoting any winner. Record exclusions.
- [x] Rebuild/verify affected delivered routes, document speed/quality evidence and final weight/activation policy.
All heavyweight jobs serial; C<=4; build cap14 (normally4); memory safety limits; power auto.
Other paused work remains paused. Stop this pass when credible bounded mechanisms are exhausted,
not on an unproved assertion of a hardware ceiling.
Checkpoint: compact T2048 SiLU/A8 fusion preserves six exact NLL sidecars and raises
C1 prefill1494→1519tok/s. Ordinary T2–4 MLP successor pipeline passes independent
oracles/ISA/direct A-B and18 whole repetitions; C2/3/4 aggregate decode38.15/53.90/65.79
→47.41/67.19/81.50tok/s, exact baseline tokens. C1 and K5 decode unchanged.
Closure: promote K5 T12/18/24 tiled MLP and ordinary N5120/K6144 T2–4 pipeline.
Final C2/3/4 aggregate ordinary49.98/71.04/86.03 and K5 119.66/133.66/140.39tok/s;
all18 repetitions exact prior same-C ordinary tokens. C1 remains30.15/96.28.
MLP14cells, dedicated down2cells and output5cells pass full-output FP64 criteria;
retained instructions exact. All delivered binaries rebuilt; focused host checks pass,
limits nonbinding, independent final review SHIP. Installed model README and active
performance/qualification docs updated; unchanged weight/activation recipe.
Full-output oracle exposed inherited generic N4096/T6 2.0341%>2% failure, retained
explicitly: no global A8 qualification or hardware-ceiling claim. Other paused work
is not reopened by this bounded closure. Evidence:
`profiles/rocprof/r9700-compact-mixed-speed-20260923/`; baseline authority is
`baseline-corrected.json`, final authority `final-verified-summary.json`.

COMPLETED USER REQUEST (2026-09-23): deliver the selected compact Q4-head model with
gate/up-only A4 prefill and A8 decode. Reuse the installed base/companion bytes, not a new quantization.
- [x] Select family1 in default and delivered builds; keep the qualified fast gate/up kernels.
- [x] Record exact installed model paths, weight recipe creation, and activation/build commands.
- [x] Verify routing, matched quality, and real ordinary/DFlash execution; record measured delivery status.
Heavyweight work is serial, builds explicitly capped at14 or fewer jobs; unrelated paused work stays paused.
Closure: default/delivered family1; six mixed-profile NLL sidecars byte-exact; all installed
base/companion bytes reused unchanged with adjacent creation receipts and updated README.
C1 codeP4096/G128/chunk2048:1494.30tok/s ordinary prefill,30.15 ordinary decode,
83.20 K4,96.05 K5,92.53 adaptive(maxK5) outputtok/s; all3 repetitions/modes exactordinarytokens.
No binding memory limits/CPU throttling. CLI/server/PPL/bench and focused contracts pass;
independent review SHIP. Evidence: `profiles/bench/r9700-compact-mixed-delivery-20260923/`;
selected-delivery section in `docs/performance.md`. Uniform-control tools reject mixed snapshots.

COMPLETED USER REQUEST (2026-09-23): endpoint/mixed-activation quality search with theoretical
performance ranking. No speed benchmarks or kernel tuning in this pass. Preserve all prior
reference/candidate evidence and the delivered default.
- [x] Create exact-copy W8 embedding-only, head-only, and both-endpoint variants of selective-cap26
  and the near-miss no-late-MLP22 recipe; qualify inventory, binding and endpoint layouts.
- [x] Measure matched three-text prefill/decode PPL for useful endpoint/activation combinations.
  Start with selective-cap endpoints crossed with the six existing A8/mixed profiles; use controls
  and promising profiles to refine the smaller22-protection recipe. Retain quality failures.
- [x] Rank measured quality against modeled active weight bytes and integer matrix work, explicitly
  excluding unsupported tok/s predictions. Distinguish embedding residency from per-token traffic,
  output-head cost, A4 compute opportunity, FP8 work and unmodeled KV/state/launch costs.
- [x] Record a concrete quality-qualified optimization shortlist and next kernel work; do not
  promote unmeasured performance or claim global optimality. Preserve BF16 DFlash codebooks/state.
Use the immutable default5090 reference, C1, chunk2048, and unchanged scoring spans. PPL2% is a
preferred screen, not a statistical cliff; report per-text NLL, severe positions, and near misses.
No new NVIDIA measurement is possible while that GPU is absent. Other paused work remains paused.
Closure: 27 configurations / 162 successful scoring cells; six exact-copy endpoint artifacts.
Cap26 W8-head/gate-up A4 passes (worst prefill +1.116%, decode -0.879%); no tested endpoint
enables MLP-wide/all-projection A4 within2%. Q4-endpoint cap26/gate-up remains the compact
speed-oriented shortlist leader; W8 head adds644.14MiB and4.73% logical decode weight bytes.
Cap22 W8-head/gate-up is a near miss (+2.156%). BF16-head controls do not change selection;
all six decode NLL sidecars match their A8-head controls exactly. No speed or production promotion.
Results, limitations, and conditional future priorities: `docs/performance.md`, endpoint precision
section; full cells/commands: `profiles/ppl/r9700-endpoint-precision-20260923/`. Preserve them.

COMPLETED USER REQUEST (2026-09-23): bounded NVIDIA-aligned AMD precision search.
Keep the saved default NVFP4 reference immutable. Compare actual weight protections and scale
storage with both NVIDIA artifacts; do not equate integer A4 with floating NVFP4 A4.
- [x] Inventory protected roles and scale/codebook bytes in the installed artifacts.
- [x] Screen broader A4 prefill coverage (MLP, then eligible Text projections) on the saved
  selective-cap weights using the same three texts and prefill/decode scoring schedules.
  Decode/verify stay A8. Qualify newly selected public Linear shapes before timing.
- [x] Use the protection inventory and quality evidence to test a bounded NVIDIA-aligned weight
  recipe where it could improve the speed/quality frontier; retain W4A8 for sensitive operations
  and promote to W8/FP8 only where needed. Measure eligible whole-inference candidates, not
  theoretical bit-width speedups. Preserve BF16 DFlash codebooks/state.
- [x] Record the measured choice, scale costs, failures and reproducible commands; retain the
  current delivered default unless a qualified candidate is a demonstrated better tradeoff.
All42 new quality cells completed. No new candidate passes2% on every text/schedule, so no
timing or promotion. Closest alternatives: gate/up+attention-input A4 worst+2.42%; removing
late MLP protections saves250,994,688bytes but reaches+2.232% technical decode. Keep existing
selective-cap/uniformA8 default and its separately evaluated gate/up-only mixed profile.
Inventory, scale costs, failures and commands: `docs/performance.md` and
`profiles/ppl/r9700-nv-aligned-precision-20260923/`; runner `tools/ppl/select_nv_aligned.py`.
Frozen NVIDIA reference is intact; no visible NVIDIA GPU for additional328MiB PPL. No
unbounded layer search or unrelated paused work was opened.
The near-NVFP4 screen remains <=2% worse PPL per text/schedule with severe-position deltas
reported separately. This is a bounded local comparison, not global optimality or BF16-source
production admission. Unrelated paused tasks remain paused.

COMPLETED USER GOAL (2026-09-23): optimize the saved Q4/FP8 selective-cap model to at least
30 output tok/s ordinary decode and 60 output tok/s DFlash at C1; continue past these targets
when bounded profiling identifies clear material wins. This explicitly reopens kernel work,
recipe-specific DFlash integration, and mixed-A4 cooperative prefill, with decode first.
Historical rates on other artifacts/workloads are motivation, not matched baselines or ceilings.
- [x] Establish matched C1 baseline and whole-inference attribution for the installed selective-cap
  artifact; determine whether mixed-recipe dispatch bypasses admitted Q4 fused routes.
- [x] Qualify and optimize base-decode bottlenecks toward >=30 tok/s; measure whole inference.
- [x] Bind a correctly quantized DFlash companion retaining BF16 codebooks/private state; qualify
  greedy-token/state correctness and optimize K4/K5 toward >=60 output tok/s at C1.
- [x] Implement a well-motivated cooperative mixed-A4 prefill challenger, qualify against the
  public-input oracle and retained quality samples, and promote only whole-prefill wins over A8.
- [x] Evaluate remaining clearly evidenced material wins, record achieved rates and limitations,
  and commit coherent implementation/results at natural milestones.
Preserve the immutable 5090 reference and selected near-NVFP4 quality; distinguish the <=2%
per-text PPL comparison from the separate BF16-source admission gate and report severe positions.
Use matched commands, prompt/context, graph mode, auto power, and C<=4. Do not silently substitute
another model, speculative throughput for ordinary decode, or profiler timing for admission.
Unrelated XAttention and broad capacity campaigns remain paused.
Evidence and exact commands: `profiles/bench/r9700-compact-decode-20260923/`;
completed results belong in `docs/performance.md`, not a growing experiment diary here.
- Base-only C1/P4096/G128: 20.2346 baseline → 30.1973/30.1730 tok/s in balanced final A/B;
  controls29.8817/29.8317. Local Q4 fusions, ordered PV staging and down-only two-group raw
  loads preserve every retained token. Full FP64, explicit BF16 seams and graph checks pass.
- New17.0025GB DFlash companion preserves all base payloads and BF16 codebooks; actual1190-object
  binding passes. C1 code4K K4/K5=89.6352/104.8799 tok/s; shortchat=75.6452/69.5402, with
  ordinary30.8765. Every speculative repetition matches ordinary tokens exactly. Draft-length
  preference is workload-dependent. The graph-update allowance fix retains fail-closed checks.
- Mixed A4 ping/pong prefill=1489–1498 tok/s versus uniformA8's1431–1446; all six mixed PPL
  sidecars unchanged, public FP64 passes, original A8 ISA preserved. Fast kernel is selected
  within the mixed evaluator; uniformA8 remains default because mixed has10 versus6 new severe
  technical positions. This is not final BF16-source production admission.
- Rejected and removed: PV stage64 spills, explicit page pointers regress, DFlash row-sharing
  gives no material consistent whole win. Do not repeat prior losing T1 CTA or gate/up-prefetch
  sweeps. Context3's separate FP8-query oracle failure is retained, not waived; changed PV passes.
- Depth4 is rejected:0.1264000ms versus depth2's0.1277205, below material whole saving bound.
- Final linked4K confirmation and balanced comparison show29.45–30.20tok/s across both
  binaries with identical ordinary-kernel machine code and exact tokens. No consistent build
  regression, but no guaranteed30tok/s floor on every4K run. Retain all slow cells.
- Final linked shortchat ordinary/K4/K5=30.7130/75.5710/70.7222tok/s, all repetitions exact;
  both C1 targets achieved on the delivered build. CLI/serve/PPL/bench builds and focused
  oracle/static/graph checks pass. No remaining material winner in this bounded investigation;
  code/results are committed with this ledger. Prior checkpoints:be19fb30,1e83910f,807fc291,
  23bc2389. No universal ceiling or C2–4 speed claim; unrelated paused work stays paused.

COMPLETED USER REQUEST (2026-09-23): identify a smaller Q4/FP8 recipe close to the retained
5090 NVFP4 PPL, then establish where A4 can replace A8 without unacceptable quality loss.
This reopens only bounded recipe/activation selection; unrelated XAttention/capacity work stays
paused. Cap large promoted base projections at FP8; preserve specified direct norms, controls,
and persistent state. Do not claim faster execution until whole-inference measurement supports it.
- [x] Register/convert four fixed candidates: early-attention FP8, all-attention FP8,
  attention+GDN-QK FP8, and selective's26 protected projections capped at FP8 with Q4 endpoints.
- [x] Compare each against immutable matched NVFP4 inputs/NLLs; prefer at most2% per-text PPL
  regression in both schedules, report5% sensitivity and newly severe positions separately.
  Use existing all-Q4/A8 and four-role controls; do not rerun unchanged references.
- [x] Measure eligible candidates at matched C1/P4096/G128/chunk2048, select a concrete base
  recipe and save under local_llm/models. If needed, refine only a material observed tradeoff.
- [x] Evaluate a bounded mixed-A4/A8 precision choice for the selected base; retain A8 where
  quality or actual speed rejects A4. Record resulting recipe, supported routes, and limitations.
Immutable 5090 reference: `tools/ppl/fixtures/nvfp4-5090-20260922/`, committed `0b00efa1`.
All four artifacts pass payload/binder/FP8 state/planner checks (`03240dd5`). Thirty quality
cells and twelve speed cells complete. Selected selective-cap15.793 GB meets2% on every
text/schedule (worst prefill+0.13%, decode+1.81%) at1431–1446 prefill/20.18–20.20 ordinary
decode tok/s. Saved and revalidated under
`/ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-q4-fp8-selective-cap/`.
Mixed prefill gate/up A4 (`0a439704`) also meets2% PPL, but measures900–901 prefill tok/s and
adds10 versus6 new severe technical-prefill positions. Keep uniform A8 default. Public-input
FP64/codec, finite/poison/arena and both-build metadata/selector checks pass; installed artifact
and current default build reproduce frozen code NLLs exactly. No DFlash companion attached.
Results/commands: `profiles/ppl/r9700-fp8-capped-selection-20260923/selection.json` and
`docs/performance.md`. A tuned cooperative A4 gate/up kernel is a future candidate, not a
completed speedup; unrelated kernel/DFlash/XAttention/capacity work remains outside this task.

COMPLETED USER REQUEST (2026-09-23): compare multi-text PPL against the standard 5090 NVFP4
artifact using `ninfer-dylan2`, and measure AMD recipe/activation prefill and ordinary-decode
speed to expose the measured speed/quality frontier. This bounded campaign supersedes the pause
only for these comparisons; it does not reopen XAttention, capacity, or kernel optimization.
- [x] Prepare three matched text/token samples and verify scoring/tokenizer alignment across GPUs.
- [x] Measure all-Q4 and source-MSE mixed Q4/W8, each with Q4 A4/A8 (W8 stays A8), plus
  selective-protected A8 and four-role A8 against the specified NVFP4 reference.
- [x] Include the currently deployed selective-protected tiled-head companion as a separate
  ordinary-mode row; the retained base selective artifact has the older RowSplit W8 head.
- [x] Measure AMD C1 fresh-prompt prefill and ordinary decode with chunk2048, fixed cache,
  warm1/r3; report actual routes, per-text PPL/NLL deltas, speed and measured Pareto frontier.
Keep all model bytes fixed and no implicit download. Record A4 evaluator route corrections if
needed so reported activation width matches execution. Snapshot the NVIDIA scorer without
changing its active checkout; preserve other agents' work. Similarity is reported as paired
NLL/PPL ratios and severe-position differences, not an unsupported equivalence assertion.
All42 AMD and6 NVIDIA quality cells are finite/aligned; all21 AMD speed cells pass three-repeat
token checks. A4 selector correction/public dispatch qualification: `3540705b`; ordinary graph
profile-update budget correction and host C1..4 checks: `44013616`. All-Q4/A8 is15.17 GB,
1437–1440 prefill/21.90–21.92 ordinary decode tok/s at P4096/G128, within5% NVFP4 PPL on every
text in both schedules. Four-role is1625–1628/15.12 tok/s and within2%; current tiled selective
is928–939/19.41–19.42 tok/s. These finite PPL screens are not production or DFlash admission.
Full tables/limitations: `docs/performance.md`; exact commands, retained failures and selected
retry receipts: `profiles/ppl/r9700-nvfp4-multitext-pareto-20260922/` (`comparison.md/json`).
No recipe was promoted; broader work remains paused outside this completed bounded comparison.

COMPLETED USER REQUEST (2026-09-22): qualify DFlash K4/K5 at C2–4 and explain C2 scaling.
The selected artifact and head math are unchanged; target verification remains genuinely batched.
- [x] Localize and correct width-dependent arithmetic and accepted-token replay state.
- [x] Qualify C1–4 graph and C2–4 eager against fresh same-C ordinary output and measure rates.
- [x] Attribute the original C1/C2 scaling and record closure.

Four corrections retain ordinary arithmetic across small-token batches: K5120 RMSNorm T1–24,
the two protected BF16 Linear shapes T1–24, projected GDN controls T1–24, and replay-fold
key normalization/dot reduction matching ordinary snapshot state. Independent FP64 Op gates,
same-input cross-width checks and replay-state regression pass. Obsolete BF16 staging and
paired-wave32 control implementations are removed. Head replay ruled out a head-only cause.

Final clean-build matrix passes all21 cells: C1–4 ordinary/K4/K5 graph warm1/r3 and C2–4
ordinary/K4/K5 eager warm0/r1, P89/G128 under auto. Speculation matches same-C ordinary
tokens exactly; repeats and eager/graph tokens/accounting are exact. C4 ordinary changed from
its historical width-dependent stream; shared ordinary lanes now match across all four C values.
Median aggregate ordinary/K4/K5 tok/s:
C1 27.51/58.75/57.29; C2 34.04/69.31/64.42; C3 45.34/87.94/77.15; C4 57.67/97.99/86.73.
Correctness costs about8% C1 speculative speed against the earlier measurement; C2 K4 is
now18% faster in aggregate than C1. These are finite-workload checks, not universal parity,
an optimization ceiling, or broad recipe admission.

Evidence and exact commands are in
`profiles/bench/r9700-dflash-concurrent-correctness-20260922/`:
`run_final.py`, `summarize_final.py`, `final-summary.json`, per-cell reports and retained
localization/FP64 logs. Scripts use create-only output directories; do not rerun into this package.
The historical C1/C2 trace explains the old scaling: target-layer cost87.180 vs38.585ms/graph,
including tiny BF16 controls15.660 vs0.761ms. Profiling is attribution-only, not current timing:
`profiles/rocprof/r9700-dflash-c1-c2-k4-20260922/analysis.md`.
Current rates and limitations are in `docs/performance.md`. Broader recipe, prefill and
XAttention campaigns remain paused; this closure does not reopen performance optimization.

COMPLETED USER MEASUREMENT REQUEST (2026-09-22): test the selected build's memory-bandwidth
headroom, then measure C2–4 speedups against a matched C1 reference. This reopens measurement,
not kernel optimization, model conversion, or broad recipe/prefill/XAttention campaigns.
- [x] Measure sustained device-memory bandwidth under `auto`; relate source-accounted active
  weight traffic and useful tokens per round to current ordinary/K4/K5 decode. Distinguish
  modeled traffic from physical counters; do not claim a proven tok/s ceiling from a stream probe.
- [x] Measure matched C1–4 ordinary/K4/K5 on the selected artifact, reporting aggregate and
  per-request rates, acceptance, same-workload speedups and numerical/token limitations.
- [x] Record results and whether bandwidth saturation or a practical ceiling is actually supported.
Fresh read-stream median636GB/s does not establish inference saturation. Conditional C1
weight-stream references41.08/125.53/128.57tok/s are not achievable-speed predictions.
C1 ordinary/K4/K5=27.53/63.76/62.15; C4 aggregate=56.82/108.92/98.77tok/s.
C2–4 DFlash fails same-C ordinary exact-token parity on some lanes: timing observations only,
not production admission. Benchmark rotates added-lane prompts; scaling ratios include that mix.
Results, acceptance, commands and limitations are recorded in `docs/performance.md`, backed by
`profiles/bench/r9700-decode-bandwidth-concurrency-20260922/`.
This measurement request is complete; no kernel work or paused campaign was reopened.

DFLASH OPTIMIZATION PASS COMPLETE (2026-09-22): final admitted C1 chat decode is
K4 **64.81318 tok/s**, K5 **62.92418 tok/s**; whole-output57.02881/55.64922tok/s.
All16 matched runs preserve exact tokens/accounting; every balanced chat pair and raw regression
gate passes. Ordinary T1 is unchanged; latest measured C1 ordinary rate remains27.46441tok/s.
Final authority: `profiles/bench/r9700-bf16-controls-wave32-20260922/whole-inference/summary.json`.
No currently evidenced material mechanism remains untested in this fixed-artifact C1 pass.
This is not global optimality, memory-bandwidth saturation or broad context/concurrency admission.
Earlier rates below are retained history, not current work directives. Unrelated prefill/XAttention
and broad recipe campaigns remain explicitly paused; do not resume them automatically.
Final selected numerical-only T1/T5/T6 qualifier, existing GDN regression, build and diff checks
pass after removal of temporary comparisons and duplicate oracle ownership.

- [x] Reattribute the packed-W8 build on the measured chat K5 workload; rank remaining owners.
  `profiles/rocprof/r9700-dflash-packed-w8-chat-k5-20260922/attribution.json`: exact129tokens,
  37graphs/91accepted/zero fallback. Profiled pergraph: targetlayers68.491ms, head7.600ms,
  draft5.170ms. Leading owners: gate/up13.509ms, normalization12.340ms, down9.742ms,
  protectedBF16projections9.327ms. These durations are attribution only.
- [x] Bound and evaluate remaining W8 scale/operand scheduling, Q4 verify, normalization/fusion,
  protected BF16 projection, draft and round-service mechanisms; reuse prior rejections unless
  a genuinely new mechanism changes the bound. Maintain useful independent implementation/review.
- [x] Implement and independently qualify credible challengers, retain exact public greedy parity,
  and admit only matched unprofiled whole-inference wins. Repeat attribution when it changes
  target selection; commit at natural checkpoints.
- [x] Close only when no credible material mechanism remains untested or an external prerequisite
  genuinely blocks progress; document the practical remaining bound and actual achieved rates.

Exact-order normalization block-ahead is rejected at the whole-gain screen:
numerical/ISA checks pass but129-call savings0.382/0.481ms are below2%current round at T5/T6.
Evidence and retained executable are in `profiles/bench/r9700-rmsnorm-block-ahead-20260922/`;
temporary code is removed, and no whole rerun is justified.
Integrated public qualification and linked ISA checks pass for gate/up and the narrowed bundle:
N5120/K6144 and N4096/K5120 pipeline at T5/T6; N12288/K5120 retains scale-gather.
Selected-public screens save3.710/3.391ms for gate/up and1.569/1.666ms for projections.
The extracted down-body public regression also passes. The immutable Q4-only benchmark is
`profiles/bench/r9700-verify-pipeline-family-20260922/whole-inference/candidate-ninfer_bench`;
whole admission PASS: chat K4/K5=63.02158/61.24468 decode tok/s versus
57.42168/55.93994 matched control, +9.75%/+9.48%; all16streams/accounting exact.
Raw single-pair checks30.58844/29.02133tok/s also pass. Evidence is adjacent `summary.json`.
These are bundle results, not isolated operator gains, and exclude subsequent BF16 changes.
Selected public qualifiers are consolidated and temporary comparison controls removed;
the numerical-only four-shape qualifier is `ninfer_r9700_a8q4_verify_projection_qual`.
The BF16 follow-up tested the complete paired projected-control Op: one joint split-K16
projection kernel plus merge/gating, preserving both explicit BF16 projection seams. Its
independent full-formula qualifier compares against two incumbent projections plus gating,
weighted48completecalls rather than96individual projections. Complete-Op numerical, own-graph,
guard and native-BF16 ISA checks pass; all warm/cold/alignment cells clear the material screen,
estimating1.848–2.073ms/round. Evidence:
`profiles/bench/r9700-bf16-projected-control-splitk-20260922/screen/report.json`.
Public selected qualification and planner/T1 regression passed, but real-model chat K4 failed
exact greedy parity at output index52 (413 versus3470), changing76/129positions and round
accounting. The campaign stopped immediately; no speed is admitted. Retained decision:
`profiles/bench/r9700-bf16-projected-control-splitk-20260922/whole-inference/rejection.json`.
Admitted T1-only controls are restored and rejected implementation/targets deleted; never
rerun this unchanged reduction profile or waive the exact-token gate.

Gate/up two-group lookahead is rejected at the native-code gate. Rotating payload copies
force early waits at76/77VGPR; the one bounded fixed-slot repair still hoists future scale
conversion across current compute, forcing waits at101/107VGPR. Neither implements the
intended two-generation overlap, so no GPU campaign is justified. Evidence:
`profiles/bench/r9700-gate-up-depth2-20260922/README.md`; temporary implementation is removed.

The final admitted mechanism is exact-order wave32-per-(head,token) paired BF16 controls,
preserving the original160FMA lane chains, shuffle reduction and BF16 seams. It attacks
wave8's repeated CTA barriers and inactive token waves, unlike rejected WMMA reassociation.
Pure launch savings are insufficient: require48completecalls to save>2%of53.4487/56.4859ms
rounds, independent oracle/exact control/graph guards, then exact real-model whole admission.
The refreshed admitted Q4 trace (`profiles/rocprof/r9700-dflash-verify-pipeline-chat-k5-20260922/attribution.json`)
confirms2.081ms of control projections and~0.079ms gating; target layers42.598ms, gate/up10.524ms,
down5.741ms and normalization6.147ms are attribution only, not unprofiled performance claims.
The scalar wave32 candidate passes exact outputs, independent oracle and graph/guard checks,
but three cold-cache cells miss the material gate (0.970–1.030ms savings versus1.069/1.130ms
thresholds). Warm cells pass; mixed evidence does not admit the route. Retained first screen:
`profiles/bench/r9700-bf16-controls-wave32-20260922/screen.json`.
The bounded four-step load refinement passes fresh static and numerical/timing gates:
25VGPR/19SGPR, no LDS/private/spills, actual blocked-load overlap and all160ordered FMAs.
All warm/cold/alignment cells preserve exact outputs and save1.260–1.502ms/round, clearing
the unchanged material gate. Evidence: same directory `four-step-screen.json`.
T5/T6 are admitted in the owning projected-controls Op, preserving T1 and reusing the existing
GDN module without scratch. Actual-public qualification, existing GDN regression and all16
same-artifact whole runs pass. K4/K5 reach64.81318/62.92418tok/s; mean paired gains3.83%/3.71%.
The last control pair is slower, so report variability: K4 pairs+2.88..5.72%, K5+2.67..5.54%;
even the first two pairs exceed2%. Raw one-pair checks31.45723/29.82590tok/s pass.
Beyond this controls family, the independent
frontier review finds no currently justified untested material mechanism; do not invent a new
campaign from theoretical headroom alone. Any closure is limited to this fixed artifact and
measured C1 chat K4/K5 workloads, not a global or multi-context/concurrency optimum.

Live mechanism evidence: exact-order K5120 RMSNorm token8 at T5/T6 is admitted by balanced
three-pair chat whole-inference A/B, retaining exact token IDs and speculative accounting.
K4 decode improves 38.02294 → 41.27905 tok/s; K5 improves 38.54031 → 42.17567 tok/s,
with mean paired gains of 8.57% and 9.48%. Raw-text one-pair checks are regression evidence only.
See `profiles/bench/r9700-dflash-rmsnorm-exact-20260922/summary.json`.
All three W8 scale-gather variants are rejected; the generic half-wave screen and sustained
fixed-T full-wave shuffle/readlane screens lose despite exact outputs. Production source is
restored to the packed-load baseline; evidence and timing-protocol distinctions remain in
`profiles/bench/r9700-dflash-scale-gather-20260922/README.md`.
The exact RMSNorm one-CTA-per-row mapping is also rejected: approximately 0.070 ms versus
token8's 0.050 ms, with exact outputs and independent FP64 checks retained in
`profiles/bench/r9700-dflash-rmsnorm-exact-20260922/row-cta-screen.json`.
Q4 gate/up scale gathering is rejected at whole-inference scope: mean chat K4/K5 paired gains
1.29%/1.20%, each with a negative pair; the selector is removed. Retain
`profiles/bench/r9700-dflash-gate-up-scale-gather-20260922/planner-retry/summary.json`.
Packed normalization is admitted only at K5120 rows5: chat K4 41.14269 → 42.60744 tok/s,
3.56% mean paired gain, all three pairs winning. Rows6's K5 gain of 1.56% with one losing pair
is not admitted, so rows6 stay scalar token8. Preserve the original combined `NO_WIN` summary
at `profiles/bench/r9700-dflash-rmsnorm-packed-20260922/summary.json`; raw checks are regression-only.
Protected BF16 staging's original diagnostic passes exact outputs/public FP64 and shows about
3x operator gains for both qualified shapes/T5/T6, including two-byte-offset buffers; retain
`profiles/bench/r9700-bf16-staging-20260922/screen/report.json`. Actual selected generic-dispatch
qualification now passes. The combined rows5-packed-plus-BF16 whole A/B is admitted:
chat K4 40.78243 → 45.45008 tok/s and K5 42.15744 → 45.65818 tok/s, all three
pairs winning for each width with exact tokens and speculative accounting. Raw one-pair checks
also improve but remain regression evidence. See the same package's `summary.json`.
The unchecked frontier above remains active; these results do not establish exhaustion.

Selected lossless W8-head bundle: storage/binder and all ordinary/prefill consumers are complete.
The explicit artifact is
`qwen3.8-27b-r9700-q4-selective-protected-n16k16-dflash2-q4-head-n16k16-eval.ninfer`
in `/ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-q4-selective-protected-dflash2-q4/`.
Its receipt verifies 1,189 unchanged payloads and inverse-byte-exact head codes/scales, with the
same size/recipe identity and no duplicate head, runtime packing or requantization. The original
artifact and BF16-staging control binary remain unchanged. Real binding/old-layout rejection,
public-consumer exact/FP64, graph and guard checks pass.
The initial tiled ordinary route failed C3 (-3.89%); the qualified coalesced T1–3 replacement
preserves the exact 256-chain reduction and removes that regression. The selected bundle also
includes T5/T6 Q4 projection gathering at N5120/K6144, N12288/K5120 and N4096/K5120.
`profiles/bench/r9700-w8-tiled-head-20260922/whole-inference-combined/summary.json` is PASS:
24 exact-token/accounting runs; three balanced chat pairs each give K4 45.89039 → 53.56033
tok/s (+16.72%), K5 45.68180 → 52.43020 (+14.77%) and ordinary C1 24.87909 → 27.46441
(+10.39%). The single C3 ordinary pair improves aggregate 38.56344 → 45.08326 (+16.91%).
Raw K4/K5 single regression pairs improve to 26.08567/24.87419 tok/s; no ordinary/raw decode
or whole rate breaches the 2% regression bound. These are combined, not isolated-projection gains.
Preserve the failed `whole-inference` and `whole-inference-retry` packages. C2/K4 and C4/K5
correctness/startup smoke passes exact lane tokens, accounting and graph allocation bounds;
`concurrency-smoke/summary.json` is not a throughput admission. Its first control was reused
after correcting the harness's per-lane versus aggregate decode-count assumption.
Fresh attribution in `profiles/rocprof/r9700-dflash-tiled-projections-chat-k5-20260922/attribution.json`
retains exact tokens/accounting: target layers51.679ms, head2.238ms, draft4.688ms per graph.
Gate/up13.509ms, down9.740ms and scalar normalization6.141ms remain leading owners (profiled,
not performance-admission durations). The down-only next-group pipeline is now admitted for
N5120/K17408 T5/T6 after public-Op exact/FP64/graph/guard checks and all 16 matched whole runs.
Three balanced chat pairs each improve K4 53.70092 → 57.31582 tok/s (+6.73%) and K5
52.48043 → 55.87248 (+6.46%); whole-output rates are 51.08892/49.88149 tok/s. Raw single
regression pairs reach 27.86485/26.48399 tok/s. Same artifact, exact streams/accounting/config;
ordinary is not repeated because its route is unchanged. Retain
`profiles/bench/r9700-dflash-down-pipeline-20260922/whole-inference/summary.json`.
The earlier C2/C4 smoke predates this pipeline. No 60 tok/s or terminal quality claim follows.
Draft-side audit finds no independent roughly 1.3 ms/round mechanism: append totals 0.908 ms
(dead Q-row elimination ceiling 0.262 ms), proposal head+selector 1.044 ms, already split SWA
attention 0.139 ms and device round service including folding 0.672 ms. These are pre-pipeline
profiled bounds, not speed measurements or exclusion of host overhead. Normalization block-ahead
remains unselected; this historical frontier is superseded by the completion above.

COMPLETED DFLASH SPEED CHECKPOINT (2026-09-22): established matched baselines, attributed
verification, and qualified the earlier packed-load improvement. That checkpoint measured
39.00 tok/s for chat K5, superseded by the normalization result above;
The later completed checkpoint above establishes60+ for the stated C1 chat workload. Other prefill, XAttention and broad
recipe campaigns remain explicitly paused. Main owns serialized GPU runs; independent agents
review sources.

- [x] Measure current ordinary/K4/K5 on identical target weights, prompt, greedy sampling,
  context/cache and graph configuration; retain tokens and acceptance. Use the new
  selective-protected base: add its missing DFlash companion without changing base payloads,
  then measure all modes on that same combined artifact. The four-role screen at
  `profiles/bench/r9700-dflash-post-port-baseline-20260922/` was stopped after the user clarified
  the intended model; retain partial evidence, but do not use it as this goal's primary baseline.
  The selective companion is now created in
  `/ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-q4-selective-protected-dflash2-q4/`:
  18,887,772,672 bytes; all1124 base and66 canonical-Q4/BF16 companion objects verified byte-exact.
  Independent source review,14 Python tests, C++ registry/workspace tests and real graph execution
  pass. C1/P128+G64/context1024 screen at
  `profiles/bench/r9700-dflash-selective-baseline-20260922/summary.json` retains exact65tokens
  across all9 measured runs: ordinary24.78694,K4 17.51063,K5 17.26805 decode-output tok/s.
  Both speculative widths emit1.65789 tokens/round; service-inclusive decode/round is96.18/97.53ms.
  This raw-text corpus is not chat-templated; add one representative matched chat case before
  assigning low acceptance to quantization. Fixed-order screen is not challenger admission.
  Chat check now complete at `profiles/bench/r9700-dflash-selective-chat-20260922/summary.json`:
  P89+G128/context1024, exact129tokens across all9runs. Ordinary24.77555,K4 35.89543,K5 35.97194
  decode-output tok/s; acceptance lengths3.36842/3.45946. Raw-text poor acceptance does not alone
  establish a quantization problem. Combined artifact/support checkpoint: `f1160deb`.
- [x] Attribute draft, verify and other round costs on the measured workload; distinguish low
  accepted/output tokens per round from slow verification. Graph markers alone are insufficient.
  Source-mapped graph trace `profiles/rocprof/r9700-dflash-selective-k5-20260922/attribution.json`
  matches tokens/accounting and covers39graphs (38rounds plus one full-graph zero-extent fallback).
  Profiled kernel means: target layers68.616ms, target head17.394ms, draft layers5.155ms,
  append0.953ms, proposal head0.686ms, selector0.423ms. Intercepted durations are attribution only.
- [x] Evaluate base-decode mechanisms at verify T5/T6. T1-only predicates must not be widened
  blindly. Initial source audit ranks exact-order gate/up scale gather ahead of normalization/
  codec and residual fusions, conditional on current attribution; retain rejected split-K and
  small-T remap evidence rather than repeating them.
  Fresh attribution supersedes that initial ranking: W8 full target head is the largest single
  kernel owner. Actual linked ISA contains32 scalar byte-load sites and41 load waits per G32,
  versus two IU8 WMMA instructions. First challenger packs the same eight operand bytes into two
  dword loads using the existing4-byte alignment contract; no weight/codec/math-order change,
  no scale-gather bundle. Preserve old linked control under
  `profiles/bench/r9700-w8-packed-loads-20260922/control/`; qualify tails/4-not8 alignment,
  public numerical oracle, graph replay and exact output bits before matched whole A/B.
- [x] Implement and qualify the strongest supported mechanism, address acceptance if limiting,
  and verify a material matched whole-inference gain with exact greedy-token parity. Record
  actual rates and remaining bottlenecks; commit at natural checkpoints.
  Selected packed W8 operand loads:32byte sites→4B64,41load waits→11,83VGPR→61, same2IU8WMMA,
  no scratch/LDS or arithmetic changes. Ordered control/candidate FP64, exactcodec, tails,
  4-mod8 alignment and poisonedgraph tests pass; all6complete output tensors match exactly.
  Retained initial fixture failures were a default/nonblocking-stream initialization race,
  repaired without changing tolerances. Balanced24run wholeA/B passes all12paired comparisons
  and exacttokens/accounting: rawK4 17.50045→18.52400, rawK5 17.24277→18.38723,
  chatK4 35.02727→37.96226, chatK5 36.35116→39.00057 decode tok/s. ChatK5 wholeoutput36.06263.
  Ordinary chat regression preserves all129tokens (one repetition24.36863 tok/s; baseline3rep
  mean24.77555). CLI/server/PPL relinked to selected core. Evidence/commands:
  `profiles/bench/r9700-w8-packed-loads-20260922/whole-summary.json` and adjacent runners;
  stable result and limitations in `docs/performance.md`.

For a subsequent DFlash speed task, reattribute the selected build before choosing another
mechanism. Pre-change target layers were68.616ms/graph, versus5.155ms draft; Q4 gate/up,
normalization, Q4 down and protected BF16 projections remain candidates, not measured new wins.
Do not repeat rejected split-K/remap experiments without a new bound/mechanism. Raw-text
acceptance is independently limiting; unchanged outputs/accounting show this load optimization
improves round execution, not acceptance. C2–4 performance and60+tok/s remain unclaimed.

COMPLETED FULL FEATURE PARITY (user-authorized 2026-09-21): port all applicable non-kernel
features from upstream experimental `e04fad37`, including p-less/epsilon sampling and its
1/1024 probability floor, tool constraints/reasoning recovery, scheduling/host overhead,
durable SSD cache and startup improvements. Implement necessary HIP sampling semantics;
evaluate upstream DFlash/GDN/attention kernel changes individually for porting. Preserve the
sole Qwen3.8-27B/gfx1201 target, fixed FP8-K/INT4-V/FP16-scale growing cache, BF16 DFlash
selector/private state, C<=4 and model artifacts. This explicitly authorizes the feature changes
previously awaiting a product decision. Do not restore CUDA or additional model targets.

- [x] Sampling/API: p-less, epsilon/floor, suppression, masks/cycle exclusion, ordinary and speculative correctness.
- [x] Host runtime: reasoning/tool generation and recovery, chain/adaptive DFlash, vision integration, scheduling/telemetry.
- [x] Storage/startup: durable SSD cache, RAM recovery/ownership, parallel loading and pinned initialization.
- [x] Product/tooling: CLI/server contracts, xgrammar, Docker/build/test/evaluation tools and active documentation.
- [x] Kernel delta assessment: DFlash/GDN/attention/projection mechanisms, equivalence/port/rejection evidence.
- [x] Integrated qualification and coherent commits/push; close all applicable feature gaps.

Kernel assessment for `c450798c..e04fad37` (source review, not a performance claim):

Integration verification checkpoint (2026-09-21, worktree
`ninfer-amd-r9700-upstream-integration-20260921`, build
`build-r9700-upstream-integration`): final full build and graph-allowance host tests passed.
R9700/auto physical sampling, speculative-round, selector and RoPE independent oracle qualifiers
pass; fixed RAM/SSD exact-byte/raw/zstd/restart/Vision-identity/CRC test passes. Real C1 ordinary
and DFlash K4 graph tool requests return the same schema-valid echo call. Positive-temperature
p-less adaptive DFlash and image-chart Vision+DFlash K5 both complete. RoPE required an exact
occupied-range alias check for interleaved Vision Q/K; FP64 output and untouched V are verified.
C2 DFlash graph startup/tool isolation/disconnect/streaming now passes after distinguishing
the W5/W6 attention topology change at 8192 visible keys. Short final-assistant PPL now scores
four finite targets from the actual rendered token boundary. Public SSD restart restores
38 tokens and matches all16 cold greedy output tokens; recovered statistics publish at startup.
Native Vision traces capture471 boundaries for one image and942 for two, with exact aggregate
versus individual outputs. Focused host suite17/17 and benchmark Python tests10/10 pass.
Selected-Vision planner/validator now closes over the actual native tracer and requires all471
captures in each profile, exact identities, finite metrics and replayed shared numerical gates;
17 Python tests plus3 corruption subtests pass with independent source review.
Per-sequence DFlash selector projection passes independent FP64 and exact C1/packed checks,
including K4/C4 and K5/C4 graph replay. Preserved MTP K3 now completes the same schema-valid
echo call: measured58MiB fits the corrected62MiB graph allowance. The planner counts actual
instantiate/update/restore operations; the strict observed-allocation guard remains enabled.
Docker tooling is source/syntax reviewed, not image-build/runtime qualified. Kernel alternatives
below are assessed, not claimed as R9700 performance improvements.
Native-context MTP C4/K5 startup passes with148MiB observed against190MiB planned. Fresh
DFlash C2 tool isolation, disconnect-survivor and subsequent streaming pass after the selector
port. Evidence is retained under the integration worktree's
`profiles/bench/upstream-parity-20260921/` and `upstream-parity-20260921-tool-isolation-final/`.
The feature implementation is checkpointed in `2eab0a50`; the following qualification closure
commit includes the final host regression and these results. Earlier optimization/recipe
campaigns remain paused by this feature-parity deliverable; do not launch them as parity work.

- **Ported for feature correctness:** full-domain p-less ordinary/speculative sampling,
  configured argmax eligibility, root-only cycle exclusion, DFlash p-less one-hot proposal,
  and per-row position offsets. HIP uses wave32/hipCUB and device-scope release/acquire
  completion counters with caller-owned shared-layout workspace. Independent CPU-only review
  admits these sources to the sampling/speculative/path-select physical oracle qualifiers;
  compile success alone does not qualify numerical results or speed.
- **DFlash selector parallel merge: applicable optimization, retain incumbent for this port.**
  Upstream `src/ops/kernel/dflash2_path_select.cuh` replaces thread-zero insertion of 512
  split candidates with sixteen block-wide rank reductions. AMD
  `src/ops/r9700/dflash/dflash2_path_select_kernels.h` still has the serial exact merge;
  it is not already equivalent in performance. Both preserve value-descending/id-ascending
  ordering, so no new public behavior requires replacement. The challenger adds sixteen
  block reductions/barrier chains; NVIDIA improvement does not establish R9700 benefit.
  No promotion or speed claim is made without matched physical selector and whole-DFlash
  evidence. P-less greedy proposal semantics were ported independently of this optimization.
- **DFlash selector per-sequence projection: applicable numerical isolation, ported.**
  Both chain and tree wrappers now project one sequence's T columns at a time, reusing
  caller-owned scratch. Flattening T*C could select a different BF16/W8 reduction route as
  other requests entered the batch. Projection workspace is sized for T, while selector
  scratch remains sized for T*C. The selector qualifier compares packed outputs exactly with
  separate C1 calls (IDs, proposal probabilities and every tree state output), alongside the
  independent FP64 oracle; companion K4/K5 C4 runs cover the width-dependent route boundaries.
- **GDN replay/state: supported behavior already covered by the AMD route.** Upstream
  recurrent `OverlayAccess` and replay changes use transient state overlays plus raw
  key/value/gate records. AMD `gdn_recurrence/gated_delta_net.hip` and
  `gdn_replay_fold.hip` already record raw BF16 K/V and FP32 gates and fold the committed
  prefix into FP32 persistent state. Program retains its typed KV rollback/publication and
  selected-path fold. Copying CUDA state addressing would replace this qualified ownership
  design without adding a required host feature.
- **GDN control/projection: retain AMD qualified leaves.** Upstream specializes packed BF16
  control GEMV for T5/T6 and extends packed T to20, and adds FP8/projected-convolution paths.
  AMD owns BF16 control in `gdn/gdn_ops.hip` and the package's
  `Variant::gdn_norm_control_projection`; its integer/FP8 projection contracts and workspace
  differ. Adaptive W4/W5/W6 uses these existing leaves, not NVIDIA dispatch. Fusing away
  convolution scratch is not valid without replacing and qualifying the AMD leaf, so its
  `GdnPrefillConvRoots.convolved` storage is deliberately retained.
- **Chunked GDN precision: not a mechanical port.** Upstream adds FP16 W/U staging alongside
  BF16 using `mma_f16`, `ldmatrix` and CUDA shared-memory layouts. AMD has independent
  recurrence/affine-chunk implementations and numerical qualification. Importing the CUDA
  intermediate precision would change the R9700 arithmetic profile without evidence;
  host-feature parity neither requires that profile nor proves it faster or more accurate.
- **Attention/projection: target-specific implementations retained.** Upstream sparse
  prefill/rank-sort changes operate on its NVFP4/U8 cache route; AMD's only growing cache is
  typed FP8-K/INT4-V/FP16-scale, consumed by `kv/fp8_int4_kv_attention.hip` and its separate
  XAttention route. Upstream's minimum-length dispatch adjustment is not the AMD dense
  product predicate. New FP8/NVFP4 linear/add/SwiGLU routes do not replace the selected
  Q4/W8/FP8 AMD leaf implementations or BF16 selector codebooks. These are implementation
  alternatives, not missing host semantics; no CUDA kernel or new weight/cache format was
  admitted on an unsupported equivalence assumption.

HISTORICAL BOUNDED UPSTREAM INTEGRATION (superseded by full parity above): upstream `experimental` fetched at `e04fad37`; AMD
checkpoint `41e272ee` was preserved while integration was verified on
`integration/upstream-experimental-20260921` in the sibling `ninfer-amd-r9700-upstream-integration-20260921`
worktree. This is selective porting, not a wholesale rebase or a claim of full upstream parity.
Current batch preserves existing product behavior: prelaunch cancellation, speculative tail
validity/committed statistics, resident/RAM reuse readiness, host-image DMA lifetime and pinned
allocator exception safety. Latest upstream semantics take precedence over reverted intermediate
fixes. New p-less defaults, constrained-tool/reasoning recovery and SSD-cache features require a
separate product decision; CUDA kernel optimizations and startup parallelization are not part of
this correctness batch. Existing performance campaigns remain paused.

Source provenance: `c075cc38` prelaunch filtering with `64840a95`'s corrected ordinary
postlaunch semantics (clear, do not fabricate rollback); `0412f177` speculative tail invalidation;
`b2b1b7e4` committed-prefix statistics; `64840a95` RAM readiness/ownership fixes adapted to HIP.
Independent review passed. Fresh build `build-r9700-upstream-integration` builds Engine/PPL;
host runtime/admission/statistics tests pass, and pinned-arena injected allocation failures and
allocation-free fragmented release pass. Manual `ninfer_r9700_engine_cache_cancel_qual ARTIFACT`
passed on the saved selective model: C1 greedy, cancelled after one published token, exact cold
continuation parity, confirmed RAM restore of39 tokens with advancing capture/restore counters.
This does not force a specific cancellation race or physically qualify speculative cancellation
and every checkpoint-reader interleaving. No kernel arithmetic, sampling defaults or model bytes
changed; no PPL/performance campaign was repeated. Optional cache-allocation cold-fallback
recovery and startup loading parallelization remain separate, unported work, not claims of this batch.

The exact selective artifact is also saved under
`/ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-q4-selective-protected/` with a README and original
conversion provenance. Destination SHA-256 matches the tested artifact; no requantization occurred.

COMPLETED NARROW TASK: created `r9700-q4-selective-protected-n16k16-eval`, exactly
17,678,295,040 bytes, from all-Q4 with 2 W8 vocabulary endpoints,15 BF16 protections and11
selective FP8 objects; all other payloads copied byte-exact. Source/runtime review, CPU format
checks, actual-artifact binding and new-shape FP64/eager/graph checks passed. Matched dense/G16,
chunk2048,C1,skip-half PPL at8K/32K: selective6.641352/5.804673, all-Q4 6.723230/5.891989,
four-role6.614002/5.797376. All three pass capacity-speed quality and fail strict accuracy.
Selective is17.98% smaller than four-role; no speed or production-selection claim. Exact
commands, reports, sidecars and comparison are in
`profiles/ppl/r9700-selective-protected-comparison-20260921`. This bounded user request is
complete and does not resume the paused speed/XAttention/DFlash campaigns below.

USER PAUSE 2026-09-21: wrap up current work; pause XAttention. All active work is now stopped,
including the whole-comparison queue. Unchecked tasks below are retained future work, not an
instruction to restart during this pause. No GPU or agent job should remain active. The corrected
phase-sum owner retains complete all-Q4 dense G16/G32 C1..4 and XAttention G16 C1..3. XAttention
G16 C4 was interrupted by the user pause and has no admissible completed report. See
`profiles/bench/r9700-terminal-base-phase-sum-20260921/whole/closure.json`. Do not rerun that
create-only whole stage or execute controls/select against its incomplete matrix. On explicit
resumption, reuse completed valid cells and create a reviewed missing-cell continuation; do not
repeat completed chunk, numerical quality, capacity or whole cells. No terminal base is selected.
Latest confirmed DFlash production evidence remains K4/W5 `17.40551875 tok/s` and K5/W6
`17.60527624 tok/s`, versus matched ordinary about `19.82 tok/s`, under the retained scoped
production-confirmation workload. No new selected-recipe DFlash benchmark has run.
Priority correction: the main objective is DFlash throughput. Recent base/reporting work has not
advanced that throughput. On resumption, first resolve the minimum base decision needed for a
matched DFlash experiment using retained evidence; do not automatically restart the broad base/
XAttention campaign. If that requires changing the current terminal-selection dependency or
ranking contract, surface that explicit decision rather than silently claiming a final base or
spending another long campaign on preparation. Preserve exact public-token and numerical gates.

Retained pre-pause context:

2026-09-21: bounded prefill optimization, chunk selection (2048), and all24 numerical-quality
cells are complete. Twenty-two quality cells pass; mixed XAttention G16/G32 are excluded at32K.
All48 base capacity cells now pass after the qualified FP8 ownership/accounting fix. Active owner:
`profiles/bench/r9700-terminal-base-fp8-context-recovery-20260921`; its numerical/chunk bridge
and capacity validation are closed. Real-model XAttention keep distributions also pass.
Whole timing is PAUSED after all-Q4 dense G16 C1/C2: schema20 inflates C>1 aggregate prefill
throughput by using max lane service time with all lanes' token counts. Whole wall and decode
measurements remain valid, but phase objectives must not use that prefill metric. The owned
campaign was stopped during C3; `whole/closure.json` preserves the exact scope. Reporter fix
`5eff7490` now emits schema21 serial-lane prefill-service sums; all four fresh builds, host
tests/planners and a short C2 real-model reporting check pass. The reviewed successor is
`profiles/bench/r9700-terminal-base-phase-sum-20260921`: its reporting-only bridge is frozen
and independently validated; `commands.sh whole` was stopped at the user pause above.
Controls and selection have not run. Do not rerun freeze or whole.
Keep the existing24 phase/whole objective policy. Do not rerun completed chunk, quality,
capacity, resource qualification, host checks or the timing-ineligible reporting smoke. Finish graph/eager controls,
publish base selection, and
materialize and optimize the three DFlash recipes at K4/W5 and K5/W6. Do not restart completed
chunk, quality, reference, or capacity campaigns. See terminal-selection tasks for exact authorities.

### Retained DFlash history

`DFLASH-SCHEDULE` closed 2026-09-08: the small-T packed-Q4 candidate (N34816/K5120 + MLP-down T5,
`NINFER_R9700_DFLASH_SMALL_T_CANDIDATE`) was A/B-tested on the 6fe53d53 combined companion
(C1/P129+G27, K4/W5 + K5/W6, eager, 3 reps) and found to have no material decode-speed effect
(K4/W5 candidate/control ratio 0.9993, K5/W6 ratio 0.9977, both within noise); it stays
default-off. The matched-A/B parity gate caught a K5/W6 verify-context correctness bug (emits
109600 where base decode emits 96917 at token index 5; deterministic, pre-existing), filed as
`DFLASH-K5W6-VERIFY`. Investigation (2026-09-09) found the index-5 root cause: the fused
attention leaf (the only W=6 verify route) used a full-precision BF16->FP32 query while the
ordinary/W5-WMMA routes quantize the query to FP8 E4M3FN; the fused leaf now round-trips its
query through the FP8 codec (uncommitted, `fp8_int4_kv_attention.hip:503`), and K5/W6 now matches
ordinary for 17 tokens. A residual index-17 divergence remained because the default build
(`NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE=0`, formerly `ATTENTION_PARITY_CANDIDATE`) routed
BOTH K4/W5 and K5/W6 to the fused leaf; the sealed A/B ran with `=1` (K4/W5 on the W5 WMMA leaf,
which matched ordinary). RESOLVED by option (b) (generalize the W5 batched-WMMA route to W5/W6
and promote it to a production route); see the closed `DFLASH-K5W6-VERIFY` item. K5/W6 is now
production-admissible and matches ordinary exactly on the default build.
Evidence:
`profiles/bench/r9700-dflash-small-t-whole-ab-6fe53d53-20260906` (sealed). Prior:
`DFLASH-TEXT-P129` append-versus-fresh Text parity closed (token-level exact; first intermediate
divergence at layer 13 post_mixer, one BF16 bit, no generated-token change) and the
mechanical-protocol K4/W5 directional screen closed with a material win; both are recorded under
their task items and the after-parity router.

Prior checkpoint (commits `6fe53d53` and `dce80877`): retained the default-off attention-parity
candidate and focused qualification. Dense T129 keeps rows 0..127 and overwrites row 128 with
canonical W1 arithmetic; DFlash K4/W5 uses a DFlash-only, G16, three-launch batched W5 WMMA route.
Both semantic rows, graph replay, independent oracles, routing/capacity scope, and gfx1201
resources passed. Diagnostic W5 time is `0.06228868 ms` versus `0.09549904 ms` fused. Authority:
`profiles/bench/r9700-attention-parity-candidate-focused-20260906/summary.json`. Do not rerun it.

### XAttention checkpoint

XAttention is implemented as a compile-isolated, qualification-only ordinary Text-prefill leaf;
the default build does not link it and decode, DFlash verification/proposal, MTP, and GDN remain
dense. The corrected B128/S16/tau900 ranker and Bq16 sparse consumer use the fixed typed cache and
passed G16/G32 physical FP64-oracle gates. Concentrated-fixture 8K/32K operator speedups were
28.56x/32.40x for G16 and 28.81x/32.72x for G32, but are not whole-model admission. Current status
and evidence are in `docs/maintainer/softmax-attention.md`,
`profiles/bench/r9700-xattention-b16-requal-s16-tau900-g16`, and
`profiles/bench/r9700-xattention-b16-requal-s16-tau900-g32`. Remaining work is owned by
`XATT-KEEP-DIST`, the matched quality/capacity/whole chain, `XATT-ADMISSION`, `SELECTED-NIAH`, and
the conditional scale/cutover tasks. A chunk whose first absolute query position is not B128-aligned
intentionally uses exact dense attention; never locally reanchor its estimator. No new XAttention
task is needed.

### Retained whole-model parity result

The reviewed three-load gate ran once and is retained under
`profiles/bench/r9700-attention-candidate-combined-whole-parity-gate-template-20260906/results`;
the SHA-256 of its complete `result.sha256` closure is
`3e262f85e9807ef2e72f3e76524624835293f2f358c4defe1d47b581d64fba06`.
It ended `text_parity_failed`: all three 28-token continuations match; target tokens, DFlash
decision reconstruction, and layer3 attention are exact; the Text final normalized tail differs
in 4,908/5,120 BF16 values (first index 0; maximum absolute difference 1.0). This is completed
diagnostic evidence. Never rerun it or use its synchronous traces for timing.

The create-only Text append-versus-fresh layer-boundary package it prescribed was prepared,
reviewed, and run as `profiles/bench/r9700-text-layer-boundary-traces-6fe53d53-20260906`, closing
`DFLASH-TEXT-P129`; the mechanical-protocol K4/W5 directional screen closed as well (see the
after-parity router).
The bounded DFlash verify-down split-K task is closed default-off: S=8 and S=2 both failed exact
public-token parity, and S=4 has no mechanism for restoring the incumbent serial-FMA semantics.
Do not rerun either sealed package or infer an S=4 command. The reviewed selector-free
BASE-DECODE-BW measurement completed. The BF16 GDN T1, all-Q4 attention T1, and all-Q4 C2..4
direct and whole gates all passed, and independent result audits admitted their exact production
routes. Their three qualification selectors were removed together after independent code review;
exact predicates and mixed/other-width fallbacks remain. A fresh selector-free C1..4 production
confirmation at `profiles/bench/r9700-three-route-production-confirmation-20260920` passed exact
tokens at C1..4 and reproduced the admitted performance. The first projected-residual invocation
stopped before candidate qualification because its local incumbent harness passed a null stream to
an eager Op requiring an explicit stream; preserve its `attempt-1` and never rerun that package.
The reviewed fresh retry owned one explicit stream and completed qualification, but its report
embedded raw newlines and failed strict JSON parsing after the qualifier exited 0. Preserve that
sealed diagnostic attempt. Retry2 passed strict JSON, complete numerical qualification, static
checks, and the direct timing bound. The integrated public Op now also passes production-symbol
qualification in the sealed production retry2 attempt. Preserve every prior attempt, including
the production failures caused by unstable DRM card numbering and incomplete public malformed-input
validation. The reviewed whole C1 A/B now passes, and its independent result audit admits the
exact all-Q4/A8 base Text T1 projected-residual route for promotion review. All six runs preserve
the retained 257 public tokens; candidate decode is about `29.00 tok/s`, with median paired
decode-time ratio `0.9787725`. Preserve its sealed `attempt-1`; never rerun it. Selector removal,
implementation review, and fresh linked public-Op qualification then passed. The selector-free
C1 production confirmation at
`profiles/bench/r9700-projected-residual-t1-production-confirmation-retry1-20260920/attempt-1`
retained all 257 tokens and measured `28.9947`, `29.0153`, and `28.9945 tok/s`; its median decode
time was `0.9788658` of the retained control and within `0.0225%` of the admitted candidate median.
Independent audit reported `SHIP`, closing promotion for exactly the all-Q4/A8 base Text T1 route.
The `result.sha256` digest is
`f629374d5dba788ba93837ee06b74eab6be1650f95d05772694af848d37c9d5a`. This does not prove
physical bandwidth saturation or stall freedom.

The deterministic queue after reset is:

1. complete numerical accuracy at the selected shared chunk2048, then terminal base-artifact
   selection; bounded-panel attention and the full chunk-selection campaign are closed;
2. immediately make `DFLASH-RECIPE`, `DFLASH-QUALITY`, and `DFLASH-WHOLE` the primary performance
   work, targeting at least `60 decode-output tok/s` at C1 with exact public greedy-token parity;
3. retain only clearly reusable recipe-independent DFlash work while the selected artifact is
   unavailable. Do not start another base-decode mechanism. The qualified GDN combined-grid route
   remains preserved for a later matched whole C1 A/B but is not production-routed.

No additional recipe-independent DFlash GPU experiment is admitted merely because the final recipe
is blocked. The retained down scale-gather promotion has exhausted its scoped mechanism, and its
whole result proves that kernel savings alone cannot approach the terminal target at current
acceptance. Select another pre-recipe experiment only when exact retained attribution identifies a
distinct owner and a conservative whole-round bound that can materially change the decision; use a
fresh independently reviewed create-only package. Other remaining unchecked tasks retain their
declared dependencies. No absolute prefill-ceiling proof is required before proceeding.

- [x] `PREFILL-BOUNDED-REVIEW` Review retained whole-prefill attribution and exhausted candidates
  once for an overlooked large concrete gain. Record either one new mechanism with a quantitative
  whole-prefill bound, or that no immediate implementation is justified. In the latter case proceed
  directly to chunk selection and numerical accuracy. Preserve reusable BF16 references; later
  changes rerun only the numerical/quality evidence affected by their arithmetic or representation.
  CLOSED 2026-09-21: independent CPU review found no new large implementable gain. Retained Text
  attribution is Q4 `334.763 ms`, FP8 `307.311 ms`, attention `147.247 ms`, GDN `139.397 ms`.
  Larger Q4 tiles/pipelines, FP8 library/custom paths, fused attention and GDN alternatives already
  have negative timing or resource evidence. Dual-FMAC's `8.144 ms` saving is an exhausted small
  result; hypothetical peak substitution supplies no new mechanism. Evidence:
  `profiles/rocprof/r9700-retained-production-p2048-trace-plan-20260906/evidence.json`,
  `profiles/bench/r9700-q4-prefill-dual-fmac-whole-p2048-c1-full-v2-20260906/report.json`, and
  `profiles/bench/r9700-fp8-gate-up-m128n256-retained-reopen-20260906/attempt-4/decision.json`.
  Proceed to the current-build chunk campaign and numerical accuracy. No ceiling claim follows.

- [x] `PREFILL-CHUNK-ATTENTION` CLOSED 2026-09-21. Bounded query panels extend the existing
  tiled dense route to appended chunks and 128..8192 query rows through context262144 while
  retaining absolute causal positions, page mapping, and at most 384.1875 MiB score/max storage.
  G16/G32 FP64, active-count/panel/graph, typed-leaf, host-planner and five ISA/resource gates pass.
  Whole8K/chunk1024 improves `184.8640144 → 1184.242683 tok/s` (median duration ratio
  `0.155599830`). The fresh matched P2048 pair is `1664.303599 → 1659.247929 tok/s`
  (ratio `1.003513977`, within the 2% bound); the older1904 result was not reproduced.
  Independent result audit: SHIP. Admission:
  `profiles/bench/r9700-chunked-attention-candidate-leaf-retry1-20260921/whole/result.json`.
  Preserve the original raw-pass/public-fixture-failure evidence and completed8K cell; the
  corrected fixture and deferred-VRAM completion reran only missing checks. Full findings are in
  `docs/performance.md`. Resume the fresh panel-attention chunk campaign, then accuracy/DFlash.
Every future physical experiment must be launched through a reviewed package-local `commands.sh`,
not an ad-hoc reconstructed command. A package is runnable only when its plan contains no
`UNBOUND`, its runner has a non-mutating preflight mode, and its package validation passes. Record
the exact invocation in that package's plan so
the next agent never has to infer CLI flags from prose.

## After-parity router

### After parity: determine whether K4/W5 can beat base decode

**CLOSED (2026-09-06, 6fe53d53 build).** The C1/P129+G27 K4/W5 screen passed: DFlash K4/W5
beats base decode by a material margin (total 1.3178 s vs 1.6830 s, ratio 0.783; whole-output
21.25 vs 16.64 tok/s; decode-output 23.98 vs 18.07 tok/s). Both arms produce the identical
28-token sequence (exact match). DFlash speculative accounting: 15 rounds, 59 drafted, 11
accepted, 1 fallback, acceptance rate 0.186 (vs historical 0.561), acceptance length 1.733,
per-position [6,5,0,0]. The low acceptance rate (18.6% vs 56.1%) is below the historical
break-even assumption, yet DFlash still wins because multi-token-per-round processing
outweighs the low acceptance. Evidence:
`profiles/bench/r9700-dflash-k4w5-after-parity-screen-6fe53d53-20260906` (committed).
K5/W6 remains conditional: its historical 72/591 drafts over 120 rounds accepted no
fifth-position draft, so run it only if new K4 evidence could make K5 decision-relevant.

### If K4 fails or is marginal

Use a current-four-role unprofiled run first; the legacy all-Q4/G32 trace cannot rank the FP8 target
path. If phase accounting is insufficient, profile proposal, target verification,
acceptance/repair/commit, host/copy gaps, and graph launch, then pursue only an owner able to change
the `57.6915 ms` bound. Current leverage is target-verification Linear first: qualified T5
N5120/K17408 MLP-down changed `0.207386→0.057680 ms` (optimistic `9.5812 ms` over 64 layers), while
W5 attention changed `0.09549904→0.06228868 ms` (only `0.5314 ms` over 16 layers). Ablate RMSNorm,
GDN, or launch scheduling only if whole performance is marginal; preserve Op oracles, state
transactions, and graph-stable storage.

Do not reopen the rejected FP8 T5/T6 target gate/up challenger, the serial fifteen-launch W5 WMMA
route, adjacent M128N256 prefill tiles, non-temporal dot8, grouped-PV split512, or K1..11 DFlash
shortlist without a materially different source mechanism and a bound showing it can change the
end-to-end decision.

Recipe/acceptance, C1..4 admission, selected hardware profiling, and held prefill work are owned by
the corresponding unchecked tasks below. Package owners may prepare their fail-closed CPU artifacts
in parallel, but no prepared package bypasses its dependency or authorizes GPU execution.

## Deferred: selector-free base decode bandwidth

- [ ] `BASE-DECODE-BW` Usefully maximize selector-free base-decode memory throughput on the fixed
  R9700 at `C=1..4`; do not optimize or schedule a `C>4` cell. The retained production baseline is
  `27.05729956 tok/s` at C1/P8192+G256 with Device Graph. Its decoded packed-weight payload divided
  by wall time is `367.956 GB/s`, or `57.9%` of the same-session `635.9 GB/s` stream ceiling. Treat
  this only as a defensible *useful payload rate*: it omits other reads/writes and cache effects and
  is not a physical HBM-bandwidth measurement. Existing attribution makes the T1 native-dot8 Q4
  family the material owner. DEFERRED by the deterministic queue above: do not start another base
  mechanism before the numerical-accuracy, terminal-selection, and DFlash sequence. The saturation
  objective remains unchecked because reliable physical counters are unavailable and the retained
  result does not establish saturation or stall freedom. When explicitly resumed, begin at Layer 0
  with its real decode shapes and a wave-cooperative
  challenger whose bound can materially improve whole decode by raising useful memory-level
  parallelism while preserving the stored N16/K16 contract and serial semantic accumulation order.
  Do not rerun the rejected simple non-temporal dot8, grouped-PV split512, or split-K candidates
  unless a materially different source mechanism and end-to-end bound first justify one.

  Block-size selection is closed and must not be rerun. The `auto` cold sweep in
  `profiles/bench/r9700-base-decode-dot8-cold-block-sweep-20260919.json` passed the independent
  FP64/full-output bit-exact/status checks for all seven tuples and rotated more than 64 MiB of
  disjoint weight bytes between samples. Only N4096/K5120 favored 64 over the production
  256-thread block (`0.9824215` ratio); the other six tuples tied or lost and the call-weighted
  whole-decode bound is below `0.1%`. The same retained cold report also rejects the
  one-group-ahead source pipeline: its seven candidate/control ratios are `1.02449`, `1.00173`,
  `1.01111`, `1.01236`, `0.99819`, `0.99712`, and `1.00147`. ISA retained native dot8 with 26
  VGPR, 32 SGPR, and no LDS or scratch, but scheduled the next group's B64 loads after the current
  dot8 sequence and therefore did not implement the intended overlap. The losing pipeline is
  removed. Retain one production geometry, do not add a block-size selector, and require a
  materially different mechanism with a new whole-decode bound before further kernel work.

  The next admitted mechanism is the T1 GDN paired-Q4 projection: the N4096/K5120 query-key and
  N12288/K5120 value-z matrices consume the identical represented BF16 hidden vector, so one
  A8G64 quantization and one combined row grid can replace two quantize/Linear launches without
  changing either represented BF16 projection or the following snapshot convolution/state Op.
  The retained cold medians bound the pair at about `4.365 ms/token`; streaming the combined
  weights at the gate-up route's observed rate plus eliminating 48 duplicate quantizations and
  launches predicts roughly `0.75 ms/token` before ancillary savings. Qualify the complete paired
  boundary against the independent represented-weight FP64 oracle and both serial outputs, then
  require a cold complete-sequence win before any compile-gated whole-Engine C1 A/B. Select only
  T1 with both exact Q4N16K16/G64/FP16-scale shapes; mixed FP8/Q4 profiles and T>1 retain their
  existing routes. Do not direct-scatter past the convolution/state boundary in this challenger.

  Before GPU execution, prepare the normal create-only package and obtain independent `SHIP` review.
  Qualification must include the independent mathematical oracle at real shapes and the applicable
  exact comparison, gfx1201 ISA/resource/static routing evidence, cold and repeated direct-shape
  candidate/control timing, then a production Device-Graph whole C1 A/B at P8192+G256 with exact
  public-token parity. Extend whole admission through C2..4 only after C1 wins materially. Promote
  only a qualified, physically faster candidate that improves selector-free whole decode; otherwise
  retain it default-off and close it with the measured bound. A physical bandwidth-saturation or
  stall-freedom claim additionally requires reliable hardware counters; if gfx1201 counters remain
  unavailable, report useful payload rate and whole speed only, with profiled timing used solely for
  attribution. The bound selector-free package completed three authority-matching exact-token
  runs at `27.0059`, `27.1906`, and `27.0281 tok/s` (median `27.0281083 tok/s`). Its exact logical
  Linear-boundary rate is `367.8525 GB/s`, or `57.85%` of the same-session `635.9 GB/s` stream
  ceiling; this is not physical HBM traffic or stall-freedom proof. The optional profile is not
  needed for the live paired-projection decision. Its balanced direct gate passed with exact serial
  outputs and a zero-step represented-weight FP64 result; the allocation-balanced medians were
  `0.1303605 ms` serial and `0.1024000 ms` paired. The independently reviewed whole A/B passed exact
  public-token parity in all six processes. Its three candidate/control decode-time ratios were
  `0.9784431`, `0.9794007`, and `0.9784902` (mean `0.9787780`, upper two-standard-error bound
  `0.9794013`); candidate rates were `27.9897`, `27.6197`, and `27.6044 tok/s`. This authorizes
  selector-free promotion of the exact all-Q4 T1 pair, not a physical-HBM saturation claim. The
  qualification selector and benchmark-report field are now removed: production selects the pair
  only for one token with both exact Q4G64_F16S projection bindings; mixed weights and T>1 retain
  the unchanged query-key selected-Linear plus serialized value-z fallback, and the following
  snapshot convolution/state boundary is unchanged. Compile-time route receipts cover C1 selection,
  C2..4 rejection, and both mixed-weight orders. The create-only selector-free confirmation at
  `profiles/bench/r9700-gdn-q4-pair-t1-production-confirmation-20260919` passed independent review
  and confirmation. All C1 runs matched the retained token authority and reproduced the admitted
  speed (`9.266168907 s` median, `27.6274 tok/s`, production/retained time ratio `0.9997211`). The
  unchanged C2..4 fallback repeated exact tokens, with median aggregate rates `32.1617`, `43.4327`,
  and `50.5869 tok/s`. This closes promotion of the T1 GDN Q4 pair but not BASE-DECODE-BW.

  The recipe-independent T1 BF16 GDN projected-control direct gate fused the two
  N48/K5120 projections and control formula while preserving each BF16 projection cast boundary.
  Its corrected logical saving is 10,432 bytes/layer; the retained service ceiling is about
  `0.701 ms/token`. The package compared serial, combined-grid, and fused routes with an independent
  FP64 oracle, exact represented-boundary parity, balanced cold timing, and exact gfx1201 static
  evidence. Its first wrapper invocation did not reach the GPU because the linked executable could
  not locate `libamdhip64.so.7`; no report was created. The corrected wrapper adds the ROCm runtime
  search path, and independent re-review proved that all ROCm dependencies resolve. A second
  pre-kernel attempt exposed an invalid assumption that only one HIP device is visible; the
  corrected qualifier explicitly selects device 0, verifies its exact R9700/gfx1201 identity, and
  derives the checked power node from its HIP PCI identity. The final measurement passed: serial,
  combined-grid, and fused medians were `0.0337600`, `0.0240800`, and `0.0209200 ms/layer`;
  every one of the three disjoint allocation copies favored fused. The observed fused saving is
  `0.0128400 ms/layer`, or `0.6163200 ms/token` across 48 layers, exceeding the `0.2 ms/token`
  admission threshold by 3.08x. The represented BF16 projection/control outputs are bit-exact to
  serial; independent FP64 maxima are `0.031108081` for projection and `1.19e-7` for control.
  Static evidence reports wave32, no scratch/spills, and fused resources of 11 VGPR, 25 SGPR, and
  2 KiB LDS. Evidence:
  `profiles/bench/r9700-bf16-gdn-control-t1-20260919/report.json`. This admits a production-symbol
  no-a/b candidate and then a separate matched whole-model C1 A/B; it does not itself authorize
  promotion. Preserve the explicit BF16 rounding boundary in registers, exact T1-only routing,
  and the existing T2..4 fallback/workspace. The compile-gated production candidate, direct
  production-symbol qualifier, and whole C1 A/B package are now implemented. Independent review
  found the semantic Op/raw kernel and exact BF16 seam coherent, and accepted the repaired direct
  device/PCI/power/invocation/artifact receipts plus the identity-closed future whole wrapper. The
  direct production-symbol run passed: the candidate is bit-exact to the two-production-linear
  control path, passes the independent FP64 oracle and malformed boundaries, and the exact symbol
  is wave32 with 11 VGPR, 22 SGPR, 2 KiB LDS, no scratch/spills, exactly two FP32 stores, and no
  BF16 stores. All retained device, process, source, executable, assembly, and receipt identities
  passed independent audit. Evidence:
  `profiles/bench/r9700-bf16-gdn-control-t1-production-qualification-20260919/report.json`. Run the
  separately reviewed matched whole gate began but stopped at run 3/6 before inference with
  `hipMalloc arena: hipErrorOutOfMemory`. Runs 1 control and 2 candidate completed; run 3 candidate
  retained the failure. After process exit, device 0 remained at 83% VRAM with no KFD PID; the
  maintainer later identified a stopped llama.cpp Vulkan container as the owner. Preserve this
  package/results; never rerun or append it. The maintainer reset device 0, and a separate reviewed
  create-only retry package reran the complete balanced campaign from the start. That independently reviewed
  package is `profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-retry-20260919`. It rebinds the
  retained failure and all original authorities, reruns all six roles, and refuses plan/results or
  any role unless PCI-derived device-0 VRAM is at most both 1 GiB and 5% with power `auto`. Its
  reset preflight passed at 59,912,192/34,208,743,424 bytes, but its first control process exposed a
  wrapper teardown race: inference exited 0 with the exact retained 257-token hash at
  `9.258297426 s` decode (`27.65087232 tok/s`) and power `auto`, while the immediate post-exit sample
  still saw 14,536,441,856 bytes. VRAM then settled to 59,912,192 bytes without another reset. The
  four retained files match `results/result.sha256`; no candidate run or pair exists. Seal this
  retry package and never append or rerun it. A fresh retry2 package must bind this failure, rerun
  all six roles, and preserve the same thresholds while polling post-exit VRAM for bounded
  asynchronous reclamation before deciding that the device is dirty. That create-only package is
  `profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-retry2-20260920`; two independent reviews
  reported `SHIP` after strict-deadline, telemetry-failure, power/capacity-drift, helper-binding,
  and non-mutating-preflight checks. Its complete six-role campaign passed exact 257-token parity.
  Candidate/control decode-time ratios were `0.9876921`, `0.9870972`, and `0.9868352`; the median
  was `0.9870972` and mean-plus-two-standard-errors was `0.9877152`. Candidate rates were
  `27.9803`, `27.9999`, and `27.9901 tok/s` versus control `27.6359`, `27.6386`, and
  `27.6217 tok/s`. Every process exited 0 with power `auto`; each immediate ~14.536 GB post-exit
  sample drained to 59,912,192 bytes in 0.10074--0.10109 seconds within the unchanged gates.
  Evidence: `profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-retry2-20260920/results`; the
  SHA-256 of `result.sha256` is
  `e6fb6b0cf14aa8a28273e5a32b35f76fba7700dd31a971a35a9d4a016ad69569`. This admitted production
  promotion. The selector is now removed after both attention campaigns completed; the fresh
  selector-free C1..4 confirmation named below is the remaining composition gate.

  The all-Q4 T1 attention paired-projection direct gate compared the two
  complete N7168/K5120 Q4 linears plus four incumbent extracts against one shared A8G64
  quantization and one combined N14336 native-IU4 kernel. Independent review passed after correcting
  the whole-token bound to the 16 full-attention layers and giving the query-key and gate-value
  matrices distinct represented weights. The final gate passed with complete BF16 output parity,
  zero BF16 steps against the represented-weight FP64 oracle, native IU4/wave32, 20 VGPR, 24 SGPR,
  and no LDS/scratch/spills. Allocation-balanced serial and paired medians were `0.1401195` and
  `0.1039595 ms/layer`; all three disjoint weight allocations favored paired by at least 8.4%.
  Across exactly 16 full-attention layers the observed saving is `0.5785600 ms/token`, or 1.5637%
  of the retained 36.99852 ms/token baseline. Evidence:
  `profiles/bench/r9700-attention-q4-pair-t1-direct-20260919/qualification.json`. This admits only
  a production-symbol candidate and matched selector-off/on whole C1 A/B with exact public tokens;
  it does not authorize promotion. Select only T1 with both exact all-Q4 Q4G64_F16S bindings;
  mixed weights and T2..4 retain their existing routes until separately qualified.
  The compile-gated production implementation exists. Its direct production-symbol package is
  `profiles/bench/r9700-attention-projection-t1-production-qualification-20260919`, and the dependent
  whole package is `profiles/bench/r9700-attention-q4-pair-t1-whole-ab-20260919`. Independent review
  reported `SHIP` for the direct package's create-only hardening and the whole package's complete
  cache, source, and exact-invocation authorities. The whole package must not be prepared until the
  direct qualification receipt exists. That direct production-symbol qualification now passes:
  complete outputs are bit-exact to serial plus four extracts, the independent represented-weight
  FP64 oracle is zero BF16 steps, and all guards pass. Static evidence is native mixed-IU4 dot8,
  wave32, 20 VGPR, 24 SGPR, and no LDS/scratch/spills. Allocation ratios are `0.613515`, `0.842832`,
  and `0.648888`; the global median changes `0.126980→0.092000 ms/layer` (1.380217x), bounding
  `0.559680 ms/token` or `1.512709%` of whole decode. Evidence:
  `profiles/bench/r9700-attention-projection-t1-production-qualification-20260919/qualification.json`
  (SHA-256 `8bfebf95b5b4a6e47db7e5d41dff686e5b145e41b75850ef669de80d06e95d74`).
  This authorizes only preparing the dependent whole gate, not promotion.
  That whole gate was prepared and two independent reviews reported `SHIP`; its complete live
  caches differ only at the T1 attention selector, while BF16 GDN and C2..4 paired selectors remain
  off. The complete six-role C1 campaign passed exact 257-token parity. Candidate/control ratios
  were `0.9887575`, `0.9873370`, and `0.9869812`; median `0.9873370` and mean-plus-two-standard-errors
  `0.9887771` clear every gate. Candidate rates were `27.9817`, `28.0051`, and `28.0088 tok/s`
  versus control `27.6671`, `27.6504`, and `27.6442 tok/s`. Evidence:
  `profiles/bench/r9700-attention-q4-pair-t1-whole-ab-20260919/results`; the SHA-256 of
  `result.sha256` is `9054b308fb675b236532b847c72b7774165824dce1e43bceff4cbe83c0db17f1`.
  This admitted production promotion. The selector is now removed after the C2..4 campaign
  completed; do not rerun this gate or claim physical HBM bandwidth from its unprofiled timing.

  The independently reviewed C2..4 paired-WMMA direct gate covers both the GDN N4096+N12288 pair
  (48 calls/round) and attention N7168+N7168 pair (16 calls/round). It compares two complete
  production WMMA linears against shared-quantize/two-WMMA and shared-quantize/combined-grid
  routes using distinct role-salted packed weights, complete exact BF16 outputs, and an independent
  decoder-based FP64 oracle. Its fail-closed bound uses the retained selector-free confirmation
  round medians `62.1857104`, `69.0723372`, and `79.0718535 ms` for C2, C3, and C4. Source and
  wrapper reviews passed after fixing role indistinguishability, balanced warmup coverage,
  create-only report writes, and retained device/power/invocation/static provenance. Its first
  authorized invocation stopped before HIP device selection or timing: a stream-state check
  incorrectly rejected a successful baseline-file iterator read. No qualification report was
  created; the failure is retained in `attempt-1-read-failure.json`. The reader now rejects only
  `badbit`; rebound identity and non-GPU preflight passed same-review approval. The final direct
  gate passed all six cases with complete three-arm BF16 parity, independent represented-weight
  FP64 error of at most one BF16 step, native IU4/wave32, 57 VGPR, 32 SGPR, and no LDS,
  scratch, or spills. Every candidate allocation beat its paired serial allocation. Exact
  layer-weighted savings are `6.1705039`, `6.5730799`, and `6.7305520 ms/round`, or 9.9227%,
  9.5162%, and 8.5119% against the retained C2, C3, and C4 round medians. Evidence:
  `profiles/bench/r9700-a8q4-pair-wmma-c2c4-design-20260919/qualification.json`. This admits a
  distinct exact-C2..4 all-Q4 production candidate and matched whole C2..4 A/B; it does not
  authorize promotion or extend the route to mixed weights, T1, or another token width. The
  compile-gated production implementation is now independently reviewed: one shared raw WMMA
  kernel feeds semantically owned attention direct outputs and GDN query-key/value-z outputs, with
  exact T2..4 plus both-Q4G64_F16S routing and unchanged T1/mixed/other-width fallbacks. The direct
  production-symbol package is
  `profiles/bench/r9700-paired-projection-c2c4-production-qualification-20260919`; its non-GPU gate
  passes native IU4/wave32 at 57 VGPR, 32 SGPR, and no LDS/scratch/spills. Its production-symbol
  qualification now passes all six GDN/attention T2..4 cases: complete three-arm BF16 parity,
  independent FP64 error at most one BF16 step, every guard, and all 18 allocation medians faster
  than serial. Weighted savings are `6.2195198`, `6.4822403`, and `6.7388880 ms/round`, bounding
  `10.0015%`, `9.3847%`, and `8.5225%` gains at C2, C3, and C4. Evidence:
  `profiles/bench/r9700-paired-projection-c2c4-production-qualification-20260919/qualification.json`
  (SHA-256 `d884a7a97fdbc694ef08ad57c5b3f6cc57323c3e2c6f98fe404802fe5b4c0b06`). This
  authorizes only whole-package preparation, not promotion. Independent review reported `SHIP` for
  the whole package at
  `profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919`; no separate review artifact was
  created. A reviewed lexical PCI power-path repair removed a false symlink-resolution rejection.
  The complete 18-process campaign passed exact 257-token parity. Its C2 pair ratios were
  `0.9250746`, `0.9245411`, and `0.9246617`; C3 ratios were `0.9283638`, `0.9272089`, and
  `0.9269505`; C4 ratios were `0.9338502`, `0.9341644`, and `0.9341877`. The corresponding
  mean-plus-two-standard-error bounds were `0.9250822`, `0.9283767`, and `0.9342851`. Evidence:
  `profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919/results`; the SHA-256 of
  `result.sha256` is `26c4ae49e7a24ce97d1f56c891e841942507678aaab8a0df0ada315996e93cda`.
  Two independent result audits reported `SHIP`. This admitted production promotion for exactly
  all-Q4 T2..4, not mixed weights or another width and not a physical-bandwidth claim. The BF16 GDN
  T1, all-Q4 attention T1, and C2..4 paired selectors and benchmark fields are now removed; exact
  predicates and all fallbacks remain. Focused selector-free builds and routing qualifiers passed,
  and independent code review reported `SHIP`. The fresh composition package
  `profiles/bench/r9700-three-route-production-confirmation-20260920` passed exact retained tokens
  at C1..4. Decode times were `9.0186748`, `14.7080029`, `16.4007540`, and `18.8960216 s`, or
  aggregate rates `28.3855`, `34.8110`, `46.8271`, and `54.1913 tok/s`; ratios to the retained
  candidate medians were `0.9865955`, `0.9997177`, `1.0002975`, and `1.0000448`. All process,
  power, VRAM-drain, identity, and closure checks passed. The SHA-256 of `result.sha256` is
  `9be2f67ff0f74f92e509bbfa6bb0909e7feb5e38993aed184d71ce1cd7033148`. This closes the
  three-route promotion composition gate, not physical HBM saturation or a new A/B claim.

  After the admitted paired routes are resolved, the direct projected-residual mechanism covers
  all-Q4 T1 projected-residual fusion for N5120/K6144 and N5120/K17408. These are respectively the 16
  attention-output plus 48 GDN-output projections and 64 MLP-down projections: exactly 128
  residual publications/token. The retained ordinary C1 trace measures the removable residual-add
  family at `0.2114 ms/token` (`0.20956 ms` interval union); fusing the exact
  `delta=BF16(dot); x=BF16(FP32(x)+FP32(delta))` boundary into the native-IU4 epilogue also removes
  2.62144 MB/token of delta store/read traffic. Qualify both complete Linear-plus-residual
  boundaries with distinct actual packed weights, an independent represented-weight FP64 oracle,
  exact residual bits, three-copy balanced cold timing, and exact-symbol native-IU4 static evidence.
  Require `64*(saving_N5120K6144+saving_N5120K17408) >= 0.2 ms/token` before whole C1 A/B; keep
  mixed weights and T>1 on the current path. After projected-residual closure, the next ranked
  mechanism is T1 RMSNorm+A8G64 preparation fusion. Layer 0 identifies 128 potential normalization
  boundaries/token; start with the 64 MLP boundaries through a public normalized-linear Op and
  the admitted native-dot8 consumer. Screen the complete boundary under Device Graph against an
  independent oracle and require at least `0.2 ms/token` weighted direct saving before preparing
  a whole C1 A/B. Neither the 128-boundary count nor eliminated staging proves a speed gain.
  The reviewed direct public-Op qualification at
  `profiles/bench/r9700-normalized-linear-t1-qualification-20260920/attempt-1` passed the independent
  FP64 normalization-to-BF16-to-exact-A8-to-signed-Q4 oracle with zero BF16-step error, exact codec
  and public-control parity, graph recovery, and native-IU4 embedded ISA. The complete-boundary
  median improved `0.2406990→0.2347595 ms`; all three allocations won, mean paired ratio plus two
  standard errors was `0.9790384`, and the 64-call estimate saved `0.3801284 ms/token`, clearing the
  `0.2 ms/token` gate. Independent audit reported `SHIP` for whole C1 A/B preparation only. The
  `result.sha256` digest is
  `01e1e0571d92541311195202df919d3750759f3bcedf7033944f604bcf2baf2c`; closure SHA-256 is
  `71576628e039d62dcdc3a31d17c60804765e95c859ec2118d16602ae2bd64dc4`. The subsequent reviewed whole C1 A/B at
  `profiles/bench/r9700-normalized-linear-t1-whole-ab-20260920/attempt-1` retained all 257 tokens.
  Candidate/control decode-time ratios were `0.9895615`, `0.9896042`, and `0.9897497`; median
  `0.9896042` and mean plus two standard errors `0.9897524` passed the gate. Mean decode improved
  `29.0050→29.3087 tok/s` (about `1.05%`). Independent audit reported `SHIP` for scoped promotion
  review; the `result.sha256` digest is
  `2484b71a86314ec8bf4b116fe0b5caecb707d77747666a41dec5f2ee95ee32ee`. Selector removal and independent
  implementation review passed for exactly these 64 all-Q4/A8 ordinary base Text C1 MLP boundaries;
  prefill, verify/DFlash, MTP, C2..4, mixed inventory, A4, attention, and GDN retain their fallbacks.
  Fresh linked public-Op qualification at
  `profiles/bench/r9700-normalized-linear-t1-promoted-qualification-20260920/attempt-1` passed with
  `0.3647995 ms/token` direct saving. Selector-free C1 confirmation at
  `profiles/bench/r9700-normalized-linear-t1-production-confirmation-20260920/attempt-1` retained
  all 257 tokens and measured `29.3174`, `29.3157`, and `29.3116 tok/s`; median decode-time ratio
  to the retained control is `0.9893966`, with every run faster than the admitted candidate median.
  Its `result.sha256` digest is
  `f28d5a0e89bc8cfaa438d7c7ef86405c5eb486c650838c57bf086d9b40c2aebb`. Independent confirmation
  result audit reported `SHIP`, closing this scoped promotion. Do not rerun these sealed attempts.
  No result proves physical bandwidth saturation or stall freedom.

  The explicit next-group prefetch challenger for T1 N34816/K5120 gate/up is terminally rejected.
  Its linked complete normalized-linear boundary qualified numerically and in embedded gfx1201
  ISA, but measured `0.2361795→0.2369395 ms` (control→candidate), a weighted saving of
  `-0.0486398 ms/token`, with every allocation slower and paired ratio mean plus two standard
  errors `1.0094922`. This fails the `0.2 ms/token` admission gate; no whole C1 A/B is justified.
  Evidence is sealed at `profiles/bench/r9700-gate-up-prefetch-qualification-20260920/attempt-1`;
  `result.sha256` digest is
  `5ea1c61d7d02fdeb971c151ded538872cf46121c31d299686bf0b42390b68e15`, closure SHA-256
  `8e08b7a134601a6dddb5c16c5a74b7b19ed6f635bf435f50ee50b742c75ac4db`.
  The independent FP64 oracle had zero BF16-step error and maximum relative L2 `0.001801662`;
  codec and public-control parity were exact. Embedded native-IU4 ISA preserved four successor
  B64 loads ahead of current compute at 26 VGPR, 36 SGPR, occupancy 16, zero LDS/scratch/spills.
  Correct overlap therefore did not deliver a useful complete-boundary gain on this workload.
  Independent result review accepted the rejection. Removed the temporary qualifier/kernel/checker
  ownership; retained assembly, embedded objects, receipts, and package scripts are historical
  evidence, not rerunnable commands. Production was never changed by this challenger. Do not
  reopen this overlap, geometry
  remapping, non-temporal loads, or split-K without a distinct mechanism and new bound.

  **GDN projection/control combined-grid direct decision closed; whole promotion deferred.** Define one
  semantically closed qualification-only Op from the same represented BF16 hidden input to
  explicit QK, value-Z, g, and beta outputs. Combine the existing 48-CTA BF16 projected-control
  branch and 64-CTA paired-Q4 branch in one heterogeneous grid; keep convolution and persistent
  state transitions after the Op boundary. Ownership remains in `src/ops`, called through the
  existing GDN execution-leaf family. Scope is ordinary base Text T1 with exact Q4+BF16_CTRL
  weights. Any scratch is caller-owned and explicit; do not introduce hidden prepared workspace.
  This is projection/control overlap. Finish its bounded numerical/timing decision, then follow
  the user's direction to recipe-independent DFlash optimization before another base-decode
  mechanism; defer normalization fusion until base-decode work resumes.

  Retained Layer-0 bounds are `0.102400 ms/layer` for the Q4 pair and `0.020920 ms/layer` for
  controls, with `44,564,480` Q4 bytes and `983,040` BF16 bytes. Estimated additional streaming
  costs `0.002259 ms/layer`; ideal overlap saves about `0.896 ms/token` across 48 layers.
  This is a feasibility bound, not a measured gain. Clearing the `0.2 ms/token` admission margin
  requires only `4.167 us/layer` complete-boundary saving. CPU/static feasibility now has
  independent `SHIP`: the qualification-only `112x256` grid maps CTAs `0..63` to paired-Q4
  QK/value-Z outputs and CTAs `64..111` to BF16 control heads `0..47`, publishing FP32 g/beta.
  CTA-uniform branches keep all nine control barriers out of Q4 CTAs. Emitted gfx1201 code
  preserves native IU4, the exact incumbent control reduction and both BF16 rounding seams;
  the combined envelope is 34 VGPR, 52 SGPR, 2048 bytes LDS, compiler occupancy 16, and zero
  scratch/spills. Compiler occupancy is a resource ceiling, not measured active occupancy.
  An explicit unresolved risk is four vector B128 activation loads replacing the incumbent's
  two scalar B256 loads; their issue cost may erase the overlap benefit. Static evidence is
  reproducible through `make -C tools/r9700 gdn-projection-control-grid-static`. This is no
  numerical-correctness or performance claim by itself. The independently reviewed complete-boundary
  package at `profiles/bench/r9700-gdn-projection-control-grid-qualification-20260920/attempt-1`
  subsequently passed: control/candidate medians were `0.17841950/0.17215950 ms`, giving
  `0.30048001 ms/token` projected saving across 48 layers; ratio mean plus two standard errors was
  `0.98806244`, all three allocation medians won, and 23/24 individual pairs won. Removing the one
  long control sample still gives `0.29380846 ms/token` and ratio upper `0.97378743`.
  Projection relative L2 was `0.00154624` with zero BF16-step difference; codec, incumbent output,
  graph replay/recovery, guards, and input immutability passed, and control errors were below
  `7.18e-7`. Result-manifest SHA-256 is
  `ccd4b33664c3be9bd4d35811debe81c4ec520ea7693e44addae9e03b6bd1e23c`; independent result audit
  reported `SHIP` for whole C1 A/B preparation. Production routing remains unchanged. Preserve the
  qualification-only route and defer its matched whole C1 A/B until base-decode work resumes;
  recipe-independent DFlash optimization is active next, before any other base mechanism.
  The first reviewed package,
  `profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919`, passed compile/static preflight
  but stopped on the first K6144 incumbent launch before candidate parity or timing:
  `launch_incumbent(...): invalid argument`. The qualifier had used a null stream while
  `eager::residual_add_bf16` requires an explicit stream. This is a harness defect, not a candidate
  rejection; `attempt-1` retains compile/assembly/static evidence but no qualifier stdout/stderr or
  exit receipt. Never rerun or append it. The fresh package
  `profiles/bench/r9700-a8q4-projected-residual-t1-design-retry-20260920` binds the original evidence,
  threads one owned nonblocking stream through both arms, transfers, scrub, warmups, events, and
  synchronization, and retains process output/exit/closure on every ordinary outcome. Its
  compile/static preflight passed and independent review reported `SHIP`; no numerical, timing, or
  admission criterion changed. That qualifier then exited 0 and printed a `0.884448 ms/token` pass,
  but the wrapper sealed the attempt failed because the shared JSON escape helper emitted literal
  newlines inside the embedded compile receipt. A read-only diagnostic recovery found K6144
  `0.0764400→0.0694605 ms` and K17408 `0.1420800→0.1352400 ms`, zero BF16 steps for both delta and
  residual, exact complete residual parity, every allocation ratio below one, and aggregate saving
  `0.8844481 ms/token`; this merits correction but cannot itself authorize a whole A/B because the
  retained report is not strict JSON. Preserve the retry attempt and its closure; never rerun it.
  The independently reviewed retry2 package at
  `profiles/bench/r9700-a8q4-projected-residual-t1-design-retry2-20260920` changes only JSON control
  escaping, binds both prior evidence inventories, passes exact strict-JSON round-trip for every
  control byte plus quote/backslash/UTF-8, and retains the unchanged explicit-stream/static gate.
  Retry2 passed with strict JSON and a complete verified closure. N5120/K6144 improved
  `0.0765195→0.0695995 ms` and N5120/K17408 improved `0.1424995→0.1350995 ms`; every allocation
  median beat its incumbent. Both delta and residual had zero BF16-step error, complete residual
  bits matched exactly, and all guards passed. Native IU4/wave32 used 18 VGPR, 32 SGPR, and no
  LDS/scratch/spills under `auto`. Weighted saving was `0.9164801 ms/token`, exceeding the
  `0.2 ms/token` gate. Evidence:
  `profiles/bench/r9700-a8q4-projected-residual-t1-design-retry2-20260920/attempt-1`; the SHA-256 of
  `result.sha256` is `0583ef9e618ba2bc8b80a91a8312425f2035a109991a9743e71c9e4cc38c1b0d`.
  This admits only the exact all-Q4 T1 candidate and preparation of matched whole C1 A/B;
  production-symbol qualification, exact public tokens, and whole-inference performance admission
  remain required before promotion.
  Production integration now uses the public `ops::projected_residual_t1` contract and its A8G64
  workspace query; only selector-on, all-Q4 base Text T1 residual projections qualify. Mixed
  inventory, MTP, prefill, T>1, and selector-off retain the incumbent path. The first production
  package, `profiles/bench/r9700-a8q4-projected-residual-t1-production-qualification-20260920`, failed
  before numerical qualification because hardcoded DRM `card2` identified the integrated GPU.
  Production `-retry1-20260920` fixed power authority to PCI `0000:13:00.0` but aborted when a
  malformed public binding reached `HIP_CHECK`. Production `-retry2-20260920` adds complete public
  address/alignment/extent/alias rejection and host preflight with GPU visibility disabled.
  Preserve both failures; never append or rerun them.
  The passing production retry2 validates the actual public Op and linked production code object:
  independent FP64 oracle and complete residual bits are exact, malformed/poison/canary checks
  pass, and three captured graph replays per shape match eager output. At `auto`, K6144 improved
  `0.0759595→0.0692600 ms`, K17408 `0.1423595→0.1352800 ms`, every allocation median improved,
  and weighted saving was `0.8818557 ms/token` against the unchanged `0.2 ms/token` bound. Native
  IU4/wave32 retains zero LDS/scratch/spills. Evidence:
  `profiles/bench/r9700-a8q4-projected-residual-t1-production-qualification-retry2-20260920/attempt-1`;
  `result.sha256` SHA-256 is `206b15a42c9e1cb7df2150d834659b8ed3d167dd643fc74a771213a83b787e6d`,
  and `closure.json` SHA-256 is `9fbab6cfe0332847fcbd52f15c24dc95d80cc543349daaec67ed3be496533e03`.
  The subsequent reviewed whole C1/P8192+G256 A/B at `auto` passed all three adjacent balanced
  pairs with ordinary Device Graph and the fixed FP8-K/INT4-V cache. In execution order,
  control/candidate/candidate/control/control/candidate rates were `28.3689630`, `29.0009861`,
  `29.0069622`, `28.3912167`, `28.3819193`, and `28.9849878 tok/s`. All six retained 257-token
  vectors matched both their pair and the selector-free C1 authority. Candidate/control decode-time
  ratios were `0.9782068408`, `0.9787724938`, and `0.9791937627`; their mean was `0.9787243658`,
  mean plus two standard errors `0.9792961944`, and median `0.9787724938`, passing the unchanged
  every-pair/upper-bound `<1` and median `<=0.99` gates. Evidence:
  `profiles/bench/r9700-projected-residual-t1-whole-ab-20260920/attempt-1`; `result.sha256` SHA-256
  is `e7337ec8e9bc19d8519f52f06ffcb50359270fa47a4416ff859a4aee8afa9b79`.
  The independent result audit admitted promotion for exactly all-Q4/A8 base Text T1,
  N5120/K6144 or K17408. Mixed inventory, MTP, prefill, T>1, and A4 remain excluded. Selector
  removal and implementation review passed. Fresh linked public-Op qualification then passed with
  `0.8704314 ms/token` weighted direct saving. The selector-free C1 production confirmation at
  `profiles/bench/r9700-projected-residual-t1-production-confirmation-retry1-20260920/attempt-1`
  retained all 257 tokens and measured a `29.0015 tok/s` mean; its median decode time was
  `0.9788658` of the retained control and all three runs stayed within `1.01` of the admitted
  candidate median. Independent audit reported `SHIP`; the `result.sha256` digest is
  `f629374d5dba788ba93837ee06b74eab6be1650f95d05772694af848d37c9d5a`. Promotion is closed.
  This whole speed gain establishes neither physical memory-bandwidth saturation nor stall freedom.

## Active now: DFlash semantic and schedule work

- [x] `DFLASH-SCHEDULE` Finish recipe-independent exact-shape work only for K4/W5 and K5/W6; retain
  the K1..11 input contract, not its campaign. Packed Q4 is qualified only for N34816/K5120 at
  T=4,5,6,8,10,12,18,20; T15/T16/T24 and unlisted shapes remain WMMA. Qualified T5
  N5120/K17408 MLP-down stays default-off pending exact current-companion whole evidence and a
  material win. Do not rerun the rejected FP8 T5/T6 target gate/up route (3.77x/3.64x incumbent).
  Rows5/6 RMSNorm is qualified but stays off. Consume the retained failed three-load gate; do not
  rerun it. Text parity closed 2026-09-06 (`DFLASH-TEXT-P129`), unblocking the schedule speed
  work above. CLOSED 2026-09-08: the small-T packed-Q4 candidate (N34816/K5120 + MLP-down T5,
  `NINFER_R9700_DFLASH_SMALL_T_CANDIDATE`) was A/B-tested on the 6fe53d53 combined companion
  (C1/P129+G27, K4/W5 + K5/W6, eager, 3 reps) and found to have no material decode-speed effect
  (K4/W5 candidate/control ratio 0.9993, K5/W6 ratio 0.9977, both within noise over 3 reps); it
  stays default-off. The matched-A/B parity gate also caught a K5/W6 5th-column verify-context
  correctness bug, filed as `DFLASH-K5W6-VERIFY`. Evidence:
  `profiles/bench/r9700-dflash-small-t-whole-ab-6fe53d53-20260906` (sealed).

- [x] `DFLASH-K5W6-VERIFY` Fix the K5/W6 verify-context correctness bug. The greedy K5/W6 DFlash
  path (draft_window=5, verify_width=6, chain-verify) emits a wrong token: at C1/P129+G27 on
  6fe53d53 it produces token 109600 where the base (ordinary) decode produces 96917 (token index
  5 of the 28-token sequence; all other 27 positions match). Both the small-T control and
  candidate builds diverge identically, so it is deterministic and pre-existing (not caused by the
  small-T selector).

  Investigation progress (2026-09-09, current HEAD 437881fc, build-r9700):
  (1) INDEX-5 ROOT CAUSE FOUND + FIXED (uncommitted): the fused attention leaf
  (`fp8_int4_kv_attention_fused`, the only route the W=6 verify takes) used a full-precision
  BF16->FP32 query, while the ordinary decode and the W5 batched-WMMA route quantize the query to
  FP8 E4M3FN before the QK dot product. The fused leaf now round-trips its query through the same
  FP8 codec (`fp8_int4_kv_attention.hip:503`). On the rebuilt binary, K5/W6 now matches ordinary
  for the first 17 tokens (index 5 is 96917, was 109600).
   (2) RESIDUAL INDEX-17 DIVERGENCE — RESOLVED (option b): the default build had
   `NINFER_R9700_ATTENTION_PARITY_CANDIDATE=0` (now `NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE`),
   which gated off the W5 batched-WMMA route, so BOTH K4/W5 and K5/W6 fell through to the fused
   leaf and emitted identical tokens that diverged from ordinary at index 17 (118178 vs 96560).
   The sealed A/B ran with `=1`, where K4/W5 used the W5 WMMA leaf (matched ordinary for all 28
   tokens) and only K5/W6 used the fused leaf. So the residual divergence was a second, deeper
   numerical inconsistency in the fused leaf (its reduction tree differs from the WMMA leaves'),
   not the query-precision issue fixed in (1), and not the DFlash verify logic (the sealed A/B's
   K4/W5 proved that correct on the W5 WMMA leaf).
   DECISION (made 2026-09-10, user chose (b)):
     (a) Rebuild with the flag `=1` (the sealed A/B config): K4/W5 -> W5 WMMA leaf (matches
         ordinary); K5/W6 stays on the fused leaf and still diverges at index 17 unless (b) or (c)
         is also done. Fastest way to confirm the diagnosis on the GPU.
     (b) Add a W6 batched-WMMA route (generalize the W5 WMMA leaf to W=6) so both widths run on the
         same WMMA numerics as ordinary. Cleanest correctness fix; new kernel + qualification.
         CHOSEN.
     (c) Make the fused leaf's reduction tree match the WMMA leaves' (changes a production kernel's
         numerics, including the short-context decode fallback). Hardest.
   The earlier "phantom-entry" and "suspected locations" leads (dflash_impl.h:765-772,
   speculative_round.hip:479-500/427-441) are SUPERSEDED by (1)/(2): the fault was in the
   attention leaf's numerics, not the KV transaction or verify position-filling. RESOLVED by (b):
   K5/W6 now runs on the W5/W6 batched-WMMA route and matches ordinary exactly (see the closure
   note below). Evidence:
   `profiles/bench/r9700-dflash-small-t-whole-ab-6fe53d53-20260906/results` (sealed); fresh
   pre-fix runs in `/tmp/opencode/k5w6-fix-test/` (ordinary/k4w5/k5w6 .json).

   OPTION (b) IMPLEMENTED (2026-09-10, HEAD 437881fc + uncommitted W5W6 generalization): the user
   chose (b). The W5 batched-WMMA route was generalized to W5/W6: `use_dflash_w5_batched_wmma`
   -> `use_dflash_w5w6_batched_wmma` (admits query_rows in {5,6}); the three kernels renamed
   `*_batched_w5*` -> `*_batched_w5w6*` (grid-parameterized, no hard-coded 5); the PV-launch and
   QK grids now use `a.query_rows`; workspace sizing is parameterized by `rows`; routing +
   workspace planning in `r9700_full_attention.hip` admit 5 or 6 rows. The diagnostic fused-query
   FP8 round-trip from (1) was REVERTED (diagnostic-only, not the production solution; K5/W6 no
   longer needs it because it now runs on the WMMA route). VERIFIED: on a build matching the
   sealed control's four candidate flags (DFLASH_MLP_DOWN_T5 / DFLASH_RMSNORM_ROWS56 /
    FP8_PREFIX_COMMON_ALGO / GDN_VERIFY_WAVE_QK all =1, plus the attention-parity flag =1,
    since renamed `NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE`), the
   ordinary arm reproduces the sealed reference exactly AND K5/W6 (W6 WMMA) matches the ordinary
   arm for all 28 tokens (96560@17; the pre-fix index-5 109600 is gone). The earlier ordinary-arm
   divergence (118178@17) was a build-config difference (those four candidate flags OFF in the
   plain build), not a source regression. REMAINING: full W6 qualification (FP64 oracle / serial
    W1 / canaries / eager+graph / ISA / route-rejection), W6-vs-fused speed measurement, and
    promotion of the direct route (remove the qualification-only branch) if it passes. Evidence:
    `/tmp/opencode/k5w6-w5w6/` (matched-ordinary.json, matched-k5w6.json, noedit-*.json,
    sealed-ordinary.json).

    CLOSED 2026-09-12 (HEAD 437881fc + uncommitted W5W6 generalization + promotion). All remaining
    work is done and verified:
    (3) FULL W6 QUALIFICATION PASSED (discriminator tool, exit 0, no stderr): route-rejection
    accepts W5/W6 and rejects W4/W7/tree/out-of-range; W6 batched WMMA matches the FP8-Q profile
    oracle to 1.01e-7 (max_abs) and is bit-exact to serial WMMA and to graph-capture replay;
    canaries intact. Diagnostic W6-vs-fused: batched 0.0617 ms vs fused 0.0955 ms (~1.55x win).
    ISA/static check (`check_attention_parity_static.py` on `build/kv_op_qual.s`) PASS.
    (4) SURGICAL PROMOTION: `use_dflash_w5w6_batched_wmma` is now a PRODUCTION route (no flag
    gate) for the exact qualified cell — DFlash target verification, rows 5 or 6, non-tree,
    context 64-8191, G16, token-fastest FP8 keys, feature-fastest INT4 values/FP16 scales. The
    G16/layout predicate is enforced at the routing site in `r9700_full_attention.hip`; G32,
    wrong-layout, tree, and out-of-range cells retain the fused fallback. Workspace planning
    reserves the W5W6 workspace on shape conditions (no flag gate). The umbrella
    `NINFER_R9700_ATTENTION_PARITY_CANDIDATE` flag was renamed to
    `NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE` (CMake cache var, `kTextP129WmmaTailCandidate`,
    bench report field `text_p129_wmma_tail_candidate`, planner/discriminator/bench-support
    tests) and now gates ONLY `use_text_p129_wmma_tail`.
    (5) BOTH FLAG STATES SELECT IDENTICAL W5/W6 ROUTING: discriminator exit 0 on flag=0
    (`build-r9700`) and flag=1 (`build-r9700-w5w6`); `dflash_w5w6_route_is_production=true` in
    both, `compiled_text_p129_candidate_enabled` flips with the flag. W6 numerics clean in both
    (FP8-Q profile 1.01e-7, bit-exact to serial + graph).
    (6) DEFAULT-BUILD K4/W5 AND K5/W6 EXACT WHOLE-TOKEN PARITY (build-r9700, flag=0,
    --whole-pg 129,27, lane 0): ordinary (`--spec mtp --draft-tokens 0`) == K4/W5
    (`--draft-tokens 4 --dflash-verify-width 5`) == K5/W6 (`--draft-tokens 5
    --dflash-verify-width 6`), all 28 tokens. Evidence: `/tmp/opencode/k5w6-w5w6/parity-{ordinary,
    k4w5,k5w6}.json` and `discriminator-{promoted,flag0}.json`.
    SELECTED PREDICATE: production W5W6 batched-WMMA route for the qualified DFlash cell (above),
    fused fallback preserved for G32/wrong-layout/tree/out-of-range; the renamed flag controls
    only Text P129. K5/W6 is now production-admissible.

- [x] `DFLASH-DRAFT-ATTN-LAYER0` Evaluate a kernel-iteration challenger for the DFlash draft
  (proposal) attention `bidirectional_gqa_bf16_kernel` (D128/Hq32/Hkv8/group4/page64, the
  recipe-independent DFlash lane). LAYER-0 REJECTED 2026-09-12 (no challenger implemented).
  Baseline (auto, 200-event median, `ninfer_r9700_bidirectional_gqa_qual`): tree T12/B2 = 0.0921 ms,
  chain T5/B2 = 0.0110 ms; FP64 oracle passes (max_abs 0.00098). The kernel is a serial per-key
  online-softmax (one KV head/block, 4 waves, one query head/wave), occupancy-16, real context =
  the DFlash Full-BF16 cyclic capacity (2048). The verify-side W5W6 WMMA trick does NOT transfer:
  the 16x16x16 WMMA computes 16 query rows (M-dim) at once, but each wave owns exactly one query
  head (group 4), so the M-dim is wasted and the QK dot is still a per-key scalar reduce — no
  issue-cost win. The mechanism that would help is split-KV / flash-tiling (parallelize the
  ~2048-key loop across more waves/CTAs and merge partial online-softmax states), but that is a
  larger, higher-risk rewrite of a single draft-attention Op, not a whole-model bottleneck.
  CONDITIONAL FUTURE HYPOTHESIS (not an active task): if whole-DFlash phase attribution later shows
  the draft-attention Op owns a material whole-inference ceiling, revisit a split-KV/flash-tiling
  rewrite with a full oracle + ISA + timing qualification. Evidence:
  `/tmp/opencode/k5w6-w5w6/dflash-draft-attn-layer0.md`.

- [x] `DFLASH-VERIFY-DOWN-SPLITK` Pursue the verify-stage Q4 GEMM owner (57% of the
  DFlash round per the retained owner-trace) via a fresh Layer-0 mechanism + roofline, NOT another
  blind small-T variant. LAYER-0 BOUND (2026-09-12, unprofiled event timing, auto power, exact
  T=5/6 shapes, standalone probe `/tmp/opencode/q4-verify-roofline/q4_verify_roofline`): the
  production `a8q4g64_linear_wmma32` route (grid=(rows/16,tokens/16), block=32, N16K16 tiled
  Q4G64 weights) achieves **gate_up [34816,5120] = 430 GB/s** (2176 waves, 90.3 MiB, 68% of the
  retained 633 GB/s pure-stream ceiling) but **down [5120,17408] = 230 GB/s** (320 waves, 45.2 MiB,
  **36% of ceiling**). The down GEMM is the weak owner: only 320 waves (vs 2176) cannot hide HBM
  latency, so it sits at 36% of the stream ceiling — a 64% gap, the largest in the family. The
  prior small-T candidate (N34816/K5120 one-row-per-thread) exhausts only that mechanism and the
  gate_up shape; it never touched the low-occupancy down shape.
  MECHANISM (tile/wave remapping = split-K): split the down GEMM's K=17408 reduction across more
  waves to raise occupancy toward the gate_up's 430 GB/s. Projected saving: down at 430 GB/s saves
  ~99 us/GEMM x 64 layers ~= **6.3 ms/round ~= 5.3% of the ~118 ms round** (material, clears the
  whole-inference admission margin). BOUND THE SPLIT-FACTOR SWEEP (e.g. K-split in {2,4,8}); do
  not sweep unbounded. GATES before promotion (stop immediately on any failure): (1) independent
  FP64 oracle at the exact down shape; (2) canaries; (3) gfx1201 ISA/resources; (4) graph/workspace
  safety (Device-Graph address stability, arena lifetime); (5) direct paired unprofiled timing at
  the exact T=5/6 shapes; (6) exact public-token parity (K4/W5 and K5/W6 vs ordinary); (7) matched
  whole-DFlash A/B. Do NOT begin the gate_up remap unless this task closes AND gate_up
  independently proves a material remaining bound.
  OPERATOR QUALIFICATION PASSED 2026-09-19 on the physical R9700 at `auto`: S=8 was best and is
  bit-exact to the incumbent and the independent represented-input FP64 oracle at T=5 and T=6;
  output/partial canaries, alias rejection, two Device-Graph replays, wave32/native-IU4 ISA, zero
  LDS/private/scratch/spills, and occupancy 16 all passed. T5 median fell 0.20543 -> 0.07451 ms
  (upper paired ratio 0.3658); T6 fell 0.21545 -> 0.08024 ms (upper paired ratio 0.3733). Evidence:
  `profiles/bench/r9700-dflash-down-splitk-qualification-20260919/results/summary.json`. Remaining
  before promotion: exact public-token parity and matched whole-DFlash Engine A/B for both K4/W5
  and K5/W6 with Device Graph. Use fresh `--whole-pg 128,64`, the optimized proposal head, and
  balanced control/candidate launch orders; do not add the known-non-equivalent isolated-decode
  diagnostic stream. The candidate remains compile-gated and default-off.
  S=8 WHOLE GATE FAILED 2026-09-19. The sealed C1/P128+G64 production-graph campaign found every
  ordinary/control repetition exact, while every S=8 candidate repetition deterministically first
  diverged at retained token index 6 (`96917` control versus `109600` candidate), identically for
  K4/W5 and K5/W6 and both launch orders; 11/65 public tokens differed. Its whole timing is therefore
  semantically confounded, though per-speculative-round decode remained about 12--13% faster.
  Root cause is FP32 reassociation: the incumbent has one serial 272-group FMA chain, while S=8 has
  eight 34-group chains plus a plain-add reduction. The sparse synthetic qualifier did not expose
  the real dense-activation rounding boundary. Evidence:
  `profiles/bench/r9700-dflash-down-splitk-whole-ab-20260919/results` (sealed failure). Do not
  promote S=8. Screen already-qualified S=2 eagerly next (least reassociation, retained ~1.77x
  operator win); only if exact may it advance to a fresh matched Device-Graph whole A/B. Test S=4
  only if S=2 fails and a concrete rounding-direction rationale remains; otherwise close the
  mechanism default-off.
  CLOSED DEFAULT-OFF 2026-09-19. The reviewed S=2 eager discriminator also failed exact parity:
  ordinary and both controls were exact, while both K4/W5 and K5/W6 candidates changed only token
  index 31 (`100131` -> `101642`). Evidence:
  `profiles/bench/r9700-dflash-down-splitk-s2-eager-parity-20260919/results` (sealed failure).
  Because S=2 is the least-reassociated split and still crosses a real target decision, S=4 has no
  correctness rationale and is not run. Keep the candidate selector default-off; no production
  routing changes. CONDITIONAL FUTURE HYPOTHESIS (not active): an exact-order two-stage design could
  store every per-G64 integer dot in parallel, then replay the incumbent's 272 FP32 FMAs in group
  order. Reopen only after a Layer-0 bound includes the full integer-dot workspace traffic and
  serial reduction and still proves a material whole-round ceiling.

- [x] `DFLASH-DOWN-SCALE-GATHER` Qualify one recipe-independent exact-order T5/T6 verify-down
  challenger for BF16 `[T,17408]` -> fresh A8G64 -> Q4N16K16/G64 `[5120,17408]` -> BF16
  `[T,5120]`. The incumbent gfx1201 schedule executes five/six separately masked activation-scale
  loads followed by `s_wait_loadcnt 0` for every G64 group. Load one scale per participating token
  lane and distribute it with full-wave shuffles, removing four/five serialized load phases while
  preserving every native-IU4 integer dot, FP16 conversion and scale product, the serial 272-group
  FP32 `fmaf` order for each output, and final BF16 rounding. Token scales are 544 bytes apart, so
  this is issue/dependency-wait consolidation, not contiguous coalescing or reduced weight traffic.
  Static ISA/resource inspection must prove the gather/broadcast schedule before physical timing.
  Then require a decoded signed-Q4 FP64 oracle on dense varied inputs, exact A8 codec/status and
  exact incumbent BF16 outputs, guards/immutability and malformed rejection, captured
  poison/stale/finite recovery, explicit stream ordering, and balanced cold complete-boundary T5/T6
  timing across three allocations. Retained direct down medians are `0.205429/0.215447 ms`; a 1%
  whole-round improvement needs about `0.01425/0.01453 ms` saved per call across 64 layers. This is
  credible experiment headroom, not a guaranteed bound or current ownership attribution. Only a
  material direct winner advances to ordinary/K4W5/K5W6 exact public-token parity and matched
  Device-Graph whole A/B. Do not alter production routing or reopen split-K.

  This mechanism cannot establish the user's `>=60 decode-output tok/s` target by itself. Retained
  valid controls are about `16.16/16.35 tok/s` with `1.488/1.524` output tokens per roughly
  `91.19/92.98 ms` round; making every down GEMM free predicts only about `19.07/19.24 tok/s` at
  unchanged acceptance. Final companion recipe/quality work must materially improve acceptance;
  at retained round time, even perfect K4/W5 acceptance is only about `54.3 tok/s`, whereas K5/W6
  can theoretically exceed 60. Recipe-independent kernel work remains useful but must not be
  represented as sufficient for the terminal target.

  CLOSED DIRECT GATE 2026-09-21. The independently reviewed complete-boundary qualification at
  `profiles/bench/r9700-dflash-down-scale-gather-qualification-20260921/attempt-1` passed both
  widths with exact incumbent BF16 and codec results. T5 measured `0.32185949→0.23085950 ms`,
  robust ratio upper `0.72017304`, and conservative 64-call round-saving lower `5.76329845 ms`;
  T6 measured `0.33231950→0.23130000 ms`, ratio upper `0.69679859`, and lower saving
  `6.44497011 ms`. All 48 pairs and every allocation/order subgroup won. FP64 relative errors were
  `0.00171474/0.00166648`; graph recovery, guards, immutability, five finite cases, and 19 malformed
  cases passed. The colder complete boundary includes fresh codec preparation, unlike the older
  warm prepared-activation kernel medians, so their absolute latencies are not cross-compared.
  Result-manifest SHA-256 is
  `172bc3e277821cb3f2977c6edc8e9359a0b48cd15d67d8a7136ce8b6373ccf50`; independent result audit
  reported `SHIP` for exact-token and matched whole-graph preparation. Production is unchanged.

- [x] `DFLASH-DOWN-SCALE-GATHER-WHOLE` Integrate the exact qualified T5/T6 scale-gather route behind
  one build-bound candidate predicate for DFlash target-verify down only; do not reuse or widen the
  rejected split-K selector. Retain identical workspace planning because the route uses only the
  existing A8 scratch. Through one fresh reviewed create-only package, require ordinary controls
  to match across builds, candidate K4/W5 and K5/W6 public tokens to match their corresponding
  controls exactly, and balanced matched Device-Graph whole A/B at C1/P128+G64 with the optimized
  proposal head. Admit production only if both widths remain materially faster at decode-output
  scope; otherwise remove the candidate route and retain the direct evidence as a rejected
  whole-level mechanism. Do not infer the `>=60 tok/s` terminal target from this gate.

  CLOSED CANDIDATE GATE 2026-09-21. The reviewed campaign at
  `profiles/bench/r9700-dflash-down-scale-gather-whole-ab-20260921/results` retained exact 65-token
  ordinary/K4W5/K5W6 parity in all 30 repetitions, exact per-width speculative accounting, and
  unchanged workspace/graph capacity. K4/W5 decode improved `16.29055→17.37589 tok/s` with robust
  ratio upper `0.97047898`; K5/W6 improved `16.38600→17.58698 tok/s` with upper `0.93464897`.
  Whole-output gates and both launch orders also passed; all 12 paired decode and whole comparisons
  won. Result-manifest SHA-256 is
  `78ba0d0a95ff1888adc43d54044fd3b6b59cdfe052487c70d4b28ddb912bed30`; independent audit reported
  `SHIP` for narrow in-mode production promotion. Matched ordinary decode remained about
  `19.82 tok/s`, so this neither selects DFlash as default nor closes `DFLASH-WHOLE`.

- [x] `DFLASH-DOWN-SCALE-GATHER-PROMOTE` Promote only the exact admitted C1 base-Text Verify
  K4/W5 and K5/W6 route. Move the kernel/launcher out of qualification naming into the owning
  Linear Op, make the target-owned A8/Q4/T5/T6/`route_tokens==0` predicate unconditional, retain
  one fresh A8 preparation and existing graph-stable workspace, and remove the temporary selector
  and report fields. Retire the overlapping default-off split-K production selector, kernels,
  partial-workspace planning and report fields because it owns the same cell and failed public-token
  parity; preserve sealed historical evidence. Rebuild linked numerical/ISA qualification and run
  one fresh exact-token production confirmation against the retained candidate/control. Do not
  widen to C2..4 or treat this scoped promotion as the final recipe/acceptance decision.

  CLOSED PRODUCTION PROMOTION 2026-09-21. The owning Linear Op now links the qualified kernel
  unconditionally and the target schedule selects it only for C1 (`route_tokens==0`) base-Text
  Verify, genuine DFlash target verification, layers 0..63, A8/Q4N16K16/G64 T5/T6
  N5120/K17408. It performs one fresh A8 preparation in the unchanged graph-stable activation
  workspace. Ordinary decode, MTP, prefill, C2..4 compact batches, other phases, shapes and
  inventories retain their prior routes. The temporary scale-gather selector/report fields and
  the overlapping failed split-K selector, workspace, kernels and active qualification tools are
  removed; sealed historical evidence remains.

  Fresh linked qualification and exact-token production confirmation are sealed at
  `profiles/bench/r9700-dflash-verify-down-production-confirm-20260921/results`. K4/W5 production
  decode was `17.40551875 tok/s`, with median decode time `0.99829774` of the retained admitted
  candidate and `0.93594169` of its control. K5/W6 was `17.60527624 tok/s`, with ratios
  `0.99896050` and `0.93074366`. Both widths retained exact 65-token output and exact speculative
  accounting; ordinary tokens and workspace capacity were unchanged. The linked direct qualifier
  retained independent numerical/static admission and measured T5 `0.3221000→0.2303795 ms` and
  T6 `0.3316390→0.2307600 ms`. Plan SHA-256 is
  `b7560805d482b6324b50e67f74b3fa5b2cfef1aa069fe6ebd16c925360c3817c`; result-manifest SHA-256 is
  `f981a79b0880185e19965a178265b639a3733f713cc9cae2e4f8dad8acc7839f`; closure SHA-256 is
  `142abd07272548b1ba79140fde80f709e9031eeb1aa73caebcffe565e09107ca`. Independent result audit
  reported `SHIP`. This closes only the scoped kernel promotion. Matched ordinary remains about
  `19.82 tok/s`; final DFlash recipe and acceptance selection remain responsible for the
  `>=60 tok/s` target.

## Durable decision rule (added 2026-09-12)

A failed candidate exhausts only its mechanism and qualified cells, not the attributed owner. When
exact-shape Layer-0 evidence identifies one new mechanism whose conservative bound clears the
whole-inference admission margin, autonomously pursue the single highest-ceiling mechanism through
qualification and matched whole A/B; do not ask the user merely because implementation is
substantial. Do not bundle a secondary mechanism. Ask only when alternatives retain a material
unresolved tradeoff or the product contract must change.

- [x] `DFLASH-TEXT-P129` Close append-versus-fresh P129 parity. The retained combined-selector gate
  has exact public continuations but its final normalized tail differs in 4,908/5,120 BF16 values.
  Its ordinary-append arm captured only the tail, so it cannot localize this failure. Prepare a
  reviewed Text-pair layer-boundary package on the same build and find the first current divergent
  boundary; do not assume the previous layer3 frontier survived the attention change. Evidence:
  `profiles/bench/r9700-dflash-semantic-traces-aebd5f82-20260906/results`,
  `profiles/bench/r9700-qwen3-layer-boundary-traces-43e5e4cc-20260906/results`, and
  `profiles/bench/r9700-fp8-e2-t129-text-parity-gate-20260906/results`, plus the retained gate above.
  CLOSED 2026-09-06 on the retained 6fe53d53 combined-selector build: the reviewed Text-pair
  layer-boundary package
  (`profiles/bench/r9700-text-layer-boundary-traces-6fe53d53-20260906`) ran both arms on the R9700.
  Token-level parity is closed — fresh (whole-pp129+tg1) and append (pp128+tg1) both produce the
  retained P129 ordinary control `[96558, 96917]`. The first visible intermediate divergence is
  layer 13 post_mixer (snapshot 27): 202/5,120 hidden units differ by one BF16 bit (48683 vs
  48682), predecessor boundary exact. The previous layer3 frontier did not survive the attention
  change; the divergence now localizes to layer 13. This is a silent association-order difference
  between the prefill (all-129-in-one-chunk) and append (128-prefix + 1-decode) routes; it does not
  change the generated tokens and does not authorize production routing or performance claims.

- [x] `DFLASH-TARGET-P129` The retained gate closes ordinary-W1 versus DFlash-W5 target semantics:
  all 28 public tokens, the DFlash decision/accept/commit reconstruction, and layer3 attention are
  exact. A supplementary residual trace first differs at layer10 post-mixer (252/5,120 values;
  predecessor exact), which does not invalidate the represented target contract.

- [ ] `DFLASH-RECIPE` [depends: TERMINAL-SELECTION, CHUNK-SELECT] Select DFlash matrices from the
  real BF16 DFlash2 checkpoint rather than inheriting the base recipe. Compare canonical Q4G64,
  source-MSE Q4G64, and source-MSE W8G32 while preserving BF16 selector codebooks and private state.
  Keep the optimized 131072-row Q4G64 head with its exact I32 token map. Admit row-scaled E4M3 only
  after exact-shape speed, DFlash quality, storage, and DFlash-owned prepared Linear execution.
  Rank by qualified acceptance and whole throughput, not matrix MSE. Materialize survivors by
  byte-exactly extending the selected N16 base with the recipe-aware 66-object inventory and bind
  base receipt, BF16 source, recipe, plan, index, and ranking. The old
  `profiles/bench/selected-dflash-prepare-20260905` and fixed-Q4/K1..11 schema-v3 owner are
  historical and non-runnable; create a fresh receipt-bound recipe-aware two-width successor.
  CPU prerequisite audit 2026-09-21: local source
  `/ssdpool2nvme/local_llm/models/qwen3.8-27b-dflash2` validates all 81 BF16 tensors and both
  selector codebooks. `tools/convert/qwen3_8_27b_r9700/dflash2_matrix_recipes.py` already implements
  all three encoders (32 matrices plus 34 unchanged BF16 objects). Recipe-aware conversion,
  receipts, six additional evaluation identities and C++ binder/Variant storage planning are now
  implemented; 14 CPU Python tests (including real encoder byte-oracles) and the isolated host
  registry/execution-workspace tests pass. This is not physical companion admission. Remaining:
  real artifact binding/graph qualification, recipe-aware selected companion preparation and
  selection at K4/W5 and K5/W6. Preserve the base's 131072-row draft head and exact token map byte-for-byte.
  Benchmark admission now validates all nine companion identities against their exact base and
  recipe receipts; schema2 shortlist permits only K4/W5 and K5/W6. Independent review SHIP,
  80 focused benchmark/prefill CPU tests pass, and selected2048 authority revalidates unchanged.
  The selected successor tools are now implemented: `tools/bench/prepare_selected_dflash.py`
  creates three recipe-separated CPU conversions and create-only two-width campaigns;
  `assemble_dflash_selection.py` binds recipe+K/W, declared capacity-eligible C subsets, exact
  public parity, and matched speed. Primary C1 winner/per-C frontiers are evaluation only, not
  runtime recipe switching or all-C production admission. Independent review SHIP; 102 unit and
  14 cutover tests pass, including actual diagnostic writer/validator roundtrips and failed-output
  preservation. Exact commands are in `tools/bench/README.md`. Physical launch still needs the real
  terminal base and the matching existing `build-r9700-phase-sum-{g16,g32,xattention-g16,xattention-g32}-20260921`
  benchmark/planner pair, which supports all nine companions for every base recipe. Old panel
  binaries lack new IDs but stay frozen as selection provenance; no new build is required.
  New evaluator runs both DFlash and matched ordinary/token-parity controls; exact profile mapping
  is in `tools/bench/README.md`. All four configuration/host-only W5/W6 planner checks pass.
  Final cutover rejects superseded evaluations and evaluation-only schema5: separately qualify one
  resident companion and its supported capacity before final admission. Base PPL owns target NLL;
  DFlash evaluation owns acceptance/generated output. Never add companions to base Pareto candidates.
  The explicit single-resident `admit`/`validate-admission` workflow is now implemented and reviewed
  (`a9111d2e`): one recipe/K/W must qualify acrossC1..4; per-C frontier winners cannot be mixed.
  Final cutover recomputes the distinct resident authority, while schema5 remains evaluation-only.
  Schema5 retains the passing C1 screen and a separate C2..4 followup without rerunning C1;
  both raw namespaces, their common corpus and public-token checks are independently replayed.
  A recipe-capable selected-base benchmark may also evaluate companions (`5133c7b6`); different
  executable hashes are not a capability requirement. Exact recipe/identity/parity gates remain.
  W8 feature/QKV/output/conv/selector shapes currently use existing BF16×W8 execution, not
  unqualified A8 routes. Qualify exact DFlash shapes and widths before any speed/accuracy claim.
  The reviewed numerical-only package `profiles/bench/r9700-dflash-companion-ops-20260921`
  now provides CPU-only `commands.sh preflight` and create-only `commands.sh run` for the
  fresh three-qualifier build. It covers42 sampled full-K FP64 Linear cases,16 full-formula
  selector cases and8 convolution cases, each eager plus two poisoned-workspace graph replays.
  W5/W6 and K4/K5 row extents derive from C1..4 call sites; feature/QKV also cover chunk2048.
  The selector oracle now retains full FP64 projection/Markov arithmetic rather than copying
  private BF16 staging. All three targets compile; six CPU tests and independent review SHIP.
  Physical qualification now PASS: all66 cases completed with empty stderr; Linear maximum
  normalized error0.148498 (limit1), convolution maximum relativeL2 error0.00257126.
  Independent result audit SHIP. Retain `physical/result.json`; never rerun this completed package.
  This admits represented-W8 public Ops only, not real companion binding, recipe quality or speed.
  Python 3.11 CPU conversion dependencies are available without installation via
  `PYTHONPATH=/ssdpool2nvme/local_llm/ninfer/out/numerical-reference-venv/lib/python3.11/site-packages`
  with `/home/battlefront/.local/bin/python3.11` (explicit CPU only; torch is CUDA, not ROCm).
  The separate Python 3.11 ROCm reference environment is now installed and qualified; see QUALITY-8K32K.

- [ ] `DFLASH-QUALITY` [depends: DFLASH-RECIPE, DFLASH-TEXT-P129] For each
  surviving companion retain aligned target/draft outputs, deterministic proposals and final target
  tokens, exact ordinary-target parity, per-position acceptance, accepted drafts/round, repair and
  fallback counts, and exact artifact/profile provenance under the selected cache group. Base-model
  PPL owns target NLL; synthetic operator error cannot select a recipe.

- [ ] `DFLASH-WHOLE` [depends: DFLASH-QUALITY, DFLASH-SCHEDULE, DFLASH-VERIFY-DOWN-SPLITK] Advance only a valid, materially
  faster C1 K4/W5 route, then retain fresh recipe-aware K4/W5 and decision-relevant K5/W6 evidence
  at C1..4: exact output, resolved W, per-position acceptance, fallback/repair, prefill/graph-decode
  throughput, graph startup/replay, resolved workspace and fixed-family graph allocation, and
  capacity/headroom. A capacity failure excludes that exact recipe/K/W/C cell. Compare with exact
  spec-none controls; no matrix result or manual frontier closes this task. After final base/artifact
  dependencies, the user's C1 DFlash optimization target is at least `60 decode-output tok/s` in
  matched whole inference. Admission first requires a material win over the current production
  base and exact public greedy-token parity; 60+ is the optimization target, not permission to
  waive either gate. This work targets DFlash, not MTP.

## Terminal-selection chain

The former `DENSE-FLOOR-DECISION` prerequisite is satisfied. Execute these tasks after the bounded
prefill review, preserving their remaining data dependencies.

- [ ] `WHOLE-MATRIX` [depends: CHUNK-SELECT, QUALITY-8K32K, CAPACITY-WHOLE-12] Retain matched
  schema-v21/spec-none ordinary 8K+256 and 32K+256 whole reports for each quality- and
  capacity-eligible profile at every C1..4. A measured quality or capacity failure is a retained
  exclusion with no whole objective.
  Require `phase_timing_semantics=serial-lane-service-sum_shared-decode-max_v1` for corrected
  serial-prefill accounting; retained schema20 C>1 prefill cannot supply selection objectives.
  Completed schema20 C1 chunk and capacity evidence remains valid and replayable without reruns.
  Reopen exact commands, artifacts, executables, receipts, planner bytes, and `auto` endpoints.
  MTP rows are optional regression diagnostics and never rank or block the base selection.

- [x] `CHUNK-PHYSICAL-12` [depends: DENSE-FLOOR-DECISION, PREFILL-CHUNK-ATTENTION] Create fresh receipt-bound N16/K16
  no-overwrite screen/finalist/campaign owners and run all twelve candidates at C1 over aligned
  chunks 1024, 2048, 4096, and 8192 at 8K, then both global finalists at 32K. The old
  `profiles/bench/prefill-chunk-screen-twelve-candidate-20260905` is non-runnable history.
  PAUSED predecessor: `build-r9700-selection-{dense,xattention}-g{16,32}-20260921` and
  `profiles/bench/r9700-chunk-selection-receipt-bound-n16k16-20260921` retain two measured cells
  and the discovered attention fallback; do not rebuild those frozen binaries or resume it.
  Prepared successor `profiles/bench/r9700-chunk-selection-panel-attention-20260921` requires
  admitted `PREFILL-CHUNK-ATTENTION` evidence and fresh
  `build-r9700-selection-panel-{dense,xattention}-g{16,32}-20260921` binaries. It owns read-only preflight,
  explicit input freeze, preparation, screens, finalists and selection through `commands.sh`.
  Its README gives the exact ordered commands. The shared runner now checks the R9700 PCI
  identity and matching HIP ordinal rather than unstable DRM card numbering. No timing is
  credited until each report passes the existing matrix and selector validation.
  All 48 C1/8K screens and 24 C1/32K finalists completed successfully on 2026-09-21.
  Independent reconstruction and frozen-input review reported SHIP. Do not rerun this completed
  campaign or rebuild its binaries; its results admit chunk selection, not numerical quality.

- [x] `CHUNK-SELECT` [depends: CHUNK-PHYSICAL-12] Publish one shared startup chunk through
  the successor `r9700-chunk-selection-panel-attention-20260921` package, maximizing the worst normalized
  throughput across the complete candidate/prompt objective set, then workspace and smaller-chunk
  tie-breaks. Bind each N16 migration receipt and the four-role planner identity. Current 4096 rows
  remain diagnostic unless selected.
  Publish to `profiles/bench/prefill-chunk-selection-panel-attention-20260921.json`;
  the 20260905 selection pipeline remains historical and is never resumed.
  Selected 2048: global maximin 0.9959696531 versus 4096's 0.8884210782; maximum workspace
  608,387,072 versus 813,924,352 bytes. It wins all twelve 32K pairs. Independent review SHIP.
  Fresh chunk2048 BF16 references are required; chunk4096 references cannot be substituted.

- [ ] `XATT-KEEP-DIST` [depends: CHUNK-SELECT] Validate the redesigned sparse consumer on real
  8K/32K model keep distributions and rerun affected whole evidence. Synthetic concentration and
  standalone operator fixtures cannot replace this route-level evidence.
  Real-model distribution evidence now PASS, independent review SHIP:
  `profiles/bench/r9700-xattention-real-keep-distribution-20260921`. All8 eligible sparse
  recipe/group/length cases cover1,280 dispatches and491,520 head/query-block slots. Mean keep
  fractions are about60.8%/51.4% for all-Q4 and59.3%/49.9% for four-role at8K/32K.
  These traced runs are timing-ineligible; do not rerun them. Only affected uninstrumented whole
  evidence remains for this item; the phase-sum whole successor owns it.

- [ ] `Q4-CTA-TERMINAL` [depends: CHUNK-SELECT] Retain rebuilt selected-route attribution and whole
  evidence for the promoted production-extent A8Q4 prefill CTA; historical direct screens do not
  establish the terminal artifact/profile result.

- [ ] `W8-CTA-TERMINAL` [depends: CHUNK-SELECT] Retain rebuilt mixed-artifact attribution and whole
  evidence for the target-specific A8W8G32 prefill CTA, preserving its separate exact fallback and
  crossover semantics.

- [x] `QUALITY-8K32K` CLOSED 2026-09-21. All three recipes, dense/XAttention and G16/G32
  completed paired BF16-source 8K/32K quality at shared chunk2048. Fresh BF16 A/B sidecars match
  exactly after full-span GDN FP64 qualification; PPL is6.463887635/5.632510588. Twenty-two of24
  cells pass, yielding ten eligible profiles. Mixed XAttention fails strict accuracy at32K:
  G16/G32 introduce19/18 new severe positions against budget17 despite passing mean-NLL deltas.
  Both exclusions and all raw sidecars are retained in `df325432`; no tier changes or reruns.
  Quality-exclusion publication/selection repair `1e3c0f5e` passed independent review and133
  focused tests. All24 actual cells and the complete authority map replay successfully.
  Published authority:
  `profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921/quality-authorities-receipt-bound-n16k16.json`.
  The package retains exact commands, isolated CP311 ROCm setup, reference repeat and failed
  original stage exit. Do not rerun reference, quality or publication. Continue terminal capacity.

- [ ] `POSTCHUNK-ON` [depends: CHUNK-SELECT] Through
  `profiles/bench/r9700-terminal-base-panel-attention-20260921`, run both ON-profile capacity matrices
  with the selected chunk and execute each eligible schema-v14 whole matrix with the same
  group-specific executable and artifact.
  The fresh package provides preflight/capacity/whole/select commands, binds the published six
  quality authorities and frozen panel artifacts/builds, collects all twelve capacity outcomes,
  and runs only quality- and capacity-eligible whole profiles before schema-v7 publication.
  Whole reports retain public tokens without changing timed work. After whole, run its reviewed
  `controls.sh preflight` and `controls.sh run`: exact ordinary8K/32K graph/eager tokens atC1..4
  are required before select can create output. All graph repetitions/lanes are compared with one
  eager execution; failure outputs are create-only. Controls review SHIP; eight CPU tests pass.
  Root review SHIP after
  repairing tuple/list capacity identities; 25 focused CPU checks pass. No physical result is
  implied. Old post-chunk/terminal 20260905 launchers are historical; do not republish chunk selection.
  STOPPED: all eight all-Q4/mixed profiles completed C1..4 capacity, but four-role C2+ fails
  unstructured hipBLASLt OOM. C1 planned slack5,244,103,936 bytes becomes1,314,914,304 after startup.
  All144 FP8 projections create separate library handles after capacity planning; this is an
  ownership/accounting defect, not a valid capacity exclusion. `capacity/closure.json` seals the
  original package, interrupted during the last hybrid profile's C2. Do not resume it or run its
  whole/select stages. Repair one explicit shared target-owned FP8 library context created before
  the authoritative free-memory snapshot, qualify actual allocation/graph behavior, then use a
  fresh recovery package. Reuse the eight valid capacity matrices and numerical/chunk evidence
  when the resource-only change preserves arithmetic; bind fresh hybrid benchmark/planner bytes.
  Recovery implementation `7d4eb810` shares one loaded-target context before the final capacity
  snapshot; all four fresh recovery builds match their original compiled profiles. Independent
  source and physical qualification review SHIP: public-input FP64 sampled outputs match exactly,
  two poisoned graph replays pass, and all ten startup algorithm fingerprints remain unchanged.
  Context allocation is172MiB, two-execution preparation8MiB, later execution6MiB; this does not
  establish full-model startup accounting. Retained proof is under
  `profiles/bench/r9700-terminal-base-fp8-context-recovery-20260921/qualification`.
  The single-model check in `profiles/bench/r9700-fp8-context-model-startup-20260921` must establish
  actual startup free memory at least planned slack before fresh hybrid capacity proceeds.
  First startup succeeds but its strict accounting gate fails: actual free5,213,519,872 versus
  planned slack5,214,743,808 bytes. The1,223,936-byte shortfall reconciles exactly to8MiB library
  preparation plus3,321,088 bytes of arena rounding minus10MiB unused graph reserve. Preserve this
  diagnostic; finish explicit physical-cost accounting before the successor capacity campaign.
  Accounted successor now PASS: loaded-target preparation precedes capacity resolution; workspace
  binding allocates zero bytes, and a hybrid-only4MiB bound covers two arenas' allocation rounding
  without changing nonhybrid plans. Fifteen prebind/bind-first algorithm fingerprints match;
  public-input FP64 sampled eager/graph outputs match exactly. The fresh `qualification-accounted`
  and `startup-accounted` results retain actual free5,213,519,872 versus planned5,202,160,896 bytes.
  The11,358,976-byte surplus equals unused graph reserve plus the rounding bound minus actual
  arena rounding. No unexplained startup allocation remains. Do not rerun either completed proof;
  The reviewed bridge is frozen and all16 new hybrid capacity cells pass. Combined with32
  retained nonhybrid cells, `capacity/validation.json` records48/48 successes across12 matrices.
  Quality intersection still admits only10 whole matrices; mixed XAttention remains excluded.

- [ ] `ALLQ4-PAIRS` [depends: CHUNK-SELECT] Complete current dense and XAttention G16/G32 capacity
  and whole pairs and admit all four all-Q4 candidates to the same schema-v7 Pareto decision.

- [ ] `MIXED-PAIRS` [depends: CHUNK-SELECT, QUALITY-8K32K] Complete dense and XAttention G16/G32
  mixed-recipe PPL plus fresh exact C1..4 capacity/whole pairs using compile-bound binaries.

- [x] `FOURROLE-CAPACITY` [depends: CHUNK-SELECT] Reconcile the admitted four-role 8K/32K quality
  with fresh selected-chunk C1..4 capacity. The old report's feature-materialization accounting is
  invalid; its failed cells remain retained. All four accounted G16/G32 dense/sparse matrices
  now pass C1..4 with actual startup free memory above planned slack; recovery proof and capacity
  reports reside in `profiles/bench/r9700-terminal-base-fp8-context-recovery-20260921`.

- [ ] `CAPACITY-WHOLE-12` [depends: POSTCHUNK-ON, ALLQ4-PAIRS, MIXED-PAIRS, FOURROLE-CAPACITY]
  Complete the twelve schema-v14 capacity outcomes and corresponding whole matrices only for
  capacity-eligible profiles, with symmetric dense/XAttention eligibility inside each
  recipe/cache pair.

- [ ] `MODEL-EVIDENCE` [depends: QUALITY-8K32K, CAPACITY-WHOLE-12] Complete real FP8-K/INT4-V 8K
  and 32K paired quality, diagnostic greedy-token, graph/eager, and spec-none ordinary phase
  evidence for the remaining candidates. DFlash evidence is selected-only and separate.

- [ ] `XATT-ADMISSION` [depends: XATT-KEEP-DIST, QUALITY-8K32K, CAPACITY-WHOLE-12] Complete native
  gfx1201 B128/S16/tau900 admission for both G16/G32 typed-cache instances against each matching
  dense control. Preserve ordinary decode and DFlash verification on dense attention. Require
  route-level PPL, exact shape/ISA/resources, capacity, and whole inference for admission;
  `SELECTED-NIAH` remains a post-selection cutover gate.

- [ ] `WHOLE-PARETO` [depends: WHOLE-MATRIX, MODEL-EVIDENCE, XATT-ADMISSION] Select one G16/G32 and
  dense/B128-S16-tau900 static profile only from complete matched quality, capacity, phase, and
  whole evidence. Do not infer it from an isolated operator sweep.

- [ ] `TERMINAL-SELECTION` [depends: WHOLE-PARETO] Publish exactly one schema-v7 artifact, cache
  group, static Text-prefill profile, and selected chunk through
  `profiles/bench/r9700-terminal-base-phase-sum-20260921`. Its reporting-only bridge retains
  all48 completed capacity cells and the numerical/chunk authorities. The preceding FP8 recovery
  whole attempt is sealed for its schema20 concurrent-prefill reporting defect; the panel-attention
  package and20260905 launcher are historical.
  Preserve every measured exclusion and do
  not materialize the final artifact or cut over XAttention here.

## Post-selection, conditional, and final gates

- [ ] `LOWCTX-LADDER` [depends: TERMINAL-SELECTION] Run the exact selected dense C1/spec-none/auto
  ladder at P=128,512,1024,2048,4096 using the selected chunk through
  `tools/bench/run_ninfer_bench_matrix.py --preset low-context-prefill` and
  `tools/bench/validate_low_context_prefill.py`; use fresh
  `profiles/bench/r9700-selected-low-context-20260921` and its sibling `-evaluation.json`.
  Exact selected-route commands are in `tools/bench/README.md`. Record progress toward 2,000+ tok/s at
  P2048; this performance target no longer blocks accuracy, DFlash, or artifact admission.

- [ ] `PREFILL-TAIL-CONDITIONAL` [depends: SELECTED-PROFILE] [if: selected profiling shows
  nonqualified final-chunk Linear fallback is material] Qualify arbitrary-tail extensions of the
  promoted A8Q4/A8W8 CTAs. Otherwise close this task without an operator campaign. Preserve exact
  production token predicates until resolved.

- [ ] `SELECTED-PROFILE` [depends: TERMINAL-SELECTION] Profile only the selected 8K/32K prefill,
  ordinary decode, and admitted speculative route. First separate Text, proposal, verification,
  host gaps, copies, and dominant kernels; then use focused counters. Verify actual native INT4,
  FP8, or other expected ISA, near-max useful memory streaming/bandwidth where applicable, no
  material stalls, and L2/TCP behavior. Use `profiles/bench/post-terminal-selected-hardware-use-20260905`
  plus `profiles/rocprof/selected-ordinary-decode-profile-build-20260905` and
  `profiles/rocprof/selected-ordinary-decode-memory-prepare-20260905`; do not duplicate
  candidate-screen profiling.

- [ ] `XATT-G32-SCALE-CONDITIONAL` [depends: TERMINAL-SELECTION] [if: G32 XAttention survives]
  A/B subgroup broadcast of FP16 value scales against duplicate per-feature loads. Require the
  independent oracle, materially lower VGPR with legal occupancy/LDS and no spills, direct speed,
  and selected whole confirmation.

- [ ] `SELECTED-NIAH` [depends: TERMINAL-SELECTION] Run the selected candidate's 64K five-position
  long-context needle retrieval through `tools/bench/prepare_selected_niah.py` into fresh
  `profiles/bench/r9700-selected-niah-20260921` after its matched model and whole evidence;
  run the generated `commands.sh` (exact preparation command in `tools/bench/README.md`).
  Dense remains active until this passes.

- [ ] `XATT-CUTOVER-CONDITIONAL` [depends: XATT-ADMISSION, SELECTED-NIAH, TERMINAL-SELECTION,
  XATT-G32-SCALE-CONDITIONAL]
  [if: B128/S16/tau900 wins] Promote that leaf and remove the qualification-only branch; if dense
  wins, delete the sparse branch. Update build/report/profile ownership coherently and retain no
  runtime selector.

- [ ] `BF16-PARITY` [depends: TERMINAL-SELECTION] Complete BF16-source model parity for the selected
  integer artifact: retain the completed teacher-forced prefill quality authority and run the
  selected C1 8K/32K decode BF16 comparison using `tools.ppl.prepare_selected_exact_token`
  against `profiles/bench/r9700-terminal-base-phase-sum-20260921/select/result.json`
  (exact preparation command in `tools/ppl/README.md`). The historical focused-verification
  launcher runs host/Op tests, not BF16 model parity, and is not this gate's producer.
  The selected launcher now reuses the bound BF16 prefill reference and fresh-process repeat
  proof: its layer-major formula, chunk spans and half-score positions are identical under the
  decode label. Original reports remain prefill evidence; candidate T1 decode and graph/eager
  comparisons still run fresh. Independent review and64 focused tests pass; all24 retained
  quality cells replay unchanged. Do not rerun the invariant BF16 reference for this gate.
  Per-Op represented-input oracles are already complete but do not replace selected-model evidence.

- [ ] `SELECTED-PARITY-VISION` [depends: BF16-PARITY] Complete selected-artifact decode and
  same-route graph/eager comparisons plus the source-BF16 Vision diagnostic where applicable.
  Use fresh `profiles/ppl/r9700-selected-exact-token-20260921` and
  `profiles/bench/r9700-selected-vision-20260921` prepared from that same published selection;
  retain the selected whole C1..4 graph/eager controls rather than repeating them.
  The Vision producer is an artifact-backed Python/source-BF16 comparison, not C++ Engine Vision
  evidence. MTP comparisons remain optional
  non-ranking diagnostics and Vision remains a finite/shape diagnostic without a numeric threshold.

- [ ] `FINAL-ARTIFACT` [depends: TERMINAL-SELECTION, LOWCTX-LADDER, SELECTED-PROFILE,
  PREFILL-TAIL-CONDITIONAL, SELECTED-NIAH, SELECTED-PARITY-VISION, DFLASH-WHOLE] Through
  `profiles/bench/final-artifact-cutover-admission-prepare-20260905`, join the selected hardware-use,
  low-context, conditional-tail, NIAH, parity, and DFlash authorities for the same winner before
  promoting the selected converter identity and materializing one fresh final base artifact plus
  its selected DFlash companion. Remove evaluation identities and alternate project-owned recipe
  paths only after this pre-promotion admission passes.

- [ ] `E2E-PROFILE` [depends: FINAL-ARTIFACT, SELECTED-PROFILE] Reuse the twelve base capacity
  outcomes rather than duplicating them, then retain final C1..4 complete-inference throughput and
  focused attribution for the final base and DFlash companion. Recipe-independent operator traces
  are not final-artifact evidence.

- [ ] `FINAL-SUITE` [depends: FINAL-ARTIFACT, E2E-PROFILE, BF16-PARITY, SELECTED-PARITY-VISION,
  XATT-CUTOVER-CONDITIONAL] Run the bounded focused correctness, artifact integration, serving
  schema, real-model inference, PPL/token, graph/eager, speculative acceptance, and speed suite.
  Physical qualifiers and selected-model gates cannot be replaced by CPU preparation.
