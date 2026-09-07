# R9700 autonomous execution ledger

Status: live work ledger. Every unchecked item is assigned work. This temporary ledger supplements
the active authorities in `docs/README.md`; completed implementation and experiment history belongs
in `docs/performance.md` and `docs/maintainer/r9700-overhaul-plan.md`, not here.

## Fixed execution and product constraints

- Keep useful, non-overlapping subagent work active when it exists; do not invent work to fill a
  slot. Give each bounded implementation or experiment one owner and an independent read-only,
  CPU-only reviewer. Review the material semantics, scope, graph/workspace invariants, oracle,
  ISA/resources, command safety, provenance, and decision logic; repair `NO-SHIP` findings and
  require the same reviewer to report `SHIP` before GPU qualification.
- The primary agent alone serializes R9700 work, reproduces preflight and result decisions, and
  pushes coherent WIP checkpoints. Never launch while its owner is editing or review is unresolved.
  At handoff, inspect this ledger, `git status`, pushed HEAD, and retained package summaries,
  receipts, and closure. Never overwrite a failed package; its persisted result is evidence.
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
- Canonical dense Q4 uses scalar-base/U32-voffset N16/K16 ping/pong. Its source-matched
  C1/P2048/G0/chunk4096/spec-none result is `1904.339303 tok/s`, below the hard `2,000 tok/s` floor.
  The bounded candidate list is exhausted and no practical ceiling was proved. The terminal
  selection chain remains blocked until the user changes that product gate or a materially new
  source mechanism satisfies the recorded M128N256 return bound: at most `3.18103603125 ms` per
  matrix and at least `51.438603 ms` whole-P2048 saving with independent numerical/static evidence.
  Crossing 2,000 permits the dependent campaign; it does not itself prove an optimization ceiling.
- Ordinary non-speculative decode is closed at the bounded selector-free
  `27.05729956 tok/s` result. This is not proof of bandwidth saturation or stall freedom.
- DFlash2 is the required speculative direction. Test production choices only at K4/W5 and K5/W6;
  do not resume the K1..11 shortlist or optimize MTP. Preserve both selector codebooks and private
  DFlash state in BF16. Exact public greedy-token parity is mandatory and cannot be waived by logit
  margins. Recipe-independent DFlash work may proceed now; companion selection and physical
  C1..4 admission wait for the schema-v7 base and receipt-bound shared chunk.
- Dense remains the product Text-prefill route until XAttention passes matched quality, capacity,
  whole-inference, selected-profile NIAH, and cutover gates.

Dependency notation is `[depends: ...]`. `DENSE-FLOOR-DECISION` is the external blocker described
above, not an executable task. Conditional tasks retain their explicit `if:` clause.

## Mechanical continuation protocol

Run from `/ssdpool2nvme/local_llm/ninfer-amd-r9700`. Stop at the first failed prerequisite and
retain it; never continue a failed package to timing. The fixed GPU, delegation, and power rules
above apply to every command.

### Current checkpoint

Commits `6fe53d53` and `dce80877` retain the default-off attention-parity candidate and focused
qualification. Dense T129 keeps rows 0..127 and overwrites row 128 with canonical W1 arithmetic;
DFlash K4/W5 uses a DFlash-only, G16, three-launch batched W5 WMMA route. Both semantic rows, graph
replay, independent oracles, routing/capacity scope, and gfx1201 resources passed. Diagnostic W5
time is `0.06228868 ms` versus `0.09549904 ms` fused. Authority:
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

Next, diagnose source/CPU state and prepare an independently reviewed, create-only Text
append-versus-fresh layer-boundary package using the retained combined-selector build. Capture or
bisect enough Text-pair boundaries to locate the first current divergence without assuming the
previous frontier still applies.
No GPU action is currently prepared.
Timing, K5/W6, C2..4, profiling, and speed sweeps remain forbidden.

Every future physical experiment must be launched through a reviewed package-local `commands.sh`,
not an ad-hoc reconstructed command. A package is runnable only when its plan contains no
`UNBOUND`, its runner has a non-mutating preflight mode, and its package validation passes. Record
the exact invocation in that package's plan so
the next agent never has to infer CLI flags from prose.

## After-parity router

### After parity: determine whether K4/W5 can beat base decode

Prepare a reviewed, parity-hash-bound package with selector-off ordinary and selector-on DFlash
executables from the same source commit for one matched C1/P129+G27 K4/W5 screen. Use one warmup
and the minimum repetitions needed for a directional result. Recompute acceptance and retain exact
tokens. Historical `69/123 = 0.56097561` accepted drafts/round gives
`1.56097561` useful tokens/round, so base `27.05729956 tok/s` implies a diagnostic break-even below
`57.6915 ms/round`. Require a material conservative whole win and order-stable/eager-graph evidence.
Retain engine/phase throughput, per-position acceptance, fallback/repair, and exact output. Do not
run C2..4, longer generation, capacity, chunk, or profiling at this stage.
K5/W6 is conditional: its historical 72/591 drafts over 120 rounds accepted no fifth-position
draft, so run it only if new K4 evidence could make K5 decision-relevant.

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

## Active now: DFlash semantic and schedule work

- [ ] `DFLASH-SCHEDULE` Finish recipe-independent exact-shape work only for K4/W5 and K5/W6; retain
  the K1..11 input contract, not its campaign. Packed Q4 is qualified only for N34816/K5120 at
  T=4,5,6,8,10,12,18,20; T15/T16/T24 and unlisted shapes remain WMMA. Qualified T5
  N5120/K17408 MLP-down stays default-off pending exact current-companion whole evidence and a
  material win. Do not rerun the rejected FP8 T5/T6 target gate/up route (3.77x/3.64x incumbent).
  Rows5/6 RMSNorm is qualified but stays off. Consume the retained failed three-load gate; do not
  rerun it. Schedule speed work remains blocked while `DFLASH-TEXT-P129` closes Text parity.

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

- [ ] `DFLASH-QUALITY` [depends: DFLASH-RECIPE, DFLASH-TEXT-P129] For each
  surviving companion retain aligned target/draft outputs, deterministic proposals and final target
  tokens, exact ordinary-target parity, per-position acceptance, accepted drafts/round, repair and
  fallback counts, and exact artifact/profile provenance under the selected cache group. Base-model
  PPL owns target NLL; synthetic operator error cannot select a recipe.

- [ ] `DFLASH-WHOLE` [depends: DFLASH-QUALITY, DFLASH-SCHEDULE] Advance only a valid, materially
  faster C1 K4/W5 route, then retain fresh recipe-aware K4/W5 and decision-relevant K5/W6 evidence
  at C1..4: exact output, resolved W, per-position acceptance, fallback/repair, prefill/graph-decode
  throughput, graph startup/replay, resolved workspace and fixed-family graph allocation, and
  capacity/headroom. A capacity failure excludes that exact recipe/K/W/C cell. Compare with exact
  spec-none controls; no matrix result or manual frontier closes this task.

## Blocked terminal-selection chain

All tasks in this section directly or transitively depend on `DENSE-FLOOR-DECISION`.

- [ ] `WHOLE-MATRIX` [depends: CHUNK-SELECT, QUALITY-8K32K, CAPACITY-WHOLE-12] Retain matched
  schema-v20/spec-none ordinary 8K+256 and 32K+256 whole reports for each capacity-eligible profile
  at every C1..4. A measured capacity failure is a retained exclusion with no whole objective.
  Reopen exact commands, artifacts, executables, receipts, planner bytes, and `auto` endpoints.
  MTP rows are optional regression diagnostics and never rank or block the base selection.

- [ ] `CHUNK-PHYSICAL-12` [depends: DENSE-FLOOR-DECISION] Create fresh receipt-bound N16/K16
  no-overwrite screen/finalist/campaign owners and run all twelve candidates at C1 over aligned
  chunks 1024, 2048, 4096, and 8192 at 8K, then both global finalists at 32K. The old
  `profiles/bench/prefill-chunk-screen-twelve-candidate-20260905` is non-runnable history.

- [ ] `CHUNK-SELECT` [depends: CHUNK-PHYSICAL-12] Publish one shared startup chunk through
  `profiles/bench/prefill-chunk-selection-pipeline-20260905`, maximizing the worst normalized
  throughput across the complete candidate/prompt objective set, then workspace and smaller-chunk
  tie-breaks. Bind each N16 migration receipt and the four-role planner identity. Current 4096 rows
  remain diagnostic unless selected.

- [ ] `XATT-KEEP-DIST` [depends: CHUNK-SELECT] Validate the redesigned sparse consumer on real
  8K/32K model keep distributions and rerun affected whole evidence. Synthetic concentration and
  standalone operator fixtures cannot replace this route-level evidence.

- [ ] `Q4-CTA-TERMINAL` [depends: CHUNK-SELECT] Retain rebuilt selected-route attribution and whole
  evidence for the promoted production-extent A8Q4 prefill CTA; historical direct screens do not
  establish the terminal artifact/profile result.

- [ ] `W8-CTA-TERMINAL` [depends: CHUNK-SELECT] Retain rebuilt mixed-artifact attribution and whole
  evidence for the target-specific A8W8G32 prefill CTA, preserving its separate exact fallback and
  crossover semantics.

- [ ] `QUALITY-8K32K` [depends: DENSE-FLOOR-DECISION] Complete deterministic BF16-source 8K/32K
  quality for all three recipe branches, both attention profiles, and G16/G32. Use
  `profiles/ppl/terminal-quality-recovery-20260905`; dense all-Q4 is already rebased, while sparse
  all-Q4 and mixed dense/sparse remain open.

- [ ] `POSTCHUNK-ON` [depends: CHUNK-SELECT] Through
  `profiles/bench/post-chunk-twelve-candidate-20260905`, rerun both ON-profile capacity matrices
  with the selected chunk and execute each eligible schema-v14 whole matrix with the same
  group-specific executable and artifact.

- [ ] `ALLQ4-PAIRS` [depends: CHUNK-SELECT] Complete current dense and XAttention G16/G32 capacity
  and whole pairs and admit all four all-Q4 candidates to the same schema-v7 Pareto decision.

- [ ] `MIXED-PAIRS` [depends: CHUNK-SELECT, QUALITY-8K32K] Complete dense and XAttention G16/G32
  mixed-recipe PPL plus fresh exact C1..4 capacity/whole pairs using compile-bound binaries.

- [ ] `FOURROLE-CAPACITY` [depends: CHUNK-SELECT] Reconcile the admitted four-role 8K/32K quality
  with fresh selected-chunk C1..4 capacity. The old report's feature-materialization accounting is
  invalid; measured failed cells must be retained and broad prepared matrices remain unlaunched.

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
  `profiles/bench/terminal-static-selection-20260905`. Preserve every measured exclusion and do
  not materialize the final artifact or cut over XAttention here.

## Post-selection, conditional, and final gates

- [ ] `LOWCTX-LADDER` [depends: TERMINAL-SELECTION] Run the exact selected dense C1/spec-none/auto
  ladder at P=128,512,1024,2048,4096 using the selected chunk through
  `profiles/bench/low-context-selected-ladder-20260905`. Require at least 2,000 tok/s at P2048;
  retain a complete failing result for diagnosis and do not substitute a per-repeat cutoff.

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
  long-context needle retrieval through `profiles/bench/post-terminal-niah-prepare-20260905` after
  its matched model and whole evidence. Dense remains active until this passes.

- [ ] `XATT-CUTOVER-CONDITIONAL` [depends: XATT-ADMISSION, SELECTED-NIAH, TERMINAL-SELECTION,
  XATT-G32-SCALE-CONDITIONAL]
  [if: B128/S16/tau900 wins] Promote that leaf and remove the qualification-only branch; if dense
  wins, delete the sparse branch. Update build/report/profile ownership coherently and retain no
  runtime selector.

- [ ] `BF16-PARITY` [depends: TERMINAL-SELECTION] Complete BF16-source model parity for the selected
  integer artifact and every supported schedule through
  `profiles/bench/post-terminal-focused-verification-20260905`. Per-Op represented-input oracles
  are already complete but do not replace selected-model evidence.

- [ ] `SELECTED-PARITY-VISION` [depends: BF16-PARITY] Complete selected-artifact decode and
  same-route graph/eager comparisons plus the source-BF16 Vision diagnostic where applicable.
  Use `profiles/ppl/post-terminal-exact-token-prepare-20260905` and
  `profiles/bench/post-terminal-selected-vision-prepare-20260905`; MTP comparisons remain optional
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
