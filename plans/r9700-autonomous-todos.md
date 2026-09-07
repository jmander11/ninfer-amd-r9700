# R9700 autonomous execution ledger

Status: live work ledger. Every unchecked item is assigned work. This temporary ledger supplements
the active authorities in `docs/README.md`; completed implementation and experiment history belongs
in `docs/performance.md` and `docs/maintainer/r9700-overhaul-plan.md`, not here.

## Fixed execution and product constraints

- Keep useful independent subagent work active whenever it exists, while the primary agent owns
  serialized R9700 execution and coordinates shared-file edits. Do not invent work to fill a slot.
- The product workload is exactly one R9700 and startup-fixed `C=1..4`. Never schedule or require an
  active `C>4` cell. Retained `C=5..8` rows are historical only.
- Performance-admission timing requires device 0, Radeon AI PRO R9700, `gfx1201`, wave32, and power
  profile `auto`. Profiled timing is attribution-only. Do not monitor temperature or fan speed
  unless observed evidence first indicates throttling.
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

This section is the command-level handoff for an agent that should execute bounded work rather
than redesign the campaign. Run every command from
`/ssdpool2nvme/local_llm/ninfer-amd-r9700`. Never run two GPU processes concurrently. CPU builds,
static inspection, package preparation, and report review may be delegated while the primary agent
owns the GPU. Before any timing run, require device 0 to be the R9700/gfx1201 and the power profile
to be `auto`; do not change clocks, power, fan, or temperature policy. Stop a package on its first
failed prerequisite and retain the failure instead of continuing to timing.

### Current bounded task: attention arithmetic parity candidate

The current source candidate has two independent changes behind the default-off
`NINFER_R9700_ATTENTION_PARITY_CANDIDATE` selector:

1. Dense Text T129/C129 retains dense rows 0..127 and overwrites only row 128 with the existing
   W1/C129 FP8-Q WMMA arithmetic.
2. DFlash K4/W5 target verification uses one three-launch batched W5 FP8-Q WMMA route. The route
   must receive explicit DFlash identity from the DFlash caller; ordinary Text and MTP must remain
   false. Runtime selection and workspace-capacity planning must use the same identity. The exact
   admitted physical profile is G16, token-fastest FP8 keys, feature-fastest INT4 values, and
   feature-fastest FP16 value scales for context 64..8191. Tree verification and every other width
   retain their existing route.

Configure and build the focused selector-on qualifier with:

```sh
cmake -S . -B build-r9700-attention-parity-on -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DNINFER_R9700_ATTENTION_PARITY_CANDIDATE=1
cmake --build build-r9700-attention-parity-on -j4 --target \
  ninfer_r9700_dflash_attention_route_discriminator \
  ninfer_r9700_runtime_planner_qual \
  ninfer_bench_support_test
```

Run the CPU/static checks before occupying the GPU:

```sh
git diff --check
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 \
  tools/r9700/test_check_attention_parity_static.py
ctest --test-dir build-r9700-attention-parity-on --output-on-failure \
  -R 'ninfer_bench_support|attention_parity_routing|runtime_planner'
```

Then run exactly one focused GPU process and capture its JSON rather than copying terminal text:

```sh
mkdir -p profiles/bench/r9700-attention-parity-candidate-focused-20260906/results
build-r9700-attention-parity-on/src/ninfer_r9700_dflash_attention_route_discriminator \
  > profiles/bench/r9700-attention-parity-candidate-focused-20260906/results/qualifier.json
```

Do not interpret the JSON as a promotion by itself. It must establish all of the following before
the source is committed as a qualified candidate: exact oracle/status/canary checks; dense Text
rows 0..127 unchanged; overwritten Text row 128 bit-exact to standalone W1/C129; batched W5 all
rows bit-exact to serial W5 WMMA; eager/graph replay identity; exact G16/static gfx1201 resource
contract; no ordinary/MTP/tree selector escape; and a consistent DFlash-only runtime/capacity
predicate. Workspace poison is not a semantic output: the dense score plane writes only its causal
triangle, so check outer canaries and complete output, not full scratch overwrite.

If the focused qualifier passes, make and push one WIP checkpoint before preparing model runs:

```sh
git status --short
git diff --check
git add CMakeLists.txt bench src tests tools plans/r9700-autonomous-todos.md
git commit -m "wip: qualify text and dflash attention parity candidate"
git push origin experimental
```

Inspect the staged set before committing and omit unrelated user-owned changes. Do not promote the
selector default in this checkpoint.

### Whole-model parity gate after the focused candidate

The prepared package is
`profiles/bench/r9700-attention-candidate-combined-whole-parity-gate-template-20260906`. It is
currently intentionally non-runnable. This command validates only its structure and must exit
successfully without touching the GPU:

```sh
bash profiles/bench/r9700-attention-candidate-combined-whole-parity-gate-template-20260906/commands.sh \
  --validate-template
```

Before any model load, replace the package's unbound placeholders with the committed source/tree,
a fresh selector-combination build receipt, benchmark hash, linked ROCm identities, exact prior
authority objects, and a reviewed prepared-execution closure. The combined build must enable
FP8-prefix-e2, GDN-wave-QK, DFlash-MLP-down-T5, DFlash-RMSNorm-rows56, and attention parity; it must
leave DFlash-small-T off. Convert `run.py` from a refusal stub only after its preflight enforces
device0 R9700/gfx1201/wave32, `auto`, absence of profiler/injection/remapping variables, exclusive
output creation, and exact artifact/corpus hashes. Preserve three and only three model loads:

- ordinary fresh P129/G27, shared by Text-fresh and target-ordinary comparisons;
- ordinary append P128/G27 with isolated prompt/decode;
- DFlash fresh P129/G27 at K4/W5.

The whole gate passes only if the full 28-token ordinary-fresh sequence exactly equals both the
ordinary-append and DFlash-fresh sequences, the Text tail traces are exact, and the DFlash decision
trace reconstructs every licensed token, accepted position, fallback, frontier transition, and
report counter. Any public-token mismatch keeps timing inadmissible and returns to the first
divergent recorded boundary; do not launch a speed sweep.

### First admissible DFlash speed screen

Only after the whole-model parity gate passes, create a fresh immutable package with matched
selector-off ordinary and selector-on DFlash executables from the same source commit. Run C1,
P129+G27, K4/W5 first, with one warmup and the minimum repetitions needed for a directional screen.
Retain token IDs and require exact parity again. Compare engine decode throughput, target-verify
time, proposal time, acceptance per round and by draft position, fallbacks, and graph/eager
identity. Do not run K5/W6 unless K4/W5 is valid and the result could change the decision. Do not
run C2..4, long generation, chunk selection, capacity matrices, or profilers at this stage.

Every future physical experiment must be launched through a reviewed package-local `commands.sh`,
not an ad-hoc reconstructed command. A package is runnable only when its plan contains no
`UNBOUND`, its runner has a non-mutating preflight mode, and `commands.sh --validate-template` or
the package-equivalent validation passes. Record the exact invocation in that package's plan so
the next agent never has to infer CLI flags from prose.

## Active now: DFlash semantic and schedule work

- [ ] `DFLASH-SCHEDULE` Finish recipe-independent
  schedule and exact-shape work for only K4/W5 and K5/W6. Retain the delivered K1..11 input
  contract but no broad shortlist. The qualified packed-Q4 route remains limited to N34816/K5120
  at T=4,5,6,8,10,12,18,20; T15/T16/T24 and every unlisted shape stay on WMMA. The T5
  N5120/K17408 MLP-down candidate is standalone-qualified but remains off by default. Require
  current-companion whole A/B, token parity, and a material end-to-end win before any route change.
  The exact T5/T6 FP8 target gate/up challenger is terminally rejected at 3.77x/3.64x incumbent
  time; do not rerun it without a changed source mechanism. The K5120 one-CTA RMSNorm route is
  standalone-qualified at rows5/6 near 0.083x incumbent time and integrated behind an off-by-default
  exact compile selector. Its clean matched builds pass 108-case eager and rows5/6 graph numerical
  gates in both selector states at at most one BF16 step, but selector-on fails the exact P129 token
  gate earlier at index21 and remains off. Use the layer1 normalized-input trace to distinguish the
  expected row-independent CTA equality from downstream GDN state/record behavior; require exact
  parity and matched whole evidence before admitting that static width set.

- [ ] `DFLASH-TEXT-P129` Localize the shared Text append-versus-fresh dependence for the identical
  effective P129 history. The tail differs in 4,978/5,120 BF16 elements; exact boundary tracing is
  equal through layer0 MLP and first differs after the layer1 GDN mixer in 142/5,120 elements. Now
  detail tracing is exact through h, controls, and q/k/v/z, then first differs at layer1 recurrence
  output o (106/6,144 elements). Compare the FP32 recurrent state immediately before the selected
  transition to distinguish state input from wide-prefill/append recurrence arithmetic. The exact
  selector-one combined build now makes both layer-zero and layer-one frontier states and every
  captured layer-one GDN boundary byte-exact; its first visible residual difference is instead
  layer-three post-mixer (3,467/5,120 BF16 elements, first hidden 0 bits 48413 versus 48414, exact
  predecessor). This is consistent with the e2 T129 choice removing the earlier layer-zero/layer-one
  path difference, but the combined build is not a matched isolated-causality test. Text now converges
  with the target result on the first full-attention layer as the remaining arithmetic owner.
  Evidence is under `profiles/bench/r9700-dflash-semantic-traces-aebd5f82-20260906/results` and
  `profiles/bench/r9700-qwen3-layer-boundary-traces-43e5e4cc-20260906/results`, and the current
  selector-one result is under `profiles/bench/r9700-fp8-e2-t129-text-parity-gate-20260906/results`.

- [ ] `DFLASH-TARGET-P129` On the fresh-P129 common prefix through generated index 26, localize the
  first ordinary-versus-DFlash difference. Retain target-verification top-two/argmax by column,
  accepted column, licensed tokens, and next frontier so target arithmetic, acceptance selection,
  and post-commit state remain distinct. Target logits already differ at the first generated step;
  exact boundary tracing is equal through layer0 MLP and first differs after the layer1 GDN mixer
  in 1,219/5,120 elements. The rows5 CTA candidate changes the generated trajectory at index21 and
  does not restore overall parity, but selector-on makes every captured layer1 GDN field byte-exact
  and moves the first visible target residual difference to the layer3 full-attention mixer
  (2,796/5,120). The exact two-arm layer3 trace under
  `profiles/bench/r9700-qwen3-layer3-attention-trace-617672eb-20260906/results` now proves the
  selected column is byte-exact through input, RMSNorm, q/gate/k/v projection, q/k normalization,
  RoPE, causal visibility, and canonical cache positions 0--129; the first difference is
  attention_fp32 element 0 (maximum absolute difference 0.02367246150970459). Qualify the ordinary
  W1 and DFlash W5 full-attention arithmetic directly against the independent represented-input
  oracle and identify the route/reduction difference; pairwise parity remains supplementary, but
  the final public greedy-token gate remains exact. Eager already equals graph; that does not waive
  parity. The default-off combined attention candidate has now passed its focused physical
  qualifier: batched W5 is all-row bit-exact to serial W5 WMMA and graph replay, its selected row0
  is exact to ordinary W1, and it measures a diagnostic `0.06228868 ms` versus `0.09549904 ms` for
  incumbent fused W5. The Text P129 dense prefix is unchanged and its overwritten tail is exact to
  append W1. The concise result is
  `profiles/bench/r9700-attention-parity-candidate-focused-20260906/summary.json`. This is not yet
  public-token or whole-inference evidence; execute the three-load exact parity gate described
  above before any speed screen or route promotion.

- [ ] `DFLASH-RECIPE` [depends: TERMINAL-SELECTION, CHUNK-SELECT] Select DFlash matrices from the
  real BF16 DFlash2 checkpoint rather than inheriting the base recipe. Compare canonical Q4G64,
  source-MSE Q4G64, and source-MSE W8G32. Admit row-scaled E4M3 only after exact-shape speed,
  DFlash quality, and DFlash-owned prepared Linear execution exist. Materialize survivors by
  byte-exactly extending the selected N16 base with the recipe-aware 66-object inventory and bind
  base receipt, BF16 source, recipe, plan, index, and ranking. The old
  `profiles/bench/selected-dflash-prepare-20260905` and fixed-Q4/K1..11 schema-v3 owner are
  historical and non-runnable; create a fresh receipt-bound recipe-aware two-width successor.

- [ ] `DFLASH-QUALITY` [depends: DFLASH-RECIPE, DFLASH-TEXT-P129, DFLASH-TARGET-P129] Retain matched
  quality for every surviving companion
  under the selected cache group: aligned target/draft outputs, deterministic proposal and final
  target tokens, exact ordinary-target parity, and exact artifact/profile provenance. Base-model
  PPL owns target NLL; synthetic operator error cannot select a DFlash recipe.

- [ ] `DFLASH-WHOLE` [depends: DFLASH-QUALITY, DFLASH-SCHEDULE] Retain K4/W5 and K5/W6 acceptance,
  exact output,
  capacity, and whole-inference evidence at C1..4: resolved W, per-position acceptance, fallback and
  repair, prefill/graph-decode throughput, capacity/headroom, and focused attribution when needed.
  Select only with fresh recipe-aware two-width evidence and a material conservative win over the
  exact spec-none ordinary controls; no individual matrix or manual frontier closes this task.

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
