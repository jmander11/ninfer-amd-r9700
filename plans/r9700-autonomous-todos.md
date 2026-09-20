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
- Canonical dense Q4 uses scalar-base/U32-voffset N16/K16 ping/pong. Its source-matched
  C1/P2048/G0/chunk4096/spec-none result is `1904.339303 tok/s`, below the hard `2,000 tok/s` floor.
  The bounded candidate list is exhausted and no practical ceiling was proved. The terminal
  selection chain remains blocked until the user changes that product gate or a materially new
  source mechanism satisfies the recorded M128N256 return bound: at most `3.18103603125 ms` per
  matrix and at least `51.438603 ms` whole-P2048 saving with independent numerical/static evidence.
  Crossing 2,000 permits the dependent campaign; it does not itself prove an optimization ceiling.
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

Dependency notation is `[depends: ...]`. `DENSE-FLOOR-DECISION` is the external blocker described
above, not an executable task. Conditional tasks retain their explicit `if:` clause.

## Mechanical continuation protocol

Run from `/ssdpool2nvme/local_llm/ninfer-amd-r9700`. Stop at the first failed prerequisite and
retain it; never continue a failed package to timing. The fixed GPU, delegation, and power rules
above apply to every command.

### Current checkpoint

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
BASE-DECODE-BW measurement completed. The BF16 GDN, all-Q4 T1 attention, and all-Q4 C2..4
paired-projection *design* qualifiers completed and admitted production candidates; none of those
later candidates is promoted merely by its design result. The BF16 GDN production symbol is
direct-qualified, but its first whole gate retained an environmental OOM attempt, and device 0 now
requires a maintainer GPU reset before more physical work. The deterministic queue after reset is:

1. run the complete six-role BF16 GDN whole gate from `profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-retry-20260919`;
2. direct-qualify T1 attention with `profiles/bench/r9700-attention-projection-t1-production-qualification-20260919`, then prepare/review/run `profiles/bench/r9700-attention-q4-pair-t1-whole-ab-20260919`;
3. direct-qualify C2..4 with `profiles/bench/r9700-paired-projection-c2c4-production-qualification-20260919`, then prepare/review/run `profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919`;
4. resolve the BF16 GDN and both paired-route promotion decisions, then run `profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919`.

Do not promote or remove any candidate selector anywhere between queue items 1 through 3 because
doing so invalidates downstream cache and source authorities. The maintainer reset lowered device-0
VRAM below both 1 GiB and 5%; the exact next action is
`bash profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-retry-20260919/commands.sh --measure`.
Every other remaining unchecked task depends
directly or transitively on this queue or the external `DENSE-FLOOR-DECISION`.

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

## Active now: selector-free base decode bandwidth

- [ ] `BASE-DECODE-BW` Usefully maximize selector-free base-decode memory throughput on the fixed
  R9700 at `C=1..4`; do not optimize or schedule a `C>4` cell. The retained production baseline is
  `27.05729956 tok/s` at C1/P8192+G256 with Device Graph. Its decoded packed-weight payload divided
  by wall time is `367.956 GB/s`, or `57.9%` of the same-session `635.9 GB/s` stream ceiling. Treat
  this only as a defensible *useful payload rate*: it omits other reads/writes and cache effects and
  is not a physical HBM-bandwidth measurement. Existing attribution makes the T1 native-dot8 Q4
  family the material owner; begin at Layer 0 with its real decode shapes and a wave-cooperative
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
  retained the failure. After process exit, device 0 remained at 83% VRAM with no KFD PID, proving
  stale ROCm allocation rather than candidate execution failure. Preserve this package/results;
  never rerun or append it. Further GPU work requires the maintainer to run
  `sudo /opt/rocm/bin/rocm-smi --gpureset -d 0` and a separate reviewed create-only retry package
  must rerun the complete balanced six-role campaign from the start. That independently reviewed
  package is `profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-retry-20260919`. It rebinds the
  retained failure and all original authorities, reruns all six roles, and refuses plan/results or
  any role unless PCI-derived device-0 VRAM is at most both 1 GiB and 5% with power `auto`. Its
  current preflight correctly fails at 28,413,235,200/34,208,743,424 bytes. After reset and only
  after the low-VRAM preflight passes, run the exact package invocations recorded in its plan.

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
  direct qualification receipt exists.

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
  passes native IU4/wave32 at 57 VGPR, 32 SGPR, and no LDS/scratch/spills. The separately reviewed
  Independent review reported `SHIP` for the whole package at
  `profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919`; no separate review artifact was
  created. Physical
  qualification remains queued behind the required device-0 reset, the GDN whole retry, and the
  T1 attention campaign. Do not prepare the whole package before its direct receipt exists.

  After the admitted paired routes are resolved, the reviewed-ready direct package
  `profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919` is the next bounded mechanism:
  all-Q4 T1 projected-residual fusion for N5120/K6144 and N5120/K17408. These are respectively the 16
  attention-output plus 48 GDN-output projections and 64 MLP-down projections: exactly 128
  residual publications/token. The retained ordinary C1 trace measures the removable residual-add
  family at `0.2114 ms/token` (`0.20956 ms` interval union); fusing the exact
  `delta=BF16(dot); x=BF16(FP32(x)+FP32(delta))` boundary into the native-IU4 epilogue also removes
  2.62144 MB/token of delta store/read traffic. Qualify both complete Linear-plus-residual
  boundaries with distinct actual packed weights, an independent represented-weight FP64 oracle,
  exact residual bits, three-copy balanced cold timing, and exact-symbol native-IU4 static evidence.
  Require `64*(saving_N5120K6144+saving_N5120K17408) >= 0.2 ms/token` before whole C1 A/B; keep
  mixed weights and T>1 on the current path. If admitted, the next ranked mechanism is T1
  RMSNorm+A8G64 preparation fusion. Do not reopen geometry remapping or split-K; an explicit
  gate-up prefetch challenger is later work only if ISA proves early loads and wait scheduling.

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
