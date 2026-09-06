# R9700 autonomous execution ledger

Status: live work ledger. Checked items have direct implementation and verification evidence;
unchecked items remain active work. This temporary file supplements the active authorities listed
in docs/README.md and is removed after the migration is integrated. The original overhaul plan is
a historical migration record, not a live product authority.

Execution rule: keep available subagent slots occupied with useful, bounded, independent work
whenever such work exists. Delegate continuously rather than waiting for the primary thread to
finish a long build or physical measurement. The primary agent retains ownership of serialized
R9700 execution and coordinates shared-file boundaries so delegated work neither contends for the
GPU nor creates conflicting edits. Do not invent low-value work merely to occupy a slot.
Global execution cap: never schedule, generate, benchmark, profile, or require a product cell above
`C=4`. All active matrices are exactly `C=1..4`; retained `C=5..8` rows are historical evidence
only and must not be reused as an active command or product requirement.
Immediate execution priority: scalar-base/U32-voffset addressing is now the sole canonical dense
Q4G64/A8G64 ping/pong route. Its source-matched whole result is `1,904.339303 tok/s`, still below
the 2,000 tok/s hard gate, so do not resume the dependent chunk/capacity/whole campaign. The next
active optimization priority is ordinary non-speculative decode. Native dot8 and the exact K5120
rows1..4 RMSNorm CTA are canonical; the latter's source-matched C1 8K+256 decode median is
`9.467485694 s` (`27.03991411 tok/s`), and its selector-free final smoke is `27.05729956 tok/s`.
The grouped-PV mechanism passed its reduced screen and exact direct product qualification but lost
the C1/P8192+G32 whole screen at decode ratio `1.0038023342` and was removed without a full gate.
Run bounded bandwidth/stall-proxy profiling next and make the ordinary-decode roof decision; do not
open another grouped-PV variant.
After ordinary decode, implement and optimize DFlash2. Do not schedule MTP optimization:
existing MTP support may remain, but its `19.244583 tok/s` MTP3 result is diagnostic only and does
not rank a product route. Dense prefill remains open for the two bounded future mechanisms recorded
below, without reopening an unbounded topology search, FP8 substitution, Q4G128/A8G128
representation change, or XAttention as a dense-floor surrogate.
The 2,000 tok/s P2048 value is an acceptance floor based on an existing llama.cpp observation, not
an optimization target or a performance ceiling for this fixed-model, fixed-R9700 engine. Crossing
it permits the dependent campaign to resume but does not close prefill performance work: continue
until selected-route whole and operator evidence shows the remaining dominant work is near its
relevant compute or memory ceiling and no material avoidable dispatch, dependency, cache, memory-
streaming, or synchronization stall remains.
Do not schedule temperature/fan monitoring unless an observed result first indicates thermal
throttling. Performance-admission timing requires the R9700 power profile to be exactly `auto`;
stable non-`auto` counter captures are attribution-only and their durations are inadmissible.

Current selection contract: the cache is fixed to FP8 E4M3FN keys, signed INT4 values, and FP16
value scales. Cache-only G16/G32 changes 93/94 of 4,095 BF16-greedy choices, so BF16 argmax
identity is diagnostic rather than an admission condition. Quality eligibility requires an
explicit tier plus complete finite aligned sidecars. Accuracy uses delta mean NLL at most 0.02 and
`max(4, ceil(0.001 * scored positions))` new severe positions; capacity-speed uses at most
`ln(1.05)` and `ceil(0.0025 * scored positions)`. Final classification retains the exact
twelve-profile input set after matched 8K/32K quality and resolved C=1..4 capacity outcomes. A
measured capacity failure is a retained exclusion with no whole-inference objective;
`pareto-whole` evidence is required only for capacity-eligible profiles, and dense/XAttention
eligibility must match within each recipe/cache-group pair. Quality is an admission constraint,
not the terminal ranking objective. The classifier first retains one same-recipe cache/execution
winner for each recipe with an eligible pair, then emits exactly one
`terminal_production_selection` across those recipe winners under
`global_maximin_whole_then_capacity_then_quality_then_canonical_v1`: maximize worst-cell
normalized matched spec-none ordinary whole-inference throughput, then normalized capacity, then remaining declared-tier quality budget,
and use canonical artifact/static-profile identity only for a complete measured tie. Selection is
deterministic on the retained per-cell means; raw repetition spread remains evidence but is not a
tolerance or alternate decision rule. Raw scorer wall time and greedy flip rate are not speed objectives.
The deterministic BF16 quality authority is now closed by exact fresh non-trace 8K/32K repeats;
BF16-derived eligibility and schema-v7 selection remain open only for candidate sets not yet
rebased against that authority. The nondeterminism was localized to the attention backend and
removed by fixed-order PV, binding hipBLAS, disabling rocBLAS atomics, and enabling strict
deterministic algorithms before backend import. Dense all-Q4 G16/G32 has been rebased and passes
the capacity-speed tier; mixed and sparse candidate evidence remains outstanding.
XAttention admission additionally compares each compile-bound B128/S16/tau900 candidate with the
matching dense G16/G32 static profile for each eligible weight recipe in that same frontier.
Sparse-only whole matrices can
select a cache group but cannot establish that sparse Text prefill improves whole inference; PPL
scorer wall time and the standalone Op fixture are not substitutes for this dense control.
Independent device sanitizer and complete gfx1201 VALU/LDS/stall counters remain unavailable in the
installed ROCm 10 toolchain/profiler release. Dispatch PMC cache counters are usable when the R9700
is temporarily held in `profile_standard`: the first real-model consumer pass produced positive
GL2C/TCP controls and measured 98.371% aggregate GL2 and 77.692% aggregate TCP hit ratios. Absolute
cache/HBM bytes remain unavailable because the gfx1201 request-size buckets are known-zero.

## Completed foundations

- [x] Select Radeon AI PRO R9700 gfx1201 as the sole new production target.
- [x] Define FP8 E4M3FN K, signed INT4 V, FP16 V-scale cache semantics.
- [x] Implement independent host codec and FP64 attention oracle.
- [x] Add exact argmax sidecars and paired PPL gate machinery.
- [x] Qualify direct gfx1201 FP8 WMMA opcode emission and owned asymmetric raw fragment map.
- [x] Build a BF16-input R9700 KV qualification route with fragmented pages, sentinel checks,
      positions 63/64/65, and in-place tree compaction.
- [x] Extract reusable HIP A2 append codec source at src/ops/r9700/kv/fp8_int4_kv_append.hip.
- [x] Add device status for nonfinite source, FP16-scale overflow, and invalid page addressing;
      qualify all rejection paths on the physical R9700.
- [x] Change generic paged physical storage to record page order per plane and verify mixed-order
      zeroing plus spill/restore compilation.

## Active: dual A8 integer profiles and Pareto selection

- [x] Lock the sole growing-cache contract to FP8 E4M3FN K, signed INT4 V, and FP16 V scales;
      do not add an INT8-K alternative or runtime cache selector.
- [x] Keep both all-Q4G64+A8G64 and role-mixed Q4G64/W8G32+A8G64 as directly bound,
      production-optimized gfx1201 profiles with complete real-shape operator qualification.
      The W8 half of the mixed profile is an adaptive represented-BF16/A8G32 production Tensor
      route: interleaved physical measurements set conservative shape-specific crossovers across
      all 13 unique shapes/256 W8 tensors, while unknown all-W8-only shapes remain exact. Exact
      codec/FP64 formula, malformed/alias/nonfinite, caller-owned workspace, ISA/resources, and
      production Tensor boundary evidence is retained in the two W8A8 JSON reports. The final
      quality campaign measured the source-MSE mixed artifact within the accuracy tier under G16/G32 at
      8K (+0.012077/+0.012658 mean NLL, 3/2 new severe) and 32K
      (+0.015790/+0.015485, 16/16 new severe against a 17-position budget). Adaptive A8 is the
      default and represented-BF16 W8 activation is the explicit control. These fixed candidate
      sidecars are historical: their exact scorer binaries are no longer present, so the mixed
      dense route requires fresh scoring against the exact v3 BF16 authority and the mixed sparse
      route has not yet been acquired.
- [x] Keep the authority-bound four-role rowwise-FP8/all-other-Q4G64 artifact as a distinct third
      recipe branch. Its exact role inventory, artifact/receipt identity, dense-G16 8K/32K quality,
      executed-role proof, and planner/capacity tooling exist; it does not inherit the all-Q4
      recipe. Its retained capacity pass is invalid because it used `20,707,768,320` bytes of
      dense/spec-none materialization while claiming MTP3 plus the optimized head. Exact
      materialization is `21,290,468,352` bytes, making the chunk-4096/P8192/G16/C4 plan
      startup-inadmissible by `245,140,480` bytes. Its remaining G32 and sparse quality plus fresh
      selected-chunk capacity outcomes and eligible whole rows remain open under the shared
      terminal gate.
- [x] Replace impossible zero-BF16-flip admission with explicit accuracy and capacity-speed tiers
      over finite/aligned, paired mean-NLL, and position-aware severe-NLL guardrails. Both
      mixed-Q4/W8+A8 and all-Q4+A8 measured within a declared tier. Dense all-Q4 G16/G32 has now
      passed offline rebase against the exact v3 authority; mixed and sparse profiles remain open.
      Keep exact-token parity for same-route
      graph/eager, MTP/ordinary, and draft-window variants; compare prefill/decode schedules under
      an explicit finite aligned NLL bound with their flips diagnostic. BF16 flip count/rate is
      likewise diagnostic. Schema-v5 campaign output and focused tests cover the new contract.
- [ ] Held behind the immediate dense C1/P2048/spec-none `>=2,000 tok/s` floor and
      practical-ceiling gate above; the user-authorized smaller-gain exact-Q4 continuation does not
      authorize this downstream campaign. After those gates pass, produce and retain matched whole-model benchmark artifacts for every capacity-eligible
      recipe/cache/execution candidate under the C=1..4 product cap. Bind each candidate's
      completed 8K/32K quality sidecars to fresh exact C=1..4 ordinary capacity plus prefill/decode
      and whole-inference throughput and relevant profiler attribution. The retained
      mixed-recipe C7/C8 failures are out-of-scope stress history and no longer exclude that recipe;
      it now requires the same matched admission evidence as all-Q4.
      DFlash2 is the preferred and required speculative production path and owns downstream
      acceptance evidence. Existing MTP3 rows remain
      useful only to preserve the already-supported MTP graph/eager, ordinary-target token,
      acceptance/state, cache, row-view, and draft-window semantics. Do not schedule the prepared
      MTP shortlist-head trace, alternate head precision, MTP-bulk tuning, or any other new MTP
      optimization as a prerequisite. Schema v7 carries no MTP shortlist-head or downstream-
      readiness status; selected-profile NIAH and DFlash remain independent downstream gates.
      - [x] Close the current schema-v14 whole-evidence admission contract. The assembler accepts
            exact C=1..4 capacity outcomes for every all-Q4, mixed, or four-role base, then accepts
            whole-inference pairs only for capacity-eligible profiles using one selected chunk,
            artifact, executable, cache group, attention profile, and the exact adjacent N16
            migration receipt; four-role additionally requires its matching hybrid planner bytes.
            A whole matrix must have no failure marker and must bind `auto` both
            before and after timing; every report path is campaign-owned and every raw schema-v20
            report is reopened against its exact command. Each of the four C values requires one
            spec-none ordinary timing report containing matched 8K+256 and 32K+256 rows. Retained
            MTP3 rows serve only as diagnostic exact-token/state/graph regression evidence, cannot
            enter base recipe/cache/attention ranking, and are not a prerequisite for it. Any
            prepared runner or validator that still ranks or requires those MTP3 rows must be
            revised before physical execution. The two retained schema-v13 all-Q4/G16 sparse
            manifests are therefore
            inadmissible. After the shared chunk is selected, acquire twelve exact capacity
            outcomes (48 commands), retain measured failures as exclusions, and run the four-report
            `pareto-whole` matrix (one report per C, eight workload rows total) only for each
            capacity-eligible profile. Still-missing quality
            cells remain a separate prerequisite, and only the terminal selected route proceeds to
            required profiling and DFlash evidence.
      Classify cache-group/execution-profile dominance only over the complete matched objective set
      and preserve every raw report for the schema-v7 terminal production choice.
      Execute the remaining physical gates in dependency order: rebuild the four compile profiles
      and bind them to all twelve recipe/cache/attention candidates; acquire only the bounded
      chunk-selection screens and finalist confirmations; select and bind one chunk;
      acquire the post-chunk quality, capacity, and whole candidate matrices; classify one terminal
      artifact/group/static profile; then run the low-context, selected-route profiling and tail
      gates only for that identity, with MTP retained solely as optional supported-behavior
      regression diagnostics. A diagnostic trace used to explain a
      failed selection screen is not terminal profiling. If a retained selected-route optimization
      changes whole timing, rerun its affected terminal rows and classification, not unrelated
      candidate campaigns unless the selected identity or contract changes.
      The prepared 20260905 ownership chain is now explicit: `prefill-chunk-screen-twelve-candidate`
      feeds `prefill-chunk-selection-pipeline`; the selected chunk unlocks
      `terminal-quality-recovery` and `post-chunk-twelve-candidate`; those feed
      `terminal-static-selection`. Only its schema-v7 winner unlocks the selected low-context,
      hardware-use, ordinary-decode-memory, focused-verification, NIAH, and DFlash
      packages; their joined final cutover admission is prepared at
      `profiles/bench/final-artifact-cutover-admission-prepare-20260905`. That CPU-only owner
      recomputes every input and can publish only after all physical authorities pass; its prepared
      closure is not a completed cutover receipt. This chain remains held at its first physical step by the immediate P2048 floor
      above; it does not authorize resuming the stopped broad campaign. These are prepared
      boundaries, not completed physical evidence.
  - [ ] Held behind the immediate dense C1/P2048/spec-none `>=2,000 tok/s` floor and
        practical-ceiling gate above; the user-authorized smaller-gain exact-Q4 continuation does not
        authorize this downstream campaign. After those gates pass, select the production text-prefill chunk instead of treating the current 4,096-token
        campaign control as final. After localizing the unexpectedly low native prefill throughput,
        sweep the supported aligned chunks 1,024, 2,048, 4,096, and 8,192 at 8K, then confirm the
        same two global finalists at 32K for all twelve candidates; qualify numerical behavior and workspace/capacity,
        and bind the winning startup-fixed value into every terminal whole-inference candidate.
        Existing 4,096-token whole rows are diagnostic until this selection either retains 4,096 or
        replaces and reruns them consistently.
        - [x] Add the strict selection owner before physical execution. The schema-v2 record requires
              the exact three recipes (all-Q4, mixed Q4/W8, and four-role FP8/Q4) x G16/G32 x
              dense/B128-S16-tau900 Cartesian set, all four 8K
              chunk rows per candidate, and the same two globally shortlisted 32K finalists for
              every candidate. It selects one shared chunk by maximin normalized throughput across
              all candidate/prompt objectives, preserving the complete four-chunk 8K denominator
              while normalizing 32K against the two finalists, then maximum workspace and
              smaller-chunk tie-breaks,
              reopens every raw report, and binds every source hash. Schema-v4 Pareto input now
              requires this record and rejects PPL, capacity, or whole evidence at another chunk.
        - [x] Close selected-chunk receipt and hybrid-planner provenance propagation in the matrix
              tooling. Every all-Q4, mixed, and four-role screen/finalist and downstream
              selected-chunk campaign requires its own exact adjacent N16 migration receipt.
              Four-role additionally queries and validates its compile-matched planner at each
              exact requested chunk instead of retaining a hardcoded 4,096-token width proof; the
              selector requires one planner identity across that candidate's 8K screen and 32K
              confirmation. The terminal low-context validator reopens the selected base receipt
              and selected-chunk width authority for every recipe. It requires the hybrid
              selection/planner fields only for four-role and rejects those hybrid-only fields for
              all-Q4 and mixed. The supported candidate
              set remains exactly four chunks, screening is C=1 only, and all downstream product
              matrices remain capped at C=1..4. Paths and identities are explicit; no glob,
              modification-time, or `latest` selection is admitted.
        - [x] Retain the superseded twelve-candidate screen preparation at
              `profiles/bench/prefill-chunk-screen-twelve-candidate-20260905` as non-runnable
              history. Its eight already-created all-Q4/mixed manifests predate their now-published
              N16 migration receipts and cannot be resumed, rebound, or selected. After the P2048
              gate opens chunk preparation, create a fresh no-overwrite screen, finalist,
              campaign, pipeline, and selection authority whose names end in
              `-receipt-bound-n16k16-20260905`. This credits no physical evidence yet; all C1
              screen/finalist measurements and the schema-v2 selected-chunk authority remain
              pending.
  - [ ] After terminal artifact, value group, static execution-profile, and prefill-chunk
        selection, establish the separate C1 low-context dense-prefill performance ladder at
        prompt lengths
        `P={128,512,1,024,2,048,4,096}` under `auto`. Use one compile-matched dense-attention
        control with speculative execution disabled, the exact selected artifact and value group,
        the selected startup chunk (including its natural smaller effective/final chunks), and the
        same fixed corpus-prefix and timing semantics for every row. Require at least `2,000`
        prefill tok/s at `P=2,048` unless the user later refines the intended low-context boundary;
        this floor applies only to the terminal selected artifact/profile, not separately to every
        comparison candidate that preceded terminal selection. Retain the full ladder so
        startup/launch and context scaling remain visible. If that row
        fails, trace the matched row before changing kernels and apply the selected-route gate below:
        prove the executed dominant-kernel gfx1201 ISA/resources and quantify its logical FLOP/byte
        roof position from matched `auto` timing before choosing an optimization. This is a
        low-context dense latency/throughput gate, not evidence against the 8K quadratic dense cost
        or a substitute for selecting and qualifying XAttention at longer context.
        The fail-closed executable package is prepared at
        `profiles/bench/low-context-selected-ladder-20260905`: it validates schema-v7, derives the
        unique compile-matched dense control for the winner's exact artifact and value group,
        reopens the selected-chunk hybrid planner when applicable, runs only the five C1 ordinary
        rows, and publishes the recomputed evaluation without overwrite. It is intentionally
        blocked on the terminal schema-v7 authority; no ladder or evaluation output exists yet.
        The `P=2,048` floor is a hard `2,000 tok/s` gate on the benchmark's mean of
        per-repetition `prompt_tokens/prefill_seconds`; the package publishes a complete failing
        result for diagnosis but returns nonzero, and does not substitute an inequivalent
        per-repeat time cutoff. After the dense-Bq16,
        LDS-barrier, Q4-N128, and token8-RMSNorm promotions, an all-Q4/G16 dense C1 run under
        `auto` with speculative execution disabled measures `875.173 tok/s` and `2.340110 s`
        mean prefill over three repetitions (report SHA-256
        `8de226d544edad4b2cbed1c7b7fd5927a4dbd5b33e8a372937d1f0c124c4b922`). This is about
        `5.1x` the broken pre-promotion observation but still misses the floor. Its one-repetition
        selected-region trace accounts for essentially the complete wall time: 320 production Q4
        N128 CTA calls consume `1.087014 s` (`46.58%`), 16 production Bq16 dense-attention calls
        consume `0.804238 s` (`34.46%`), GDN recurrence consumes `0.136528 s` (`5.85%`), and
        the promoted 129 K5120 RMSNorm calls consume only `0.012974 s`; the 32 K256 RMSNorm calls
        that correctly retain the incumbent consume `0.025190 s`. Causal convolution remains only
        `0.033675 s` and activation quantization `0.034045 s`, so neither may displace the two
        dominant optimization problems. This diagnostic is not the terminal five-row ladder, but
        it replaces the old scalar-attention and pre-N128 projections for choosing the next kernel
        work: material reductions in both Q4 Linear and dense attention are necessary to reach the
        literal floor.
        After the full-score GQA6, gated-RMSNorm, split-view SiLU, and K256 token8 RMSNorm
        promotions, the authoritative matched P2048 rerun improves to `1,214.498278 tok/s` and
        `1.686293799 s` mean prefill (`0.001380049 s` standard deviation), but still misses the
        unchanged `2,000 tok/s`/`1.024 s` gate. The report is
        `profiles/bench/prefill-p2048-post-k256-rmsnorm-20260904.json` with SHA-256
        `fe0598f603b9a3d3e5d25475b27c39da897ed0b4069566495053f04c1246b6b4`. This result
        supersedes the prior timing as the current low-context optimization baseline, not as a
        completed five-row terminal ladder. A matched current mixed-Q4/W8 diagnostic with A8
        activation coding, C1, chunk 4,096, and speculative execution disabled is slower at
        `920.8914604 tok/s` and `2.223935366 s`. Its report is
        `profiles/bench/prefill-p2048-mixed-q4-w8-current-20260904.json`, SHA-256
        `590f3e226d01a11269b79b959f7daf3800b3d392f7c6c8cea76c974bc1ec76df`; it does not
        provide a shortcut to the dense gate or supersede the all-Q4 authority.
        The post-full-score trace retained at
        `profiles/rocprof/diagnostic-p2048-post-fullscore-causal-20260904/trace_kernel_stats.csv`
        (SHA-256 `afba65658aea8dd344b8c81b17730c8e35402b61868d7a1f39c5733e0a3e0c88`)
        captured both the warmup and measured pass, so its global percentages are not the decision
        attribution and do not replace matched `auto` timing. The unique measured marker instead
        contains 2,249 complete kernels and no boundary crossing. Its selected-region report is
        `profiles/rocprof/diagnostic-p2048-post-fullscore-causal-20260904/measured-selected-region-attribution.json`
        (SHA-256 `3993cb56bdc620f13e9ba223463f041120104d28cf91293dbe491d3bd3d976c7`).
        Independent kernel service is 1,729.283668 ms and its timestamp union is 1,719.161926 ms
        inside 1,753.456094 ms of marker wall; the 34.294168 ms kernel-inactive gap is not proven
        GPU idle without runtime/API activity. Measured-only attribution is 320 Q4 M64xN128 CTA
        calls/64.5355%, 48 ordinary GDN-recurrence calls/8.4243%, 16 full-score PV calls/5.0115%,
        16 full-score QK calls/4.8159%, 64 then-generic split-view SiLU calls/4.1999%, 48
        then-incumbent gated-RMSNorm calls/3.7309%, 321 activation-quantization calls/2.0451%, and
        32 then-incumbent K256 RMSNorm calls/1.4637%. The global trace's 644 Q4 WMMA32 calls are not ordinary
        two-pass P2048 work: 640 layer calls are the startup T=1 ordinary code-warm execution plus
        Device Graph capture; four vocabulary-head calls cover those two startup executions and the
        warmup/measured prefills. Only the measured vocabulary head (`2.585003 ms`, 0.1495% of
        measured kernel service) belongs to the
        measured P2048 pass, and startup service is excluded from steady-prefill attribution. That
        trace predates the gated-RMSNorm, split-view SiLU, and K256 RMSNorm promotions, so it
        supports those completed choices but does not supersede the current `1,214.498278 tok/s`
        whole-prefill authority or establish current residual timings.
        A fresh attribution-only trace from the cleaned, rebuilt all-Q4/G16/dense production
        checkout is retained under
        `profiles/rocprof/diagnostic-p2048-current-production-selected-region-20260904/`.
        Its schema-v20 benchmark report (SHA-256
        `9a21a6e10c2eb12f245f37049150a10fb2d0f86b757fc06ba989f7c06cf27e51`) records
        `1,337.361666 tok/s` and `1.531373339 s` for one C1/P2048/chunk-4096 measured pass with
        speculation disabled under `auto`; it is a current bottleneck diagnostic, not a terminal
        timing authority. The unique database (SHA-256
        `3130f21e834125bc32de16399fa2c4fb0fea5cb061182cefd122c1d418235aef`) and analyzed
        selected-region record (SHA-256
        `a427b0d5d2d23cf12cbfcdacc98394a121636f06d3443386ccee01adabd1f27b`) bind unchanged
        executable/artifact/corpus/tool/source identities before and after capture. Kernel-active
        union is `1,494.565572 ms` of `1,526.733259 ms` Text-prefill wall (`97.8930%`), so host
        launch gaps cannot explain the deficit. The 320 selected Q4 ping/pong CTAs consume
        `950.143999 ms`; dense attention consumes `197.841899 ms`, GDN recurrence
        `149.470363 ms`, gated RMSNorm `66.257030 ms`, and all 320 activation quantizers
        `36.346063 ms`. Sharing the 64 duplicate K5120 quantizations can recover only about
        `4.684 ms`, so it is not a material prefill repair; Q4 remains the decision-owning stage.
        - [x] Promote the exact represented-BF16 K5120 token8 RMSNorm route for `T>=128`, while
              retaining the serial kernel for other feature widths and smaller rows. The
              no-clobber gfx1201 report
              `profiles/bench/r9700-rmsnorm-token8-ab-20260904.json` (SHA-256
              `a56039f86fc2333a8a5523cddd07e3c3712ece49e22b8b82d7e3b87b02a2ebf2`) proves
              incumbent-bit-exact output plus the independent FP64 criterion. At P2048 the
              production specialization reduces RMSNorm from `0.872519` to `0.101720 ms`
              (`8.5777x`) with unit offset disabled and from `0.874641` to `0.101400 ms`
              (`8.6257x`) with unit offset enabled; every retained T128-and-larger row wins.
              LLVM reports 22 VGPR, occupancy 16, four 128-bit vector loads, and zero
              LDS/private/scratch. This closes only the measured RMSNorm subproblem and does not
              displace the dominant Q4 Linear and dense-attention work.
        - [x] Qualify and reject the exact adjacent residual-add to K5120 token8 RMSNorm prefill
              fusion. The trace established 127 directly consecutive pairs, but preserving the
              observable BF16 residual publication before the ordered FP32 `fmaf` chain made the
              one-launch candidate slower at every T={1,024,2,048,4,096,8,192} cell. At P2048 it
              regressed from `0.188480005` to `0.554238975 ms` per pair (`2.94057178x`,
              `-0.365758955 ms`); the full ratio range was `1.89297712x`--`2.98276377x`. Static
              qualification had passed at 18 VGPR, occupancy 16, and zero LDS/private/scratch,
              and exact residual/output, complete independent-oracle, poisoned-rewrite, alias,
              and status checks passed. The immutable physical report is
              `profiles/bench/r9700-residual-rmsnorm-k5120-prefill-ab-20260904.json`, SHA-256
              `6ed367d0438f0f7f8dbfe4bd01f7da126a6857c1953da8f1ed6e294eec20c134`; the retained design is
              `profiles/bench/r9700-residual-rmsnorm-k5120-prefill-static-design-20260904.json`,
              SHA-256 `40dcc021e59e8c80272ed6c36c9ab4ef36e1bd6c2e9939c439671426a44eca73`.
              The qualification kernel/API/harness/checker/validator surfaces are removed,
              production remains the separate residual-add plus selected RMSNorm pair, and this
              terminal result does not authorize an adjacent fusion variant.
        - [x] Promote the exact represented-BF16 K256 token8 RMSNorm route for rows `>=128`, while
              retaining the incumbent below the measured boundary and for other feature widths.
              The immutable report `profiles/bench/r9700-rmsnorm-k256-token8-ab-20260904.json`
              (SHA-256 `87e4be62c5c7574273f7ee22eaf4858fa8442a496d738fa68a42d8b4452edbce`)
              passed incumbent-bit-exact and independent FP64 formula checks and won every
              measured row. At 2048 rows the median fell from `0.0939209983` to `0.0412399992 ms`
              (`2.27742481x`); at the real 49152-row query extent it fell from `1.10108304` to
              `0.0953209996 ms` (`11.5513163x`). The static gate reports 23 VGPR, occupancy 16,
              four 128-bit loads, and zero LDS/private/scratch.
        - [x] Promote the exact represented-BF16 causal-convolution token-tile-4 route for
              `T>=64`, while retaining the serial kernel below the conservative stable crossover.
              Tile 4 loses at T16 and its T32 win is only `0.01856` versus `0.01900 ms`, so neither
              point is selected. It wins every selected measured extent T64/128/256/1024/2048/
              4096/8192; at P2048 it reduces one call from `0.659199` to `0.163760 ms` (`4.0254x`).
              The production and retained direct-regression routes are incumbent-bit-exact and
              preserve exact final BF16 history publication. LLVM reports 21 VGPR, occupancy 16,
              and zero LDS/private/scratch for tile 4. This closes only the measured convolution
              subproblem; the P2048 trace bounded all causal convolution to `0.033675 s` before
              promotion, so it does not displace the dominant Q4 Linear and dense-attention work.
        - [x] Reject the FP32 affine-chunk GDN recurrence challenger. The no-clobber report
              `profiles/bench/r9700-gdn-affine-chunk-rejected-20260904-r2.json` (SHA-256
              `f574df9e6e1207b8177e527289a6bc804fb978960415b366405b5bf7ad5fb7fe`) preserves
              complete sequential-FP64 numerical admission and all raw/median C32/C64/C128 rows.
              C64 at P2048 measured `5.622350 ms` versus `2.892155 ms` for the incumbent, a
              `0.514403x` speedup (`1.9440x` slower), and therefore remains disconnected from
              production.
        - [x] Reject the normalized GDN grouped-head challenger. Grouping the three value heads
              sharing each Q/K head reduced duplicate normalization but collapsed the selected
              192-CTA grid to 64 longer-lived CTAs. The no-clobber report
              `profiles/bench/r9700-gdn-grouped-heads-ab-20260904.json` (SHA-256
              `014c442f916466a8d460783bb9f18fc7d5ee4a69663827faf3536d4ab43f6bd4`) records losses at every
              T128/512/2048/4096 row; P2048 is `4.105878 ms` versus `2.962798 ms`, a `0.721599x`
              speedup. Production remains the four-row-tile route.
        - [x] Reject the ordinary GDN LDS-scope-only challenger. It preserves the selected
              192-CTA/256-thread arithmetic and removes all four static `global_inv` sites while
              retaining workgroup-scoped LDS ordering. The no-clobber report
              `profiles/bench/r9700-gdn-ordinary-lds-scope-ab-20260904.json` (SHA-256
              `4927845416d01aa6267d72df89342e91ab30ef0b9416bdf221d0116230acb444`) wins every
              T128/512/2048/4096 row, but P2048 improves only from `3.113205` to `2.761526 ms`
              (`1.127350x`), below the fixed `1.5x` admission gate. Production remains unchanged.
        If that trace confirms the current scalar dense full-attention leaf is material, replace it
        with one bounded causal tiled-attention challenger. The current compile/runtime audit
        confirms that bulk `P<=4,096`, `T>2` Text prefill cannot select either the T1/T2 FP8-QK
        WMMA leaf or the context-8,192 split-512 leaf: it executes scalar fused streaming, rereads
        K/V/scales independently for every query row, and has no cross-query cache reuse or matrix
        instruction. Use page-aligned `Bk=64`, sweep only
        `Bq={4,8,16}`, and first preserve represented-query numerics by decoding FP8 K into
        cooperative LDS staging for native BF16 WMMA. Reuse each staged tile across query rows
        first, then add the six-query-head GQA reuse only where resources and timing support it;
        cooperatively resolve each page and stage its token-fastest K plus feature-fastest packed V
        and FP16 scales. Keep FP32 online softmax and direct INT4-V accumulation without a global
        `T x context` score/probability matrix. Qualify the selected tile directly against the
        independent represented-input FP64 causal oracle at `P={128,512,1,024,2,048,4,096}` for
        the selected G16/G32 cache. Require a specialization-specific fail-closed
        ISA gate for the numerically selected native BF16- or FP8-WMMA profile, plus exact
        VGPR/LDS/occupancy and zero-private-scratch gates, online-softmax/direct-INT4-V oracle
        agreement and matched `auto` operator timing before route promotion; the post-promotion
        whole ladder and same `P=2,048` floor remain the enclosing completion gate. Do not
        select the arithmetic profile before the independent design/numerical review; FP8-Q WMMA
        remains a separate implementation profile with its own BF16-source quality gate. Consider
        cache-policy hints or software
        prefetch only after the tiled algorithm is selected and a dispatch-scoped GL2C/TCP/SQ A/B
        identifies a remaining cache-path limit; hints do not substitute for row/GQA data reuse.
        - [x] Promote the physically selected Bk64/Bq16 BF16-WMMA route for both G16/G32 over the
              exact initial-prefix P=128..4096 envelope. The retained no-clobber report
              `profiles/bench/r9700-dense-prefill-attention-ab.json` has SHA-256
              `d7a0f50a6ae91448583b07d7477d2926b950b65eeadd3ac924f28a8354ed95a6`; Bq16 won every
              cell. Its selected/incumbent median ratios at increasing P are
              0.561224173/0.566735009/0.504308119/0.437316979/0.416703457 for G16 and
              0.546212923/0.554317816/0.495417690/0.429230227/0.417418378 for G32. Production
              retains the previous route below P128, for later-prefix chunks, tree attention, and
              nonselected plane layouts. Bq4/Bq8 runtime paths were removed. Rebuild, rerun the
              Bq16 oracle/static gate through the production entry, then acquire the affected
              post-promotion whole ladder before closing the parent item. The rebuilt production
              oracle and both static specialization gates pass: each G16/G32 kernel has exactly
              sixteen native BF16 WMMA instructions, wave32/WGP execution, 217 VGPRs, 37,160 bytes
              LDS, occupancy 6, and zero scratch/private/spills. The diagnostic P2048 trace above
              confirms exactly 16 production Bq16 calls and no scalar bulk-prefill fallback.
        - [x] Replace fused Bq16 in the production initial-prefix envelope with the physically
              selected three-stage full-score GQA6 route. Its retained operator report has SHA-256
              `0f80353256105a6b759b1f906f8cfc1cee1ef204870c83f1d3dff7ca0d69dcca`; at P2048
              it measures 14.669395 versus 61.569540 ms for G16 and 14.819379 versus 61.336494 ms
              for G32. QK shares each K tile across six heads, maximum reduction and PV are
              separate kernels, and the caller-owned FP32 score/max workspace is reserved at the
              planner's stable peak. The post-promotion whole-prefill result above remains below
              the enclosing 2,000 tok/s gate, so fresh whole-prefill attribution—not another
              operator-only extrapolation—owns the next optimization decision.
        - [x] Reject and remove the qualification-only full-score Bq32/GQA6 QK route. Each of six
              waves evaluated two sequential M16 query fragments against one staged K16 tile,
              halving QK workgroups while preserving the exact FP32 score plane, maximum, and PV
              stages. Independent FP64 represented QK and complete-attention oracles, sampled
              Bq16/Bq32 bit-exact score parity, finite evidence, poisoned active-prefix rewrite,
              page-row, and zero/invalid active-row cases passed at P=128/512/1,024/2,048/4,096
              for G16/G32. Static gfx1201 evidence shows 32 native BF16-to-FP32 WMMAs, 33 VGPR,
              8,360-byte LDS, occupancy 15, and zero scratch/spills versus Bq16's 16 WMMAs,
              29 VGPR, 8,296-byte LDS, and the same occupancy. Physical `auto` timing found the
              longer serial wave schedule slower despite equal occupancy: Bq32 QK lost all ten
              cells and was `1.210299x`/`1.193247x` slower at P2048 G16/G32. Complete-Op P2048
              ratios were `0.879411x` and `1.075306x`, but nine of ten complete cells and every
              QK cell violated the no-regression boundary, so one variable cell cannot select it.
              The corrected retained report is
              `profiles/bench/r9700-dense-full-score-bq32-ab-20260904.json`, SHA-256
              `cc1ccf4e845482f91b78f4e022969a19198fe51ea2a18d3d4dc9d8037ac5a7c0`.
              Production remains Bq16; the Bq32 implementation and temporary qualifier modes are
              removed after this decision.
        - [x] Promote the physically qualified dense full-score Bk16/Bk32 crossover. The immutable
              pure-Bk32 report `profiles/bench/r9700-dense-full-score-bk32-ab-20260904.json`
              (SHA-256 `d85a9ce25c8bee3dc6e4138d4dcfc6a9aec8034e8586fed63573b0fd346f158e`)
              loses at P128 but wins at every measured P512/1,024/2,048/4,096 cell for both G16
              and G32. The production entry point now dispatches Bq16/Bk16 below P512 and
              Bq16/Bk32 throughout P512..4,096. The fresh v1 report
              `profiles/bench/r9700-dense-full-score-bk16-bk32-crossover-ab-20260904.json`
              (SHA-256 `7941bb518ce2a2287e06403952ad9d90f2ada115550c1967e474906f5a600a2f`)
              passes both groups at P128/512/1,024/2,048/4,096: QK and complete-Op ratios are at
              most `1.01` in every cell and P2,048 QK is `0.643247x` for G16 and `0.656254x` for
              G32, both at most `2/3`. Production now uses Bk32 throughout P512..4,096, with the
              selected kernel named and checked directly by production resource/static tooling;
              obsolete fused-Bq16, pure-Bk32, and composite qualification APIs/harness modes are
              removed while both decision reports remain immutable.
              The rebuilt production whole-model P2,048/C1/no-spec result is `1,371.444888`
              tok/s (`1.493317223 s`) under `auto`, improving the preceding `1,337.361666`
              tok/s diagnostic by `2.5486%` but remaining below the `2,000 tok/s` floor. The
              retained report is
              `profiles/bench/r9700-dense-bk-crossover-p2048-c1-20260904.json`, SHA-256
              `43e447a46e6400b0a419540036a2c2cf7e3d9c37668172822927703bd9426fe1`.
        - [x] Qualify and reject the dense full-score P2048 FP8-Q/Bk32 route. The isolated
              candidate encoded represented BF16 Q once into the dead first 12,582,912 bytes of
              the existing score workspace, then used raw E4M3 Q and stored-E4M3 K in native
              WMMA without growing workspace or changing maximum/PV. Its exact FP8-Q panel,
              independent FP64 score and complete-attention oracles, active-output rewrite,
              causal/page/status/tail checks, and static gates all passed: 32 native FP8 and zero
              BF16 WMMAs, 63 VGPR, 8,192-byte LDS, occupancy 16, and zero scratch/spills. The
              immutable physical report
              `profiles/bench/r9700-dense-full-score-fp8-q-bk32-ab-20260904.json` (SHA-256
              `ef0d6557a59145c3f632b96b24f41c922369d9f42fa48ab2c7b24d5b1c7e7f30`) measured
              2.983591080 versus 3.442310095 ms per call (`0.8667409378x`), or 47.73745728 ms
              over 16 calls and only 7.33950424 ms matched projected saving. That misses the
              predeclared 32.439611 ms / 15 ms whole-saving gate, so no whole run is admitted.
              Production remains the BF16-WMMA Bq16/Bk32 route; all candidate-only kernel, API,
              quantizer, harness, static-checker, and test surfaces are removed, with no adjacent
              FP8-Q variant sweep. The retained static design report is
              `profiles/bench/r9700-dense-full-score-fp8-q-bk32-static-design-20260904.json`
              (SHA-256 `aed480c787313c280aefc83d8cdf943a6b81711872786677f905f3d4036cd521`).
  - [x] Replace the serial-per-token XAttention consumer with a B16-query sparse
        FlashAttention-style gfx1201 route. The retained C1 32K selected-region benchmark report
        (SHA-256 `a22837755d61d295ed00e160a0ee272b96c6bff87e90121ef9d11803f21a02d9`)
        and rocprof DB (SHA-256
        `83a42f5ce11d73f03354cf5ec2872b29fa8e1d53a45ae537d96f519be40357ae`) show
        `443.849 s` of independently summed base Text-prefill kernel time, only `0.318 s` of
        MTP-prefill work, and `0.348 s` (`0.078%`) of host/idle gap. Full attention totals
        `325.124 s`; its current consumer is the dominant kernel at `302.770 s` over 128
        dispatches (`68.13%` of prefill), versus `15.353 s` for the ranker. Post-mixer totals
        `92.908 s` and GDN `25.817 s`, so neither MTP-prefill nor host scheduling explains the
        low rate. Each row/head consumer CTA walks retained tokens serially and synchronizes once
        per token. The 32K qualifier's `32.450 ms` consumer measurement covers one 128-row block,
        so the comparable 4,096-row/layer baseline is `1.038 s`, not `32.450 ms`; the final runtime
        chunk is `4.146 s` per layer (`3.99x`). Linear scaling from the qualifier's `11.71875%`
        keep fraction suggests roughly `46.8%` real-input retention, but this is an inference rather
        than a measured keep statistic. The replacement parallelizes QK/online-softmax/INT4-V work
        across retained pages with native BF16 WMMA, FP32 online softmax, and direct INT4-V
        accumulation without a global score workspace. Its schema-v4 G16/G32 physical qualifiers
        passed the independent FP64 oracle at `1.6023e-8`/`1.6986e-8` maximum absolute error and
        measured `28.56x/32.40x` and `28.81x/32.72x` dense-over-sparse at 8K/32K on the concentrated
        fixture. Reports with executable/source hashes are retained under
        `profiles/bench/r9700-xattention-b16-requal-s16-tau900-g{16,32}/`.
  - [ ] Held behind the immediate dense C1/P2048/spec-none `>=2,000 tok/s` floor and
        practical-ceiling gate above; the user-authorized smaller-gain exact-Q4 continuation does not
        authorize this downstream campaign. After those gates pass, validate
        the redesigned consumer at real 8K/32K model keep distributions, select the
        production prefill chunk, and rerun affected whole evidence. The concentrated 11.72%-keep
        qualifier corpus and the current-tree 16.2502%-keep production-scale refresh establish
        operator correctness and potential, not model-level speed. The existing serial-consumer C1
        whole row remains diagnostic and cannot enter selection. Resume no separate operator sweep:
        after that decision authorizes resumption, the next physical gate is the existing exact
        twelve-candidate C1 8K chunk screen, followed only by its two globally selected 32K finalists.
        The prepared screen/finalist owner is
        `profiles/bench/prefill-chunk-screen-twelve-candidate-20260905` plus
        `profiles/bench/prefill-chunk-selection-pipeline-20260905`; operator refresh evidence is
        complete, but no model-distribution screen or terminal selection row is credited here.
  - [ ] Held behind the immediate dense C1/P2048/spec-none `>=2,000 tok/s` floor and
        practical-ceiling gate above; the user-authorized smaller-gain exact-Q4 continuation does not
        authorize this downstream campaign. After those gates pass, obtain
        matched rebuilt real-model attribution and terminal whole-selection evidence for
        the already-promoted production-extent multi-wave A8Q4 prefill CTA. The retained 32K `auto`
        trace that motivated the completed operator replacement attributes
        `121.708 s` to 25,229 `a8q4g64_linear_wmma32_kernel` dispatches. At the dominant
        T=4,096/R=34,816/K=5,120 gate/up shape, each 127.79 ms dispatch achieves only about
        11.43 effective TOP/s: independent 16x16 waves request about 94.1 GB of operand/scale data
        from only about 0.116 GB of unique represented inputs and weights. Implement and qualify a
        prefill-only 64-token x 64-row, 16-wave CTA with cooperative K64 LDS staging and four-way
        A/B tile reuse, while retaining the low-overhead one-wave route for T<=128. Preserve the
        A8-low/A8-high signed reconstruction, exact G64 scale boundary, FP32 group accumulation,
        one BF16 output rounding, direct row-major artifact binding, and status/alias semantics.
        Qualify at T=1,024/2,048/4,096/8,192 over the dominant real shapes with native IU4 ISA,
        resource/no-scratch evidence, an independent complete smaller oracle plus spread real-shape
        checks, and finally matched real-model prefill attribution. Promote only if whole-prefill
        improves; do not replace the selected decode/small-T kernel on extrapolation.
        Operator implementation/admission and its production dispatch are complete in the checked
        children below. The parent remains unchecked only for the selected-chunk artifact-bound
        trace and whole-model gate owned by `post-chunk-twelve-candidate-20260905`.
        - [x] Close the physical linear-Op admission subgate. The schema-v3 report passed the
          exact 8-shape x T=1,024/2,048/4,096/8,192 Cartesian matrix with zero maximum BF16 steps
          in every sampled row and the exhaustive tail case. The cooperative CTA won all 32 rows
          by `2.5261x` through `8.1587x`. The retained report is
          `profiles/bench/r9700-a8q4g64-prefill-cta-qualification.json` (SHA-256
          `b3c78cab21dfc0f97521269245b8442009048aadcd9db5ffd947a4d7d3baf889`). This closes only
          the exact tuple predicate, which is now the production linear-Op dispatch boundary. The
          report is immutable pre-promotion provenance; matched rebuilt real-model prefill
          attribution and whole selection keep the parent item open.
        - [x] Promote the LDS-scoped CTA barrier after direct physical A/B. The immutable Q4 report
          `profiles/bench/r9700-a8q4-prefill-cta-lds-scope-ab-20260904.json` selects the production
          barrier at a weighted P2048 ratio of `0.9857744040`; every tuple is bit-exact and faster.
          Production now emits workgroup-local release/signal/wait/acquire ordering, while the old
          global-invalidating barrier remains only as an explicitly named regression control. The
          rebuilt static gate binds the production symbol's native IU4 path and zero-scratch
          resource envelope. The diagnostic trace above confirms exactly 320 production CTA calls.
        - [x] Promote the M64xN128 persistent-N2 CTA after direct physical A/B. The immutable report
          `profiles/bench/r9700-a8q4-prefill-cta-n128-ab-20260904.json` (SHA-256
          `9f6a49e725bc3f13ade5775332b95848a36ad796cb4ff25ea9b55ae4838f674e`) selects N128 at a
          weighted P2048 challenger/incumbent ratio of `0.8319113784`, with no material tuple
          regression. The production specialization has eight native IU4 WMMA instructions, two
          workgroup-local signal/wait pairs, 92 VGPR, 8,576-byte LDS, and zero private/scratch
          storage. It is now the sole CTA for the exact qualified eight-shape by
          T=1,024/2,048/4,096/8,192 envelope; M64xN64 remains only as the explicitly named direct
          regression control and the one-wave route retains nonqualified/tail extents. The parent
          remains open for rebuilt real-model attribution and whole-prefill evidence.
        - [x] Reject the qualification-only M64xN128 ping/pong staging challenger without changing
          production. Its exact oracle and every measured tuple are nonregressing, and weighted
          P2048 improves from `1126.778368` to `914.413888 ms` (ratio `0.8115295`, saving
          `212.364480 ms`, or `1.23224x`). That does not meet the predeclared `>=1.5x` promotion
          gate. The terminal report is
          `profiles/bench/r9700-a8q4-prefill-cta-pingpong-ab-20260904.json`, SHA-256
          `06a2846e1ae5ba90989dcb401e578a1b96479a8e472b7aa0d5befdcbcf6281f7`.
          Production therefore remains the selected M64xN128 persistent-N2 CTA; subsequent work
          requires a structural mapping/reuse improvement rather than another staging-only claim.
        - [x] Reject the M64xN256 plus ping/pong Q4 challenger without changing production. Its
          512-thread/16-wave mapping assigns four N16 row fragments to each wave, halves CTA,
          wave, and activation reread counts over production M64xN128, and preserves full WMMA
          utilization for all eight production shapes. The weighted T2048 represented request
          floor is `842.961535 -> 639.334490 GB` (`1.31850x`), but physical performance did not
          compound with the independently measured staging gain.
          The compiled static gate passes at 16 IU4 WMMAs, 160 VGPR, 25,856-byte LDS, occupancy 9,
          a 512-thread maximum workgroup, zero private/scratch, and six next-bank loads proven
          outstanding across the current-bank WMMAs. The independent exact/FP64 gate and every
          tuple passed, but weighted P2048 improved only from `1109.850230` to `920.208770 ms`
          (ratio `0.8291288`, `1.20609x`, saving `189.641460 ms`), below the fixed `>=1.5x` gate.
          The terminal report is
          `profiles/bench/r9700-a8q4-prefill-cta-m64n256-ab-20260904.json`, SHA-256
          `e1d611ff2a74ebad9c300297c93ea1722a74c57d01c71a75836987fe7fe3ba3b`.
          At that operator-only decision point, production remained the selected single-bank
          M64xN128 CTA; the later matched whole-P2048 gate below supersedes it with ping/pong.
        - [x] Reject and remove the K128-superstep Q4 challenger at the static design boundary.
          It staged two adjacent G64 groups per ping/pong bank to halve dynamic barrier pairs,
          but left every IU4 WMMA, conversion, scale/FMA, and requested payload byte unchanged.
          The compiler preserved eight independent accumulator destinations with same-chain
          instruction gaps `5/4/13/5/9/9/10/17` (production's four chains are
          `10/13/9/11`); explicitly preloading both members before round-robin issue produced
          identical emitted scheduling and resources. Relative to production, the candidate
          increased LDS from 17,152 to 34,304 bytes and VGPR from 88 to 138, reducing reported
          occupancy from 16 to 10, while retaining zero private/scratch storage. With no work or
          traffic reduction and no credible `>=1.5x` bound, a slow physical matrix had no decision
          value after the benchmark queue was stopped. Its qualification kernel, CLI, report
          schema, and static-checker mode were removed; production remains unchanged.
        - [x] Reject one-scale-per-row signed Q4 before artifact/runtime implementation. The
          deterministic CPU-only screen sampled eight fixed rows from each of all 439 BF16
          matrix recipes (`3,512` rows and `22,160,256` represented elements). Relative to
          Q4G64, aggregate relative-L2 worsened from `0.1120950135` to `0.1970425118`
          (`1.757816924x`) and maximum absolute error from `0.0391845703` to `0.0559082031`
          (`1.426791277x`); every matrix's relative-L2 worsened, and
          `mtp/input_projection` reached `7.77488x`. The retained diagnostic is
          `profiles/bench/r9700-q4-row-scaled-sampled-quality-20260904.json`, SHA-256
          `a6dcebd55da1a915e55df57de961c796d130d1fce9fb57860e0a21711abfd323`, with the
          config/index/ranking, all 18 source shards, and implementation hashes. The format is
          not a credible integer recipe despite its safe I32 bound and scale-traffic reduction.
        - [x] Bound larger signed-Q4 groups with the identical deterministic 439-matrix CPU
          screen. The retained report
          `profiles/bench/r9700-q4-group-size-sampled-quality-20260904.json` (SHA-256
          `5e5fa075d1113242cf3400c2c0c192180eb014bdc2c17fc0c1feefd46ff8256f`) finds
          aggregate relative-L2 candidate/control ratios of `1.104192022x` for G128,
          `1.207007178x` for G256, and `1.310485874x` for G512; every one of 439 matrices
          worsens at every larger group. At the exact weighted P2048 production shape inventory,
          scale requests and FP32 group accumulations fall from `37,017,354,240` bytes and
          `789,703,557,120` operations at G64 to respectively 1/2, 1/4, and 1/8 at
          G128/G256/G512. Conservative split-code I32 maxima are only `146,432`, `292,864`,
          and `585,728`, far below `INT32_MAX`. G256 and G512 are rejected on their source-error
          increase. G128 alone remains a bounded evaluation candidate: its `10.42%` aggregate
          relative-L2 increase must pass fresh BF16-source PPL/exact-token gates before any
          artifact/runtime implementation or production claim.
          The source scorer itself cannot select a weight group: its scheme choices affect only
          the full-attention KV cache, and every matrix load remains BF16. A separate diagnostic
          owner now provides the smallest non-artifact gate by replacing only rank-two Text
          weights with canonical Q4G64/Q4G128 FP16-scale decodes before the unchanged BF16
          formula. Its completed CPU preflight is
          `profiles/bench/r9700-q4g128-source-ppl-preflight-20260904.json` (SHA-256
          `6b0192a10dddb9ef3691c9f0ab5bc57cb79b89326ac61468e9a2f2c0c37b3422`): all 18 shards,
          1,199 indexed tensors, 851 Text tensors, and 498 raw/322 non-speculative scorer-consumed
          logical Text matrix views are complete and group-compatible. The next executable step is
          the fresh paired 8K G64/G128
          commands in `tools/ppl/README.md`, followed by the strict comparison against the retained
          exact v3 BF16 cell. This remains weight-codec-only evidence because activation coding and
          matmul stay BF16; a pass permits Q4G128 artifact/runtime implementation but cannot replace
          later product A8Q4 PPL.
          The paired source-only gate is now complete and rejects G128 before artifact or kernel
          implementation. Against the exact retained 8K BF16 authority, G64 has mean-NLL delta
          `+0.0361149393` and 11 new severe positions at the 11-position budget, so it passes the
          capacity-speed tier. G128 has mean-NLL delta `+0.0530273908`, exceeding the fixed
          `ln(1.05)=0.0487901642` limit, although its 10 new severe positions pass. The incremental
          G128-minus-G64 delta is `+0.0169124515`; exact argmax mismatches are diagnostic only.
          The immutable comparison is
          `profiles/ppl/q4-group-source-8k-20260904/comparison.json` (SHA-256
          `21d0b436f38e48c4cd8ced92be9a291f915e1e2cf39b8acd1be04aee7ffafea2`), binding G64/G128
          report SHA-256 values `4750c0e18c8f5c9c21e80e5beb2a2ff108dc4feb771466010de4c3b5f7aacd8f`
          and `932c3ec8788d1c73749639a92bdc60b71523370e60e76c62943a8bbdfac26c51`.
          The comparison reader now resolves the raw BF16 cell through its unique enclosing
          campaign cell because the raw descriptor does not itself carry the retained sidecar
          hashes, and normalizes the campaign dictionary versus diagnostic pair-list shard forms;
          eight focused tests pass. Do not escalate G128 to 32K or add a G128 product path.
          A later exact-role selective gate narrowed the scope to 48 value-Z, 48 GDN-output, and
          64 MLP-down calls and passed its activation-inclusive 8K quality boundary, but the only
          admitted disconnected A8G128-by-Q4G128 kernel did not establish material P2048 speed.
          Batched report
          `profiles/bench/r9700-a8g128-q4g128-n16k16-batched-ab-20260905.json` (SHA-256
          `62928d0d4d6054b55fb067457ef79e418db4a533e10b2cc4e31eeec7051dfddf`) measured
          `-0.155958 ms` weighted saving with a conservative
          `[-2.511666,+2.199750] ms` interval. Five isolated range outliers failed stability:
          value-Z T1 control `50.7115%`, GDN-output T12 candidate `26.2646%`, GDN-output T36
          control `27.3622%`, MLP-down T36 candidate `15.5050%`, and MLP-down T2048 candidate
          `10.8517%`. Its one allowed
          frozen-byte repeat,
          `profiles/bench/r9700-a8g128-q4g128-n16k16-batched-repeat1-ab-20260905.json`
          (SHA-256 `2bcbd267a99bf0316695732d9431cde65d16706dc405de68985220ed23c04287`),
          measured `-0.423443 ms` with `[-3.595587,+2.748701] ms` and one GDN-output P2048
          candidate range outlier at `15.3891%`. Exact P2048 control/candidate medians were respectively
          `2.331719/2.334753`, `1.233698/1.220992`, `3.555652/3.565343 ms` in the first run and
          `2.345716/2.351652`, `1.235031/1.223824`, `3.562162/3.572732 ms` in the repeat for
          value-Z/GDN-output/MLP-down; their conservative ratio bounds were
          `[0.995490,1.007142]`, `[0.981957,0.997511]`, `[0.997276,1.008189]` and
          `[0.994598,1.010529]`, `[0.977557,1.004339]`, `[0.996479,1.009496]`.
          These are independent, non-pooled reports. Both robust point estimates and computed
          intervals remain below `3 ms`, but instability prevents treating either interval as a
          stable physical speed bound. The repeat allowance is exhausted, so no third run follows.
          Small-T wins are diagnostic only; no Q4G128 product or dual-scale compatibility lane
          follows, no live `>=30 ms` hypothesis remains, and the separate `>=2,000 tok/s`
          whole-model gate remains unchecked. Chunk selection remains blocked behind that unmet gate.
        - [x] Rebase the stopped P2048 recovery decision on the final N16/four-role production
          path and close the previously enumerated bounded exact-Q4 candidates. The current dense
          C1/P2048/chunk-4096/spec-none authority is
          `profiles/bench/r9700-four-role-n16k16-dense-g16-p2048-c1-spec-none-20260905.json`
          (SHA-256 `612d667eea61aebdee9a7d12add99bfd90d31b21f155b6114e0b3ca2361d0338`):
          `1,847.942898 tok/s` and `1.108835254 s`. Its selected-region attribution at
          `profiles/rocprof/r9700-n16k16-production-p2048-trace-20260905/analysis.json`
          (SHA-256 `3466e40d50ec96b9377db06791eda7f7947c1811f0be3a23d619b392490accbd`)
          measures the remaining 176 Q4 calls at `356.276973 ms` for
          `21.99023255552` useful TMAC, or `61.7223 TMAC/s`; the literal floor needs
          `84.835254 ms` less whole time. Two independent scalar-base/u32-voffset qualifiers
          consistently improved those three Q4 shapes by about `6.4%`, but their robust lower
          savings were only `22.833545/22.766320 ms`, below the frozen `25 ms` gate, and each run
          had one unstable control arm. Their immutable SHA-256 values are
          `1941b3c6a262a61928d3971fc2b0afe06f7dac16db68a4221e79ae42940afb9b` and
          `6501bc10d6dcbfb7638fd3ad977555231e263e3642b184ad15151f3feb9714dc`;
          the repeat allowance is exhausted and production remains unchanged. A4 failed its
          numerical-first 8K gate at `+0.070500086` mean NLL versus the fixed
          `0.048790164` limit (report SHA-256
          `cd449ecddbf7f2d260fb43d630b1274ae986631114bf6114c99f3253f0c6b8f5`).
          The final distinct M96xN256/twenty-four-wave exact-Q4 candidate passed full incumbent
          BF16 parity, sampled complete-K represented-FP64, tail/VMM guard, graph/status, and
          static hardware gates at 105 logical/112 allocated VGPR, 30,208-byte LDS, two active
          CTAs/48 resident waves, 16 native signed-IU4 sites, and zero spills. It nevertheless
          stably lost every physical cell by about `5%`--`6%`, delivering only
          `56.283693 TMAC/s`; its immutable terminal report is
          `profiles/bench/r9700-a8q4-n16k16-m96n256-p2048-ab-20260905.json` (SHA-256
          `c09f5b716cf8bce90ceac8e0c7c81e2621713610061582e0e66941a61cd9e20f`).
          Split-K and unchanged-tile persistent/cooperative forms are also terminal by traffic and
          launch-gap bounds. Qualification-only code is removed. A later independent review found
          the omitted M64xN192 topology below, so it received the final bounded qualification.
        - [x] Terminally reject the exact-Q4 M64xN192/twenty-four-wave topology compounded with
          scalar-base/u32-voffset addressing. The physical `auto` gate passed full incumbent-BF16
          parity, 126 complete-K represented-FP64 samples, both N=5,120 five-role VMM guards,
          wrapper/status/input/alias/graph checks, exact resource/ISA proof, and runtime occupancy
          at two CTAs/48 waves. All three isolated cells were faster and stable, but the exact
          48/64/64 call-weighted saving was only `9.1266002655 ms` with `0.5012054885 ms` robust
          uncertainty, leaving a robust lower bound of `8.625394777 ms` versus the frozen `25 ms`
          gate. Candidate service was `363.091501236 ms`, so the whole-floor diagnostic also failed.
          The immutable terminal report is
          `profiles/bench/r9700-a8q4-n16k16-m64n192-scalar-base-p2048-ab-20260905.json` (SHA-256
          `ece998b8688acf4a76aff5592bdecf26aedc9b3497448db76ac1cf03d0a91baf`). Production remains
          unchanged, qualification-only code is removed, and no adjacent M64xN160/N224 sweep is
          authorized. This closes the final bounded exact-Q4 topology and returns prefill work to
          the explicit representation/context/performance-contract decision above.
        - [x] Apply the user-authorized smaller-gain continuation without reopening rejected
          representations or topology families. A mechanism enters the composition pool only after
          the independent represented-format oracle, incumbent BF16 parity, status/tail/alias/guard,
          graph, and loaded-ISA/resource gates pass; every exact P2048 shape must have robust
          candidate/control upper ratio `<=1.01`, and the exact call-weighted robust saving lower
          bound must be at least `5 ms`. Permit one frozen-byte repeat only for instability or
          interval overlap, never to rescue a stable miss. Savings touching the same kernel, call
          set, addressing, LDS, or schedule are non-additive and require a directly measured
          compound. Trigger whole-P2048 confirmation at a directly measured `>=10 ms` robust
          operator lower bound, after three pool additions, or whenever a point estimate could
          cross the remaining floor deficit. Production promotion requires positive robust whole
          improvement of at least `10 ms`, no semantic/workspace/dispatch regression, and rebasing
          every later comparison on the promoted source. Stop after two consecutive stable compound
          misses, a remaining non-overlapping credible whole bound below `10 ms`, or selected Q4
          service at `>=120` useful TMAC/s or an equivalent physical ceiling bound.
          The first candidate is the scalar-base/u32-voffset addressing mechanism because its two
          retained sessions agree on about `23.8 ms` point / `22.8 ms` robust-lower Q4 saving while
          preserving the exact arithmetic, LDS, grid, ABI, and Q4G64/A8G64 representation. Integrate
          it into the actual product source, prove unsigned-offset bounds over every admitted
          shape/extent while retaining `size_t` output addressing, requalify the actual wrapper and
          loaded object, run one fresh exact `48/64/64` operator A/B, and immediately run a
          source-matched dense C1/P2048/chunk4096/spec-none whole A/B if it passes. Do not combine it
          with M64xN192: the retained compound's `8.625 ms` robust lower demonstrates destructive
          interaction. No other previously rejected mechanism is silently reopened; a subsequent
          candidate must be a new, predeclared hypothesis informed by the post-promotion loaded ISA
          and physical attribution.
          The product-source scalar-base candidate passed on `auto`: the actual-wrapper/static gate
          reports 88 logical/allocated VGPR, 17,152-byte LDS, eight native signed-IU4 sites, zero
          scratch/spills, exact scalar-base/single-VGPR-voffset loads, all 32 admitted shape/extent
          bounds, full-bit incumbent parity, represented-FP64 samples, graph/status/guard coverage,
          and a `21.057251555 ms` exact-call-weighted robust saving lower. Its immutable operator
          report is
          `profiles/bench/r9700-a8q4-n16k16-scalar-base-product-p2048-ab-20260905.json`
          (SHA-256 `325cad2e3c53b620f2864014bd16d3f4e510fdeb846b19a2c33ad31ee7c28fd8`).
          The source/emitted-compile-matched OFF/ON whole gate also passed promotion: control versus
          candidate median total was `1097.650069/1075.475953 ms`, robust total saving lower was
          `20.306733144 ms`, robust ratio upper was `0.9811942024`, all sixteen token vectors were
          exactly `[[271]]`, and workspace/environment identities matched. Its immutable report is
          `profiles/bench/r9700-scalar-base-production-p2048-c1-ab-20260905.json` (SHA-256
          `08463f48aae28aa0dd5ad4f5f05155a11deacda2d458db38e2f512a974be8e9f`). The promoted
          prefill median is `1075.438603 ms` / `1904.339303 tok/s`, so the 2,000 tok/s floor remains
          open by `51.438603 ms`. Scalar-base is now the canonical exact-Q4 product body under the
          final `m64n128-pingpong-n16-k16-scalar-base-production` profile. The qualification build
          selector, duplicate full-tile kernel/wrapper, and terminal qualification runners are
          removed; the authoritative 8-shape x 4-extent predicate plus exact four-plane U32 bound
          selects it, while every other valid direct tuple uses the predicated `size_t` fallback.
          The default final object retains 88 VGPR, 17,152-byte LDS, occupancy 16, zero
          scratch/spills, eight IU4 WMMAs, and ten scalar-base/single-U32-voffset loads.
          Two non-overlapping mechanisms are recorded for bounded future prefill work after the
          ordinary-decode and DFlash2 priorities. First, remap the canonical LDS index by adding
          word bit 2 to row bit 3 while retaining word bit 1 to row bit 4, giving
          `f=[0,0,16,16,8,8,24,24]`; the mapping is bijective, preserves two-address strides,
          reduces activation publication from 8 banks x 4 lanes to 16 x 2, and has an optimistic
          `8.39 ms` whole-weighted envelope at 2.5 GHz over 671,088,640 iterations. Second, pair the
          complete eligible accumulation stream with dual-FMAC, whose optimistic envelope is
          `5.84`--`7.18 ms`. Each remains future-only and must independently satisfy the existing
          `5 ms` exact-call-weighted robust-lower composition gate with every-cell upper ratio
          `<=1.01`; neither reopens the rejected topology or representation families.
        - [x] Record the initial design rejection of direct signed-A8 by unpacked-signed-Q4 IU8
          WMMA. The
          gfx1201 builtin is K16 with signedness controls `(true, A, true, B, C, false)`. A G64
          output fragment needs four IU8 WMMAs, exactly the same native instruction count as the
          current two K32 halves by low/high IU4 planes. It removes the final `low + 16*high`
          combine and one accumulator bank, but adds A8 reconstruction, signed-Q4 expansion, and
          one serial accumulator chain; on-chip unpacking leaves global traffic unchanged and
          raises M64xN128 LDS from 8,576 to 12,672 bytes by doubling staged weight storage. A
          separate materialized unpack plane would also double weight traffic. This has no
          credible material bound, much less the roughly `2.5x` Q4 reduction now needed for the
          whole-prefill objective, so no qualification kernel was added at this initial stage. The
          later matched ceiling and fair-pipeline static qualification below supersede this
          design-only disposition.
        - [x] Reassess that IU8 rejection against the externally proposed `152.5` useful
          split-IU4 versus `148` native-IU8 TMAC/s comparison. The matched M64xN128/G64 model in
          `profiles/bench/r9700-a8q4-native-iu8-design-reassessment-20260904.json` accepts those
          values as a favorable hypothesis rather than a local measurement. They do not overturn
          the decision: one wave still needs eight matrix instructions for its two N16 outputs,
          either `2 K32 halves * 2 activation planes * 2 outputs` with IU4 or
          `4 K16 quarters * 2 outputs` with IU8. The proposed ceilings therefore put IU8
          `2.95%` below split-IU4 before overhead. On-chip conversion preserves the same 8,576
          global payload bytes per CTA/G64. Expanding Q4 once before LDS raises ping/pong LDS from
          17,152 to 25,344 bytes and operand LDS reads by `1.5x`; retaining packed Q4 in LDS keeps
          those resource counts but repeats signed-nibble expansion in every consumer wave. A
          direct-A8 qualification quantizer could avoid reconstructing the current split planes
          without changing activation bytes, but it does not remove either weight-expansion choice
          or the change from four two-WMMA accumulator chains into two four-WMMA chains.
          Restoring four chains also restores an I32 combine and forfeits the accumulator saving.
          No native mixed A8/I4 or nibble-to-byte instruction exists on gfx1201, so the exact IU8
          formulation reduces neither matrix issue nor global traffic and remains rejected without
          inventing a physical benchmark or adding a qualification kernel. The supplied comparison
          therefore does not reopen an IU8 challenger. The only bounded experiment capable of
          overturning this static rejection would be one qualification-only, exact-P2048 matched A/B
          over `[N,K]=[5120,17408]`, `[5120,6144]`, and `[12288,5120]`, weighted by the observed
          `64/64/48` production calls. It would compare the current split-IU4 CTA with both legal IU8
          staging choices, require exact final-BF16 parity and complete activation-preparation plus
          matrix timing under `auto`, and require aggregate matrix service at or below
          `387.093037 ms` from the fresh `417.093037 ms` matrix baseline to save 30 ms. The later
          local ceiling measurement and static fair-pipeline qualification below supersede this
          pre-implementation decision; no GPU timing was admitted because the complete candidate
          failed its resource gate first.
        - [x] Prepare the focused production-Q4 P2048 dispatch-PMC attribution after exhausting
          structural candidates. The original single-pass command aborted before publishing data:
          gfx1201 advertises 16 SQ PMC slots but the installed register table contains only eight,
          so its eleven SQ events reached an absent ninth descriptor despite `pmc-check`. The failed
          attempt is terminal no-data evidence at
          `profiles/rocprof/r9700-production-q4-p2048-pmc-plan-20260904/runtime-failure.json`
          (SHA-256 `910beebbbc559f72417fbac70d84cb45b0af5dfb3f046b9a24bbc3b4ba97c798`),
          and the power profile was restored to `auto`. The replacement command-only plan is
          `profiles/rocprof/r9700-production-q4-p2048-pmc-two-pass-plan-20260905/plan.json`
          (SHA-256 `71479ff4f5a9c7fadf3e6d519f7cb9957ace1f4fb3d91eb28cfb75b5132ab224`),
          with guarded runner SHA-256
          `90079df7a90eb94a444a10ff0f7777b80054e0dc5ad6ae96cc35bb628e3f07d8`,
          fixed-name postprocessor SHA-256
          `3447218e28343ae35aaefd616072526874dca8c36af34f77aa914ffb4db6007d`, and analyzer SHA-256
          `55d17d882330b6d24dad35a18874a7b5cd1648b7a5fc8647846a9f5c67bd8b1c`. It filters only
          `a8q4g64_linear_prefill_cta_kernel` in the selected measured region and expects the exact
          `[2048,5120,17408]`/64, `[2048,5120,6144]`/64, and
          `[2048,12288,5120]`/48 call inventory, separated by ROCTX stage and launch geometry.
          Two independently guarded identical workloads split the SQ block to four and eight events.
          Each pass must independently validate the complete dispatch/shape/resource inventory and
          its positive controls; dispatch IDs and ratio operands are never joined across passes.
          Together they cover GL2C/TCP hit ratios, relative GL2 read requests, vector/VMEM/LDS issue,
          LDS conflict, broad waits, write backpressure, and occupancy clues. They cannot measure physical read
          bytes/bandwidth or isolate WMMA cycles on this installed gfx1201 stack: the required GL2
          request-size buckets are known-zero and no `SQ_INST_CYCLES_VALU_WMMA` event exists. Combine
          it only with exact static IU4 counts and the already-retained `auto` timing; all PMC
          durations remain attribution-only. Each pass requires a temporary sudo switch from `auto`
          to `profile_standard` and restores and verifies `auto` after that pass or any failure.
          Both passes then completed with exact independent 176-dispatch inventories; the valid
          attribution is `profiles/rocprof/r9700-production-q4-p2048-pmc-two-pass-plan-20260905/attribution.json`
          (SHA-256 `0385a2f226c7f31f36184d4bfca9584b639183710386e0e5076029085874d646`).
          It binds both exact embedded workload commands and rocprofiler-sdk `1.3.5`
          (`6b0e43f341195e203754e08f850e437ff2fc09f9`, ROCm `10.0.0`). Official gfx1201
          occupancy is retained as residency, while the unavailable per-instance TA/GL2 maxima and
          VALU/VMEM/wave-life counts are labeled instance-sum or per-wave relative activity rather
          than utilization. Across exact shapes, occupancy is `94.94`-`98.22%`, dependency wait is
          `70.95`-`94.72%`, issue wait is `3.97`-`24.98%`, GL2 hit is `84.25`-`95.56%`, and the
          gfx1201 L0 vector-cache hit is `1.73`-`4.00%`; these establish locality and dependency
          pressure but not physical HBM bandwidth or causal stall decomposition.
        - [x] Qualify and terminally reject the strongest source-level latency-hiding response to
          that PMC signature: a two-bank/two-entry register FIFO retaining the current `17,152`
          LDS bytes, `8,576` represented global bytes/G64, eight IU4 WMMAs, and one publication
          barrier. The best full scalar form met the static resource gate at `94` VGPR,
          occupancy `16`, and zero private/scratch bytes. A recognized
          `llvm.amdgcn.s.wait.loadcnt(4)` emitted after the current eight WMMAs, however, was
          immediately followed by compiler operand waits `2/1/0` before oldest publication; loop
          rotation independently added waits `2/0` before the next compute, draining all four
          younger requests. The peeled rescue rose to `118` VGPR/occupancy `12`, direct
          buffer/global-to-LDS is not a gfx1201 feature, and a third LDS bank does not extend the
          global-load distance. The immutable no-GPU rejection is
          `profiles/bench/r9700-a8q4-prefetch2-static-rejection-20260905.json` (SHA-256
          `4a4ffd1c0f5d759fd6debff02987572a233a2bb599ccc965f0231386d56e582b`). Production is unchanged.
        - [x] Attempt and terminally close the one selected-dispatch ATT fallback. Under installed
          rocprofiler-sdk `1.3.5`, the exact R9700 GPU-index-0/SE0/WGP1/SIMD0 P2048 dispatch filled
          an exact `1,610,612,704`-byte trace from a `1,610,612,736`-byte buffer and emitted SQTT
          plus thread-trace buffer-full diagnostics. The invalid/no-authority receipt is
          `profiles/rocprof/r9700-production-q4-p2048-att-retry3-20260905/overflow-failure.json`
          (SHA-256 `da77c12c2659556c793ba714a1092affcd44519a2255f3578b0aaddf185dc428`). The
          prior 384-MiB overflow, parser-rejected command, and wrong-gpu-index no-data attempts are
          also retained only as invalid receipts. Do not decode or interpret any of them. The CLI
          has no narrower intra-dispatch trace scope, so do not enlarge the buffer or retry this
          ATT route. Keep the valid two-pass PMC report as attribution authority; production stays
          on the selected M64xN128 ping/pong Q4 route. The PMC's `70.95`-`94.72%` broad dependency
          waits, high residency, and healthy GL2 locality are non-causal clues, not PC-stall or
          bandwidth proof. The production aggregate's `21.99023255552` TMAC / `417.093037 ms` =
          `52.722608` useful TMAC/s versus the isolated `200.880451` TMAC/s IU4 ceiling remains
          unexplained headroom. Mark this selected-Q4 iteration tool-limited and terminal; do not
          reopen another Q4 architecture. Return the active performance lane to the pending
          chunk/whole-profile selection.
        - [x] Qualify and terminally reject the scheduling-only dependency-latency candidate.
          The disconnected two-bank route added a zero-instruction value anchor and
          `sched_barrier(0)` after each current-G64 totals update. Its emitted gfx1201 loop placed
          all eight IU4 WMMAs, 16 recombinations, 16 I32-to-FP32 conversions, 16 scale products,
          and 16 FP32 accumulations before the first successor publication drain while retaining
          exactly 17,152 LDS bytes, 96 VGPR, occupancy 16, and zero private/scratch storage. The
          independent FP64, full incumbent/candidate bit parity, tail, alignment, and nonfinite-
          status regression passed. The direct `auto` exact-shape P2048 A/B then regressed the
          48/64/64-call aggregate from `394.126793` to `401.131914 ms`, ratio `1.0177738`, for
          `-7.005121 ms` saving; the three per-shape ratios were `1.000735`, `1.018073`, and
          `1.026857`. The terminal report is
          `profiles/bench/r9700-a8q4-epilogue-schedule-p2048-ab-20260905.json` (SHA-256
          `3a2430a6bb07850d6f49d0d9ea62e695b1103838f578ee7ebbe6ed0d3aaf7ffc`). The isolated
          harness/kernel/header/checker and build products are removed; production remains the
          selected M64xN128 ping/pong route.
        - [x] Reopen, qualify statically, and terminally reject the direct-A8 / packed-W4-to-IU8
          route. The matched local register-only report
          `profiles/bench/r9700-q4-hardware-peak-iu8-20260905.json` (SHA-256
          `1e4f5a5ac863a1939e560794758e335692998121bc5172a6f5696f1af8653756`) measured
          `200.880451` useful split-IU4 TMAC/s and `195.123788` topology-matched IU8 TMAC/s, a
          `2.865716%` IU8 deficit that closely confirms the supplied `152.5` versus `148`
          comparison. The disconnected qualifier independently derived direct-A8 codes, FP16
          scales, status, fused-SiLU BF16 boundaries, full tail output, and fault/no-clobber cases.
          Its fair expand-once ping-pong kernel retained packed Q4 globally, loaded each packed
          dword once, used two `12,672`-byte LDS banks, and emitted exactly eight signed IU8 WMMAs
          plus one publication barrier, but compiled first to `115` and finally to `111` VGPR at
          occupancy `12`, failing the fixed `<=96` VGPR / occupancy-`16` gate. Scheduling barriers
          removed the four-K16 operand hoist but could not eliminate the deferred expansion peak;
          the best legal nine-VALU compact expansion still bounded the route near `100` VGPR.
          The immutable rejection is
          `profiles/bench/r9700-a8q4-direct-iu8-pingpong-static-rejection-20260905.json` (SHA-256
          `d4f2c610b973935aa60b732c3ee39e9dc84a512781839458a5caadcca78f2a2c`). No GPU timing or
          production change was admitted, all qualifier surfaces were removed, and counters remain
          required only to choose a non-IU8 next Q4 architecture.
        - [x] Reject a Q4G64 CTA-row-tile/group-major artifact relayout at the design boundary.
          In the selected M64xN128 kernel, each eight-lane subgroup already fetches one exact
          contiguous 32-byte G64 code slice, and four consecutive groups fill the same 128-byte
          row line. Reordering the same payload can group a wave's addresses more closely in
          time, but it changes neither the unique 32-byte sectors nor the amortized 64/128-byte
          cache-line count: the K5,120/N128 CTA consumes 327,680 code bytes plus 20,480 scale
          bytes under either layout, with the same equality at K6,144 and K17,408. The selected
          next-group loads already issue ahead of eight native-IU4 WMMAs. Thus there is no byte,
          matrix-issue, or launch-count bound for the required 30 ms saving from the traced
          421.413627 ms Q4 service. The relayout would also replace the direct row view across
          converter, artifact, binder, and kernel indexing and would make decode's successive
          per-row G64 slices stride by a 4,096-byte tile. Duplicating both layouts or runtime
          repacking is outside the product contract. No artifact identity, converter path, or
          qualification kernel is added.
        - [x] Reject a wave-specialized producer/consumer variant at the design boundary. The
          retained ping/pong ISA already proves next-group global loads outstanding across current
          WMMAs and measured only `1.23224x`; dedicated producer waves neither reduce the four-WMMA
          per-fragment/G64 floor nor represented traffic, while consuming waves and CTA residency.
          The N256 structural traffic reduction also realized only `1.20609x`. There is therefore
          no credible path for producer specialization to approach the required `~2.5x`.
        - [x] Reject the signed-A4G64 x signed-Q4G64 M64xN128 candidate before model gates. The
          qualification-only kernel halves native IU4 issue from eight to four instructions per
          loop, activation code storage/reads from one byte to one nibble per K, and LDS from
          8,576 to 6,528 bytes; gfx1201 compilation reports 84 VGPR, occupancy 16, and zero
          private/scratch. The independent host codec image, bit-exact signed-A4 WMMA reference,
          separate A4/A8 FP64 represented-code formulas (maximum two BF16 steps), tails/status,
          and all 32 production tuples passed before timing. Weighted P2048 improved only from
          `1121.674698` to `1006.050599 ms` (`1.11493x`), below the fixed `>=1.5x` operator gate,
          and N1024 regressed at every extent by ratios `1.21819` through `1.35600`. The terminal
          report is `profiles/bench/r9700-a4q4-prefill-m64n128-vs-a8-production-ab-20260904.json`,
          SHA-256 `2ac938b230197384a476bd57ccfd39d72a1b73c03bee82c58bf3be9df7de5b1f`.
          Production remains A8; PPL/token/whole gates were not run, particularly because the
          existing A4 quality evidence already rejects that activation profile.
        - [x] Close the installed-library packed-INT4 architecture audit without adding another
          weak challenger. CK Tile and rocWMMA expose homogeneous packed-I4 gfx12 WMMA fragments,
          but no mixed A8/packed-Q4, direct per-G64 scaled GEMM. CK's group-quant pipelines convert
          packed inputs to FP8/BF8 compute, hipBLASLt has no gfx1201 I4 solution family, and the
          CK two-stage pipeline's direct `global_load_lds` path is CDNA3-only. The remaining gfx12
          cache modifiers are hints, while production already has full K32 IU4 utilization and
          the physically rejected ping/pong and N256 schedules bound staging/mapping-only gains at
          `1.23224x` and `1.20609x`. No local-library-derived design has a credible `>2x` bound;
          further A8Q4 work requires a genuinely new target-specific kernel architecture or a
          separately qualified artifact arithmetic change.
          A direct ROCm-10/gfx1201 recheck makes the arithmetic limit explicit: CK Tile and
          rocWMMA declare only homogeneous `i8*i8` K16 and `i4*i4` K16/K32 integer builtins, and
          llvm-mc accepts `v_wmma_i32_16x16x32_iu4` while rejecting candidate mixed-width
          `i8*i4`/`iu8_iu4` mnemonics. For each wave's two output fragments and one G64 group,
          exact A8 reconstruction therefore needs eight IU4 instructions (two K32 halves times
          low/high activation planes times two fragments). Expanding Q4 weights to I8 merely
          substitutes eight K16 IU8 instructions, doubles weight bytes, and raises ping/pong LDS
          from 17,152 to 25,344 bytes before unpack overhead. Nor can Q4G64 integer dots be summed
          before scaling: every group owns a generally distinct represented
          `activation_scale[g]*weight_scale[row,g]`, so combining groups changes the public
          represented formula. The only bounded amortization candidate is the separately screened
          Q4G128 format, which halves scale requests and FP32 group accumulations but worsens
          sampled aggregate relative-L2 by `1.104192022x`; it remains gated on BF16-source PPL and
          exact-token evidence before any artifact/runtime implementation.
        - [x] Retain matched native-IU4 issue and Q4G64 code/scale streaming ceilings, and use
          them only as roofline references. The qualification-only probe report is
          `profiles/bench/r9700-q4-hardware-peak-20260904.json` (SHA-256
          `72e4c91eb9177ff5969cce6c06febde251a4852e1e69bd317b77646ddcb7baa4`). Under `auto`,
          its register-resident eight-chain K32 IU4 kernel measured `801.669240` median issued
          TOPS, while its 272-times-L2 separate packed-code/FP16-scale stream measured
          `633.264977 GB/s` (`98.9477%` of the nominal 640 GB/s bus rate). Static qualification
          binds eight signed/signed unclamped `v_wmma_i32_16x16x32_iu4` instructions, 122 VGPR,
          occupancy 10, and zero private/scratch for the issue probe; the stream has exactly two
          128-bit read sites at the represented G64 weight ratio of 32 code bytes to two scale
          bytes, 16 VGPR, occupancy 16, and zero private/scratch. The report's `compute_units: 32`
          is the raw HIP `multiProcessorCount` scheduler value on this gfx1201 runtime, not a
          revision of the architectural 64-CU hardware fact.

          Reconciliation against the production M64xN128 weighted P2048 operator aggregate uses
          `101.082` tera-issued-operation-equivalents and `842.961535 GB` of source-requested
          traffic over `1.109850230 s`. Those give `91.077154` issued TOPS, or `11.3609%` of the
          measured register-resident ceiling. The source-request rate is `759.527288 GB/s`, or
          `119.938%` of the matched streaming ceiling. That value is deliberately not reported as
          physical HBM bandwidth: the traffic model counts represented source requests (including
          activation, output, status, and repeated CTA requests), whereas caches, coalescing, LDS
          reuse, and overlap determine physical transactions. The ceiling run proves that native
          IU4 and near-nominal memory streaming are available and that production is not saturating
          the isolated IU4 issue ceiling; it does not quantify production HBM traffic or identify
          scheduler/memory stalls. Final selected-route GL2C/TCP/SQ attribution therefore remains
          in the later profiling item.
        - [x] Reject a global-FP32-partial, two-stage large-M/N split-K design before
          implementation. The strongest viable large tile is M128xN128 with 13 G64 groups
          (`Kchunk=832`): its one 64-KiB LDS image is exactly 65,024 bytes, comprising an
          8,448-byte full-M activation group and 56,576 bytes of packed weights/scales. Sixteen
          waves can retain the production-sized 16 FP32 accumulators per lane and emit the two
          M64 slices sequentially. At P2048, the exact split counts for K=5,120/6,144/10,240/17,408
          are 7/8/13/21. Across the declared weighted operator inventory, it preserves the exact
          `202.164110622720` tera-issued-operation-equivalents (`12,339,118,080` dynamic native
          IU4 WMMAs) and `789.703557120` billion scaled group FMAs, reduces CTA barrier encounters
          from 96,399,360 to 48,199,680, and adds 58,818,822,144 FP32 reduction additions.

          The corresponding exact source-request accounting is 617.217261568 GB of operands and
          status, 267.630149632 GB of FP32 partial writes, the same 267.630149632 GB of partial
          reads, and 16.177430528 GB of final BF16 writes: 1,168.654991360 GB total versus
          production's 842.961534976 GB. The largest P2048 partial workspace is 1.996488704 GB
          (7 planes for N=34,816/K=5,120), far beyond the 8-MiB L2; it becomes 7.985954816 GB at
          P8192. Even granting nominal 640 GB/s, perfect stage-one compute/write overlap, and zero
          operand traffic, the mandatory partial write followed by reduction read/final write has
          an optimistic 0.861621453-second service floor, limiting the current 1.109850230-second
          aggregate to at most 1.28810x. Every M>=128, N>=128 alternative has at least as many
          split planes under the same LDS bound or less operand reuse; M256xN128/Kchunk704, for
          example, rises to 1,144.813015040 GB and a 0.988420506-second partial-only floor.

          Chunk-local FP32 FMA sums followed by ordered chunk addition are deterministic but not
          bit-identical to production's single per-G64 FMA chain. Preserving that exact association
          requires one partial per G64 group: 3,158.814228480 GB each to write and read, up to
          22.817013760 GB of P2048 workspace, and a 9.896571699-second nominal-bandwidth floor.
          Thus neither admissible numerical interpretation has a credible `>2x` bound, and no
          qualifier, workspace contract, or production path is added.
        - [x] Reject packed-Q4-to-BF16 LDS expansion as the dense-prefill replacement. A bounded
          exact-shape `[2048,34816,5120]` structural probe retained resident signed-Q4/FP16-scale
          weights, consumed represented BF16 activations directly, expanded each G64 weight tile
          to BF16 in double-buffered LDS, and accumulated with native gfx1201 BF16 WMMA. Its nine
          independent probes matched the explicit BF16-materialized-weight FP64 formula exactly
          (not the canonical Q4G64 formula), but the complete
          challenger took `38.879177 ms` versus `7.311391 ms` for the current activation-quantize
          plus A8Q4 path (`0.188054x` control-over-challenger speedup). The emitted kernel required
          exactly `65,536` LDS bytes, `144` VGPRs, and 512 threads. On-chip scalar dequantization
          and the resulting residency loss dominate; the exploratory target/source were removed.
          This rejects that decomposition only. Further Q4 work retains native IU4 WMMA and must
          improve its staging/issue behavior at the complete-operator boundary.
        - [x] Reject and remove the 16-wave M128xN128 A8Q4 challenger after exact and physical
          qualification. Each wave owned a 2x2 token/row fragment set and evaluated its two token
          fragments sequentially, halving persistent-weight global rereads relative to production
          without the earlier M128xN128 route's 1,024-thread CTA. The complete independent FP64
          represented A8Q4G64 oracle passed for all eight real Text/MTP shapes at
          T=1,024/2,048/4,096/8,192, including tail delegation and nonfinite-status poisoning.
          It compiled to 124 VGPR, 12,800 bytes LDS, zero scratch, and a 512-thread launch bound.
          The retained `auto` A/B report
          `profiles/bench/r9700-a8q4-prefill-cta-m128n128-w16-ab-20260904.json` has SHA-256
          `021e331c44f759fdc0589d67313bda033f8ae803e417592b6c24dc376ca02a2e`.
          The challenger lost every one of 32 measured shape/extent cells; weighted P2048 service
          regressed from `887.735771` to `1,157.479890 ms` (`1.303856x`). Reduced global weight
          requests do not repay the longer per-wave dependency schedule and reduced latency hiding,
          so the qualification-only implementation is removed and production remains ping/pong.
        - [x] Reject and remove 8-wave M64xN128 consumer consolidation after two exact physical
          variants. Each wave owned a 2x2 token/row microtile while preserving production's CTA
          grid, two-bank LDS, native IU4 count, scale order, and represented formula. Full-output
          parity plus the independent FP64 oracle passed across the same eight real shapes and four
          token extents. The first form retained next activation and weight payloads across WMMA,
          compiled to 165 VGPR/17,152-byte LDS/zero scratch, and regressed weighted P2048 from
          `910.749758` to `958.495028 ms` (`1.052424x`), losing all 32 cells. Its report SHA-256 is
          `134f2cd17df05d65395a4fdb8e34018bb4453fa902e5cf8653165ddfe1256285`.
          A final form retained only the likely-HBM weight prefetch and deferred the cacheable
          activation loads, reducing allocation to 162 VGPR but regressing further to
          `979.352448 ms` versus `910.981094 ms` (`1.075052x`), again losing all cells. Its report
          `profiles/bench/r9700-a8q4-prefill-cta-m64n128-w8-weight-prefetch-ab-20260904.json`
          has SHA-256 `ae86ab9cf9ce84f4920141c3868168d5bb337d5ecc4f1f2f3b8a77b2222e4c09`.
          Halving consumer waves removes too much latency hiding; the route is removed.
        - [x] Reject and remove the 16-wave M128xN64 A8Q4 challenger against the actual selected
          M64xN128 ping/pong production control. The candidate used a 128-token by 64-row CTA so
          every wave owned two M16 token fragments for one N16 row fragment, interleaved both
          token fragments across each K32 half, and halved persistent-weight CTA rereads. The
          independent complete FP64 represented formula, bit-exact production parity, tails,
          alignment, and nonfinite-status poisoning passed over all eight real Text/MTP shapes at
          T=1,024/2,048/4,096/8,192. Symbol-local assembly contains the required eight native IU4
          WMMAs; resources are 96 VGPR, 21,248-byte LDS, 512 threads, and zero scratch.
          Direct matched `auto` timing nevertheless lost all 32 cells. Weighted P2048 service
          regressed from `918.109962` to `1,048.477159 ms` (`1.141995x`). The corrected retained
          report is `profiles/bench/r9700-a8q4-prefill-cta-m128n64-w16-ab-20260904.json`, SHA-256
          `fec88f2e8da42de75c9a5d1bf76422518a67f692095f0e6fed4791a15d530e38`.
          An earlier report that mistakenly timed the single-bank regression control while
          labeling it ping/pong was deleted rather than retained as evidence. A whole-model run
          against an older, non-source-matched baseline was likewise discarded: it could not
          attribute its apparent difference to a candidate that is directly slower. Production
          remains M64xN128 ping/pong.
        - [x] Admit one directly bound E4M3 hipBLASLt profile only as a bounded escape-path
          evaluation candidate. The installed hipBLASLt 1.4.1 gfx1201 catalogs contain
          E4M3/E4M3, BF16-output, FP32-HPA solutions: four scalar-scale transpose families have
          536/559/224/779 solutions, and the ordinary-layout family has 11 FP32 outer-vector-scale
          solutions. The decoded records bind unswizzled A/B and ordinary leading dimensions, so
          a row-major K128-padded E4M3 plane plus one FP32 scale per output row can be stored in and
          bound directly from a new artifact; activation uses one FP32 scale per token. There is no
          persistent prepack API or accepted inner-K block-scale catalog for this route, and no
          runtime weight repack is permitted.

          The existing inventory implies 28,503,368,096 tensor bytes for 28,424,681,472 padded
          E4M3 matrix positions, 4,874,224 FP32 row scales, and 59,189,728 unchanged direct bytes.
          That leaves 4,782,628,448 bytes after the default 1-GiB reserve on 32 GiB. A conservative
          four-byte-per-output heuristic workspace is 285,212,672 bytes at P2048 and
          1,140,850,688 at P8192 for N=34,816; the largest P8192 activation image is 142,639,104
          bytes. This passes a static residency plausibility gate, not resolved capacity.

          P2048 dense linear work is 101.08205531136 TFLOP-equivalents. The 2,000 tok/s whole gate
          permits 1.024 seconds, requiring 98.713 TFLOP/s if linear work consumed all of it or
          197.426 TFLOP/s if half remains for other Ops. Scaling the measured 48.930-billion/s
          native-WMMA issue rate by the K16 FP8 instruction's 8,192 operations gives a non-measured
          architectural reference of 400.835 TFLOP/s, making the half-budget case 49.3% of that
          scale. Exact-shape heuristics and physical speed are still unproven. E4M3 rowwise quality
          cannot be inferred from W8G32: any implementation first requires a distinct BF16-source
          codec/oracle and the same 8K/32K PPL/severe-position gates. This audit is a go only for
          one isolated evaluation candidate, not an artifact, product route, or selection result.

          The implementation boundary is now pinned in
          `docs/maintainer/gfx1201-low-precision-operations.md`: a distinct
          `F8E4M3_ROW_F32S`/`row-scaled-k128-v1` artifact layout, direct K128-padded code and FP32
          row-scale plane binding, caller-owned E4M3 activation/token-scale/hipBLASLt workspace,
          and a Program-owned `ops::LinearExecution` with explicit pre-execution heuristic
          preparation. No function-static handle/cache or per-call descriptor allocation is
          admissible. The first vertical slice is the real
          `text/layers/0/mlp/gate_up[34816,5120]` projection at T=2048 with exact codec/layout,
          independent decoded FP64-oracle, direct-pointer, workspace, heuristic-support, and
          matched-Q4 timing gates. Do not convert/register the full 439-matrix candidate until
          that slice passes.
        - [x] Reconcile the selectable gfx1201 instruction catalog against the installed ROCm 10
          headers rather than datatype names alone. The active catalog in
          `docs/maintainer/gfx1201-low-precision-operations.md` now includes native IU4 K16/K32,
          IU8 K16, FP16/BF16 K16 with FP32 or narrow accumulation, all four FP8/BF8 K16-to-FP32
          WMMA forms, and native FP8/BF8 encode/decode conversions. It records the absence of a
          mixed A8-by-I4 instruction, nibble-to-byte/group-scale fusion, and gfx1201 asynchronous
          global-to-LDS copy. It also records the available temporal/cache-scope hints and separate
          load/store/DS wait domains. FP32 K4, FP16/BF16 K32, IU8 K64, FP8/BF8 K64/K128,
          F4/F6/MX scaled forms, matrix-A/B reuse controls, async-copy wait, and non-speculative
          cache hints are explicitly gfx1250-only and cannot enter an R9700 candidate. Every
          selected route still requires symbol-local emitted-ISA, resources/stalls/cache, operator,
          and whole-schedule evidence; the catalog is an admissible-choice inventory, not proof a
          named builtin was used.
        - [x] Implement the isolated converter-side rowwise-E4M3 codec vertical slice without
          adding an artifact inventory or runtime route. The scalar oracle independently exhausts
          the finite E4M3FN codebook with RNE/even ties, finite saturation, signed zero, and
          malformed-word rejection. The CPU vector encoder requires represented BF16 `[N,K]`,
          stores direct row-major E4M3FN bytes with canonical zero-filled K128 padding, and aligns
          the FP32 row-scale plane to 256 bytes. Vector bytes match the scalar oracle exactly at
          Qwen K=5,120/6,144/17,408. Eight real source rows from MLP gate, attention output, MLP
          down, and LM head measure relative-L2 error
          0.026347/0.026398/0.026425/0.026473 respectively, with maximum sampled absolute errors
          0.001726/0.002686/0.001744/0.003069. This establishes only the codec/numerical boundary;
          complete evaluation conversion, runtime binding, Op qualification, and 8K/32K quality
          gates remain open.
        - [x] Map the complete target-private evaluation inventory and add the bounded gate/up
          fixture boundary without registering or converting the candidate. All 439 source
          non-direct matrix objects retain their names/order/shapes and map to
          `F8E4M3_ROW_F32S`/`row-scaled-k128-v1`; the 582 BF16, 96 FP32, one I32, and six resource
          objects remain unchanged. The resulting tensor plan is 28,503,368,096 bytes and its
          256-aligned device arena is 28,503,382,016 bytes. The isolated fixture writer admits
          exactly `text/layers/0/mlp/gate_up[34816,5120]` under an evaluation-only identity and
          atomically streams its exact 178,397,184-byte payload. The CPU test reopens that artifact
          and proves its sole object metadata, direct E4M3 code bytes, K padding, FP32 scale offset,
          and complete payload digest. Full-model conversion remains prohibited until the real
          single-object Op gate passes.
        - [x] Add the target-private full evaluation converter CLI without running conversion.
          `convert_e4m3.py` admits only `r9700-f8e4m3-row-eval`, refuses overwrite, performs the
          complete source/frontend/ranking/object-plan preflight before opening output, and streams
          row-chunked E4M3 code payloads through the atomic container writer while retaining only
          the small FP32 scale plane. Its no-output CLI preflight passes the real 18-shard source:
          1,199 BF16 source tensors, six hash-qualified resources, 1,124 objects, 439 row-scaled
          matrices, 28,503,368,096 tensor bytes, and a 28,503,382,016-byte aligned arena. A bounded
          synthetic safetensors/resource conversion test reopens the artifact and proves its
          explicit evaluation identity, exact object shape/format/layout, scalar-oracle payload,
          conversion receipt, and no-overwrite behavior. The full evaluation artifact remains
          intentionally unmaterialized pending the single-object Op gate.
        - [x] Implement the decision-owned role-consistent E4M3/Q4 artifact and execution boundary.
          Gate-up is the admitted initial slice, not the terminal role inventory. The
          retained pre-`LinearExecution` decision report
          `profiles/bench/r9700-fp8-gate-up-decision-shared-source-20260904.json`
          (SHA-256 `3da19d842bba606b76fbd12625e8c5b65fa32e3bc64b74144865e92cc7a02809`)
          admits the 64 `text/layers/{0..63}/mlp/gate_up[34816,5120]` objects: their
          complete T=2,048 route is `1.671757x` faster than production Q4 and increases resident
          bytes by `5,356,650,496`. Fixed-shape qualification of the 16 full-attention query-key,
          16 full-attention gate-value, and 48 GDN query-key objects is complete: FP8 beats Q4 by
          `1.439930x` at `[2048,7168,5120]` and `1.464192x` at `[2048,4096,5120]`, with zero BF16
          steps at all nine represented-format probes. The terminal role decision
          `profiles/bench/r9700-fp8-post-gate-up-decision-20260904.json` (SHA-256
          `7d12a2d962606d967d905c010cf0b506398b3dd0be09841df9cdd62ee8563128`) selects all
          three additions. Replacing those 144 objects in total adds `6,380,716,032` bytes and
          projects only `1,536.993` whole-P2048 tok/s; it cannot close the 2,000 tok/s floor alone.
          Its claimed `2,004,481`-byte P=8,192/G16/C4 remainder is invalid because the historical
          capacity input predates the current N16 artifact, selected chunk, and exact ordinary
          physical-capacity contract and is not admission evidence.
          The owner refactor deliberately changed the shared qualifier source and rebuilt the gate
          and attention executables, so these retained reports no longer pass the live
          source/executable validator. Their numerical/performance result remains bounded design
          evidence only; fresh reports from the prepared owner must reproduce it before promotion.
          The current gate source/executable SHA-256 values are
          `3d1220e07d3ca0c0789798811759520af0aafaa732ef28f919437c68f358b8f0` and
          `fd8da33fc8a3655696eeec66e319980d88daa4b62692a0524ddc47e422a8888c`; the current
          attention executable SHA-256 is
          `b8c1b6581f52edd253066f7ab30c204942e248210e2b0719a043d27823046118`.
          Fresh prepared-owner physical qualification now reproduces every role decision under
          `auto`. Gate-up FP8 is `4.188108` versus `6.853187 ms` Q4 (`1.636344x`), attention
          query-key/gate-value is `1.044207` versus `1.493470 ms` (`1.430243x`), and GDN
          query-key is `0.610164` versus `0.892485 ms` (`1.462698x`); all represented-format
          probes are zero BF16 steps and direct binding, nonfinite poisoning, and no-clobber gates
          pass. The report SHA-256 values are
          `290dfd9229b710996008b997fe34f11ce7932892c84cf1e6653eeb31a82ea661`,
          `5ae29ddabb70974c758cace97cbf3e532dfe1737f2c0b05328dc88aa4edcd50c`, and
          `8e24c3c52422c0ed4f64aa419835eb8d227b0834f8c387d3151294a7e6552b5d` respectively.
          Regenerated gate and four-role decisions both emit `proceed`, with SHA-256
          `bc9cae520c54cda55a4f2b6a7864cd5cb9ead45409784e082204a3c5f234f425` and
          `804513574cd45be17d71db13f866f259a4a734f8701c9f2ffc666ba0edebbda1`.
          Fresh symbol-local hardware proof now passes for both distinct algorithms. Gate-up proof
          SHA-256 `06136e3853e8831c9e51dbbb1d6130c3195cfecba07955f9fa12e34aa209bf3c`
          binds 16 stable FP8 dispatches whose exact symbol interval contains 112 native FP8
          matrix instructions and 16 Q4 controls whose production interval contains eight IU4
          WMMAs; its trace DB SHA-256 is
          `40a9d030fa6d0a302ee99cf156df86de60c11daa83fbcd925eba0a62e89a1f01`.
          The distinct attention proof SHA-256
          `84142eef7e3633b7d308296c0c70f97321f12616108b3922773866112b53a545`
          binds its selected `dce001...` algorithm interval with 160 native FP8 matrix
          instructions and the same eight-IU4 Q4 control; its trace DB SHA-256 is
          `d8b6110438d266309f2ad9adf783f2c06aa090bf9a542b91bfccf15a35a64b0b`.
          Artifact/runtime integration and host width planning are complete, and the evaluation
          recipe is now an explicit third schema-v7 base-selection branch with its own artifact,
          conversion-receipt, execution-profile, and prospective DFlash-companion identities. This
          closes the implementation boundary only: fresh graph/eager and speculative execution
          parity plus the prepared whole-performance/capacity matrices remain separately required
          before production promotion.
          A single
          decision owner must consume the completed role reports plus every capacity cell and emit
          the exact compile-time object-name set, format counts, byte totals, weights/recipe
          identity, and digest used by both converter and binder. Nothing before that retained
          decision may freeze `64 E4M3 + 375 Q4` as the final recipe. The result is one target
          identity and recipe, not a runtime dtype selector or a compatibility lane.

          Supporting executed-hardware evidence exists, but live admission is not complete. The
          retained pre-shared-source proof
          `profiles/rocprof/r9700-fp8-gate-up-isa-proof-hsa-20260904/proof.json` (SHA-256
          `eb0988a0a55e3e8a5af16e480dd8d315845df6ca320e62401f7b27921773b7df`) joins all 16
          traced complete calls per route through rocprof dispatch, kernel-symbol, and loaded
          code-object identities. The exact pointer/size-keyed 894,736-byte RocRoller ELF for the
          selected hipBLASLt kernel contains 112 native
          `v_wmma_f32_16x16x16_fp8_fp8` instructions in that symbol's bounded interval. The exact
          file-backed production Q4 symbol contains eight native
          `v_wmma_i32_16x16x32_iu4` instructions. It binds the overwritten pre-refactor executable
          and old gate report, so it is supporting symbol-local ISA evidence rather than current
          promotion authority. Refresh it against the shared-source executable/report, and capture
          the distinct selected attention algorithm, before production promotion.

          The qualification-only hardware-proof tooling now fail-closes over either the existing
          gate-up profile or the distinct fixed `[2048,7168,5120]` attention query-key/gate-value
          profile without changing the gate-up v1 plan/proof schemas. The prepare tool records an
          explicit profile, shape, executable, admission report, profiler, capture library, and
          fresh disjoint outputs; the validator applies the matching report contract and requires
          the traced algorithm identity before extracting its exact dispatched symbol. The prepare
          and validator source SHA-256 values are respectively
          `cc05cd753b72d86bd1718b4b5282fea5610265fff34e98e6e0e54c8e0f64dca5` and
          `3e02e7b990c13526ab71010386496ec7b66539e1e01d4d4ffbbb5863adb7c15e`;
          focused CPU tests pass. The validator recognizes both the common prepared-owner
          `poison_nonfinite_output` dispatch and the retained fixed-gate boundary; qualifier tests
          remain the semantic poisoning authority. This tooling prerequisite and both fresh traces
          are now complete.

          Implement it as a target-private inventory derived from the all-Q4 object plan, replacing
          only the decision-owned exact names with `F8E4M3_ROW_F32S`/`row-scaled-k128-v1`; give the
          hybrid a distinct weights/recipe identity and conversion receipt, and reuse the existing
          generic row-scaled descriptor, reader, materializer, and codec unchanged. Register one
          exact `WeightsProfile`; add explicit role-format functions for each selected main-Text
          binder; and leave unselected Text roles, MTP binding, `row_view`, DFlash companion
          binding, and generic `ops::linear` free of an E4M3 fallback. Under the all-four-role
          frontier the exact inventory is 144 E4M3 and 295 Q4 matrices with a
          `21,540,531,712`-byte aligned resident arena; the converter must derive rather than assume
          that conditional total.

          - [x] Complete the decision-owned artifact, converter, and binder slice without
            materializing the full model. The target-private
            `fp8_hybrid_selection.inc` is the single record for the distinct
            `r9700-q4g64-f8e4m3-four-role-eval` /
            `r9700-q4g64-f8e4m3-four-role-eval-v0` qualification identity and all 144 exact
            matrix names; its newline-delimited name-set SHA-256 is
            `b2ceeb63c581c0f26aab5a4d8c0958da34d836fcc5c47d377bce709eaf37e3e8`.
            Both the Python inventory/converter and compiled binder consume that record. The
            inventory is derived from the all-Q4 plan and proves 144
            `F8E4M3_ROW_F32S` matrices, 295 `Q4G64_F16S` matrices, unchanged remaining
            objects, 21,540,517,792 tensor bytes, and a 21,540,531,712-byte aligned arena.
            Explicit main-Text role binders select E4M3 only for all 64 MLP gate/up, 16
            full-attention query/key, 16 full-attention gate/value, and 48 GDN query/key
            objects; global Text, unselected Text, MTP, Vision, row views, DFlash, and generic
            Linear retain no E4M3 fallback. A metadata-complete sparse binding qualification
            proves the 1,124-object/1,118-device/6-host plan, exact arena, direct role formats,
            and rejection of an all-Q4 artifact under the hybrid identity. Five focused CPU
            inventory/conversion tests, the registry qualifier, the binder qualifier, and a full
            `build-r9700` build with `-j4` pass. No full conversion or GPU execution was run, and
            the identity remains evaluation-only rather than the selected production recipe.

          - [x] Complete the no-write full-conversion readiness and deterministic validation
            boundary. `convert_fp8_hybrid --preflight-only` now requires one explicit `.ninfer`
            output, rejects an existing artifact or receipt, verifies the index contains exactly
            1,199 required BF16 tensors across the canonical 18 nonempty shard files, performs
            the complete tensor-metadata/config/resource/ranking/object-plan checks, derives the
            serialized artifact size, and requires that size plus 1 GiB of destination-filesystem
            headroom. It emits exact conversion and post-conversion validation argv arrays;
            ordinary conversion repeats the same destination/free-space gate before creating its
            writer. `--validate-only` independently reopens the completed artifact, compares its
            identity and all 1,124 descriptors to the preflight plan, hashes the artifact, and
            verifies the receipt's source, identity, recipe, selection, object-plan, byte-size,
            and digest bindings. The real CPU-only preflight passes against
            `/ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16` and the retained draft ranking:
            source-index SHA-256
            `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`,
            object-plan SHA-256
            `3cd973e5c8fc802d2e1042807fd3b021c89afeab7fe06ea62f507450222923bc`,
            exact projected artifact 21,553,545,216 bytes, required free space 22,627,287,040
            bytes, and 2,021,823,873,024 bytes free at the check. The intended output is
            `out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-eval.ninfer`; the preflight proved both
            it and its receipt absent afterward. The five focused CPU tests pass, including
            low-space, occupied-path, missing-shard, synthetic complete-artifact, and tampered
            receipt rejection. The subsequent real conversion and independent `--validate-only`
            pass materialized and reopened all 1,124 objects at exactly 21,553,545,216 bytes,
            matched the identity/recipe/selection/object-plan/source receipt bindings, and computed
            artifact SHA-256
            `1dfe9626fd6412592f87480a2f6934e8494a4b267693a25831dff490959542ce`.
            The first complete P2,048/C1/no-spec execution through that artifact also passes and
            measures `1,626.272735 tok/s` (`1.259509372 s`) under `auto`, `18.58%` faster than the
            rebuilt all-Q4 `1,371.444888 tok/s` baseline. This pre-shared-workspace result is
            retained at
            `profiles/bench/r9700-fp8-four-role-pre-shared-workspace-p2048-c1-20260904.json`,
            SHA-256 `6003f0ca2c66ab7ebda5958dd673b27a5d345702bca8e804e8d0f3d96d9fb907`;
            it proves full role routing but remains below the `2,000 tok/s` floor and precedes the
            capacity-preserving activation-workspace relocation.
            After that relocation, the matched rebuilt reports have identical 422,171,648-byte
            sequence and 608,387,072-byte workspace capacities. All-Q4 measures
            `1,369.269112 tok/s` (`1.495692741 s`) and the hybrid measures
            `1,636.830337 tok/s` (`1.251199252 s`), a `1.195407x` throughput improvement with no
            memory-capacity penalty. Their report SHA-256 values are
            `7dc01033692eb54f98582792f1b30422c3a91f2f62d61b312674aca9da926115` and
            `67497301b435910886898535e8a006078c6ee0d71cfb29b047853e64a70c8e8c`.
            The corrected source-codec 8K gate also passes against the deterministic BF16
            authority: mean NLL is `1.888154558`, delta `+0.022781640` versus the allowed
            `ln(1.05)=0.048790164`, with five new severe positions against budget 11. The source
            result and comparison SHA-256 values are
            `d8e3061c055c33064e315c7c5693b02c7c1ee3973182c71683ae1e066b6f8a89` and
            `67cd05967ba207faa85e558f198ed84c943e68a7831e9a8fd103f6fe99ed1773`.
            Its 379/4,095 argmax flips are diagnostic rather than a lossy-codec gate.
            Artifact-backed product PPL now passes at both staged lengths. At 8K the hybrid mean
            NLL is `1.889554244`, delta `+0.024181326`, with four new severe positions against
            budget 11; at 32K it is `1.756849826`, delta `+0.027958477`, with 25 new severe
            positions against budget 41. The campaign result SHA-256 values are
            `f30ef0d102ba5b54ed1c9c689641065839b0523c065692494ec4db1af63c2d20` and
            `97aacde821b0e570995b2fdc86cf9167763961ad92cfeb70d034a11fffdfa8f0`.
            Greedy flips remain diagnostic: 389/4,095 at 8K and 1,420/16,383 at 32K. Both
            create-only greedy receipts revalidate the live artifact/conversion receipt, retained
            BF16 campaign/repeat proof, scorer reports, and sidecar hashes; their SHA-256 values are
            `1dfe19d3359bf3f71be82d860278b8608e71d354298a68a6ebbae034910bfab8`
            and `fc9c5629c49c83b8435f24484505f2aba7567a21726f08a97ac252263ade2c22`.
            Thus source-codec and product NLL/new-severe gates pass and greedy evidence is complete;
            graph/eager and speculative parity plus final whole-performance/capacity admission are
            still pending. This does not select or promote the evaluation recipe.

          - [x] Bound the post-hybrid P2,048 bottleneck without starting another sweep. The
            auditable retained-input projection at
            `profiles/bench/r9700-post-hybrid-prefill-projection-20260904.json` (SHA-256
            `38663598f41a586a1322519f0fb7ed8661f6904047e321793242ccdefb00db0c`)
            records the `235.509372 ms` gap between the pre-relocation hybrid whole time and the
            `2,000 tok/s` floor. The 144 selected role calls project from `529.234288 ms` Q4 to
            `330.741408 ms` FP8 using their complete-Op medians, a `198.492880 ms` saving versus
            the `233.807851 ms` observed whole saving; the `35.314971 ms` favorable cross-run
            remainder is explicitly unassigned rather than attributed to an Op. The old dense
            trace projects the 176 remaining Q4 calls at `420.909711 ms`, while dense full-score
            attention is `197.841899 ms` over 16 semantic layer calls and GDN recurrence is
            `149.470363 ms` over 48. Neither latter family alone can close the gap; their combined
            upper bound is `347.312262 ms`, requiring a `67.81%` combined reduction. XAttention
            has zero calls in the production trace and no matched P2,048 timing, so longer-context
            evidence is not projected onto this cell. The next evidence is exactly one
            post-relocation selected-region dense trace, bucketed into 144 selected FP8 calls,
            176 remaining Q4 calls, 16 dense-attention layer calls, 48 recurrence calls, other
            kernel service, and kernel-inactive wall; it determines the next operator and prevents
            broad role/chunk/XAttention sweeps. The generator's two focused CPU tests pass.

          - [x] Prepare that single post-relocation selected-region trace without launching it.
            `profiles/rocprof/diagnostic-p2048-post-relocation-hybrid-selected-region-20260904/`
            contains a no-reuse plan and guarded command bound to the current shared-linear-
            workspace hybrid source report, executable, 21,553,545,216-byte artifact, corpus,
            rocprofv3 binary, and postprocessor by path, size, and SHA-256. The command fixes
            dense C1/P2,048/chunk-4,096/G16/spec-none, one warmup and one measured repetition,
            captures only the selected ROCTX region, records `auto` before/after, names one exact
            database rather than discovering it by glob, and refuses every occupied output path.
            Its validator conserves the lossless dispatch inventory and fails closed unless it
            sees 144 selected FP8 calls (quantize/matrix/poison), 176 remaining Q4 calls
            (quantize/matrix), 16 dense QK plus 16 dense PV layer dispatches, and 48 recurrence
            calls before emitting the six decision buckets. Plan SHA-256 is
            `1834ba5454f103c046eeb7cc6f8d0c4771b06b8158b1143560712d67ef183b17`;
            two synthetic bucketing tests, Python compilation, shell syntax, the live identity/
            freshness preflight, and duplicate-plan rejection pass. No profiler or GPU command
            was run.

          - [x] Run the post-integration host-only gate for the hybrid/shared-workspace slice.
            A full `cmake --build build-r9700 -j4` completed all 63 scheduled actions. The 21
            focused CTest cases covering artifact reading/materialization, prefill dispatch,
            FP8 activation and Program execution state, all target-binding fixtures including
            the FP8/Q4 hybrid, target binding, registry, and runtime planning pass. The 34 focused
            Python tests covering hybrid conversion/inventory, source-PPL diagnostics, whole-
            profile analysis/preparation/validation, post-hybrid trace bucketing, and the retained
            bottleneck projection pass under the documented isolated ROCm Python environment.
            Repository-wide `git diff --check` passes. No GPU-labelled CTest or GPU command ran;
            the initial system-Python collection failed only because that interpreter lacks Torch,
            and the complete rerun under the documented interpreter is the valid result.

          - [x] Promote the fixed K128 ordinary-GDN gated-RMSNorm eight-row CTA at only the exact
            flattened extents `48*T` for T=1024/2048/4096/8192. The immutable direct report
            `profiles/bench/r9700-gated-rmsnorm-k128-rows8-ab-20260904.json` (SHA-256
            `bdada371d626a6f4398ac350f5aaebdf3318d1743fb507e32f626b3c062e2e5c`) passed incumbent
            BF16-bit parity, the independent FP64 complete formula, poison rewrite, alias, and
            malformed-input gates and won all four extents; P2048 fell from `1.16636395` to
            `0.143720999 ms` (ratio `0.123221397`). LLVM reports 24 VGPR, occupancy 16, and zero
            LDS/private/scratch. The matched hybrid whole rerun improved from `1.251199252 s` /
            `1,636.830337 tok/s` to `1.202762421 s` / `1,702.749429 tok/s`; the post-promotion
            report SHA-256 is `d1f67fb21427c5378e87afec4f4a273a34425223dd4c941d58e01fa3bc45a5aa`.
            Production retains the general kernel for every other shape. Qualification-only API
            and one-shot executable surfaces are removed; the production ISA/resource checker
            remains. This operator admission neither selects the hybrid artifact nor closes its
            graph/eager, speculative, capacity, or terminal performance gates.

          - [x] Qualify and reject the dense full-score P2048/G16 16-wave PV head-partition
            candidate. The independent represented FP64 oracle, full incumbent-bit parity,
            poison/error semantics, and static checks passed at 70 VGPR, occupancy 16,
            9,208-byte LDS, 512 threads, and zero private/scratch storage. The immutable terminal
            report `profiles/bench/r9700-dense-full-score-pv-w16-head-partition-ab-20260904.json`
            has SHA-256 `38ebd0c64352ccb0b27c1b60f92fa537865a92827901879f89b2c189c212feee`.
            Candidate/incumbent medians were `6.84690714/7.28727818 ms` for PV (ratio
            `0.93956989`) and `10.1469564/10.8013163 ms` for the complete Op (ratio
            `0.939418495`), failing the predeclared `<=0.80` and `<=0.90` gates. No whole run was
            admitted. Qualification APIs, kernel, executable, Make targets, and candidate-only
            checker/tests are removed; the eight-wave PV kernel remains production. No adjacent
            head/query/page mapping sweep is authorized by this result.

          - [x] Qualify and reject the capacity-preserving FP32 hipBLASLt dense-PV replacement.
            The only legal query-major mapping used four zero-workspace strided-batched calls,
            each `batch_count=6` and `[M,N,K]=[256,2048,2048]`, with A `lda/stride=256/0`, B
            `ldb/stride=49152/2048`, and D `ldd/stride=6144/256`. Exact in-process solution 140189
            was GSU1/SK0 and passed the complete independent FP64 causal oracle, bitwise repeat,
            graph replay, invalid-frontier, fixed-domain, and no-clobber gates. The immutable report
            `profiles/bench/r9700-dense-fp32-gemm-pv-p2048-ab-20260904.json` has SHA-256
            `b231a114f4b50684f2e2fdc3eb8a8ce08a9702053dc32e5ff9346a80132a8538`.
            Candidate/incumbent medians were `4.524150848/5.141388893 ms`; only
            `9.875808716 ms` was saved over 16 matched calls, and the old-bucket projection was
            `80.216934204 ms`, failing the `>=30 ms` / `<=61.161073 ms` gate. The disconnected
            qualifier source, executable target, and generated binary are removed; production is
            unchanged. Do not revisit this layout or sweep adjacent hipBLASLt algorithms.

          - [x] Fuse the selected Text-MLP producer/consumer boundary only for row-scaled-E4M3
            gate/up plus Q4G64 down at T=2,048 in the 64 main Text layers. The fused Op consumes
            concatenated BF16 gate/up, computes FP32 SiLU-multiply, explicitly rounds its internal
            result to BF16, preserves the existing signed-A8G64 codec/status and M64N128 Q4 matrix,
            and removes one 142,606,336-byte BF16 write/read handoff and launch per layer. The
            immutable direct report
            `profiles/bench/r9700-fused-silu-a8q4-down-p2048-ab-20260904.json` (SHA-256
            `4d4143bc096d759c71c7d12cca2e3d68aa2a5c2bf0a4c74d2e972dfbc195f2b3`) passed full workspace and
            output bit parity, independent formula/codec, poison, tail/alias, resource, and both
            timing gates: stage `0.342199489/0.659238994 ms` (ratio `0.519082598`) and complete MLP
            `3.644394517/3.926753998 ms` (ratio `0.928093412`). The matched hybrid P2048/C1 whole
            report `profiles/bench/r9700-fused-silu-a8q4-down-production-p2048-c1-20260904.json`
            (SHA-256 `5daa98aafa34efcc5a55ee2eeca464ae94575c95ed60e6303cd9216ae253ab76`)
            improved the accepted K128 baseline from `1.202762421 s` / `1,702.749429 tok/s` to
            `1.178537057 s` / `1,737.747715 tok/s`, saving `24.225364 ms` with unchanged
            `608,387,072`-byte workspace and clearing the `>=10 ms` gate. Every other profile,
            width, and layer retains the ordinary boundary. Qualification aliases and the one-shot
            executable are removed; the production ISA/resource checker and immutable reports
            remain. Its current static report SHA-256 is
            `aea9061d6bb27fb9e0fec9508b242816cf5cf69118fe1f36b5616a3fcdbf5336`: 12 VGPR, zero
            LDS/private/scratch, exact load/store/native-exp inventory, and no intermediate BF16
            global traffic. This admission does not select the hybrid artifact or close the 2,000
            tok/s product floor.

          - [x] Run and bucket the single fresh post-MLP selected-region trace, then terminally
            reject its residual `other_base_text_kernel_service` bucket as a `>=20 ms` owner.
            The attribution-only `auto` capture measured `1,188.524422 ms` marker wall and
            `1,183.706883 ms` prefill, with `446.817900 ms` remaining Q4, `321.775102 ms`
            selected FP8, `148.139860 ms` dense attention, `145.579372 ms` GDN recurrence,
            `98.049622 ms` other kernel service, and `31.942668 ms` kernel-inactive wall. It
            requires exactly 64 fused SiLU-to-A8 preparations (`20.783684 ms`) and 112 ordinary
            A8 preparations (`8.941179 ms`), so the accepted fusion has removed the prior 64-call
            `24.328006 ms` standalone SiLU bucket rather than hiding it in `other`. Attribution
            SHA-256 is `fd563a4c3760f6f302fbdd7e0580356eceafeadb3a1d94ed326b1009b3a7bfe8`;
            database SHA-256 is `6ba449f2ef3efddb511a38d5ce7a30827bddd2b52e9d4671d9f5350fd5c08dd6`.

            The largest non-excluded coherent residual boundary is the paired 96-call BF16 GDN
            a/b projection plus 48 control calls at only `23.549814 ms`. A six-wave shared-input
            fusion retains all `11,796,480` BF16 WMMAs and all weight requests, removes only
            `37,748,736` BF16 handoff bytes, and reduces modeled activation-plus-weight requests
            from `12,079,595,520` to `7,046,430,720` bytes (`1.714286x`) while adding K-step CTA
            handoffs. Saving 20 ms would require at most `3.549814 ms`, a `6.634098x` speedup, so
            it has no credible physical bound. The immutable CPU/static audit is
            `profiles/bench/r9700-post-mlp-other-service-static-audit-20260904.json`, SHA-256
            `032e19afd7bb0b097bbf215bfb165243dd8c1bad3d932017eba6b37567f71a8f`.
            No residual-bucket qualifier or GPU follow-up is admitted.

          - [x] Qualify and reject the exact ordinary Text P2048/C1 GDN scale-sidecar admission. The
            provenance-complete v2 direct report passes at `2.252036095 ms` candidate versus
            `3.259594917 ms` incumbent (`108.0977325 ms` across 48 calls), with preparation
            `0.03379999846/0.07719899714 ms` and recurrence `2.164277077/2.893904924 ms`.
            It is `profiles/bench/r9700-gdn-scale-sidecar-p2048-ab-v2-20260904.json`, SHA-256
            `20288276c005bf469f99d6a7dccddc497c5d49c6a2301b65dac746f71fc38304`; the incomplete v1
            report is removed and not authoritative. The clean isolated whole run measured
            `1.153886920 s` / `1774.872034 tok/s`, a `24.650137 ms` saving from the accepted
            `1.178537057 s` baseline, but missed the fixed 30 ms gate by `5.349863 ms`; workspace
            remained exactly `608,387,072` bytes. Its immutable report is
            `profiles/bench/r9700-gdn-scale-sidecar-production-p2048-c1-v2-20260904.json`, SHA-256
            `110dbb4090dfa4a92ac993bd70ac4c21666b7a259abd17f63a4dbb913d9f6e95`. The candidate API,
            kernel, runtime/workspace integration, test, and qualifier are removed; production
            retains the general recurrence, and no adjacent recurrence variant is admitted.

          - [x] Audit and reject exact C1/P2048 two-stream projection overlap. GDN query-key and
            value-Z consume `29.039348/116.964267 ms` across 48 layers, making query-key's
            `29.039348 ms` the perfect-overlap ceiling and requiring `68.87%` of that ceiling to
            clear a 20 ms gate. The two attention FP8 branches add at most `16.1977975 ms` under
            perfectly balanced overlap. Today both formats bind one mutable activation region;
            although an aligned GDN partition fits the existing persistent capacity, the Q4 CTA
            consumes all 16 occupancy waves and both full-device branches contend for the same
            matrix/cache/memory service. With the trace already kernel-active for `97.30%` of
            Text-prefill wall, the required saving is not credible. Do not add streams/events or
            run a GPU overlap sweep. The terminal static audit is
            `profiles/bench/r9700-p2048-projection-stream-overlap-static-audit-20260904.json`.

          - [x] Bound and reject the exact eight-result zero-workspace hipBLASLt algorithm check
            for the dominant P2048 FP8 MLP gate/up shape `[34816,5120,2048]`. All eight returned
            fingerprints passed the independent represented FP64, malformed/alias/poison, and
            same-address eager/capture/replay gates. Rank 1
            `dde00100000000000000000000000000` was the best non-default result but its paired
            matrix median was `4.064851046 ms` versus rank 0's `4.061550617 ms` (ratio
            `1.000812603`); complete timing was `4.186669827/4.189990520 ms` (ratio
            `0.999207470`), projecting only `0.212524414 ms` saving over 64 calls. It failed the
            exact `3.808516172 ms`/`0.960590362` matrix and `4.031858 ms`/`0.962691984` complete
            gates; ranks 2 through 7 were slower. Production retains rank 0 fingerprint
            `e0e00100000000000000000000000000`, zero matmul workspace, unchanged graph addresses,
            and unchanged capacity. The terminal report
            `profiles/bench/r9700-fp8-gate-up-algorithm-p2048-ab-20260904.json` has SHA-256
            `5dd95d8f01d911e34e437175e028ed973ce025361ba61e9a269239f97c32de31`; the design report remains
            as its predeclared authority. The one-shot executable, CMake target, and harness are
            removed, and no whole run or adjacent algorithm/shape campaign is admitted.

          - [x] Reject converter-time row-interleaved or CTA-tiled FP8 gate/up storage as a
            P2048 fused gate/up-to-A8G64 route before implementation. The exact operation still
            requires `89,128,960` native FP8 WMMAs for `[2048,34816]x[34816,5120]`. The strongest
            pair-owned mapping, M128 by 64 logical outputs (128 stored gate/up rows) by K32,
            requires 4,352 1,024-thread CTAs, 160 synchronized K steps, 16,384 bytes of ping-pong
            LDS overlaid by the BF16 epilogue, and an estimated at least 81 VGPR from the emitted
            M128N256 resource result. More decisively, the measured rank-0 matrix plus accepted
            fused stage is `4.403750106 ms` per layer, so a 30 ms whole saving requires the new
            complete kernel at or below `3.935000106 ms`: faster than the tuned hipBLASLt matrix
            alone while also retaining both BF16 boundaries, SiLU, G64 reduction, and A8 coding.
            The removed 285,212,672-byte write/read traffic is worth only 28.825 ms over 64 layers
            at the retained 633.265 GB/s stream reference before overlap. Full-row permutation is
            byte-neutral but supplies no matrix-core advantage; true K/CTA tiling breaks the
            hipBLASLt decode and non-P2048 fallbacks, while duplicating all 64 objects would require
            11,417,419,776 bytes against only 819,200 bytes of C4 slack. Do not implement, time, or
            sweep adjacent storage/tile variants.

          - [x] Qualify and reject ordinary-GDN output gated-RMSNorm-to-A8G64 fusion at P2048.
            The 256-thread/eight-wave candidate passed exact incumbent parity, an independent
            represented-FP64 formula and codec, poison/alias checks, and static inspection at
            25 VGPR with zero LDS/private/scratch. Preparation improved from `0.239998996` to
            `0.196279004 ms`, but its 48-call aggregate was `9.421392202 ms` versus the
            `5.699527 ms` ceiling; complete output saved only `3.050880432 ms` versus the required
            `5 ms`. The immutable report
            `profiles/bench/r9700-gdn-output-gated-rmsnorm-a8-fusion-p2048-ab-20260904.json` has
            SHA-256 `8396c5e76b4166405acab70abce11f7f7137e264a1eb61eec3b25d0a098ba962`.
            Qualification code/tooling are removed and production retains separate K128
            gated-RMSNorm, A8G64 preparation, and Q4 output projection. Value-Z remains rejected
            because it lacks a unique producer boundary and cannot meet the same saving gate.

          - [x] Qualify and reject the fixed-P2048 ordinary-GDN projection/convolution
            direct-scatter cutover after the accepted MLP fusion. The immutable direct report
            `profiles/bench/r9700-gdn-prefill-projection-conv-direct-scatter-ab-20260904.json`
            (SHA-256 `fe852ad70f33f2e654cf31f2ef0da874c5355cdd491f9f97e027cb68142cf310`)
            passed at `0.288159013/0.559597015 ms` per layer (ratio `0.514940202`), deriving
            `13.0290241 ms` over all 48 calls. The isolated whole report
            `profiles/bench/r9700-gdn-prefill-projection-conv-direct-scatter-production-p2048-c1-20260904.json`
            (SHA-256 `8ceaa1d948e0a76d29765a4255dd806a2bc3ffeba4b5ec009fc39f8b893e7c2f`)
            improved the MLP-selected baseline from `1.178537057 s` / `1,737.747715 tok/s` to
            `1.168570885 s` / `1,752.573288 tok/s`, saving only `9.966172 ms` with
            `1.00852852x` throughput speedup. Those miss the fixed `>=12 ms` and `>=1.01x` gates;
            workspace remained `608,387,072` bytes. Production is restored to the ordinary
            projection copies, causal convolution, and three extracts. Candidate APIs, kernels,
            harness, Make/static/report-validator surfaces, and candidate-only tests are removed;
            the design and immutable direct/whole reports remain as the terminal record.

          Replace the qualification-only fixed-T owner with prepared fixed-N/K row-scaled
          `LinearExecution` objects for each selected role shape, whose hipBLASLt handles, layouts,
          algorithms, weight/row-scale pointers,
          activation image, token-scale plane, status word, and matmul workspace are owned by each
          Program and created before graph capture. Add a Variant execution-state hook to
          `ProgramImplCore`; thread it through `schedule::ExecutionCore` and `TextContext`; pass the
          Text layer index through `mlp_tail` to `Variant::post_mixer`; and select the prepared owner
          only when that layer's directly bound weight is E4M3. `LoadedModel` remains immutable
          weight storage. One Program-owned activation/matmul region may be shared by all
          serialized selected-role calls, but it must be excluded from `WorkspaceArena` reuse,
          included explicitly in the sequence memory plan/capacity curve, and keep stable addresses
          for every captured Device Graph.

          The fixed-shape qualification owner has now been replaced by the ops-owned
          `LinearExecution` vertical slice. It derives N/K and the K128 physical leading dimension
          from the directly bound row-scaled-E4M3 `Weight`, accepts caller-owned stable activation
          and hipBLASLt workspace regions, and creates idempotent per-T descriptor/algorithm
          profiles only through explicit `prepare(T)`. `run(T)` rejects unprepared widths and
          overlapping or invalid regions; the three gate-up, attention, and GDN qualifiers now use
          this common owner. A full `build-r9700` build (`-j4`), the CPU activation-workspace
          contract, and ten existing report/decision tests pass. The implementation source SHA-256
          values are `e0052e9b3980974e6f2f2865e7c7550e66e2dd23476b91e35c5bb46a88a0908d`
          and `93f6e593d8d6e35a523f23e67b1cb950f33168286e8052c542d250c9177a81da` for
          `linear_execution.h` and `.hip`. This closes the prepared Op-owner slice only; target
          inventory, Program ownership/threading, and artifact conversion are completed by the
          later bullets; production admission remains open.

          The evaluation-only four-role profile now has the corresponding Program execution
          boundary. `Variant::ExecutionState` owns all 144 layer/role-specific prepared owners,
          while one serialized Program-lifetime region supplies their shared activation image and
          zero-byte-selected matmul workspace. The region is now also the sole caller-owned
          activation image for all sequence Q4/W8 linears: at P=2,048 its controlling A8Q4
          K=17,408 requirement is `36,765,700` bytes (`36,765,952` after Program alignment), and
          the superseded equal-size per-phase WorkspaceArena tail is removed. Thus the FP8
          K=5,120 image (`10,494,208` aligned bytes) aliases only the later, mutually dead Q4
          activation image rather than a live tensor, and adds no affine peak bytes; non-hybrid
          integer profiles relocate their existing reserve without changing the selected matrix
          storage. Program constructs the owners before graph capture. The shared eager-width
          authority now enumerates ordinary C=1..4, MTP C*(K+1), DFlash C*verify-width, and the
          selected full prefill chunk with sorted deduplication and checked products; focused CPU
          contracts pin the current MTP set `{1,2,3,4,8,12,16,2048}` and DFlash set
          `{1,2,3,4,12,24,36,48,2048}`. An irregular final prefill tail is intentionally absent
          and may prepare lazily only after `hipStreamIsCapturing` proves capture is inactive;
          capture fails closed otherwise. The global Text layer
          index and execution state now reach attention query-key/gate-value, GDN query-key in all
          update/snapshot/record paths, and MLP gate-up; an E4M3 call without its exact retained
          owner fails closed. `ninfer_r9700_core`, the exact Qwen runtime, runtime-planner and
          target-GDN qualifier link checks build with `-j4`; the existing activation contract and
          new execution-state capacity/profile contract both pass as CPU-only tests. No GPU run or
          production-profile promotion is claimed.

          The retained post-relocation planner report
          `profiles/bench/r9700-fp8-hybrid-current-capacity-p8192-g16-20260904.json`
          (SHA-256 `122ec14fe78aba37a9cbb4f53189aee3e8f85da670137b35d6634e69edaccaac`)
          is historical invalid capacity evidence. It used the dense/spec-none materialization
          (`20,707,768,320` bytes) while claiming the MTP3 optimized feature set. Exact inventory
          accounting adds `582,700,032` bytes for that feature set, yielding
          `21,290,468,352` materialized bytes. Recomputing the same P=8,192/G16/MTP3 graph plan
          therefore leaves C1--C3 admissible at aggregate capacities
          `262,144/281,792/267,584` tokens, while C4 is startup-inadmissible by
          `245,140,480` bytes. The former C1--C4 slack values and the still older
          pre-relocation C4 claim are not current authority. The schema-v2 capacity producer now
          derives feature materialization from the exact inventory and binds the source schema,
          tool, target, weights identity, device, graph mode, and concurrency; a fresh physical
          selected-chunk capacity campaign remains required. The report validator fails closed against the
          live CPU planner executable and weight-report hashes; its two focused tests and
          `py_compile` pass without GPU allocation.

          - [x] Add the exact 8K source-weight-codec diagnostic boundary for the canonical hybrid.
            `tools/ppl/fp8_hybrid_source_diagnostic.py` consumes the one target-owned
            `fp8_hybrid_selection.inc` authority through `fp8_hybrid_decision.py`; it does not
            carry a second matrix-name list. It mechanically follows the existing source recipes,
            deriving 224 affected raw source tensors and 2,654,208 selected raw rows for the 144
            E4M3 logical Text matrices, including the head-interleaved Q projection and partial
            GDN QKV source. Another 178 scorer-consumed logical Text matrices use canonical
            Q4G64, while all 144 rank-two direct-BF16 matrices remain unchanged and the bound full
            inventory remains exactly 144 E4M3/295 Q4. Both lossy codecs decode to BF16 before
            the unchanged deterministic source formula. The paired
            comparator requires the retained 8K prefill/half BF16 source and corpus identity,
            verifies complete NLL/argmax sidecars, and applies the existing `ln(1.05)` mean-NLL
            and 0.25% new-severe-position bounds. Five focused CPU tests and `py_compile` pass.
            Its physical source-codec pass and the subsequent product results are recorded above;
            this isolated pass remains codec-only rather than product PPL by itself.

          - [x] Prepare the staged artifact-backed 8K then conditional 32K hybrid quality gate.
            `tools/ppl/run.py --require-fp8-hybrid` now consumes the target selection authority and
            fails closed unless the candidate has its exact hybrid identity plus the adjacent
            conversion receipt binding recipe, selection, format/byte counts, object plan,
            source index/ranking, artifact size, and artifact digest. The retained joint 8K/32K
            BF16 campaign importer now permits a requested length subset while retaining exact
            corpus/source/scorer/command/sidecar validation; the existing exact fresh-process BF16
            repeat comparison remains mandatory. The commands in `tools/ppl/README.md` run only
            G16 dense prefill/half with A8Q4, adaptive-A8 W8, and selected FP8-Q/K, apply the
            capacity-speed `ln(1.05)`/0.25%-new-severe gates, report diagnostic exact-token flips,
            and launch 32K only after the 8K campaign passes. The real hybrid receipt and retained
            BF16 subset revalidate CPU-only; all 44 focused campaign tests and `py_compile` pass.
            The subsequently executed physical 8K/32K product campaigns and receipts are recorded
            above; this bullet records their fail-closed preparation boundary rather than a second
            outstanding quality task.

          - [x] Add the post-PPL exact-token diagnostic validator. The create-only
            `validate_fp8_hybrid_greedy.py` requires a passing single-length dense G16 hybrid
            campaign, reopens the selected artifact and adjacent conversion receipt, revalidates
            the retained deterministic BF16 campaign plus its exact fresh-process repeat proof,
            rehashes both scorer binaries/reports/NLL/argmax sidecars, and recomputes the aligned
            I32 greedy comparison instead of trusting derived campaign fields. Greedy identity is
            explicitly diagnostic-only for this lossy recipe. The BF16 subset importer now treats
            path-valued historical scorer arguments by resolved identity while keeping flag order,
            inventory, scalar workload values, scorer/source/corpus identity, sidecars, and repeat
            proof exact. Three focused validator tests, all 44 campaign tests, and `py_compile`
            pass CPU-only. Both completed product campaigns revalidate through the tool: the 8K
            receipt covers 4,095 positions and 389 diagnostic flips, while the 32K receipt covers
            16,383 positions and 1,420 flips. Their receipt SHA-256 values are
            `1dfe19d3359bf3f71be82d860278b8608e71d354298a68a6ebbae034910bfab8`
            and `fc9c5629c49c83b8435f24484505f2aba7567a21726f08a97ac252263ade2c22`;
            no scorer or GPU work was launched by receipt validation.

          - [x] Localize the completed 8K source-codec and artifact-backed product results against
            the identical retained BF16 positions. The create-only CPU report
            `profiles/ppl/fp8-hybrid-execution-localization-8k-20260904.json` (SHA-256
            `cd8c2f429aae7efdd2e8569f682cedb32c626e870ce35656ac1a6c4cacbb299a`)
            revalidates selection/recipe/artifact receipt, source shards, corpus prefix, workload,
            BF16 repeat proof, scorer identities, and all aligned sidecars before comparison.
            Weight representation alone is `+0.022781640` mean NLL versus BF16; the complete
            product is `+0.024181326`, leaving a collective runtime increment of `+0.001399687`
            (paired SE `0.001335026`). Product versus source differs at 117/4,095 greedy positions:
            it repairs 50 source BF16 disagreements, introduces 60, changes seven already-wrong
            predictions, and preserves 322 shared disagreements. It repairs source-only new-severe
            position 3,689, introduces no product-only new-severe position, and retains the other
            four. The increment collectively includes activation quantization, product matrix
            arithmetic/reduction, cache codecs, and fusion, which final-token sidecars cannot
            separate. It is diagnostic and changes no gate. Three focused numerical tests and
            `py_compile` pass; no GPU or scorer ran.

          - [x] Extend the localization validator across the conditional 32K product workload.
            Token length is now explicit and restricted to 8,192 or 32,768; the latter requires
            exactly 16,383 aligned product/BF16 NLL and argmax positions from the retained BF16
            subset and repeats every artifact/receipt/source/corpus/scorer/sidecar check. Because
            the source-codec producer and retained source sidecars are currently 8K-only, the 32K
            report refuses cross-length attribution and records
            `source_localization_available=false`. The passing real result is retained at
            `profiles/ppl/fp8-hybrid-execution-localization-32k-20260904.json` (SHA-256
            `1e722acbd6c2a07223d2878b039b79ef74afdc838745173e630ca85ce5a8b69c`):
            product mean-NLL delta is `+0.027958477` versus BF16, with paired SE `0.002196653`,
            25 new severe positions under budget 41, 11 repaired severe positions, and 1,420
            diagnostic greedy flips among 16,383 positions. No threshold changed and no scorer or
            GPU ran.

          - [x] Prepare the bounded post-PPL selected-artifact execution validator. The planned
            execution campaign is subordinate to the explicit representation/context/performance-
            contract hold and is diagnostic, non-ranking, and not a prerequisite for base
            or DFlash selection. If retained as supported-behavior regression, its C1 decode extras cover
            MTP3 Device Graph versus eager, MTP3 versus ordinary, and MTP3 versus MTP4 with
            complete aligned NLL sidecars and exact argmax identity. The new CPU-only validator
            reopens the hybrid artifact and adjacent conversion receipt, rehashes both scorer executables,
            rejects an incomplete or nonexact execution matrix, and queries the compiled runtime
            authority for C=1..4 ordinary, MTP3, and MTP4 eager-prepared widths. At chunk 4,096 the
            required inventories are respectively `{1,2,3,4,4096}`,
            `{1,2,3,4,8,12,16,4096}`, and `{1,2,3,4,5,10,15,20,4096}`. Exact commands are in
            `tools/ppl/README.md`; focused Python tests and the host-only width query pass. No GPU
            execution or product evidence is claimed.

          - [x] Prepare the final hybrid whole-inference and native-capacity command matrices.
            `run_ninfer_bench_matrix.py --require-fp8-hybrid --prepare-only` now fails closed unless
            the real current artifact and adjacent conversion receipt validate, the selected dense
            G16 benchmark and shared-workspace runtime-planner executables exist and are hashed,
            chunk 4,096 is explicit, and concurrency is exactly C=1,2,3,4. The base capacity and
            whole matrices use the ordinary planner widths and spec-none route. The existing
            revised `pareto-whole` matrix supplies ranking spec-none ordinary rows; any separately
            retained MTP3 rows are diagnostic exact-token/state/graph regression only and cannot
            enter ranking.
            `pareto-capacity` supplies the base profile's product-capacity rows. Prepared manifests can
            resume only after revalidating the corrected ordinary-only command coverage, input
            identities, and `auto` power.
            Exact commands are in `tools/bench/README.md`. This closes tooling preparation only;
            no GPU benchmark or performance/capacity admission is claimed.

          The current T=2,048 owner is not sufficient to materialize even the admitted gate-up
          slice. Keeping Q4 for T<=128 by storing both planes is not capacity-admissible: gate-up
          alone would add its full `11,417,419,776` E4M3 bytes to the all-Q4 arena, exceeding even
          the most permissive retained cell by `1,953,581,567` bytes and the strict P=8,192/G16/C4
          cell by `5,034,699,263`. If all four candidate roles pass, dual planes add
          `13,600,161,792` bytes and miss the strict cell by `7,217,441,279`. Therefore the strongest
          fixed contract replaces each decision-selected role with E4M3 at every invocation; it
          does not retain a short-width Q4 plane or call it a fallback. Before conversion, prepare
          and qualify descriptors/algorithms for the selected prefill chunk and every realizable
          final tail plus ordinary C1..4 and all MTP/DFlash target-verification widths that traverse
          any decision-selected main Text projection. Heuristic search,
          descriptor mutation, allocation, and repacking must stay outside graph capture and timed
          execution. Each prepared width remains bound to its exact algorithm/workspace requirement;
          direct pointer, nonfinite-status, and decoded-FP64 Op checks are complete. The existing
          8K/32K PPL/severe-position gates have now executed on the single hybrid identity.
          Numerical quality and diagnostic greedy evidence now pass at 8K/32K, but graph/eager and
          speculative execution parity have not executed and remain the architectural coverage
          blocker. Whole-performance/capacity admission also remains pending, so neither the hybrid
          artifact nor its runtime profile is selected for production.
        - [x] Promote ping/pong after the matched parent-level whole-P2048 gate. The control report
          `profiles/bench/prefill-p2048-q4-pingpong-matched-control-20260904.json` (SHA-256
          `66a4c832b67156e713fbe5d3de57598086ec4a264acf513b212d149e70ec1254`) identifies
          `m64n128-production` and measured `1219.355187 tok/s` in `1.679577815 s`; the challenger
          `profiles/bench/prefill-p2048-q4-pingpong-whole-20260904.json` (SHA-256
          `a187d24e1ed78154f40a71ec1243f1d64ef46fbb25d3f0a2b2af3937667bb244`) measured
          `1348.188927 tok/s` in `1.519077416 s`. The `1.105657x` whole throughput improvement and
          `0.9044400` time ratio select ping/pong despite the preserved earlier `>=1.5x`
          operator-only rejection. Ping/pong is now the sole exact eight-shape by
          P=1,024/2,048/4,096/8,192 production route; the single-bank M64xN128 kernel is explicitly
          regression/tail-only. The temporary compile selector and alternate identity were removed.
          The later scalar-base promotion below preserves this topology while superseding that
          profile identity with `m64n128-pingpong-n16-k16-scalar-base-production`.
        - [x] Reject and remove the trace-selected 32-wave M64xN256 MLP-down challenger. Its static
          gate passed with eight IU4 WMMAs, 87 VGPR, 25,856-byte LDS, a 1,024-thread workgroup,
          and zero private/scratch storage, while preserving full bit parity, status poisoning,
          and the independent represented-FP64 oracle. Physical challenger/control ratios at
          T=1,024/2,048/4,096/8,192 were
          `1.0146974/1.0120804/1.0203344/1.0149652`; T2048 regressed from `3.379034` to
          `3.419854 ms`. The immutable terminal report is
          `profiles/bench/r9700-a8q4-m64n256-w32-mlp-down-ab-20260904.json`, SHA-256
          `a41e2a2ad65809ade16b629e698244834d2c2da065335f8fab62869813aaef5b`.
          Both rejected M64xN256 executable/tooling paths are removed, M64xN128 ping/pong remains
          production, and the bounded stop rule excludes an adjacent N256 or cache-hint sweep.
  - [ ] Held behind the immediate dense C1/P2048/spec-none `>=2,000 tok/s` floor and
        practical-ceiling gate above; the user-authorized smaller-gain exact-Q4 continuation does not
        authorize this downstream campaign. After those gates pass, obtain
        the matched rebuilt mixed-artifact trace and whole-selection evidence for the
        already-promoted target-specific A8W8G32 prefill CTA. The W8 implementation remains separate
        from A8Q4 and owns its distinct G32 signed-I8 payload, two native
        `v_wmma_i32_16x16x16_iu8` calls per group, FP32 scale accumulation, and one BF16 output
        round. Its selected 64-token x 64-row, 16-wave tile and exact production dispatch boundary
        are closed by the two completed subgates below: current emitted assembly passes at 50 VGPR,
        4,352-byte LDS, occupancy 16, 512 threads, and zero private/scratch storage. The earlier
        design-stage 32-token x 64-row alternative was not retained as a second production or
        qualification lane after the selected CTA won every exact Op cell by at least `14.8205x`;
        do not reopen that adjacent tile without a new whole-trace bottleneck. T<=128 and all
        unqualified tuples retain the incumbent one-wave route. The remaining gate is physical,
        not an implementation gap: after global chunk/static-profile selection, obtain an honest
        current-production mixed trace and matched whole result before the mixed recipe can win.
        That matched capacity/whole acquisition is owned by
        `profiles/bench/post-chunk-twelve-candidate-20260905`. Existing MTP execution remains covered
        as regression behavior; it adds no post-selection optimization gate.
        - [x] Close the physical linear-Op admission subgate. The schema-v3 report passed the exact
          4-shape x T=1,024/2,048/4,096/8,192 Cartesian matrix with zero maximum BF16 steps in every
          sampled row and the exhaustive tail case. The cooperative CTA won all 16 rows by
          `14.8205x` through `18.4695x`. The retained report is
          `profiles/bench/r9700-w8a8g32-prefill-cta-qualification.json` (SHA-256
          `1547f7d62fae4077f32f203b61831fc97b4f98bffcc930c6a29de7284f5c40c0`). This closes only
          the four exact mixed-Text tuple predicates, which are now the production linear-Op
          dispatch boundary. The report is immutable pre-promotion provenance; matched rebuilt
          mixed-model attribution and whole selection keep the parent item open.
        - [x] Promote the LDS-scoped CTA barrier after direct physical A/B. The immutable W8 report
          `profiles/bench/r9700-w8a8-prefill-cta-lds-scope-ab-20260904.json` (SHA-256
          `66cd50be917ee81b6f575fd7944b853bd4aa9a3ef3c2f3e0b0e8bdd378276784`) selects the
          production barrier at a weighted P2048 ratio of `0.9431145048`; every tuple is bit-exact
          and faster. Production uses the same qualified LDS-only ordering, and the global-invalidating
          form remains only as the direct regression control. Mixed real-model attribution remains
          open with the parent item.
        - [x] Retire the conditional mixed-artifact MTP-bulk optimization gate. The prepared
          `[5,120,10,240]` and `[1,024,5,120]` qualifier/trace package remains historical tooling,
          but it is not scheduled and is not required for recipe selection, finalization, or
          cutover. Existing MTP graph/eager execution, state, cache, row-view, acceptance, and
          output behavior remain regression requirements. DFlash/DFlash2 is the required and
          preferred speculative path for new support and performance work.
  - [ ] Qualify an arbitrary-tail extension of the promoted A8Q4/A8W8 prefill CTAs only after the
        global chunk and terminal static-profile selection and current selected-route profiling
        confirm that nonqualified final-chunk Linear fallbacks remain material. If they are not
        material, close this conditional subgate without an operator campaign. Until this subgate
        closes, preserve the exact production token predicate T in
        {1,024, 2,048, 4,096, 8,192}; do not infer admission from
        the launchers' mechanical support for arbitrary T>128, and keep T<=128 on the incumbent.
        The bounded operator matrix is paired CTA-versus-incumbent `auto` timing at
        T={129,192,193,256,257,512,513,1,023,2,047,4,095,8,191} over only the selected profile's
        reachable CTA tuples and formats; the eight Q4 plus four W8 tuples (132 timing cells) are a
        ceiling used only if the selected mixed profile requires both. Apply the
        independent represented-code FP64 oracle with the retained <=2-BF16-step criterion to all
        selected real tuples at T={129,257,1,023,8,191} (at most 48 numerical cells), reusing the
        existing exhaustive synthetic N65/K193/T129 padding/status/alignment evidence. Admit only
        measured per-format/per-shape minimum-T boundaries whose tested widths at and above the
        boundary win; do not replace them with a blanket T>128 rule. After any operator promotion, check
        selected-artifact whole-prefill timing at prompt lengths `selected_chunk+129`,
        `selected_chunk+257`, and `selected_chunk+1,023` before closing the gap. This token-tail
        matrix does not qualify the mixed MTP-only W8 [5,120,10,240] or [1,024,5,120] shapes.
  - [ ] Optimize ordinary non-speculative decode immediately from the dense all-Q4 G16 exact C1
        P8192+G256 spec-none Device Graph route. Its pre-dot8 one-repetition diagnostic was
        `15.13345066` output tok/s over `16.91616841 s` of decode in
        `profiles/bench/ordinary-none-dense-all-q4-g16-c1-8k-current-20260905.json` (SHA-256
        `829d19d4eff2364e0d782f0da3cfe7dfa7c3bee313614799f5aa53a333e17f7f`). It supersedes the
        older `8.354688852` tok/s result. This work is explicitly exempt from the still-open
        dense-prefill floor and from the held chunk/capacity campaign:
        identify the dominant ordinary-only kernels and host/launch gaps, qualify bounded direct
        changes against the fixed cache and exact output semantics, and confirm wins on ordinary
        whole decode. The one-round trace report is
        `profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/benchmark-report.json`
        (SHA-256 `7d9da725902c04e208525ae2d02dd33fc696ab452ad6673579a730f6e8f7eee6`), its ROCPD database is
        `profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/raw/ordinary-decode-c1-one-round_results.db`
        (SHA-256 `2cae81419c0cf360eb032537ca95a4030a1c80df1e9221dc5d6dfcd06f3aa0ff`), and the corrected
        analysis is `profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/analysis-v2.json`
        (SHA-256 `ca8937876bc16259aeaf793a1c6070c5b99dac74d71f2eca4b348ff0b2f7a9d1`). The ordinary host
        marker spans `78.193701 ms` and has 1,806 exact-associated dispatches totaling
        `64.388840 ms` of independent dispatch duration. The leading contributors are Q4 WMMA
        (`42.301380 ms`, 321 calls), RMSNorm (`13.642834 ms`, 161 calls), split512 PV
        (`4.276113 ms`, 16 calls), and QK (`1.030131 ms`, 16 calls). Dispatch-interval union is
        only a union of rocprofiler start/end intervals: it is not GPU-active time, GPU wall time,
        utilization, or CU occupancy, so optimization follows source-matched whole A/B rather than
        converting that union into a physical claim. The native-dot8 T=1 operator gate passed with
        a `14.1025415618 ms` robust exact-call-weighted saving per token in
        `profiles/bench/r9700-a8q4-t1-native-dot8-all-text-20260905.json` (SHA-256
        `300626f0d45b5b9bb8f6b420f44b7e5652e3a848738c636959f02f3448d2b5b0`). Its seven exact
        full-K tuples are selected by the Linear shape contract; `[5120,17408]` and
        `[34816,5120]` also occur in DFlash2. The four-pair source-matched whole gate promoted the
        route: median decode fell from `16.9262812925 s` (`15.1244089340 tok/s`) to
        `12.5301623450 s` (`20.4307009719 tok/s`), robust ratio upper was `0.7427031671`, robust
        saving lower was `17.0000856399 ms/token`, prefill ratio upper was `1.0016298424`, and all
        generated tokens and matched identities were exact. The immutable report is
        `profiles/bench/r9700-a8q4-t1-dot8-whole-p8192-g256-full-20260905.json` (SHA-256
        `19278ca8c8df5d417bbd5d36ce1689760e3cb6dbe606c3a597a6b2f0b847fee1`). Native dot8 is now
        canonical for those tuples with no selector; off-inventory T=1 and logical-K tails retain
        WMMA. The strengthened production regression covers varied N16/K16 weights/scales, dense
        activations, complete output parity, FP64 checks, poisoning, rewrites, and canaries.
        A fresh selector-free production build then measured `20.45440879 tok/s` with exact 257
        generated IDs in `profiles/bench/r9700-dot8-production-final-p8192-g256-c1-20260906.json`
        (SHA-256 `c341f1eeb2f5d5597272dbecc532982b008701c7ff6a22d946541f98c9c94b2c`); its extracted loaded
        gfx1201 object (SHA-256 `6a4e9eed7804da2321e112a3c520f65eb3d6a1b88e8352c39f9b364a30033312`)
        has exactly 16 native dot8 instructions, no WMMA, 18 VGPR, and zero LDS/private/spills.
        The parallel K5120 RMSNorm CTA subsequently passed every rows1..4 operator cell. Its
        immutable report is
        `profiles/bench/r9700-rmsnorm-k5120-rows4-qualification-8192i-20260906.json` (SHA-256
        `e58b2e56980fb083548f1c357c26fc98a8a5bae27b1723ccf712451eaf4b8103`), with a
        `11.7782198046 ms` minimum robust ordinary-round saving lower bound. The exact K5120
        rows1..4 domain is now canonical; K5120 rows5..127 retain generic RMSNorm, K5120 rows>=128
        retain token8, and other widths retain their existing routes. Its four-pair source-matched
        whole gate reduced median decode from `12.521242570 s` to `9.467485694 s`, robust ratio
        upper `0.7598847464`, robust saving lower `11.72510638 ms/token`, and prefill ratio upper
        `1.0029548902`, with exact generated tokens. The immutable whole report is
        `profiles/bench/r9700-rmsnorm-rows4-whole-p8192-g256-full-20260906.json` (SHA-256
        `3f5c7a678f29b09537b46ebf7692e6f8d9b422e08f9ffe656367815932b2d0f6`). The selector-free final
        smoke measured `9.461402437 s` (`27.05729956 tok/s`) with all 257 IDs retained; its report
        SHA-256 is `b05db0068a4f1c73ce9c2092443b42f9f48b0fdb80ff8b3335db60cd5bdca74b` and executable SHA-256 is
        `a7c9303bd213fa3dbdb29ca0cee73addef1de8ab6a6e6b231509e25239776425`. The selected-region trace
        records exactly 129 CTA dispatches plus 32 legitimate generic RMSNorm dispatches; its
        database SHA-256 is `8fe71be97e77c2651cb0c75fe203cedb13f200f8ac76082e4310bbfb855e5370`.
        The extracted loaded gfx1201 object SHA-256 is
        `c3dcad45559a112f42f07b1d7e87fbd1d494d1d024083efc0498500ee678cd72`; the selected kernel uses
        17 VGPR, 32 bytes LDS, wave32, occupancy 16, and zero scratch/spills.
        The following grouped-PV mechanism report is screen-only:
        `profiles/bench/r9700-split512-grouped-pv-qualification-v2-20260906.json` (SHA-256
        `f690accdc47bd85096aa412aa415e3f55446e8d9b737b9928cef671b1054ea6d`). The integrated product
        route passed exact numerical, rejection, and Device Graph qualification in
        `profiles/bench/r9700-split512-grouped-t1-direct-20260906.json` (SHA-256
        `958b38efaced4b671222c32de93ce72264df6a58aa6fbb5797f0738e965ba65b`). Its C1/P8192+G32 whole
        screen nevertheless rejected it at decode ratio `1.0038023342`, saving
        `-0.141697625 ms/token`, and prefill ratio `1.0033795393`, with exact generated tokens. The
        report is `profiles/bench/r9700-split512-grouped-t1-whole-p8192-g32-screen-v2-20260906.json`
        (SHA-256 `bcd24621462dbb404ecc77b79a6b324f0c4d4b67a7e0afdcbacc69281cd1d9f2`). No full gate was run;
        the selector, challenger, and temporary tools were removed. Continue ordinary decode with
        bounded bandwidth/stall-proxy profiling and its roof decision, not another grouped-PV
        variant. This work does not open the held prefill campaign or discharge the selected-recipe
        DFlash2 gates. Do not optimize or rank MTP; its `19.24458338` tok/s MTP3 row remains diagnostic.
        After ordinary decode, the next implementation priority is DFlash2.
        Final selected-route profiling and roof accounting remain in the dependent item below
        because they require the eventual selected chunk/profile authority.
  - [ ] After chunk and terminal static-profile selection, profile only the ultimately selected 8K
        and 32K prefill and decode routes, including the admitted speculative mode; candidate
        screens used to make those selections are not a second terminal profiling campaign. First
        use selected-region tracing to separate base Text prefill, speculative/MTP-prefill work,
        decode, host gaps, copies, and dominant kernels; then use a focused counter pass on only the
        identified kernels to measure the usable relative GL2C/TCP hit rates and SQ activity. For
        every executed selected-route Op family, retain a dispatch-to-specialization inventory plus
        the disassembly and resource gate proving that its gfx1201 kernel uses the intended hardware path
        (IU4/INT4, INT8, or FP8/BF16 WMMA as applicable), rather than inferring instructions from
        the semantic dtype. The current Q4/W8, XAttention, and GDN grep/awk reports are observational,
        not sufficient final gates. Interpret the required path per traced specialization: promoted
        Q4 prefill CTAs require native `v_wmma_i32_16x16x32_iu4` low/high pairs, I32 recombination,
        and FP32 scaling; promoted W8 CTAs require native `v_wmma_i32_16x16x16_iu8` and FP32
        scaling. Any selected-route W8 tuple outside those CTA predicates is interpreted only as
        observed route inventory; MTP-only tuples do not create a new optimization or promotion
        gate. Sparse XAttention rank and B16 QK consumption require native
        `v_wmma_f32_16x16x16_bf16` with FP32 online softmax/direct INT4-V,
        while its FP8-to-BF16 pack is a conversion path rather than a matrix path. The incumbent
        dense `T>2` attention is scalar/VALU FP32 Q-by-decoded-FP8-K reduction, online softmax, and
        signed-INT4/FP16-scale PV; GDN recurrence/control uses scalar FP32 FMA/transcendentals and
        subgroup reductions with BF16 I/O and FP32 state. Absence of WMMA is not a failure for
        either incumbent; a promoted tiled dense-attention challenger must
        instead prove its selected BF16-WMMA specialization. Elementwise RMSNorm, SiLU, and residual
        kernels likewise remain VALU unless separately replaced. Keep every executed family's path
        check fail closed; after selected-region tracing names the dominant specialization, upgrade
        that route's check to its exact specialization inventory,
        required opcode and count, VGPR/LDS/occupancy ceilings, and zero private scratch; bind the
        checked assembly and source identity to the retained per-Op claim. For every executed family,
        record weight/data residency and access plus whether next-operation movement is dependency-safe.
        For every material path, retain stream/launch evidence for any pipelining or overlap claim,
        plus a controlled `auto` A/B; otherwise record overlap as inapplicable, including serial GDN
        token-state recurrence and gate-up-to-SiLU-to-down dependencies. A single-buffer or serialized
        pack/rank/consumer implementation is not overlap evidence. Combine matched `auto` timing
        with represented
        logical bytes and FLOPs to quantify compute and logical-memory roof proximity. Use the
        `profile_standard` counter
        pass only to establish relative GL2C/TCP hit behavior and productive SQ activity. Absolute
        HBM traffic and VALU/LDS/scheduler-stall counters remain unavailable on this stack, so make
        no absolute-bandwidth or stall attribution claim from the known-zero request-size buckets.

        The post-selection ordinary-decode proxy capture is prepared at
        `profiles/rocprof/selected-ordinary-decode-memory-prepare-20260905`. The immutable terminal
        executable remains the timing and selection authority; because it predates the explicit
        `DecodeOrdinaryRound` marker, it is not the profiler executable. After the live finalist
        campaign exits, `profiles/rocprof/selected-ordinary-decode-profile-build-20260905` must
        produce a distinct marker-bearing build and derivation receipt that reopens the schema-v7
        winner and binds its whole-matrix reports, artifact, cache group, attention profile,
        selected chunk, compile definitions, marker source, and hybrid receipt/planner when
        applicable. The capture preparer requires that receipt and independently revalidates the
        complete selected runtime route. The bounded future workload is non-speculative 8K+256
        decode at C1..4, one measured repetition after one warmup. All timing from the
        instrumentation build is inadmissible. A single gfx1201-valid PMC pass reports GL2/L0 hit
        ratios, relative GL2 reads,
        occupancy, broad overlapping SQ waits, and instance-summed GL2 write-stall activity, with
        positive-activity and exact-inventory gates. It deliberately leaves physical GDDR6 bytes/s,
        peak-bandwidth fraction, and stall freedom null: this installed profiler's request-size
        buckets are known-zero, and the available wait counters cannot prove causal stall absence.
        Therefore the user's near-bandwidth-roof request can be answered only with these measured
        proxies plus logical roof analysis unless a future tool exposes valid physical-byte counters.

        The concise low-precision path/streaming coverage snapshot below is the selection ledger.
        Here, "executed ISA" means a dispatch joined through the loaded code object to its bounded
        symbol, while "static ISA" means a source-matched emitted-symbol/resource check. Do not
        reopen a closed opcode question merely because terminal whole-route provenance is pending.

        | Selected or selectable production Op family | Stored/artifact dtype | Activation / public-input dtype | Expected gfx1201 instruction path | Executed/static proof | Cache, prefetch, and following-work evidence |
        |---|---|---|---|---|---|
        | Text/MTP/DFlash Q4 Linear, including the existing draft head | signed `Q4G64_F16S` codes/scales | represented BF16 -> dynamic two-plane A8G64 | native low/high `v_wmma_i32_16x16x32_iu4`, I32 recombination, FP32 G64 scaling | Exact CPU-only extraction from all four current candidate executables reconfirms identical selected symbol bytes: the M64xN128 CTA has eight IU4 sites, 88 VGPR, 17,152-byte LDS, and zero private/scratch; the wave32 route has four IU4 sites, 64 VGPR, zero LDS/private/scratch. Executable/code-object/symbol hashes are bound by `profiles/bench/r9700-twelve-candidate-hardware-path-static-audit-20260905.json`. This closes current-build presence/resources; MTP uses it as retained regression behavior rather than a new optimization target | The converter fixes `text/draft_head` to `Q4G64_F16S[131072,5120]` in `row-split-k128-v1` for every registered recipe. The binder materializes those code/FP16-scale planes directly, and the shared Linear boundary performs BF16 -> two-plane A8G64 once per call before the wave32 kernel; there is no runtime weight repack. The CTA stages each packed group once, reuses it across four waves, and overlaps next-group code loads with current WMMAs under default caching. Next-scale prefetch and non-temporal weight loads were physically rejected. The selected-Q4 PMC capture remains the non-causal activity authority; selected-dispatch ATT is terminally unavailable after bounded buffer overflows, and absolute HBM bytes/stall freedom remain unavailable. |
        | Mixed-profile W8 Linear | signed `W8G32_F16S` codes/scales | adaptive represented BF16 -> A8G32 above the qualified crossover; represented-BF16 fallback below it | two signed `v_wmma_i32_16x16x16_iu8` per G32 group for A8; scalar FP32 FMA for the exact fallback | Static and direct operator execution/resource gates pass for the four promoted Text CTA tuples. The current-four-binary extraction proves identical CTA bytes with two IU8 sites, 50 VGPR, 4,352-byte LDS, and zero private/scratch, plus an identical two-IU8-site A8 wave32 route. A selected mixed-model trace remains useful route evidence; MTP-only tuples are regression coverage, not an optimization gate | The CTA stages each code/scale fragment once in LDS and reuses it across four waves. Its source and emitted ISA do not overlap a successor G32 group or following Op; no such claim is made. All selected loads use default-temporal policy. Physical cache attribution is required only if the mixed winner's selected trace makes this path material. |
        | Four-role hybrid Text projections: MLP gate/up, attention query/key and gate/value, GDN query/key | row-scaled `F8E4M3_ROW_F32S` | represented BF16 -> row-scaled E4M3 plus FP32 token scale | selected hipBLASLt loaded ELF contains native `v_wmma_f32_16x16x16_fp8_fp8` with FP32 HPA | Fresh executed-ELF proofs pass for the gate/up and distinct attention algorithms. The exact loaded-ELF joins retain gate/up (and GDN-shared) at 128 SGPR, 192 architectural/zero accumulator VGPR, 25,088-byte LDS and zero private/scratch, while attention is 128/192/0, 12,544-byte LDS and zero private/scratch; the installed rocprof schema exposes no spill counts. The terminal hybrid verifier requires its actual selected dispatches to match the same ELF SHA and available resource fields. GDN shares the proved gate/up algorithm identity and passes direct execution/numerics, but its exact production dispatch still belongs in the terminal inventory if the hybrid wins | Weights bind directly and remain resident; there is no runtime repack. hipBLASLt's internal fetch schedule is opaque, no next-Op prefetch is claimed, and two-stream projection overlap was statically rejected. Profile memory only if the selected trace makes an FP8 projection material. |
        | Dense initial-prefix full attention | FP8 E4M3FN K, signed INT4 V, FP16 V scales | BF16 Q; FP32 scores/probabilities | Bq16/Bk16-or-Bk32 `v_wmma_f32_16x16x16_bf16` QK; integer nibble decode plus FP32 PV | Production oracle/static gates pass with exact opcode counts/resources/zero scratch, and the P2048 trace proves the selected dispatches | QK stages decoded K in LDS and reuses it across six GQA heads; maximum and direct INT4-V PV are dependency-ordered stages. No inter-stage overlap claim or physical-byte claim is required for the already selected algorithm; profile it only if terminal attribution again makes it material. |
        | Sparse B128/S16/tau900 XAttention | the same fixed three-plane cache | represented BF16 Q; rank/consumer use decoded BF16 K panels and FP32 probabilities | native BF16 WMMA for rank and B16 QK consumption; integer nibble decode plus FP32 direct PV; FP8-to-BF16 pack is a conversion path | Source-matched static/operator gates pass; current real-model trace/counter evidence is diagnostic, so an XAttention winner still needs exact terminal dispatch-to-ELF/resource binding | Pack/rank/consumer are serialized, not overlap evidence. A real consumer pass measured 98.371% GL2 and 77.692% TCP hit ratios under attribution-only `profile_standard`; repeat on the selected profile only if it wins and remains material. |
        | Ordinary/split-512 decode QK plus cache append | FP8 E4M3FN K, signed INT4 V, FP16 V scales | BF16 Q/source; fixed T1 or T4 decode | native FP8 WMMA for admitted T1 QK, BF16 WMMA for fixed T4 QK, and `v_cvt_pk_fp8_f32` for cache encode; scalar FP32 direct INT4-V PV | Static emitted ISA and physical operator/state admission pass; terminal route inventory must bind the selected metadata-aware specialization | Append, publication, and cache consumption remain on the ordered execution stream. Publication/state dependencies make next-matrix overlap inapplicable; split-512 scratch aliases only across sequential consumers. |
        | Vision attention | direct BF16 Q/K/V activation tiles; its surrounding projection weights use their bound recipe | BF16 | native `v_wmma_f32_16x16x16_bf16` for both QK and PV with FP32 accumulation | Current source-matched device-only gfx1201 emission contains 15 BF16 WMMA sites at 256 VGPR, 4,096-byte LDS, and 336-byte private/scratch storage; this is an accurate static path proof, not a zero-scratch promotion claim. Exact terminal Text-profile selection does not substitute for a multimodal executed-dispatch proof | Q/K/V are caller-owned resident tensors and the kernel reuses wave fragments locally. Segment/tile preparation precedes attention on the same stream; no cross-Op prefetch or overlap is claimed. |

        The post-selection closure is prepared at
        `profiles/bench/post-terminal-selected-hardware-use-20260905`. It revalidates the exact
        schema-v7 artifact/executable/cache/attention/chunk tuple, requires the reconciliation's
        modeled inventory to equal a hashed selected-route trace, joins the selected Q4 CTA and
        dense or XAttention symbols to the current four-binary audit, requires W8/IU8 only for the
        mixed winner. Existing MTP dispatch checks are regression evidence only; the separate
        optimized-MTP-head proof is no longer a finalization gate. A
        four-role winner additionally requires both loaded-ELF FP8 proofs and rejects them unless
        their exact dispatched hipBLASLt kernel names occur in the selected inventory for MLP,
        GDN, attention-query/key, and attention-gate/value roles. The verifier now recomputes the
        complete reconciliation from its snapshotted trace and static schedule instead of trusting
        a caller-authored modeled-symbol list. Retained MTP dispatch checks may still be reopened as
        regression evidence, but neither their performance nor a separate selected-chunk MTP-bulk
        W8 proof is required for finalization. The reconciliation
        producer now accepts dense or fixed-tau XAttention C1 P2048 schedules and classifies exact
        Q4/W8 pairs or the four-role FP8 quantize/matmul/poison sequence from artifact format and
        source order. The post-selection trace preparer/validator binds the actual schema-v7 route;
        neither sparse nor hybrid may reuse the dense counterpart. The output
        explicitly leaves physical bandwidth and stall freedom unset. It is blocked on schema-v7
        selection and those future selected-dispatch inputs; no GPU work was run to prepare it.
        | Embedding gathers and DFlash selector codebooks | recipe-bound Q4/W8 or direct BF16 embedding; model-specified direct BF16 selector tables | I32 token indices / selector indices | scalar/vector packed-code decode plus FP32 scale for quantized gathers, or direct BF16 loads; no matrix opcode is expected | Source/static eager and DFlash gates establish the codec/load paths; terminal tracing is required only when these become material in the selected workload | All tables are directly materialized and resident with no runtime repack. Index dependencies make speculative next-table movement unsafe; cache behavior is left to normal temporal access with no overlap claim. |
        | Ordinary GDN recurrence/control and elementwise normalization/SiLU/residual | Q4/W8/FP8 applies only to surrounding projections; persistent state is FP32 | BF16 I/O with FP32 recurrence state | scalar FP32 FMA/transcendentals and subgroup/VALU reductions; no matrix opcode is expected | Production specializations have direct numerical/resource/timing evidence; terminal tracing must identify them but absence of WMMA is correct | The token-state recurrence and gate-up -> SiLU -> down boundaries are true dependencies. Record overlap as inapplicable unless a separately qualified fused boundary removes a material handoff. |

        The current-four-binary audit above found no mismatched matrix instruction family or
        runtime weight repack across the twelve selectable candidates. It also records that none
        of the selected embedded matrix symbols carries an explicit cache modifier: those loads
        use default-temporal policy, while only the Q4 CTA has proved successor-group overlap.
        Presence in an executable remains distinct from executed-dispatch evidence.

        A single dispatch-scoped FP8 gate/up counter package is prepared, but not physically
        executed, at `profiles/rocprof/r9700-selected-fp8-gate-up-p2048-pmc-20260905`.
        It binds the current four-role G16 artifact/executable, the selected hipBLASLt solution
        123104 and loaded-ELF 112-site native-FP8 proof, and only the exact
        `[T=2048,N=34816,K=5120]` 64-call C1/chunk4096/non-speculative workload. Two legal
        `profile_standard` passes measure GL2/L0 hit ratios, official occupancy, broad SQ wait,
        LDS-conflict, and normalized issue activity, restore `auto` after each pass or failure,
        and reject foreign symbols/resources/regions or stale input bytes. The exact same loaded
        symbol also serves 48 GDN projections at a distinct grid/region, so the command binds the
        64 one-based occurrences after the exact selected-symbol filter, derived from the
        same-command retained trace; the
        analyzer rejects any extra captured row rather than silently reducing a mixed inventory.
        Its roofline records
        4.061550617 ms / 179.7699 useful TFLOP/s against the 400.835-TFLOP/s architectural
        ceiling; even that ceiling gives about 1.822 ms, above the 1.646909-ms time inferred to
        close the whole gap alone. Profiler durations remain attribution-only and neither
        physical bandwidth nor WMMA utilization is claimed. Broad overlapping wait and cache
        counters are decision-ready diagnostics, not causal proof or an automatic rewrite gate;
        this PMC evidence alone cannot warrant a fundamentally different architecture. This
        preparation is not schema-v7
        selection, whole admission, or physical profiling closure.

        The remaining instruction-path finalization gate is the dispatch-to-ELF inventory for the single terminal
        recipe/cache/attention profile, including GDN FP8 or XAttention only when selected. The
        selected-Q4 GL2C/TCP/SQ pass is additionally required before declaring the low-context
        memory path near its roof or free of an actionable cache limit, but it does not precede the
        recipe/cache ranking. No unavailable absolute-HBM or detailed stall counter is a gate.

        Open XAttention memory-path work only when the selected-route trace makes the affected
        stage material. In order: move the ranker's CTA-private mass/plane-mass/rank/selection
        scratch into LDS for the selected 8K/32K extents; if rank time remains material, eliminate
        its duplicate full Q/K estimator traversal; stage the flash consumer's invariant B16xD256
        query tile once per CTA; and try an LDS transpose of token-fastest FP8 keys in the packer
        only if the selected trace overturns its currently negligible bound. Keep these as separate
        challengers rather than one compound change. Every challenger must pass the independent
        numerical oracle, exact-specialization ISA plus VGPR/LDS/occupancy and zero-private-scratch
        checks, matched selected-route `auto` operator and whole-prefill timing, and a dispatch-scoped
        `profile_standard` GL2C/TCP A/B with positive activity controls. Do not infer benefit from
        hit-rate changes alone or apply cache-bypass hints to packed K/query/value data that is
        intentionally reused.
        - [ ] If terminal selection retains G32 XAttention as a selected or Pareto route, A/B a
              subgroup-broadcast FP16 value-scale load in its flash consumer instead of one
              duplicate scale load per feature. The selected feature-fastest specialization is
              currently 120 VGPR versus G16's 68 despite G32 storing half as many scales; require
              the independent oracle, materially lower VGPR with legal LDS/occupancy and zero
              scratch, alternating production-scale timing, and matched selected-route whole
              timing. G32 trades about 2.5% lower current 8K sparse prefill speed for about 4%
              greater memory-limited C3/C4 context capacity, so this gate does not preselect either
              group. If terminal selection does not retain G32 XAttention, close this subgate N/A.
        Only if the selected-region trace leaves XAttention consumer scaling materially unresolved
        because the real keep distribution is unknown, add a bounded, opt-in selected-route
        diagnostic that snapshots the existing device `keep_count` vector for each full-attention
        layer/chunk. Enqueue copies or device aggregation on the execution stream while the caller
        workspace is live, with no per-layer host read or synchronization, and perform one
        post-timing flush. Retain exact layer/chunk geometry, raw counts and recomputable keep
        fractions, and distinguish the dense T=1 route, forced full retention from an unaligned
        B128 origin, naturally selected all-page sets, and invalid input rather than conflating them
        as one fallback. Mark the sidecar timing-ineligible and bind it to the exact benchmark
        report, artifact, executable, selected group/profile/chunk, `auto` pre/post evidence, and
        profiler database hashes. Do not implement or collect this telemetry when the selected
        trace already identifies the bottleneck without keep-distribution attribution.
        Where the selected-route trace shows Q4/W8 prefill CTAs are material, test their current
        memory path as separate challengers. The present source and rebuilt gfx1201 objects use
        default-policy aligned/coalesced dword code loads plus halfword scale loads with no explicit
        cache hint; each fragment is staged once per CTA and reused by four waves. The host traffic
        model's requested/unique ratios are `37.35x`--`55.00x` for representative Q4 shapes and
        `42.09x`--`57.27x` for representative W8 shapes. Those are source-request hypotheses about
        cross-CTA reuse, not physical HBM measurements. First A/B an LDS-scope-only correctly
        ordered barrier that removes the current per-barrier `global_inv`: use workgroup
        local-address-space release/barrier/acquire fences, never a bare `s_barrier` or a global
        barrier redefinition, and reject unless each exact CTA retains two barrier pairs, zero
        `global_inv`, the required WMMA, zero scratch, and its resource ceilings. Only if that
        conditional A/B wins, then test ping-pong LDS with next-group global-load/WMMA overlap.
        That ping/pong challenger was exact and faster but failed its initial predeclared `>=1.5x`
        operator gate (`1.23224x`); the later matched parent-level whole-P2048 gate at lines 828--839
        superseded that disposition and promoted it to production. Test block traversal or
        cache-policy changes only if the matched counter pass shows remaining opportunity, and do
        not infer benefit from a changed hit rate without an `auto` timing win.
        - [x] Qualify the bounded M64xN128 macro-swizzled CTA traversal against the actual ping/pong
          production route. It changes only the one-dimensional CTA visit order so four token tiles
          reuse each output-row/weight tile consecutively; production arithmetic, per-CTA staging,
          eight native IU4 WMMAs, 88 VGPR, 17,152-byte LDS, and zero-scratch ceilings remain fixed.
          Require the independent oracle, active-output rewrite, exact tail/status behavior, all
          eight qualified shapes at T={1,024,2,048,4,096,8,192}, and direct alternating `auto`
          timing. Admit only with no cell slower than `1.01x` and a strict weighted-P2048 win; if it
          loses, remove the qualification-only route and retain ping/pong production unchanged.
          The fresh retained report
          `profiles/bench/r9700-a8q4-prefill-cta-m64n128-swizzle-ab-20260904.json`
          (SHA-256 `f4243888b6c1c70100c475a35a78d0d08e9e93d0a5926e6b27d6658b818ecfc8`)
          rejected it: weighted P2048 improved slightly to `0.9941328147x`, but the
          N=12,288/K=5,120/T=2,048 cell regressed to `1.0151797184x`, violating the
          predeclared `1.01x` per-cell ceiling. The qualification API, traversal template,
          resource fields, and harness modes were removed; production remains the unswizzled
          M64xN128 ping/pong route.
        - [x] Qualify one weight-only non-temporal-load variant of the restored production CTA.
          Preserve its 2-D traversal, activation/status/output policies, arithmetic, barriers, and
          staging; apply gfx1201 `TH_LOAD_NT` only to the four static packed-weight dword sites and
          two weight-scale halfword sites. Require symbol-local absence of new buffer/flat loads,
          exactly eight IU4 WMMAs, 88 VGPR, 17,152-byte LDS, zero scratch/spills, complete numerical
          coverage, direct alternating `auto` timing, no cell above `1.01x`, and a strict weighted
          P2048 win. Cache counters are diagnostic and cannot admit a timing loss. The fresh retained
          report `profiles/bench/r9700-a8q4-prefill-cta-weight-nt-ab-20260904.json`
          (SHA-256 `ec543f022dbfe8c50f3d9679dfd7763c03ae193ceebf3c459401ec59a4d45f65`)
          rejected the challenger: its weighted P2048 ratio was `7.6551502799x`, and every one of
          the 32 measured shape/token cells lost. Bypassing cache destroyed essential cross-CTA
          weight reuse, so the qualification route and harness modes were removed and production
          remains the ordinary cached M64xN128 ping/pong route.
        - [x] Qualify and reject next-G64 scale prefetch in the production M64xN128 ping/pong
          geometry. The candidate moved the activation-scale and weight-scale halfword loads into
          the existing four-code-dword WMMA overlap window without changing arithmetic, LDS,
          barriers, or output ownership. Static qualification passed at eight native IU4 WMMAs,
          89 VGPR, 17,152-byte LDS, 512 threads, occupancy 16, and zero scratch. Exact production
          parity, the sampled independent FP64 formula, full poisoned-output rewrite, and
          tail/alignment/nonfinite-status gates passed over MLP-down, GDN value-Z, and GDN output
          at T={1,024,2,048,4,096,8,192}. Physical timing rejected it: all 12 cells lost
          (`1.016370x`--`1.048674x`), and trace-call-weighted P2048 regressed from
          `375.835487` to `391.499994 ms` (`1.041679x`, `-15.664507 ms`). The immutable report is
          `profiles/bench/r9700-a8q4-prefill-cta-scale-prefetch-ab-20260904.json`, SHA-256
          `aaeeaf16a0040d5a23046d40fff532f34805124cdd3e5bbeb5d0a721962ad8ef`; its design authority is
          `profiles/bench/r9700-post-k128-q4-scale-prefetch-static-design-20260904.json`, SHA-256
          `577a639c825e7125b6b3b1da37ac9e40ab6dc475dd5e78b5ca3e4bb4815b5161`. Carrying the extra
          scale values lengthens live ranges and worsens the selected pipeline despite removing
          the serialized loads. The qualification API/template/checker/harness surfaces are
          removed and production remains the ordinary post-compute-scale-load route.
        Second, test one shared
        activation quantization for the independent full-attention query-key/gate-value and GDN
        query-key/value-Z projection pairs. The current source audit proves that sharing is exact
        only for all-Q4: each pair consumes the identical represented BF16 K=5,120 view, whereas
        the mixed profile's second member uses the distinct A8W8 boundary. At P2048, removing all
        64 duplicate all-Q4 quantizations avoids 2,034,237,696 logical bytes and 64 launches, but
        the fresh selected-region trace bounds their summed service to about 4.684 ms, only 0.31%
        of the 1.531-second pass. Do not implement this as the current prefill repair; reconsider it
        only after material Q4/FP8 and attention work if a sub-percent cleanup remains relevant.
        Only then, if cache/latency evidence supports it, test a
        narrow read-only next-matrix prefetch in the fixed target schedule with explicit ordering
        and storage ownership. Weights are already resident in ordinary device memory, so reject
        full-matrix cache warming and `hipMemPrefetchAsync` as inappropriate. Preserve dependent
        state ordering and admit no A/B without legal ISA/resources, numerical qualification, and
        matched-`auto` whole-prefill or decode improvement without moving the bottleneck elsewhere.
        Attribute the current low
        150.47/73.33 prefill tok/s and any material gap in the current MTP3
        19.24458338/7.54 decode tok/s at
        8K/32K before calling performance final; retain only the profiler evidence needed to support
        the identified bottleneck or roofline claim.
        - [x] Establish the usable gfx1201 profiling boundary and retain one honest counter control.
          Runtime/kernel/memory traces, SQ activity, and relative GL2C/TCP hit ratios are usable;
          absolute cache/HBM bytes and VALU/LDS/scheduler-stall attribution are not. The retained
          `profile_standard` counter CSV is
          `profiles/rocprof/xattention-b16-all-q4-g16-c1-8k-cache-pmc-20260904/cache-pmc_counter_collection.csv`
          (SHA-256 `7bc45f6fc1342ec4d49a42e3a195ecbf9e02e3e59c219c1682eae730989a9cae`).
          This closes counter-capability research only; selected-route `auto` roof analysis and
          controlled cache/prefetch/overlap A/Bs remain in the open parent item.
        A preliminary real-model 8K pass over the redesigned G16 consumer establishes that dispatch
        PMC is operational under `profile_standard`: all GL2C/TCP/activity controls were positive,
        aggregate GL2 hit rate was `98.371%`, and aggregate TCP hit rate was `77.692%`. The raw CSV
        SHA-256 is `7bc45f6fc1342ec4d49a42e3a195ecbf9e02e3e59c219c1682eae730989a9cae`.
        Its 32 selected dispatches have the exact `24 heads * 256 B16 tiles * 256 threads` grid and
        total `4.5698 s` (`126.027 ms` median), only `2.76%` of that pass's prefill time. This rules
        out the new consumer as the owner of the stable-profile whole-prefill duration, although its
        real 4,096-row scaling remains about `3.16x` slower than the hot 128-row fixture extrapolation.
        Its timing is explicitly inadmissible because `profile_standard` pinned low clocks; repeat
        cache attribution on the selected route and retain ordinary `auto` timing separately.
        The fixed ordinary diagnostic completed that separate measurement. Its strict validator
        reconstructs the exact device-0 C1 8K+256, chunk-4096, graph, three-repetition/one-warmup
        command; requires draft window zero to resolve to `spec=none`; rehashes the executable,
        artifact, and corpus; and derives every retained aggregate from the raw repetitions while
        requiring all speculative counters to remain disabled and zero. The current executable is
        `build-r9700-xattention-model-s16-tau900/bench/ninfer_bench` (SHA-256
        `a70a034a9d8161b2a828eb581c5b0e423b6f493ff4af793e0819a2e1fc5380f6`). Its designated
        output directory passed with every speculative counter disabled and zero. Three `auto`
        repetitions measured `239.5738442` prefill tok/s (`34.1947324 s`) and `8.354688852`
        ordinary decode output tok/s (`30.64198944 s`), with `3.963623176` whole output tok/s over
        `64.84022237 s`. The manifest SHA-256 is
        `a2ce50522910bd1442156383c85f359315c33bc5964578ba3ca28bf9685e104a` and the raw report
        SHA-256 is `049f3e4712a65ba618f28b47d830a96019cd16dea8dd70a920e1385aca4f1c96`.
        Thus the retained `19.24458338` tok/s row is explicitly MTP3, not ordinary decode.
  - [x] Physically admit and promote the split-512 long-context decode-attention leaf. The retained
        schema-v2 report at `profiles/bench/r9700-split512-attention-admission-20260904.json`
        (SHA-256 `bb4f0f2fba4773ab7c16e1f9c2e746fa9105f19ab4073fdffaa810108f2e4f1e`)
        passed the complete stored-byte FP64 oracle, invalid-input/native-workspace boundary, and
        fixed-address Device Graph suites before timing. Its 20-row/20-sample interleaved matrix
        measured complete incumbent/split ratios from `6.323x` through `17.646x` across G16/G32,
        8K/32K, T=1, and every active prefix of fixed-width T=4. Production now selects split-512
        only for context>=8,192 with ordinary T=1 or fixed-width T=4, retains the fused route for
        metadata-bearing T=1, retains the short-context WMMA/score-streaming routes, supplies exact
        caller-owned storage, and assigns distinct ordinary/MTP/DFlash graph topology classes. The
        native 262,144-token scratch is 37,847,040 bytes for T=1 and
        151,388,160 bytes for T=4; it aliases across request slots, all 16 full-attention layers,
        and sequential Text/MTP calls, so modeled Text-prefill scratch still dominates the arena.
        The first chunk-screen attempt exposed a metadata-dispatch regression outside the admission
        fixture: a T=1 call at visible frontier 8,192 carried a device active-row pointer, the old
        row/context-only decision selected split-512, and that leaf correctly returned
        `hipErrorInvalidValue` because T=1 has no admitted active-row form. Workspace sizing and
        launch now share one metadata-aware selector: metadata-bearing T=1 retains the fused leaf,
        while admitted ordinary T=1 and all fixed-width T=4 metadata forms keep their prior routes.
        `ninfer_r9700_split512_routing_host` passes the 8,191/8,192 row/metadata/workspace boundaries,
        and a direct 8K MTP reproduction completes in
        `profiles/bench/prefill-split512-failure-diagnostic-20260904.json`. This diagnostic fixes the
        campaign blocker but does not select a prefill chunk or supply terminal performance evidence.
        Post-selection command preparation is complete, without physical evidence, in
        `profiles/bench/post-terminal-selected-hardware-use-20260905` and
        `profiles/rocprof/selected-ordinary-decode-memory-prepare-20260905`. The historical
        `profiles/rocprof/selected-mtp-shortlist-head-prepare-20260905` package and its standalone
        producer/finalizer tooling are removed rather than retained as a dormant product lane. The ordinary-decode package reports
        only supported relative cache/activity proxies and fails closed on unavailable or zero
        counters; it does not claim physical HBM bandwidth.
  - [ ] Held behind the immediate dense C1/P2048/spec-none `>=2,000 tok/s` floor and
        practical-ceiling gate above; the user-authorized smaller-gain exact-Q4 continuation does not
        authorize this downstream campaign. After those gates pass, use the
        four rebuilt static compile profiles to run all twelve recipe/cache/attention
        prefill-chunk screens, then rerun both globally selected finalists for all twelve candidates
        before selecting one chunk. After selection, run all 48 post-promotion capacity cells
        (dense/XAttention x all-Q4/mixed/four-role-hybrid x G16/G32 x C=1..4),
        remeasure the additional Device Graph executable allocation, and discard every
        pre-promotion capacity/whole result from terminal assembly. This remains the capacity
        uncertainty even though exact layout replay shows no increase in the modeled global
        workspace arena.
        - [x] Rebuild the four compile profiles shared by the twelve recipe candidates after
          the K128 gated-RMSNorm promotion, rejected M64N256 cleanup, and current target/package
          changes. The four explicit benchmark/runtime-planner targets build successfully,
          all four `--help` boundaries pass, and the binaries bind Release/gfx1201, A8Q4/A8W8,
          native FP8-QK, M64xN128 ping/pong, the intended G16/G32 cache group, and either dense or
          B128-S16-tau900 XAttention. The fresh SHA-256 identities are
          `40940e9e9f31c4cd18f6d7267a3da703fa20a3132075e2ae6013f3c87c6002f4` (dense G16),
          `62853a231c801bbfe696fad8d42bb566ae683bb230d4f52b6349b0fbc2132cf6` (dense G32),
          `4d950a6938cd90377969cce200d44bd432a7ae760d50dd4d017c40e90dc21823`
          (XAttention G16), and
          `5209481b69406da7833bc291b3a8724571de11c66695c10d8d9dc74d7c0b4782`
          (XAttention G32). The parent remains open for physical artifact-bound screening,
          finalist reruns, selected-chunk binding, and the 48 capacity cells. All eight intended
          all-Q4/mixed screen directories and five finalist directories predate these executable
          bytes and are inadmissible; no four-role screen or finalist exists, and the global
          selection record remains absent. The same four binaries are reused, but each of the
          twelve physical runs must bind its own artifact/profile identity and exact adjacent N16
          migration receipt; the four-role runs must additionally bind their hybrid
          selection/planner authority.
        - [x] Close the CPU campaign contract. Every Pareto capacity invocation now requires the
          one schema-v2-selected prefill chunk instead of silently using 4,096, binds it in the
          schema-v14 manifest and all four C=1..4 commands, and rejects any raw capacity report
          whose Device Graph allowance/observed allocation is missing, mistyped, or exceeds its
          plan. Flattened evidence retains both graph byte values. The terminal contract now spans
          twelve candidates and the exact 48-command Cartesian inventory. The executable owner is
          `profiles/bench/post-chunk-twelve-candidate-20260905`; physical execution remains open
          until the 20260905 chunk pipeline publishes its validated schema-v2 selection. The
          retained schema-v14 manifest must be published atomically in its own directory. The
          shared-runner durable-publication and DFlash-hybrid preset patch is complete; its focused
          tests cover the exact hybrid companion authority and allowed preset boundary. Refresh the
          affected downstream closures only after the P2048 floor reopens base selection, not as a
          substitute for the still-missing physical rows.
        - [x] Close the matching whole-model CPU contract. Each capacity-eligible post-promotion
          profile receives one ranking spec-none ordinary `pareto-whole` matrix fixed to C=1..4,
          the schema-v2-selected chunk, one 8K+256 and one 32K+256 fresh-prompt row per C, and one
          warmup plus three measured repetitions. Any retained MTP3 report requires the optimized
          proposal head, nonzero drafted rounds/tokens, every output token, and exact per-repetition/
          lane target parity against its matched ordinary diagnostic; it is exact-token/state/graph
          regression evidence only and cannot enter base recipe/cache/attention ranking. The
          runner now checks and records `auto` before and after terminal speed evidence, while
          schema-v7 assembly requires the manifest-level chunk to equal every command. The dated
          output authorities remain absent pending the selected-chunk record and capacity runs;
          the prepared runner/validator must be revised to this ordinary-ranking contract before use.
        - [x] Close selected-chunk propagation through the remaining product command boundary. The
          compiled Engine, CLI, serving, benchmark, and PPL defaults share the one provisional
          product constant, while physical campaign commands remain explicit and terminal assembly
          has no silent 4,096 fallback. Serving corpus schema v6 and serving-concurrency execution
          now require the selected chunk and validate the exact cache group and dense/XAttention
          compile profile before accepting evidence; resume binds all three fields. The combined
          CPU suites pass 91 benchmark-tool, 99 PPL-tool, and 7 serving-corpus tests. This closes
          command preparation only: it neither selects a chunk nor creates capacity/whole evidence.
  - [x] Add the focused diagnostic `pareto` matrix preset for matched 8K/32K MTP3 prefill and graph
        decode at C=1..4, plus `pareto-whole` fresh-prompt end-to-end and `pareto-feasibility` automatic
        required-workload checks. The initial schema-v18 raw reports and schema-v11 matrix manifests bind Q4/W8 activation
        profiles, the exact FP8-Q/K T1>=64/T2>=320/T>=3-stream classifier, artifact identity/hash,
        and benchmark-executable hash; strict validation rejects missing rows, non-finite timing,
        incoherent acceptance, and same-path artifact or executable replacement. Feasibility output
        labels the binding constraint and cannot masquerade as an uncensored maximum-capacity
        objective. The distinct
        `pareto-capacity`/`dflash-capacity` presets now set the model-native 262,144-token ceiling
        and classify the exact resolved effective maximum as device-memory- or model-context-bound;
        Earlier C=1..8 base-capacity runs are retained only as superseded history. Fresh exact
        schema-v20/schema-v13 C=1..4 capacity was completed for all four all-Q4 dense/sparse
        G16/G32 candidates and all four mixed dense/sparse G16/G32 candidates at chunk 4,096. After
        the B16 consumer rebuild, only the four dense matrices remain byte-matched to their current
        executables, and they remain admissible only if chunk selection retains 4,096. All four
        sparse capacity matrices must be rerun for current executable provenance; if another chunk
        wins, all eight must be rerun because chunk/query-row workspace is part of the capacity
        contract. The mixed C7/C8 failures are out of scope and do not end its whole or potential
        DFlash path. Every whole-inference matrix remains outstanding, so the compound item stays
        unchecked.
        Historical mixed C1..4 maxima were G16 262144/314112/301888/289664 and G32
        262144/326656/313984/301248 tokens; they justify restored eligibility but are not current
        admission inputs. The runner and schema now make `pareto-whole` exactly one matched
        spec-none ordinary ranking row per C=1..4; MTP3 remains optional diagnostic evidence.
  - [ ] Held behind the immediate dense C1/P2048/spec-none `>=2,000 tok/s` floor and
        practical-ceiling gate above; the user-authorized smaller-gain exact-Q4 continuation does not
        authorize this downstream campaign. After those gates pass, establish
        the final matched 8K/32K prefill quality authority for all three recipe branches,
        both attention profiles, and G16/G32.
        Historical two-recipe acquisition is complete: its eight candidate cells have finite, complete, index-aligned
        FP32-NLL and I32-argmax sidecars with retained hashes. Against one retained BF16 realization,
        the source-MSE mixed recipe measured within the accuracy tier in all
        cells: 8K G16/G32 are +0.012077/+0.012658 mean NLL with 3/2 new severe positions, and 32K
        are +0.015790/+0.015485 with 16/16 against a 17-position budget. All-Q4 measured outside
        accuracy but within capacity-speed in all cells: 8K is +0.039509/+0.041056 with 9/10 new severe, and
        32K is +0.044561/+0.046032 with 41/39 against a 41-position budget. The exact paired
        distributions, memberships, artifact/sidecar hashes, and direct group diagnostics are in
        `profiles/ppl/q4-a8-final-8k-32k-quality-comparison.json`; neither quality nor scorer wall
        time selects cache group or recipe without the remaining capacity/whole-inference matrix.
        These original one-realization labels are retained history. The dense all-Q4 candidate
        sidecars have since passed the exact-v3-authority rebase recorded below; sparse all-Q4 and
        both mixed routes remain open.
        `profiles/ppl/terminal-quality-recovery-20260905` is the prepared six-authority recovery
        owner and is blocked on the selected chunk; it has not produced any of the missing physical
        candidate sidecars. Its CPU preflight now reopens the target-owned complete 18-shard BF16
        checkpoint for every selected chunk, including the retained-4096 branch, before any
        GPU-capable subprocess. It rejects occupied output names with `lexists` semantics so a
        dangling symlink cannot bypass freshness, and publishes the validated six-entry authority
        map by an exclusive same-directory hard link rather than an overwriting rename. The current
        retained-evidence validation and package checksum closure pass; this hardening creates no
        quality evidence and does not close the physically pending parent.
    - [x] Localize the BF16-source nondeterminism originally attributed to recurrence with targeted
          repeated spans and first-divergence
          state/logit evidence; bind the complete imported scorer implementation, interpreter,
          PyTorch/FLA/ROCm environment, and device identity rather than only the `ppl.py` wrapper;
          eliminate the nondeterminism or replace the execution with a deterministic authority and
          validate it by exact repeated sidecars; then recompute every BF16-derived mean-NLL,
          severe-membership, and greedy diagnostic from the already fixed candidate sidecars before
          allowing schema-v7 quality eligibility or terminal selection.
          This item is closed: the retained 12,289-token fresh-process traces have identical bound
          inputs/provenance and match through layer-3 GDN, masked QK, and FP32 softmax; their first
          differing finite checkpoint is KV-head 0's FP32 PV result at rows 12,224--12,287. Thus the
          defect was the backend-dependent probability-times-value reduction order, not recurrence
          state or scorer orchestration. The replacement uses ascending absolute 8,192-row FP32
          `torch.mm` chunks with ordered in-place FP32 accumulation. The current strict comparator
          reopens both replacement campaigns and reproduces the retained schema-v1 comparison
          byte-for-byte, including exact untraced 8K and 32K NLL/argmax sidecars. No further GPU
          repeat is required for this issue. Candidate cells whose old executable or rebuilt sparse
          consumer provenance cannot be reopened require fresh candidate acquisition under the
          separate quality parent; they are not unresolved BF16-authority nondeterminism.
          - [x] Implement the opt-in bounded BF16 stage trace and strict CPU comparison tool. The
                trace binds the scorer implementation and execution environment and hashes first/last
                hidden and logit slices without becoming quality evidence. The documented physical
                procedure starts with two fresh 4,097-token processes and escalates to 8,192 only if
                the first comparison is exact. Physical detailed traces localized the prior first
                divergence to layer 3's attention PV result. At that diagnostic stage the parent
                remained open for full deterministic-profile repeats and gate recomputation; the
                subsequent children below completed both.
          - [x] Promote the localized deterministic execution profile into the ordinary scorer:
                before importing PyTorch/FLA/backend code it requires hipBLAS, disables rocBLAS
                atomics, enables strict non-warn-only PyTorch deterministic algorithms, and rejects
                conflicting caller settings. Ordinary results and reusable campaigns now bind the
                complete scorer/FLA trees, interpreter/package/runtime/device environment, and
                deterministic/reduction state. Two independent trace-enabled 8,192-token runs are
                byte-exact at every traced stage and in their NLL/argmax sidecars, proving the bound
                profile removes the observed recurrence. They remain diagnostic because tracing is
                enabled. Candidate-campaign reuse now validates every original candidate identity,
                command, raw JSON, and sidecar hash, strips old BF16-derived fields, and recomputes
                them against a new authority without rerunning unchanged dense candidates. The full
                non-trace A/B campaigns are now complete: 8,192 is byte-exact and matches the traced
                repeats, but 32,768 remains nondeterministic despite identical bound provenance.
                At 32K, 15,758 of 16,383 NLL values and 125 argmax values differ; mean NLL is
                `1.7285508015` versus `1.7288484543`, with maximum paired NLL difference `0.675375`.
                The failed exact comparison is retained at
                `profiles/ppl/bf16-reference-deterministic-repeat-comparison-20260904.json`; it is
                diagnostic only and cannot authorize reuse or schema-v7 selection. Long-context
                first-divergence localization is now decisive: paired fresh-process 12,289-token
                traces match through layer 3 GDN, masked QK, and FP32 softmax, then first diverge at
                `layer-03.attention-detail.kv-00.pv-fp32.last`, proving the growing-prefix
                probability-times-value reduction is the source rather than recurrent state or QK.
                The diagnostic comparison is retained at
                `profiles/ppl/bf16-deterministic-12289-repeat-comparison-20260904.json` (SHA-256
                `361990647cf2be46974c0f1e0f2b152dd174a23571945b21bcf9356349ed1c98`).
                The fixed-order PV replacement and v3 fused-recurrent GDN route are implemented.
                The v3 process boundary now canonically fixes TunableOp, TF32, Triton, rocBLAS,
                ordinary launch scheduling, and allocator-cache controls before imports; forbids
                architecture, Tensile-library, and custom-allocator overrides; and binds highest
                matmul precision, actual gfx1201, plus resolved Triton/AMD lowering controls in
                ordinary and schema-v8 trace provenance. Earlier incomplete v3 output is rejected.
                A host-defined full-span numerical probe now covers fused-recurrent at T=4,095 and
                T=4,096. Its independent serial FP64 oracle traverses every row for five selected
                value-head/value-feature columns, retains all K=128 prediction/output terms, and
                checks five output rows plus all 128 final-state elements per column with exact
                input/result/source/execution hashes. The retained physical report
                `profiles/ppl/bf16-gdn-full-span-probe-v3-20260904.json` has SHA-256
                `61dc9b76ee21e04ccb277dd21952848fe5353683da724fa45531db14c31bc34a`, passes both
                extents, and bounds the largest sampled BF16-output absolute errors to
                `7.8929979e-05` at T=4,095 and `8.5443700e-05` at T=4,096, with corresponding
                complete sampled-column FP32 final-state maxima `1.3657522e-07` and
                `1.3202331e-07`. This closes the production-span GDN operator diagnostic but
                cannot substitute for fresh complete scorer repeats.
                Two standalone untraced 8K runs have byte-identical NLL sidecars (SHA-256
                `6ba4009ac23da9c831b475ce38fa98c377fd504ade81733de711be95e4092c0e`) and argmax
                sidecars (`c9904bd5b09d00666aee8500023cb52cc6f22d5c369cc00ceba57ea04a6d80c1`), with
                4,095 finite scores, mean NLL `1.8653729179`, PPL `6.4583438691`, and 46 severe
                positions. Their scorer-tree hashes differ, so they are retained arithmetic
                diagnostics rather than the same-current-tree repeat authority. Both current-tree
                v3 campaign 8K cells independently reproduce those exact two
                sidecar hashes under scorer tree
                `b09714374eed958d9e508ba415c6a4b2b913e944bada39a8d177be4b8cb113b7`.
                The complete v3-A campaign is retained at
                `profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json`
                (SHA-256 `9b402ad02be0f27d80530bd1eba4ead7c0179c658aa9654d43ca5b0fb9ca41f0`).
                Its 8K cell has the metrics above; its 32K cell has 16,383 finite scores, mean NLL
                `1.7288913487`, PPL `5.6344038553`, and 148 severe positions. The 32K NLL/argmax
                sidecars are respectively SHA-256
                `f5da949265ff566ebc368e3521cadc3f2ea7c687f1072ec993dbaec101ab1ab4` and
                `ec4e4d6b731d837f98b5f8bac7161ddacb9e55f0cf1842d68b58b9a2957a314b`.
                The matching v3-B campaign is retained at
                `profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-b-20260904/results.json`
                (SHA-256 `7b3d9f496c02185e467408c8aa67bd4af4fcd3b1e3bcc436be72ab3ac877b009`).
          - [x] Retain and strictly validate the exact current-tree non-trace BF16 authority repeat.
                `profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json`
                has SHA-256
                `9b13916a2a3ca8b8e5d01352f98a8306414f4e13a79989c19b1d1248d5eb00de`
                and binds the two complete v3 campaigns above. Both 8K and 32K semantic reports,
                finite sidecars, and exact outputs match: 8K NLL/argmax SHA-256 are
                `6ba4009ac23da9c831b475ce38fa98c377fd504ade81733de711be95e4092c0e` /
                `c9904bd5b09d00666aee8500023cb52cc6f22d5c369cc00ceba57ea04a6d80c1`,
                and 32K are
                `f5da949265ff566ebc368e3521cadc3f2ea7c687f1072ec993dbaec101ab1ab4` /
                `ec4e4d6b731d837f98b5f8bac7161ddacb9e55f0cf1842d68b58b9a2957a314b`.
                The comparator reopens both raw campaigns, recomputes corpus/manifest provenance
                and NLL-derived aggregates, validates exact sidecar cardinality/content, and
                rejects aliased repeat inputs or output paths.
          - [x] Rebase the fixed dense all-Q4 G16/G32 sidecars offline against that exact authority.
                `profiles/ppl/xattention-dense-q4g64-v3-rebase-20260904/results.json` has SHA-256
                `fc05174d70421f60351c0cd87c587d04f74eabd205e0d9431887a60c7c7bc4fa`
                and binds comparison SHA-256 `9b13916a2a3ca8b8e5d01352f98a8306414f4e13a79989c19b1d1248d5eb00de`.
                G16/G32 pass the capacity-speed tier at 8K with mean-NLL deltas
                `+0.0397932579` / `+0.0413409097` and 10/11 new severe positions against budget
                11, and at 32K with `+0.0447215401` / `+0.0461926079` and 40/38 against budget 41.
                This closes only dense all-Q4 quality admission; fresh sparse all-Q4 and mixed
                dense/sparse acquisition, matched whole evidence, and schema-v7 selection remain
                open under the parent.

## Active: DFlash2 companion for the selected base A8 profile

- [x] Implement an explicit R9700 DFlash2 evaluation artifact identity and converter from the
      source safetensors inventory, retaining the model-specified BF16 selector codebook and using
      directly bound persistent integer draft matrices without runtime repacking or hidden device
      allocation. Define the exact INT4/W8 role assignment from source evidence and capacity rather
      than inheriting NVFP4-era choices.
      - Completed: the exact 81-source-tensor/66-artifact-object inventory, three explicit
        DFlash2-Q4 evaluation identities, converter preflight, binder, and DFlash-sized caller-owned
        A8G64 workspace are implemented. Publication is recoverable: atomic pending/final reports
        retain source/base/output SHA-256 provenance, and report finalization validates an existing
        completed artifact without reconversion. Both real base artifacts pass preflight. Actual
        multi-GiB conversions are complete for the all-Q4 and mixed base profiles; the hybrid
        companion remains deliberately absent pending base selection. The retained atomic
        conversion reports bind all-Q4 output `ba39608b...9cdc` (16,382,310,912 bytes) and mixed
        output `c2dcb265...a43c` (24,090,686,464 bytes) to their exact base and source hashes;
        both actual sizes equal the projected sizes. The companion matching the base recipe chosen
        by the C=1..4 Pareto gate remains eligible for the draft-quality, acceptance, capacity, and
        whole-inference gates below; the other remains exact conversion provenance.
- [x] Qualify every added DFlash2 integer Linear route at its real shapes with canonical represented
      BF16-to-A8 activation codecs, independent exact packed-code and FP64 formula oracles, malformed/
      alias/nonfinite coverage, native gfx1201 ISA/resources, and interleaved timing against the
      corresponding exact/W8 control. Use "adaptive A8" for INT4 draft matrices; do not describe
      this as W8 activations.
      - The retained schema-v2 A8-Q4 artifact sweep now covers all five new DFlash matrix shapes,
        22 unique shapes and 220 token extents in total, with all 32 DFlash matrix occurrences
        attributed. Every extent passes the independent complete FP64 formula within two BF16
        steps and retains seven-trial interleaved timing samples. Physical composite qualifiers
        additionally pass Q4 through grouped dynamic-convolution prepare and both DFlash selector
        chain/tree arena routes, including mapped selection and malformed Q4 plane rejection.
- [ ] Select the base recipe before DFlash GPU execution. The earlier mixed G16 C7/C8 and G32 C8
      failures remain exact out-of-scope stress provenance, but cannot exclude the mixed recipe
      under the C=1..4 product cap. The four-role FP8/Q4 recipe remains a distinct eligible third
      branch and may not inherit the all-Q4 identity or companion. Run DFlash only for the
      companion of the selected base recipe;
      do not duplicate the shortlist and matrices for the losing recipe.
      After selection, materialize the selected recipe's currently absent registered N16 DFlash2
      companion by byte-exactly copying that exact authority-bound base and appending the registered
      66-object Q4/BF16 DFlash inventory.
      Its conversion report must bind the base conversion-receipt digest plus recipe, selection,
      object-plan, source-index, and ranking hashes. Projected size/capacity arithmetic is not
      admission: fresh selected-K/W C=1..4 capacity and whole evidence is mandatory.
      The prepared entry point is
      `profiles/bench/selected-dflash-prepare-20260905/prepare.sh`. It fails closed until the exact
      schema-v7 winner exists, accepts all-Q4, mixed, or four-role hybrid without a fallback, and
      binds the selected artifact/build/chunk/group/Text-prefill profile to exactly its registered
      companion. All three current-N16 companions are absent; exactly the winner's companion is
      converted, while obsolete pre-N16 companions cannot satisfy the byte/report/base-identity
      preflight. Conversion binds the exact ROCm Python launcher, its `pyvenv.cfg`, Torch/HIP and
      safetensors identities, and the required ROCm library path; preparation fails before
      publication if that environment is unavailable. The staged future
      command runs the fixed C1 shortlist, derives its K/W frontier, runs C1..4 capacity for every
      frontier tuple, runs `dflash-pareto` only for capacity-eligible tuples, and then invokes the
      no-overwrite schema-v2 owner. DFlash proposal and target verification remain dense, its 32
      persistent matrices are Q4G64 with dynamic A8G64 activations, all 34 non-matrix/codebook
      objects remain source BF16, and the separate fixed runtime state remains private BF16.
- [ ] Run and retain matched DFlash2 quality evidence for the selected base recipe's companion under
      the cache group carried by the schema-v7 `terminal_production_selection`. DFlash itself
      remains on its qualified dense attention boundary: retain aligned target/draft outputs,
      deterministic proposal and final target tokens, exact ordinary-target output parity, and
      exact artifact/profile provenance. Target-model NLL is not applicable because DFlash changes
      proposal execution rather than the teacher-forced target distribution; the base artifact PPL
      campaign owns that numerical gate. Do not select a recipe from standalone synthetic error.
- [ ] Run and retain DFlash2 acceptance and whole-inference evidence for the selected base recipe and cache group at the
      startup-fixed C=1..4 workload, including resolved verify width, accepted tokens per round,
      fallback/repair behavior, prefill and graph decode throughput, capacity/headroom, and focused
      profiler attribution for any unresolved bottleneck before choosing the production companion.
      - [x] Add the focused `dflash-pareto` C=1..4 8K/32K matrix. It requires explicit startup K,
            retains requested/resolved verify W, artifact and benchmark hashes, per-position
            acceptance, and exact ordinary-vs-DFlash greedy token parity for every repetition and
            lane. Two isolated eager C=1 proposal/selector probes emit finite-logit, tree/head,
            top-16/64/256, reject, and by-depth counters plus ordered per-round proposal IDs,
            parent topology, and aligned licensed target tokens bound to the exact profile. Exact
            proposal traces and final target outputs are required across the repeated runs; both
            syncing runs are excluded from timing evidence. Physical candidate runs remain unchecked.
      - [x] Add the schema-v2 provenance-bound DFlash selection owner. The authoritative command is
            `python3 tools/bench/assemble_dflash_selection.py`; it consumes the selected base
            schema-v7 record, companion conversion report, physical shortlist, one C=1..4 capacity
            campaign for every shortlist-frontier K/W (including exact failure provenance), and a
            complete 18-point `dflash-pareto` campaign only for each capacity-eligible K/W. It
            reopens schema-v20 reports under schema-v14 manifests, recomputes
            shortlist/parity/determinism/generated-quality
            evidence, preserves exclusions and the full eligible frontier, and selects one static
            K/W by maximin whole throughput, capacity, acceptance, then numeric K/W only for a
            complete tie. The open parent items close only when the retained
            recipe/group-qualified DFlash selection record passes and its winner
            supplies the production companion K/W; no individual matrix or manual frontier choice
            closes them.

## Active: typed growing-cache ABI

- [x] Replace homogeneous dtype and quant-group identity for Text/MTP KV with a semantic
      Fp8KInt4V cache format.
- [x] Define typed K-FP8, V-INT4, and V-scale-FP16 plane views with no K-scale or K-mean fields.
- [x] Plan exactly three planes per text and MTP layer, with G16/G32 qualification selection tracked
      explicitly and no NVFP4 or INT8 growing-cache branch.
- [x] Make page-table, position uniqueness, status, and transaction-commit ownership explicit at
      the cache/Op boundary.
- [x] Port in-place compaction to copy the exact three-plane representation through the typed view.
- [x] Add cache-layout and physical-address tests at Qwen D256/Hkv4 geometry.
- [x] Bind the typed cache view into the Text and MTP decoder-state owners; remove their shared
      dtype, quant-group, Sage, K-scale, and K-mean cache identity.
  - [x] Add a Qwen decoder-state owner that plans and binds independent typed Text/MTP three-plane
        views with no legacy-format fields. It is now the canonical `DecoderState`/`PagedKVCache`
        owner; the parallel homogeneous BF16/INT8/NVFP4 decoder owner and its tests are deleted.
- [x] Wire typed cache append/compact transactions to the HIP A2 sources, including device-status
      clear/read and valid-frontier publication rejection.
  - [x] Qualify a HIP-resident all-layer typed transaction owner: contiguous append and monotone
        compaction require each layer on one ordered status word, advance a shared frontier only
        after zero status, and poison a failed/abandoned launched cache; exercise fragmented
        positions 63/64/65 and a nonfinite device-source rejection on the physical R9700.
  - [x] Bind that transaction directly to the canonical Qwen Text/MTP cache pool and one
        allocation-owned block-table row. The physical state qualifier launches every Text layer,
        commits host/device append and compaction frontiers, rejects a positive gap and a negative
        device-I32 position, and proves that invalid-position and nonfinite A2 status preserve the
        old frontier while poisoning only the affected sequence publication. The schedule's
        existing device I32 suffix is consumed directly; positions and status are caller-planned
        workspace, so no mutation performs hidden device allocation, H2D position staging, or
        runtime repacking.
  - [x] Bind causal cache reads to a non-forgeable `PagedKVLayerRead`. Committed reads capture the
        exact allocation, block-table row, allocation mapping generation, mapped-page count, closed
        publication frontier, and transaction generation; append transactions expose a pending read
        only after that layer's codec launch and only on the same ordered HIP stream. Mapping
        materialize/trim/bind/unbind and begin append/compaction invalidate older reads, compaction
        exposes no pending read, and commit/abort closes every pending capability. The physical
        persistent-state and target-attention qualifiers together cover pre-launch,
        unlaunched-layer, cross-stream, stale-generation, post-commit, compaction, and `row 0 ->
        row 1 -> row 0` rebind rejection while proving a freshly captured read remains usable.
- [x] Reject aliased logical-to-physical page maps at cache construction and RAM restore, with
      no-write rejection evidence for a malformed RAM image.
- [x] Make every raw QK corrupt-map output conspicuously NaN; reject nonfinite attention scales
      and malformed plane-layout selectors at every A2/A3 raw and typed-cache boundary, then
      requalify the complete oracle corpus.
- [x] Add independent byte-offset oracle coverage for all eight independent K/V/V-scale plane
      layout combinations, fragmented page tables, positions 63/64/65, sentinels, and compaction.
  - [x] The host cache-layout test derives byte offsets without planner formulas for every G16/G32
        × independent-plane-layout combination and fragmented physical-page coordinates; the
        device qualifier combines that layout matrix with nonidentity page tables, sentinels,
        exact 63/64/65 append, and exact three-plane compaction.
- [x] Add typed cache host spill/restore coverage for mixed K/V/V-scale plane layouts; all packed
      physical bytes round-trip exactly through the generic transport.
- [x] Bind the typed cache semantic fingerprint into the R9700 RAM-state owner and reject an
      incompatible typed-cache RAM image before restore.
  - [x] Define and qualify the typed FP8-K/INT4-V fingerprint (codec version, page/D256 geometry,
        capacity, layer/head count, V group, and all three physical orders); a mismatched image is
        rejected before any byte restoration. The HIP cache owner now snapshots/restores one typed
        page map, frontier, and exact three-plane image only after that fingerprint and every byte
        extent validate; a physical-device round trip and no-write mismatch rejection pass.
  - [x] Compile the production Qwen host-RAM tier against HIP and bind independent Text/backend
        semantic fingerprints into its version-5 image. A gfx1201 D256/Hkv4 qualifier round-trips
        two Text layers and one MTP layer through fragmented source/destination maps without an
        external host fence, proves both mismatch paths leave every destination byte unchanged,
        and exercises safe in-flight FIFO eviction.
- [ ] Held behind the immediate dense C1/P2048/spec-none `>=2,000 tok/s` floor and
      practical-ceiling gate above; the user-authorized smaller-gain exact-Q4 continuation does not
      authorize this downstream campaign. After those gates pass, select one
      G16/G32 plus dense/B128-S16-tau900 static profile only from the full matched
      quality, capacity, separately timed prefill/decode, and whole-inference Pareto gate. During
      the subsequent production
      cutover, delete the losing cache-group qualification/dispatch branches and the unselected
      compile-isolated attention profile; retain the selected profile's required dense T=1, decode,
      MTP/tree, DFlash, and non-prefill boundaries.
      `profiles/bench/terminal-static-selection-20260905` is prepared to consume the exact twelve
      candidate-local quality/capacity-outcome rows plus whole rows only for capacity-eligible
      profiles and publish one schema-v7 winner atomically; its absent output keeps this physical
      selection unchecked.
  - [x] Make the pending comparison a compile-time identity rather than a runtime cache selector.
        Separate G16 and G32 builds compile the selected group through the complete runtime and
        PPL executable, reject every other configured value, and emit `kv_value_group` plus the
        exact K/V/V-scale plane layouts for the campaign to verify against its profile label and
        fixed static layout. Both builds pass the C=1..4 runtime planner,
        persistent-state fingerprint/lifetime checks, and full D256/Hq24/Hkv4 physical attention
        oracle on gfx1201; the final group still requires the complete model-level Pareto gate.
  - [x] Supersede the schema-v18/v11 capacity runs and the interrupted phase run as historical,
        non-selection evidence: their reports bind the group and exact executable bytes but do not
        carry an explicit plane-layout identity. Retain their raw files; do not assemble them into
        the final frontier.
  - [x] Complete fresh exact C=1..4 product capacity for all eight artifact/cache/execution
        candidates after the tooling migration. Schema-v20 reports under schema-v13 manifests bind
        all-Q4 and mixed artifact bytes, dense or B128/S16/tau900 executable bytes, exact G16/G32
        layouts, and uncensored model-context- or device-memory-bound maxima. The earlier C=1..8,
        `pareto-capacity-layout-*`, `pareto-capacity-unprofiled-layout-*`, and
        schema-v18/v11 `pareto-capacity-max-*` campaigns remain superseded historical evidence and
        cannot enter the final C=1..4 frontier. Exact current values and hashes are retained in the
        candidate children below and `docs/performance.md`.
  - [ ] Held behind the immediate dense C1/P2048/spec-none `>=2,000 tok/s` floor and
        practical-ceiling gate above; the user-authorized smaller-gain exact-Q4 continuation does not
        authorize this downstream campaign. After those gates pass, complete
        the twelve schema-v14 capacity outcomes named in the XAttention admission
        children below plus the corresponding whole matrix for every capacity-eligible profile,
        then assemble one schema-v7 Pareto decision against each candidate's matched quality
        evidence. This is the aggregation owner for those same physical campaigns, not a second
        G16/G32 run. The schema-v7 decision must retain every eligible per-recipe winner and
        one `terminal_production_selection` binding the selected artifact, fixed cache, and compile-bound
        dense or B128/S16/tau900 execution identity.
        The shared physical owner is `profiles/bench/post-chunk-twelve-candidate-20260905`; do not
        launch a second capacity or whole campaign from this aggregation item.
  - [x] Make that final handoff fail closed across the selection boundary. The schema-v4 assembler
        accepts a candidate-local native schema-v6 PPL campaign directly, so dense and sparse
        profiles cannot be conflated through one global quality label. It reopens raw cells and
        hashed sidecars, requires the exact cache and compile-bound execution identity to match the
        capacity matrix for all twelve candidates and the whole matrix exactly for each
        capacity-eligible profile, and carries both identities into the schema-v7 classifier.
        Capacity-failed dense/sparse pairs remain non-comparable; deterministic same-recipe
        selection retains a winning cache and execution profile only for recipes with an eligible
        pair. The terminal rule must then retain exactly one production artifact/cache/execution
        winner. Schema v7 records every branch's distinct weight-storage profile and retains and
        reopens the complete schema-v4 input by path/SHA-256. Schema v7 carries no MTP-head or
        downstream-readiness marker; selected-profile NIAH and DFlash remain separate gates.
        DFlash and NIAH reject a missing terminal
        winner or any selected identity drift. The chunk selector and assembler now require the
        full twelve-candidate Cartesian set, so the older two-recipe chunk result cannot silently
        admit the hybrid.

## Active: HIP core cutover

- [x] Qualify an isolated HIP-only R9700 device context with gfx1201/wave32 enforcement, ordered
      compute/load/copy streams, blocking host events, and device event timing on the physical card.
- [x] Qualify HIP device-buffer transfer and aligned scoped arena ownership on the physical R9700;
      this is the allocation substrate for the pending paged-KV/materialization cutover.
- [x] Qualify HIP graph RAII capture, instantiate, and replay against a device-buffer byte oracle
      on the physical R9700; Engine graph replay remains pending eager model execution.
- [x] Replace CUDA-only root and source build ownership with a HIP-only gfx1201 production-core
      configuration. The root resolves the active ROCm Clang rather than the rejected `hipcc`
      wrapper, links `hip::host` directly, and builds the closed `ninfer_r9700_core` archive plus
      its physical-device qualifier; no CUDA target is conditionally retained in that CMake graph.
- [x] Expand the closed HIP production-core CMake boundary through artifact materialization,
      request memory, native paged/cyclic/GDN state, public Ops, target scheduling, Engine, and
      applications only as each CUDA contract is replaced; the current archive must not be
      represented as a buildable inference Engine.
  - [x] Add generic artifact framing/binding and direct final HIP materialization to the CMake
        graph; its physical qualifier passes exact H2D bytes and rejects legacy NVFP4 storage.
  - [x] Add generic `.ninfer` reader, binder, typed binding, and direct final materialization to
        the HIP build graph. The canonical HIP pinned event ring performs final H2D copies with no
        runtime repacking; its physical-device qualifier verifies out-of-order exact tensor bytes,
        retained resources, plan/accounting, and NVFP4/blockscale rejection.
  - [x] Add the exact Qwen3.8-27B R9700 family-runtime HIP instantiation and force its entire static
        archive through one link boundary. A physical planner qualifier constructs and finalizes
        ordinary, Vision, MTP3, MTP3+Vision, and DFlash4 profiles at every fixed concurrency C=1..4;
        this closes schedule symbol ownership but does not yet claim a public Engine/application.
  - [x] Link the sole-target public Engine over the exact runtime and restore native HIP ownership
        for the CLI, HTTP server, and PPL applications. A whole-archive Engine qualifier has zero
        project-owned or CUDA/NVTX unresolved symbols, physically exercises gfx1201 context
        construction/exception teardown, and rejects invalid artifact contracts; all three product
        executables build, their help is free of CUDA/NVFP4/cache-selector surfaces, and invalid
        artifacts fail through the public Engine contract. Real loading remains at the real-artifact
        gate rather than being simulated with a sparse synthetic file.
  - [x] Add startup-frozen request memory, typed FP8-K/INT4-V paged allocation/block-table
        ownership, BF16 cyclic state, and BF16/FP32 linear-attention state to the HIP core. The
        physical qualifier covers mapping publication, allocation release, ordered spill/restore,
        D2D lane/slot copies, and zeroing; legacy homogeneous growing-cache views are removed
        from the active core contract.
    - [x] Delete the duplicate R9700-prefixed cache owner and narrow `hip_device_qual` to its
          device/arena/graph boundary. Canonical decoder-state, RAM-cache, raw-Op, and target
          attention qualifiers retain the physical cache evidence without a second owner.
  - [x] Add the target-private Qwen3.8 D256/Hq24/Hkv4 full-attention leaf to the HIP archive. Its
        physical gfx1201 qualifier consumes the canonical generation-bound layer-read capability,
        matches an independent FP64 oracle for committed T=1..9/17/128 and same-stream pending
        attention over a fragmented 257-token table, rejects pre-launch/cross-stream/stale/poisoned/invalid
        state before dispatch, and makes negative/over-frontier device causal-position rows NaN
        without contaminating adjacent valid rows. The provisional G16/token-K/feature-V/
        feature-scale route family uses score streaming for wider/masked calls and the selected
        ordinary-causal FP8-Q/K WMMA crossover below; final static group/layout still requires its
        remaining product gates.
    - [x] Qualify exact packed-tree visibility in the same production leaf: each row attends its
          prefix plus only ancestor-mask-selected suffix tokens. The physical gfx1201 test uses
          context 257, prefix 253, positions 253..256, and masks 1/3/5/13 against an independent
          FP64 oracle in both rowwise and scalar-broadcast prefix modes; incomplete host pairs,
          unsupported prefix strides, and the WMMA route reject before launch, while bad device
          prefix/context/suffix/empty-selection metadata poisons only its own row.
    - [x] Add a device-I32 active-query-row scalar to the fused target leaf for fixed-width MTP
          panels. Null preserves the all-row specialization; counts in `[0,T]` execute the active
          prefix and write exact positive-zero tails before inspecting row query/position/tree/cache
          inputs, while negative or over-T values make every row NaN. The physical qualifier passes
          zero/partial/full/error/post-error causal cases and a partial tree panel against the same
          independent FP64 oracle while replaying one captured fixed-address Device Graph. The
          selected G16 token-K/feature-V/feature-scale kernels remain
          at 23 VGPR fixed or 24 VGPR device-count, 52 bytes LDS, zero scratch, and occupancy 16.
- [x] Port DeviceContext, arenas, transfers, timers, events, and memory reporting to HIP.
  - [x] Promote the qualified HIP ownership substrate into the sole project `DeviceContext`,
        `DeviceEventTimer`, `DeviceBuffer`, `DeviceArena`, `PinnedHostBuffer`, and
        `HostPinnedArena` contracts. The CUDA device/arena implementations and duplicate
        R9700-prefixed contracts are deleted; the HIP core qualifier covers Tensor allocation,
        range rejection, alignment/scoped rewind, pinned allocation, and ordered async copies.
- [x] Rename project-owned CUDA graph and transfer terminology to device/HIP terminology.
  - [x] Audit the closed, buildable HIP core/artifact/request-memory/persistent-state boundary.
        Its graph owner is the device-neutral `DecodeGraphDefinition`/`DecodeGraphExecutable`
        contract backed directly by `hipGraph*`; transfers use device-neutral buffer/cache verbs
        and explicit HIP streams. No CUDA-specific field, diagnostic, CLI/report key, or transfer
        timing name remains in this bounded slice.
  - [x] Replace the public and product-wide graph option, memory fields, emitted JSON/CSV keys,
        CLI flags, help/errors, application/serve consumers, benchmarks, tests, Python campaign
        tools, and active authorities with `use_device_graph`, `device_graph_*`, Device Graph,
        and `--no-device-graph`. The superseded CLI spelling has no compatibility branch and is
        explicitly rejected by CLI and serving parser tests. Historical plans and model cards
        retain their period-specific wording until their separately assigned cleanup.
- [x] Port canonical graph RAII capture, instantiate, update, upload, and launch to HIP; remove the
      temporary R9700-prefixed duplicate and verify all operations on the physical R9700. Captured
      Engine replay remains deferred until eager model execution passes.
- [x] Remove CUDA, NVTX, SM120 TMA, PDL, and NVIDIA-only build ownership rather than retaining a
      compatibility backend.
  - [x] Replace the common runtime tracing facade with native ROCtx while preserving the fixed
        `ScopedRange`/`Name`/`Category` caller contract. Category, name, and payload remain visible
        in an allocation-free range label; the production core links ROCtx directly, the header and
        namespace are ROCtx-owned rather than a retained NVTX compatibility spelling, and a focused
        qualifier proves nested RAII ranges leave the ROCtx push/pop stack balanced.
  - [x] Close the complete project-owned C++/HIP build boundary: exact case-insensitive scans of
        root/source/application/benchmark/test/R9700-tool CMake and 358 C/C++/HIP/header files find
        no retired backend, tracing, SM120, TMA, or PDL spelling, API, target, source, or filename.
        A fresh HIP-only configure with applications, benchmarks, and tests enabled builds every
        target; all five product/benchmark help routes and 19 focused CPU/API/protocol tests pass.
- [x] Replace every internal cudaStream_t/graph/error/event/transfer contract and all direct CUDA
      calls in core, artifact materialization, runtime, Ops, tests, applications, and tools.
  - [x] Remove direct CUDA contracts and calls from every source owned by the currently buildable
        HIP core, generic artifact materializer, startup-frozen request memory, typed paged/cyclic
        cache, and linear-attention persistent-state boundary. The active sources include HIP
        runtime types directly and the physical qualifiers link the required native ROCm owners.
        The unused PDL helper is deleted; broader runtime, tests, applications, and tools are
        tracked by their respective cutover items.
  - [x] Replace the serving request-log environment's CUDA runtime ownership and observable schema
        with HIP compile/runtime/driver versions plus the gfx architecture name. Schema version 17
        is qualified by a physical R9700 query reporting gfx1201 and ROCm HIP 7.15.26333; the old
        CUDA/compute-capability fields are removed rather than aliased.
  - [x] Replace the whole-Engine benchmark harness's CUDA device/version/profiler boundary and
        report keys with HIP device/runtime/driver and gfx architecture values. The benchmark and
        report sources compile through the ROCm 10 toolchain, and the report formatter host test
        passes with the R9700 identity and new schema fields.
  - [x] Replace the disabled legacy test graph with one native `BUILD_TESTING=ON` R9700 graph.
        CPU/API/protocol tests link the current Engine and serving targets; physical device tests
        are the production-owned exact/FP64 `tools/r9700` qualifiers, serialized through one GPU
        resource lock. Superseded per-Op device fixtures, old real-artifact identities, and retired
        format tests are deleted. The complete 53-test CTest run passes on physical gfx1201,
        including repeatable sparse target-binding fixtures and current CLI/serve identity and
        option contracts.
  - [x] Recheck every built archive/executable: undefined-symbol scans contain no retired runtime or
        tracing owner, and direct dynamic dependencies contain HIP/ROCtx but no alternate device
        runtime. Distro FFmpeg's generic OpenCL dependency is external media plumbing rather than a
        project launch/API backend.
- [x] Convert every .cu/.cuh implementation and CUDA-language CMake target to HIP/gfx1201 or
      delete it when it belongs to the superseded NVFP4/5090 path.
  - [x] Delete the final unbuilt CUDA-only Op microbench files and unused PDL helper after proving
        the native qualifier and whole-Engine benchmark graphs have no references to them. Port
        the retained sustained-memory bound probe to an exact-device gfx1201 HIP owner under
        `tools/r9700`: it compiles with the selected ROCm 10 toolchain, rejects a non-R9700 device,
        and uses a 4 GiB working set. After strengthening the checksum to cover every block, five
        0.25-second trials physically measured best/median read `636.2/636.0 GB/s`, kernel write
        `588.0/587.9 GB/s`, kernel copy `549.0/548.6 GB/s`, and HIP D2D `543.8/543.4 GB/s` against
        the R9700's 640 GB/s advertised bandwidth; the aggregate read checksum matched exactly.
        This records a hardware bound without entering a kernel-selection or end-to-end claim.
  - [x] Prove the final source-language closure directly: the owned source/application/benchmark/
        test/R9700-tool tree contains zero `.cu` or `.cuh` files, its CMake graph declares only
        C/C++/HIP and `gfx1201`, and the fresh compile database contains no CUDA-language source,
        architecture, or toolchain entry.
- [x] Establish the available HIP sanitizer and focused lifetime/error checks for arenas, views,
      cache mapping, graph capture, and asynchronous transfers.
  - [x] Add and pass the focused host AddressSanitizer build of the physical HIP ownership
        qualifier. The ROCm 10 build uses C++20, instruments host allocation/ownership while
        explicitly disabling unavailable GPU instrumentation, and exercises device/arena/graph
        lifetime plus ordered transfers on the physical R9700. LeakSanitizer alone is disabled
        because the process-global ROCm HSA runtime retains internal allocations through exit;
        AddressSanitizer remains active and passes with halt-on-error.
  - [x] Classify device sanitizer support for gfx1201 as unavailable in the installed toolchain.
        The CPU-only retained audit uses HIP 7.15.26333/Clang 23 commit `8f497e0`: the plain
        `gfx1201` compile succeeds only after warning that device AddressSanitizer is ignored, and
        `gfx1201:xnack+` is rejected as an invalid target ID. The exact commands/output are in
        `profiles/r9700/rocm10-tool-capability-audit.json`. This is an external target/toolchain
        limitation, not a missing project build flag.
- [x] Establish ROCm profiler recipes, trace capture, PMU counter capture, ISA disassembly, and
      reproducible workload metadata for every production kernel decision.
  - [x] Add the reproducible HIP runtime/kernel/memory trace recipe for the D256/Hq24/Hkv4
        page-boundary qualification workload; ISA disassembly is checked directly from the
        generated gfx1201 source. PMU selection remains driver-gated.
  - [x] Capture the first ROCm runtime/kernel/memory trace of that workload. The trace contains
        every append, compact, QK, PV, and fused-A3 dispatch; ROCm corrected inverted HSA dispatch
        timestamps, so its timing fields are retained only as provisional bring-up attribution.
  - [x] On the rebooted ROCm 10 baseline, capture a fresh runtime/kernel/memory trace and separate
        PMU databases. SQ busy/wave events are valid. The initial isolated generic and gfx12 wave32
        VALU/LDS plus TCP and GL2C probes remained zero despite known work. Subsequent
        dispatch-scoped collection resolved the TCP/GL2C scope failure and made relative hit rates
        usable; isolated VALU/LDS and absolute request-size traffic remain unavailable.
  - [x] Classify complete gfx1201 VALU/LDS/stall and absolute cache/HBM attribution as unavailable
        in the installed profiler release. The `rocm-tool-audit-test` and `rocm-tool-audit` targets
        under `tools/r9700` now validate the explicitly named, hashed SQLite captures without
        launching HIP: the retained trace has
        1,024 dispatches, 224 copies, and 3,130 runtime regions; `SQ_BUSY_CYCLES` and `SQ_WAVES`
        are nonzero. Isolated generic/wave32 VALU and wave32 LDS samples remain identically zero;
        gfx1201 request-size buckets are also known-zero and base counts may undercount. A later
        dispatch-scoped real-model pass produced positive TCP request/miss and GL2C hit/miss
        controls, admitting relative hit ratios but not absolute byte rates. rocprofiler-sdk 1.3.5
        commit `6b0e43f` produced the audit databases, and ROCm Compute Profiler 3.8.0 commit
        `16adc4d` advertises no gfx1200/gfx1201 analysis architecture. Unavailable events are not
        interpreted as zero hardware activity and cannot support a kernel decision.

## Active: production attention kernel family

### XAttention prefill sparsity

- [ ] Held behind the immediate dense C1/P2048/spec-none `>=2,000 tok/s` floor and
      practical-ceiling gate above; the user-authorized smaller-gain exact-Q4 continuation does not
      authorize this downstream campaign. After those gates pass, complete
      the native gfx1201 XAttention prefill admission gate over both G16/G32 instances of
      the typed FP8-E4M3FN-K/INT4-V/FP16-scale cache. Keep ordinary T=1 decode, MTP/DFlash verification,
      tree masks, and GDN layers on their qualified dense paths. The evaluator must use an
      explicit antidiagonal ranker and 64-token cache-page keep list; `tau=1` is the exact dense
      identity and the sparse threshold/stride are compile-isolated qualification profiles, not a
      second cache format or an unmeasured production default. Physical requalification selected
      the compile-isolated S16/tau900 candidate; dense attention remains the sole production route
      until the complete twelve-candidate, three-recipe decision retains capacity-failed pairs as
      exclusions, measures every eligible whole matrix, and closes the selected-profile
      NIAH gate close.
  - [x] Add an independent host keep-set oracle for the XAttention antidiagonal algorithm and an
        FP64 attention oracle over exactly the retained pages. Qualify causal/tail pages, sinks and
        recent-page retention, ragged prefill rows, malformed/nonfinite inputs, deterministic keep
        ordering, `tau=1` dense identity, and the real D256/Hq24/Hkv4 geometry.
  - [x] Correct the evaluator to use paper B128 query/keep blocks expanded to ordered B64 cache
        pages, count mandatory sink/current-block mass inside the tau budget, zero-pad rather than
        discard ragged Q/K groups, and take dense fallback for a B128-unaligned chunk. Keep `900`
        as integer-permille build encoding for mathematical `tau=0.9`; the threshold direction is
        the minimum cumulative probability mass.
  - [x] Restore a bounded skinny BF16-WMMA estimator: decode each FP8 K exactly once into
        caller-owned BF16 packed storage, reuse it across Q heads/blocks, and run separate WMMA
        maximum and exp/mass passes without a full logit matrix. A wave-reduced diagonal correction
        preserves each plane's causal zero-padded K tail. Static gfx1201 ISA contains native BF16
        WMMA and no private scratch spill.
  - [x] Retain direct ISA/resources and interleaved 8K/32K timings with separate ranker/consumer
        attribution. The schema-v1 runs are historical only: zero Q/K
        forced about 90% page retention, while missing binary/source identity and the inverted
        S16/S8 rank-time relationship make them invalid cross-profile evidence. Schema v3 uses the
        concentrated represented fixture and must retain the measured physical-oracle maximum
        absolute error and acceptance tolerances, executable/source hashes, and raw samples.
        Prior RTX 5090 B128/tensor-core results are direction evidence only: tau=.9 improved 32K
        prefill about 6.5--6.9% and 64K about 14.6--15.7%, with 32K NLL and 64K needle passing;
        they do not admit the distinct R9700 typed-cache implementation.
        Those S16/S8 reports are retained as pre-redesign history. The current schema-v4 G16/G32
        B16-query consumer reports bind the actual value group and passed at `1.6023e-8`/`1.6986e-8`
        maximum absolute error. Their structured fixture retained 14.06%/11.72% of pages and
        measured G16 `28.56x/32.40x` and G32 `28.81x/32.72x` dense-over-sparse at 8K/32K.
        Reports with executable/source hashes are retained under
        `profiles/bench/r9700-xattention-b16-requal-s16-tau900-g{16,32}/`.
        The stricter schema-v6 production-scale admission passed under `auto` at 8K/T=4,096 with
        four rotating address-distinct replicas, nonzero signed INT4 values, varying FP16 scales,
        and an all-element FP64 softmax/PV oracle. The current-tree refresh also binds the updated
        dense control and closes the former stale source-hash gap. At a 16.2502% keep fraction, G16
        measured `53.4330635 ms` total (`26.5418129 ms` rank, `26.3109531 ms` consumer) and G32
        measured `56.6925545 ms` (`26.3639278 ms`, `30.2011223 ms`); both production oracles
        reported exact zero relative-L2 and maximum-absolute error. The retained current-tree
        report SHA-256 values under
        `profiles/bench/r9700-xattention-current-tree-refresh-20260905/` are
        `205bac511ff61511a5e61233942beeead49829a417689a0ceaeafd4ef5427d11` (G16) and
        `ab5df29c798cdab9aae233da8433f84e2da8e36f8d3fde27a85f6e56d9bdcaaf` (G32).
        This closes the physical production-scale operator and source-matched ISA/resource subgate.
        Here the remaining "static profile" gate means choosing one compile-isolated G16/G32 plus
        dense/B128-S16-tau900 product profile, not rerunning operator admission. Operator reports are
        qualification-only and are not Pareto inputs. The current matrix owner emits schema v14, so
        every retained schema-v13 pre-redesign or pre-promotion matrix is structurally rejected;
        schema-v7 additionally requires the exact twelve recipe/group/profile candidates, candidate-
        local schema-v6 quality, one selected chunk, and matched artifact/executable identities.
        Recipe, chunk, selected-profile NIAH, terminal dispatch-to-ELF/resource binding, and fresh
        whole-model selection remain open.
  - [ ] Retain the completed matched model PPL/NLL/argmax input to static-profile selection, then run
        the long-context needle retrieval gate on the selected candidate before production admission.
        Dense remains the sole Engine route until every gate passes. The compile-isolated
        G16 S16/tau900 planner passed, and a fresh 128-token dense/XAttention diagnostic was
        byte-exact. A standalone 8K sparse cell completed in 50.47 seconds; comparing it numerically
        with the older 72.09-second dense sidecar gives +0.00048794 mean NLL, two new/six resolved
        severe positions, and 152 diagnostic greedy flips, but the old dense cell lacks the
        compile-bound profile and scorer-hash envelope and is not matched admission evidence. Fresh
        schema-v6 dense and XAttention campaigns are complete for all-Q4 G16/G32; their retained
        pass flags are relative to one shared BF16 realization and are not current quality admission. Their
        schema-v1 direct comparison is retained at
        `profiles/ppl/xattention-q4g64-route-comparison-20260903.json`. Only the long-context needle
        portion of this child remains open, downstream of static selection. Because the C=1..4 cap
        restores mixed-recipe eligibility, equivalent mixed G16/G32 dense and XAttention PPL gates
        are now required before the complete product-candidate selection.
        The required needle gate runs after the capacity/whole evidence produces the
        schema-v7 `terminal_production_selection`; it gates promotion of that selected artifact/cache/execution
        candidate and is not an input to the selection. Run one provenance-bound complete 64K
        start/q25/mid/q75/end ladder with the selected G16/G32 and dense/B128-S16-tau900 server:
        five exact-format, fresh-full-prefill requests using explicit model `qwen3.8-27b`, 64 output
        tokens, and `--no-prefix-reuse`. Retain it in the group/profile-qualified path from
        `tools/bench/README.md`. The
        8K/200K ladders and complete 6x5 matrix are optional broader/post-change coverage rather
        than assigned admission work.
        The exact required five-position 64K C1 fresh-prefill gate is prepared at
        `profiles/bench/post-terminal-niah-prepare-20260905`; it remains blocked on and dynamically
        bound to the schema-v7 winner, so no NIAH evidence is credited yet.
        - [x] Retain resolved C=1..4 capacity for compile-isolated ON-profile G16 and G32 before
              admission. The fresh schema-v20/schema-v13 campaigns resolve G16 to 262144, 524288,
              553280, and 541056 tokens and G32 to 262144, 524288, 575424, and 562688; C1/C2 are
              model-context-bound and C3/C4 device-memory-bound. They bind benchmark SHA-256
              `ec85dfe59aa9fc8ca369f4ffb785bd33fda5959d2ac5889010643a8a01e4e775` and
              `5780de5e08b5317c54e2212f4d3088dfeaad8bde290079c8adb42565b5e28e02`
              respectively plus all-Q4 artifact SHA-256
              `19d029a89c1ef1cf87420067555021a7c7b435c31a92bea7c64ccf42c03d80e9`.
              The exact 4,096-row/262,144-context planner workspace is 1,085,967,619 bytes versus
              603,619,587 bytes for dense. All values exactly reproduce the corresponding C1..4
              rows of the superseded C1..8 campaigns; those older manifests and the schema-v12
              `pareto-capacity-markerfree-layout-*` dense evidence remain historical only.
        - [ ] After selecting the chunk, let
              `profiles/bench/post-chunk-twelve-candidate-20260905` rerun both ON-profile capacity
              matrices with the rebuilt B16 consumer and then run each fresh schema-v14 whole
              matrix with the same group-specific executable and all-Q4 artifact bytes. Its four
              spec-none ordinary timing commands per matrix supply the ranking rows. Any separately
              retained MTP3 evidence is optional diagnostic exact-token/state/graph regression and
              does not enter this matrix, its prerequisites, or ranking. The in-progress/retained standalone phase directories are diagnostic
              history and are not selection inputs. Whole-profile preparation is optional
              attribution only after a complete unprofiled whole matrix.
        - [ ] Complete matching current dense G16/G32 capacity and whole pairs and include all four
              all-Q4 dense/sparse candidates in the same schema-v7 Pareto decision. Fresh dense
              schema-v20/schema-v13 capacity is complete: G16 resolves to 262144, 524288, 570304,
              and 558080 tokens and G32 to 262144, 524288, 593152, and 580416; C1/C2 are
              model-context-bound and C3/C4 device-memory-bound. The G16/G32 benchmark hashes are
              `78b4f3ba85436badaaccb4695848406d4d8f5d4f4f1439235d8d3200758490fb` and
              `5351c7c8fc4c7cd8fc136c2e6ab0107c65201d3fc0c4ab9fa4000f07e7587729`.
              Their whole matrices remain required, using the same executable and artifact bytes.
              The retained schema-v12/v19 capacity and phase evidence stays historical, and the
              dense schema-v6 PPL campaign already covers both groups with one-realization pass flags.
        - [ ] Complete the four restored mixed-recipe static candidates: run matched schema-v6 PPL for
              dense and B128/S16/tau900 G16/G32, then fresh exact C=1..4 capacity/whole pairs from
              their four compile-bound benchmark binaries. The prepared
              `terminal-quality-recovery-20260905` and `post-chunk-twelve-candidate-20260905`
              packages own those acquisitions. Assemble these with the four all-Q4 and four
              four-role candidates so weight recipe, cache group, and attention route are selected
              from one complete twelve-candidate objective set. The existing mixed dense quality cells are
              reusable only where the schema-v6 provenance/profile contract matches. Fresh mixed
              dense capacity is complete: G16 resolves to 262144, 314112, 301888, and 289664
              tokens, while G32 resolves to 262144, 326656, 313984, and 301248. C1 is
              model-context-bound and C2--C4 are device-memory-bound for both groups. The G16/G32
              manifests bind benchmark SHA-256
              `78b4f3ba85436badaaccb4695848406d4d8f5d4f4f1439235d8d3200758490fb` and
              `5351c7c8fc4c7cd8fc136c2e6ab0107c65201d3fc0c4ab9fa4000f07e7587729`,
              mixed artifact SHA-256
              `8fbadf14e355b1943ef9386a91ebafff0d852295a9adc0054bdd430291505ebd`,
              and manifest SHA-256
              `cf87ab0c3cfd2e574c0c114652658973b53785af8b34026a077ad8bf5cbbed58` and
              `93bcb29aaf2aceef9558cfd5221cb49b32688e691dd194c11f4bcf990a24995c`.
              Pre-rebuild mixed B128/S16/tau900 G16 capacity measured 262144, 297088,
              284800, and 272576 tokens; C1 is model-context-bound and C2--C4 are
              device-memory-bound. Its schema-v13 manifest binds benchmark SHA-256
              `ec85dfe59aa9fc8ca369f4ffb785bd33fda5959d2ac5889010643a8a01e4e775`,
              the same mixed artifact SHA-256, and manifest SHA-256
              `c4229e99a882c51158fe433a43cb44bc17b0c20c88cbaf352ecae178aec160b0`.
              Pre-rebuild mixed B128/S16/tau900 G32 capacity measured 262144, 308928,
              296192, and 283520 tokens with the same binding pattern. It binds benchmark SHA-256
              `5780de5e08b5317c54e2212f4d3088dfeaad8bde290079c8adb42565b5e28e02`,
              the same mixed artifact SHA-256, and manifest SHA-256
              `f12dfac84dc0d005e51e95142862dce1cc5abd020aeb7f98e1e354c76f937b03`.
              Both sparse manifests are retained historical measurements: their executable hashes
              predate the B16 consumer rebuild and they must be rerun at the selected prefill chunk.
              The fresh mixed dense schema-v6 campaign is also complete as fixed candidate-sidecar
              evidence at `profiles/ppl/xattention-dense-q4-w8-mse-20260903/results.json` (SHA-256
              `7530b27e503baf8bf29c2fd83135c0e2395a9a04bd050a2872dd2a7b341b9220`). It binds mixed artifact
              SHA-256 `8fbadf14e355b1943ef9386a91ebafff0d852295a9adc0054bdd430291505ebd`, dense G16/G32 scorer
              SHA-256 `fb36b2fca7871fbcbdbbce32cb1da8d05965e2164fea0d2773dd8259d686b6d4` and
              `552e73200a351691d5451adccfa7dd91e9381631388e366d3c03c051338a967d`, and imports the one retained
              BF16 realization from all-Q4 dense results SHA-256
              `68a6de3098c2726e2a0526226e63e50d0f52ed610afeab7256b05337410f1abf`. Relative to that realization,
              G16/G32 measure +0.011503/+0.012084 mean NLL with 4/3 new severe positions at 8K and
              +0.015520/+0.015216 with 14/14 at 32K. The schema pass flags are provisional and do not
              close BF16-derived admission; fresh mixed dense and sparse PPL acquisition remains open.
              All four mixed whole matrices remain pending; no old C1..8 capacity manifest is a
              current input.
    - [x] Prepare the provenance-bound paired PPL boundary: schema-v6 `tools/ppl/run.py` requires
          either compile-bound dense or `b128-s16-tau900` reports and records the selected profile
          with scorer/artifact/corpus/source and sidecar hashes. Exact fresh G16/G32 commands and
          output paths use the all-Q4 `capacity-speed`/`ln(1.05)` gate in `tools/ppl/README.md`;
          pre-profile-field dense sidecars are not reused. PPL acquisition now uses the same four
          current dense G16/G32 and B128/S16/tau900 G16/G32 selection build trees as capacity/whole
          evidence and rebuilds them immediately before acquisition. The runner records the
          resulting executable bytes; the ledger does
          not designate a fixed scorer hash because any linked core or unpromoted-challenger source
          change invalidates earlier executable provenance and requires a fresh rebuild/campaign.
          The strict schema-v1 `tools/ppl/compare_xattention.py` assembler requires two completed
          campaigns and directly retains paired route NLL, severe membership, greedy flips, and
          scorer timing without replacing the independent BF16 gates.
          The prepared BF16 v3 authority fixes FLA fused-recurrent exponent dispatch with
          `FLA_USE_FAST_OPS=0` and the complete canonical TunableOp/TF32/Triton/rocBLAS/ordinary-
          schedule/allocator environment before accelerator imports. It rejects architecture,
          Tensile-library, and custom-allocator overrides and binds actual gfx1201 plus resolved
          Triton/AMD lowering controls in ordinary and schema-v8 stage-trace provenance;
          conflicting or older unbound executions are rejected. Exact non-trace v3 8K/32K repeats
          now close the BF16 authority, and the dense all-Q4 campaign has been rebased against it;
          fresh sparse all-Q4 and mixed dense/sparse acquisition remains open. The retained schema-v1 comparison reports dense-over-
          XAttention scorer speedups of 1.44x at 8K and 2.39x at 32K, with diagnostic-only route
          deltas and flips. It is quality/timing input to the four-candidate decision, not the
          all-Q4 subcomparison owner. Equivalent mixed-recipe paired-route evidence remains required after
          the C=1..4 product-cap change.
  - [x] Add an OFF-by-default, compile-isolated target-private Text-prefill attention leaf and exact
        capacity API for the model gate. Only `Phase::Prefill` in `attn_mix` selects it; dense T=1
        remainders, decode, MTP/tree verification, and DFlash proposal/verification leaves stay
        dense; DFlash feature capture follows the selected ordinary Text-prefill route. The
        isolated PPL report binds `b128-s16-tau900` explicitly, and the campaign runner rejects a
        dense/sparse binary mismatch. Native benchmark schema-v20 reports and schema-v14 matrix
        manifests now carry and validate the same compile-bound profile for capacity, phase, and
        whole-inference evidence. Schema-v20 `server_start` carries the same compile-bound identity
        and cache group, and durable NIAH derives both expected values plus exact artifact identity
        from the schema-v7 `terminal_production_selection` before accepting matching executable/artifact bytes and
        fresh-full-prefill request evidence.
  - [ ] After model, whole-inference, and selected-profile NIAH admission, cut over the schema-v7
        `terminal_production_selection`. If B128/S16/tau900 wins, promote that Text-prefill leaf into the ordinary build; if
        dense wins, remove the qualification-only sparse branch. Dense remains the ordinary build
        until those gates pass. The sparse-winner cutover is one coherent contract change: delete the
        `NINFER_R9700_XATTENTION_QUALIFICATION` option and configurable stride/tau cache knobs;
        compile the selected B128/S16/tau900 Op and Text-prefill workspace/dispatch unconditionally;
        retain dense execution for T=1, decode, MTP/tree verification, DFlash proposal/verification,
        and every non-prefill call. DFlash feature-capture prefill uses the selected Text-prefill
        route. Replace qualification-named benchmark/request-log/PPL/matrix provenance with the one
        production attention-profile identity, bump the affected project-owned schemas together,
        and update their fail-closed validators and focused host tests. Remove the isolated-build
        commands only after the admission evidence and selected profile have been retained; do not
        carry an OFF branch or runtime selector into the production tree.

- [x] Add raw owned gfx1201 fragment-map qualification with sparse/nonuniform FP8 operands.
- [x] Implement vector QK baseline at D256, Hq24, Hkv4 with an independent score oracle.
- [x] Implement the raw FP8 WMMA QK candidate at D256, Hq24, Hkv4; qualify its E4M3-Q private
      profile directly against the independent FP8-Q score oracle and inspect its own ISA.
- [x] Implement FP32-probability INT4-V PV vector baseline and qualify every output against an
      independent dequantization oracle at D256/Hq24/Hkv4.
- [x] Reject WMMA PV for the FP32-probability, exact INT4-times-FP16-scale contract: a raw
      FP16xFP16-to-FP32 experiment passed its distinct oracle at page-boundary and 256-token
      corpora but necessarily rounded both operands, so it was removed rather than admitted.
- [x] Fuse append, paged QK, online FP32 softmax, and PV where whole-Op measurements justify it.
  - [x] Implement and qualify a score-streaming paged QK + online FP32 softmax + exact
        INT4×FP16-scale PV HIP baseline for all G16/G32 independent layouts; it preserves the
        FP32 PV contract without a context-sized score buffer. Append remains a separate A2 launch
        until a whole-Op fusion measurement justifies merging it.
  - [x] Implement and qualify raw-WMMA QK plus reusable FP32 scores, parallel FP32 softmax, and
        exact vector PV. It is 2.7--3.1x faster at T=1 over 1K--32K; repeat 4K timing favors it
        for T=1..2 and score streaming for T>=3. It is now compile-isolated from the represented-
        BF16-Q score-streaming control for direct PPL attribution.
  - [x] Promote only the physically winning ordinary-causal WMMA region into the target leaf and
        runtime workspace plan. Serialized boundary medians select T1 at context>=64 and T2 at
        context>=320 below the separately admitted split-512 boundary; other short-context shapes
        stream.
        The target leaf and separately compiled control both pass the independent represented-input
        FP64 oracle, fragmented committed cache, and same-stream pending device-table-row gates.
        At context 257, selected T1 is 0.104 ms versus 0.191 ms for streaming; T2 correctly stays
        streaming. Machine-readable classifier/timing/oracle evidence is retained in
        `profiles/bench/r9700-fp8-qk-wmma-crossover.json`.
  - [x] Treat the selected FP8-Q WMMA profile as qualified private precision rather than an
        observable semantic cast. A 63-position real decode comparison against the compile-isolated
        score-streaming control has two greedy flips, mean signed ΔNLL +0.004319, and maximum
        absolute ΔNLL 0.122795. Schema-v4 schedule comparison retains those flips diagnostically
        and requires a finite aligned NLL bound supplied from the complete 8K campaign; it does not
        guess a universal threshold from this short probe. Exact argmax remains required for
        same-route graph/eager, MTP/ordinary, and draft-window variants.
  - [x] Repair and qualify the runtime paths exposed by the isolated scorer: ordinary T=1 GDN now
        permits the intentional alias between its two read-only slot selectors while retaining
        disjointness from every mutable/data operand; MTP acceptance advances a dedicated
        fixed-address frontier vector instead of mutating the segmented transaction base. The GDN
        independent oracle passed, ordinary enabled/control scoring completed, and paired MTP3
        scoring is exactly identical across all 63 NLL and argmax positions. Cell JSON, raw
        sidecars, and the comparison record are retained in
        `profiles/ppl/fp8-qk-wmma-crossover/`.
  - [x] Run the complete independent-layout device qualification at D256/Hq24/Hkv4 and 4,096
        context tokens (one timing iteration): all G16/G32 × K/V/V-scale layout combinations
        passed A2 bytes, QK scores, FP32 PV, full attention, compact bytes, and error rejection.
        Timing is deliberately provisional until the coherent-driver repeat gate.
  - [x] Repeat the all-layout fused-attention gate at 8,192 and 32,768 tokens. Both contexts
        passed the independent FP64-attention oracle, exact three-plane state tests, and error
        paths; the long-context timing is a correctness baseline only, not a layout selection.
  - [x] Measure the production-shaped T-row A2 suffix plus selected A3 at decode T=1..8 and staged
        prefill T=9/17/128 over 1K/4K/8K/32K on the physical gfx1201, after direct codec/FP64-oracle
        gates. Three-repeat medians bound even an impossible free append to at most 0.80% at 1K,
        0.20% at 4K, 0.10% at 8K, and 0.03% at 32K for decode; T=128 bounds are 0.40%, 0.10%,
        0.05%, and 0.01%. Ordinary HIP provides no inter-workgroup barrier between the independent
        T×Hkv writers and T×Hq readers; a cooperative launch cannot admit the T=128 grid, while
        reader-local encoding duplicates work sixfold and evades single-writer/status ownership.
        Ordered-pair timings showed no repeatable gain beyond clock variance, so A2 remains a
        separate ordered launch and the already-fused QK + online FP32 softmax + exact PV A3 is
        the maximal admitted production operation. Direct metadata reports A2/A3 at 13/23 VGPR,
        zero scratch, occupancy 16, with zero/52-byte LDS respectively.
- [x] Benchmark G16/G32 and each plane order at T=1..8 and 1K/4K/8K/32K contexts.
  - [x] All 32 post-reboot workload points and all 16 layouts passed. The provisional timing leader
        is G16/token-K/feature-V/feature-scale at 1.004041 mean normalized latency, mean rank 2.188,
        and 14 wins; this is not the final ABI until paired quality and whole-inference gates run.
- [ ] Finalize the weight recipe, static cache layout, and Text-prefill execution identity from the
      same schema-v7 `terminal_production_selection` after repeatable latency, quality, capacity, phase, and
      whole-inference Pareto gates; do not make a second layout decision from the isolated Op sweep.
      The prepared publication owner is `profiles/bench/terminal-static-selection-20260905`; it
      remains blocked on the selected chunk, six quality authorities, and twelve physical
      capacity/whole pairs.
- [x] Qualify the independent sparse raw WMMA QK map and reject WMMA for PV because no gfx1201
      raw operand form preserves FP32 probabilities plus exact INT4-times-FP16-scale values; no
      rocWMMA/CK fragment layout enters production.
- [x] Add full Op oracle cases for empty/tail pages, causal length, ragged decode T=1..8 and staged
      prefill T=9/17/128 rows, GQA mapping,
      signed zero, FP8 saturation, INT4 ties, scale underflow/overflow, and invalid transactions.
  - [x] Extend the independent all-layout qualifier through single-token and 63/64/65-token causal
        tails, including exact append/attention/PV/compaction and invalid-map rejection. Empty
        inputs remain explicitly rejected by the raw Op contracts rather than silently interpreted.
  - [x] Extend fused A3 to T=1..8 independently represented BF16 query rows with per-row causal
        frontiers over one shared typed paged cache. All T values passed the independent FP64
        oracle across the complete G16/G32 × K/V/V-scale-layout matrix at a 65-token boundary
        corpus; zero/out-of-range device frontiers poison only their row. Exact A2 comparisons
        include signed zero, finite E4M3 saturation, INT4 RNE ties, canonical scale underflow,
        and scale-overflow rejection; every raw A2/A3 transaction rejects an empty invocation.
  - [x] Remove the target leaf's artificial T=8 ceiling and stage arbitrary positive raw U32 row
        extents across legal gfx1201 grid-Y launches without a score workspace or allocation. The
        fixed Qwen leaf admits T through its 262,144-token native capacity. Selected-layout raw and
        typed-leaf T=9/17/128 cases at a fragmented 257-token frontier passed the independent FP64
        oracle; the target leaf measured 0.204/0.442/2.504 ms over 50 ROCm-event iterations.
- [x] Profile and tune launch shape, vector widths, LDS staging, cache behavior, VGPR occupancy,
      page-table translation, softmax reduction, and wave32 versus wave64 only when a measurement
      changes the selected fixed implementation.
  - [x] Rebuild and profile the post-parallel-softmax WMMA decode candidate on physical gfx1201.
        Full oracles passed at 65-token T8, 4K T1/2/3, and 32K T1; a 50-event 4K sweep retains the
        provisional WMMA T1..2/score-streaming T>=3 split, with T3 tied within measurement noise.
        The focused trace attributes only 7.7 us to parallel softmax versus 818.8 us to exact PV.
        LLVM metadata reports QK/softmax/PV at 24/23/15 VGPR, 0/76/0 bytes LDS, zero scratch, and
        occupancy 16, with native FP8 WMMA present. No softmax/launch edit is measurement-justified;
        the finite context/row classifier above owns production admission while the compile-
        isolated score-streaming profile owns subsequent model-level attribution.
  - [x] Close the selected score-streaming A3 launch/profile decision without speculative tuning:
        gfx1201 exposes wave32, D256 maps exactly to eight waves and an eight-partial LDS merge,
        and the selected 256-thread kernel remains at 23 VGPR, 52-byte LDS, zero scratch, occupancy
        16. The complete layout/context/T event sweep, post-softmax WMMA trace, and production-shaped
        A2+A3 repeat gate changed no selection. Installed VALU/LDS counters remain unavailable;
        dispatch-scoped TCP/GL2C hit ratios became usable later, but do not retroactively provide
        absolute traffic or stall evidence for this decision. A future static-layout selection must
        reopen profiling for that newly selected layout rather than perturbing this route family in
        advance.

## Remaining Engine and artifact work

- [ ] Select the one measured integer weight recipe, promote the converter authority to its final
      identity, and materialize the fresh final artifact.
      Selection is owned by the prepared schema-v7 terminal package, after which
      `post-terminal-focused-verification-20260905`, `post-terminal-niah-prepare-20260905`, and
      `selected-dflash-prepare-20260905` must pass for that same winner before the cutover plan may
      rename or materialize anything. The machine-joined owner at
      `profiles/bench/final-artifact-cutover-admission-prepare-20260905` is implemented and
      closure-bound, but its final admission JSON remains absent until all of those physical gates,
      selected exact-token, low-context, and hardware-use authorities pass. Preparation
      does not authorize conversion or promotion.
  - [x] Complete the BF16-source R9700 evaluation-converter construction and retained real-artifact
        conversion evidence needed to reach the measured selection gate.
  - [x] Add a provisional direct-BF16-source W8G32 candidate converter with exact 1,118-tensor
        inventory, deterministic scalar packing oracle, preflight-before-output, and synthetic
        byte tests. It deliberately cannot become the product recipe without real-model gates.
  - [x] Add two separate unregistered low-bit evaluation converters without changing all-W8 or
        C++ bindings: the 439-Q4G64 `r9700-q4g64-eval` capacity floor is 15,159,801,760 tensor
        bytes (14.12 GiB), while `r9700-q4-w8-eval` retains 183 source-Q4 roles and promotes the
        other 256 matrices to W8G32 for 22,868,177,312 bytes (21.30 GiB). Dependency-light exact
        1,124-object plan tests and complete 18-shard BF16/frontend/ranking preflights pass for
        both. Q5/Q6 remain fallback measurements rather than primary candidates; the existing
        439-W8 candidate remains the accuracy ceiling. No identity is selected by this evidence.
  - [x] Replace the retired-model draft-ranking dependency with an explicit Qwen3.8 provenance
        tool. The builder requires the schema-2 Qwen3.8 PPL-corpus identity with raw/no-special
        tokenization, validates decimal tokenizer IDs, positive token count, domain, and payload
        hash against each sibling manifest, rejects tiled throughput corpora, and emits
        exactly one little-endian I64 frequency row plus a JSON sidecar. Converter preflight
        revalidates and recounts every named corpus before byte-comparing the row. The current
        32,768-token untiled PPL corpus produced a 1,986,560-byte row with 3,933 distinct IDs and
        SHA-256 `205b2d6c5b58d946c87b425578da13b5cfc32a0abe1ff201e95b47d032bf004e`;
        the complete 18-shard BF16 source preflight accepts its 131,072-entry shortlist.
  - [x] Qualify the raw signed-INT4 gfx1201 primitive needed to evaluate a Q4G64 challenger without
        selecting it. Literal wave32 fragment maps for native dense `16x16x16` and `16x16x32`
        signed IU4 pass all 256 I32 outputs exactly against sparse asymmetric host GEMM oracles.
        LLVM emits both native opcodes with 35/55 VGPR, zero LDS/scratch, and occupancy 16. Across
        shared-packed-input K64/256/1024/5120 tile loops, WMMA32 is 7.25x/6.25x/11.81x/13.00x
        faster than the straightforward packed-INT4 vector dot. This primitive alone made no
        recipe decision; the complete candidate below owns the activation and scale boundaries.
  - [x] Qualify a production-shaped, dispatch-isolated Q4G64 challenger. Its allocation-free raw
        boundary quantizes represented BF16 to caller-owned signed A4G64/FP16-scale planes with
        finite/RNE semantics, K128 row padding, exact partial-G64 and K192-to-K256 tail images,
        device status, and alias/extent rejection. Native signed `16x16x32` WMMA combines each
        A/W G64 I32 dot with FP16 scales in FP32 and rounds output once to BF16. Every output at
        `[7168,5120]` T=1..8/16/32/64/128 matches the independent FP64 A4-times-Q4 formula exactly
        at the BF16 boundary; the independent represented-BF16-times-decoded-Q4 comparison has
        maximum error 0.256836 within its computed 0.649838 decoded-weight quantization bound.
        Complete unprofiled median latency is 0.06025--0.08957 ms for decode and
        0.11118/0.12674/0.15120/0.28566 ms for prefill, versus current W8 at
        0.12410--0.59626 and 1.15381/2.22700/4.35739/8.48574 ms. ISA reports native signed IU4,
        55 VGPR, zero LDS/scratch, and occupancy 16. This admits Q4G64 to real-model recipe gates;
        production Tensor dispatch remains W8 until quality and whole-inference evidence.
  - [x] Integrate the qualified Q4G64/A4G64 route as two explicit real-runtime evaluation
        identities without selecting it: `r9700-q4g64-eval` binds all 439 matrices as Q4G64 and
        `r9700-q4-w8-eval` binds the 183 source-Q4 roles as Q4G64 plus 256 W8G32 matrices. Text,
        MTP, PPL scoring, DFlash-head, and Vision call sites use caller-owned, profile-sized A4G64
        workspace; persistent codes/scales are consumed directly with no hidden allocation or
        runtime repack. The all-Q4 token table uses a direct signed-Q4/FP16-scale gather with an
        independent exact-code fixture and malformed-plane/alias rejection. Sparse
        metadata-complete binder fixtures protect both format maps while the all-W8 identity
        retains its existing W8 dispatch. At integration time the real BF16-source PPL,
        quality, capacity, and whole-inference gates were outstanding; the completed 8K comparison is
        recorded below and still makes no selection.
  - [x] Qualify the higher-precision activation route for persistent Q4G64 weights without
        forfeiting gfx1201 INT4 execution. Encode each canonical signed A8G64 activation into an
        unsigned low-nibble plane and signed high-nibble plane, evaluate both with native IU4
        wave32 WMMA, combine `low + 16 * high` exactly in I32, compose FP16 group scales in FP32,
        and round the output once to BF16. The compile-time evaluation profile must report its
        activation precision explicitly and leave the existing A4 evidence reproducible; it must
        not add a runtime recipe selector or duplicate byte-identical weight artifacts. Admit the
        route only after independent codec/formula oracles at real Text shapes, malformed-workspace
        and nonfinite checks, ISA/resource evidence, and interleaved complete-Op timing against
        A4G64 and W8 baselines.
        The separately compiled A8 evaluator retains all three existing Q4 artifact identities
        and reports `q4_activation_bits=8`; A8 is now the production-style compile default and A4
        requires an explicit evaluator build that reports 4. Exact
        activation images, codec edge/RNE/FP16-floor/nonfinite/overflow, K192-to-K256 tail,
        malformed extent, and complete-candidate alias checks pass. Quantizer status is consumed
        asynchronously by both A4 and A8 WMMA routes and maps any failure to exact BF16 NaN output
        instead of silently consuming zero codes. The real-shape fixture rotates four active lanes
        per G64 group and covers all 64 packed K-lane/WMMA positions. Every output of the
        independent A8-times-decoded-Q4 FP64 formula passes with zero BF16 steps at
        `[7168,5120]`, T=1..8/16/32/64/128; represented-source maximum error is 0.015625 versus
        A4's 0.212891. Complete A8 medians are 0.05874--0.08699 ms decode and
        0.11070/0.12538/0.15366/0.30146 ms prefill, 8.7--24.3 percent slower than A4 but
        faster than current W8 in this isolated shape test. ISA is native low U4xI4 plus high
        I4xI4 `v_wmma_i32_16x16x32_iu4`; A8 quantize/WMMA use 10/64 VGPR, zero LDS/scratch, and
        occupancy 16. Timing uses five alternating forward/reverse route rounds, retains
        each route sample, and reports medians. Separately compiled A4/A8 qualifiers also exercise
        the public Tensor/WorkspaceArena selection, exact workspace allocation, and rewind. The
        complete hardware/toolchain/result records are retained under `profiles/bench/`. The
        production-style A8 and explicit A4 Engine planners additionally pass C=1..4 ordinary,
        Vision, MTP, and DFlash schedule construction while preserving
        W8<mixed-Q4/W8<all-Q4 workspace ordering; the combined record is
        `profiles/bench/r9700-q4-activation-runtime-planner.json`.
        The selected wave32 quantizer additionally passes the exact activation codec and sampled
        independent FP64 formula with zero BF16 steps at all 17 distinct all-Q4 shapes, the six
        mixed-artifact Q4 shapes, and the five additional DFlash shapes over T=1..8/32/128
        (22 unique shapes and 220 extents). The refreshed seven-trial interleaved timing shows
        near-ties at isolated points and up to 2.034% count-weighted gain for all-Q4 and 1.436% for
        DFlash. The complete inventory, samples, and medians are retained in
        `profiles/bench/r9700-a8q4g64-artifact-shape-sweep.json`.
  - [x] Isolate Q4 activation error from Q4 weight error with matched real 8K runs of the all-Q4
        and source-role Q4/W8 artifacts under the qualified A8 profile. Compare complete aligned
        NLL/argmax sidecars against their A4 executions and BF16, including terrible-position
        membership. If higher-precision activations do not meet the existing numerical gate, keep
        coherent important role families at W8 rather than adding Q4-to-FP8/BF16 expansion paths.
        A8 reduces all-Q4 PPL from 7.397805 to 6.720524 and mixed Q4/W8 from 6.765289 to
        6.550048. It repairs 475/258 A4 greedy failures while introducing 125/75 new ones.
        The mixed candidate meets the accuracy tier at +0.013815 mean NLL and two new
        NLL-at-least-10 positions; it differs at 237 BF16-greedy diagnostic positions. Complete JSON,
        FP32-NLL, and I32-argmax sidecars for both activation widths remain under
        `profiles/ppl/`; no Q4-to-floating expansion path was added.
  - [x] Add a same-format source-MSE-refined Q4/W8 control before changing role allocation. Apply
        the canonical-baseline-inclusive alternating least-squares scale objective independently
        to Q4G64 and W8G32 groups, preserve the existing 183-Q4/256-W8 inventory and byte layout,
        and give it a distinct evaluation identity and complete conversion provenance. Use no
        activation corpus, token labels, PPL traces, or per-layer decisions in this control.
        The 22,881,204,736-byte real artifact has exactly 183 Q4 and 256 W8 matrices and no runtime
        repack. Its A8-Q4/represented-BF16-W8 8K control is PPL 6.544746, +0.013005 mean NLL, 234 BF16-greedy flips, and
        three new severe positions. Against non-MSE A8 it improves mean NLL by 0.000810 and PPL by
        0.005302 while changing 196 greedy positions. Conversion metadata, artifact validation,
        smoke output, and complete 8K sidecars are retained.
  - [x] If the A8 plus source-MSE Q4/W8 control leaves enough quality margin, evaluate a locked,
        nested all-layer role ladder that moves only complete source-Q5 families to native Q4 in
        independently justified error-per-byte order: Text MLP down, then GDN value/output, then
        full-attention gate-value/output. Stop at the first predeclared numerical-gate failure;
        never select individual layers from the final PPL corpus. Production selection still
        requires the chosen rung's 32K, decode, graph/eager, speculative, capacity, and C=1..4
        whole-inference gates. Under the current position-aware budget, the starting 183-Q4 rung is
        quality-eligible at 8K; no further precision demotion is selected without matched 32K and
        whole-inference Pareto evidence.
  - [x] Use measured 8K evidence to isolate the strongest next feasible precision promotion
        without selecting a recipe. All-Q4 is rejected at +0.1355 mean NLL / 800 BF16-greedy
        flips; all-W8 is +0.002053 / 90, and a source-BF16 output head regresses to +0.002425 / 92.
        The rejected head artifact and sidecars remain evidence, while its converter/profile was
        removed. The source inventory ranks only token embedding and output head at Q6, internal
        gate/value/output/down roles at Q5, and query/key/gate-up roles at Q4. With the head
        eliminated, `r9700-w8-bf16-embed-eval` uses the remaining Q6 role: 438 matrices stay W8G32
        and only `text/token_embedding` is source BF16 through the existing direct gather. Its exact
        tensor/device/projected artifact sizes are 31,452,349,792 / 31,452,361,984 /
        31,465,375,488 bytes, a 1,191,936,000-byte promotion. Promoting both vocabulary matrices
        would leave only 641,698,560 bytes after default headroom, below the 838,860,800-byte main
        32K G32 cache alone. Exact inventory and sparse full-binder tests pass. Its matched real 8K
        G16 result is +0.001624 mean NLL with 80 BF16-greedy flips: the fewest flips of the tested
        candidates; flips are diagnostic. Resolved 32K capacity and
        whole-inference comparison remain selection-dependent.
  - [x] Prepare the coherent Q5 fallback without disturbing the active BF16-embedding gate.
        `r9700-w8-bf16-attn-vo-eval` restores source BF16 for the fused gate/value and attention
        output projections across all 16 full-attention layers—32 matrices total—while the other
        407 matrices remain W8G32. It adds exactly 1,022,361,600 bytes over all-W8; exact
        tensor/device/projected artifact sizes are 31,282,775,392 / 31,282,787,584 /
        31,295,801,088 bytes. Its binder routes only those 32 weights through the existing BF16
        Linear path with zero hidden allocation, repacking, or activation-quantization workspace.
        Exact inventory and metadata-complete sparse binding tests pass. Its matched real 8K G16
        result is +0.001098 mean NLL with 84 BF16-greedy flips: the lowest mean-NLL penalty of the
        tested candidates; flips are diagnostic. Capacity and performance remain
        selection-dependent and no recipe is selected.
  - [x] Prepare the coherent attributed GDN fallback while preserving the embedding and
        attention-VO profiles. `r9700-w8-bf16-gdn-qk-eval` restores source BF16 for
        `gdn/query_key` in all 48 GDN layers and leaves the other 391 matrices W8G32. It adds
        exactly 943,718,400 bytes over all-W8; tensor/device/projected artifact sizes are
        31,204,132,192 / 31,204,144,384 / 31,217,157,888 bytes, leaving 2,081,852,160 bytes after
        default 1 GiB sizing headroom. The explicit binder routes only those 48 matrices through
        the existing BF16 Linear path with zero repacking, hidden allocation, or quantization
        workspace. Exact inventory and metadata-complete sparse tests pass. Its matched real 8K
        G16 result is +0.001375 mean NLL with 98 diagnostic BF16-greedy flips and is
        quality-eligible. Capacity and performance remain
        selection-dependent without a selection.
  - [x] Add the omitted capacity-feasible full-attention query/key evaluator without splitting the
        already measured value/output family. `r9700-w8-bf16-attn-qk-eval` restores source BF16
        for the fused query/key projection in all 16 full-attention layers and leaves the other
        423 matrices W8G32. It adds exactly 550,502,400 bytes over all-W8; tensor/device/projected
        artifact sizes are 30,810,916,192 / 30,810,928,384 / 30,823,941,888 bytes, leaving
        2,475,068,160 bytes after default 1 GiB headroom. The explicit binder uses the existing
        BF16 Linear path without repacking, hidden allocation, or quantization workspace. Exact
        inventory and metadata-complete sparse binder tests pass. Its matched real 8K G16 result is
        +0.001996 mean NLL with 81 diagnostic BF16-greedy flips and is quality-eligible. Capacity
        and performance remain selection-dependent without a selection.
  - [x] Complete the matched real 8K G16 prefill gate for the coherent resident candidates. The
        BF16 source scores mean NLL 1.865657. Relative to it, all-Q4 is +0.135526 / 800 flips,
        source-Q4/W8 is +0.046148 / 420, all-W8 is +0.002053 / 90, BF16 output head is
        +0.002425 / 92, BF16 token embedding is +0.001624 / 80, BF16 attention value/output is
        +0.001098 / 84, BF16 GDN query/key is +0.001375 / 98, and BF16 attention query/key is
        +0.001996 / 81. The all-W8 G32 control is +0.001909 / 98. The same-size source-MSE-refined
        all-W8 G16 evaluator is the mean-NLL leader at +0.000212 / 88 flips, but not the flip-count
        leader. Every result has 4,095 aligned finite NLL and argmax entries. Under the current
        numerical contract, all-W8 and coherent BF16 promotions are quality-eligible, while A4 Q4
        profiles are not. Greedy mismatch counts remain diagnostic and visibly non-monotonic, so
        selecting individual tensors/layers against this same corpus would still be arbitrary
        tuning. No integer recipe or cache group/layout is selected until matched 32K quality,
        resolved capacity, and whole-inference Pareto evidence is complete; parent items remain
        open and no evidence profile is deleted.
  - [x] Add a same-size source-only calibrated W8G32 evaluator. The explicit
        `r9700-w8g32-mse-eval` identity preserves the 439-W8G32 plan, 30,260,413,792 tensor bytes,
        30,260,425,984-byte arena, existing binder profile, and runtime dispatch. Each represented
        BF16 G32 group chooses the earliest minimum decoded-weight SSE scale from the canonical
        absmax baseline and an eight-step deterministic RNE-code/least-squares trajectory; it uses
        no calibration text, PPL/argmax sidecars, evaluation labels, or GPU results. Full
        representative Text tensors improve source decoded-weight SSE by 4.94--5.33%, Vision QKV
        by 2.65%, and deterministic vocabulary samples by 5.16--5.33%. Independent scalar codec,
        vectorized codec, exact inventory, metadata-complete sparse binder, and non-GPU build tests
        pass. Its completed matched real 8K G16 trace has mean NLL 1.865868912, delta +0.000211577
        versus BF16, PPL 6.461547962, 88 BF16-greedy flips, no new terrible position, and two
        repaired terrible positions. Relative to canonical all-W8 G16 it repairs 34 flips, creates
        32 new flips, and improves mean NLL by 0.001841799 at identical artifact size. It is retained
        as a quality-eligible mean-NLL evaluator; it does not select the recipe or close the parent
        item without complete Pareto evidence.
  - [x] Add and execute the same-format calibrated mixed control.
        `r9700-q4-w8-mse-eval` preserves the exact 183-Q4G64/256-W8G32 assignment, 1,124-object
        order, 22,868,177,312 tensor bytes, 22,868,191,232-byte arena, and existing Q4/W8 binder
        and runtime dispatch. Both integer formats select the earliest minimum decoded-weight SSE
        scale from their canonical FP16 absmax baseline plus eight deterministic source-only
        alternating code/least-squares steps; there is no activation, draft-ranking/corpus-token,
        PPL/argmax, or GPU-measurement input to that objective. The Q4 scalar/vector codec passes
        canonical packing and 35
        fixed/random refined parity cases, and a complete source `[7168,5120]` attention query/key
        matrix reduces decoded Q4 SSE by 10.9413% without changing stored format or layout. The
        converter has full source/frontend/draft-ranking preflight and durable conversion JSON;
        dependency-light plan/dispatch/report tests and the metadata-complete C++ binder fixture
        pass. The complete 22,881,204,736-byte real artifact was converted atomically and passes
        exact CPU inventory plus the real C++ binder. Its matched A8-Q4/represented-BF16-W8 8K control is PPL 6.544746,
        +0.013005 mean NLL, 234 BF16-greedy flips, and three new severe positions. All conversion,
        artifact, JSON, FP32-NLL, I32-argmax, and checksum provenance is retained. The current
        five-position severe budget is satisfied, but complete Pareto evidence is still required
        before selection.
  - [x] Qualify and integrate a production-shaped W8G32 plus dynamic signed-A8G32 evaluator.
        Caller-owned code/FP16-scale/status workspace matches the device codec
        byte-for-byte to an independent host RNE implementation. Two native signed INT8 WMMA
        instructions form each G32 INT32 dot before FP32 scale composition; represented-formula
        FP64 checks pass at `[7168,5120]` T1--8/16/32/64/128 with zero sampled BF16 error. A
        physical sweep over all 13 mixed-artifact W8 shapes retains exact execution below each
        conservative crossover and uses A8 at T3/T4 for nine shapes, T32 for the MTP MLP, and T64
        for the three 1152-row Vision shapes. The route is compile-selected and reported, uses the
        production Tensor/WorkspaceArena path, and leaves persistent identity/bytes unchanged.
        LLVM reports 9 VGPR for quantization and 83 VGPR for WMMA, zero LDS/scratch. The subsequent
        matched real-model A/B selected adaptive A8 as the default W8 execution profile at PPL
        6.538677, +0.012077 mean NLL, and three new severe positions; the A16 build remains the
        explicit control.
- [x] Port eager exact transforms, norms, activations, embedding, sampling, and BF16 reference
      linear paths.
  - [x] Qualify owned gfx1201 eager position/I32 state movement, BF16 scatter/gather,
        FP32-to-BF16 RNE and BF16-to-FP32 casts, BF16 residual add, and deterministic BF16
        argmax at Qwen D5120/vocabulary shapes for ragged T=1..8. The stochastic route is
        qualified below; norms, activations, embedding, and linear math were separate work.
  - [x] Qualify FP32-reduction RMSNorm/LayerNorm/L2Norm, exact/tanh GELU, SiLU/sigmoid gates,
        dense BF16 embedding gather, and no-randomness greedy selection at D5120, vocabulary
        248320, and ragged T=1..8 against independent host FP64 or exact-bit oracles. The
        stochastic route is qualified below; GDN and linear paths were separate work.
  - [x] Promote the exact K6144 gated-RMSNorm token8 prefill route. Its immutable physical report
        `profiles/bench/r9700-gated-rmsnorm-k6144-token8-ab-20260904-r3.json` (SHA-256
        `31c50ee9f42edd61dae21266b64b099af0e5b3a94051eecb4f55fa783f5ed77f`) passed incumbent
        BF16-bit-exact and independent FP64 formula checks, improved P2048 from 1.150442 to
        0.155001 ms (7.42215872x), and had no T>=128 regression. The measured T64 point also won
        by 1.78739786x, so production selects the zero-workspace token8 route for K6144/T>=64 and
        retains the incumbent for smaller rows and all other feature widths. The gfx1201 static
        gate reports 22 VGPR, occupancy 16, four 128-bit loads, and zero LDS/private/scratch.
  - [x] Promote the exact Text-MLP split-view SiLU-multiply 2D route. Its immutable physical report
        `profiles/bench/r9700-silu-mul-split17408-2d-ab-20260904.json` (SHA-256
        `a561f8b7981b836b2a5a582d093d67d087f2841efede95791aa5969a13b7bc10`) passed generic
        BF16-bit-exact and independent FP64 formula checks and won every measured T128/512/2048/4096
        row; P2048 improved from 1.00915897 to 0.374680012 ms (2.69338894x). Production selects
        the grid-X-feature/grid-Y-token route only for `[17408,T]` split views with exact element
        strides `[1,34816,34816*T,34816*T]` at T>=128, preserving the generic route for smaller
        rows and all other layouts. The static gate reports 9 VGPR, occupancy 16, and zero
        LDS/private/scratch.
  - [x] Bind the qualified eager kernels to the repository-internal Tensor/Weight Op ABI with
        native HIP streams, including scalar state, structured scatter/gather/extraction, fused
        residual RMSNorm, gated RMSNorm, L2 diagnostic intermediates, and Q6G64/W8G32 embedding
        decode. A clean gfx1201 build and physical public-ABI qualifier pass the real D5120 and
        248320-vocabulary corpus plus exact structured-state and integer-codec oracles. The W8
        embedding case owns the complete `[248320,5120]` code/scale extents, samples IDs through
        248319, compares every selected element bit-exactly with an independent signed-code times
        stored-FP16-scale oracle, and rejects short payload and wrong-padding metadata; the
        superseded CUDA wrappers, launchers, and kernels for this bounded Op set are removed.
        The corresponding CUDA-only pointwise/norm/movement/embedding/argmax microbenchmarks and
        obsolete composite GDN-layer benchmark are also removed: they forced deleted NVIDIA
        candidates and supplied no gfx1201 route decision beyond the active eager/GDN qualifiers.
  - [x] Port the public `SamplingConfig`/`sample` stream boundary to native HIP and qualify the
        positive-temperature two-stage gfx1201 route at the full padded 248320-token vocabulary
        for B=1..4. Independent host ordering and FP64 probability oracles cover exact BF16 ties,
        presence/frequency penalties, top-k clamp cases, top-p/min-p truncation, counter keys over
        seed/position/purpose, null and distinct count arrays, and mixed greedy/stochastic rows;
        every selected token and the complete count-array publication image match exactly. The
        retained pre-cap wave-local/hierarchical B=8 top-20 route measured 0.58 ms over 50
        unprofiled events with 620800 bytes of caller-owned scratch (the exact sorting-network
        baseline was 0.67--0.74 ms); that timing is historical, not supported-product evidence.
        LLVM metadata reports partial/final kernels at 20/16 VGPR, 1440/304 bytes
        LDS, no private scratch, wave32, and occupancy 16.
- [x] Port optimized decode and prefill linear, GDN, MTP, DFlash2, and required vision paths.
  - [x] Replace the public MTP next-round transition with one native HIP/gfx1201 implementation in
        the closed production archive. A physical exact-integer qualifier covers every K=1..5 and
        B=1..4 product shape, zero/full/mixed acceptance, budget/context exhaustion, AR row pitch
        and untouched padding, all output planes, input immutability, and malformed public shapes;
        the superseded CUDA wrapper, launcher, kernel, and duplicate CUDA test are deleted.
  - [x] Replace the complete public speculative-round family with one native HIP/gfx1201 owner:
        verify input/id preparation, chain Leviathan acceptance with optional selector-q residual,
        packed-tree SpecInfer membership, accepted-hidden selection, and proposal-id remap. Exact
        independent oracles exhaust K=1..8, W=2..16, and B=1..4, exercise mixed greedy/stochastic
        rows, penalties, filtering, counter RNG, count publication, immutable inputs, and fixed-
        address graph replay at the 248320-token product vocabulary. The caller-owned two-stage
        retained pre-cap distribution route measured 2.01 ms for chain K=8/B=8 and 0.59 ms for
        mixed product-tree W=12/B=8 over 20 physical R9700 events; those timings are historical,
        not supported-product evidence. No hidden allocation or host readback remains, and
        the superseded CUDA owner/test is deleted.
  - [x] Qualify standalone gfx1201 BF16 and provisional W8G32 decode-linear routes at the real
        Qwen3.8 full-attention query/key shape `[7168,5120]`. Both decode exactly from their
        represented storage boundaries, accumulate FP32, and pass independent FP64 oracles at
        synthetic T=3 and real T=1/T=8. A staged-LDS wave32 cooperative candidate is selected
        only for the measured candidate-width ranges: BF16 T=4..8 and W8G32 T=5..8; the baseline
        remains selected for narrower widths. This is Op bring-up evidence, not a final weight
        recipe or complete-model performance claim.
  - [x] Qualify standalone gfx1201 prefill-linear routes at `[N,K]=[7168,5120]` and fixed
        T=16/32/64/128. BF16 selects native wave32 BF16×BF16→FP32 WMMA after an independent
        full FP64 real-T16 gate plus spread direct-FP64 real-matrix checks at every wider width;
        all observed BF16 errors are zero. The 50-event timings are 0.203/0.267/0.698/2.418 ms.
        W8G32 selects a 16-wave 1-KiB-LDS exact-code/FP16-scale FP32-FMA route because no WMMA
        form preserves that codec; it passes the same direct oracle structure and measures
        1.119/2.225/4.371/8.751 ms. This does not select a final weight recipe or integrate an
        Engine path.
  - [x] Replace the public Linear contract with the sole native HIP BF16/W8G32 route and bind it
        directly to canonical Tensor/Weight views. The W8 raw ABI now receives exact independent
        code/FP16-scale planes, so target row views require no payload repack; byte extents,
        physical geometry, aliases, and unsupported formats reject before launch. Decode T=1..8
        and arbitrary prefill T>=9 share the selected gfx1201 dispatch. The physical public-ABI
        qualifier passes direct FP64 oracles for BF16/W8, row-sliced W8 storage, short-plane
        rejection, real `[7168,5120]` T=1/8/16/32/64/128 cases, and non-multiple-of-16 T=9/15/17
        tails. Memory-bounded sparse represented-weight cases additionally own the complete
        `[248320,5120]` output-head and `[131072,5120]` draft-head extents at T=1/3, compare every
        BF16 output bit with the direct FP64 formula, reject short code/scale and wrong-padding
        metadata, and compose the full output head with exact valid-vocabulary argmax at a one-BF16-
        ULP winner margin.
  - [x] Qualify standalone gfx1201 correctness-first Gated DeltaNet semantic Ops at real Qwen
        C=10240, Hq=16, Hv=48, and K=V=128 shapes: width-four causal BF16 convolution/SiLU,
        FP32 decay/correction controls, FP32 persistent recurrence, BF16 readout, and exact FP32
        checkpoint copy. Independent FP64 and FP32 recurrences cover T=1..8, zero/intermediate/
        full accepted-prefix transactions, immutable verify checkpoints, and exact same-path
        fold clones. Projection fusion remains separate work.
  - [x] Replace the correctness-first ordinary Gated DeltaNet recurrence with the measured
        arbitrary-width gfx1201 prefill route. Four 32-row/256-thread CTAs per value head expose
        enough work for all 64 CUs; one wave loads and normalizes Q/K, uniform controls and values
        are broadcast, and per-token workgroup synchronization falls from 19 points to 3. The
        public represented-input FP64 oracle passes normalized T=128 and non-normalized T=64 for
        distinct and in-place state, while snapshot/replay W<=16 remains unchanged and qualified
        (max BF16-output abs 7.63e-6, rel-L2 0.001798, FP32-state abs 8.42e-9). Seven-run physical
        medians at T=16/32/64/128 are 0.033244/0.054612/0.099263/0.188758 ms, respectively
        21.37/27.02/28.47/28.97 percent below the original route; two/eight row tiles and the
        grouped-head, 16-lane, and split-accumulator challengers lost and are deleted. LLVM reports
        56 VGPR, 1040 bytes LDS, zero private scratch, wave32, and occupancy 16.
  - [x] Integrate the provisional W8G32 target Variant's real C10240 GDN projection/convolution
        boundary for fixed B=1..4 verification: direct no-repack row-split projections, mixed-width
        snapshot publication, immutable replay records, and tree-parent histories now use native
        HIP with caller-owned workspace. The physical raw qualifier corrected feature-fastest
        `[channels,4]` weight and `[channels,3,slots]` state indexing, then passed ordinary T=1..8,
        mixed-width B=3 snapshot/tree cases, exact BF16 records/state, and independent FP64
        convolution/recurrent oracles. A target-level sparse-W8 composition gate passes B=2/W=4
        snapshot and tree record with exact workspace accounting. Broader recurrent schedule,
        fusion, MTP, DFlash2, vision, and final recipe selection remain under the open parent.
  - [x] Port the DFlash2 selector boundary to native HIP over the already-bound BF16 predecessor
        and successor codebooks, with BF16/W8G32 hidden projection delegated to the canonical
        R9700 Linear Op and no NVFP4 codebook field, runtime repack, or CUDA launcher. The physical
        gfx1201 qualifier uses all 248320 codebook rows at T=2/B=2 and passes exact greedy support,
        probability, identity-token, mapped draft-head token, and packed-tree topology oracles.
        The complete DFlash proposal/verification schedule remains open because its broader family
        schedule still imports CUDA-only generic attention and other unported Ops.
  - [x] Replace the runtime scalar-schedule and Vision Op dependencies exposed by exact runtime
        linking. Native add-bias, NLL, masked-block, ragged-prefix, Vision attention, and Vision
        positional embedding directly own HIP public contracts; physical qualifiers cover exact
        transforms, full represented-input FP64 formulas, guarded bounds, immutable inputs,
        packed/uniform Vision sequences, interleaved isolation, and real product dimensions. Their
        obsolete CUDA wrapper/launcher/kernel owners are deleted.
  - [x] Replace the public DFlash Full bidirectional-attention boundary with one native gfx1201
        wave32 BF16 route for D128/Hq32/Hkv8 and T=1..16/B=1..4. The Op owns a cache-specific BF16
        paged view rather than reviving the removed Text/MTP cache view, performs online-softmax
        with no hidden allocation or transient workspace, and passes an independent FP64 physical
        qualifier at chain W=5 and tree W=12 over mixed B=2 context/valid widths and fragmented
        table rows (110592 active outputs, max absolute error 0.000977; inactive tails exact zero).
        LLVM reports 31 VGPR, 42 SGPR, wave32, zero LDS/private scratch, and no spills.
  - [x] Replace DFlash2 SWA and grouped dynamic convolution with native gfx1201 Ops. SWA uses a
        measured finite wave32 direct/split schedule, caller-owned split workspace, FP32 softmax
        statistics, exact-zero inactive tails, and both W2048/W4096 cyclic contracts; its physical
        FP64 qualifier passes mixed lanes, wrap boundaries, short/direct and saturated widths with
        max absolute error 0.00125. The selected real W2048 T1 point measures 0.063 ms versus
        1.17 ms for unsplit bring-up. Grouped convolution consumes BF16/W8 projection storage
        through native Linear, fuses phase-0 convolution with phase-1 stash extraction, and passes
        complete represented-input FP64 formulas at D5120/G320 with max absolute error 1.59e-5.
        Their obsolete CUDA owners and duplicate CUDA tests are deleted, and the linked runtime
        planner passes ordinary/Vision/MTP/DFlash at C=1..4.
- [x] Restore fixed C=1..4 scheduling, state transaction, RAM spill/restore, and device graphs.
  - [x] Rerun every directly cap-sensitive physical qualifier after the shared request-lane maximum
        changed to four. Runtime planner, GDN recurrence, target GDN composition, scalar schedule,
        sampling, DFlash KV append-prefix, SWA, MTP round, and speculative round all pass on the
        physical gfx1201 R9700 at their active B/C<=4 contracts. SWA initially exposed a stale
        qualifier-only B8 workspace-capacity probe: the production Op correctly rejected that
        out-of-contract request before launch. The probe now uses the shared B4 maximum and the
        complete qualifier rerun passes; its retained direct W2048 T16/B4 point is 0.053 ms with
        the existing maximum-absolute oracle error 0.00125. This support requalification closes no
        capacity, whole-inference, quality, or terminal-selection item.
  - [x] Retire the NVIDIA-calibrated adaptive-draft lane and make speculative width a single
        startup-fixed product contract. Engine, CLI, serving, benchmark, campaign tooling,
        workspace planning, request state, and Device Graph ownership now contain only the
        configured K and resolved DFlash verify width; the obsolete EWMA policy, timing tables,
        multi-K captures, tests, and temporary plans are deleted. Any future automatic-K policy
        requires real selected-artifact same-candidate parity, acceptance, and whole-round R9700 timing
        gates for every candidate K and C=1..4.
  - [x] Cut the Qwen sequence layout and Program ownership metadata to the canonical typed cache:
        one fixed G16/token-K/feature-V/feature-scale schedule profile constructs Text/MTP state,
        no dtype/legacy quant-group/Sage/K-stat/sparse flags remain, and sequence state carries
        healthy/closed publication objects. Program transfers/events are native HIP and RAM
        capture/restore binds the exact Text/MTP semantic fingerprints. Numerical schedule calls
        still need to launch the canonical all-layer transactions before this parent item closes.
  - [x] Physically qualify the typed per-row MTP cache lifecycle needed by fixed scheduling:
        device-prefix alignment uses a smaller row maximum and ignores invalid inactive tails, one
        active AR append advances the provisional frontier, a zero-count AR append is byte-inert,
        and licensed, partial, and cancelled resolution truncate publication without rewriting
        retained cache bytes or poisoning the sequence. A separately allocated contiguous append
        is the exact three-plane byte oracle for the split alignment/AR mutations.
  - [x] Bind the eager Text and MTP family schedule to caller-owned typed transactions. Ordinary
        and target verification publish exact Text suffixes; MTP alignment consumes device active
        counts, while one round-scoped segmented authority now spans alignment and every AR append
        with fixed device status/cursor and a device-selected live block-table row. Eager and Device
        Graph use the same captured addresses and do no per-segment host readback; one post-round
        resolution publishes the complete Text verify suffix and only the licensed MTP alignment
        suffix. Prefix reuse, cancellation, partial acceptance, and checkpoint restore no longer
        repair MTP frontiers directly. Independent exact three-plane byte oracles prove contiguous
        equivalence, zero-count inertness, partial retention, and eager/graph parity; dynamic-row
        pending attention passes its FP64 oracle. The complete runtime/Engine boundary and C=1..4
        planner build and execute on gfx1201.
  - [x] Keep Qwen3.8 DFlash2 state separate from growing Text/MTP KV: Local,
        rewrite-checkpoint, and staging are fixed BF16 cyclic caches with native HIP lane copies.
        The physical qualifier passes an exact all-layer lane-1 capture/clear/restore oracle; the
        dormant family Full-cache shape has its own two-plane BF16 owner rather than reusing the
        asymmetric three-plane cache.
- [ ] Held behind the immediate dense C1/P2048/spec-none `>=2,000 tok/s` floor and
      practical-ceiling gate above; the user-authorized smaller-gain exact-Q4 continuation does not
      authorize this downstream campaign. After those gates pass, produce
      real FP8-K/INT4-V 8K and 32K paired quality, diagnostic greedy-token, graph/eager, and
      spec-none ordinary prefill/decode evidence; after base selection, produce required DFlash
      speculative acceptance evidence for the winner only.
      Missing candidate-local quality is owned by
      `profiles/ppl/terminal-quality-recovery-20260905`; matched C1..4 ordinary whole evidence is
      owned by `profiles/bench/post-chunk-twelve-candidate-20260905`, while selected-only acceptance
      is owned by the DFlash pipeline. These remain blocked behind the selected chunk and no CPU
      preparation closes this physical parent.
  - [ ] Reconcile the completed four-role hybrid prefill-quality evidence with fresh
        selected-chunk capacity evidence.
        The fail-closed CPU report is
        `profiles/bench/r9700-fp8-hybrid-quality-capacity-admission-20260904.json` (SHA-256
        `8d6cfd8525905046c6c3837a77eeb58b9268559ca37409cdcc3692b64e67c99d`). It reopens the
        current artifact and adjacent conversion receipt, rehashes both scorer identities and all
        NLL/argmax sidecars, and recomputes the paired gates. The 8K campaign (SHA-256
        `f30ef0d102ba5b54ed1c9c689641065839b0523c065692494ec4db1af63c2d20`) has delta mean NLL
        `0.0241813264` and 4/11 new-severe positions; the 32K campaign (SHA-256
        `97aacde821b0e570995b2fdc86cf9167763961ad92cfeb70d034a11fffdfa8f0`) has delta mean NLL
        `0.0279584773` and 25/41. Both beat the `ln(1.05)` and 0.25% gates. The regenerated current
        C1..4 capacity report (SHA-256
        `e81d966017cc464938cdc612ec2f9da57fd99e84557f23763fef4242ece17fa4`) is not admissible
        current capacity evidence: it used `20,707,768,320` dense/spec-none materialized bytes
        while claiming MTP3 plus the optimized head. The exact feature materialization is
        `21,290,468,352` bytes, which makes the same P=8,192/G16/C4 plan startup-inadmissible by
        `245,140,480` bytes; the old apparent C1--C4 slack values are invalid. Quality remains
        admitted, but capacity must be rerun only after production chunk selection and must retain
        any measured failed cells. Performance promotion is also still open: the current
        dense C1/P2048/chunk-4096/spec-none authority reaches `1,847.942898 tok/s`, below the
        `2,000 tok/s` floor and without practical-ceiling proof. The prepared broad C1..4 matrices
        remain intentionally unlaunched; decode graph/eager and speculative execution evidence
        also remains open.
  - [x] Implement the independent BF16-source scorer that emits the campaign's
        `bf16-reference` JSON plus index-aligned FP32 NLL and exact I32 argmax sidecars. The
        checkpoint-native evaluator validates the exact 1,199-tensor, 55,562,855,904-byte,
        18-shard source index and all 851 text BF16 tensor shapes, then stages one original
        source layer at a time instead of loading 51.75 GiB on the 32-GiB device or reloading it
        for every decode token. Layer-major prefill/decode preserve the product score indices but
        intentionally use the same efficient oracle chunks; the paired product binary, not the
        mathematical oracle, owns the T=1 shape gate. Multi-token GDN uses external FLA and an
        explicit FP32 recurrence remains its short-tensor authority. Full attention retains BF16
        K/V with FP32 softmax/PV, and the untied BF16 head scores only the 248,077-token public
        domain. Schedule/spec/graph labels cannot alter the formula. The
        dependency-light protocol/mapping/sidecar suite and the aligned quality-gate runner pass.
        On physical gfx1201, FLA 0.5.2 matches the explicit recurrence on the
        short oracle, all real source metadata passes, and a four-token/three-score traversal of
        all 64 original BF16 layers completes in 97.351 seconds with finite, aligned sidecars and
        no retained device allocation. The current in-tree C++ scorer remains correctly only the
        fixed FP8-K/INT4-V product route. Repeated full-model runs under the former default BLAS
        route produced different BF16 NLL and argmax sidecars. The localized deterministic
        fixed-order-PV/hipBLAS/no-atomics/strict-algorithm profile is now mandatory and fully
        provenance-bound. Fresh non-trace v3 8K/32K repeats are exact and establish the quality
        authority; dense all-Q4 gate recomputation is complete, while mixed and sparse candidate
        rebase remains open under the localization child above.
  - [x] Obtain the complete 18-shard BF16 source, its six frontend resources, and a supported
        Python environment. The official source is complete at the maintainer-provided path with
        1,199 indexed tensors and 55,562,855,904 bytes. The isolated Python 3.12 environment has
        ROCm PyTorch 2.9.1, safetensors 0.8.0, and flash-linear-attention/FLA Core 0.5.2; the real
        source preflight, tensor metadata pass, and shortest full scorer execution succeed.
        The current target-owned prerequisite receipt is
        `profiles/ppl/r9700-bf16-source-checkpoint-preflight-20260905.json` (SHA-256
        `4c605982c84fbfc8803fea24ccdd0230277f4f20c8547a38098f6f440bf99fa9`); it validates the
        exact config/index and all 18 named nonempty regular shards without claiming payload
        hashes. Source acquisition is therefore unblocked, while fresh conversion, PPL,
        exact-token, and physical admission remain separate pending gates.
        Runtime applications continue to read only the `.ninfer` artifact.
  - [x] Re-audit the no-output conversion boundary after all source shards became local. The
        executable environment is
        `/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python` with
        `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib`; unqualified `python3` has neither
        Torch nor safetensors, and the isolated interpreter cannot import ROCm Torch without that
        library path because `libroctx64.so.4` is otherwise unresolved. The official all-Q4
        `--preflight-only` command passes with 18 shards, 1,199 BF16 tensors, six resources, 1,124
        objects, 439 Q4 matrices, ranking SHA-256
        `205b2d6c5b58d946c87b425578da13b5cfc32a0abe1ff201e95b47d032bf004e`, and provenance
        SHA-256 `0f8dafeb34abf5729b789906343535f1cd8d43712e62fa6d83538222d47339bb`.
        The mixed source-MSE converter now exposes the same no-write `--preflight-only` CLI
        boundary and passes it with the exact 18-shard/1,199-BF16 source, six resources, validated
        ranking/provenance, 1,124-object plan, and expected 183-Q4/256-W8 inventory. Its
        machine-readable summary binds the config/index paths, sizes, and SHA-256 digests, all 18
        exact shard names and sizes (without reading 52 GiB solely to add unrequired payload
        hashes), all six ordered resource names/sizes/digests, and the complete object-plan digest
        and byte accounting. The mode accepts neither `--out` nor `--device`, never selects a
        device, and creates no artifact or temporary output. The DFlash
        no-output preflight does pass against the existing mixed evaluator: base SHA-256
        `8fbadf14e355b1943ef9386a91ebafff0d852295a9adc0054bdd430291505ebd`, 81 source tensors,
        66 appended objects, 1,190 combined objects, byte-exact base copy, and projected
        24,090,686,464-byte file. No new evaluation conversion is needed before selection: the
        22,881,204,736-byte mixed evaluator and its conversion report already exist. A fresh
        production artifact remains blocked on the selected-chunk and six BF16-source quality
        authorities, terminal schema-v7 recipe/cache/execution decision with matched C=1..4
        spec-none ordinary whole timing, selected dense-control C1 P2048
        admission at the 2,000 tok/s floor, selected-route NIAH admission, and schema-v2 DFlash
        decision. Only after all bind the same winner may cutover add the selected
        recipe's no-output CLI boundary, replace evaluation converter identities with the one final
        authority, and materialize the final base followed by its DFlash companion. Candidate
        quality/whole runs also require freshly rebuilt G16 and G32 scorers because the available
        paired scorer binaries predate the current production kernel promotions.
  - [x] Complete the no-GPU paired-campaign boundary before the real measurement. The schema-v6
        runner now validates and hashes the exact shared G16/G32 artifact, scorers, and Qwen3.8
        corpus manifest; rejects per-cell model, weights, group, schedule, graph, speculative,
        chunk, prompt-length, or threshold drift; requires explicit candidate PPL gates; applies
        finite/aligned, mean-NLL, and new-severe-position quality gates against BF16. Prefill/decode
        schedule comparison requires an explicit measured max-absolute-NLL bound and treats greedy
        flips as diagnostic; same-route graph/eager, MTP/ordinary, and draft-window pairs retain
        exact argmax plus bounded per-token NLL parity. Native benchmark schema v20 records the
        G16/G32 group, exact K/V/V-scale plane layouts, Q4/W8 activation profiles, and exact FP8-Q/K
        classifier and compile-bound attention route; the schema-v14 matrix
        runner binds artifact and executable hashes and rejects a wrong-profile, incomplete, or
        stale resumed report while retaining coherent speculative acceptance counters. This preparation
        selects neither group. Retained base-recipe and dense/sparse-profile 32K sidecar acquisition is
        complete for all-Q4, and dense all-Q4 BF16-derived eligibility has been recomputed against
        the exact v3 authority; all four fresh all-Q4 C=1..4
        capacity matrices and both fresh mixed dense and sparse capacity matrices are complete. The
        sparse all-Q4 and mixed dense/sparse PPL acquisition, all mixed and
        all-Q4 whole matrices, and ordinary same-route execution evidence remain. Speculative
        acceptance belongs to the selected-only DFlash gate rather than base selection.
- [ ] Held behind the immediate dense C1/P2048/spec-none `>=2,000 tok/s` floor and
      practical-ceiling gate above; the user-authorized smaller-gain exact-Q4 continuation does not
      authorize this downstream campaign. After those gates pass, benchmark
      complete inference and concurrency C=1..4 with profiler attribution. The base
      closure is supplied by the same twelve schema-v14 capacity outcomes and corresponding
      eligible whole matrices used by
      the schema-v7 `terminal_production_selection`; do not schedule a duplicate C=1..4 campaign. DFlash
      and any bottleneck-specific profiler evidence remain downstream of that selection. The
      prepared owners are `post-chunk-twelve-candidate-20260905` before selection and the selected
      hardware-use and ordinary-decode-memory packages afterward; none has supplied the
      pending physical evidence.
- [x] Delete every remaining CUDA, RTX 5090, sm_120a, NVFP4 artifact/cache/kernel/tool/test/doc path.
  - [x] Remove the retired artifact format from the C++ and Python numeric/layout registries,
        delete its block-scale codec, converter, model-card, specialized evaluation, and final
        kernel-header ownership, and make the generic Python container framing/geometry path
        independent of Torch. Exact scans of core/artifact/converter/R9700-tool/eval ownership are
        clean; dependency-free v2 round-trip and v1 rejection, physical HIP materialization, and
        complete 1,124-object W8 candidate binding pass. The final active-path audit removed the
        stale CUDA clangd override, SM120/NVFP4 editor rule, CUDA formatter ownership, ignored
        NVIDIA profiler artifacts, 5090-branded schema fixture, obsolete handoff document and
        NVIDIA tensor-core spreadsheet, empty NVFP4 source directories, CUDA audit stubs, and
        legacy CUDA/NVFP4 build tree. Active source, build, tool, test, root-document, and filename
        scans are clean. Only the live migration ledger and historical overhaul record retain
        explanatory references.
- [x] Replace target identity, registry, CLI defaults/help, serving reports, README, performance
      tables, artifact authorities, Op authorities, and kernel-development tooling with the one
      R9700 integer-profile product contract.
  - [x] Port the sole-target registry to native HIP memory queries on the selected device and
        align its runtime owner and load-summary key with the provisional Qwen3.8-27B R9700
        package constants. The active CMake boundary compiles the registry and directly qualifies
        sole-target rejection plus explicit/automatic page-aligned capacity behavior.
  - [x] Remove the public `KvCacheStorage` selector and all NVFP4/INT8/BF16 growing-cache and
        Sage/Sparge/XAttention knobs from `EngineOptions`, `MemorySummary`, CLI, serve, and PPL.
        These product surfaces now own the one fixed FP8-K/INT4-V format; their positive option and
        help contracts contain only fixed-cache capacity and Device Graph controls, schema-v20
        server-start JSON reports `fp8-k-int4-v`, the compile-bound cache group, and attention
        profile; the physical HIP environment qualifier passes,
        and PPL compiles with no format selector or backend-specific environment route. The PPL
        campaign selects an independent
        BF16-reference executable/artifact and separately configured R9700-G16 and R9700-G32
        evaluator/runtime builds rather than mutating an Engine option; those runtime-state
        profiles may consume the same explicit weight candidate unless the weight recipe is also
        under test. It enforces paired-NLL schedule gates with diagnostic flips plus exact per-token
        argmax for same-route execution variants, and passes a complete
        synthetic launcher/report smoke test. Serve corpus/concurrency drivers likewise launch no
        cache selector and validate the one schema-v20 `fp8-k-int4-v` format.
  - [x] Audit all active product and maintainer documents, excluding the historical overhaul
        record, against the sole Qwen3.8-27B R9700 contract. README, CLI, serving, performance,
        artifact, Op, model, cache, and concurrency authorities now agree on native HIP, fixed
        draft K, FP8-K/INT4-V/FP16-scale runtime state, provisional G16 and W8 selection, and the
        distinction between cache build profiles and artifact identity. All 17 active local
        Markdown link targets resolve, exact retired-backend/identity/option scans are clean, and
        the affected documentation diff passes whitespace validation. The parent remains open for
        non-document product surfaces and final real-model selection.
- [x] Remove superseded Qwen product lanes, converter paths, fixtures, source references, aliases,
      and docs as the selected Qwen3.8 R9700 route replaces them.
  - [x] Make the provisional R9700 converter self-contained by moving the neutral Qwen3.8
        1,118-tensor inventory, BF16 source recipe, config validation, exact frontend-resource
        hashes, and draft shortlist into its target-owned package. Delete the old Qwen3.6-27B and
        Qwen3.8 groupwise/low-bit converter packages, specialized evaluation scripts/configs, and
        published model cards; synthetic inventory/codec and sparse complete-binding checks pass.
        Final integer-recipe and artifact-weights identity remain at the real-model gates.
  - [x] Remove the superseded Qwen3.6-35B-A3B lane: its target package, registry/Engine variant,
        sparse-MoE-only Op, converter/reference tooling, target tests and benchmark, evaluation
        configs, model card, dedicated authorities, and active source/tool/document references are
        gone, including generic-named kernels/tests/benches whose admitted geometry belonged only
        to that lane. The sole-target Engine owner no longer carries a runtime target variant. The
        active gfx1201 build and artifact, core, DFlash state, eager, full-attention, KV-RAM,
        sampling, persistent-state, and target-binding qualifiers pass; exact scans find no
        remaining nonhistorical retired identity, geometry symbol, or sparse-MoE owner.
  - [x] Move the target-private Python Text/Vision/MTP reference and source-BF16 Vision diagnostic
        to `qwen3_8_27b`; bind only the exact 1,124-object
        `qwen3.8-27b/r9700-int-candidate` W8G32 inventory; and replace the reference cache selector
        with the fixed FP8-E4M3FN-K/INT4-G16-V/FP16-scale logical codec. Dependency-free binding
        signatures, static cache geometry, and Python compilation pass. The complete BF16 source
        and converted evaluation artifacts are now present; real-model parity remains open on the
        final recipe/artifact selection and its still-pending physical comparisons.
  - [x] Cut the C++ product leaf from the superseded `qwen3_6_27b` package to exact
        `qwen3_8_27b` ownership and rename the identity-free shared runtime from `qwen3_6` to
        `qwen3`, including exported headers, namespaces, instantiation macros, private leaf
        symbols, registry construction, CMake targets, Engine/PPL callers, R9700 qualifiers,
        benchmarks, family tests, and the active model authority. No registry compatibility alias
        or old C++ include/source path remains. The configured apps+benchmarks+tests build passes,
        the exact package/registry and family CPU checks pass, and the serialized physical gfx1201
        suite passes 52/52.
  - [x] Remove superseded Python/tool identities from benchmark, serving-smoke, CLI-example,
        converter, reference, parity, and Python-test surfaces. The retired ranking manifests and
        reports are deleted, their raw counts no longer feed conversion, and the candidate now
        requires an explicit external Qwen3.8-derived draft ranking recorded in its report. Active
        JSON references and stored hashes validate; focused benchmark/serve checks and Python
        compilation pass. The edited Markdown translation fixture is now recounted through the
        exact Qwen3.8 target frontend at 357 prompt tokens with thinking disabled; all-Q4, mixed,
        and four-role artifacts agree on the same committed message bytes and frontend resources.
- [x] Validate `.ninfer` framing, converter tensor inventory, bound layouts, source-checkpoint
      provenance, and no-runtime-repacking with a real BF16 source checkpoint. A read-only CPU
      audit validates the six active real artifacts (all-W8, source-MSE all-W8, all-Q4, Q4/W8,
      source-MSE Q4/W8, and W8 with BF16 token embedding) against their exact ordered 1,124-object inventories, registered encoded
      sizes, aligned spans, conversion reports, complete 18-shard/1,199-tensor BF16 source recipe,
      independently rebuilt draft ranking and I32 shortlist, and all six official frontend
      payloads. The current binder accepts each active identity; the structurally valid retired
      BF16-head evaluator is rejected as intended. Reconstructed device plans account for every
      tensor byte, and the materializer plus typed row-split bindings directly copy and expose the
      stored code/scale planes without decode, requantization, transpose, or per-weight allocation.
      The machine-readable evidence is retained under `profiles/artifact/`.
  - [x] Cut every active Q4 artifact writer, reader, binder, consumer, benchmark validator, and
        command example to the Q4-only `r9700-q4g64-n16-k16-v1` persistent layout. The canonical
        four-role evaluation input for all forthcoming P2048, capacity, focused-profile, PPL, and
        DFlash preparation is
        `out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer`, identity
        `qwen3.8-27b/r9700-q4g64-f8e4m3-four-role-n16k16-eval`, size 21,553,549,312 bytes, SHA-256
        `040c6e7ed29c856718a638c00181975710d987b7d5f49f4cafbdf68911f7e7d2`. The former RowSplit
        artifact remains only immutable historical evidence and the explicit offline-transcoder
        source; active validators require the layout-distinct identity, so passing the former path
        fails before a benchmark launch. The lossless migration independently matched every Q4
        logical code/scale plane and every non-Q4 byte, reopened all 1,124 ordered objects with 295
        Q4 descriptors, and left the source SHA-256
        `1dfe9626fd6412592f87480a2f6934e8494a4b267693a25831dff490959542ce` unchanged.
- [ ] Complete BF16-source model parity for the ultimately selected integer artifact and every
      supported execution schedule before finalizing its identity.
      The prepared selected-route boundary is
      `profiles/bench/post-terminal-focused-verification-20260905`; it cannot run or publish until
      schema-v7 identifies the exact artifact, build, group, profile, and chunk.
  - [x] Establish per-Op numerical correctness for every promoted integer route against its
        independent exact-code or complete FP32/FP64 represented-input oracle at the real model
        shapes and route boundaries. Pairwise agreement with a BF16 implementation is
        supplementary evidence, not the Op oracle, and is not substituted for these completed
        qualifications.
  - [x] Implement the checkpoint-direct BF16 Text scorer and retain complete, finite, aligned 8K
        and 32K prefill comparisons for both G16/G32 builds of the all-Q4+A8 and source-MSE
        mixed-Q4/W8+A8 candidates. The retained comparison binds the 1,199-tensor/18-shard BF16
        source plus two BF16 realizations and eight candidate JSON/NLL/argmax triplets by hash.
        Exact current-tree v3 8K/32K repeats now establish the BF16 authority, and dense all-Q4
        G16/G32 has passed offline rebase; mixed and sparse candidate admission remains open under
        the localization child above.
  - [ ] Complete the still-open selected-artifact decode/same-route execution comparisons and
        source-BF16 Vision diagnostic where applicable. The complete BF16 source and converted
        evaluation artifacts are present; the remaining parity waits on final recipe/cache
        selection and physical comparisons using the selected artifact and executable. Host
        inventory or synthetic Op-oracle success cannot close it. Use the prepared focused verifier
        for the selected-route comparisons rather than creating a second broad suite. The exact
        PPL/token owner is prepared at
        `profiles/ppl/post-terminal-exact-token-prepare-20260905`: it runs only the selected winner
        at C1 and publishes only after both required 8K/32K ordinary graph/eager exact-parity cells
        revalidate. MTP comparisons are optional non-ranking diagnostics.
        Its schema-v7-dependent physical campaign and admission output remain absent.
        The source-BF16 Vision diagnostic is separately prepared at
        `profiles/bench/post-terminal-selected-vision-prepare-20260905`. It resolves only the
        schema-v7 winner, freezes the committed single-image/no-thinking input with an explicit
        CPU frontend environment, runs the selected-artifact and source-BF16 Vision towers in the
        ROCm environment at C1, and publishes only after exact block-0/13/26/merger shapes and
        finite metrics revalidate. This remains diagnostic with no numerical threshold; CPU
        preparation does not close the unchecked physical parent.
- [ ] Run the complete focused correctness suite, artifact integration, serving schemas, real-model
      inference, PPL/token gates, graph/eager parity, speculative acceptance, and end-to-end speed
      suite on the selected final artifact.
  - [x] Reconcile the final focused runner with the current Qwen3.8/R9700-only CTest and Python
        authorities. The bounded CPU layer is exactly 15 native host contracts, eight external
        serving-schema contracts, and 23 explicitly named artifact/converter/report/reference and
        interpreter-boundary Python files in `r9700-final-artifact-cutover.md`; it does not run a
        device-labelled qualifier, real model, or performance campaign. The native list now includes the live
        tensor, linear-prefill-dispatch, FP8 activation/execution-state, and XAttention oracle
        contracts that the stale regex omitted. The device list includes the still-selectable
        XAttention implementation instead of calling it rejected, and continues to exclude the
        instruction micro-probe. All named CTest targets and Python files exist. The Python command
        now requires one explicit absolute interpreter path providing pytest, Torch, and
        safetensors and fails its import preflight instead of silently selecting the
        dependency-free system Python. It preserves the venv launcher rather than resolving its
        symlink to the base interpreter, and the transient route binds the launcher, executable,
        Python/package versions, prefix, and `pyvenv.cfg`; the existing
        `/ssdpool2nvme/local_llm/ninfer-dylan2/eval/.venv/bin/python` passes that preflight.
        The flattened benchmark fixture now carries the mandatory production ping-pong Q4 CTA and
        speculative-execution fields. CPU execution passes; physical qualifiers and all
        selected-final-artifact/model gates remain open under this parent.
        The post-selection entry point is prepared at
        `profiles/bench/post-terminal-focused-verification-20260905`: it refuses to run without a
        revalidated schema-v7 winner, reopens the winner's exact capacity/whole manifests and
        current artifact/benchmark/planner bytes, verifies the selected build's G16/G32 and
        dense/tau=.9 CMake profile, then runs only the named host/schema/Python boundary or the
        separately authorized selected physical qualifiers. The physical mode adds XAttention
        only when selected and all future runtime scope remains bounded by C=1..4. Its transient
        route cleanup and final success publication are dev/inode-owned, require the exclusive
        hard link to retain that exact inode, and preserve a concurrently replaced pathname.
        Package-local tests pass; the one terminal-selection closure row remains intentionally
        pending the deferred shared-runner/downstream closure cascade. It is currently
        blocked on the absent terminal schema-v7 authority; no physical test was run.

## Driver-dependent repeat gate

- [x] Install a coherent supported AMD driver and ROCm release, reboot, and requalify device
      identity, graphs, ISA, profiler trace, PMU collection, launch tuning, and every published
      performance selection.
- [x] Record the supported driver/ROCm/kernel versions and rerun all rejected/failed/unstable
      profiler or graph checks; do not publish pre-update performance as final evidence.
