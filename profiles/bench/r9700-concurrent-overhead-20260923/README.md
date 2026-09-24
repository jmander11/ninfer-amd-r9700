# Concurrent projection, round-overhead and attention/FP8 continuation

User continuation from8562efdd. Same installed cap26 Q4-head/gate-up-A4 model,
Q4 DFlash/BF16 selector codebooks, G16 fixed cache, dense, chunk2048; prefill paused.
R9700 gfx1201 wave32 PCI0000:13:00.0 ROCm10, auto. C<=4.
Heavy jobs serial, memory20Ghigh/24Gmax/no swap, builds4 (hard cap14).
No CPU quota on timings. One independent CPU reviewer, no agent GPU/build jobs.

First decision: identify exact concurrent generic Q4 projections and contribution
at K4/C4 (T20), compare against known T5/6 successor pipeline. New shape admission
requires original-input/represented-A8 FP64 bounds, codec/exactgeneric/graph/guards,
ISA/resources and measured whole benefit. No weights/arithmetic change planned.
Profiler capture includes graph/runtime/kernel/copy traces to separate useful
service from intra-graph/inter-round gaps. Profiled times are attribution only.
Protected FP8 and attention candidates require an independently justified bound;
prior rejected split-K/down/PV/depth2 candidates remain excluded.

Run `PYTHONPATH=. /home/battlefront/.local/bin/python3.11
profiles/bench/r9700-concurrent-overhead-20260923/run_cell.py --label <fresh>
--concurrency <1..4> --draft <4|5> [--trace]` inside the bounded systemd scope.
The script retains exact command/report/receipt and compares every repetition
with same-C ordinary tokens. Final selection uses unprofiled timings only.

## Concurrent projection Layer0

Fresh C4K4 graph4 contains31 steady rounds. Generic Q4 consumes635.22ms, of
which vocab/selector account111.93ms; the remaining523.29ms (20.25% of steady
kernel service) belongs to the eight missing projection shapes. GDN N4096 and
N12288 each execute48times/round; output/feature N5120 calls are distinguished
by graph node and schedule, not grid alone. No selector/vocabulary admission.
Cold complete-Op T20 baseline ms, three weight copies/80MiB scrub/24 samples:
N4096/K5120 .11464; N12288/K5120 .16158; N5120/K6144 .14082;
N7168/K5120 .12874; N6144/K5120 .12356; N1280/K5120 .10890;
N5120/K4096 .10556; N5120/K25600 .46086. Original/public and represented FP64,
exact generic and codec pass. Hypothesis: the generic ascending group-load
dependency exposes latency; existing successor pipeline overlaps memory with
current IU4/FP32 work. A20% saving on these calls offers~3.8% kernel-service
benefit (about4.05% using the exact20.25% share). Extend only
T10/12/15/18/20/24 with exact same mathematical profile.
N12288 T5/6 must retain scale-gather; concurrent widths require tiled pipeline,
not the <=8-row scale-gather accumulator. Existing output T12/18/24 unchanged.
No hidden allocation, repacking, workspace or accumulation-order change.

The task-local screen's `--timing-only` skips repeated host FP64 work only after
the exact linked core and all48 same shape/fixture cells passed canonical
`--concurrent-only` qualification; exact generic/codec/guards remain checked in
each timed pair. Baseline T20 screen ran its own oracle before candidate creation.

## One paired-QK challenger

The complete-attention baseline screen qualifies W4/5/6 at context64,
133/134/135 and4100/4101/4102. At4K, warm complete-Op medians are
0.28178/0.28830/0.34542ms; 80MiB-scrub cold medians are
0.28648/0.29308/0.35492ms. Use the same `attention_screen.hip` source and
frozen control/candidate cores; no Text-prefill campaign is part of this screen.

Mechanism: put two query rows' six heads per KV head into two eight-M-lane
groups of the same16x16x16 FP8 WMMA. Input M slots0..5 describe row2p;
slots8..13 describe row2p+1; other slots are zero. Output lane-half selects
the causal row and accumulator j selects head j<6. W5's last pair masks its
nonexistent second row before query loads and score stores. Score layout,
FP8-Q encoding, sixteen ascending K chunks, softmax and PV remain unchanged.
Invalid page-table rows/pages poison all represented scores; no workspace,
graph-key, format or precision change. This is independent-row packing, not
split-K, deeper lookahead, or the rejected PV row-sharing implementation.

QK CTA/K-load/WMMA counts ideally fall W4 4→2, W5 5→3, W6 6→3.
The old Q loader fetched50 head vectors across four KV heads per query row
(16+16+12+6) although only24 were used; masking now fetches exactly24,
eliminating52% of Q loads/conversions at every width. This is instruction/load
traffic, not a claim about DRAM transactions or realized runtime. FP32
accumulator width stays eight; added row predicates/indexing may increase
register pressure or erase the benefit. Graph4 QK is5.90% of kernel service;
the W5 ideal40% reduction is about1.97ms/round before those costs. One bounded
prototype is permitted: reject spills, any oracle/serial/graph/guard failure,
or insufficient complete-Op and matched whole-inference gain. No tuning sweep.

## Projection qualification and operator result

All48 concurrent cells pass canonical original-input/represented-A8 FP64 bounds,
exact generic/eager/graph, codec, stale workspace/poison, guards and immutability;
144 deliberate output corruptions rejected. Maximum public norm-budget use0.88825.
All44 preexisting device instruction/resource streams remain exact. The45 new
kernels use nativeIU4/wave32,70–75VGPR/22SGPR, zero LDS/private scratch.
All45 new cells save3.66–51.66% in matched cold complete-Op timing. The three
retained N5120/K6144 T12/18/24 cells are compared with generic only and are not
counted as new production wins. `projection-speed-summary.json` records every
cell; whole-inference admission remains a separate measurement.

## Paired-QK qualification and operator result

Canonical short/long discriminator passes its unchanged public BF16-Q/FP8-profile
oracles, serial-bit-exact and graph checks; new invalid-device-row/physical-page
tests poison all represented W4/5/6 rows and preserve guards. Nine task-local
complete-attention cells also pass independent public/profile oracles and exact
serial/graph comparisons. QK remains24VGPR/32SGPR, native FP8 WMMA, no LDS/private
scratch; softmax and PV instruction/resource streams are unchanged.
At contexts4100/4101/4102, cold complete-Op W4/W5/W6 changes
.28648→.25604/.29308→.26876/.35492→.29476ms (10.63/8.30/16.95% reductions).
Warm reductions are11.96/9.60/17.48%. Short cells are effectively unchanged:
mixed-sign differences below1% do not justify a separate short-context route.
Independent review accepts this for whole-Engine qualification, not yet as a
whole-inference speed claim.

## Other owner decisions

`offline-owner-review.md` and `c4-attribution.json` retain the runtime/FP8 audit.
Between-round uncovered time is31.48ms over3.591s of traced decode (<0.9%). The
first pageable D2H/synchronization calls overlap actual device execution; their
whole API durations cannot be claimed as removable CPU overhead. Intra-graph
uncovered590.50ms is materially confounded by profiling (most gaps5–12us,
profiled142.58 versus unprofiled162.21tok/s), and the copy table is empty despite
the requested trace. No claim of hardware idle/stall freedom follows from it.
Shared protected-FP8 input preparation removes at most0.261ms/round, insufficient
for a standalone material improvement. FP8 matmul is4.45% of steady kernel
service; no specific algorithm defect was identified. No custom FP8 rewrite or
repeat of rejected FP8 small-T kernels is justified. These are bounded exclusions,
not proofs of absolute ceilings. Prefill stays paused and chunk2048 is unchanged.

## Final whole-inference admission and closure

All bounded targets are complete. The results below supersede the provisional
whole-inference admission wording above. Same installed cap26/Q4 DFlash artifacts,
dense G16, auto power, P4096/G128, chunk2048, warmup1/repetitions3; aggregate tok/s:

| Mode | Fresh control | Projection only | Final paired QK |
|---|---:|---:|---:|
| C1 K5 | 101.26 | 101.26 | 102.73 |
| C2 K4 | 122.58 | 144.27 | 146.20 |
| C3 K4 | 160.20 | 186.82 | 190.96 |
| C4 K4 | 162.21 | 178.75 | 181.89 |
| C4 K5 | 154.05 | 162.85 | 166.90 |

C1 control is projection-only because its device bodies are unchanged. C4 adaptive
maxK5 is180.21 aggregate (45.05/request), without a fresh pre-change adaptive
control. All18 final repetitions match same-C ordinary decode exactly; fixed-K
rounds and accepted-token counts also match controls. All44 cold K3/adaptive
graph/eager transition cases pass exact ordinary tokens. Six pending rows at K4
remain unobserved; append/storage are unchanged and maximum-capacity checks pass.
Four focused host tests and two static attention tests pass; CLI/server/PPL/bench
are rebuilt. Independent review reports SHIP. No weights or precision changed;
ordinary decode and prefill were not remeasured.

Final trace confirms attribution, not a throughput claim: over31 graph4 rounds,
generic Q4 service falls635.22→112.03ms (remaining vocabulary/selector), QK
152.40→106.09ms, and total kernel service2583.95→2248.14ms. Final inter-round
uncovered time is28.15ms; intra-graph uncovered595.85ms remains profiler-confounded.
No new host/FP8 rewrite follows from this capture.

`final-summary.json` joins the completed measurement and correctness reports.
Reproduce explicit measurement cells with `run_cell.py --help`; inspect its fixed
artifact paths before launch. Qualification/screen helpers are `build_screen.py`,
`projection_screen.hip`, `build_attention_screen.py`, `attention_screen.hip`,
`build_tail.py` and `verify_cold_tails.py`. Saved command/receipt pairs retain the
exact invocation. Run heavyweight jobs serially; build at most14 jobs, normally4.
`summarize_final.py` and `analyze_trace.py` are offline analysis only. Raw databases
and frozen binaries remain local; structured evidence and reproduction sources
are versioned. This is a bounded improvement, not an absolute-ceiling claim.
