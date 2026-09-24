# Remaining candidate admission

Start afd4bf1f. User authorizes testing and retaining beneficial gate/up cross-tile
reuse, adjacent-feature PV and exact-tree GDN normalization. Bounds and exclusions:
`../r9700-concurrent-overhead-20260923/remaining-investigation-20260924.md`.
Same installed cap26/Q4 DFlash model, dense G16, chunk2048, auto power, C<=4.
Heavy jobs strictly serial under20GiB MemoryHigh/24GiB MemoryMax/zero swap;
build normally4 jobs, never more than14. Primary owns GPU and builds; independent
agent owns GDN source/helper editing only until explicitly frozen for build.
Every candidate requires independent oracle, exact codec/state/graph checks where
applicable, ISA/resources and unprofiled complete-Op timing before promotion.
Keep fresh outputs and losing-result evidence; remove losing implementation code.

## Initial screens (not final admission)

Gate/up T20 paired tiles pass the public-input FP64 budget, exact incumbent,
codec, graph and guards. Complete-Op alternating medians: rotating-address
.344119→.204819ms;80MiB-scrub cold .464118→.324259ms. Native IU4,
103VGPR/26SGPR, no LDS/private scratch. This admits a full canonical T15 control
and T18/20/24 candidate qualification, not automatic promotion of other widths.

Initial GDN exact-tree screen passes all six C1/C4 W4/5/6 cells plus W2/W16
boundaries; warm gains12–19%, scrub-cold7–22%. Its emitted square/add still
contracted despite HIP rounded intrinsics. A register-only square materialization
barrier was added; final ISA/qualification/timing must use the revised build.
Initial barrier instruction sites40→6 represent20→3 signal/wait pairs, not
40→6 independent barriers. Initial resources54→62VGPR,1536→1024LDS, zero scratch.

Broad PV pairing passed all nine complete-attention oracle/serial/graph cells,
but regressed W4/W5. W6/context4102 improved9–10%. Reject broad promotion;
screen only W6/G16/feature-fast V+scale at4096<=context<8192, retaining the exact
old route elsewhere. Paired resources118VGPR/48SGPR/0LDS/0scratch versus111VGPR.
`pv_boundary_screen.hip` tests2048,4095,4096,4097,8191,8192 on both saved/current
cores. No crossover is admitted from the single initial4K result alone.

## Selected operator evidence

Gate/up: all four canonical public cells T15/18/20/24 pass original-input and
represented-A8 FP64 bounds, exact generic/codec/eager/graph, poison/stale/zero/scaled
fixtures and guards/immutability;12 deliberately corrupted outputs rejected.
T18/20/24 cold complete-Op medians improve .459963/.462223/.461623ms to
.315501/.323281/.331122ms (31.41/30.06/28.27%). Final emitted bodies are identical
to the qualified/timed candidate:108/103/100VGPR,26SGPR, native IU4, no LDS/scratch.
The temporary pilot duplicate source was removed; production owns the selected
`a8q4_paired_tiles.h` implementation. Unqualified widths/shapes retain old dispatch.

PV: selected W6/4096..8191 route passes public/profile FP64, exact serial/graph and
guards at2048,4095,4096,4097,8191,8192; canonical short/long tests also pass invalid
page metadata. Cold complete-attention improvements at4096/4097/8191 are
11.95/11.69/9.11%; warm improvements7.07/7.83/2.46%. Unchanged outside-range
controls fluctuate but are not claimed as wins. Original PV remains111VGPR and
paired PV118VGPR; both occupancy12, zero LDS/scratch. Static checks cover both.

GDN: final square materialization eliminates normalization FMAC contraction.
Resources60VGPR/52SGPR/1024LDS/0scratch, with20→3 emitted barrier signal/wait pairs.
Six C1/C4 W4/5/6 cells and W2/W16 boundaries pass public FP64 outputs/snapshot
state/records, exact snapshot/graph and preserved input state. Actual replay-fold
passes ordinary-state-exact at B2/W6, max_state_abs3.89485e-9. Warm Op savings9–15%;
cold C4 is mixed (-2.13% to+2.13%). Isolated whole tests resolve the product question:
with gate/up/PV identical, GDN raises C1K5 104.78834→105.15468tok/s and C4K4
202.72736→203.51040 aggregate. Every selected repetition exceeds its control range,
exact ordinary tokens throughout. Do not attribute the initial contracted version's
larger gains to this final implementation.

## Final GDN selection correction

Fresh C2K4 control145.91585 versus candidate145.37681 and repeat145.31439 shows
a small regression. C3K4 control191.02224 versus candidate190.39311 and repeat
190.82436 shows no benefit. Retain exact-tree only at actual batch1/4; batch2/3
keeps the incumbent. This is an execution-geometry choice, not startup capacity
or caller identity. Distinct compiled bodies preserve each route's allocation.
Both instruction/resource streams match their qualified controls exactly (ignore
objdump's non-instruction zero-padding ellipsis). Tree/alternate-profile behavior
is unchanged. All12 B1..4 W4/5/6 public oracle/record/graph cells and W2/W16
boundaries pass in `gdn-batch-selected.json`; actual replay-fold passes again.

The frozen all-candidates binary retains the earlier broad-GDN measurements;
they are not the final concurrent dispatch. CLI/server/PPL/bench are rebuilt with
the batch-specific selection. C1's unchanged instruction body retains its whole
measurement; concurrent modes and cold transitions are remeasured. Four focused
host and two static tests pass. `summarize.py` records final closure after all
selected reports exist, retaining the rejected broad-GDN/broad-PV evidence.

## Final closure

`final-summary.json` joins the qualified routes and final batch-specific dispatch.
Best measured P4096/G128 modes: C1K5 105.15, C2K5 149.40, C3K4 191.05,
C4K4 203.26 aggregate tok/s. C4K5 is186.88; adaptive maxK5 is201.04.
C3K5 improves157.525→178.293 against a fresh control. C2/C3 K4 remain effectively
unchanged against fresh controls; C4K4 improves11.75% against the retained previous
pass181.89 baseline, not a fresh same-pass comparison. All24 final repetitions
and44 final batch-selected cold cases match ordinary tokens; fixed acceptance
and round counts are unchanged. Cold cases observe9 padded and4 zero-extent
transitions; no six-pending K4 transition was observed. No weights changed.

Reproduce from repository root with explicit Python3.11 and serial resource scopes
described above. `run_cell.py --label <fresh-label> --concurrency <1..4> --draft
<4|5>` runs the selected build; add `--adaptive` only with draft5. Saved per-cell
`command.json` contains the complete artifact/corpus/workload invocation. Build
`tail-check-selected` with `build_tail.py tail-check-selected`, then run
`verify_cold_tails.py` in a fresh evidence copy (outputs are create-only).
`build_screen.py profiles/bench/r9700-remaining-candidates-20260924/gdn_screen.hip
<fresh-label>` builds the12-cell final GDN screen;
the gate/up and PV sources use the same builder. `summarize.py` validates the
retained canonical result set, also create-only. Ordinary decode and prefill were
not remeasured. Older broad-GDN results are retained as rejected-scope evidence.
