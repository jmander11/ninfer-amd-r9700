# R9700 kernel iteration (layers 0–3)

This is the required speed-work procedure for the native HIP backend on Radeon AI PRO R9700.
The physical target is ROCm 10, `gfx1201`, wave32. The executable recipes and independent-oracle
qualifiers live under `tools/r9700`; there is no compatibility control plane or host-only
classifier.

Op admission, numerical oracles, candidate ownership, and public-Op measurement follow
[`op-development.md`](op-development.md). The R9700 qualifier inventory, exact commands, selected
routes, and recorded baseline evidence are maintained in [`tools/r9700/README.md`](../../tools/r9700/README.md).

Do not select a kernel from a datasheet peak, simulator, compiler output, or profiler-intercepted
duration. Those inputs can reject an impossible idea or explain a measured result, but only an
unprofiled event timing on the selected physical R9700 can choose a route.

## Layer 0 — contract and bound hypothesis

Before writing a challenger, record the decision point:

1. public Op and exact represented inputs, formats, layouts, dimensions, and execution extent;
2. independent oracle and its exact, tolerance-based, or behavioral acceptance criterion;
3. current production route and its unprofiled physical latency at the same point;
4. minimum bytes and useful operations for the complete Op boundary;
5. the suspected limiting resource and the observation supporting it;
6. the concrete mechanism by which the challenger reduces that limit.

Classify the hypothesis as bandwidth, arithmetic/issue, occupancy/latency hiding, synchronization,
launch overhead, or cross-Op materialization traffic. Arithmetic intensity is evidence, not a
universal classifier: include codec bytes, scales, page tables, workspace traffic, and fused
epilogues actually crossed by the Op. If the proposed change does not reduce the suspected limit,
stop before implementation.

## Layer 1 — gfx1201 legality and numerical qualification

Implement a temporary challenger inside the owning R9700 Op family. Preserve wave32 semantics,
caller-owned workspace, stream ordering, and the public Tensor/Weight contract. A private cast or
reduction order is allowed only where the semantic contract permits it.

Build the qualifier that owns the changed route. Common standalone targets are:

```bash
make -C tools/r9700 build/eager_op_qual
make -C tools/r9700 build/linear_op_qual
make -C tools/r9700 build/kv_op_qual
make -C tools/r9700 build/gdn_op_qual
make -C tools/r9700 build/sampling_op_qual
```

Run the resulting executable on the selected R9700 before timing it. Use
`NINFER_R9700_PCI_BUS_ID` when the host has more than one matching device. The qualifier must
compare the production/challenger result directly with its independent exact or FP64 oracle at the
real model shapes and relevant boundary cases. Pairwise parity with another kernel is not the
oracle.

Inspect ISA only for a named legality or code-generation question:

```bash
make -C tools/r9700 isa
make -C tools/r9700 linear-isa
make -C tools/r9700 linear-resources
make -C tools/r9700 eager-isa
make -C tools/r9700 gdn-isa
make -C tools/r9700 sampling-isa
make -C tools/r9700 kv-resources
```

Relevant gfx1201 facts include wave32 execution, the emitted AMD WMMA opcode when a WMMA route is
claimed, VGPR/LDS/private-segment use, and reported occupancy. ISA presence proves code generation,
not speed or correctness.

When retained operation evidence needs the exact embedded gfx1201 code object, use the safe
`tools/bench/extract_embedded_code_object.py` owner documented in `tools/bench/README.md`. It hashes
the selected executable before and after extraction and gives `llvm-objcopy` a distinct temporary
output ELF. Never run `llvm-objcopy --dump-section SECTION=FILE INPUT_ELF` without a distinct output
ELF operand: that form rewrites `INPUT_ELF` in place.

## Layer 2 — physical candidate sweep

After the oracle passes, compare candidates at exactly the decision point. Use identical inputs,
cache state, warmup, stream, event timing, and iteration count. Keep profiler interception disabled
for selection. Report median or another declared robust statistic and enough repeated events to
distinguish the expected change from noise.

Performance-admission timing on the R9700 must fail closed unless
`/sys/class/drm/card2/device/power_dpm_force_performance_level` is exactly `auto` before device
construction. Recheck it after numerical qualification and after timing, and bind the observed
value into retained evidence. A stable non-`auto` profile may be used for a focused counter
diagnostic, but its durations are profiler controls and cannot select or admit a route.

Sweep parameters within one coherent family first: workgroup shape, waves, tiles, staging depth,
layout, vector width, fusion boundary, or a finite dispatch crossover. A temporary qualifier may
expose candidate timings, but it must not turn private candidate names into a product option.

`--timing-only` is valid only where the owning qualifier documents it and only after that identical
shape, format, layout, and route passed a full-oracle run. It removes host reference time from a
repeat sweep; it does not waive the oracle gate.

If attribution could change the implementation decision, collect one focused profile after the
unprofiled result is known. The maintained examples are:

```bash
make -C tools/r9700 profile-trace
make -C tools/r9700 profile-attention-trace
make -C tools/r9700 profile-pmc
make -C tools/r9700 rocm-tool-audit-test rocm-tool-audit
```

For a whole-inference bottleneck remaining after Pareto selection, do not substitute a raw Op
qualifier for the production workload. Use `tools/bench/prepare_whole_profile.py` against the exact
completed `pareto-whole` or `dflash-pareto` directory. Its generated rocprof command selects only
the synchronized `ninfer_bench_measured` region of one already-measured geometry. Start with
marker/kernel/memory-copy trace under `auto`; generate a separate dispatch-scoped GL2C/TCP/SQ PMC
command under temporary `profile_standard` only if the trace leaves that specific cache/activity
question; the generated command must begin in `auto` and restore and verify `auto` with an EXIT trap.
The preparation command requires the selected weights identity, KV group, compile-bound
XAttention profile, and prefill chunk, and a focused PMC additionally requires the named kernel
regex. Inspect ISA/resources only for the dispatch family named by the trace.

For the selected dense P2048 gate, follow the complete provenance-bound pipeline in
`tools/bench/README.md`: prepare the P2048 trace/PMC plan from the validated low-context authority;
validate the trace; prepare its source-derived static schedule; reconcile every dispatch; build the
roofline report; validate the separate profile-standard PMC capture; produce one strict
ISA/resource report for every executed operation family; and assemble the final selected-P2048
evidence. The reconciliation must retain modeled and explicitly uncovered dispatch duration/count
coverage. Static reports bind the profiler display symbol separately from the mangled code-object
symbol. The final assembler joins PMC only through exact recognized stage/display-symbol families,
never through dispatch IDs from another capture. Unprofiled `auto` timing remains the performance
authority; profiler durations and `profile_standard` counters are attribution-only, and unavailable
physical HBM-byte or stall counters remain explicitly unavailable.

Name the question before collection: dispatch split, wave count, occupancy, cache traffic, VALU or
LDS issue, or a specific memory transfer. Treat unavailable or zeroed ROCm counters as unavailable,
not as proof that the hardware performed no work. Give every capture a new explicit `PROFILE_DIR`,
then pass its exact database path to the offline audit rather than selecting an output by glob or
time. The audit report preserves the database hash, embedded workload command and R9700 identity,
row/counter summaries, and exact installed tool versions. It never launches HIP and is not a
performance measurement.

On the installed ROCm 10 baseline, rocprofiler-sdk 1.3.5 provides runtime/kernel/memory-copy traces
and validated nonzero `SQ_BUSY_CYCLES`/`SQ_WAVES`. Isolated generic/wave32 VALU and wave32 LDS
captures remain zero despite known matching work. The earlier known-zero TCP/GL2C result was a
collection-scope failure: dispatch-scoped collection now produces nonzero TCP request/miss and
GL2C hit/miss controls, so relative hit ratios are usable. The gfx1201 request-size buckets remain
known-zero and base event counts may undercount, so those events do not support absolute cache/HBM
byte rates. ROCm Compute Profiler 3.8.0 advertises no gfx1200/gfx1201 analysis architecture;
copying a gfx115x configuration would not make its architecture-specific formulas valid.

## Layer 3 — production selection

Promote the fastest oracle-qualified route into the public Op dispatch. Encode only a measured
finite crossover supported by the changed contract. Delete losing challengers, temporary forcing
controls, duplicate launchers, and comparison-only benchmark ownership.

Then rebuild and run the qualifier through the same public boundary linked by `ninfer_r9700_core`.
Recheck the selected route at its real shapes and boundary extents, and repeat the unprofiled timing
at the claimed scope. Use whole-Engine or serving measurements only when the requested claim is
end-to-end; an Op-level change is established at the public Op boundary.

## Decision card

A completed speed decision records:

1. Op, represented inputs, exact shape/extent, format/layout, and phase;
2. oracle and acceptance result for every promoted route;
3. baseline and challenger unprofiled physical timings;
4. bound hypothesis and how the challenger attacks it;
5. gfx1201 ISA/resource facts material to the decision;
6. focused profiler result only when it changed or explained the decision;
7. selected route/crossover and the public qualification that reaches it;
8. confirmation that losing candidates and temporary controls were removed.
