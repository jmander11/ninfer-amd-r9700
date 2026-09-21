# Selector-free projected residual production qualification

This fresh, create-only package confirms the promoted public
`ninfer::ops::projected_residual_t1` and capacity query linked from the default
`build-r9700/src/libninfer_r9700_core.a`. The reviewed implementation removes the
candidate selector. The plan binds current source, CMake cache and compile database,
archive, public qualifier, and host route/malformed-binding qualifier. It also binds
the accepted production-qualification-retry2 result and passing whole C1 A/B result;
their sealed manifests are checked without modifying historical packages.

Preflight runs the host route and public malformed-binding qualifier with both
ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES set to -1. This protects the supported
shape/phase/inventory exclusions and rejection before HIP dispatch. It checks strict
JSON serialization, compiles production device assembly, and extracts the actual
linked gfx1201 code object with a separate extraction output. Public Op/capacity
symbols, archive ownership, native IU4 dot8, wave32, bounded registers, and zero
LDS/scratch/spills are required. Preflight uses only temporary files and creates no
attempt or GPU context.

Measurement repeats those checks and exercises N5120, K6144/K17408 directly through
the public Op: independent FP64 represented A8/Q4 oracle (at most two BF16 steps),
BF16 projection then BF16 residual-add boundary, exact incumbent residual bits,
nonfinite/status poisoning, canaries, malformed/alignment/alias rejection, and three
captured-graph replays. The stable R9700 PCI function must match HIP device 0; power
must be auto before HIP, after each correctness/graph check before timing, and after
timing. Timing retains three disjoint allocations, 80 MiB scrubs, twelve balanced
samples per arm, allocation ratios at most 1.01, and at least 0.2 ms/token aggregate
64-layer saving. These are the previously accepted direct admission criteria.

Run from the repository root after independent package review:

```bash
bash profiles/bench/r9700-a8q4-projected-residual-t1-promoted-qualification-20260920/commands.sh --preflight
bash profiles/bench/r9700-a8q4-projected-residual-t1-promoted-qualification-20260920/commands.sh --measure
```

Measurement creates attempt-1 exclusively and retains process logs, strict JSON,
closure, and SHA-256 manifest. A failure is sealed and cannot be rerun in place.
Neither preflight nor measurement rebuilds or substitutes binaries. Passing this
direct gate admits selector-free C1 production confirmation; the exact-token and
whole-speed confirmation is still required before promotion is considered closed.
The qualifier's existing `qualified_for_whole_ab_only` status means direct numerical
and timing qualification only; the package records the current confirmation scope.
