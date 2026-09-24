# GDN complete projection/control grid qualification

Temporary challenger only; no product routing or production archive changes.
Build with `make -C tools/r9700 build/gdn_projection_control_grid_qual gdn-projection-control-grid-static`.
Then `bash profiles/bench/r9700-gdn-projection-control-grid-qualification-20260920/commands.sh --prepare`
creates the immutable plan. `--preflight` performs only CPU/static inspection in temporary storage;
it never launches HIP or creates the retained attempt. Independent review precedes `--measure`.

Both arms accept represented BF16 hidden[5120], exact Q4N16K16/G64 weights for QK[4096]
and value-Z[12288], and BF16 control matrices [48,5120]. Outputs are both BF16 projections
and FP32 g/beta[48]. The candidate freshly resets/prepares exact A8G64 scratch, then runs
the reviewed 112x256 grid. Control runs the existing BF16 projected-control Op followed
by fresh A8 preparation and the existing paired-Q4 projection, on the same explicit stream.

Projection oracle independently decodes every stored signed nibble and FP16 scale, retains
the FP64 dot, and requires relative L2 <=0.01 and at most two BF16 steps. A8 codec/status
are exact. Control oracle independently accumulates both represented BF16 dot products in
FP64, applies the established FP32-to-BF16 a/b oracle seams, and evaluates gating in FP64;
the established pointwise bounds are 0.0625 for g and 0.0003 for beta. Both arms additionally
require exact incumbent parity.
Three address-distinct guarded weight allocations, immutable inputs, malformed inputs,
and complete captured poison/stale/finite replay are checked before timing.

Timing is 24 balanced pairs of complete graph replay, three rotating weight copies, with
80 MiB cache scrub before each arm outside the timed interval. Admission requires every
allocation median to win, paired ratio mean+2SE <1, and >=0.2 ms/token saved over 48 layers.
A numerically valid loser is completed/rejected, not a failed or abandoned experiment.
The source and embedded instruction schedules/resources bind the unchanged production
consumers and combined kernel; the merged grid's scalar-to-vector load risk is resolved
only by physical timing. Passing permits whole-inference A/B preparation, never promotion.

The create-only `attempt-1` retains per-process output/exit receipts, the plan identity,
qualification, independent timing summary, closure and `result.sha256`. Existing output
is never overwritten, including failed attempts. No profiler, model artifact or network is used.
