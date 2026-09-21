# Gate-up prefetch complete-boundary qualification

Qualification-only T1 N34816/K5120 A8G64/Q4N16K16 challenger. Production stays
unchanged. This package reuses the create-only receipt/closure lifecycle from the
bound normalized-linear promoted qualification runner. Measurement summaries,
closures and disposition banners are specific to this prefetch experiment.

Both arms capture the public production `normalized_linear` boundary. The candidate
replaces the terminal projection with the complete seven-argument parameter record
from a separately captured one-node challenger graph. The qualifier requires the
original two-kernel, one-edge topology, exact geometry, exact operand addresses,
incumbent N/K, and unchanged preparation. Graph edits and instantiation are outside
timing; preparation and projection are inside timing in both arms.

Numerical authority: independent FP64 RMSNorm, explicit BF16 seam, exact A8G64
codec, independent complete FP64 signed-Q4 projection using stored FP16 scales.
The normwise criterion is relative L2 <= 0.01 and a gross cap of two BF16 steps;
the incumbent BF16 result must also match exactly. Four finite fixtures cover
nonzero index-varying input, an independently perturbed input, zero and tiny input
on all three weight allocations. Output/workspace guards, input/weight immutability,
public malformed/alias rejections, private null-operand/stream rejection, and
NaN/Inf/stale-status recovery run before timing. The private fixed-shape launcher
requires complete aligned nonoverlapping planes; it has no extent/alias validation
contract beyond its explicit null checks.

Timing: unprofiled R9700/gfx1201/wave32 in `auto`, three disjoint 90 MiB weight
copies, 80 MiB cache scrub before each event, 24 total paired samples (eight per
allocation) with balanced arm order per allocation. Admit only when every allocation wins, paired ratio mean+2SE
< 1 and 64*(median control - median candidate) >= 0.2 ms/token. This admits only
independent result review and whole C1 A/B preparation, never production promotion.

Static admission reuses the reviewed schedule checker (four successor B64 loads,
16 current IU4 operations, serial FP32 FMA before draining successors, two loop
generations and terminal dependencies). It assembles the checked source assembly
and requires its exact instruction bytes to equal the consumer embedded in the
measured binary. Resources remain wave32, occupancy16, <=32 VGPR/<=64 SGPR,
zero scratch/spills/LDS. The linked incumbent/preparation are also inspected.

From the repository root, build then prepare:

```sh
cmake --build build-r9700 --target ninfer_r9700_a8q4_gate_up_prefetch_qual -j 2
bash profiles/bench/r9700-gate-up-prefetch-qualification-20260920/commands.sh --prepare
bash profiles/bench/r9700-gate-up-prefetch-qualification-20260920/commands.sh --preflight
```

`--preflight` compiles/disassembles in temporary storage only, never initializes HIP
and never creates `attempt-1`. After independent package review and exclusive GPU
availability, the assigned GPU owner may run:

```sh
bash profiles/bench/r9700-gate-up-prefetch-qualification-20260920/commands.sh --measure
```

`attempt-1` is create-only. Source/build/package changes fail identity checks.
Every retained subprocess gets stdout/stderr and a process receipt. A valid analyzed
measurement writes its summary and a `completed` closure with explicit `passed` or
`rejected` disposition and exits zero. A rejected timing is a completed experiment,
not an operational failure. Operational errors write a `failed` closure and exit
nonzero. All outcomes get a SHA256 manifest. Attempts are never overwritten.
