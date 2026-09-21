# Selector-free normalized-linear production qualification

Fresh linked public-Op qualification after the passing direct and whole-inference
A/B authorities. Passing prepares the final selector-free whole C1 confirmation;
the package does not authorize any further routing changes.

The qualifier links the current `build-r9700/src/libninfer_r9700_core.a` with A8,
Release, gfx1201. No normalized-linear candidate selector is permitted in the
production cache or compile command. The executable must be built separately
before preparation; this package never builds it.

At T1/N34816/K5120, compare public RMSNorm then Linear with public
normalized_linear. Both complete public boundaries are captured and timed by
graph replay. Preserve the existing independent FP64 RMSNorm, explicit BF16
seam, exact A8 codec, independent signed-Q4 oracle, two-BF16-step/1e-2 relative-L2
criterion, exact codec and public-control checks, malformed/alias rejection,
immutable inputs, canaries, T2 fallback, and three poison/stale/finite graph replays.
Static checks inspect source assembly and both actual embedded gfx1201 kernels,
including the public host symbol, wave32, scratch/spill freedom, and native IU4
consumer. Preparation LDS is allowed.

Timing remains 24 balanced adjacent pairs over three disjoint allocations with
80 MiB scrub outside each timed boundary. Every allocation must win, paired
ratio mean plus two standard errors must be below one, and
64*(median control - median candidate) must save at least 0.2 ms/token.
Physical GPU 0000:13:00.0 must be AMD 1002:7551, gfx1201 wave32, auto power.
Only `--measure` invokes the GPU qualifier.

The plan binds current sources, scripts, compile commands, production archive and
executable, plus prior passing direct and whole A/B summaries, closures and
manifests. Those historical identities provide admission context; fresh results
come only from the newly linked executable. No artifact or workload is selected
implicitly.

From the repository root:

```bash
bash profiles/bench/r9700-normalized-linear-t1-promoted-qualification-20260920/commands.sh --prepare
bash profiles/bench/r9700-normalized-linear-t1-promoted-qualification-20260920/commands.sh --preflight
```

After independent implementation and package review:

```bash
bash profiles/bench/r9700-normalized-linear-t1-promoted-qualification-20260920/commands.sh --measure
```

Preparation writes only a fresh plan. Preflight compiles source assembly and
inspects linked embedded code in temporary storage; it does not initialize HIP,
alter the package, or create the retained attempt. Measurement creates
`attempt-1` exactly once, retains plan identity, commands, stdout/stderr, exit
receipts, static evidence, raw samples, independent analysis, closure and
`result.sha256`. All bound identities are rechecked around GPU execution.
Failures remain immutable; fixes need a new explicitly named package.
