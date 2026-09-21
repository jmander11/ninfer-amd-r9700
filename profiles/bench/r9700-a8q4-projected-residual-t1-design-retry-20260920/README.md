# Projected-residual T1 harness retry

This retry preserves the original 20260919 package and failed attempt unchanged. The
original qualifier passed a null stream into a production Op requiring an explicit
stream; the original runner also failed to retain the qualifier's output and exit
receipt. This package corrects those harness defects only. Its plan binds every
original package/attempt file, the corrected qualifier, current production sources,
and all retry scripts. The kernel mechanism, FP64 oracle, shapes, twelve samples per
arm over three allocations, per-allocation 1.01 ratio limit, and 0.2 ms/token
aggregate admission threshold remain unchanged.

The qualifier owns one nonblocking HIP stream. Both arms, cache scrub, event timing,
warmups, input/output transfers and synchronization use that stream. Host transfers
finish before host payloads leave scope and before timing starts. `--package` binds
the actual retry package into qualification provenance.

From the repository root, prepare once and run disposable compilation/static checks:

```
bash profiles/bench/r9700-a8q4-projected-residual-t1-design-retry-20260920/commands.sh --prepare
bash profiles/bench/r9700-a8q4-projected-residual-t1-design-retry-20260920/commands.sh --preflight
```

After independent review and ledger authorization, the sole future GPU command is:

```
bash profiles/bench/r9700-a8q4-projected-residual-t1-design-retry-20260920/commands.sh --measure
```

The create-only `attempt-1` retains stdout, stderr and timestamp/exit receipts for
every compiler, static-check and qualifier process. Success, ordinary failure and
catchable interruption all produce `closure.json` and `result.sha256`; SIGKILL or
power loss necessarily leaves an incomplete attempt requiring external inspection.
An existing plan/attempt cannot be overwritten. A pass authorizes only preparation
of a whole C1 A/B, never production routing. No GPU execution is part of preparation
or preflight.
