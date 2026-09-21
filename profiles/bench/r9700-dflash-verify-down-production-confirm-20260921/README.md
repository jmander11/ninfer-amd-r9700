# Production DFlash verify-down confirmation

This fresh package confirms promotion of the admitted scale-gather route, not a new
mechanism sweep. The production source is src/ops/r9700/linear/dflash_verify_down.hip.
Only target-owned C1 base-Text DFlash Verify A8/Q4 T5/T6 N5120/K17408 selects it.
Production has no scale-gather or split-K selector and no partial-sum workspace.

From the repository root, CPU-only preparation and preflight:

```
bash profiles/bench/r9700-dflash-verify-down-production-confirm-20260921/commands.sh --prepare
bash profiles/bench/r9700-dflash-verify-down-production-confirm-20260921/commands.sh --preflight
```

After independent review and exclusive R9700 access on auto power:

```
bash profiles/bench/r9700-dflash-verify-down-production-confirm-20260921/commands.sh --measure
```

No argument means CPU preflight. The plan binds current sources/build/archive,
both executables, exact artifact/corpus, and admitted direct/whole evidence.
CPU preflight runs host routing, malformed public-Op, report and analyzer checks;
it binds the renamed kernel and codec/incumbent ISA/resources embedded in both
the benchmark and the archive-linked operator qualifier. No HIP launch or rebuild.

Measurement first runs the archive-linked complete public Op on both widths:
independent decoded-Q4 FP64 oracle, exact A8/status and incumbent BF16, three
allocations, guards/immutability, malformed rejection, poison/stale/finite graph
replay and balanced cold complete-boundary timing. A valid direct timing loser
closes completed/rejected without launching whole inference.

Then ordinary, K4/W5 and K5/W6 use C1 whole P128/D64, Device Graph, three reps and
one warmup. All 65 public tokens and speculative accounting must exactly match
admitted evidence; workspace capacity and graph allowance must hold. Decode and
total median time must be <=1.02 times the retained admitted candidate and <=0.99
times the retained control. This is a production confirmation against historical
matched measurements, not a fresh randomized whole A/B or a new throughput claim.

First functional failure stops. Valid timing losers close completed/rejected;
all receipts, summary, closure and checksums are create-only in results/. Never
overwrite or rerun a closed attempt. Sealed qualification/A-B evidence is unchanged.
