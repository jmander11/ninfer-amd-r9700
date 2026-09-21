# Projected-residual T1 report-serialization retry

This retry preserves the original 20260919 package and the first 20260920 retry,
including both failed attempts. The first retry completed the numerical/timing work
but emitted literal newlines in its embedded compile receipt, making its report
invalid JSON. The shared report-string serializer now escapes every U+0000–U+001F
control character, quotes and backslashes. No kernel, stream-ownership, oracle,
shape, sample-count or threshold change is part of this retry.

The plan binds both prior packages/attempts, current production and qualifier
sources, and all retry2 scripts. Disposable Python bytecode caches are excluded
from the first retry inventory. The unchanged admission gate requires exact residual
parity, the FP64 criterion, twelve samples per arm over three allocations,
per-allocation ratios at most 1.01, and at least 0.2 ms/token aggregate savings.

Preflight compiles the actual qualifier and uses its CPU-only string-serialization
mode to round-trip all 32 JSON-required controls, quotes, backslashes, UTF-8 and
multiline compile-receipt-like text through a strict JSON parser. The measurement
runner repeats that regression before invoking any GPU work, retaining its process
receipt, input and output. Preflight and preparation do not initialize HIP.

From the repository root:

```
bash profiles/bench/r9700-a8q4-projected-residual-t1-design-retry2-20260920/commands.sh --prepare
bash profiles/bench/r9700-a8q4-projected-residual-t1-design-retry2-20260920/commands.sh --preflight
```

After independent review and ledger authorization, the future GPU command is:

```
bash profiles/bench/r9700-a8q4-projected-residual-t1-design-retry2-20260920/commands.sh --measure
```

The create-only attempt-1 retains compiler, static-check, serializer and qualifier
stdout, stderr and timestamp/exit receipts. Success, ordinary failure and catchable
interruption seal closure.json and result.sha256; SIGKILL or power loss can leave an
incomplete attempt. Existing plans and attempts cannot be overwritten. A passing
direct result authorizes only whole C1 A/B preparation, never production routing.
