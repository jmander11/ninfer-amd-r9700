# R9700 C2-C4 paired-projection whole A/B

This create-only package is blocked until the direct production-symbol qualification passes.
Preparation builds exact selector-off/on binaries and binds their caches, executables, routing
symbols, source identities, device-0 PCI/power identity, and retained C2/C3/C4 public-token
authorities. Measurement runs three adjacent order-controlled pairs at each concurrency with
Device Graph enabled and speculative decode disabled.

```
bash profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919/commands.sh --prepare
bash profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919/commands.sh --preflight
bash profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919/commands.sh --measure
```

Passing requires exact retained public tokens, every candidate/control ratio below one, the mean
plus two standard errors below one, and median ratio at most 0.99 independently at C2, C3, and C4.
