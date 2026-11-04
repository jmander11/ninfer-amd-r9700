# R9700 maintainer backlog

This backlog is separate from public installation and release-preparation work. An item becomes active only when requested. Product and numerical requirements are in `AGENTS.md`; completed measurement results belong in `docs/performance.md`. Historical campaigns and the previous ledger remain in Git history.

## Constraints

- Source private host configuration and use the shared GPU lock as described in `docs/maintainer/local-development.md`. Builds use at most 8 jobs.
- Follow the current cache and numerical contract in `AGENTS.md`. The old blanket prohibition on FP8 query/key arithmetic was superseded by qualified production FP8-Q/K WMMA routes; it must not override the admitted implementation.
- Leave clocks, power and voltage unchanged unless the user explicitly requests that experiment. Performance claims need measurements at the claimed scope; numerical changes need the owning oracle and relevant model quality gates.

## Follow-ups awaiting a separate request

- `QUALITY-GAINS`: evaluate a larger usage calibration set against the BF16 8K/32K cells. GPTQ output-head and attention gate/value/output improvements were promoted; activation ordering was rejected. Keep the admitted 21 FP8 protections and the NVFP4 comparison bar until a measured replacement is qualified.
- `CLOCKS-README`: compare stock clocks with an explicitly supplied alternate clock and voltage profile. This remains user-gated; the published workload matrix does not qualify an alternate hardware profile.
