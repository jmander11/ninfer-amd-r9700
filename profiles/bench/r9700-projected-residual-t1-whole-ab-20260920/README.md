# Projected-residual T1 whole C1 A/B

This package measures the integrated projected-residual route against its selector-off
composition on the same source, using the fixed FP8-K/INT4-V cache and ordinary Device Graph.
Preparation requires an explicit **passing production public-Op qualification attempt**.
The failed production qualification attempt-1 is not an admission authority.

Run from the repository root with Python 3.11 via `commands.sh`:

```sh
bash profiles/bench/r9700-projected-residual-t1-whole-ab-20260920/commands.sh --prepare --qualification-attempt /absolute/path/to/passing/attempt
bash profiles/bench/r9700-projected-residual-t1-whole-ab-20260920/commands.sh --preflight
# Only after independent review of the sealed plan:
bash profiles/bench/r9700-projected-residual-t1-whole-ab-20260920/commands.sh --measure
```

Preparation configures/builds fresh `build-r9700-projected-residual-whole-control-20260920` and
`build-r9700-projected-residual-whole-candidate-20260920`, runs their host-only route assertions,
and creates `plan.json` exclusively. It also builds and binds the existing GDN qualifier's
`--device-info` helper without executing it. Preparation does not launch HIP. The two configurations
must differ only in the projected-residual candidate selector. Preparation can proceed
only after the production qualification passes and its bound source inputs still match.

The GPU workload is C1, P8192/G256, chunk 4096, draft tokens 0, one warmup and one
measured repetition per process. The CLI's `--spec mtp --draft-tokens 0` normalizes to
`spec=none` in the report. Three adjacent independent pairs run in this fixed order:
control/candidate, candidate/control, control/candidate. Every report must retain all
257 exact public tokens matching the accepted three-route production C1 authority.

Admission requires every candidate/control decode-time ratio below 1, mean + 2 standard
errors below 1, and median at most 0.99. Auto power is checked before and after each
process; before every benchmark, the bound `--device-info` helper must resolve HIP ordinal 0
to R9700/gfx1201/wave32 at PCI `0000:13:00.0`, the same function used for sysfs telemetry.
Its command, executable identity, stdout/stderr, parsed device identity, and process receipt
are retained in the attempt. Initial VRAM must be below 1 GiB and 5%, and post-exit drain polls for up to
30 seconds. Report device, cache, configuration, geometry, graph priming, and selector
are checked. Results are create-only under `attempt-1`; failed attempts remain intact.

A passing result sets `promotion_review_authorized=true` while keeping
`production_routing_authorized=false`; it supports a separate promotion decision only.
It does not prove physical bandwidth
saturation or absence of stalls. No broader concurrency or speculative speed claim is made.
