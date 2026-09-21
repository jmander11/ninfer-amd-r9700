# Selector-free normalized-linear production confirmation

This create-only package confirms the promoted normalized-linear route for the 64
all-Q4/A8 MLP gate/up boundaries in ordinary base Text C1 decode. Preparation binds
the existing production build and fresh passing selector-free public-Op qualification.
It never builds, repairs inputs, initializes HIP, or runs a workload. Preflight is
read-only: file identities, retained manifests, build dry-run, and sysfs checks.
Review the prepared plan before measurement. Neither phase creates an attempt.

Run from the repository root:

```bash
bash profiles/bench/r9700-normalized-linear-t1-production-confirmation-20260920/commands.sh --prepare --build-dir /ssdpool2nvme/local_llm/ninfer-amd-r9700/build-r9700 --qualification-attempt /ssdpool2nvme/local_llm/ninfer-amd-r9700/profiles/bench/r9700-normalized-linear-t1-promoted-qualification-20260920/attempt-1
bash profiles/bench/r9700-normalized-linear-t1-production-confirmation-20260920/commands.sh --preflight
bash profiles/bench/r9700-normalized-linear-t1-production-confirmation-20260920/commands.sh --measure
```

Three production runs use C1/P8192/G256, chunk 4096, ordinary Device Graph,
the explicitly bound Q4 artifact, fixed FP8-K/INT4-V cache, no speculation,
one warmup, and one measured repetition. All 257 public tokens must match the
retained authority. Each decode time must be at most 1.01 times the admitted
candidate median (8.735625429 s), and the production median divided by the
retained control median (8.826095962 s) must be at most 0.99. This reproduces
the admitted improvement; it is not a fresh randomized A/B claim.

The retired normalized-linear selector must be absent from the production cache,
compile commands, and benchmark config. The core archive, cache, compile commands,
and public-Op qualifier must match the fresh passing qualification. The existing
benchmark must be up to date according to a nonmutating Ninja dry-run.
The admitted historical A/B and original qualification remain immutable evidence;
their old source and binary identities need not match the promoted implementation.

Only measurement invokes the bound HIP identity helper before each inference child.
Receipts check R9700/gfx1201/wave32/PCI identity, runtime and driver versions,
auto power, low VRAM before initialization, and bounded VRAM drain afterward.
There are no sudo calls, power writes, downloads, resets, or automatic retries.
The attempt, JSON closure, partial failure receipts, and SHA-256 manifest are
create-only. Passing closes this production confirmation, not bandwidth saturation.

The retained normalized-linear whole-A/B campaign supplies read-only helpers for
identity checks, manifests, device receipts, token geometry, telemetry, and A/B
summary recomputation; its exact script is bound into the prepared plan.
