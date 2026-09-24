# Normalized-linear T1 whole C1 A/B

This package measures the integrated MLP RMSNorm+A8 preparation fusion at 64 ordinary
base-decode boundaries against the selector-off composition. Both arms use current
production routes, the fixed FP8-K/INT4-V cache, and ordinary Device Graph execution.
Admission requires the passing direct normalized-linear qualification attempt.

Run from the repository root:

```sh
bash profiles/bench/r9700-normalized-linear-t1-whole-ab-20260920/commands.sh --prepare --qualification-attempt /ssdpool2nvme/local_llm/ninfer-amd-r9700/profiles/bench/r9700-normalized-linear-t1-qualification-20260920/attempt-1
bash profiles/bench/r9700-normalized-linear-t1-whole-ab-20260920/commands.sh --preflight
# Only after independent review of the sealed plan:
bash profiles/bench/r9700-normalized-linear-t1-whole-ab-20260920/commands.sh --measure
```

Preparation creates fresh control/candidate build directories, builds both benchmark
executables and the execution-state/report contract tests, runs both tests with GPUs
hidden, and exclusively creates plan.json. The two caches must differ only in
NINFER_R9700_NORMALIZED_LINEAR_T1_CANDIDATE=0/1. Other candidate routes are disabled.
The control build also provides the existing GDN qualifier's device-info helper;
preparation and preflight never execute it or launch HIP. Preflight only validates
bound files, retained evidence, and read-only sysfs state; it creates no attempt.

The sole GPU entry is --measure. Workload: C1, P8192/G256, chunk 4096, draft tokens 0,
one warmup and one measured repetition per process. The CLI's --spec mtp with zero
draft tokens reports spec=none. All 257 public tokens must exactly match the retained
selector-free projected-residual production confirmation run-1. Three adjacent pairs
use control/candidate, candidate/control, control/candidate ordering.

Admission requires every candidate/control decode-time ratio below 1, mean plus two
standard errors below 1, and median at most 0.99. Reports must identify the corresponding
selector as false/true, fixed cache, graph priming, and expected geometry. Before every
benchmark the bound helper must identify ordinal 0 as R9700/gfx1201/wave32 at PCI
0000:13:00.0, matching sysfs. Power must be auto before and after processes. Initial
VRAM must be at most 1 GiB and 5%; post-exit drain polls for up to 30 seconds.

Results are create-only in attempt-1, with process receipts, stdout/stderr, closure,
and complete SHA-256 manifest, including failures. Passing authorizes promotion review,
not production routing. This experiment measures base-decode latency; it does not claim
physical bandwidth saturation or speculative/concurrent performance.
