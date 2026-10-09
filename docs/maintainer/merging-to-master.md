# Merging into master

`experimental` carries development and custom upstream AMD ports. `master` receives candidates
that pass the full R9700 promotion gates. This authority adapts upstream's master-merge process;
it does not merge CUDA upstream ancestry into this fork.

## Rule

Fetch this fork's `origin/master` and integrate it into the working branch before validation.
Every gate runs on the committed candidate that master will receive, on one gfx1201 R9700 with
the selected ROCm toolchain. Every gate must pass; missing prerequisites, skipped tests and tools
that could not run block promotion. Fix findings on the branch and restart the gates on the new
candidate. Never disable or weaken a check to pass. Ordinary development retains the focused
verification rules in AGENTS.md; the full suite applies specifically to promotion into master.

## Gates

Run the native host workflow with an explicit maintainer-provided production artifact:

```bash
scripts/master-gates.sh /ssdpool2nvme/local_llm/models/qwen3.8-27b-r9700-fp8lut4/qwen3.8-27b-r9700-fp8lut4.ninfer
```

| Gate | Required result |
|---|---|
| 1. Whole-tree pre-commit | All pinned formatters, linters, spelling and hygiene pass; no rewrites |
| 2. Release build | Tests, apps and benchmarks enabled, warnings as errors, no host sanitizers |
| 3. Whole-tree clang-tidy | Zero diagnostics, including HIP translation units |
| 4. All non-real CTest and retained Python contracts | Every host and physical qualifier passes; no slow-test exclusion |
| 5. Real-artifact Engine integration | Every real CTest runs, plus cache/cancellation and MTP/DFlash recovery qualifiers |
| 6. Device memcheck | Every registered gpucheck qualifier passes guarded device allocations |
| 7. Device racecheck | Every registered gpucheck qualifier passes native gfx1201 LDS instrumentation |
| 8. Device initcheck | Every registered gpucheck qualifier passes poisoned device allocations and its oracle |
| 9. Host ASan/UBSan | Separate instrumented build and every host CTest pass without sanitizer reports |

These device checks have the documented coverage and limitations in `code-quality.md`; they
are the R9700 implementation, not NVIDIA compute-sanitizer. The supported cache is always
FP8-K/INT4-V/FP16 scales. Other artifacts and CUDA/NVFP4 lanes are excluded. Artifact quality and
performance admission remain governed by their existing authorities; promotion does not invent
additional conversion or benchmark campaigns.

The selected Python 3.11 interpreter must provide pytest, Torch and safetensors. Set
`NINFER_PYTHON` to its explicit path when needed. Dependencies are not installed by this runner.
Builds use at most eight jobs; `NINFER_DEV_JOBS` may lower the count. All GPU execution holds the
shared coordination lock; device-check tools acquire it per qualifier. Coordinate resident
workloads with their owner before testing. Do not delete the lock or stop another owner's job.

## Promotion and evidence

1. Fetch origin, reconcile master into experimental if needed, and commit the candidate.
2. Push experimental, then run the gates. A hook fix or other source change requires another
   commit, push, and full gate run on that candidate.
3. Record the candidate, master base, artifact, host kernel/driver, ROCm toolchain and each gate's
   summarized result in the merge description. The runner retains commands, results and outputs
   under `profiles/master-gates/<candidate>/`; raw logs need not enter the repository.
4. Fetch master again. If it moved, integrate it and repeat validation. Otherwise merge the
   candidate into master with a real merge commit and push master.

The runner only validates. It never commits, merges, pushes or deploys. Those actions require the
user's request; a request to promote a branch authorizes its candidate and merge commits and
pushes. No hosted CI is assumed. Never fabricate upstream ancestry with an ours merge or rewrite
published history to mark upstream reconciled.
