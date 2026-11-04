# Merging into master

Develop and port upstream changes on `experimental`. Merge into `master` only after the candidate passes all R9700 promotion gates. Port upstream changes without merging CUDA ancestry.

## Rule

Fetch this fork's `origin/master` and integrate it into the working branch before validation. Every gate runs on the committed candidate that master will receive, on one gfx1201 R9700 with the selected ROCm toolchain. Every gate must pass; missing prerequisites, skipped tests and tools that could not run block promotion. Fix findings on the branch and restart the gates on the new candidate. Never disable or weaken a check to pass. Ordinary development retains the focused verification rules in AGENTS.md; the full suite applies specifically to promotion into master.

## Gates

Run the native host workflow with an explicit maintainer-provided production artifact:

```bash
scripts/master-gates.sh /path/to/models/qwen3.8-27b-r9700-lq4/qwen3.8-27b-r9700-lq4.ninfer
```

| Gate | Required result |
|---|---|
| 1. Whole-tree pre-commit | All pinned formatters, linters, spelling and hygiene pass; no rewrites |
| 2. Release build | Tests, apps and benchmarks enabled, warnings as errors, no host sanitizers |
| 3. Whole-tree clang-tidy | Zero diagnostics, including HIP translation units |
| 4. All non-real CTest and retained Python contracts | Every host and physical qualifier and Python codec check passes; no skips or slow-test exclusion |
| 5. Real-artifact Engine integration | Every real CTest runs, plus cache/cancellation and MTP/DFlash recovery qualifiers |
| 6. Device memcheck | Every registered gpucheck qualifier passes guarded device allocations |
| 7. Device racecheck | Every registered gpucheck qualifier passes native gfx1201 LDS instrumentation |
| 8. Device initcheck | Every registered gpucheck qualifier passes poisoned device allocations and its oracle |
| 9. Host ASan/UBSan | Separate instrumented build and every host CTest pass without sanitizer reports |

See `code-quality.md` for device-check coverage and limitations. These are R9700 checks; NVIDIA compute-sanitizer does not run here. Use the production artifact and fixed FP8-K/INT4-V/FP16-scale cache. Existing artifact-quality and performance requirements still apply; promotion adds no conversion or benchmark campaigns.

The selected Python 3.11 interpreter must provide pytest, Torch and safetensors. Set `NINFER_PYTHON` to its explicit path when needed. Dependencies are not installed by this runner. Builds use at most eight jobs; `NINFER_DEV_JOBS` may lower the count. All GPU execution holds the shared coordination lock; device-check tools acquire it per qualifier. Coordinate resident workloads with their owner before testing. Do not delete the lock or stop another owner's job.

## Promotion and evidence

1. Fetch origin, reconcile master into experimental if needed, and commit the candidate.
2. Push experimental, then run the gates. A hook fix or other source change requires another commit, push, and full gate run on that candidate.
3. Record the candidate, master base, artifact, host kernel/driver, ROCm toolchain and each gate's summarized result in the merge description. The runner retains commands, results and outputs under `profiles/master-gates/<candidate>/`; raw logs need not enter the repository.
4. Fetch master again. If it moved, integrate it and repeat validation. Otherwise merge the candidate into master with a real merge commit and push master.

The runner validates without committing, merging, pushing or deploying. Those actions require the user's request; a promotion request authorizes the candidate and merge commits and pushes. Run the gates locally; hosted CI is not assumed. Never use an ours merge or rewrite published history to claim upstream was merged.
