# NInfer documentation

Start with the project `README.md` for the current Radeon AI PRO R9700 build, artifact status, and run commands. Each executable’s `--help` lists its options and defaults.

## User guides

| Document | Purpose |
|---|---|
| `model-artifact.md` | selected artifact, availability and checksum |
| `acknowledgements.md` | upstream credits, model sources and licenses |
| `cli.md` | text, structured messages, media, sampling, MTP/DFlash2, context, and memory |
| `serving.md` | OpenAI/Anthropic HTTP schemas, streaming, state, concurrency, and logging |
| `performance.md` | R9700 benchmark methods and results |
| `containers.md` | explicit ROCm/Python build contexts, dedicated R9700 containers, test runner and app deployment |
| `../examples/cli/` | committed input examples |

## Maintainer references

Runtime and numerical behavior:

- `maintainer/concurrent-inference-architecture.md`: request/lane lifecycle, scheduling, and Device Graphs.
- `maintainer/paged-kv-cache.md`: typed FP8-K/INT4-V cache, publication, capacity, and RAM spill.
- `maintainer/softmax-attention.md`: Text/MTP and DFlash2 attention ownership.
- `maintainer/replayssm-gdn.md`: Gated DeltaNet mathematics and state transitions.
- `maintainer/op-development.md`: semantic Op admission and independent-oracle rules.
- `maintainer/kernel-iteration.md`: gfx1201 optimization and profiling procedure.
- `maintainer/gfx1201-low-precision-operations.md`: native INT4/INT8/BF16/FP8 matrix operations, ROCm BLAS support boundaries, and the current source-to-ISA map.

Artifacts and model implementation:

- `maintainer/artifact-container.md`: generic `.ninfer` framing and binding.
- `maintainer/tensor-formats.md`: persistent numerical formats.
- `maintainer/storage-layouts.md`: registered physical layouts.
- `maintainer/qwen3.8-27b-model.md`: shared Qwen3 family mathematics used by Qwen3.8-27B.
- `maintainer/r9700-integer-artifact-candidate.md`: provisional/evaluation integer artifacts and final recipe-selection contract.
- `maintainer/qwen3.8-27b-artifact.md`: sole target inventory, conversion, and binding contract.

## Maintainer workflow and evidence

- `maintainer/local-development.md`: private host paths, GPU coordination and test prerequisites.
- `../plans/r9700-autonomous-todos.md`: maintainer work ledger, separate from installation.
- `maintainer/upstream-sync.md`: upstream baseline and port dispositions.
- `maintainer/merging-to-master.md`: full AMD promotion gates for experimental-to-master merges.
- `maintainer/code-quality.md`: static gates and the gfx1201 device checks (memcheck, initcheck, racecheck).
- `../tools/bench/README.md`: physical matrix commands, DFlash selection, safe embedded-code extraction, and the selected-P2048 trace/PMC/reconciliation/roofline/static-evidence workflow.
- `../tools/r9700/README.md`: target-specific oracle, ISA/resource, and physical admission commands.

Use current references for production measurements and artifact choices; superseded backend documentation is historical.

## Historical migration record

`maintainer/r9700-overhaul-plan.md` records the migration's original decisions and acceptance background. It is not an active implementation, option, artifact, or performance authority.
