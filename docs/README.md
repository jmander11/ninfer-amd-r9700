# NInfer documentation

Start with the project `README.md` for the current Radeon AI PRO R9700 build, artifact status, and
product commands. Executable `--help` output is the exact authority for option spelling/defaults.

## User guides

| Document | Purpose |
|---|---|
| `cli.md` | text, structured messages, media, sampling, MTP/DFlash2, context, and memory |
| `serving.md` | OpenAI/Anthropic HTTP schemas, streaming, state, concurrency, and logging |
| `performance.md` | R9700 measurement policy and admitted results |
| `../examples/cli/` | committed input examples |

## Active maintainer authorities

Runtime and numerical ownership:

- `maintainer/concurrent-inference-architecture.md`: request/lane lifecycle, scheduling, and Device Graphs.
- `maintainer/paged-kv-cache.md`: typed FP8-K/INT4-V cache, publication, capacity, and RAM spill.
- `maintainer/softmax-attention.md`: Text/MTP and DFlash2 attention ownership.
- `maintainer/replayssm-gdn.md`: Gated DeltaNet mathematics and state transitions.
- `maintainer/op-development.md`: semantic Op admission and independent-oracle rules.
- `maintainer/kernel-iteration.md`: gfx1201 optimization and profiling procedure.
- `maintainer/gfx1201-low-precision-operations.md`: native INT4/INT8/BF16/FP8 matrix operations,
  ROCm BLAS support boundaries, and the current source-to-ISA map.

Artifact and target ownership:

- `maintainer/artifact-container.md`: generic `.ninfer` framing and binding.
- `maintainer/tensor-formats.md`: persistent numerical formats.
- `maintainer/storage-layouts.md`: registered physical layouts.
- `maintainer/qwen3.8-27b-model.md`: shared Qwen3 family mathematics used by Qwen3.8-27B.
- `maintainer/r9700-integer-artifact-candidate.md`: provisional/evaluation integer artifacts and
  final recipe-selection contract.
- `maintainer/qwen3.8-27b-artifact.md`: sole target inventory, conversion, and binding contract.

## Live execution state

- `../plans/r9700-autonomous-todos.md`: authoritative completion ledger.
- `../tools/bench/README.md`: physical matrix commands, DFlash selection, safe embedded-code
  extraction, and the selected-P2048 trace/PMC/reconciliation/roofline/static-evidence workflow.
- `../tools/r9700/README.md`: target-specific oracle, ISA/resource, and physical admission commands.

Superseded backend documentation is not an active product contract and must not supply production
measurements or artifact decisions.

## Historical migration record

`maintainer/r9700-overhaul-plan.md` records the migration's original decisions and acceptance
background. It is not an active implementation, option, artifact, or performance authority.
