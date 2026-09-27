# Upstream synchronization record

Upstream is the RTX 5090/CUDA NInfer `experimental` branch. It is reconciled into this R9700
repository by custom AMD ports, not merged ancestry; `AGENTS.md` states the sync rule.

## Baseline

Reconciled through upstream `e04fad3728573a0109236929f5d473475a8657f2` (2026-09-21) by the AMD
ports `7187d95d`, `2eab0a50` (feature implementation) and `49c896dd` (qualification closure). The
next sync reviews upstream changes after this commit against current AMD behavior.

## Ported features (`c450798c..e04fad37`)

- Sampling/API: full-domain p-less and epsilon sampling with the 1/1024 probability floor,
  suppression, masks and root-only cycle exclusion, configured argmax eligibility, DFlash p-less
  one-hot proposal, per-row position offsets; ordinary and speculative. HIP uses wave32/hipCUB
  with device-scope release/acquire completion counters and caller-owned workspace.
- Host runtime: reasoning/tool generation and recovery, chain/adaptive DFlash, Vision integration,
  scheduling and telemetry.
- Storage/startup: durable SSD prefix cache, RAM recovery/ownership, parallel loading and pinned
  initialization.
- Product/tooling: CLI/server contracts, xgrammar, Docker/build/test/evaluation tools.
- Earlier bounded correctness batch: prelaunch cancellation (`c075cc38` with `64840a95`'s
  postlaunch semantics), speculative tail invalidation (`0412f177`), committed-prefix statistics
  (`b2b1b7e4`), RAM readiness/ownership (`64840a95`).

## Kernel dispositions

- DFlash selector per-sequence projection: ported (numerical isolation; packed outputs equal
  separate C1 calls).
- DFlash selector parallel merge (sixteen block-wide rank reductions): applicable optimization,
  not ported; the AMD serial exact merge keeps the same ordering, and no R9700 benefit is shown.
- GDN replay/state overlays: already covered by the AMD raw-record replay and FP32 fold.
- GDN control/projection specializations (packed T5/T6, T to 20, FP8/projected convolution):
  AMD leaves retained; their workspace and precision contracts differ.
- Chunked GDN FP16 W/U staging: not ported; AMD owns an independently qualified chunked route.
- Attention/projection (NVFP4/U8 sparse prefill, FP8/NVFP4 linear routes): not applicable to the
  fixed FP8-K/INT4-V cache and the AMD weight formats.

CUDA paths, NVFP4 and additional model targets are excluded by the product contract.
