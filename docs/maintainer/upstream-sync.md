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

## In progress: `e04fad37..931617b2` (42 upstream commits)

The baseline above advances once every commit below is ported, equivalent or dispositioned.

- Ported: `5d366aeb` sampling-count rollback ordering, `909ff2d7` text-part tool results,
  `565161ad` Anthropic thinking signature, `610615cd` swscale alignment, `82fb7c5f` generated
  UTF-8 repair, `1c0b8f5f` tool-schema keyword relaxation, `0fe9b7ff` + `8f0082d3` literal client
  text encoding, `dfc818ae` unread disk scratch and grammar heap trim (its tokenizer blob packing is
  not ported: the AMD tokenizer tables differ and the saving is host memory only), `5b5d3caa`
  worker and disk-thread device binding, disk tier `cb6ec221`, `8c232d01`, `931617b2`, recovery
  on resident KV `d29841e0` + `550560fe` with merge `92388a91`'s literal-span adaptation (the
  executor probe test binds a probe device instead of a HIP device; the Program-level real test is
  `ninfer_r9700_recovery_kv_qual`; the CUDA test-only stream-order fix in `test_state_store.cpp`
  and the real prefix/disk/DFlash-cancel test edits have no AMD counterpart).
- Not ported: `45bef20a` (report instead of fail on the device-wide graph memory delta). R9700 GPU
  jobs are serialized by the shared GPU lock, and the startup check is the only calibration guard
  for the graph allowance here.
- Equivalent: `b71eebf3` (R9700-calibrated DFlash graph allowance), `c36f38c4` (host round
  preparation during the commit tail).
- Excluded (NVIDIA, NVFP4/TMA/tensor-core, or C > 4): `c62f7ead`, `c1da30a8`, `fda2972d`,
  `fccf6613`, `b83885e3`, `79033d70`, `4ca53273`, `6ab11dfa`, `43b49f9e`, `9b8fcce1`, `e800379e`,
  `41f331af`, `724dd314`, `91e41952`; upstream docs `7e303491`, `c40469f3`, `f74915f9`.
- Pending: DFlash2 p-less draft temperature `5d6f6bf2`; Prometheus `/metrics` `a7cbe5c9`;
  graph-update diagnostics `9f020ed7`; to measure: `3e18ef63`, `f3c15618`, `724de290`.
