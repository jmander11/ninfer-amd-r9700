# Upstream synchronization record

Upstream is the RTX 5090/CUDA NInfer `experimental` branch. It is reconciled into this R9700
repository by custom AMD ports, not merged ancestry; `AGENTS.md` states the sync rule.

## Baseline

Reconciled through upstream `34c7119b00fd10b64553cdcf4c295f01ac84f605` (2026-09-28). The previous baseline
`e04fad3728573a0109236929f5d473475a8657f2` (2026-09-21, AMD ports `7187d95d`, `2eab0a50`,
`49c896dd`) was advanced by the dispositions below. The next sync reviews upstream changes after
`34c7119b` against current AMD behavior.

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

## `e04fad37..34c7119b` (47 upstream commits, reconciled 2026-09-28)

- Ported: `5d366aeb` sampling-count rollback ordering, `909ff2d7` text-part tool results,
  `565161ad` Anthropic thinking signature, `610615cd` swscale alignment, `82fb7c5f` generated
  UTF-8 repair, `1c0b8f5f` tool-schema keyword relaxation, `0fe9b7ff` + `8f0082d3` literal client
  text encoding, `dfc818ae` unread disk scratch and grammar heap trim (its tokenizer blob packing is
  not ported: the AMD tokenizer tables differ and the saving is host memory only), `5b5d3caa`
  worker and disk-thread device binding, disk tier `cb6ec221`, `8c232d01`, `931617b2`, recovery
  on resident KV `d29841e0` + `550560fe` with merge `92388a91`'s literal-span adaptation (the
  executor probe test binds a probe device instead of a HIP device; the Program-level real test is
  `ninfer_r9700_recovery_kv_qual`; the CUDA test-only stream-order fix in `test_state_store.cpp`
  and the real prefix/disk/DFlash-cancel test edits have no AMD counterpart) as `059eb9f9`,
  `9f020ed7` HIP graph-update diagnostics (`91fc4b45`), `5d6f6bf2` DFlash2 p-less draft
  temperature (`1461d0ae` + `44f14b09`; R9700 default 0.4 measured +9.9% p-less T1.5 decode),
  `a7cbe5c9` Prometheus `/metrics` (`a13ddefd`; runtime sparse-attention knobs dropped, CUDA
  Graph names become Device Graph), `67753776` RAM KV spill on admission (`d0e58c0a`, plus
  exclusion of an attempt's own captures from reclaim targets, which upstream lacks).
- Ported with R9700 measurement: `724de290` causal-conv crossover, here T>=32 rather than T>8
  (`bfbec72c`; the AMD serial/tiled crossover lies between 16 and 32 at C=10240); `f3c15618`
  headroom half only (`4b4588f9`, `--kv-capacity-headroom`, default 64 MiB). Its host-resident
  token embedding is not ported: in the Q4G64 N16K16 tile layout the PCIe row gather cost 1.0%
  C1 prefill for about 24K tokens of C4 capacity.
- Not ported: `45bef20a` (report instead of fail on the device-wide graph memory delta). R9700 GPU
  jobs are serialized by the shared GPU lock, and the startup check is the only calibration guard
  for the graph allowance here. `3e18ef63` (rewrite checkpoints in lane-owned pinned host memory):
  on the R9700 it freed 29K C4 DFlash tokens but added 18-27 ms to checkpoint-capturing and
  31-61 ms to checkpoint-restoring requests' TTFT, with no capacity gain at C1.
  `482274b9` (tool-grammar mask overlap with DFlash2 C4/C6 verify on a second stream): this
  ROCm serializes concurrent graph branches (`docs/performance.md`), so the overlap cannot occur;
  its research documents are NVIDIA-specific.
- Equivalent: `b71eebf3` (R9700-calibrated DFlash graph allowance), `c36f38c4` (host round
  preparation during the commit tail), `c7bc4cbe` (8 build jobs; `9e0795ac`), `34c7119b`
  (adaptive DFlash K locked at the cheapest k on warm requests; fixed by `a7880a27`, which credits
  unseen deeper hops with the deepest observed acceptance, measured 1.074x vs adaptive K5).
- Excluded (NVIDIA, NVFP4/TMA/tensor-core, or C > 4): `c62f7ead`, `c1da30a8`, `fda2972d`,
  `fccf6613`, `b83885e3`, `79033d70`, `4ca53273`, `6ab11dfa`, `43b49f9e`, `9b8fcce1`, `e800379e`,
  `41f331af`, `724dd314`, `91e41952`, `7485d137` (NVFP4 T6 decode attention); upstream docs
  `7e303491`, `c40469f3`, `f74915f9`.

Evidence: `profiles/bench/r9700-upstream-sync-20260928/`.
