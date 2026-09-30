# Upstream synchronization record

Upstream is the RTX 5090/CUDA NInfer `experimental` branch. It is reconciled into this R9700
repository by custom AMD ports, not merged ancestry; `AGENTS.md` states the sync rule.

## Baseline

Reconciled through upstream `9639c32f32027630cd361407807fc004cd91ce13` (2026-09-30), advanced
from `34c7119b` (2026-09-28) and `e04fad3728573a0109236929f5d473475a8657f2` (2026-09-21, AMD
ports `7187d95d`, `2eab0a50`, `49c896dd`) by the dispositions below. The next sync reviews upstream changes after `9639c32f`
against current AMD behavior.

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
  headroom half (`4b4588f9`, `--kv-capacity-headroom`, default 64 MiB).
- Ported with an R9700 prompt route: `f3c15618` host-resident token embedding
  (`TensorPlacement::MappedHost`, `mapped_host_bytes` in the load summary, CLI and `server_start`).
  A direct port had kernels read every row from pinned memory; in the Q4G64 N16K16 tile layout a
  row touches one 8-byte word per 128-byte line, and that 16x PCIe read amplification cost 1.0% C1
  8K prefill (3385 -> 3351 tok/s) with decode unchanged. The port keeps the in-place gather only
  for generated tokens (device ids in captured Device Graphs: decode, verify, drafter, MTP steps).
  Prompt rows are host-staged: the host gathers a window's distinct rows into a compact image of
  the same format (`ops::stage_embedding_rows`), the load stream copies it into a fixed
  persistent region, and the existing gather kernel reads it there; the next chunk's window is
  staged and copied while the current chunk runs, so only a prompt's first chunk stages
  synchronously. The materializer fills the pinned backing from the direct-I/O slots rather than
  from the page cache. Measured with the card on a chipset PCIe x4 link: greedy tokens
  identical (DFlash, MTP, 8K multi-chunk, vision); C1 8K prefill at chunk 2048 3389 -> 3385 tok/s
  (noise), at chunk 4096 3257 -> 3243 (-0.4%, the synchronous first chunk is half the prompt);
  decode unchanged; C4 DFlash auto capacity 525632 -> 550528 tokens.
- Not ported: `45bef20a` (report instead of fail on the device-wide graph memory delta). R9700 GPU
  jobs are serialized by the shared GPU lock, and the startup check is the only calibration guard
  for the graph allowance here.
  `482274b9` (tool-grammar mask overlap with DFlash2 C4/C6 verify on a second stream): this
  ROCm serializes concurrent graph branches (`docs/performance.md`), so the overlap cannot occur;
  its research documents are NVIDIA-specific.
- Ported with an R9700 copy path: `3e18ef63` (rewrite checkpoints in lane-owned pinned host
  memory; its graph-allowance change is not ported, the R9700-calibrated constants apply). A first
  direct port added 18-27 ms to checkpoint-capturing and 31-61 ms to checkpoint-restoring requests'
  TTFT: its GDN images moved with `hipMemcpy2DAsync`, whose ROCm host rect path pins the host range
  per call and copies with a shader or line-staged transfers. The port copies one contiguous range
  per layer and component, captures through the staging slot (device snapshot, then D2H on
  `copy_stream` behind later work), and restores from staging on device while it still holds the
  lane's image. Measured with the card on a chipset PCIe Gen4 x4 link, versus the pre-port build:
  checkpoint-capturing TTFT 149 -> 112-121 ms (DFlash) and 131 -> 105-111 ms (MTP), staging-hit
  restores unchanged, staging-miss restores +7-8 ms; C4 DFlash auto capacity 496192 -> 525632
  tokens.
- Equivalent: `b71eebf3` (R9700-calibrated DFlash graph allowance), `c36f38c4` (host round
  preparation during the commit tail), `c7bc4cbe` (8 build jobs; `9e0795ac`), `34c7119b`
  (adaptive DFlash K locked at the cheapest k on warm requests; fixed by `a7880a27`, which credits
  unseen deeper hops with the deepest observed acceptance, measured 1.074x vs adaptive K5).
- Excluded (NVIDIA, NVFP4/TMA/tensor-core, or C > 4): `c62f7ead`, `c1da30a8`, `fda2972d`,
  `fccf6613`, `b83885e3`, `79033d70`, `4ca53273`, `6ab11dfa`, `43b49f9e`, `9b8fcce1`, `e800379e`,
  `41f331af`, `724dd314`, `91e41952`, `7485d137` (NVFP4 T6 decode attention); upstream docs
  `7e303491`, `c40469f3`, `f74915f9`.

Evidence: `profiles/bench/r9700-upstream-sync-20260928/`.

## `34c7119b..9639c32f` (14 upstream commits, reconciled 2026-09-30)

- Ported: `42d7fa75` multi-key NIAH (`1a6eb119`: 32K/240K/260K lengths, the 260K single-key
  fixtures, `run_niah_check --multikey`; all ten 260K fixtures regenerate byte-exact), `8aa4ce07`
  preserve-thinking replay at the generation opener (`b9dd0450`), `5342e7a9` p-less block
  verification (`d1f56ed8`; FP64 block oracle, greedy tokens unchanged, p-less accepted length
  within noise on R9700: C1 3.62 -> 3.57 tokens/round, C4 3.51 -> 3.51).
- Ported as an idea: `9639c32f` short-append attention (`2fa61af7`). Its decode-chunked small
  appends match the existing 1..8-row packed decode and 9..127-row mid-row routes; its grid.z
  key split for wider calls becomes the mid-row split for 128..1023 rows under a whole-wave cost
  model (128 rows at 131K: 9.56 -> 4.09 ms per layer). 1024-row calls fill whole waves and stay
  dense.
- Already equivalent: `c9a04602` (`d0e58c0a`, `23877a2e`), `db2b7153` (`455e45ef`), `1e4dc811`
  (`3c4b4ce6`), `9b7d1277` (`2a8b033f`), `37c63461` (the host tool-mask mailbox already runs plain
  rounds at the no-exchange ceiling), and the `94d97c11` DFlash2 draft-temperature contract.
- Excluded: `34ed7bc6` (NVFP4 A8 gate/up scratch), the `94d97c11` sparse-attention flags (no
  runtime sparse-attention knobs here), and the NVIDIA measurement records `f33ff218`, `163c6b38`,
  `ceeac05d`.
