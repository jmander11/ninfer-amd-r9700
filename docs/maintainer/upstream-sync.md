# Upstream synchronization record

Upstream is the RTX 5090/CUDA NInfer `experimental` branch. It is reconciled into this R9700
repository by custom AMD ports, not merged ancestry; `AGENTS.md` states the sync rule.

## Baseline

Reconciled through upstream `593c2d0c95ce2e34ef137af70c94d0417b37a890` (2026-10-03), advanced
from `574a8d91` (2026-10-02), `f7f70d89` (2026-10-02), `9639c32f` (2026-09-30), `34c7119b` (2026-09-28) and
`e04fad3728573a0109236929f5d473475a8657f2` (2026-09-21, AMD ports `7187d95d`, `2eab0a50`,
`49c896dd`) by the dispositions below. The next sync reviews upstream changes after `593c2d0c`
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
- Excluded: `34ed7bc6` (NVFP4 A8 gate/up scratch), the `94d97c11` rows for the NVFP4-KV
  `--sage`, `--keep-frac` and `--xattn-tau` flags (XAttention exists here only as the
  compile-time qualification build with fixed tau, not a runtime flag; Sparge is not implemented;
  Sage is rejected by the precision policy), and the NVIDIA measurement records `f33ff218`,
  `163c6b38`, `ceeac05d`.

## `9639c32f..f7f70d89` (4 upstream commits, reconciled 2026-10-02)

- Ported: `ca08451c` `/metrics` buckets, labels and closed-label series storage (`cc613dc3`;
  keeps the HIP/Device Graph identity, `ninfer_device_memory_bytes`, the `/v1/score` route and
  the AMD API code set; `reservation_exceeded` stays absent because the R9700 executor never
  publishes it). `a1d289ec` closed preserve-off turns stored cut at their turn checkpoint, per-reason
  disk drop counters under the mutex, and the shared splice checkpoint-prefix fallback
  (`db1abce1`). The cut image's current-state copies use the copy-stream host callback behind the
  image fence instead of upstream's host wait and scheduler-thread memcpy. R9700 C1 serve, 4
  two-turn chats of ~2.7K tokens: RAM 427 -> 286 MiB per entry, turn 2 appends with the resident
  checkpoint's reuse length and identical greedy tokens (DFlash k4, MTP k3, DFlash K7 adaptive
  with thinking, restored from RAM and from disk). The splice fallback is unreachable here (a
  trailing assistant turn is tokenized cold for its scoring boundary) and is kept as the one
  shared helper.
- Already equivalent from `fd16c8ba`: BF16 verify projections and the W8/Q4 heads at k=6/7 (every
  verify projection and head is one aggregate launch over all W*C columns up to 64), and the
  full-width drafter-context append (the AMD path has always appended at the storage width).
- Not ported after R9700 A/B (`docs/performance.md`, 2026-10-02): the `fd16c8ba` learned
  per-law/per-k hop-hazard picker with its 1-in-32 exploration, and the p-less draft-temperature
  scale at k=6/7. Adaptive K7 already matches the per-prompt best fixed K (K7 everywhere) at C1 and
  C4; exploration cost 0.7% at C1 against an intermittent K6 lock (one lifetime in six), and the
  scaled temperatures were within noise of 0.4. The unused DFlash `append` envelope that upstream
  widened to the verify width is removed here instead.
- Excluded: the `fd16c8ba` CUDA/NVFP4 kernel changes (W8 dynamic shared memory, NVFP4 W4A8
  M48/K256, NVFP4 GDN record, GQA T=7/8 split rules). No action: `f7f70d89` relaxes an upstream
  real test the AMD tree does not carry (the cached-versus-fresh interleavings resume check).

Evidence: `profiles/bench/r9700-upstream-sync-20261002/`.

## `f7f70d89..574a8d91` (31 upstream commits, reconciled 2026-10-02)

Upstream's code-quality campaign (compiler-warning, clang-tidy, formatter, and lint gates, then
compute-sanitizer) plus the races those tools found. The quality work reached `experimental` from
the `quality/clang-tidy-backlog` branch, so `4051ebf0..26ba4203` are not descendants of
`f7f70d89`; every non-merge commit in `f7f70d89..574a8d91` is dispositioned here.

- Ported races: `2e11477f` p-less reduction helpers (trailing barriers after every shared-slot
  result read, block-uniform inverse-CDF pick without the `red_idx[0]` round trip, register
  `scan_done` in the tile scan) and the accept-kernel half of `8dba51cc` (no `done_sh` read beside
  thread 0's write). The fork's helpers were the pre-fix upstream code; the new R9700 racecheck
  reported the same sites before the port (`sampling_device.h` slot-0 reuse, moments, residual,
  top-2 broadcast, and the compiler-hoisted `*found` load) and none after.
- Ported defects from the lint census `a9c82edd`: disk-reuse planning moved from a `const
  std::optional` (a copy of the restored host image per plan), and the eval CLI's missing
  `PackageNotFoundError` import. Its other behavioral hunks are absent here or already equivalent
  (q5 aliasing, the paged-KV scatter `__shared__` descriptor, the CRC32C `[[nodiscard]]` order,
  w8 pair planning, `tools/kdev`).
- Ported tooling with AMD equivalents (`docs/maintainer/code-quality.md`):
  - `4051ebf0`, `be07884e`, `f2e67b06`: `cmake/warnings.cmake`, warnings as errors in project C++
    (GCC 13) and HIP (ROCm clang) in place of the nvcc front-end flags; the HIP host pass does not
    report device-only helpers as unused. `NINFER_SANITIZE` instruments the host compilation of
    HIP translation units only (`-Xarch_host`), because gfx1201 device code cannot take ASan.
    The fork's warning census (97 project findings: shadowing, unhandled FP8-capped profile
    enumerators, dead fields and locals, sign compares, unchecked HIP event calls) is fixed.
  - `4051ebf0`, `fe50f67d`, `edde8e88`, `26ba4203`: `.clang-tidy` and `scripts/run-clang-tidy.py`
    on the ROCm toolchain's clang-tidy, reading the HIP compile database directly (no nvcc
    translation or builder re-exec, so `a7238ed3` has no counterpart); `bugprone-signed-bitwise`,
    new in LLVM 23 after upstream's 22.1.8 pin, is disabled.
  - `4051ebf0`, `ce01f249`: `.pre-commit-config.yaml` (clang-format selects `.hip` by extension),
    `ruff.toml`, `_typos.toml` with gfx12/ROCm vocabulary, and the auto-fixing `.githooks/pre-commit`.
  - `f8ea8cf9`, `574a8d91`: compute-sanitizer has no gfx1201 counterpart (ROCm ASan needs
    `xnack+`). `tools/r9700/gpu_check` and `scripts/gpu-check.sh` provide memcheck (guard pages),
    initcheck (poisoned allocations), and racecheck (LDS instrumentation of optimized device
    bitcode through the HIP compiler launcher) over the `gpucheck` CTest set (all 31 R9700
    qualifiers). Racecheck also found a fork-only race: `nll_from_logits_kernel` (the prefill
    NLL behind PPL and scoring) read the block maximum from `partial[0]` while wave 0 could
    already be storing its sum partial there; a barrier now separates them. After the fixes the
    set passes under memcheck, initcheck, and racecheck, and plainly (47 R9700 tests, 57 host).
  - `33cf6b17`, `07d638b4`: the builder's memory cap at three quarters of host RAM without swap,
    `--ulimit core=0`, and the update-then-restart rule.
  - `6c64f009`: the dead `.codex` clang-format hook is removed.
- Ported after the sync on 2026-10-02: `ce6ad2d5` tree-wide clang-format/ruff reformat with its
  `2695ffbe` blame entry (byte-pinned candidate qualifiers excluded), and `157161de` whole-tree
  clang-tidy as the gate: the 694 findings here are fixed or suppressed at the line, and the
  commit hook is no longer opt-in. Defects found on the way: `kv_ram_cache_qual` swallowed a
  waiter exception, and `bench-matrix` tests carried a never-executed assertion with an undefined
  name and the FP8 hybrid converter a stray expression.
- Not ported: the clang-tidy backlog fixes `9ff23c4c`, `d40226e0`, `196cc237`, `9b9b9b47`,
  `341822ba`, `1bd8f518` (style or hardening of upstream code; their defect hunks are covered
  above), `ce01f249`'s `--fast` tier, `kernel` label, and master merge gates (the fork's test
  runner is host-only by default and it has no master merge flow), and `f8ea8cf9`'s
  `--sanitizer` reduced cases (the R9700 qualifiers complete under racecheck at full scope).
  Absent here: `c1a37a5a` sparse-MoE scan race, `d5d01c3c` q5 epilogue and kdev guard, the
  `8dba51cc` small-T reduce kernel (the split-KV merges here use separate scalars and barriers),
  and the `test_attn_input_proj` sanitizer cases (`ninfer_r9700_target_variant_attention_projection_qual`
  is an empty entry point).

## `574a8d91..593c2d0c` (4 upstream commits, reconciled 2026-10-03)

- Ported: `91371f42` block-cooperative p-less tile choice. `sampling_block_choose_tile` (wave32
  `__shfl_up` chunk scan, lowest owning thread, last nonempty tile for a goal past every
  interval) serves the ordinary p-less sampler and the speculative finalize kernel, whose
  residual masses and admitted sum are now block-parallel; the per-thread auxiliary-mass buffer
  is gone. Sampling and speculative-round qualifiers and racecheck pass. R9700 C1 DFlash K7
  `--lm-head-draft`, seeded p-less T1.5, 512 tokens, 3 prompts x 2 seeds x 2 ABBA passes: decode
  91.4 -> 102.1 tok/s mean, median round 32.15 -> 29.41 ms, identical round counts in every pair.
- Already equivalent: `91371f42`'s BF16 residual and DFlash conv verify aggregation at every
  width (`packed_route_tokens` makes every verify projection one aggregate launch). Excluded: its
  CUDA bench and SmallT kernel removal.
- Not applicable: `7eb0edcc` keeps a DFlash forced-K test off the 1-in-32 exploration rounds of
  the hop-hazard picker, which the fork declined (above).
- Ported with fork deltas: `4bf04efd` KV-tier rework, upstream's port of the fork's stall fixes
  (`7885114c`, `5187be80`, `14b7ac49`) plus: RAM blocks retire on fence and last I/O pin, so a
  RAM claim never waits for a spill of the same entry; a RAM restore waits for its entry's own
  capture copies in copy-hold; chunked (4 MiB) host-copy callbacks on a tier-owned host-copy
  stream; unlocked CRC, encode, compaction publication, reap, and tombstone I/O; two-phase
  eviction; serialized MANIFEST publication; claims that cancel an idle Extend/Refresh instead of
  waiting, with a stale generation becoming a `CacheRestoreFailure` cache miss; request-local
  restore failures parked like cancelled copy-holds; per-lane cached prefix hashes; and the
  startup-pinned checkpoint-image slab (`ckpt-pin=` / `ckpt-heads=`), which removes serving-time
  pinned allocation. Kept fork code: `PagedKVPublication`, the FP8-K/INT4-V RAM/disk
  fingerprints and intra-page order, mixed rounds and `decode_waiting`, `score_many`,
  `poll_idle`, the deferred GDN fold, the closed-turn cut store, and
  `synchronize_all_while_unwinding`. The disk-cache suite takes upstream's gate-held stall cases
  in place of timing windows, and the sleep-based payload stall hook is removed.
- `593c2d0c` records a 5090 A/B; the R9700 serve check is in `docs/performance.md`.
