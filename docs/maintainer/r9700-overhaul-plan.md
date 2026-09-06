# Radeon AI PRO R9700 ROCm overhaul plan

Status: active migration plan, recorded 2026-09-01. Remove this file after the selected design is
fully integrated into the active architecture, artifact, kernel-development, build, and product
references.

## Deliverable and decision

Replace the RTX 5090/CUDA implementation with one from-scratch Radeon AI PRO R9700 implementation.
The delivered engine remains a single-GPU, single-resident-model, startup-fixed C=1..4 Qwen3.8-27B
product. It is compiled only for RDNA 4 `gfx1201` and uses ROCm/HIP throughout. CUDA is removed;
Vulkan is not a second product backend. NVFP4 is removed from the artifact registry, target
identity, runtime state, kernels, tools, tests, and active documentation; it is not retained as a
bring-up representation or compatibility lane.

This is an execution-substrate and kernel redesign, not a HIP syntax conversion. Generic `.ninfer`
framing, represented tensor semantics, model mathematics, request scheduling, tokenizer/media
semantics, and serving protocols remain where their contracts do not name CUDA. Device ownership,
memory, streams, events, graphs, every tuned kernel, kernel tooling, performance evidence, and the
final low-bit storage profile are re-established for the R9700.

The implementation decisions are:

- Keep available subagent slots occupied with useful, bounded, independent CPU/static work while
  serialized GPU work or long builds are pending; the primary owner coordinates shared-file and
  GPU ownership. Do not create low-value work solely to occupy a slot.
- Never schedule or require a product, benchmark, or profiling cell above C=4. Active evidence is
  exactly C=1..4; any retained C=5..8 result is historical only.
- ROCm/HIP is the sole production programming model.
- Compile specifically with `--offload-arch=gfx1201`; do not add runtime GPU discovery or a generic
  multi-backend layer.
- Treat wave32 as the default execution unit. Qualify wave64 only for an individual kernel where a
  real measurement justifies it.
- Use direct owned HIP kernels and gfx12 compiler builtins for production. Libraries may serve as
  bring-up or measurement references but do not own the model schedule or final fixed-shape paths.
- Use Vulkan only for an optional, disposable control measurement. llama.cpp results do not select
  NInfer's backend.
- Do not transcode NVIDIA NVFP4 into the R9700 artifact. Conversion starts from the original BF16
  Qwen3.8 and DFlash2 checkpoints, selects one conversion-time R9700-native integer weight recipe,
  and never repacks live weights.
- The production KV-cache candidate is direct E4M3 FP8 keys and signed INT4 values. K has no scale
  plane. V uses one FP16 scale per contiguous 16 or 32 head-dimension values; the real R9700
  numerical and attention-path gate selects one group size, with G32 the capacity/speed lead and
  G16 the accuracy challenger. BF16 K/V exists only as the explicit numerical-reference profile.
- Bring up eager execution first. The delivered runtime regains fixed-address device graphs and
  does not preserve eager mode as an advertised compatibility product lane.

## Fixed R9700 hardware facts

| Property | Radeon AI PRO R9700 target |
|---|---|
| GPU architecture | RDNA 4, Navi 48 |
| LLVM/HIP target | `gfx1201` (`amdgpu12.01`) |
| Compute units / stream processors | 64 / 4,096 |
| AI accelerators | 128 |
| Native wavefront | wave32; wave64 is optional |
| VRAM | 32 GB GDDR6; this host reports 32,624 MiB |
| Memory interface / advertised bandwidth | 256-bit / 640 GB/s |
| Infinity Cache / L2 | 64 MiB / 8 MiB |
| LDS | 128 KiB per CU |
| Vector/scalar register files | 768 KiB VGPR / 32 KiB SGPR per CU |
| Other per-CU caches | 32 KiB L0 vector, 16 KiB scalar, 32 KiB instruction |
| Boost clock / board power | up to 2.92 GHz / 300 W |
| Host interface | PCIe 5.0 x16 |
| FP32 vector | 47.8 TFLOP/s dense |
| FP16/BF16 matrix | about 191 TFLOP/s dense |
| FP8 matrix / INT8 matrix | about 383 TFLOP/s / 383 TOPS dense |
| INT4 matrix | about 766 TOPS dense |

Theoretical peaks are classification inputs, not performance claims. Decode is expected to be
primarily weight-bandwidth-bound: compact persistent weights, fused decode/scale/epilogues, and
elimination of intermediate traffic have priority. Prefill and wider speculative work can use
gfx12 matrix throughput.

RDNA 4 gfx12 WMMA is not fragment-layout-compatible with NVIDIA MMA or RDNA 3 WMMA. The initial
matrix target is the gfx12 wave32 16x16 family. Every fragment layout, lane mapping, shuffle,
ballot, LDS bank layout, launch geometry, register budget, and reduction must be rederived and
qualified. Relevant Clang builtins include FP16/BF16, FP8/BF8, INT8, and signed/unsigned INT4
WMMA forms. Published INT4 support does not imply native NVIDIA E2M1 FP4 support.

## Backend conclusion

HIP matches this engine's ownership model: target-specific compilation, direct gfx12 WMMA
builtins, streams/events, asynchronous memory management, graph capture and replay, inspectable
AMDGPU ISA, and ROCm profiling. Custom kernels can be captured without placing library calls in
the graph.

Vulkan cooperative-matrix shapes and formats are driver-queried, shader/compiler control is less
direct, BF16/FP8 behavior depends on extension and driver combinations, and Vulkan command-buffer
ownership would be a larger mismatch for the current runtime. RADV can be fast, but llama.cpp
measurements changed materially with ROCm WMMA support, driver versions, backend kernels, and
submission heuristics. They demonstrate that stale ROCm comparisons are invalid, not that Vulkan
is intrinsically faster.

HIP graph APIs cover capture, instantiate, upload, launch, and update, but graph safety is not
uniform across ROCm libraries. In current ROCm documentation hipBLAS/rocBLAS graph support is
stronger than hipBLASLt, while rocWMMA and Composable Kernel are not generally graph-safe. The
final graph therefore contains owned kernels unless an actual R9700 test proves a library route
safe and faster.

## R9700 artifact and KV profiles

The current Qwen3.8 NVFP4 file is not a conversion source. Its NVIDIA E2M1 weights and scale words
have already crossed a lossy representation boundary; requantizing them would compound error and
make the new integer artifact impossible to qualify against the original checkpoint. The R9700
converter consumes the BF16 Qwen3.8 and DFlash2 sources directly, streams shards with bounded host
memory, and writes the one selected physical layout. The product does no runtime repacking.

The persistent-weight gate is restricted to formats with useful native R9700 execution paths:

| Candidate | Main question |
|---|---|
| Signed INT4-G64 plus FP16 scales | Which bandwidth-dominant matrices meet the BF16-source PPL and task-quality gate? |
| Signed INT8-G32 plus FP16 scales | Which endpoints, selectors, or sensitivity-proven matrices require promotion? |
| Direct BF16/FP32 | Which norms, biases, convolution parameters, or semantic state must remain direct? |
| Private A4/A8 activation staging | At which prefill/speculative widths does native integer WMMA repay quantization and scale traffic? |

Q5/Q6 and FP8 weights are measurement candidates only if Q4/W8 cannot satisfy both capacity and
quality. They are not carried as generic production families. No candidate may create a second
artifact identity or runtime selector; the selected tensor-by-tensor recipe is one R9700 integer
profile.

The growing Text and MTP KV cache uses one explicit asymmetric semantic format with independent
typed planes; it is not represented by one shared `DType` or quantization group:

```text
K code  = direct finite OCP E4M3FN FP8, no scale plane
V code  = packed symmetric signed INT4, -8 noncanonical
V scale = binary16, one per Gv values, Gv in {16,32} during qualification
```

Within V bytes, even dimension `2i` is the low nibble and odd dimension `2i+1` is the high
nibble, using four-bit two's-complement. This canonical order governs append, copy, spill, and the
oracle; a wave-native `i/i+16` experiment may be timed but cannot silently change stored bytes.

K append applies OCP `SATFINITE_RNE` once to each finite represented post-RoPE BF16 key, preserving
signed-zero bits and rejecting nonfinite model state. Q remains a represented BF16 Op input and is
privately converted to the same OCP E4M3 profile for the native gfx12 QK compute route. QK uses
wave32 FP8xFP8-to-FP32 WMMA where that wins; stable softmax and the primary PV probabilities are
FP32; PV dequantizes packed INT4 V while loading and accumulates FP32. A raw gfx12
FP16xFP16-to-FP32 WMMA PV experiment was independently qualified at the page-boundary and
256-token corpora, then rejected and removed: it rounds both represented FP32 probabilities and
the per-token INT4-times-FP16-scale value before accumulation. There is no raw gfx12 FP32/INT4
WMMA operand form that preserves the PV contract, so the selected route is vector/fused FP32 PV.

The owning append transaction validates the authorized contiguous frontier extent, page-table
range, and plane non-overlap before launch. It can either stage checked host U32 positions once or
consume the schedule's caller-owned device I32 position panel directly. The A2 kernel requires
every device bit pattern to equal `old_frontier + input_token`, so negative, gapped, aliased, or
over-frontier panels set the invalid-position status instead of addressing storage. The shared
device status word also records nonfinite source, invalid address, or a finite V group whose
`FP16_RNE(maxabs/7)` is not finite; any status bit rejects the whole transaction. The kernel does
not silently encode an infinite V scale. The cache owner clears status on the ordered compute
stream, reads it only after that stream completes, and advances the valid frontier only after the
zero result. A failed or abandoned launched transaction preserves the old frontier and poisons
only that sequence publication: partially written bytes cannot become a future valid cache image.

The Qwen host-RAM image contract defines a versioned fingerprint with the page and D256
geometry, logical/physical capacity, layer/head count, V group, and all three independent plane
orders. It has no generic dtype escape hatch. The HIP typed-cache owner serializes it together
with the exact K/V/V-scale byte streams selected by each fragmented logical-to-physical page map.
Restore compares the semantic fingerprint first, then every physical plane and image extent,
destination allocation, host identity, and optional checkpoint image before any device byte copy;
it rejects prior NVFP4/INT8/BF16 cache images or an otherwise typed cache with a different
group/layout. The production Qwen RAM tier and its HIP event lifetime are compiled into the R9700
core. Its physical qualifier round-trips two Text layers plus one MTP layer at D256/Hkv4 across
fragmented source and destination maps, proves Text and backend semantic mismatches write no
destination byte, and exercises in-flight capture eviction.

Logical-to-physical page maps are injective cache-owner state. Construction and RAM restore reject
duplicate physical page IDs before any device restoration write. Raw vector and WMMA QK make every
score belonging to a corrupt mapping NaN, rather than leaving old output or forming a plausible
score from zero-filled operands. Every raw and typed append, compact, QK, PV, and fused-attention
boundary also rejects malformed plane-order selectors; attention boundaries require a finite,
positive scale. These are qualified failure contracts, not production recovery paths.

The first fused A3 baseline streams pages without a context-sized score buffer: a D256 block
executing as wave32 waves derives each BF16-Q/FP8-K score, updates online FP32 softmax state, and
accumulates its exact signed-INT4-times-FP16-scale V contribution in FP32. It is an
oracle-qualified correctness baseline for every independent plane layout, not the selected
performance implementation. A2 append remains separate until a whole-Op measurement demonstrates
that fusion improves the fixed workload. Its A3 contract accepts any positive U32 BF16 query-row
extent and an optional device-resident causal position for each row. The production Qwen leaf
limits T only to its 262,144-token native context capacity. Rows share one verified typed cache and
page table but may see distinct prefix lengths, which is the staged prefill-suffix shape. The raw
launcher divides T into legal gfx1201 grid-Y intervals without allocation; T=9/17/128 remain one
launch. A negative or out-of-range raw device position returns NaN for precisely that row; the
typed transaction owner
must reject it before public state can be published.

A second A3 candidate targets decode-width rows with the owned raw wave32 FP8 WMMA map. It privately
casts represented BF16 Q to E4M3, writes one reusable `[Hq,context]` FP32 score workspace, performs
parallel stable FP32 softmax in place, and feeds the existing exact INT4-times-FP16-scale FP32 PV
kernel. The workspace is reused across rows. It passes the same represented-BF16-input FP64
attention oracle: across the 1K/4K/8K/32K T=1 gates its maximum absolute output error falls from
`1.66e-4` to `3.27e-6` as context grows; the 65-token T=8 edge gate is `1.84e-3`. At the provisional
G16/token-K/feature-V/feature-scale layout it measures 0.281/0.878/1.699/6.776 ms at
1K/4K/8K/32K T=1, versus 0.762/2.617/5.189/20.792 ms for score streaming. A repeated 4K sweep
favors WMMA for T=1..2 (0.873/1.724 ms versus 2.607/2.602 ms) and score streaming from T=3
(2.620 ms versus 2.634 ms at the crossover). The production classifier additionally resolves the
short-context launch crossover: T1 uses WMMA at context>=64, T2 uses it at context>=320, and all
shorter or wider calls stream. The compile-isolated score-streaming build records the private
FP8-Q profile explicitly in PPL output.

The post-parallel-softmax audit repeated full-oracle gates at 65-token T=8, 4K T=1/2/3, and 32K
T=1, with WMMA maximum absolute errors `1.84e-3`, `4.64e-5`, `6.43e-5`, `6.44e-5`, and
`3.27e-6`. A fresh unprofiled 50-event 4K sweep measured WMMA
`0.866/1.709/2.597/3.432 ms` at T=1/2/3/4 versus score streaming
`2.575/2.605/2.596/2.608 ms`; T=3 is a measurement tie and therefore does not widen the selected
T<=2 width ceiling. A focused `rocprofv3` trace under
`profiles/r9700/rocm10-post-softmax-wmma-trace-4k` attributes 26.1 us to QK WMMA, 7.7 us to the
parallel softmax, and 818.8 us to exact vector PV on its intercepted queue. Compiler metadata
reports QK WMMA at 24 VGPR/0 LDS/0 scratch, softmax at 23 VGPR/76-byte LDS/0 scratch, and the
selected-layout PV at 15 VGPR/0 LDS/0 scratch, each with reported occupancy 16. The QK image emits
`v_wmma_f32_16x16x16_fp8_fp8`. Since softmax is below one percent of the profiled candidate while
PV dominates, no softmax launch or reduction edit is justified by this audit; profiler durations
remain attribution rather than latency-selection evidence.

For D256, the retained fused leaf reduces each score inside its eight wave32s, places eight
FP32 partials in LDS, and lets the first wave merge them. This replaces a full-block reduction
tree without changing represented inputs, public output, or FP32 softmax/PV semantics. It was
requalified against the same tail and 4K all-layout oracle gates and subsequently selected by the
whole-Op measurements recorded below.

The Qwen `DecoderState` and `PagedKVCache` contracts now have only the asymmetric typed identity;
the parallel homogeneous BF16/INT8/NVFP4 owner, its `kv_dtype`/`kv_quant_group`, paired K/V scale
planes, optional Sage K-mean plane, and `keep_frac` shape have been deleted. Text and MTP each own
one `Fp8KInt4VPagedKVSpec` with the same three required plane roles and independently recorded
physical orders. One allocation-bound transaction validates a contiguous append or monotone
compaction, uses caller-planned device position/status workspace, stages checked host positions
once or reads the existing device I32 panel without an H2D copy, clears one HIP status word,
requires every layer on the same ordered stream, and reads status before publishing its
per-sequence valid frontier. It performs no hidden allocation. A failed
or abandoned launched mutation leaves the old frontier unchanged and poisons only that sequence.
The physical state qualifier covers a two-layer cross-page host append, compaction, a successful
device-I32 suffix, positive-gap and negative-device-position rejection, publication-local poison,
old-frontier preservation, and nonfinite device rejection through this canonical owner. The Qwen
sequence planner now constructs that same typed owner from one schedule profile: G16 with
token-fastest K and feature-fastest
V/V-scale. It carries no runtime cache dtype, legacy quantization group, Sage, K-statistics, or
sparse-attention knobs. The current profile is the whole-attention timing winner and remains
subject to the paired quality/capacity/whole-inference gate; a different winner replaces the constants and semantic
fingerprint rather than adding dispatch branches. Program sequence state now carries the canonical
Text and MTP publication objects. A committed `PagedKVLayerRead` is allocation-bound and remains
valid only while the publication is healthy and closed at the captured publication generation and
frontier, and while the allocation retains the captured mapping generation, table row, and
mapped-page count. Materialize, trim, bind, and unbind advance the mapping generation, so a stale
read cannot revive even if the allocation is later rebound to its original table row. An append
transaction may issue a pending capability for one layer only after that layer's codec launch; it
is valid only on the exact ordered HIP stream and disappears when the
transaction closes. Beginning append or compaction advances the publication generation, so every
older capability is invalid; compaction never issues pending reads. The global frontier still
publishes only after every layer completes with zero device status. The remaining excluded
execution schedule must launch those transactions, pass the per-layer pending capability to
attention, and commit after the full layer traversal instead of directly advancing frontiers; no
runtime KV-dtype option or legacy cache bridge may return.

DFlash2 state remains independent from the growing-cache codec. Qwen3.8 binds its Local,
rewrite-checkpoint, and one-lane staging images as fixed BF16 cyclic caches and uses native HIP
lane copies. A physical qualifier fills all five layers, captures lane 1, clears it, restores it,
and compares the complete host image exactly. The dormant family Full-cache shape has a distinct
two-plane BF16 physical owner, so it cannot accidentally inherit the Text/MTP three-plane codec.

The physical allocator carries an independent intra-page axis order for each plane. Its
`PagedKVPlaneSpec` now owns the order rather than inheriting one pool-wide order, and spill/restore
serializes each plane using its own geometry. Qualification compares token-fastest `[P,D,N,H]` K,
`[P,D/2,N,H]` V, and `[P,D/Gv,N,H]` scales against the existing feature-fastest order before the
artifact/cache ABI is frozen. A host-side independent byte-offset oracle now covers every
G16/G32 and independent-plane-order combination at fragmented physical page coordinates; the
device route combines those same combinations with a nonidentity page table, sentinels, exact
63/64/65 append, and exact compaction. Append cost alone does not select the layout: cached QK/PV
read amplification and whole-Op latency decide it.

The HIP cutover is qualified from the bottom upward as each closed ownership boundary replaces
its legacy CUDA implementation. `src/core/device.hip` and `arena.hip` are the sole canonical
DeviceContext/timer, device-buffer/arena, and pinned-memory contracts;
`src/core/decode_graph.hip` provides graph RAII, `src/core/fp8_int4_paged_kv_cache.cpp` owns
typed physical cache storage, and the Qwen decoder-state transaction owns all-layer publication.
`tools/r9700/hip_device_qual.hip` exercises exact gfx1201/wave32 selection, ordered streams,
host-yielding events, byte-exact transfers, scoped arena alignment, and captured graph replay on
the physical R9700. The state qualifier exercises append/compaction commit-or-poison behavior at
logical positions 63/64/65, while the RAM qualifier owns exact image and fingerprint evidence.
The typed cache binds its page table and three persistent planes into a non-forgeable layer-read
capability before the fused A3 Op. A normal capability authorizes the committed frontier only while
the publication remains closed; a pending capability authorizes the newly appended frontier for
that already-launched layer on the same ordered stream without publishing it globally. Empty,
stale, cross-stream, cross-generation, remapped, poisoned, and otherwise unauthorized capabilities
are rejected before dispatch.
These canonical HIP sources are not a second runtime backend or a product compatibility layer.
The target-private Qwen3.8 leaf is now part of that closed archive: it accepts only D256/Hq24/Hkv4
`PagedKVLayerRead`, validates the live capability and mapped capacity before dispatch, and applies
the finite WMMA/score-streaming classifier above. A physical gfx1201 qualifier covers committed
T=1..9/17/128 at a fragmented 257-token frontier and same-stream pending attention against an
independent FP64 formula. At that frontier selected-WMMA T1 is `0.104 ms` versus `0.191 ms` for the
separately compiled streaming control; T2 remains streaming below its 320-token threshold.
Selected-WMMA maximum absolute error is `3.83e-4`; T=9/17/128 remain streaming at
`0.213/0.453/2.513 ms`. It rejects pre-launch, stale, cross-stream,
empty, poisoned, and over-mapped states;
invalid device causal positions poison exactly their own rows. The persistent-state qualifier also
proves that begin-transaction generation changes invalidate old committed reads, unlaunched layers
cannot obtain pending reads, compaction exposes none, and reads captured on one block-table binding
remain invalid through a different-row and original-row rebind until explicitly refreshed.
The provisional G16/token-K/feature-V/feature-scale instance uses 24 VGPR, 52 bytes LDS, zero
scratch, and reports occupancy 16. Neither this target binding nor its Op oracle substitutes for
the final paired model quality and Pareto gate.

### Current HIP build boundary

The root CMake project is now a HIP-only `gfx1201` project. It resolves the active ROCm Clang at
`/opt/rocm/llvm/bin/clang++` before compiler detection because CMake correctly rejects the
`hipcc` wrapper as a compiler; the HIP package supplies `hip::host` and the selected runtime
library directory is the build-tree runtime search path. The closed `ninfer_r9700_core` archive
contains the owned device context, device and pinned arenas, graph RAII, typed FP8-K/INT4-V
physical cache and Qwen decoder-state transaction owners, A2 append/A3 attention/compact kernels,
the target-private Qwen3.8
full-attention leaf, and the qualified eager/linear bring-up operations. `ninfer_artifact` links
that archive and owns generic `.ninfer` framing,
binder/typed binding, and direct final materialization: its pinned event ring and H2D transfers
use the canonical HIP load stream and do not repack final weights. The core and artifact
qualifiers link those exact archives and run physical-device ownership and exact H2D-byte oracles.
The same HIP core now owns startup-frozen request memory, typed FP8-K/INT4-V page allocation and
block-table publication, BF16 cyclic state, and BF16/FP32 linear-attention state. Its state
qualifier executes mapping, allocation-release, spill/restore, D2D lane/slot-copy, and zeroing
oracles on the physical R9700; legacy homogeneous growing-cache views are absent from that core
contract.

The eager utility qualifier covers the native-HIP public Tensor/Weight Op boundary as well as its
flat kernel layer: exact I32 position/fill/copy/gather and scalar transitions, BF16 movement,
structured request/lane scatter and path gather, feature extraction, finite casts,
FP32-reduction norms and activations, residual/gate updates, dense/Q6G64/W8G32 embedding, and
deterministic lowest-index greedy selection at T=1..8, D=5120, and vocabulary 248077/248320.
The old per-Op CUDA microbenchmarks and their private candidate controls are not retained as an
AMD performance lane; active eager and GDN evidence comes from the physical gfx1201 qualifiers,
and a future kernel challenger must add a measurement at the decision scope that admits it.
Linear qualification uses `[N=7168,K=5120]`: an independent represented-BF16/decoded-W8 FP64
oracle reports zero observed error for baseline and wave32-cooperative routes. The fixed measured
W8 candidate dispatch uses the baseline for T=1..4 and a 1-KiB-LDS eight-wave route for T=5..8,
measuring 0.122/0.236/0.351/0.462/0.547/0.555/0.555/0.577 ms. This is kernel bring-up evidence,
not selection of W8 as the artifact recipe.

This is a real HIP compile and device-link boundary, not an Engine build or a compatibility
backend. The generic artifact reader rejects the superseded NVFP4 and blockscale storage formats.
The family sequence-layout and Program host/device transfer ownership has been cut to HIP and no
longer carries legacy growing-cache or sparse-attention selection state. DFlash2's separate BF16
cyclic checkpoint owner is compiled and physically qualified. Remaining schedule-owned numerical
consumers, public Ops and target leaves, Engine, applications, and legacy tests still carry CUDA
contracts. They are
excluded from the CMake graph and must enter only with their native HIP replacement. Enabling the
former application, test, or benchmark switches is therefore an explicit configuration error
until that cutover is complete.

For 27B Main Text, G32 is 25 KiB per logical token and 6.25 GiB at 262,144 tokens; G16 is
26 KiB/token and 6.50 GiB. These are pool payloads, not total Engine capacity. The capacity solver
also includes MTP KV, fixed GDN and DFlash state, graph allocation, workspace, and reserved
headroom.

The earlier all-Q4 capacity gate completed both cache groups under the selected physical plane
layout. Historical schema-v19 reports in the schema-v12 matrices resolve G16 C1..8 maxima to
262144, 524288, 570304, 558080, 545856, 533632, 521408, and 509184 tokens; G32 resolves to
262144, 524288, 593152, 580416, 567680, 555008, 542272, and 529536. C1/C2 were bounded by the
per-request model context and C3..C8 by device memory for both groups. These manifests are not
current C=1..4 product evidence and must be rerun under the migrated contract. Their G16/G32
SHA-256 values are
`be1d5727948512560f28a7e149547714f9b80dca96169646a2eef0310a883feb` and
`c7e5020b03cd104440011fcc17095e3166defdc59045274192e6b930b027d922`; they bind the same all-Q4
artifact SHA-256 `19d029a89c1ef1cf87420067555021a7c7b435c31a92bea7c64ccf42c03d80e9` and benchmark SHA-256
values `86763d6d3ac2ff8fc27a3815f3816aaba44a11acfaf4ebf54ca5d8991ddc7d65` (G16) and
`2c9bb5a64f0e25a307f3f2b783a527ed6b42d1671c6e826d7ca98545052e9b31` (G32). These marker-free,
layout-bound campaigns supersede all prior capacity generations, which remain retained only as
historical evidence. Capacity favors G32 at
every device-memory-bound concurrency but does not select it without matched whole-inference
results.

Every weight and KV candidate is evaluated directly against an independent complete FP32/FP64
oracle and then at model level. Record output-error distributions, paired per-token NLL,
deterministic greedy-token identity, task quality, MTP/DFlash acceptance, decode T=1..8 latency,
prefill latency, effective bandwidth, matrix utilization, VGPR/LDS use, persistent bytes,
workspace, and total model/KV capacity. Select one weight recipe and one V group size; delete all
other production candidates and qualification-only routes.

## Observed development-host state on 2026-09-01

| Item | Observed value |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Running kernel | `7.0.0-30-generic` |
| R9700 PCI identity | `1002:7551`, ASRock subsystem `1849:5413` |
| Loaded R9700 driver | AMDGPU / ROCm driver `7.1.3.31500000` |
| ROCm/HIP | ROCm 10.0 userspace at `/opt/rocm/core-10.0`, HIP 7.15.26333, AMD Clang 23 |
| HSA identity | `gfx1201`, wave32, 64 CUs, 32,624 MiB visible VRAM |
| Other AMD GPU | Ryzen `gfx1036` integrated GPU is also enumerated; the R9700 is selected by PCI identity |
| Profiling | `rocprofv3` trace and gfx1201 PMU collection work; `rocprof-compute` has no gfx1201 analysis profile in this release |
| ISA tools | AMD LLVM 23; native `v_wmma_f32_16x16x16_fp8_fp8` is emitted in both the opcode and KV paths |

The host was rebooted into the supported ROCm 10 driver/userspace baseline before the following
measurements. `tools/r9700` derives its include and link roots from the selected `hipcc`, keeping
the selected ROCm 10 headers authoritative over stale distro headers. All performance evidence
recorded after this point is from this one baseline; results from the pre-reboot ROCm 7.2 setup are
not used for selection.

The standalone `tools/r9700` qualification target has
already passed exact device OCP E4M3 conversion/decode, signed INT4-G32 pack/dequant, a direct
wave32 16x16x16 FP8 WMMA opcode probe, and an owned asymmetric raw-register fragment-map
qualification on the physical R9700. Its KV
route now passes actual BF16 K/V/Q device inputs through the reusable R9700 A2 append source at
`src/ops/r9700/kv/fp8_int4_kv_append.hip`, including exact page-boundary writes and in-place tree
compaction. The source exposes device status for nonfinite BF16 inputs, unrepresentable V scales,
and invalid page addresses; its qualifier exercises all three errors. Under ROCm 10 it also captures
a page-66 runtime/kernel/memory trace with a SQLite result and summary. Only
`SQ_BUSY_CYCLES` and `SQ_WAVES` produce valid nonzero PMU data. Isolated generic and gfx12 wave32
VALU/LDS passes plus separate TCP and GL2C passes return all zero despite known matching
instructions and traffic. The installed SDK selector numbers agree with the amdgpu driver enums,
and the isolated passes rule out counter grouping or multiplexing. This is a ROCm 10 gfx1201
counter-collection limitation; no claim uses those zero events, and the maintained recipe requests
only the two validated counters. ROCm Compute Profiler 3.8.0 also has no gfx1200/gfx1201 SoC,
profile-set, or analysis configuration. Profiling interception changes the queue ring and is
therefore used for attribution, never for the latency-selection numbers.

On this stack, the controlled D256/Hq24/Hkv4 4,096-token qualification with one timing iteration
passed all sixteen G16/G32 × independent K/V/V-scale physical-layout combinations. Every variant
matched the independent stored-byte, FP8-QK, FP32-PV, and complete FP64-attention oracles; exact
three-plane tree compaction and the append error paths also passed. This establishes the full
layout cross-product as numerically viable, not a latency selection: one-iteration values are
not selection evidence and the selected group/layout remains gated on whole-attention measurement,
PPL, and greedy-token identity.

The same exhaustive fused-A3 gate subsequently passed 8,192 and 32,768 tokens for all sixteen
G16/G32 and independent-plane-layout variants. The 32K FP64-oracle maximum absolute error was
below `3.1e-7`; the score-streaming baseline measured roughly 24--27 ms per all-head A3 launch on
this stack. That result establishes no-score-buffer long-context correctness, not a selected
latency path: the launch is intentionally untuned and all timing awaits the coherent driver gate.

The standalone eager-linear qualification also has one current ROCm 10 measurement, deliberately
kept outside the Engine and artifact selection. At Qwen3.8's full-attention query/key projection
shape `[N,K]=[7168,5120]`, BF16 inputs and either BF16 weights or the unregistered W8G32 candidate
payload accumulate FP32 and round once to BF16 output. Independent host FP64 formulas over the
represented BF16 values and decoded stored W8 bytes passed at synthetic `T=3` and real-shape
`T=1` and `T=8`, with zero observed BF16 output difference. The W8 qualifier also checks the
converter's signed-code/ties-to-even/K128-padding/FP16-scale fixture byte-for-byte before launch.

For the 50-iteration event-timed decode sweep, one 256-thread block per output row is best for
BF16 at `T=1..3` (0.207, 0.366, 0.546 ms) and W8G32 at `T=1..4` (0.122, 0.236, 0.351, 0.462 ms).
A wave32-cooperative block that stages one 256-column weight tile in 1 KiB LDS and reuses it for
up to eight token rows wins the remaining ranges: BF16 `T=4..8` (0.637, 0.652, 0.660, 0.661,
0.659 ms) and W8G32 `T=5..8` (0.547, 0.555, 0.555, 0.577 ms). The fixed raw-Op dispatch uses
those crossovers only. ISA reports FP32 `v_fmac`/`v_fma_mix`; all four kernels have zero private
scratch and 1 KiB LDS. Baseline BF16/W8 use 8 VGPRs; wave8 BF16/W8 use 16/15 VGPRs. These values
are a narrow kernel decision for the one candidate shape, not a selected artifact recipe or an
end-to-end speed claim.

The same standalone Op now separately qualifies prefill widths `T=16/32/64/128` without changing
the decode dispatch. Its BF16 route uses the native gfx1201 wave32 `16x16x16 BF16×BF16→FP32` WMMA
opcode and the already independently checked asymmetric raw fragment map; it consumes represented
BF16 X/W, accumulates FP32, then rounds once to BF16. Every baseline, 16-wave LDS challenger,
WMMA candidate, and fixed BF16 route passed the complete independent FP64 oracle at real T16 and
spread direct-FP64 samples over the full real N/K matrix at T32/64/128, with zero observed BF16
error. The 50-event ROCm 10 timings for WMMA are 0.203/0.267/0.698/2.418 ms. It is 29 VGPR, zero
LDS, and zero private scratch, with an emitted `v_wmma_f32_16x16x16_bf16` instruction. W8G32 has
no WMMA route: an integer or BF16 matrix operand would fail to preserve the raw signed code plus
per-G32 FP16-scale decode contract. Its selected 16-wave, 1-KiB-LDS FP32-FMA route likewise passed
the independent gates and measures 1.119/2.225/4.371/8.751 ms. This records a fixed candidate-Op
width decision only; it neither selects W8G32 as the artifact recipe nor claims whole-model speed.

Short causal tails are separately qualified at 1, 63, 64, and 65 tokens across the same full
G16/G32 × independent-plane-layout matrix. They use the identical complete oracle and exact
three-plane state checks; raw empty inputs remain explicitly rejected rather than treated as an
implicit zero-length cache operation. A 65-token ragged sweep also passed every T from 1 through 8
and every G16/G32 × independent-plane-layout variant against the complete FP64 attention oracle.
Its exact A2 byte comparison incorporates negative zero, finite E4M3 saturation, INT4
ties-to-even, FP16-scale underflow canonicalization, and scale-overflow rejection. Corrupt page
IDs and negative/out-of-range device causal positions are independently made conspicuous as NaN raw
output, with no frontier publication permitted by the typed-cache owner. These are numerical and
state-contract gates, not latency measurements or a group/layout selection.

The selected G16/token-K/feature-V/feature-scale score-streaming branch also passes staged-prefill
T=9/17/128 at a fragmented 257-token frontier against the same independent FP64 oracle. The raw
whole-Op timings are `0.217/0.385/1.983 ms`; the target-bound 50-event timings after 20 warmups are
`0.204/0.442/2.504 ms`. The selected WMMA crossover is restricted to ordinary T1/T2; its
`0.912/1.448/10.839 ms` at these wider widths confirms why T>=3 remains score streaming.

The post-reboot unprofiled timing matrix covers contexts 1,024, 4,096, 8,192, and 32,768 at every
row count T=1..8, five event-timed iterations per layout, for 32 workload points and all sixteen
G16/G32 and independent-plane-order variants. Every point passed its prior identical-route full
oracle and the timing-mode exact state/error gates. Ranking each static layout against the best
layout at each point makes G16/token-major-K/feature-major-V/feature-major-scale the provisional
latency leader: mean normalized latency 1.004041, mean rank 2.188, and 14 of 32 point wins. The next
candidate is G32/token-major-K/feature-major-V/token-major-scale at 1.006646, mean rank 2.688, and
11 wins. This selects neither the V group nor the persistent ABI: the full BF16-source paired PPL
quality, capacity, and whole-inference Pareto gate remains mandatory before deleting the losing branches.

Two fresh 12,289-token checkpoint-direct traces first diverged exactly at layer 3's FP32 PV result,
despite matching represented Q/K/V, masked QK, and softmax inputs. The BF16 scorer no longer leaves
that long reduction to a one-shot einsum: its owner-private authority now performs explicit FP32
`torch.mm` over fixed absolute cache-row chunks of 8,192 rows in ascending order, adds every chunk
into one FP32 result in that same order, and includes the final partial chunk. Paired untraced v2
runs then reproduced exactly at 32K but differed at 3,932 of 4,095 8K NLL positions and 43 argmax
positions, while paired trace-enabled 12,289-token runs were exact. Because retained trace clones
change device allocation lifetimes and launch scheduling, trace repeatability does not admit the
ordinary scorer. The v3 scorer replaces the external multi-kernel/autotuned FLA chunk GDN route
with fixed FLA fused-recurrent for every T>1 span while preserving the independent project FP32
recurrence at T=1. Its provenance binds both exact attention-PV and GDN-dispatch identities and
rejects every v2 result. The v3 process boundary is now also fixed before accelerator imports: it
disables TunableOp, TF32 override, hipBLASLt, launch/copy serialization, and allocator no-cache
mode; fixes Triton fusion/interpreter/kernel-override controls; rejects architecture, Tensile
library-path, and custom-allocator overrides; and requires highest matmul precision, actual
gfx1201, and an exact resolved Triton/AMD lowering profile. Ordinary and schema-v8 trace
provenance bind that complete environment and physical-codegen identity, so earlier v3 output
without it cannot be reused. Fresh paired untraced v3 campaigns are now byte-exact at both 8K and
32K and form the BF16 quality authority; dense all-Q4 has passed offline rebase, while current mixed
and sparse quality acquisition remains open. A focused diagnostic now exercises fused-recurrent at the
production T=4,095 and T=4,096 spans and checks selected independent value columns against a
complete-row serial FP64 recurrence. Every prediction retains all K=128 terms, and the check covers
sampled BF16 outputs plus every K element of the selected FP32 final-state columns without building
the full FP64 HxKxV state trajectory; it remains diagnostic rather than BF16 quality evidence.

The completed checkpoint-direct 8K cache-only diagnostic now isolates that group decision from
weight quantization: it retains every original BF16 weight and formula and applies the exact
product OCP E4M3FN K and signed-INT4/FP16-scale V codecs only at the full-attention cache append/use
boundaries. Against the aligned BF16 trace, G16 has mean NLL 1.867564646 (delta +0.001907311), PPL
6.472514326, 93 greedy-token mismatches, no new NLL-at-least-10 position, and maximum absolute NLL
delta 0.808048248. G32 has mean NLL 1.868148220 (delta +0.002490885), PPL 6.476292617, 94
greedy-token mismatches, one new NLL-at-least-10 position, and maximum absolute NLL delta
0.445980072. Both meet the current bounded 8K quality guardrails; their flips demonstrate that
strict BF16 argmax identity cannot be an admission condition. G16 is the isolated quality leader because it has lower mean, mean-absolute, and
RMS NLL error, one fewer flip, and no new terrible position; G32 retains its 32 MiB main-Text 32K
capacity advantage and lower raw diagnostic wall time. The checkpoint-direct timing is not product
kernel evidence, and neither group is selected without the remaining production execution gates.

The complete same-size source-MSE-refined W8G32 G16 evaluator also finished its matched 8K gate.
Against BF16 it has mean NLL 1.865868912 (delta +0.000211577), PPL 6.461547962, 88 greedy-token
mismatches, zero new and two repaired NLL-at-least-10 positions, paired-delta standard error
0.001077376, and absolute NLL-delta p50/p95/p99/max
0.018169/0.138614/0.240617/0.821821. Against canonical all-W8 G16 at identical size, it changes
67 greedy positions, repairs 34 prior flips, introduces 32 new flips, leaves 56 persistent flips,
and improves mean NLL by 0.001841799; its absolute NLL is closer/equal/farther at
2,019/64/2,012 positions. This becomes the mean-NLL leader but not the flip-count leader and still
retains 88 BF16 argmax differences diagnostically, so it remains an evaluator pending complete
Pareto evidence rather than selecting the recipe.

## Repository impact established by audit

The initial audit found CUDA ownership across build, core, Ops, runtime, tests, tools, and active
docs. The list below records that migration input; completed HIP replacements are described above:

- The root `CMakeLists.txt` admits only CUDA 13.1+ and `sm_120a`.
- `src/CMakeLists.txt` gives core and Ops CUDA separable compilation, links cudart/NVTX, and owns a
  special non-RDC CUDA-driver-linked TMA archive.
- The tree contains about 280 `.cu`/`.cuh` files across source, tests, and benchmarks; hundreds of
  source/internal-contract files mention CUDA APIs or constructs.
- All repository-internal Op contract headers expose `cudaStream_t`. These are project-owned APIs,
  so replace them consistently rather than adding CUDA compatibility aliases.
- `src/core/device.*`, arenas, artifact materialization, cache containers, timers, and transfer
  paths initially owned CUDA allocation, copies, streams, events, and errors. The sole-target
  registry now performs both preflight and post-load memory queries with native HIP on its
  explicitly selected device.
- `src/core/decode_graph.*` owned CUDA graph capture/instantiate/update/upload/launch; it is now
  the canonical HIP RAII owner and its physical qualifier covers all five operations.
- `src/core/pdl.cuh` is NVIDIA programmatic dependent launch and has no direct AMD equivalent.
- Common kernel primitives contain CUDA shuffles and inline PTX for MMA, `ldmatrix`, BF16 casts,
  `cp.async`, cache control, shared addressing, TMA, and `setmaxnreg` warpgroup specialization.
- Kernel iteration and roofline tools hardcode RTX 5090 bandwidth, dense FP4 peaks, and SM120
  legality.
- Public owning values and reports use CUDA-specific names such as `use_cuda_graph`,
  `cuda_graph_*`, and CUDA transfer timing.
- Active README, performance, CLI, architecture, model, Op-development, kernel-development, and
  benchmark authorities describe CUDA/RTX 5090 behavior and commands.

Generic `.ninfer` framing and descriptor/binding concepts are backend-neutral. The NVFP4 numeric
format, physical layout, target identity, converter, and execution routes are deleted rather than
used for bring-up. With 32 GB rather than 5090 capacity, the final integer artifact, workspace,
graph allocation, persistent state, and BF16/hybrid-KV capacity must be recomputed rather than
copied.

The package and registry now expose the provisional `qwen3.8-27b/r9700-int-candidate` lane plus
eight explicit recipe evaluators (`r9700-w8g32-mse-eval`, `r9700-q4g64-n16k16-eval`, `r9700-q4-w8-n16k16-eval`,
`r9700-q4-w8-mse-n16k16-eval`,
`r9700-w8-bf16-embed-eval`, `r9700-w8-bf16-attn-qk-eval`,
`r9700-w8-bf16-attn-vo-eval`, and
`r9700-w8-bf16-gdn-qk-eval`) and the registry reports target key
`qwen3_8_27b_r9700`; retired model
IDs are not aliases. README,
legacy builds, tests, and other active authorities still retain obsolete Qwen3.6, groupwise,
NVFP4, and 5090 routes. Remove those remaining product surfaces while retaining family
implementation actually used by Qwen3.8; do not preserve `nvfp4` as an alias.

## Target implementation architecture

Keep the existing semantic ownership boundaries, replacing CUDA resources with direct R9700
resources:

- `src/core` owns HIP device context, allocations, stable arenas, streams/events, timers, raw
  copies, physical state/cache containers, and device-graph RAII.
- `src/artifact` remains generic framing, descriptors, binding, and final materialization. It does
  not repack live weights.
- `include/ninfer/ops` retains semantic Op contracts with a repository-owned R9700 stream/resource
  handle, not a backend framework.
- `src/ops` owns every gfx1201 kernel and fixed-shape implementation plan.
- Qwen family and target packages retain semantic schedules, weight views, target leaf selection,
  and live Program ownership without absorbing Op kernels.
- `src/runtime` retains publication/transaction policy and Engine PIMPL without device mathematics.
- CLI, server, and benchmark continue to execute inference only through Engine.

The source may use `.hip` or CMake HIP compilation according to the selected toolchain's most direct
and reliable form. Naming should describe semantic/device concepts rather than pretending CUDA and
HIP are interchangeable. There is no `#ifdef CUDA`/`#ifdef HIP` product matrix.

## Execution plan and gates

### Stage 0: driver and gfx1201 qualification

1. Install one supported AMD driver/ROCm release without mixing old kernel and new userspace
   packages. Preserve a bootable recovery kernel.
2. Confirm the discrete R9700 can be selected deterministically despite the AMD iGPU and RTX 5090.
3. Add one small `gfx1201` qualification executable covering allocation/copy, streams/events,
   BF16/FP16, wave32 shuffle/ballot, LDS, INT4/INT8/FP8/BF16 WMMA, graph capture/replay/update,
   generated ISA, and `rocprofv3` counters.
4. Measure stable device bandwidth and the real-format candidates at representative model shapes.

Gate: the chosen ROCm release compiles and executes gfx12 instructions, graphs, profiling, and the
format experiment reliably on the actual R9700. Select the final weight/activation profile before
porting optimized linear families.

### Stage 1: product and build cutover

1. Update active product authorities to one R9700, ROCm, `gfx1201`, and the selected Qwen3.8
   profile.
2. Replace CUDA language/toolkit/architecture checks with HIP and exact gfx1201 checks.
3. Remove CUDA, CUDA Driver API, NVTX, non-RDC SM120 TMA archives, NVCC options, and NVIDIA-only
   production sources from the active build.
4. Remove obsolete target registrations and affected product tests/docs rather than carrying a
   transition lane.

Gate: configure and link reject every GPU target except gfx1201 and contain no active CUDA
dependency.

### Stage 2: core execution substrate

1. Implement HIP DeviceContext, deterministic R9700 selection, streams, events, synchronization,
   timers, allocation, pinned memory, raw copies, memset, and memory reporting.
2. Port stable arenas, artifact materialization, physical KV/state containers, and spill/restore.
3. Replace CUDA-specific public/report names with device-semantic names across CLI, apps, tests,
   and serving reports where exposed.
4. Provide device-graph RAII ownership, but leave production graph capture until eager execution is
   complete.

Gate: focused core/artifact tests and a real artifact materialization route pass with stable
addresses and correct free/capacity reporting on the R9700.

### Stage 3: eager functional model

Port in dependency order:

1. exact transforms, scalar/state movement, position, scatter/gather, and sampling;
2. norms, activations, reductions, embeddings, and residual operations;
3. BF16 reference-quality linear and fused projection routes;
4. KV codecs/addressing/append and attention;
5. Gated DeltaNet convolution, recurrence, FP32 state, and projection/control;
6. complete Qwen3.8 prefill and decode;
7. MTP accept/rollback/commit and DFlash2 selection/verification;
8. multimodal Vision paths required by the supported product.

Every floating-point route is checked directly against the independent complete FP32/FP64 oracle;
exact codecs/transforms use exact oracles. CUDA output is supplementary evidence only. Explicitly
cover represented numeric decode, persistent FP32 GDN state, KV quantization boundaries, MTP state
transactions, arena lifetimes, and real model shapes.

Gate: the complete supported Qwen3.8 eager route generates qualified text and multimodal outputs
from the real artifact without any CUDA-built object.

### Stage 4: R9700 production kernels

Replace functional routes with measured gfx1201 families:

- decode GEMV/small-width Linear for T=1..8;
- prefill and wider speculative WMMA GEMM;
- attention, GDN, MLP, output-head, bias/residual, and SwiGLU fused projections;
- GQA decode/prefill and BF16-reference/FP8-K-INT4-V production consumers;
- GDN recurrence/state transition and fused projected convolution;
- existing MTP regression preservation and DFlash2 proposal, path selection, verification, and
  state commit;
- sampling and output selection.

- [x] Physically admit and promote the split-512 long-context decode-attention leaf. Its retained
  schema-v2 report covers the complete stored-byte oracle/rejection/Device-Graph matrix and shows
  a `6.323x`--`17.646x` incumbent/split speedup at 8K/32K across G16/G32 and active T=4 prefixes.
  Production selects it only at context>=8,192 for T=1 or fixed-width T=4, retaining the prior
  short-context routes. Its exact caller-owned native-context scratch
  is 37,847,040 bytes for ordinary T=1 and 151,388,160 bytes for one MTP-width T=4 call. The
  startup planner aliases that scratch across request slots, full-attention layers, and sequential
  Text/MTP calls on the same stream; replaying the exact layout model leaves the global arena
  unchanged at C=1..4 because the Text-prefill peak remains dominant. Admission nevertheless
  changes the benchmark executable and captured graph topologies. MTP3 separately classifies the
  max+6 MTP-cache T=4 leaf and max+4 Text T=4 leaf because they cross 8K at different frontiers.
- [x] Physically select and promote the three-stage full-score GQA6 initial-prefix route for both
  G16/G32 at P=128..4096. The retained operator report has SHA-256
  `0f80353256105a6b759b1f906f8cfc1cee1ef204870c83f1d3dff7ca0d69dcca`. At P2048 its
  G16 median is 14.669395 ms versus 61.569540 ms for fused Bq16 (0.238257x, 4.197142x faster),
  and its G32 median is 14.819379 ms versus 61.336494 ms (0.241608x, 4.138938x faster).
  Production QK is Bq16/Bk16 with six query-head waves sharing one FP8-K tile (29 VGPR,
  8,296-byte LDS, occupancy 15); maximum reduction uses 17 VGPR/64-byte LDS/occupancy 16; PV
  reuses each direct signed-INT4/FP16-scale tile across all six heads while retaining FP32
  probabilities and accumulation (116 VGPR, occupancy 12, 9,208-byte G16 or 8,952-byte G32 LDS).
  Every stage has zero private/scratch/spills; only QK contains the exact 16 BF16 WMMA operations.
  The startup planner reserves the caller-owned `(24*P*P + 24*P)*4`-byte peak at a stable arena
  address. Fused Bq16 remains only as the direct operator regression comparator; P<128,
  later-prefix chunks, tree attention, and nonselected layouts retain their prior routes.
- [x] Run the hash-bound all-layer FP8 `text.mlp.gate_up` decision after its matched physical
  `[2048,34816,5120]` qualifier report exists. Qualifier SHA-256
  `8f14df460c0b413cea99309a289c41cde2b71f32431450e76a9d4527d4af6306` for
  `profiles/bench/r9700-fp8-vs-a8q4-gate-up-shared-source-20260904.json` measured
  4.083197 ms FP8 versus 6.826113 ms Q4 (1.671757x faster). The owner is
  `tools/bench/decide_fp8_gate_up.py`; it admits only schema v2 with live harness-source and
  executable hashes,
  the exact nine-point axis-sensitive FP64 represented-format oracle, seven balanced timing pairs,
  the R9700/auto-power complete-path identity, and retained P2048 trace/capacity hashes. It replaces the
  trace's measured Q4 gate/up service by the measured complete-path ratio and verifies the exact
  64-object FP8-over-Q4 byte delta against both inventories. Its original capacity calculation
  omitted MTP plus optimized-head materialization and is invalid as current admission evidence;
  only its measured operator result and projected P2048 service reduction remain design evidence.
  The retained decision report SHA-256 is
  `3da19d842bba606b76fbd12625e8c5b65fa32e3bc64b74144865e92cc7a02809` for
  `profiles/bench/r9700-fp8-gate-up-decision-shared-source-20260904.json`: verdict `proceed`,
  exact all-layer added resident bytes 5,356,650,496 and projected whole P2048
  1.359737778 s / 1506.172758 tok/s from the bound 1.535900718 s trace. Its reported
  1,026,070,017-byte G16/C4/P8192 slack is superseded and must not admit an artifact or route.
  The speed projection is not a production whole-inference measurement.
  The pre-shared-source executed-path proof is retained at
  `profiles/rocprof/r9700-fp8-gate-up-isa-proof-hsa-20260904/proof.json` (SHA-256
  `eb0988a0a55e3e8a5af16e480dd8d315845df6ca320e62401f7b27921773b7df`). Rocprof dispatch
  identities bind the pointer/size-captured selected hipBLASLt ELF to 112 native FP8 WMMA
  instructions in its exact symbol interval, and bind the production Q4 control's file-backed
  symbol to eight native IU4 WMMA instructions; neither claim relies on a kernel name alone. That
  proof binds the superseded pre-refactor executable/report and is supporting ISA evidence only;
  refreshed executed-path proof for the shared-source gate and the distinct post-gate attention
  algorithm remains required before production promotion.
- [x] Run the two fixed-shape physical inputs for the capacity-optimal post-gate/up FP8 role
  decision, then execute `tools/bench/decide_fp8_post_gate_up.py`. The exact selected addition is
  `text.attention.query_key` plus `text.attention.gate_value` at `[2048,7168,5120]` and
  `text.gdn.query_key` at `[2048,4096,5120]`: three role-consistent families, 80 objects,
  1,024,065,536 added bytes, and 87,724,946 ns of measured Q4 service. The attention roles share
  one physical qualifier, so the two required report IDs were `attention_qk_gate_value` and
  `gdn_query_key`, both under schema `ninfer.r9700.fp8_projection_qualification.v1` version 1.
  Selection was driven by the then-retained 1,026,070,017-byte post-gate/up capacity estimate.
  That estimate omitted MTP plus optimized-head materialization, so its claimed 2,004,481-byte
  G16/C4/P8192 remainder is invalid and cannot serve as capacity admission. The hash-bound owner
  independently checks role inventory, axis-sensitive oracles, balanced raw timing, and source/
  executable identity; fresh selected-chunk physical capacity owns current admission. The retained
  attention report is `profiles/bench/r9700-fp8-vs-a8q4-attention-qk-gate-value-20260904.json`
  (SHA-256 `22f93d5280aeeec447b749f683c0e9ab2673b1f17ce1244bdd7abb91ab3080f8`):
  FP8 is 1.013659 ms versus 1.459598 ms Q4 (`1.439930x`). The retained GDN query-key report is
  `profiles/bench/r9700-fp8-vs-a8q4-gdn-query-key-20260904.json` (SHA-256
  `964f87c66a563f59cf55b1509ca8f7f4743eb8107541c0d83d271e254a35a84b`): FP8 is
  0.602939 ms versus 0.882819 ms Q4 (`1.464192x`). Both report zero BF16 steps at all nine
  axis-sensitive probes. The terminal owner report is
  `profiles/bench/r9700-fp8-post-gate-up-decision-20260904.json` (SHA-256
  `7d12a2d962606d967d905c010cf0b506398b3dd0be09841df9cdd62ee8563128`), verdict
  `proceed`; its exact 144-object replacement adds 6,380,716,032 bytes and projects
  1.332472 s / 1,536.993 tok/s whole P2048. Its 2,004,481-byte tight-cell claim is invalid for the
  reason above. The speed result remains far below the 2,000 tok/s acceptance floor and is a
  projection until the selected hybrid artifact passes whole-inference and decode-preservation
  measurement.
- [x] Screen source-weight error for that exact selective set before its remaining physical
  qualifiers. `profiles/bench/r9700-selective-e4m3-vs-q4g64-sampled-quality-20260904.json`
  has SHA-256 `f94a7f6a2d65c0906f44d5b2c6d2ff553eea3c532f5a528dce0d8d3ff07568ed` and binds the
  prior all-439 E4M3 screen SHA-256
  `b7eb34eeb4a7a63db7afa6649484c7f380e9408db9d3e292bbe18af565494c9d`. Across every
  selected object (144 objects, eight deterministic rows each, 5,898,240 BF16 values), aggregate
  relative-L2 is 0.026445170 for rowwise E4M3 versus 0.112502866 for Q4G64 (0.235062x), and
  max-absolute error is 0.006766185 versus 0.024658203 (0.274399x). E4M3 improves both metrics
  on all 144 objects; the least relative-L2 improvement is still 0.247669x Q4G64 and the least
  max-absolute improvement is 0.541667x. The selected aggregate E4M3 relative-L2 is only
  1.000832x the all-matrix aggregate, its maximum absolute error is 0.411018x the all-matrix
  maximum, and it contains neither global E4M3 worst tensor. Thus capacity selection does not
  concentrate a sampled source-codec quality outlier. This is a weight-codec screen, not the
  required real-model PPL or exact-token acceptance gate.
- [x] Run the one post-relocation selected-region C1/P2048/chunk4096 trace and use it only for
  attribution. The fresh hybrid run measured 1.265174658 s / 1,618.748832 tok/s. The validated
  database has exactly 2,393 dispatches and the planned 144 selected-FP8, 176 remaining-Q4,
  16 dense-layer, and 48 recurrence semantic-call inventories. Measured service is 319.377965 ms
  for selected FP8; 237.385302 ms for the 64 Q4 MLP-down complete calls; 22.272326 ms for the
  16 Q4 attention-output calls; 183.044849 ms for the 96 Q4 GDN value-z/output calls;
  149.057877 ms for exactly 16 QK, maximum, and PV dense-attention triplets; 144.556997 ms for
  recurrence; and 179.102369 ms for other base-Text kernels. Kernel-inactive Text-prefill wall is
  35.543809 ms. The retained attribution is
  `profiles/rocprof/diagnostic-p2048-post-relocation-hybrid-selected-region-20260904/post-hybrid-buckets.json`,
  SHA-256 `0a55fd72696c75f83d73c2fb3193d4eaa92a70efc76baf0a1dd549619af6312a`.
  Two asynchronous prefill-orchestration dispatches totaling 0.107520 ms completed at most
  0.228000 ms after the host measured marker but retain the explicit selected ROCTX association;
  unmarked service is separately exposed as 98 calls / 0.362400 ms. The analyzer now admits an
  associated asynchronous tail while still rejecting unmarked out-of-range work, and classifies
  only the exact QK/maximum/PV symbols as dense attention. This trace is valid attribution, not a
  profile-timing admission artifact.
- [x] Promote the trace-selected K128 ordinary-GDN gated-RMSNorm eight-row CTA at only the exact
  flattened extents `48*T` for T=1024/2048/4096/8192. The immutable direct report
  `profiles/bench/r9700-gated-rmsnorm-k128-rows8-ab-20260904.json` (SHA-256
  `bdada371d626a6f4398ac350f5aaebdf3318d1743fb507e32f626b3c062e2e5c`) passed incumbent
  BF16-bit parity and the independent FP64 formula and reduced P2048 from 1.16636395 to
  0.143720999 ms (ratio 0.123221397), with every admitted extent faster. The matched hybrid whole
  rerun improved from 1.251199252 s / 1,636.830337 tok/s to 1.202762421 s / 1,702.749429 tok/s;
  its report SHA-256 is `d1f67fb21427c5378e87afec4f4a273a34425223dd4c941d58e01fa3bc45a5aa`.
  The general route remains for all other shapes, and promotion does not select the hybrid artifact.
- [x] Run the fresh post-MLP selected-region trace and reject the remaining `other` kernel bucket
  as a `>=20 ms` optimization owner. The attribution-only `auto` capture measured
  1,183.706883 ms prefill and split service into 446.817900 ms remaining Q4, 321.775102 ms selected
  FP8, 148.139860 ms dense attention, 145.579372 ms recurrence, 98.049622 ms other kernels, and
  31.942668 ms kernel-inactive wall. Its exact inventory includes 64 fused SiLU-to-A8 preparations
  at 20.783684 ms and 112 ordinary A8 preparations at 8.941179 ms; attribution SHA-256 is
  `fd563a4c3760f6f302fbdd7e0580356eceafeadb3a1d94ed326b1009b3a7bfe8`. The largest allowed
  residual fusion, paired BF16 GDN a/b projections plus control, owns only 23.549814 ms and would
  need a 6.634098x complete-boundary speedup to save 20 ms despite retaining all 11,796,480 BF16
  WMMAs and all weight requests. It is terminally rejected by the CPU/static report
  `profiles/bench/r9700-post-mlp-other-service-static-audit-20260904.json`, SHA-256
  `032e19afd7bb0b097bbf215bfb165243dd8c1bad3d932017eba6b37567f71a8f`; no GPU qualifier follows.
- [x] Qualify and reject the trace-selected dense full-score P2048/G16 16-wave PV head-partition
  candidate. Its complete represented FP64 oracle, incumbent bit parity, poison/error semantics,
  and static gates passed at 70 VGPR, occupancy 16, 9,208-byte LDS, 512 threads, and zero
  private/scratch storage. In the immutable terminal report
  `profiles/bench/r9700-dense-full-score-pv-w16-head-partition-ab-20260904.json` (SHA-256
  `38ebd0c64352ccb0b27c1b60f92fa537865a92827901879f89b2c189c212feee`), candidate versus
  incumbent PV medians were 6.84690714 versus 7.28727818 ms (ratio 0.93956989), and complete-Op
  medians were 10.1469564 versus 10.8013163 ms (ratio 0.939418495). These miss the predeclared
  0.80 PV and 0.90 complete-Op gates, so no whole run was admitted. Qualification-only APIs,
  kernel, executable, and checker modes are removed; production remains the selected eight-wave
  PV route. This bounded rejection does not authorize an adjacent head/query/page mapping sweep.
- [x] Qualify and reject the capacity-preserving FP32 hipBLASLt dense-PV replacement. The only
  legal current-layout mapping used four strided-batched calls (`batch_count=6`) at
  `[M,N,K]=[256,2048,2048]`, with broadcast decoded V (`lda/stride=256/0`), interleaved query-head
  probabilities (`ldb/stride=49152/2048`), and interleaved output (`ldd/stride=6144/256`). The
  exact installed solution 140189 was bound in-process as GSU1/SK0, zero workspace, and passed the
  complete independent FP64 causal oracle, repeated-bit, graph-replay, invalid-frontier, fixed-domain,
  and no-clobber gates. Candidate versus incumbent medians were 4.524150848 versus 5.141388893 ms;
  the direct 16-call saving was only 9.875808716 ms and the old-bucket projection was 80.216934204 ms,
  missing the predeclared 30 ms / 61.161073 ms gate. The immutable report is
  `profiles/bench/r9700-dense-fp32-gemm-pv-p2048-ab-20260904.json`, SHA-256
  `b231a114f4b50684f2e2fdc3eb8a8ce08a9702053dc32e5ff9346a80132a8538`. The disconnected
  qualifier is removed, production remains unchanged, and no adjacent layout or algorithm sweep
  is authorized.
- [x] Qualify and reject the dense full-score P2048 FP8-Q/Bk32 route. It passed exact FP8-Q-panel,
  independent FP64 score/complete-attention, overwrite/liveness, page/causal/status/tail, and
  static gates (32 native FP8 and zero BF16 WMMAs, 63 VGPR, 8,192-byte LDS, occupancy 16, no
  scratch/spills). Physical timing was 2.983591080 versus 3.442310095 ms per call
  (`0.8667409378x`), yielding 47.73745728 ms over 16 calls and only 7.33950424 ms matched saving.
  This fails the fixed 32.439611 ms / 15 ms saving gate, so no whole run is admitted and production
  remains BF16-WMMA Bq16/Bk32. Candidate-only implementation and qualification surfaces are
  removed without an adjacent FP8-Q sweep. The immutable design and physical reports are
  `profiles/bench/r9700-dense-full-score-fp8-q-bk32-static-design-20260904.json` (SHA-256
  `aed480c787313c280aefc83d8cdf943a6b81711872786677f905f3d4036cd521`) and
  `profiles/bench/r9700-dense-full-score-fp8-q-bk32-ab-20260904.json` (SHA-256
  `ef0d6557a59145c3f632b96b24f41c922369d9f42fa48ab2c7b24d5b1c7e7f30`).
- [x] Qualify and reject the ordinary normalized-GDN LDS-scope cutover at the fixed whole-P2048
  gate. The direct challenger preserved exact incumbent bits and the independent complete formula
  and improved the P2048 Op median from 3.113205 to 2.761526 ms. Its matched hybrid whole run under
  auto power improved the accepted K128 baseline from 1.202762421 s / 1,702.749429 tok/s to
  1.193522512 s / 1,715.930810 tok/s, a 9.239909 ms / 1.007741x gain. This misses the predeclared
  10 ms gate by 0.760091 ms, so production is restored to the incumbent ordinary recurrence and
  the challenger executable surfaces are removed without an adjacent barrier or geometry sweep.
  The immutable terminal report is
  `profiles/bench/r9700-gdn-ordinary-lds-production-p2048-c1-20260904.json`, SHA-256
  `0af07a8340f310b7f89a800bc90864377fb76ac5e45c2e22f8dd74d580a6a69f`.
- [x] Qualify and reject the exact ordinary Text P2048/C1 GDN scale-sidecar cutover. The
  provenance-complete v2 direct run passed the numerical/resource and `2.397136 ms` gate at
  `2.252036095 ms` per call
  (`108.0977325 ms` over 48) versus `3.259594917 ms`. Preparation improved from `0.07719899714`
  to `0.03379999846 ms`, and recurrence from `2.893904924` to `2.164277077 ms`. Its immutable
  report is `profiles/bench/r9700-gdn-scale-sidecar-p2048-ab-v2-20260904.json`, SHA-256
  `20288276c005bf469f99d6a7dccddc497c5d49c6a2301b65dac746f71fc38304`. The clean isolated whole
  run measured `1.153886920 s` / `1774.872034 tok/s`, saving `24.650137 ms` from the accepted
  `1.178537057 s` baseline but missing the fixed `1.148537057 s` / 30 ms gate by `5.349863 ms`.
  Workspace remained exactly `608,387,072` bytes. Its immutable report is
  `profiles/bench/r9700-gdn-scale-sidecar-production-p2048-c1-v2-20260904.json`, SHA-256
  `110dbb4090dfa4a92ac993bd70ac4c21666b7a259abd17f63a4dbb913d9f6e95`. The candidate API,
  kernel, runtime/workspace integration, test, and qualification surfaces are removed; production
  retained the general recurrence at this decision point. The later clean phase-1+2 bundle below
  reintroduced and promoted the scale-sidecar route as part of a different coherent boundary, so
  this historical standalone rejection is not the current production-route description. The
  incomplete direct v1 and the concurrently contaminated whole v1 are removed and are not evidence.
- [x] Audit and reject two-stream P2048 projection overlap without implementation. The fresh trace
  gives GDN query-key/value-Z service of `29.039348/116.964267 ms` across 48 layers, so perfect
  overlap can save only `29.039348 ms` and the 20 ms gate requires hiding `68.87%` of all query-key
  service. Attention query-key/gate-value adds only a `16.1977975 ms` balanced upper bound. The
  current calls illegally share one activation-code/scale/status region; an aligned GDN partition
  would fit inside the existing persistent reserve, but the 512-thread Q4 CTA consumes all 16
  occupancy waves while both branches saturate the same device-wide matrix/cache/memory resources.
  The trace is already kernel-active for `97.30%` of Text-prefill wall. Concurrent streams would
  spatially time-slice throughput work rather than hide independent latency, so the required
  `>=20 ms` whole saving is not credible. No stream/event implementation or GPU sweep is admitted.
  The terminal audit is
  `profiles/bench/r9700-p2048-projection-stream-overlap-static-audit-20260904.json`.
- [x] Qualify and reject ordinary-GDN output gated-RMSNorm-to-A8G64 fusion at P2048. The exact
  256-thread/eight-wave candidate passed incumbent parity, the independent represented-FP64
  formula and codec, poison/alias gates, and static inspection at 25 VGPR with zero
  LDS/private/scratch. Candidate preparation was `0.196279004 ms` versus `0.239998996 ms`, but
  its 48-call aggregate `9.421392202 ms` missed the `5.699527 ms` ceiling; complete output saved
  only `3.050880432 ms` across 48 calls versus the required `5 ms`. The immutable report is
  `profiles/bench/r9700-gdn-output-gated-rmsnorm-a8-fusion-p2048-ab-20260904.json`, SHA-256
  `8396c5e76b4166405acab70abce11f7f7137e264a1eb61eec3b25d0a098ba962`. Qualification code is
  removed and production retains the three separate stages; the non-unique value-Z boundary
  remains rejected.
- [x] Qualify and reject the fixed-P2048 ordinary-GDN projection/convolution direct-scatter
  cutover at its isolated post-MLP whole gate. The direct report
  `profiles/bench/r9700-gdn-prefill-projection-conv-direct-scatter-ab-20260904.json` (SHA-256
  `fe852ad70f33f2e654cf31f2ef0da874c5355cdd491f9f97e027cb68142cf310`) passed at
  0.288159013/0.559597015 ms per layer (ratio 0.514940202), a derived 13.0290241 ms over 48 layers.
  The isolated whole report
  `profiles/bench/r9700-gdn-prefill-projection-conv-direct-scatter-production-p2048-c1-20260904.json`
  (SHA-256 `8ceaa1d948e0a76d29765a4255dd806a2bc3ffeba4b5ec009fc39f8b893e7c2f`) improved the accepted MLP
  baseline from 1.178537057 s / 1,737.747715 tok/s to 1.168570885 s / 1,752.573288 tok/s, only
  9.966172 ms and 1.00852852x. This misses the fixed 12 ms and 1.01x gates; workspace stayed
  608,387,072 bytes. Production is restored to the ordinary copies/convolution/extracts, and the
  candidate API, kernels, harness, Make targets, static checker, and report validator are removed.
- [x] Qualify and reject the trace-selected Q4 MLP-down M64xN256 32-wave ping-pong candidate,
  leaving production
  unchanged until admission. The exact semantic shape is `[T,17408] x [17408,5120]` with stored
  `[N,K]=[5120,17408]`; the proposed 1,024-thread mapping keeps two output fragments and at most
  four next-group dwords per wave lane while halving activation rereads. At P2048 its auditable
  source-request bound is 2,271,559,680 versus 3,006,873,600 bytes with the identical 44,564,480
  native-IU4 instructions. First require at most 96 VGPR, exactly 25,856-byte LDS, zero scratch,
  one resident workgroup, unchanged instruction count and accumulation order; then require the
  complete represented FP64 oracle/full bit parity at T=1,024/2,048/4,096/8,192, a direct
  alternating P2048 ratio at most 0.90 with no extent above 1.01, and at least 10 ms whole-P2048
  saving. The first failed gate rejects it without an adjacent tile or cache-hint sweep. Design
  authority: `profiles/bench/r9700-mlp-down-m64n256-w32-static-design-20260904.json`.
  The static gate passed at eight IU4 WMMAs, 87 VGPR, 25,856-byte LDS, 1,024 threads, and zero
  private/scratch storage, but every physical extent regressed: challenger/control ratios were
  `1.0146974/1.0120804/1.0203344/1.0149652` for T=1,024/2,048/4,096/8,192. The immutable terminal
  report is `profiles/bench/r9700-a8q4-m64n256-w32-mlp-down-ab-20260904.json`, SHA-256
  `a41e2a2ad65809ade16b629e698244834d2c2da065335f8fab62869813aaef5b`. Both rejected M64xN256
  executable paths are removed; M64xN128 ping/pong remains production and the bounded stop rule
  excludes an adjacent tile or cache-hint sweep.
- [x] Qualify and reject next-G64 scale prefetch in the production M64xN128 geometry. Static
  inspection passed at eight native IU4 WMMAs, 89 VGPR, 17,152-byte LDS, 512 threads, occupancy
  16, and zero scratch, and the exact numerical/output/tail/status gates passed. Direct `auto`
  timing lost all 12 MLP-down/GDN value-Z/GDN output cells over T=1,024/2,048/4,096/8,192. The
  trace-call-weighted P2048 aggregate regressed from `375.835487` to `391.499994 ms`
  (`1.041679x`, `-15.664507 ms`). The immutable report is
  `profiles/bench/r9700-a8q4-prefill-cta-scale-prefetch-ab-20260904.json`, SHA-256
  `aaeeaf16a0040d5a23046d40fff532f34805124cdd3e5bbeb5d0a721962ad8ef`. Qualification-only
  implementation and tooling are removed; production retains post-compute scale loads.
- [x] Qualify and reject the trace-selected residual-add to K5120 token8 RMSNorm prefill fusion.
  Static inspection passed at 18 VGPR, occupancy 16, and zero LDS/private/scratch, while exact
  BF16 residual/output publication, complete independent-oracle, poisoned-rewrite, dead-delta
  alias, and status checks passed. Nevertheless, direct `auto` timing lost every
  T=1,024/2,048/4,096/8,192 cell (`1.89297712x`--`2.98276377x`). P2048 regressed from
  `0.188480005` to `0.554238975 ms` per pair (`2.94057178x`, `-0.365758955 ms`). The immutable
  terminal report is `profiles/bench/r9700-residual-rmsnorm-k5120-prefill-ab-20260904.json`,
  SHA-256 `6ed367d0438f0f7f8dbfe4bd01f7da126a6857c1953da8f1ed6e294eec20c134`; the retained design is
  `profiles/bench/r9700-residual-rmsnorm-k5120-prefill-static-design-20260904.json`, SHA-256
  `40dcc021e59e8c80272ed6c36c9ab4ef36e1bd6c2e9939c439671426a44eca73`. Qualification surfaces
  are removed, production retains the separate residual-add plus selected RMSNorm pair, and no
  adjacent fusion variant follows this terminal result.
- [x] Qualify and reject direct signed-A8 by packed-W4-to-IU8 at the fair expand-once ping-pong
  implementation boundary, while retaining the focused GL2C/TCP/SQ capture as a separate physical-
  bottleneck task. The route was not rejected from
  peak arithmetic alone. The externally supplied ceiling hypothesis is `152.5` useful TMAC/s for
  split-IU4 versus `148` TMAC/s for native IU8, only a `2.95%` IU8 disadvantage, while the current
  weighted production aggregate realizes about `45.54` useful TMAC/s. If counters attribute that
  much larger utilization gap to matrix scheduling, dependencies, reconstruction, or VALU/wait
  pressure rather than saturated physical HBM service, build one bounded exact-shape qualifier
  using direct A8 bytes and packed Q4 global storage. Compare expand-once-before-LDS with
  packed-LDS/per-consumer sign extension only when ISA/resource inspection leaves both credible;
  require the independent represented-formula oracle, native IU8 proof, tail/status behavior, and
  direct `auto` timing. Never materialize a persistent I8 weight plane or exclude the route merely
  from the close theoretical peaks.
  The original single-pass counter capture is terminal no-data evidence: the installed gfx1201
  runtime advertises more SQ events than its eight physical register descriptors and aborted on
  the ninth SQ event despite `pmc-check`. The corrected two-pass package at
  `profiles/rocprof/r9700-production-q4-p2048-pmc-two-pass-plan-20260905/` captured and independently
  validated the exact 176-dispatch inventory in each pass. Its attribution report SHA-256 is
  `0385a2f226c7f31f36184d4bfca9584b639183710386e0e5076029085874d646`. Exact-shape ranges are
  `94.94`-`98.22%` occupancy residency, `70.95`-`94.72%` dependency wait,
  `3.97`-`24.98%` issue wait, `84.25`-`95.56%` GL2 hit, and `1.73`-`4.00%` L0 vector-cache hit.
  These support a latency-hiding hypothesis but do not prove physical bandwidth or a causal stall.
  A CPU/build-only schema-v2 extension of
  `q4_hardware_peak_probe` is also ready: the exact-ISA gate proves the matched native-IU8
  two-chain/four-dependent-instruction topology at 23 VGPR/occupancy 16 and the eight-independent-
  chain saturation control at 122 VGPR/occupancy 10, both with eight IU8 sites and zero LDS or
  scratch. The completed `auto` run is
  `profiles/bench/r9700-q4-hardware-peak-iu8-20260905.json`, SHA-256
  `1e4f5a5ac863a1939e560794758e335692998121bc5172a6f5696f1af8653756`. With MAC counted once,
  it measures `200.880451` useful split-IU4 TMAC/s versus `195.123788` topology-matched IU8
  TMAC/s, ratio `0.97134284` and a `2.86572%` IU8 deficit; the eight-independent-chain IU8
  saturation control is `198.502175` TMAC/s, so the matched four-deep dependency chains cost only
  `1.70194%`. The external `148/152.5 = 0.97049180` ratio agrees within `0.08510` percentage
  points even though the absolute ceiling methodology differs. Neither register-only control
  includes Q4 expansion or staging cost. The complete 25,344-byte two-bank M64xN128 qualifier
  emitted exactly eight signed IU8 sites, one publication barrier, zero private/scratch bytes, and
  one load per packed-W dword. It initially required 115 VGPR/occupancy 12. Preventing four-K16
  operand hoisting and using direct full-G64 W loads improved this only to 111 VGPR/occupancy 12,
  versus the fixed <=96/occupancy-16 gate. A compact legal expansion probe bounded serialized
  publication near 100 VGPR, still not credibly admissible. The route is terminally rejected
  before GPU timing; its immutable evidence is
  `profiles/bench/r9700-a8q4-direct-iu8-pingpong-static-rejection-20260905.json`, SHA-256
  `d4f2c610b973935aa60b732c3ee39e9dc84a512781839458a5caadcca78f2a2c`. Qualification
  source/tool/build surfaces are removed and no adjacent IU8 topology follows.
- [x] Attempt and terminally close selected-dispatch ATT for the selected P2048 Q4 service. The
  completed PMC motivated a two-bank/two-entry register FIFO, but its
  best scalar form emitted at `94` VGPR/occupancy `16` with an explicit loadcnt-4 threshold and
  still acquired compiler waits `2/1/0` before oldest publication plus `2/0` at loop rotation,
  draining all younger requests. A peeled form required `118` VGPR/occupancy `12`; gfx1201 lacks
  direct global-to-LDS loads, and a third LDS bank does not extend global-load distance. The
  terminal static evidence is
  `profiles/bench/r9700-a8q4-prefetch2-static-rejection-20260905.json` (SHA-256
  `4a4ffd1c0f5d759fd6debff02987572a233a2bb599ccc965f0231386d56e582b`). Production remains unchanged.
  The installed rocprofiler-sdk `1.3.5` cannot retain this exact full selected dispatch: the
  correctly targeted `gfx1201`/GPU-index-0, SE0, WGP1, SIMD0 capture filled an exact
  `1,610,612,704`-byte trace from a `1,610,612,736`-byte buffer and reported both SQTT and thread-
  trace buffer overflow. Its immutable invalid/no-authority receipt is
  `profiles/rocprof/r9700-production-q4-p2048-att-retry3-20260905/overflow-failure.json`
  (SHA-256 `da77c12c2659556c793ba714a1092affcd44519a2255f3578b0aaddf185dc428`). Earlier
  384-MiB overflow, incompatible-command, and wrong-GPU attempts are separately marked invalid;
  none is interpreted. The installed CLI exposes no narrower intra-dispatch trace interval, so no
  larger-buffer retry is scheduled. The valid two-pass PMC report remains the physical attribution
  authority. It shows broad dependency wait of `70.95`-`94.72%`, high residency, and healthy GL2
  locality, but cannot establish a causal PC stall or physical bandwidth limit. The production
  aggregate executes `21.99023255552` TMAC in `417.093037 ms`, or `52.722608` useful TMAC/s,
  versus the isolated `200.880451` TMAC/s IU4 ceiling; that headroom remains unexplained rather
  than being relabeled utilization. This selected-Q4 iteration is terminally tool-limited: all
  credible bounded pipeline mechanisms were tried and rejected, no ATT claim or new Q4
  architecture follows, and the next performance decision returns to the practical-ceiling/P2048
  work below.
- [x] Qualify and terminally reject the scheduling-only response to the PMC dependency signature.
  A disconnected exact-production two-bank candidate used a zero-instruction value anchor plus
  `sched_barrier(0)` to place all eight IU4 WMMAs, 16 integer recombinations, 16 I32-to-FP32
  conversions, 16 scale products, and 16 FP32 accumulations before the successor-bank publication
  drains. It met the static ceiling exactly at 96 VGPR, 17,152-byte LDS, occupancy 16, and zero
  private/scratch storage, and its independent FP64, full bit-parity, tail, alignment, and status
  regression passed. The direct `auto` P2048 A/B nevertheless rejected it: the exact 48/64/64-call
  aggregate regressed from `394.126793` to `401.131914 ms` (`1.0177738x`, `-7.005121 ms` saving),
  with per-shape ratios `1.000735`, `1.018073`, and `1.026857`. The immutable report is
  `profiles/bench/r9700-a8q4-epilogue-schedule-p2048-ab-20260905.json` (SHA-256
  `3a2430a6bb07850d6f49d0d9ea62e695b1103838f578ee7ebbe6ed0d3aaf7ffc`). Qualification-only
  sources, checker, executable, and assembly are removed; production remains unchanged.
- [ ] Restore retained-production dense C1/P2048/spec-none prefill to at least `2,000 tok/s`, then
  establish that the retained low-context path is near its practical optimization ceiling before
  resuming chunk selection or post-promotion qualification. Crossing `2,000 tok/s` is a minimum
  recovery milestone, not completion: the crossing candidate still requires a focused whole-P2048
  confirmation followed by fresh whole-path attribution/roofline review. Continue low-context
  optimization while a dominant component retains a credible material gap to its measured
  hardware/service ceiling, including unexplained stalls, low matrix utilization, avoidable data
  movement, or an unqualified instruction path. Resume the long-context chunk decision only after
  a fresh selected-region attribution shows (a) no dominant bucket with an unexplained credible
  `>=25 ms` whole saving, (b) aggregate non-overlapping credible remainder `<=5%` of the current
  whole wall, and (c) selected Q4 service at `>=120` useful TMAC/s, or matched terminal physical
  evidence that bounds it below that rate. The current service-substitution model places the
  practical gate-opening neighborhood around `0.93`--`0.98 s` (`~2,090`--`2,200 tok/s`), but that
  interval is inferred rather than a promised result and requires whole confirmation. At exactly
  `2,000 tok/s`, Q4 would need only about `79`--`84` useful TMAC/s, still far below both the
  measured `200.880451` isolated-IU4 rate and the conservative external `152.5` estimate, so the
  floor alone cannot establish proximity to the ceiling. The retained four-role N16/K16 route's
  current exact whole measurement is `1,847.942898 tok/s` (`1.108835254 s`); the older
  `1,774.872034 tok/s` scale-sidecar result belongs to a rejected pre-cutover candidate and is not
  the production baseline. The interrupted 32K finalist campaign's ten complete reports remain
  useful diagnostic but non-resumable evidence; their pre-receipt namespaces cannot become a
  chunk-selection authority. A nominal chunk-size sweep cannot substitute
  for this gate because P2048 is a single effective chunk at either finalist setting. Re-establish
  attribution at the complete projection or fused-boundary scope, prove the expected gfx1201
  matrix instructions/resources and operand streaming for any replacement, qualify it numerically,
  and measure the whole P2048 route directly before returning to the finalist queue. The installed
  hipBLASLt catalog is now exhausted at the exact `T2048/N34816/K5120` outer-scaled descriptor:
  `791` type-level entries reduce to `10` supported algorithms, the prior qualifier had timed eight,
  and the two missing zero-workspace kernels measured `14.6055 ms` and `22.9663 ms` versus
  `3.64404 ms` for the incumbent. The raw bounded diagnostic is
  `profiles/bench/r9700-fp8-gate-up-catalog-screen-20260905/catalog-screen.txt` (SHA-256
  `f31f962da2253dafb3d0b6febbf23c4cd1eb64b6f6df84f4a6e5ba1bffd0960d`). No supported split-K or
  nonzero-workspace candidate exists for this descriptor, so the next projection route tested was
  the disconnected custom FP8 gate/up challenger rather than further library or chunk sweeps. That
  exact M128xN128xK32 challenger reached `62` VGPR, `16,384`-byte LDS, occupancy `16`, zero
  private/scratch, eight native FP8 WMMA sites, explicit successor-load overlap, independent FP64
  represented-format correctness, and same-address graph replay. It nevertheless measured
  `7.640492439 ms` versus `3.605496049 ms` for exact hipBLASLt solution `123104`, a projected
  `-258.239768982 ms` over 64 calls, and is rejected. The immutable timing receipt is
  `profiles/bench/r9700-fp8-gate-up-m128n128-ab-20260905/report.json` (SHA-256
  `36dbb9206070e10eefa113820518c0189fdf6f45fbc75e7021d44c7dbba21772`). The floor-closing search
  therefore returns to the unexplained selected-Q4 whole-prefill service gap while retaining only
  coherent smaller gains that pass their combined boundaries. The compatible ordinary-P2048 GDN
  bundle was qualified as one boundary: direct projection/convolution scatter, an exact
  262,144-byte FP32 Q/K inverse-norm sidecar with four-row recurrence ownership, and a fused
  recurrent-output/gated-RMSNorm/A8 handoff. The recurrence component has passed its direct
  represented-input FP64 gate across complete 2,048-step sampled histories, in-place/distinct
  parity, poison/immutability/overlap checks, and the retained snapshot/replay matrix. Its current
  isolated T2048 timing is `2.48721 ms`; this is component evidence only, not a floor pass or
  authority to resume chunk selection. The qualifying source and executable identities are
  SHA-256 `0df869fe9aaac5ef15a659273bf0a38b14faf1325f27798c0d6f427fbb0d0b5f` and
  `b4425e746960b95548819caa562b3cf3e07c8ab6c8d6a5c3e0bc5806c6dd1a6a`. The first purported
  phase-1+2 whole receipt at
  `profiles/bench/r9700-gdn-bundle-phase12-production-p2048-c1-20260905.json` is invalid because
  its executable was built while an incomplete phase-3 output-fusion hunk was present; it cannot
  attribute the direct-scatter plus recurrence boundary. The previously rejected output fusion
  has been removed before the corrected run. The clean phase-1+2 executable then measured
  `1.1632268 s` / `1,760.621777 tok/s`, a `15.310257 ms` saving from the retained baseline, with
  workspace unchanged at `608,387,072` bytes. Its report is
  `profiles/bench/r9700-gdn-bundle-phase12-clean-p2048-c1-20260905.json` (SHA-256
  `f1d818f68c3c243a030d64c83ac91525a212605134e6a6c337d1ad1712008651`). This exact clean
  phase-1+2 implementation is the current normalized-T2048 production route; its direct-scatter
  preparation and `scale_sidecar_recurrence_kernel` are also the route observed in the fresh
  N16/K16 trace below. At that pre-N16 checkpoint, the floor remained closed by about
  `139.2268 ms` and no chunk sweep was admitted. The first disconnected P2048/G16
  native-FP8-Q/Bk32 plus
  PV-W16 timing receipt is explicitly invalid: review found that its reconstructed QK stage omitted
  production's causal CTA pruning and computed roughly twice the intended triangular work. The
  invalid receipt is retained only as a no-authority diagnostic at
  `profiles/bench/r9700-dense-attention-combined-p2048-g16-ab-20260905.json`; it cannot reject or
  promote the corrected composition. After restoring causal pruning, recompiling, and rerunning
  every independent correctness/static gate, the corrected complete Op measured
  `11.4026193619 ms` versus `11.0173797607 ms`, a projected `-6.16383361816 ms` over 16 layers.
  The composition is therefore terminally rejected without a production edit. Its valid report is
  `profiles/bench/r9700-dense-attention-combined-corrected-p2048-g16-ab-20260905.json` (SHA-256
  `fd7b411d0532f64a3dad83dcc63623b3738c3449b76b62a157cfd1a726ca4839`). A disconnected attempt to
  express the remaining Q4 latency-hiding design as one HIP-owned M64xN128 pipeline also stops at
  its static gate: it emitted the exact eight signed IU4 sites, 17,152-byte LDS, zero scratch, and
  512-thread workgroup, but required `100` VGPR/occupancy `12`, drained the intended load FIFO with
  compiler-inserted `loadcnt0`, and moved the FP32 update outside the signal/wait interval. It is
  rejected before numerical or GPU work; the immutable receipt is
  `profiles/bench/r9700-a8q4-hand-pipeline-static-rejection-20260905.json` (SHA-256
  `52aa145ca8dc647bfc77e6ee8a25d6fc7a312069073a5f791d742abd4b93ef27`). A monolithic handwritten
  assembly replacement is a separate ABI/control-flow implementation, not an adjacent relaxation
  of this failed source-level gate. A fresh audit also corrects the scope of the historical
  split-K claim: only a design-level M128xN128/Kchunk832 topology was rejected, not a physical
  M64xN128 split-K run. The distinct tail-free S=2 current-tile form is nevertheless statically
  non-material: production already supplies 1,280--3,072 CTAs/call at 94.94--98.22% residency;
  it preserves all IU4/scaling/operand work while adding 40.802189312 GB of FP32 partial traffic
  across the 48/64/64-call inventory, at least 63.753421 ms at nominal 640 GB/s, and up to
  201.327 MB workspace per call. Netting even 30 ms would require the unchanged first stage to
  improve by another 93.753421 ms (`22.48%`), so no implementation or GPU run follows.
  The separate activation-only-LDS/direct-register-W challenger was genuinely untested and reached
  its emitted gate, but is also terminally rejected before GPU work. Its best form used the exact
  8,448-byte ping/pong activation image, eight signed IU4 sites, 512 threads, and zero spills, yet
  the current-plus-successor direct-weight live set required `116` VGPR/occupancy `12` versus the
  fixed `<=96`/occupancy-16 gate; a scheduler-barrier form was worse at `122`/`10`. The receipt is
  `profiles/bench/r9700-a8q4-partial-lds-static-rejection-20260905.json` (SHA-256
  `2e04bbff9447b46dcfc42120634424a0db83c5ed591a816c83cf80aa5c49e1a5`); qualification-only
  surfaces are removed and production is unchanged.
  A final scalar-cache-prefetch test also rejects the hypothesis that explicit gfx12
  `s_prefetch_data` requests can hide the incumbent vector-weight latency without changing its
  layout. The exact production M64xN128 arithmetic/LDS mapping emitted four rolled prefetch sites
  (16 dynamic cache-line requests per wave), eight signed IU4 sites, 95 VGPR, 17,152-byte LDS,
  occupancy 16, and zero private/scratch/spill storage; its independent represented-formula,
  incumbent-bit-parity, tail, status, alignment, guard, and publication gates passed. Direct
  balanced `auto` timing nevertheless lost every exact P2048 shape by `1.40066x`--`1.49604x`:
  the 48/64/64-call weighted aggregate regressed from `386.922674179` to `553.251506805 ms`, a
  `-166.328832626 ms` saving. The immutable terminal report is
  `profiles/bench/r9700-a8q4-scalar-prefetch-p2048-ab-20260905.json` (SHA-256
  `caf6543e4c632863fa3861160e63b67ba6ed802edab172adc4cf71c3eb874f15`). Qualification-only
  source, checker, assembly, and executable surfaces are removed; production is unchanged. The
  next Q4 candidate at that checkpoint had to alter persistent transaction/coalescing geometry
  rather than add an adjacent cache hint to the row-split layout.
  The distinct wave-role-specialized 128-bit cooperative loader did produce a real gain while
  preserving the row-split artifact and production M64xN128 arithmetic/LDS mapping. Its corrected
  qualifier emitted exactly two `global_load_b128` code sites and no narrow code loads, retained
  eight signed IU4 sites, 89 VGPR, 17,152-byte LDS, occupancy 16, and zero private/scratch/spill
  storage. A row-distinguishing fixture, complete incumbent bit parity, sampled complete-K FP64
  oracle, full bad-status poison, rejected-call no-clobber, alignment/tail, and output-guard gates
  passed on all three exact shapes. Direct balanced `auto` timing improved every cell by
  `3.3161%`--`9.3197%`, but the exact 48/64/64-call aggregate saved only `22.754199982 ms`
  (`402.936414719` to `380.182214737 ms`), below its fixed `30 ms` architectural-worthiness gate.
  It is terminally rejected without an adjacent loader-role sweep. The immutable report is
  `profiles/bench/r9700-a8q4-b128-loader-p2048-ab-20260905.json` (SHA-256
  `91dd3cda626dcb201da6ad39213e35abfd619e707ffe0f14be69ec59c431f695`); qualification-only
  source/checker/build surfaces are removed and production remains unchanged.
  The structurally distinct persistent `r9700-q4g64-n16-k16-v1` layout has now passed its
  disconnected phase-1 gate and is selected for coherent production cutover. It is a byte-size-
  preserving permutation: codes are `[N/16][G][4 K16 pairs][16 rows]` and scales are
  `[N/16][G][16 rows]`, so each cooperative prefill request is an aligned 64-bit pair and each
  decode wave reads a contiguous N16 tile. Exact row-distinguishing and independent offset/oracle
  checks, full incumbent bit parity, status/tail/null/alignment/overlap/no-clobber guards, and
  native-IU4/resource gates passed. Prefill emitted 94 VGPR, 17,152-byte LDS, occupancy 16, eight
  IU4 sites, and zero spills; decode emitted 43 VGPR, no LDS, occupancy 16, four IU4 sites, and
  zero spills. Direct `auto` timing saved `54.537488 ms` over the exact 48/64/64 P2048 inventory
  while every cell passed `<=1.01`; weighted T1 decode ratio was `0.4179170195`. The immutable
  gate report is `profiles/bench/r9700-a8q4-n16k16-layout-gate-20260905.json` (SHA-256
  `75e7a2881b1f1a256b749684829b27d6bfdb246e544d148a0b230648eb883434`). This admits one atomic
  artifact/codec/binder/reference/all-Q4-consumer migration with no old-layout compatibility or
  runtime repack. It does not itself claim a whole-prefill floor pass: the matched production
  binary and directly stored artifact must first cover every supported Q4 prefill/decode/fused,
  embedding, Vision, MTP, and DFlash consumer and then run the representative whole P2048 gate.
  The coherent codec/binder/reference/all-Q4-consumer cutover and its matched offline artifact are
  now complete. The immutable source
  `out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-eval.ninfer` remains 21,553,545,216 bytes with
  SHA-256 `1dfe9626fd6412592f87480a2f6934e8494a4b267693a25831dff490959542ce`. The fresh
  `qwen3.8-27b/r9700-q4g64-f8e4m3-four-role-n16k16-eval` artifact is
  `out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer`, 21,553,549,312 bytes, SHA-256
  `040c6e7ed29c856718a638c00181975710d987b7d5f49f4cafbdf68911f7e7d2`. Reopening it through the
  production parser confirmed the exact ordered 1,124-object selected inventory and all 295 Q4
  descriptors; independent complete logical code/scale hashes matched every migrated Q4 object,
  every non-Q4 payload hash matched its source object, the source hash remained unchanged, and no
  staging leaf remained. The matched retained-production confirmation used the exact N16/K16
  artifact, dense G16 binary, C1/P2048/chunk4096, and `draft_tokens=0` (no speculative execution).
  It measured `1.108835254 s` / `1,847.942898 tok/s` over three measured repetitions, improving
  the clean pre-cutover GDN-bundle result by `54.391546 ms` and `87.321121 tok/s`. The immutable
  report is `profiles/bench/r9700-four-role-n16k16-dense-g16-p2048-c1-spec-none-20260905.json`
  (SHA-256 `612d667eea61aebdee9a7d12add99bfd90d31b21f155b6114e0b3ca2361d0338`). This confirms the
  isolated layout gain at whole-model scope but does not pass the `2,000 tok/s` floor: another
  `84.835254 ms` must be removed before the minimum, and practical-ceiling attribution remains
  mandatory before chunk selection or the capacity gate.
  Combining N16/K16 with the previously promising wave-role `global_load_b128` loader was also
  tested as one bounded adjacent candidate after correcting its activation addressing to the
  production token-major layout. Independent oracle, bit-parity, guard, tail, and status checks
  passed; the kernel emitted 96 VGPR, 17,152-byte LDS, occupancy 16, eight IU4 sites, exactly two
  `global_load_b128` sites, and zero spills. Direct balanced `auto` timing nevertheless regressed
  every exact shape by `7.2%`--`8.1%`, losing `28.541704 ms` over the 48/64/64-call aggregate.
  It is terminally rejected and its qualification-only surfaces are removed. The immutable report
  is `profiles/bench/r9700-a8q4-n16k16-b128-p2048-gate-20260905.json` (SHA-256
  `6d1e95dd74126434c96c7d77563905da3275efcb27b79e36f95830cc6c93210b`).
  CTA ordering and grid swizzling are terminal at the same boundary. Production maps N tiles to
  grid X and token tiles to grid Y, but physical CU issue order is not contractual. The retained
  exact group-M=4 traversal saved only `5.37676 ms` on its historical mix and failed one P2048 cell
  at `1.01518x`; reweighting those medians to the current `48/64/64` inventory projects a
  `-1.16323 ms` saving. The fresh short-K addressable bucket is `165.810625 ms`, so admission would
  require a `15.077%` reduction despite already measured `93.8037%`--`95.5576%` GL2 hits. A
  hypothetical group-M=8 may reduce an optimistic resident-weight-set proxy below 8 MiB for the
  short-K shapes, but that is not scheduling evidence; group-M=4 and the exact W-only cache-policy
  qualifier both show effects far below the threshold. Full X/Y swap, Morton, and larger grouped-M
  traversals preserve the same trade and worsen the long-K resident-set model. No new swizzle
  qualifier follows.
  A fresh selected-region trace of this exact retained N16/K16 route attributes `1,068.137022 ms`
  of the `1,097.333724 ms` Text-prefill wall to active kernels, leaving only `29.196702 ms`
  kernel-inactive wall. The N16 Q4 projection kernel remains the largest actionable family at
  `356.276973 ms` across 176 calls, or about `61.7223` useful TMAC/s over the exact production
  inventory. Its same-work service time at the `120`-TMAC/s practical-ceiling gate would be
  `183.251938 ms`, a `173.025035 ms` saving, so Q4 alone retains more than enough credible gap to
  close both the `84.835254 ms` minimum-floor deficit and the `25 ms` dominant-bucket criterion.
  Chunk selection therefore remains blocked. Secondary observed families are the selected FP8
  library projections (about `306.172272 ms` across their two emitted solution families), dense
  attention (`148.044163 ms`), and GDN recurrence (`139.215284 ms` at the stage boundary). The
  attribution is `profiles/rocprof/r9700-n16k16-production-p2048-trace-20260905/analysis.json`
  (SHA-256 `3466e40d50ec96b9377db06791eda7f7947c1811f0be3a23d619b392490accbd`); it selects the next
  optimization boundary but does not replace unprofiled timing.
  A fresh trace-wide non-Q4 audit confirms that no unclosed secondary boundary meets the `25 ms`
  admission threshold. After the already-terminal FP8-library, dense-attention, and GDN-recurrence
  families, the largest unresolved bucket is only `22.662237 ms` for 96 BF16 GDN A/B projections;
  even deleting it cannot qualify, and its prior complete A/B-plus-control boundary was only
  `23.549814 ms`. The next buckets are fused SiLU-to-A8 at `20.761231 ms`, FP8 activation
  quantization at `14.616312 ms`, K5120 RMSNorm at `13.322712 ms`, GDN direct scatter at
  `12.543306 ms`, residual adds at `12.354227 ms`, ordinary A8 quantization at `8.850520 ms`, and
  gated RMSNorm at `6.589754 ms`; their coherent adjacent fusions are already terminal. The
  `29.196702 ms` marker-minus-active interval is not an attributable service bucket, and saving
  `25 ms` there would require eliminating `85.63%` of all gaps across 2,089 dispatches despite
  `97.3393%` active coverage. This leaves the Q4 boundary, rather than a fragmented non-Q4 sweep,
  as the principal exact-format optimization target.
  The `306.172272 ms` selected-FP8-library bucket is terminal for the separate `>=25 ms`
  architectural threshold. The exact trace split is `253.819158 ms` for 64 post-mixer gate/up
  calls (`184.10` useful TFLOP/s), `23.799365 ms` for 48 GDN query/key calls (`173.25` TFLOP/s),
  and `28.553749 ms` for 32 dense-attention query/key/gate/value calls (`168.47` TFLOP/s). All 791
  installed gate/up entries have already reduced to ten supported solutions and all ten were timed;
  selected solution 123104 remains best, while the nearest solution saves only a projected
  `0.212524 ms` over 64 calls. The qualified custom M128xN128 native-FP8 route regressed by
  `258.24 ms` over those calls and M128xN256 is resource-terminal. The two smaller-shape families
  total only `52.353114 ms`; saving 25 ms would require `326.60` TFLOP/s including outer scales and
  BF16 publication, versus the observed `168`--`173`. Even complete elimination of the downstream
  gate/up SiLU/A8 consumer is bounded at `20.761231 ms`, and the installed library has no fused
  paired-SiLU/group-quant epilogue. No further FP8 catalog, custom matmul, or adjacent fusion
  qualifier follows under this threshold.
  A structurally fused dense-attention retry is terminal at this boundary. The prior physical
  Bq16/Bk64 online-softmax route already combined native BF16-WMMA QK, FP32 running
  maximum/denominator/numerator state, and direct signed-INT4-times-FP16-scale PV without a global
  score or probability matrix. At P2048/G16 it measured `64.364246 ms` per layer (the later
  full-score comparator measured `61.569540 ms`) at 217 VGPR, 37,160-byte LDS, occupancy 6, and
  zero scratch/spills; the retained report is `profiles/bench/r9700-dense-prefill-attention-ab.json`
  (SHA-256 `d7a0f50a6ae91448583b07d7477d2926b950b65eeadd3ac924f28a8354ed95a6`). The fresh
  `148.044163 ms` / 16-layer bucket must fall to about `<=123.044 ms`, or `<=7.69025 ms` per layer,
  to save 25 ms, requiring more than an eightfold improvement over that physical online route.
  A six-head-shared fused CTA does not supply it: retaining all `6*16*256` FP32 output
  accumulators recreates the proven high-VGPR/low-residency envelope, while feature-group
  partitioning rereads Q/K or spills the 96-KiB output tile beyond LDS. Even before online
  rescaling and synchronization, removing the two remaining causal score-plane transfers at the
  measured streaming read/write rates plus the entire `8.381 ms` maximum stage is only about
  `18.9 ms` over 16 layers. No further online-fused tile is admitted without a changed numerical
  or cache-layout contract.
  N16/K16 does not reopen the terminal M64xN256 decision: its improved persistent-weight
  coalescing applies symmetrically to N128 and N256. N256 still changes only activation rereads and
  CTA/barrier count while preserving weight bytes, wave/IU4 work, reconstruction, and accumulation.
  Even its impossible request-only bound saves at most `48.594 ms` over the 64 P2048 down calls;
  the resource-sane 32-wave N256 was already physically slower at every tested extent, and the
  N16+b128 composition independently regressed every exact shape. No adjacent N256 challenger is
  admitted.
  The distinct N16/K16 1,024-thread M128xN128 two-bank ping-pong candidate was then tested once.
  It halved persistent-weight rereads, emitted 89 VGPR, 25,600-byte LDS, occupancy 16, eight IU4
  sites, two workgroup-local barrier pairs, the intended successor-load overlap, and zero spills;
  full production bit parity plus the independent sampled represented-formula oracle and guards
  passed. Physical `auto` timing nevertheless regressed all three exact shapes by
  `2.886%`--`3.781%`: the 48/64/64-call aggregate moved from `348.631420` to `360.949205 ms`, a
  `-12.317785 ms` saving. It is terminally rejected with no whole run or adjacent M128 variant.
  The immutable report is
  `profiles/bench/r9700-a8q4-m128n128-n16k16-pingpong-p2048-ab-20260905.json` (SHA-256
  `e17f97d896dfc65a7f9358be85830e07632aec0de770baed0fa713fd20884e85`).
  The installed library/codegen escape routes are also terminal at the exact production formula.
  CK 1.2.0 exposes native gfx1201 K16/K32 IU4 only for packed-I4 by packed-I4; its quantized
  packed-W4 GEMMs convert to FP8/BF8 compute and do not accept the stored FP16 G64 scale plane.
  rocWMMA 2.2.1 has no packed-I4 type or specialization on gfx1201. hipBLASLt 1.4.1 accepts the shared
  `HIP_R_4I` type name syntactically, but its 292-file gfx1201 solution package contains no I4
  family or code object, and its scale modes cannot express independent FP16 activation and weight
  scales at each inner-K G64 boundary. Only the compiler K32 IU4 builtin can express the required
  primitive, and retained production already invokes it directly for unsigned-low/signed-high A8
  reconstruction against signed W4 before the per-G64 FP32 scale/FMA and BF16 output. At the fresh
  `356.276973 ms` Q4 service, a `25 ms` whole saving would require `<=331.276973 ms`
  (`<=0.929830x`, `>=1.07547x`), but none of these installed surfaces yields a callable candidate.
  Splitting A8 into two library I4 GEMMs still lacks the combined per-G64 accumulation and would
  require partial storage or 80--272 group launches per projection. No library qualification or
  GPU run follows. A new target-specific kernel architecture remains a distinct possible route;
  this decision closes only wrapping the installed rocWMMA, hipBLASLt, CK, or compiler interfaces
  as if they supplied an exact complete GEMM.
  The final exact-G64 target-specific escape route is now terminal as well. A qualification-only
  standalone gfx1201 assembly kernel preserved the production ABI, N16/K16 layout, exact signed
  IU4 reconstruction and per-G64 FP32 accumulation while explicitly maintaining a two-entry
  global-load FIFO. The first physical regression exposed a real pending-request WAW hazard:
  slot-A weight requests could complete into `v55:v56` after the parity-B path had reused that pair
  for an activation address, and the resulting TCP permission-fault addresses contained the
  packed-Q4 payload pattern. The bounded repair dedicated `v93:v94` to refill address formation;
  subsequent review also corrected the G+2 activation/weight-scale cursors (`+4`/`+64` bytes) and
  the final-group skip/drain/parity sequence. The rebuilt code object passed the frozen static,
  metadata, ABI, CFG, request-order, resource, mutant, graph-replay, status, tail, alignment,
  canary, incumbent-bit-parity, and sampled complete-K FP64 gates. It used 95 VGPR, 23 SGPR,
  17,152-byte LDS, zero private/scratch/spills, compiler occupancy 16, and reported three maximum
  active 512-thread blocks per CU through the distinct HIP runtime occupancy query. Direct balanced
  `auto` timing nevertheless regressed every exact P2048 shape: candidate/incumbent ratios were
  `1.0391881546`, `1.0410894525`, and `1.0324631967`; the exact 48/64/64-call aggregate moved from
  `367.501329422` to `380.711719513 ms`, a `-13.210390091 ms` saving. It is terminally rejected
  with no production or whole-model run. The immutable report is
  `profiles/bench/r9700-a8q4-n16k16-hand-fifo-p2048-ab-20260905.json` (SHA-256
  `8d9bbc228a1ab18fb7482f809089b93e2d53d50396209675f6182dfceab9482e`); all qualification-only
  source, assembly, checker, harness, runner, validator, test, object, code-object, and executable
  surfaces are removed.
  A K128 two-G64 M64xN128 slab does not reopen this result. Packing one G64 payload in each of the
  existing 8,576-byte banks gives a 17,152-byte K128 slab and can halve barrier cadence, but leaves
  no free bank in which to publish the successor slab: its global requests serialize between pairs
  instead of overlapping the current eight-IU4-plus-epilogue body. Preserving the production
  ping-pong overlap requires two such slabs, 34,304-byte LDS, with the associated residency loss;
  either form retains all weight/activation bytes, 16 IU4 sites per K128, signed reconstruction,
  scale, and FP32 accumulation work. Even the generous issue-plus-wait ceiling for eliminating half
  the barrier cadence is about `19.41 ms` over the exact inventory, below the fixed `25 ms`
  architectural boundary and far below the `173.025035 ms` practical-ceiling opportunity. There is
  no same-input projection pair that could amortize the slab across calls. No K128 qualifier or
  adjacent exact-G64 variant follows; the exact-G64 architecture is terminal at this boundary.
  The represented-format contingency therefore advanced only through its bounded next step.
  The qualification-only CPU diagnostic samples the
  exact 160 P2048-dominant Text Q4 matrices (64 MLP-down, 48 GDN value-Z, and 48 GDN-output), with
  eight deterministic rows per matrix. Source-MSE Q4G128 measured `1.021608707x` aggregate
  relative-L2 versus canonical Q4G64; every role also passed the fixed `<=1.03x` screen at
  `1.020869417x`--`1.022098216x`. The independently reopened report is
  `profiles/bench/r9700-selective-q4g128-mse-source-screen-20260905.json` (SHA-256
  `a5389d6597be7a9bed3e75a112b64e8d7f9d8a18411fc065b153f363aa144e09`). This is neither a
  registered format nor artifact/runtime/selection authority. The terminal exact-G64 result leaves
  the P2048 floor unresolved and activates exactly one source-only 8K PPL diagnostic; it does not
  authorize an artifact conversion, kernel, chunk sweep, or 32K run. That weight-codec-only gate
  now passes. Over the fixed 4,095 scored positions, selective canonical Q4G64 measured mean NLL
  `1.8778456412`, delta `0.0124727233` versus BF16, and `5/11` new severe positions. Selective
  source-MSE Q4G128 measured mean NLL `1.8869286448`, delta `0.0215557268` versus BF16, and `6/11`
  new severe positions, passing both the `ln(1.05)=0.0487901642` mean-NLL and severe-position gates.
  Its paired mean-NLL delta versus Q4G64 is `+0.0090830036` and remains diagnostic. The independently
  revalidated comparison is `profiles/ppl/selective-q4g128-source-8k-20260905/comparison.json`
  (SHA-256 `5ba80e928150889b4876b34db23188367b75de8fd78f24b3d4d9507cf929f08e`).
  Kernel-feasibility review further narrows what that diagnostic can authorize. Retaining the
  production A8G64 activation boundary with Q4G128 weights leaves both K64 integer dots and every
  reconstruction, conversion, activation-scale product, FP32 accumulation, IU4 instruction, and
  barrier intact; it only removes `5.11705088 GB` of weight-scale requests over the exact selected
  `48/48/64` GDN-value-Z/GDN-output/MLP-down inventory, about `1.493%` of staged requests and an
  estimated `~5.2 ms` of its `350.51449 ms` service. This is terminally below the `25 ms` admission
  boundary, so no A8G64-by-Q4G128 implementation follows. The only materially credible profile was
  paired A8G128-by-Q4G128: it retains the IU4 and code-byte work but can carry the low/high integer
  accumulators across two G64 halves, halving reconstruction, I32-to-FP32 conversion, scale-product,
  and FP32-accumulation epilogues while saving `7.67557632 GB` of activation-plus-weight scale
  requests. The weight-only source diagnostic explicitly decodes weights to BF16 and keeps BF16
  activations/mathematics, so even a pass could not admit that arithmetic profile. Before any format,
  artifact, or kernel implementation, a separate activation-inclusive source/reference 8K gate was required
  over the same 160 logical roles and fixed 4,095 scored positions: compare the complete represented
  A8G64-by-canonical-Q4G64 grouped formula with A8G128-by-MSE-Q4G128, use FP32 per-group
  accumulation and one BF16 linear-output boundary, and gate the candidate directly against the
  retained BF16 authority at mean-NLL delta `<=ln(1.05)` and no more than `11` new `NLL>=10`
  positions. The paired candidate-minus-control result remained diagnostic. This gate was prepared
  at `profiles/ppl/selective-a8g128-q4g128-source-8k-prepare-20260905` (closure
  SHA-256 `13d92a7bdd065248d4c6a520164b866f6917ac3d71ec7262540d6a5ea1163a7f`), with the
  metadata-only no-GPU preflight at
  `profiles/bench/r9700-selective-a8g128-q4g128-source-8k-preflight-20260905.json` (SHA-256
  `f024f63abd035ea8a800213bf0e9800e98e296d21ef4694bb0c824a780c9379f`). Its scorer processes
  source weights in bounded 128-row chunks and streams one K64/K128 group at a time into one
  M-by-N FP32 accumulator; it does not materialize an M-by-N-by-group tensor or a whole decoded
  weight matrix. The activation-inclusive gate now passes: represented A8G64-by-Q4G64 measured mean
  NLL `1.8772519572`, delta `0.01187903925` versus BF16, and `4/11` new severe positions; represented
  A8G128-by-MSE-Q4G128 measured mean NLL `1.8862310771`, delta `0.02085815914`, and `6/11` new severe
  positions. The paired candidate-minus-control delta is `+0.00897911989` with `9` new and `5`
  repaired severe positions and remains diagnostic. The comparison is
  `profiles/ppl/selective-a8g128-q4g128-source-8k-20260905/comparison.json` (SHA-256
  `450edc009b13fb0aba4e790adf0f00edc2439645ec3e1c86b8012eb37889963d`). This admits only a
  disconnected qualification kernel, not a product format, artifact, binder, or runtime path.
  That qualifier was required to retain the 17,152-byte G64-bank pipeline, emit no more than
  96 VGPR at occupancy 16 with zero spills, pass the independent complete-formula oracle, lose no
  exact cell by more than `1.01x`, and save at least `25 ms` over the exact `48/48/64` aggregate.
  Its two banks remain K64 slabs rather than one G128 slab per bank: the even slab owns the one
  represented A/W scale pair for the logical G128 group, the odd slab extends the same low/high I32
  chains, and the epilogue must consume those scales before the even bank is overwritten by the
  next prefetched slab. Static qualification had to prove exactly one scale load per G128, scale-read
  before same-bank overwrite, successor code-load issue before the current WMMA/epilogue, and no
  compiler-inserted premature drain. Stored-layout correctness had to cover the production small-T
  union `{1,2,3,4,8,12,16,24,36,48}` at all three selected shapes, including status, tails, guards,
  and graph replay: ordinary C1--4 contributes T1--4, retained MTP width four contributes
  T4/8/12/16, and DFlash2 verify width 12 contributes T12/24/36/48 at the fixed C1--4 ceiling.
  Complete quantizer-or-fused-quantizer-plus-wave32-GEMM timing had to cover all ten reachable T
  values for each selected shape because the activation grid/predication balance differs even when
  the WMMA tile count is unchanged; T1/T4/T12/T48 were mandatory anchor and weighted-summary cells,
  not substitutes for the omitted widths. Every reachable cell and the exact call-weighted aggregate
  must be `<=1.01x`. The proposal-width-eight and proposal-head-width-seven DFlash companion calls
  remain G64 and are outside this selected G128 route. The P2048 aggregate had to include ordinary
  A8G128 preparation for value-Z/output
  and fused-SiLU-to-A8G128 preparation for MLP-down, never GEMM-only timing.
  Production admission still requires a matched dense G16 C1/P2048/no-spec whole run at or below
  `1.024 s` (`>=2,000 tok/s`), not merely the isolated `25 ms` gain.
  The disconnected qualifier is now terminally inconclusive with no admission after exhausting
  its single frozen-byte repeat. The first batched report is
  `profiles/bench/r9700-a8g128-q4g128-n16k16-batched-ab-20260905.json` (SHA-256
  `62928d0d4d6054b55fb067457ef79e418db4a533e10b2cc4e31eeec7051dfddf`): P2048
  control/candidate medians and conservative ratio bounds were
  `2.331719/2.334753 ms`, `[0.995490,1.007142]` for value-Z;
  `1.233698/1.220992 ms`, `[0.981957,0.997511]` for GDN-output; and
  `3.555652/3.565343 ms`, `[0.997276,1.008189]` for MLP-down. Its exact
  `48/48/64` weighted saving was `-0.155958 ms`, conservatively
  `[-2.511666,+2.199750] ms`. Five isolated range outliers failed the fixed 6% stability gate:
  value-Z T1 control was `50.7115%`, GDN-output T12 candidate `26.2646%`, GDN-output T36 control
  `27.3622%`, MLP-down T36 candidate `15.5050%`, and MLP-down T2048 candidate `10.8517%`. The allowed
  repeat is `profiles/bench/r9700-a8g128-q4g128-n16k16-batched-repeat1-ab-20260905.json`
  (SHA-256 `2bcbd267a99bf0316695732d9431cde65d16706dc405de68985220ed23c04287`):
  value-Z was `2.345716/2.351652 ms`, `[0.994598,1.010529]`; GDN-output was
  `1.235031/1.223824 ms`, `[0.977557,1.004339]`; and MLP-down was
  `3.562162/3.572732 ms`, `[0.996479,1.009496]`. Its weighted saving was
  `-0.423443 ms`, conservatively `[-3.595587,+2.748701] ms`; one GDN-output outlier failed
  stability because its P2048 candidate range was `15.3891%`. Both independent reports used
  128 calls per P2048 event interval, seven alternating route samples,
  and `3*1.4826*MAD` bounds. Small-T speedups remain qualification diagnostics only. Neither run
  has a robust point estimate or computed interval above `3 ms`, but their failed stability means
  those intervals are not a stable physical speed bound and the two reports are not pooled. The
  repeat allowance is exhausted, so there is no third run and no admission; no Q4G128 format,
  dual-scale artifact lane, binder, or runtime implementation follows. This does not complete or
  weaken the separate `>=2,000 tok/s` whole-model gate, and chunk selection remains blocked behind
  that unmet whole-model gate.
  A separate two-bank next-token pipeline for the remaining GDN scale-sidecar recurrence was
  admitted only to device compilation. It preserved the exact recurrence/state order and passed
  its resource envelope at 57 VGPR, 2,064-byte LDS, occupancy 16, two local barrier pairs, and
  zero scratch/spills, but the emitted ISA drained both successor vector loads and inverse/control
  scalar requests before the current-token recurrence FMAs. The intended latency overlap was
  therefore absent. It is terminally rejected at the frozen static gate without GPU regression or
  timing; a barrier-count-only variant has no credible `>=25 ms` whole saving. The immutable
  receipt is `profiles/bench/r9700-gdn-scale-sidecar-pingpong-static-rejection-20260905.json`
  (SHA-256 `5465b90c100d9b93a1644f8143efc5b042230eabc13e865fc80a0fed3700b290`), and all
  qualification-only surfaces are removed.
  Folding inverse-norm production into the direct-scatter kernel is also closed at design scope:
  it is bit-exactly feasible with a 2,048-byte BF16 shared panel and one barrier, but the retained
  sidecar preparation costs only `0.03379999846 ms` per layer, so perfect elimination can save at
  most `1.622399926 ms` over 48 layers before accounting for replacement LDS traffic and barriers.
  That is only `1.17%` of the remaining floor gap; no implementation or GPU run follows.
  The complete GDN recurrence bucket is therefore terminal under the `>=25 ms` architectural
  threshold without changing its FP32 persistent-state semantics. The fresh stage is
  `139.215284 ms` across 96 dispatches: `137.916357 ms` for 48 main recurrence calls and about
  `1.298927 ms` for sidecar preparation. Retaining preparation requires the main kernel to reach
  `<=112.916357 ms` (`>=1.2214x`) to save 25 ms. Production already keeps each 128-value FP32 state
  row in registers, coalesces its initial/final transfer, uses eight-lane row groups and precomputed
  inverse norms, and preserves the ordered FP32 FMA dependency chain. Grouped-head serialization,
  affine scan/chunking, output/RMSNorm/A8 fusion, inverse-norm folding, and two-bank successor
  prefetch are measured or statically terminal. The remaining three-head-cohort sketch removes
  only duplicated Q/K staging, roughly 2% of the arithmetic before cache effects, while retaining
  every serial state FMA/reduction and cannot credibly provide the required 18.1% stage reduction.
  Larger all-row schedules add barriers, FP32-input WMMA is unavailable, and BF16 state or parallel
  reassociation changes the state contract. No adjacent recurrence qualifier follows.
  No other semantics-preserving P2048 kernel, fusion, or launch-schedule fallback reaches the
  `25 ms` admission bound. After the terminal Q4, selected-FP8, dense-attention, and GDN-recurrence
  buckets, the remaining summed kernel service is `126.465547 ms` and fragmented: BF16 GDN a/b
  linears are `22.662237 ms`, the already-promoted fused SiLU-to-A8 path `20.761231 ms`, FP8
  activation quantization `14.616312 ms`, K5120 RMSNorm `13.322712 ms`, promoted GDN direct scatter
  `12.543306 ms`, residual adds `12.354227 ms`, ordinary A8 quantization `8.850520 ms`, GDN gated
  RMSNorm `6.589754 ms`, feature extraction `4.772466 ms`, and every other symbol `<=1.998 ms`.
  The only adjacent pair above 25 ms, residual plus K5120 RMSNorm at `25.676939 ms`, already
  regressed `2.94x` in its exact fusion. Perfect FP8-quantizer elimination is below the bound, and
  attention input sharing removes only 16 of 144 calls. The `29.196702 ms` marker-minus-active-union
  residual across 2,089 eager dispatches is not a measured host-overhead bucket: saving 25 ms would
  require eliminating `85.63%` and leave about 2 microseconds per dispatch despite profiler queue
  interception and unavoidable gaps. Decode Device Graph evidence does not admit a capture-safe or
  faster 2,089-node prefill graph. An exact follow-up of the fresh trace closes persistent/grouped
  Q4 dispatch independently: the 176 Q4 kernels consume `356.276973 ms`, while all 349 positive
  boundaries touching them total only `3.800482 ms` and no single boundary exceeds `22.04 us`.
  Same-shape calls remain separated by required attention, GDN, SiLU, residual, and normalization
  producers, so neither grouping nor a resident grid can reduce the in-kernel service loss.
  Cross-Op represented-input reuse is also closed: the 112 ordinary A8 preparations correspond
  exactly to 48 value-Z, 48 GDN-output, and 16 attention-output calls, while the other 64 inputs
  already use fused SiLU-to-A8 preparation. Every represented input is distinct, and even perfect
  removal of all ordinary A8 preparation is only `8.850520 ms`. A dedicated-loader-wave Q4 design
  has no omitted operation or occupancy gain: production already issues successor payload loads
  before all eight IU4 WMMAs and overlaps them across the compute body, while the closest measured
  wave-role b128 mechanism lost `7.659%` weighted. No qualifier follows from these three designs.
  ROCm 10 SPM is not an alternate causal-attribution route on this installation: gfx1201 is listed
  as an SPM-capable agent, but every one of the 86 exposed metrics is marked `SPM: Not Supported`.
  The fresh N16 trace instead bounds the physical question directly enough to justify one causal
  diagnostic. Its Q4 service performs `21.99023255552` useful TMAC in `356.276973 ms`, or
  `61.7223` useful TMAC/s (`30.73%` of the isolated `200.880451` IU4 ceiling), while the exact
  M64xN128 request model is `364.823707648 GB` (`1.023989 TB/s`) but the unique represented
  footprint is only `14.507049664 GB`. Neither raw GDDR saturation nor pure IU4 issue therefore
  follows. The fail-closed full/half/quarter-CU scaling experiment on the exact production N16
  P2048 Q4 service completed but is invalid/inconclusive under its frozen 10% controls. Measured
  IU4 ratios were `0.551146` and `0.279208`, just outside the nominal `0.5`/`0.25` mask-response
  tolerance; Q4 throughput ratios were `0.602902` and `0.307400`, with the quarter point
  `10.0971%` from the measured IU4 curve. The shared greater-than-L2 stream ratios were much flatter
  at `0.952047` and `0.801406`, so the result directionally favors CU-local service but is not a
  causal authority. The one authorized fresh, unpooled repeat predeclared a 15% nominal mask sanity
  tolerance while retaining the 10% measured-curve classification threshold. It validated all
  controls and exact 176-call inventories, but terminated `mixed_or_inconclusive`: production
  throughput ratios were `0.602442` and `0.307594`, IU4 ratios `0.550918` and `0.279232`, and shared
  stream ratios `0.951367` and `0.801980`; neither comparison curve matched at both reduced-CU
  points within 10%. Its immutable valid receipt is
  `profiles/bench/r9700-n16k16-q4-cu-mask-causal-repeat1-prepare-20260905/result.json` (SHA-256
  `56b6f742563fe7e2f54f94676a17245f1b35b5a1ec4be13a9ed439a67184b538`). No pooling,
  further tolerance change, or repeat follows. The immutable first-run invalid receipt is
  `profiles/bench/r9700-n16k16-q4-cu-mask-causal-prepare-20260905/invalid-result.json`.
  The one cache-policy challenger was also qualified and terminally rejected at the disconnected
  exact-MLP boundary. It applied gfx1201 high-temporal/device-scope policy only to the two packed-W payload loads for
  `T2048/N5120/K17408`, leaving A and both scale planes at their retained default policy. The long-K
  shape has the weakest legacy GL2 hit ratio (`84.25%` versus `93.80`--`95.56%` for shorter K), and
  the opposite non-temporal policy is already physically terminal. Require unchanged load widths,
  including the incumbent's two static packed-W `global_load_b64` sites, plus otherwise identical
  instruction ordering/bytes, eight IU4 sites, `<=96` VGPR, `17,152`-byte LDS, occupancy 16, no scratch/spills,
  complete represented-format correctness, and a direct ratio `<=0.86874` or saving
  `>=0.390625 ms` per call. The final candidate and baseline HSCOs differed by exactly the two
  intended policy bytes on the exact production 11-argument kernel. Stable timing was neutral:
  `3.085972 ms` control versus `3.086026 ms` candidate, ratio `1.000017`, saving
  `-0.000054 ms` per call. The immutable report is
  `profiles/bench/r9700-a8q4-n16k16-weight-device-ht-p2048-ab-20260905.json` (SHA-256
  `d1f5946a8808f855c67b138ef87e979b2af9f29e26a0806e2c56242d11d961a5`). No hint sweep
  or production change follows.
  A distinct private group-major A8G64 workspace challenger was qualified and terminally rejected.
  It preserves the two
  exact nibble planes, FP16 G64 scales, total workspace bytes, Q4N16K16 weights, two-bank
  `17,152`-byte LDS mapping, barriers, eight signed-IU4 sites, and graph-stable plane addresses,
  but changes code indexing to `[G,T,32 packed bytes]` and scales to `[G,T]`. At P2048 this changes
  each CTA/G64 activation request model from 128 disjoint code segments plus 64 strided scale words
  to 32 contiguous code segments plus two contiguous scale spans, without changing represented
  bytes or arithmetic. The disconnected exact-shape implementation emits 88 VGPR for selected
  prefill, 51 for small/T1, and 10/13 for ordinary/fused producers, all at occupancy 16 with no
  scratch/spills. After repairing a harness-only null-stream error, its complete codec, bit-parity,
  represented-FP64, T1/P2048, fused-boundary, status/guard, and graph regression passed. Stable
  direct timing nevertheless lost every shape: candidate/control ratios were `1.003918` for
  value-Z, `1.004987` for GDN/attention output, and `1.012580` for MLP-down, for a weighted
  `-3.659893 ms` saving. It fails both the every-cell `<=1.01` guard and the required `>=25 ms`
  saving. The immutable report is
  `profiles/bench/r9700-a8q4-group-major-activation-p2048-ab-20260905.json` (SHA-256
  `cccae429dda31f4bef9a71b425ea46bef2ad44388ffd7770327b50d241a684a2`). No production
  cutover or whole-P2048 follow-up follows.
  Lower/private accumulator precision and staged epilogues are terminal without a qualifier.
  Production's 16 FP32 totals are not an occupancy limiter because 96 architectural VGPR already
  retains occupancy 16 and the `17,152`-byte LDS footprint independently limits residency. Even an
  impossible clean removal of two of roughly 50 arithmetic encodings per wave/G64 bounds saving
  below `14.3 ms`; FP16/BF16 per-group accumulation introduces material rounding/overflow risk.
  A single FP32 continuation plane would add `20.4011 GB` of write/read traffic, at least `32.21 ms`
  at the retained stream ceiling, while sequential accumulator tiling removes no work and halves
  independent chains. No production or GPU work follows.
  Activation-only A8G128 with retained Q4G64 weights is also terminal without a numerical gate or
  qualifier. The K128 activation scale is mathematically well defined, but the two distinct Q4G64
  weight scales preserve both K64 epilogues, all 16 IU4 sites per K128, signed reconstruction,
  I32-to-FP32 conversion, FP32 accumulation, payload publication, and barriers. Over the exact
  `48/64/64` P2048 inventory, eliminating every redundant activation-scale request is worth at most
  `4.239 ms` at the retained stream ceiling; even an intentionally generous 50% reduction of the
  complete ordinary-plus-fused A8 producer bucket adds only `14.806 ms`, for a combined ceiling of
  `19.045 ms`, below the fixed `25 ms` admission boundary. Because sharing a K128 maximum changes
  represented activation codes, incumbent bit parity and the prior paired A8G128/Q4G128 quality
  result cannot authorize this profile. No separate BF16 gate, workspace format, producer,
  consumer, or GPU qualifier follows.
  One final exact-format instruction-level candidate was qualified: replace the four full 64-bit
  VGPR payload cursors with exact unsigned 32-bit byte offsets and use the native gfx1201
  scalar-base-plus-`voffset` forms for the unchanged A-low/A-high b32, packed-W b64, A-scale d16,
  and W-scale d16 loads. The exact maximum spans are only `17,825,792`, `44,564,480`, `1,114,112`,
  and `2,785,280` bytes respectively, so no offset can wrap. This can remove nine vector address/carry
  instructions and six associated carry-dependency waits per G64 iteration. Across the exact
  `48/64/64` P2048 inventory, nine issue slots bound at `~16.16 ms`; granting one issue-equivalent
  cycle to each dependency gate raises the optimistic ceiling to `~26.93 ms`, barely above the
  `25 ms` threshold. It is distinct from the terminal hand-FIFO, prefetch, epilogue-scheduling,
  cache-policy, and swizzle families. The one disconnected qualifier had to preserve the
  production 11-argument ABI, load widths/order/policies, successor issue point, `17,152`-byte LDS,
  barriers, eight IU4 sites, reconstruction, scale, FP32 accumulation, and output; require at most
  92 logical/96 architectural VGPR, occupancy 16, zero scratch/spills, complete represented-format
  correctness/status/tails/guards, every shape `<=1.01x`, and at least `25 ms` weighted saving. The
  disconnected wrapper had to reject every shape outside the exact three P2048 tuples because a
  later generic production route would need checked span arithmetic and fallback before using
  32-bit offsets. This was an incremental physical-ceiling probe, not by itself a credible floor
  closer: its `~26.93 ms` generous estimate projects only `~1,893 tok/s`, whereas opening the floor
  requires a directly confirmed `84.835254 ms` whole saving. A direct qualifier pass would permit only
  matched whole confirmation; unless that whole result closes the full deficit and satisfies the
  practical-ceiling conditions above, the broad item still requires a product-contract decision.
  The first run was valid but inconclusive because the GDN-output control interval failed the
  frozen range-stability gate. It nevertheless showed the same directional gain in all cells:
  robust candidate/control upper bounds were `0.936782`, `0.939132`, and `0.939203`; the weighted
  point saving was `23.812737 ms`, uncertainty `0.979192 ms`, and robust lower bound
  `22.833545 ms`. Its immutable report is
  `profiles/bench/r9700-a8q4-n16k16-scalar-base-p2048-ab-20260905.json` (SHA-256
  `1941b3c6a262a61928d3971fc2b0afe06f7dac16db68a4221e79ae42940afb9b`). The sole fresh,
  frozen-byte, independently judged, unpooled repeat was also inconclusive, this time because the
  MLP-down arm failed stability. Its robust upper ratios were `0.935613`, `0.939195`, and
  `0.940472`; the weighted point saving was `23.880331 ms`, uncertainty `1.114011 ms`, and robust
  lower bound `22.766320 ms`. Its immutable report is
  `profiles/bench/r9700-a8q4-n16k16-scalar-base-p2048-ab-repeat1-20260905.json` (SHA-256
  `6501bc10d6dcbfb7638fd3ad977555231e263e3642b184ad15151f3feb9714dc`). Neither run reaches the
  `25 ms` robust lower-bound gate; the repeat allowance is exhausted, so there is no admission,
  pooling, tolerance change, third run, address-form variant, production cutover, or whole run.
  The existing A4Q4 route is also terminal as implemented: its retained relevant P2048 cells saved
  `40.295811 ms` before N16, but current fused MLP-down has no fused-SiLU-to-A4 producer, leaving
  only a projected `18.360 ms` saving on the routes it can actually replace. Global A4 quality is
  already rejected (`+0.135526` all-Q4 mean NLL and `+0.046148` mixed). All 18 BF16 source shards
  are now locally present, which made A4 the sole remaining represented-format `>=25 ms`
  hypothesis at that checkpoint.
  - [x] Run the frozen numerical-first exact four-role N16 8K gate at
    `profiles/ppl/four-role-n16k16-a4-8k-prepare-20260905`: require mean NLL delta
    `<=0.048790164` and at most 11 new positions with NLL `>=10`. Only a pass can authorize a new
    fused-SiLU-to-A4, exact-N16 three-shape qualifier; do not rerun the stale row-major A4 harness.
    The exact 4,095-position run failed: PPL was `6.930091412` versus BF16 `6.458343869`, and mean
    NLL was `1.9358730039`, delta `0.0705000860`, exceeding the limit by `0.021709922`; new severe
    positions were `7/11`, but both gates are required. Its immutable schema-v6
    report is `profiles/ppl/four-role-n16k16-a4-8k-20260905/results.json` (SHA-256
    `cd449ecddbf7f2d260fb43d630b1274ae986631114bf6114c99f3253f0c6b8f5`). A4 is terminal with no
    fused-A4 or physical qualifier.
  A final bounded exact-Q4 architecture review found and qualified one topology not represented by
  the earlier M128xN128 or M64xN256 probes: an M96xN256, 768-thread/twenty-four-wave CTA in which
  each wave owns M16xN64 and reuses its staged A operands across four N16 output fragments. The
  disconnected gfx1201 candidate passed full incumbent BF16 parity over all three exact P2048
  shapes, 126 independent complete-K represented-FP64 samples, M96 tail/VMM-read-guard, graph,
  status, overlap/alignment, and allocation-guard checks. Its emitted object had `105` logical /
  `112` allocation-rounded VGPR, `30,208`-byte LDS, compiler occupancy `12` waves/SIMD, two derived
  active CTAs and 48 resident waves per CU, exactly 16 ordered signed-IU4 sites, a 72-byte ABI, and
  zero private/scratch/spill storage. Physical timing nevertheless stably rejected the topology in
  every cell: robust candidate/control upper ratios were `1.050602`, `1.062665`, and `1.053037`;
  matched point service was `390.703442 ms` versus `371.530018 ms`, only `56.283693` useful TMAC/s,
  and the robust upper service was `392.300470 ms` versus the exact `271.441719 ms` floor-closing
  gate. The immutable report is
  `profiles/bench/r9700-a8q4-n16k16-m96n256-p2048-ab-20260905.json` (SHA-256
  `c09f5b716cf8bce90ceac8e0c7c81e2621713610061582e0e66941a61cd9e20f`). The stable loss admits no
  repeat or production transfer; qualification-only code and tooling are removed. The accompanying
  Layer-0 review also excludes exact-Q4 split-K and unchanged-tile persistent/cooperative forms:
  minimal two-way split-K adds at least `63.753421 ms` of FP32 partial traffic at the retained
  stream ceiling, while unchanged-tile persistence can remove at most the measured `3.800482 ms`
  of Q4-adjacent boundaries; larger live reuse tiles reduce to the already rejected M128/N256
  families or require partial-result spill traffic.
  The activation-inclusive A8G128-by-Q4G128 represented-format
  contingency was the final represented-format hypothesis before these two bounded follow-ups. Its
  two independent robust point estimates
  and computed intervals above remain below `3 ms`, but instability prevents treating them as a
  physical speed bound; the permitted repeat is exhausted and admits no product path. The residual
  audit therefore leaves no live represented-format or exact-format `>=25 ms` hypothesis. The
  P2048 floor/practical-ceiling item is blocked on an explicit product-contract decision; do not
  replace that decision with unbounded candidate search or a chunk/capacity sweep.
- [ ] Rerun all 48 post-promotion capacity cells (dense/XAttention times
  all-Q4/mixed/four-role-hybrid times G16/G32, each at C=1..4). Bind the newly measured Device Graph
  executable allocation; this is
  still required even though the aliased split scratch does not raise the modeled global arena.
  The prior CPU command structure in
  `profiles/bench/post-chunk-twelve-candidate-20260905` is a non-runnable template until regenerated
  and refrozen against the new receipt-bound chunk namespace. The all-Q4 and mixed N16/K16 artifacts and
  their adjacent immutable migration receipts are published. The eight already-created
  all-Q4/mixed manifests in the earlier `-n16k16-20260905` roots predate those receipts and are
  superseded, non-runnable, and never rebound or resumed. Once the P2048 gate opens, a fresh shared
  chunk authority must be derived only from screen/finalist roots whose names end in
  `-receipt-bound-n16k16-20260905`, with a fresh campaign, pipeline, and selection authority in the
  same namespace. The post-chunk package then consumes that receipt-bound selection and owns
  exactly twelve capacity matrices x C=1..4. It runs a whole matrix x C=1..4 only for each
  capacity-eligible profile, after validating symmetric dense/XAttention eligibility within its
  recipe/cache-group pair. Each retained schema-v14 manifest must be published atomically in its
  own directory.
  Base selection ranks only matched spec-none ordinary whole rows. Existing MTP3 evidence is an
  optional exact-token/state/graph regression and is not a prerequisite or ranking input.
  Recipe-specific DFlash shortlist/capacity/Pareto remains the selected-only
  downstream gate because its companion artifact depends on the base winner. The required
  shared-runner publication/DFlash-preset patch is complete: runner-owned outputs use
  inode-checked same-directory durable publication and resume validates regular-file ownership and
  report provenance; the focused runner suite passes `48/48`. Dependent package regeneration and
  closure refreshes remain deferred behind the P2048 floor gate and must follow the dependency
  order above; preparation is not physical completion. No
  physical row or allocation measurement is credited until that package executes.

  The prior final-cutover package at
  `profiles/bench/final-artifact-cutover-admission-prepare-20260905` is likewise a non-runnable
  pre-receipt template until regenerated and refrozen. Its CPU-only validator must reopen
  the schema-v7, chunk, six-quality, exact-token, low-context, retained MTP regression in the
  whole/selected-trace evidence, NIAH, selected-source-BF16
  Vision diagnostic, focused, hardware-use, DFlash, and converter-preflight owners; join all twelve
  C1..4 capacity outcomes and the C1..4 whole matrices for capacity-eligible profiles; and publish
  create-only only after every selected-route physical authority passes.
  The final receipt remains absent, so no artifact or product-identity mutation is authorized.

Derive each route from real shapes and phase behavior. Qualify wave mode, WMMA atom, LDS layout,
register pressure, occupancy, cache behavior, launch count, fusion boundary, and scale traffic.
Decode optimization is judged at T=1..8 and the Engine round; prefill is judged at supported chunk
shapes. Do not use final-output plausibility to qualify an individual Op.

Gate: every production route meets its numerical criterion and improves the claimed scope over the
qualified functional baseline. Remove the superseded route once its replacement is accepted.

### Stage 5: graphs and scheduling

1. Capture fixed-address decode rounds after eager semantics and lifetimes are stable.
2. Prefer a finite startup-created graph set for the supported C=1..4 membership and speculative
   width tiers if HIP update constraints make dynamic updates fragile.
3. Capture owned kernels directly. Admit a ROCm library call only after an on-device graph-safety
   and speed test.
4. Replace NVIDIA PDL with fused kernels or explicit stream/event ordering, selected by round-level
   measurement.
5. Requalify graph memory allowance, address stability, eager/graph numerical criteria, and MTP
   state transitions.

Gate: graph execution is the final supported path for applicable decode rounds, is numerically
qualified, and improves real Engine timing without invalid lifetime or capacity behavior.

### Stage 6: kernel tooling and final qualification

1. Keep the gfx1201 decision procedure in `docs/maintainer/kernel-iteration.md` and the executable
   oracle, ISA/resource, timing, and profiling recipes in `tools/r9700`; the superseded
   `tools.kdev` SM120 classifier and CUDA MMA probe are retired rather than preserved as a
   compatibility backend.
2. Use `rocprofv3` for HIP/HSA dispatch, memory traces, and hardware counters; use Radeon GPU
   Profiler or Radeon GPU Analyzer only when they resolve a live attribution/ISA question.
3. Measure real Engine prefill, first-token latency, per-request decode, aggregate decode, VRAM, and
   long-context behavior at C=1..4 with the final artifact and default final KV format.
4. Run affected OpenAI/Anthropic schema and observable streaming/request behavior tests.
5. Replace RTX 5090 build, command, performance, profiler, and hardware documentation rather than
   appending an AMD compatibility section.

Gate: active authorities and executable help describe only the delivered R9700 product, focused
semantic checks pass, and performance claims have direct R9700 measurements with workload,
artifact, ROCm/driver, and relevant settings recorded.

## Required verification summary

- Exact checks: artifact framing, physical codec/layout transforms, integer state/index movement,
  and conversion-time packing.
- Numerical checks: complete logical formulas from represented inputs with decoded signed codes and
  exact stored scales; production-specific tolerances based on the selected arithmetic profile.
- Model-quality checks: each complete integer-weight/FP8-K/INT4-V candidate versus the independent
  BF16-source authority at 8K and 32K, prefill and teacher-forced T=1 decode, production MTP plus
  eager/device-graph pairs. G16 and G32 hold the candidate artifact byte-identical while varying
  only the compiled cache group; each is gated against BF16 rather than treating mutual agreement
  as correctness. Require equal scored positions and sidecar lengths, zero nonfinite values, and
  an explicit tier: accuracy is paired `delta mean NLL <= 0.02` plus new severe-position count no
  greater than `max(4, ceil(0.001 * scored positions))`; capacity-speed is paired delta no greater
  than `ln(1.05)` plus new severe-position count no greater than
  `ceil(0.0025 * scored positions)`. Record BF16 greedy mismatch count/rate diagnostically. Same-candidate
  graph/eager and speculative/ordinary executions
  require exact greedy-token parity. Report paired absolute-delta p50/p95/p99/max and the largest
  signed token positions; never truncate a mismatched pair.
- State checks: FP32 GDN state, KV page ownership/retention, MTP accept/commit/rollback, ingress
  transaction boundaries, and graph/eager address stability.
- Capacity checks: one resident 27B artifact plus workspace, graph allocation, state, and KV at each
  supported concurrency/context point within the 32 GB device. A promoted split-512 attention
  route invalidates every pre-promotion capacity executable even when its aliased scratch leaves
  the modeled global arena unchanged; rerun all 48 C=1..4 cells and bind the newly measured Device
  Graph allocation before reusing capacity evidence.
- Performance checks: Op latency for real shapes, Engine round attribution, prefill/first-token,
  per-request and aggregate decode, and end-to-end serving where claimed.
- Pareto selection: compare only quality-eligible profiles with complete matched whole-inference
  and capacity cells. A profile dominates another only when it is no worse in mean-NLL delta,
  new-severe-position rate, resolved capacity, and every required C=1..4 speed cell, and strictly
  better in at least one. Quality is admission; retain one static-profile winner per recipe, then
  emit one `terminal_production_selection` by maximin matched throughput, capacity, quality-budget
  consumption, and canonical artifact/static identity in that order. The exact retained means
  decide; raw repetition spread is evidence rather than a tolerance. Raw scorer wall time and BF16
  flip rate are not dominance objectives.
- Product checks: text and required multimodal execution, CLI help/defaults, and OpenAI/Anthropic
  request/stream behavior.

## Material risks and live decisions

- The persistent integer weight assignment remains unresolved until the receipt-bound selection,
  capacity, quality, exact-token, post-terminal, converter-preflight, and final-cutover gates;
  NVFP4 is not a candidate. The complete indexed 18-shard BF16
  source is local and target-validator accepted in
  `profiles/ppl/r9700-bf16-source-checkpoint-preflight-20260905.json` (SHA-256
  `4c605982c84fbfc8803fea24ccdd0230277f4f20c8547a38098f6f440bf99fa9`), and the three N16
  candidate artifacts and migration receipts already exist. The remaining boundary is selection
  and physical admission, not checkpoint acquisition or initial conversion.
- INT4 V group 32 is the capacity/speed lead and group 16 is the accuracy challenger. Neither is
  selected by intuition: both must pass the same represented-value attention oracle, paired PPL,
  deterministic-token, long-context, and production-speed gates.
- The R9700 has substantially less memory capacity and bandwidth than the RTX 5090 assumptions in
  current results. Context/KV/workspace limits require a fresh capacity model.
- HIP graph update behavior and library graph safety may require a finite fixed graph set.
- Mixed AMD iGPU, R9700, and NVIDIA enumeration requires deterministic device identity rather than
  ordinal assumptions.
- The recorded performance baseline is the rebooted ROCm 10, HIP 7.15.26333, AMD Clang 23,
  AMDGPU/ROCm driver 7.1.3.31500000, Linux 7.0.0-30 stack. A future stack change invalidates its
  latency selections until the driver repeat gate is rerun.
- RDNA 4 WMMA layouts and performance differ from both NVIDIA and older AMD hardware; hipified PTX
  structure is not acceptable evidence of correctness or performance.

## Completion condition

The overhaul is complete when NInfer builds only for HIP `gfx1201`, loads only the selected
Qwen3.8 R9700 artifact identity, runs its complete supported CLI/serve workload on one R9700 at
C=1..4, uses qualified R9700-native kernels and device graphs, meets numerical/state/capacity
contracts, has direct R9700 performance evidence, and contains no active CUDA/Vulkan backend,
runtime weight repacking, superseded checkpoint lane, or RTX 5090 product instruction. At that
point integrate stable content into the active references and remove this plan.

## External technical sources

- [AMD Radeon AI PRO R9700 product specifications](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)
- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/docs-10.0.0/reference/gpu-specs.html)
- [AMD RDNA 4 matrix-core programming guide](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/)
- [LLVM AMDGPU target and wavefront usage](https://llvm.org/docs/AMDGPUUsage.html)
- [Clang AMDGPU builtin reference](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html)
- [ROCm numeric precision support](https://rocm.docs.amd.com/en/latest/reference/precision-support.html)
- [rocWMMA API reference](https://rocm.docs.amd.com/projects/rocWMMA/en/docs-7.1.0/API_Reference_Guide.html)
- [HIP graph documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)
- [ROCm graph-safe library status](https://rocm.docs.amd.com/en/develop/reference/graph-safe-support.html)
- [HIP stream-ordered allocator](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html)
- [`rocprofv3` usage](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [Current ROCm Linux installation selector](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/multi-version-install/multi-version-install-rhel.html)
- [Vulkan cooperative-matrix specification](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_cooperative_matrix.html)
- [Vulkan BF16 specification](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_shader_bfloat16.html)
