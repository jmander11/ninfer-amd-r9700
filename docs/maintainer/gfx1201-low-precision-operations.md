# gfx1201 low-precision operations

This note records the low-precision matrix instructions available in the selected ROCm 10
toolchain and what NInfer actually uses on the Radeon AI PRO R9700. A datatype name in a generic
HIP header is not by itself evidence that rocBLAS or hipBLAS has a gfx1201 GEMM implementation.

## Native wave32 matrix instructions

The installed CK Tile gfx12 header exposes the following compiler builtins. Each produces one
native `v_wmma_*` instruction for a 32-lane wave. Integer forms accumulate I32. Floating forms
offer both FP32-accumulator instructions and narrower FP16/BF16-accumulator instructions; the
latter are distinct arithmetic profiles and are not substitutes for an FP32-accumulator oracle.

| Operands | Compiler builtin | Emitted gfx1201 ISA |
|---|---|---|
| signed/unsigned INT4, K=16 | `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12` | `v_wmma_i32_16x16x16_iu4` |
| signed/unsigned INT4, K=32 | `__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12` | `v_wmma_i32_16x16x32_iu4` |
| signed/unsigned INT8, K=16 | `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` | `v_wmma_i32_16x16x16_iu8` |
| FP16 x FP16, K=16, FP32 accumulate | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` | `v_wmma_f32_16x16x16_f16` |
| BF16 x BF16, K=16 | `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` | `v_wmma_f32_16x16x16_bf16` |
| FP16 x FP16, K=16, FP16 accumulate | `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12` | `v_wmma_f16_16x16x16_f16` |
| BF16 x BF16, K=16, BF16 accumulate | `__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12` | `v_wmma_bf16_16x16x16_bf16` |
| FP8 E4M3 x E4M3, K=16 | `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` | `v_wmma_f32_16x16x16_fp8_fp8` |
| FP8 E4M3/E5M2 combinations, K=16 | the corresponding `fp8_bf8`, `bf8_fp8`, and `bf8_bf8` gfx12 builtins | matching `v_wmma_f32_16x16x16_*` |

The two boolean integer operand controls select signed versus unsigned interpretation; the final
control is I32 clamp. NInfer's packed Q4 codes remain signed two's-complement nibbles even where an
activation plane deliberately uses the unsigned control. There is no native mixed A8-by-I4 WMMA:
the exact production route must decompose that formula into the available IU4 instructions.

The gfx1201-specific dense floating builtins have no selectable operand modifiers or matrix-reuse
controls. In particular, the narrow FP16/BF16 forms expose no `op_sel` argument: LLVM's generic
intrinsic definition says the legacy selector must be zero on gfx12, and the `_gfx12` Clang
builtins remove it from their signatures. Accumulator width is selected by choosing the distinct
FP32-, FP16-, or BF16-result builtin. Integer signedness and clamp are compile-time immediate
booleans, not runtime fragment metadata.

### Native 2:4 sparse wave32 matrix instructions

The installed CK Tile sparse gfx12 header also exposes `v_swmmac_*` for a 2:4-structured-sparse
matrix A and dense matrices B/C. This family was omitted from the earlier catalog. Its extra
runtime `idx` VGPR packs the positions of A's nonzero elements; it is not an output selector.

| Operands | Compiler builtin family | Emitted gfx1201 ISA |
|---|---|---|
| sparse FP16 x FP16, K=32, FP32 or FP16 accumulate | `__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32`, `__builtin_amdgcn_swmmac_f16_16x16x32_f16_w32` | corresponding `v_swmmac_*_16x16x32_f16` |
| sparse BF16 x BF16, K=32, FP32 or BF16 accumulate | `__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32`, `__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32` | corresponding `v_swmmac_*_16x16x32_bf16` |
| sparse signed/unsigned INT8, K=32 | `__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32` | `v_swmmac_i32_16x16x32_iu8` |
| sparse signed/unsigned INT4, K=32 or K=64 | `__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32`, `__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32` | matching `v_swmmac_i32_16x16x*_iu4` |
| sparse FP8/BF8, all four operand combinations, K=32, FP32 accumulate | corresponding `__builtin_amdgcn_swmmac_f32_16x16x32_{fp8,bf8}_{fp8,bf8}_w32` builtins | matching `v_swmmac_f32_16x16x32_*` |

The sparse integer forms retain the two immediate signedness controls and immediate I32 clamp.
The gfx1201 sparse floating forms add only the sparsity index; neither dense nor sparse gfx1201
forms expose matrix-A/matrix-B reuse. These instructions cannot consume the current dense Q4,
W8, or FP8 artifact bytes without a distinct 2:4 sparse representation and exact codec/oracle, so
their availability does not create a semantics-preserving shortcut for a current production Op.

Clang also declares wave64 mirrors of the gfx12 dense and sparse builtins, and the assembler
accepts the wave64 sparse register form for `gfx1201` when `+wavefrontsize64` is selected. They are
not additional arithmetic formats and are outside NInfer's fixed wave32 target contract; the
tables above intentionally enumerate the callable wave32 forms.

The `__gfx125__` block in the same header is not part of gfx1201. It adds FP32 K=4; FP16/BF16 K=32;
IU8 K=64; FP8/BF8 K=64 and K=128 variants, including narrow accumulation; F4/F6/MX scaled forms;
and matrix-A/matrix-B reuse controls. None may be selected merely because the installed header can
name it. In particular, gfx1201's builtins do not expose the gfx1250 matrix-reuse operands.

Primary toolchain references are
`/opt/rocm/include/ck_tile/core/arch/mma/wmma/wmma_gfx12.hpp`,
`/opt/rocm/include/ck_tile/core/arch/mma/sparse/wmma/sparse_gfx12.hpp`,
`/opt/rocm/include/ck/utility/amd_wmma.hpp`, and
`/opt/rocm/include/rocwmma/internal/wmma_impl.hpp`. FP8 conversion is exposed by
`/opt/rocm/include/hip/amd_detail/amd_hip_fp8.h` and `amd_hip_ocp_fp_cxx.hpp`. On gfx1201,
float-to-E4M3/E5M2 has native `v_cvt_pk_fp8_f32`/`v_cvt_pk_bf8_f32` forms, while decode has native
scalar `v_cvt_f32_fp8`/`v_cvt_f32_bf8` and packed `v_cvt_pk_f32_fp8`/`v_cvt_pk_f32_bf8` forms.
Direct packed FP16-to/from-FP8/BF8 forms are gfx1250-only; code using a higher-level conversion
must still prove the symbol-local gfx1201 ISA it actually emits. The current append/quantization
image emits native `v_cvt_pk_fp8_f32` for float-to-E4M3 conversion.

## BLAS API boundary

Classic rocBLAS does not expose INT4 or FP8 in `rocblas_datatype`: its low-precision GEMM entries
are BF16 and INT8. Classic hipBLAS delegates `hipblasGemmEx` to that backend and documents BF16 and
INT8, not INT4 or FP8. hipBLASLt ships gfx1201 F8/B8 and INT8 solution families but no INT4 family.

NInfer has no rocBLAS, hipBLAS, or hipBLASLt call site and does not link them. Every Linear,
including the row-scaled FP8 projections, runs on repository-owned HIP kernels. hipBLASLt served the
row-scaled FP8 prefill projections until 2026-09-27; the FP8LUT4 prefill GEMM with raw E4M3 staging
measured 15-44% faster per call at T2048 on every protected shape (`docs/performance.md`) and
replaced it, and the library dependency was removed.

### Row-scaled FP8 Linear

The `F8E4M3_ROW_F32S` numeric format with the `row-scaled-k128-v1` storage layout stores an
ordinary unswizzled E4M3 code plane of `N * align_up(K, 128)` bytes, zero E4M3 padding, then a
256-byte-aligned plane of `N` little-endian FP32 dequantization multipliers. An all-zero row uses
canonical positive-zero scale and zero codes. It must not be represented as `row-split-k128-v1`,
which means grouped integer codes plus one FP16 scale per K group. Conversion performs the only
weight quantization; materialization binds the two planes directly and never repacks them. The
`QType`/`QuantLayout` view belongs to `src/core/tensor.h`; framing and plane construction to
`src/artifact/`; the Python codec to `tools/artifact/` and `tools/convert/qwen3_8_27b_r9700/`.

`ops::LinearExecution` (`src/ops/r9700/linear/linear_execution.{h,hip}`) is the Linear
implementation profile. For represented BF16 `X[K,T]`, `fp8_quantize_activation` writes a
token-major E4M3 `[T,K128]` image, `T` FP32 dequantization multipliers, and one status word per
token into a caller-owned region; an all-zero token publishes positive-zero scale and zero codes,
and a nonfinite token sets its status. T <= 32 then runs `fp8_small_t_linear` (one 16-token WMMA tile, or two for T 17..32); T > 32 runs
`fp8_row_scaled_prefill_linear`, the FP8LUT4 prefill GEMM staging raw E4M3 rows. Both compute
`BF16(sum_k w[r,k] a[t,k] * ws[r] * as[t])` with FP8 WMMA and FP32 accumulation, apply both scales
in the epilogue, and publish the canonical BF16 quiet NaN for every element of a flagged token.
Construction rejects a weight whose N or K is not a multiple of 128; every Qwen3.8 protected
projection qualifies.

Each selected Text projection has a loaded-target-owned `LinearExecution`. Program construction
binds one stable caller-owned activation region (sized for the largest startup width) once; there is
no descriptor, heuristic, library handle, or matmul workspace, and nothing is allocated inside a
captured or timed call. The region, the weight, the BF16 input and the BF16 output are mutually
disjoint. Projections of one input may share one quantization (`run_quantized`).
`ninfer_r9700_fp8_row_scaled_linear_qual` checks both routes against an independent FP64 oracle of
the decoded stored codes and both scale vectors on the real shapes (T=1..16 every element; T=17,
300 and 2048 sampled tokens and rows), output guards, and per-token poisoning.

## Current source-to-ISA map

| Current source route | Hardware path and retained ISA evidence |
|---|---|
| `a8q4g64_linear_wmma32_kernel` in `src/ops/r9700/linear/r9700_linear.hip` | Two native signedness forms of K=32 IU4 WMMA reconstruct each A8 x signed-Q4 dot into I32, followed by FP32 group-scale accumulation. `tools/r9700/build/q4g64_linear.s` contains `v_wmma_i32_16x16x32_iu4`. |
| `a8w8g32_linear_wmma32_kernel` in the same file | Two native signed IU8 K=16 WMMAs per G32 group, then FP32 scale accumulation. The same assembly contains `v_wmma_i32_16x16x16_iu8`. |
| `bf16_linear_wmma16_kernel` | Native BF16 WMMA with FP32 accumulation; the assembly contains `v_wmma_f32_16x16x16_bf16`. |
| `qk_wmma_kernel` in `src/ops/r9700/kv/fp8_int4_kv_attention.hip` | Converts BF16 queries to E4M3 and consumes stored E4M3 keys directly with native FP8 WMMA. `tools/r9700/build/kv_op_qual.s` contains both `v_cvt_pk_fp8_f32` and `v_wmma_f32_16x16x16_fp8_fp8`. |
| XAttention rank and B16 sparse consumer in `src/ops/r9700/kv/fp8_int4_kv_xattention.hip` | The packing stage decodes logical FP8 keys into BF16 storage; rank scores that storage with BF16 WMMA, and the B16 sparse consumer applies BF16 WMMA to the selected subset. The serial consumer retained for the dense/tau-one case instead uses scalar FP8 decode and FP32 FMA. `tools/r9700/build/xattention_s16_tau900.s` contains `v_wmma_f32_16x16x16_bf16`. |
| Vision attention in `src/ops/r9700/vision/vision_attention.hip` | Native BF16 WMMA for QK and PV. |
| Split-512 long-context attention | Uses native FP8 WMMA for its T=1 QK form and BF16 WMMA for fixed-width T=4 QK; `tools/r9700/build/split512_attention.s` contains both opcodes. Production dispatches only tree or device-active-row T=4 at context 8,192 and above; host-fixed 1..6-row decode uses the packed BF16/FP16 WMMA route. |
| Production A8Q4/A8W8 prefill CTAs at their exact admitted Cartesian predicates | A8Q4 uses the native low/high signedness pair of `v_wmma_i32_16x16x32_iu4` at the eight qualified shapes and T={1,024,2,048,4,096,8,192}; A8W8 uses two signed `v_wmma_i32_16x16x16_iu8` operations per G32 group at the four qualified mixed-Text shapes and the same four T extents. Both use cooperative LDS reuse. Decode, partial chunks, and every unqualified shape retain the incumbent one-wave route. The immutable schema-v3 reports are the promotion evidence; current assembly remains the source-to-ISA regression boundary. |

The 2026-09-05 CPU-only refresh rebuilt these device images from the current sources. The Q4/W8
Linear image SHA-256 is `25eb70c52233e60a45443f34df2e6c1113b163b44b167bd3733a422fb719d7dd`:
the production Q4 CTA still has eight IU4 sites, 88 VGPR, 17,152-byte LDS, and zero scratch; the W8
CTA has two IU8 sites, 50 VGPR, 4,352-byte LDS, and zero scratch; the one-wave Q4/W8/BF16 leaves
have respectively four IU4 at 64 VGPR, two IU8 at 83 VGPR, and one BF16 WMMA at 29 VGPR, all with
zero LDS/private/scratch. Current KV, split-512, XAttention-S16/tau900, eager, and Vision images
have SHA-256 `7a303e300ec4498177f876294a08d2194eff7cda8c43d6ee581211308f950c91`,
`05937b9398e1ed7c6ae61248b08a103379f687dc3b4c1be5714249f1b1a28cbd`,
`4a918aeaa85d5a91e0b27dda1c55f411813437e80a7c411f08d6e9c18b551c69`,
`f51a2d9363d01d6111e77bf1cff649d2cca5e0c1c16e7aa492e82e360977dca2`, and
`e52263f89224d5cc721529153997ba08fb59ead55c6c5747ae0d268313e02b58`.
The split-512 ISA/resource checker passes; XAttention rank retains two BF16 WMMA sites and its B16
consumer sixteen; Vision attention emits fifteen BF16 WMMA sites but also 336 bytes of private
scratch, so its static evidence must not be described as a zero-scratch path. These are
source-to-ISA proofs only. The final selected Text/MTP/DFlash route still requires dispatch joins
to its embedded or loaded ELF exactly as stated in the execution ledger.

The 2026-09-05 current-candidate executable preflight is retained at
`profiles/bench/r9700-twelve-candidate-hardware-path-static-audit-20260905.json`. It extracts the
unique inner gfx1201 Linear and attention ELFs from each of the dense-G16, dense-G32,
XAttention-G16, and XAttention-G32 benchmark executables. Exact selected symbol bytes agree across
all four builds: Q4 emits four IU4 WMMAs in the wave32 route and eight in the P2048 CTA, W8 emits
two IU8 WMMAs in both routes, ordinary QK emits one FP8 WMMA, dense initial-prefix QK emits sixteen
BF16 WMMAs, and the XAttention rank/consumer emit two/sixteen BF16 WMMAs. No selected embedded
matrix symbol has an explicit load-cache modifier; that means default-temporal policy, not absent
memory traffic. The Q4 CTA alone proves next-group code-load overlap. W8 uses single-tile LDS
reuse, and the former hipBLASLt FP8 path advertised one-stage global-read prefetch, so neither is
described as cross-operation overlap. This is a twelve-candidate static preflight, not proof that
the eventual winner executed those symbols.

## Scalar and software-expanded work

Not every low-precision value operation is a matrix instruction:

- FP8 cache append uses the HIP E4M3 conversion intrinsic and emits native packed FP8 conversion.
  Scalar and packed FP8/BF8-to-F32 conversions are also native gfx1201 operations, but whether a
  higher-level E4M3-to-half expression selects them is a separate code-generation question. The
  retained vector-attention images show compiler-expanded conversion followed by ordinary FP32
  FMA. Long-enough T=1/T=2 QK dispatch instead consumes FP8 codes through native FP8 WMMA. A
  direct native-decode rewrite is worth a bounded qualification only if whole-Op profiling first
  attributes material time to that software decode.
- INT4 value-cache PV decodes signed nibbles and multiplies them by FP32 softmax probabilities and
  FP16 scales. That is not an integer GEMM, so IU4 WMMA is not applicable; the custom kernel uses
  integer unpack plus FP32 accumulation.
- W8G32 exact BF16-activation fallback routes decode signed INT8 and apply per-G32 FP16 scales with
  ordinary FP32 FMA. The qualified A8W8 shape/crossover route is the native IU8 WMMA path.
- Activation quantization, nibble packing, scale application, softmax, and reductions remain scalar
  or vector ALU work around the native matrix instructions. Calling these stages software-expanded
  must not be confused with claiming that their selected Q4/IU8/FP8 matrix multiply is emulated.
  Gfx1201 has no native signed-nibble-to-byte conversion or fused per-group Q4 scale operation;
  those stages remain shifts, masks, permutes, and scalar/vector scale arithmetic around IU4 WMMA.

The generated `.s` files above are reproducible inspection evidence for the current checkout. The
source builtin plus a fresh target-specific ISA/resource gate remains the admission authority after
any kernel change.

### Direct Q4-to-IU8 prefill mapping

Unpacking signed Q4 weights to signed bytes does not reduce the gfx1201 matrix-instruction floor for
the fixed A8G64 boundary. The IU8 builtin consumes K16, so one G64 output fragment requires four
`v_wmma_i32_16x16x16_iu8` instructions with `(true, A, true, B, C, false)`. The selected split-A8
route also requires four native instructions: two K32 halves times its low/high IU4 planes. The
direct form removes the final integer `low + 16*high` reconstruction and one accumulator bank, but
adds A8 reconstruction, signed-nibble expansion, and a serial four-instruction accumulator chain.
On-chip expansion keeps global traffic unchanged. A single-bank form increases the M64xN128 tile
from 8,576 to 12,672 LDS bytes because Q4 weight staging doubles from 4,096 to 8,192 bytes; the
fair expand-once ping-pong form requires 25,344 bytes. Materializing expanded weights would
additionally double their global traffic and is excluded.

The register-only schema-v2 hardware probe measures `200.880451` useful split-IU4 TMAC/s versus
`195.123788` topology-matched IU8 TMAC/s, ratio `0.97134284`; the eight-independent-chain IU8
control reaches `198.502175` TMAC/s. These rates count one multiply-accumulate as one MAC. The
measured relative result closely matches the separately supplied `152.5` versus `148` TMAC/s
hypothesis (`0.97049180`) while establishing that the two-chain/four-dependent-instruction IU8
topology itself loses only `1.70194%` against the opcode-saturation control. This is an isolated
issue ceiling without Q4 unpack or LDS traffic. The complete fair expand-once qualifier was then
built with direct-A8 preparation, packed-Q4 global storage, two 12,672-byte LDS banks, and eight
signed IU8 instructions. It failed the pre-timing resource gate: 115 VGPR/occupancy 12 initially,
then 111 VGPR/occupancy 12 after preventing K16 operand hoisting and removing full-G64 tail-mask
loads, against the required <=96 VGPR/occupancy 16. A compact legal W4 expansion probe bounded
more serialized publication near 100 VGPR, so scale-load reordering could restore memory overlap
but could not make the resource gate credible. Direct packed-W4-to-IU8 is therefore terminally
rejected before GPU timing; the immutable static evidence is
`profiles/bench/r9700-a8q4-direct-iu8-pingpong-static-rejection-20260905.json`, SHA-256
`d4f2c610b973935aa60b732c3ee39e9dc84a512781839458a5caadcca78f2a2c`.

### Installed packed-INT4 GEMM library boundary

The installed CK Tile, rocWMMA, and hipBLASLt surfaces do not contain an exact
mixed-A8/packed-Q4G64 GEMM that can replace NInfer's prefill kernel. CK Tile's gfx12 matrix layer has dense signed/signed
`pk_int4_t` K16 and K32 specializations, but both operands are packed INT4. Its group-quant block
GEMMs accept packed INT4 storage only by selecting FP8 or BF8 compute types; they convert the
stored codes before matrix multiplication. That is a different arithmetic profile from NInfer's
A8 low/high-nibble reconstruction followed by direct signed-Q4 integer accumulation and one FP16
activation-scale times FP16 weight-scale application per G64 group. Installed rocWMMA 2.2.1 has no
packed-INT4 type or specialization; its gfx1201 integer WMMA surface is INT8-by-INT8 K16, so it
does not even provide an IU4 fragment wrapper from which to compose the required group-scaled
kernel. The installed hipBLASLt gfx1201 solution directory contains INT8 and FP8/BF8 families but
no INT4 family or code object. CK Tile's separate `flatmm` tree is not an integer escape hatch:
its mixed-width policies are F16/FP8 by MXFP4, and the microscaling implementation explicitly
admits gfx950 (with a separate gfx1250 route), not gfx1201 A8 by signed-Q4.

CK Tile's strongest relevant scheduling pattern is its V3 two-stage global-prefetch pipeline: it
keeps the next A/B/scale tiles in registers, copies the current tiles through LDS, and overlaps the
next global loads with the current block GEMM. That pattern does not provide a gfx1201 asynchronous
global-to-LDS operation. CK Tile's `global_load_lds` helper is guarded for CDNA3 gfx940/gfx950;
gfx1201 must use ordinary buffer/global loads into VGPRs followed by LDS stores. The installed
gfx1250 path separately exposes byte/dword/dwordx2/dwordx4 asynchronous global-to-LDS operations
and `s_wait_asynccnt`; those are not gfx1201 options. Gfx1201 instead has separate
`s_wait_loadcnt`, `s_wait_storecnt`, `s_wait_dscnt`, and scalar-memory `s_wait_kmcnt` domains,
plus a combined `s_wait_loadcnt_dscnt` encoding and split workgroup barrier signal/wait. Their
nonzero immediates retain that many newer operations, so a pipeline may wait for only the current
tile while keeping newer global loads outstanding; `s_wait_loadcnt 0` before publishing loaded
VGPRs into LDS and `s_wait_dscnt 0` plus the workgroup barrier before consuming that LDS bank are
the relevant safe boundaries. `s_delay_alu` is also accepted on gfx1201, but it is a VALU
dependency-delay encoding rather than a memory-completion primitive. The selected
local-workgroup release, signal, wait, and acquire sequence remains the appropriate ownership
boundary for publishing a staged bank.

Gfx1201 buffer operations expose regular-temporal, non-temporal, high-priority-temporal, and
last-use load hints; near/far-cache combinations `NT_RT`, `RT_NT`, and `NT_HT`; `NT_WB` for
stores; and CU, shader-engine, device, or system scope. The non-speculative temporal hints in the
same installed header are guarded for gfx1250. These modifiers express reuse intent but do not
create asynchronous copies, reduce matrix issue, or reduce represented memory requests.

Local static audit provenance (2026-09-04): installed AMD clang 23.0.0git at commit
`8f497e0992fb7513f7f78a6f6b6f1056c375e961`; Clang builtin feature predicates in
`/opt/rocm/llvm/include/clang/Basic/BuiltinsAMDGPU.inc`; LLVM intrinsic signatures in
`/opt/rocm/llvm/include/llvm/IR/IntrinsicsAMDGPU.td`; cache enums in
`/opt/rocm/include/ck_tile/core/arch/amd_buffer_coherence.hpp`; and gfx12 wait wrappers in
`/opt/rocm/include/ck_tile/core/arch/arch.hpp`. Device-only compilation for `gfx1201` emitted
`v_swmmac_i32_16x16x32_iu4`, `v_swmmac_i32_16x16x64_iu4`, and
`v_swmmac_i32_16x16x32_iu8` with the requested signedness/clamp controls, and the four sparse
FP8/BF8 combinations plus sparse BF16. `llvm-mc -mcpu=gfx1201` also accepted representative
listed cache/scope modifiers, the five wait forms, and `s_delay_alu`; no GPU was executed. SHA-256 at
audit time: dense header `6f60fc1814db65ee819ea70f209bc65011de2fccdbff30cc04809eeb8e87bc87`,
sparse header `ac33f41143bf3008f78d2870e8816d6dec162a4df401cee2a8b49b004d382f63`,
Clang builtin table `7dcd613a84927311964455d6e3d47889167b5d98032664e408fb55795a600c82`, and
LLVM intrinsic table `f06a925c7ac25b092cf708ee85ddf13bd7acd5ba05f1a0a918be40762ec76a64`.

The selected M64xN128 ping/pong kernel is already close to that applicable source pattern: each
loop emits eight fully utilized K32 IU4 instructions, uses two 8,576-byte LDS banks, 88 VGPR, no
private/scratch storage, and two workgroup barrier pairs. Its ISA keeps the next global bank
outstanding during current-bank WMMA. Its original operator promotion result was only `1.23224x`;
the later matched whole-P2048 result selected it for its aggregate saving. Jointly
doubling N and adding that overlap delivered only `1.20609x`. Cache modifiers cannot credibly turn
either measured schedule into the greater-than-`2x` operator improvement now required.

There is substantial theoretical compute headroom--the weighted T=2,048 workload represents
about 202.16 TOP of issued IU4 work, or roughly 0.264 seconds at the R9700's approximately 766 TOP/s
dense-INT4 headline rate--but the installed libraries provide no exact kernel or generated
gfx1201 pattern that realizes it for this mixed-width, per-G64 scaled formula. Consequently there
is no library-derived qualification candidate with a defensible greater-than-`2x` bound. A future
attempt would require a new target-specific kernel architecture or a changed artifact arithmetic
profile, not CK/rocWMMA wrapping, cache hints, or another LDS-pipeline variant.
