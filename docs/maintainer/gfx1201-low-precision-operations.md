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
are BF16 and INT8. Its documented `rocblas_gemm_ex` type combinations likewise stop at INT8.
Classic hipBLAS delegates `hipblasGemmEx` to that backend and documents BF16 and INT8, not INT4 or
FP8. The generic `hipDataType` enumeration in `/opt/rocm/include/hip/library_types.h` contains
`HIP_R_4I` and FP8 values, but that enumeration is shared infrastructure and does not add those
types to classic hipBLAS GEMM.

hipBLASLt is different. Its API consumes `hipDataType`, provides FP8 wrapper types, and this ROCm
installation contains gfx1201 F8/B8 and INT8 Tensile solution/code-object families under
`/opt/rocm/core-10.0/lib/hipblaslt/library/gfx1201/`. It therefore offers dense production FP8 and
INT8 matmul solutions on gfx1201. That directory contains no INT4 (`I4`) solution family, so it is
not a route for NInfer's packed signed-Q4G64 GEMMs. Some block-scaling enum values in
`/opt/rocm/include/hipblaslt/hipblaslt.h` are explicitly marked not supported yet.

NInfer has no rocBLAS or classic hipBLAS call site. Custom HIP kernels remain required for its
packed Q4G64 artifact and paged FP8-K/INT4-V attention contracts. The selectable four-role
row-scaled-FP8 profile uses hipBLASLt through `LinearExecution`; it binds the converter-produced
E4M3 code/FP32 row-scale planes directly, prepares descriptors before timed/captured execution,
and performs no runtime weight repack. Both routes keep activation quantization, page mapping,
softmax, and semantic fusion boundaries under the owning Op.

### Dense FP8 prefill escape-path boundary

The installed hipBLASLt 1.4.1 library does expose a credible, separately quantized dense-prefill
evaluation path. `HIP_R_8F_E4M3` is the OCP E4M3 input type, and the matching gfx1201 problem
catalogs accept E4M3 A and B, BF16 C and D, `HIPBLAS_COMPUTE_32F`, and high-precision FP32
accumulation. The four ordinary transpose families with scalar A/B scaling contain
536/559/224/779 solutions; the directly useful ordinary-layout family additionally has 11
outer-vector-scale solutions. Those decoded records bind `aType=bType=Float8`,
`cType=dType=BFloat16`, `computeType=Float`, `highPrecisionAccumulate=true`, and
`swizzleTensorA=swizzleTensorB=false`. This is shipped executable gfx1201 coverage, not datatype-
enum inference.

The C API boundary is `hipblasLtMatmulDescCreate`, `hipblasLtMatrixLayoutCreate`,
`hipblasLtMatrixLayoutSetAttribute`, `hipblasLtMatmulDescSetAttribute`,
`hipblasLtMatmulPreferenceSetAttribute`, `hipblasLtMatmulAlgoGetHeuristic`, and
`hipblasLtMatmul`. A/B use `HIP_R_8F_E4M3`; C/D use `HIP_R_16BF`; compute and scale use FP32.
`HIPBLASLT_MATMUL_DESC_A_SCALE_MODE` and `_B_SCALE_MODE` admit scalar FP32 scales or
`HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F`, whose A and B vectors contain the logical M and N
factors. The latter exactly supports one activation scale per token and one weight scale per
output row. It does not reproduce G32/G64 inner-K group scaling. Several advertised block modes
are explicitly marked unsupported, and no matching E4M3/E4M3-to-BF16 gfx1201 catalog inspected
here binds an inner-K block-scale mode. Such a mode is therefore not part of this candidate.

Ordinary row/column layouts and leading dimensions can describe a row-major, K128-padded
`[N,K]` weight plane and transpose it logically. The matching solutions explicitly disable A/B
swizzling, and hipBLASLt exposes no persistent offline-weight-prepack object for this family.
Consequently a new artifact may store the final E4M3 code plane followed by FP32 row scales and
bind it directly; runtime weight repacking is neither needed nor allowed. Algorithm selection is
performed once at startup for the fixed shapes. The preference workspace bound and each returned
heuristic's `workspaceSize` must be incorporated into the caller-owned stable arena. Catalog
metadata permits up to four bytes per C element for global accumulation; a conservative reusable
bound is therefore 285,212,672 bytes for P2048 or 1,140,850,688 bytes for P8192 at the largest
N=34,816 output, pending actual heuristic selection.

The existing inventories contain 28,424,681,472 K128-padded matrix code positions and 4,874,224
matrix rows, plus 59,189,728 bytes of non-matrix tensor payload. An outer-vector E4M3 profile is
therefore about 28,503,368,096 resident tensor bytes: one byte per padded code, 19,496,896 bytes
of FP32 row scales, and the unchanged direct payload. It leaves about 4,782,628,448 bytes after a
1-GiB sizing reserve on a 32-GiB card, before small object-alignment differences. The largest
P8192 FP8 activation image is 142,639,104 bytes. Even the conservative 1.141-GB GEMM workspace
therefore fits alongside the fixed cache and runtime much more plausibly than the W8/BF16
fallback profiles, though resolved C=1..4 capacity remains a physical gate.

At P2048 the declared dense-linear inventory is 101.08205531136 TFLOP-equivalents. A 2,000 tok/s
whole-prefill target allows 1.024 seconds, so the linear calls require at least 98.713 TFLOP/s if
they consume the entire budget, or 197.426 TFLOP/s if half is reserved for attention,
normalization, quantization, and other work. One native K16 E4M3 WMMA carries half the operations
of the measured K32 IU4 instruction; applying the measured 48.930-billion-WMMA/s issue rate only
as an architectural scale gives an approximately 400.835-TFLOP/s FP8 ceiling. The stricter
half-budget case is thus about 49.3% of that scale. Large, aligned P2048 shapes and thousands of
shipped HPA solutions make that performance plausible, but not proven without exact-shape
heuristic and physical A/B results.

FP8 is not a drop-in reinterpretation of the selected integer artifact. E4M3 rowwise quantization
has a different represented mathematical oracle and may be worse than the already measured
W8G32 quality despite its exponent range. It needs BF16-source conversion, exact codec and
complete-Op checks, the existing 8K/32K PPL and severe-position gates, and matched whole prefill.
The audit verdict is therefore **go for one bounded evaluation candidate**, not production
selection: row-scaled E4M3 weights bound directly from the artifact, per-token E4M3 activation
quantization, FP32 HPA, BF16 output, startup-fixed algorithms, and no runtime weight repack.

#### Repository integration design

The candidate gets a distinct `F8E4M3_ROW_F32S` numeric format and
`row-scaled-k128-v1` storage layout. It must not be represented as the existing
`row-split-k128-v1`: that layout means grouped integer codes plus one FP16 scale per K group, and
both its geometry and target row-view code depend on that meaning. The new rank-two layout stores
an ordinary unswizzled E4M3 code plane of `N * align_up(K, 128)` bytes, zero E4M3 padding, then a
256-byte-aligned plane of `N` little-endian FP32 dequantization multipliers. An all-zero row uses
canonical positive-zero scale and zero codes. Conversion performs the only weight quantization;
materialization binds
the two planes directly and never repacks them.

The corresponding `QType`/`QuantLayout` weight view belongs to `src/core/tensor.h`; framing,
geometry, parsing, and direct plane construction belong to `src/artifact/reader.{h,cpp}`,
`src/artifact/storage_layouts.cpp`, and `src/artifact/typed_binding.cpp`. The matching Python
vocabulary and exact codec belong to `tools/artifact/numeric.py`, `tools/artifact/layouts.py`, and
`tools/convert/qwen3_8_27b_r9700/`. The target adds one explicitly named all-matrix evaluation
identity in `src/targets/qwen3_8_27b/export/ninfer/targets/qwen3_8_27b/package.h`,
`impl/package_identity.cpp`, `impl/load/bindings.cpp`, and `impl/variant.cpp`; the 439 currently
quantized matrix objects become row-scaled E4M3 while direct BF16/FP32/I32 objects remain
unchanged. `bindings.cpp::row_view` needs an explicit row-scaled branch with a padded-K-byte code
stride and four-byte scale stride, notably for packed attention and MTP parents.

Dynamic activation quantization and matmul execution are one Linear implementation profile owned
under `src/ops/r9700/linear/`. For represented BF16 `X[K,T]`, its HIP staging kernel writes an
ordinary E4M3 `K128 x T` image and `T` FP32 dequantization multipliers into caller-owned workspace;
each all-zero token again publishes positive-zero scale and zero codes. Nonfinite input sets a
device-resident status and zeroes the affected token; the eventual matmul route must enqueue a
post-matmul status consumer that poisons or rejects its output before publication because
hipBLASLt cannot interpret that status itself. hipBLASLt views the artifact bytes as column-major
`[K128,N]`, applies transpose to obtain `W[N,K128]`, and multiplies the column-major activation
image `[K128,T]` to produce the existing column-major/BF16 `out[N,T]`. A uses the N-element weight
row-scale vector, B the T-element token-scale vector, both with
`HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F`; C/D are BF16, beta is zero, and compute is
`HIPBLAS_COMPUTE_32F`.

hipBLASLt lifecycle is explicit rather than a function-static cache. A repository-internal
`ops::LinearExecution` owns the handle, descriptors, heuristic results, and selected algorithms;
the Qwen Program owns one instance and threads it through Text, MTP, DFlash, Vision, and scoring
linear calls alongside the existing arena and stream. It prepares fixed decode/speculative widths
at Program construction and explicitly prepares each realized prefill width before entering that
chunk, so no descriptor creation, heuristic search, allocation, or weight transformation occurs
inside a captured or timed Linear call. Its workspace contract is the aligned sum of the live
E4M3 activation image, T FP32 scales, and the selected heuristic's `workspaceSize`; the static
planner reserves the declared conservative maximum and `LinearExecution` rejects any selected
heuristic that exceeds it. Small-T may select a separately qualified
direct-E4M3 custom leaf, but it consumes the identical stored bytes and is not a second artifact
path.

The first fixed-shape owner retains the directly bound weight and caller-owned activation and
hipBLASLt workspace addresses. Those regions, its represented-BF16 input, and its BF16 output are
mutually disjoint; retained storage remains alive through owner destruction and submitted streams
complete before any input/output storage is released. Calls on one hipBLASLt handle are externally
serialized onto one stream. Its post-matmul status consumer is one CTA:
it reads status once and returns on the finite fast path, or strided-fills the complete output with
canonical BF16 quiet NaNs on rejection.

The isolated evaluation closure adds
`find_package(hipblaslt CONFIG REQUIRED)` and links `roc::hipblaslt` only into the R9700 Linear/core
closure. The C boundary is the installed `/opt/rocm/include/hipblaslt/hipblaslt.h` interface named
above; scale pointers use `HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER` and `_B_SCALE_POINTER`, and the
two scale-mode attributes use `_A_SCALE_MODE` and `_B_SCALE_MODE`.

The first vertical slice is deliberately one real dominant Text projection rather than a whole
artifact conversion: encode and directly bind `text/layers/0/mlp/gate_up` at its exact
`[34816,5120]` shape, dynamically quantize a represented BF16 `[5120,2048]` input, and execute the
same `ops::linear` contract to BF16 `[34816,2048]`. Admission requires exact byte/layout checks,
an independent FP64 oracle that decodes the stored E4M3 bytes and both FP32 scale vectors, no
nonfinite output, retained hipBLASLt heuristic support, direct artifact pointer use, stable
caller-owned workspace, and matched timing against the current Q4 projection. Only a passing
slice justifies threading `LinearExecution` through the family runtime and converting the complete
439-matrix evaluation artifact; whole-profile capacity, PPL, exact-token, decode, graph, and
P2048-prefill gates then decide whether the candidate survives.

The bounded first-slice qualifier times that complete FP8 sequence against the current
`A8Q4G64-m64n128-pingpong-production` candidate boundary at the identical shape, including each
route's activation quantization and output-status publication. Seven paired repetitions each run
FP8/Q4 and then Q4/FP8, yielding fourteen raw samples per route and an even-sample median. Nine
axis-sensitive represented-format FP64 oracle probes span three interior/corner tokens and three
interior/corner rows, so K ordering and both scale axes are observable. The fresh, exclusively
created report retains those results, the complete-time ratio, selected hipBLASLt workspace,
no-clobber/nonfinite checks, canonical source identity, and source/executable SHA-256 values; the
comparison validator rehashes those files, rechecks live power state, and derives the verdict.
Admission is fail-closed to HIP device 0 identifying as the discrete
AMD `1002:7551` Radeon AI PRO R9700, exact `gfx1201` wave32, and `auto` DPM profile both before
allocation and after timing. The report also retains the PCI identity, HIP runtime/driver and
hipBLASLt versions, ordered heuristic count and selected rank, opaque 16-byte algorithm identity,
algorithm workspace limits, waves count, and exact matrix/transpose/scale descriptors.

Executed-instruction admission is a separate post-admission trace. The host-only
`tools/bench/prepare_fp8_gate_up_hardware_proof.py` creates a no-reuse rocprofv3 plan bound to the
successful matched report. Its kernel trace runs the same executable to a separate fresh report.
`tools/bench/validate_fp8_gate_up_hardware_proof.py` requires identical executable/source and
hipBLASLt algorithm identities, joins each exact dispatch through rocprof's kernel-symbol and
code-object tables, extracts the file-backed loaded ELF, and disassembles the dispatched symbol.
It admits FP8 only when that symbol contains native FP8 WMMA/MFMA and admits the Q4 control only
when its dispatched production symbol contains `v_wmma_i32_16x16x32_iu4`; kernel names alone are
not evidence. The same loaded-ELF join retains one exact resource envelope across all sixteen
dispatches: gate/up (also shared by GDN) uses 128 SGPR, 192 architectural VGPR, zero accumulator
VGPR, 25,088 bytes of group-segment LDS, and zero private-segment bytes; attention query/key and
gate/value uses 128/192/0 registers, 12,544 bytes of LDS, and zero private-segment bytes. The
rocprof kernel-symbol schema exposes no SGPR/VGPR spill counts, so the proof records those counts
as unavailable rather than inferring zero. A terminal hybrid trace must match the qualified ELF
SHA and its available dispatch resource fields.

## Current source-to-ISA map

| Current source route | Hardware path and retained ISA evidence |
|---|---|
| `a8q4g64_linear_wmma32_kernel` in `src/ops/r9700/linear/r9700_linear.hip` | Two native signedness forms of K=32 IU4 WMMA reconstruct each A8 x signed-Q4 dot into I32, followed by FP32 group-scale accumulation. `tools/r9700/build/q4g64_linear.s` contains `v_wmma_i32_16x16x32_iu4`. |
| `a8w8g32_linear_wmma32_kernel` in the same file | Two native signed IU8 K=16 WMMAs per G32 group, then FP32 scale accumulation. The same assembly contains `v_wmma_i32_16x16x16_iu8`. |
| `bf16_linear_wmma16_kernel` | Native BF16 WMMA with FP32 accumulation; the assembly contains `v_wmma_f32_16x16x16_bf16`. |
| `qk_wmma_kernel` in `src/ops/r9700/kv/fp8_int4_kv_attention.hip` | Converts BF16 queries to E4M3 and consumes stored E4M3 keys directly with native FP8 WMMA. `tools/r9700/build/kv_op_qual.s` contains both `v_cvt_pk_fp8_f32` and `v_wmma_f32_16x16x16_fp8_fp8`. |
| XAttention rank and B16 sparse consumer in `src/ops/r9700/kv/fp8_int4_kv_xattention.hip` | The packing stage decodes logical FP8 keys into BF16 storage; rank scores that storage with BF16 WMMA, and the B16 sparse consumer applies BF16 WMMA to the selected subset. The serial consumer retained for the dense/tau-one case instead uses scalar FP8 decode and FP32 FMA. `tools/r9700/build/xattention_s16_tau900.s` contains `v_wmma_f32_16x16x16_bf16`. |
| Vision attention in `src/ops/r9700/vision/vision_attention.hip` | Native BF16 WMMA for QK and PV. |
| Production split-512 long-context attention | Uses native FP8 WMMA for the ordinary T=1 QK route and BF16 WMMA for fixed-width T=4 QK; `tools/r9700/build/split512_attention.s` contains both opcodes. At context 8,192 and above those admitted forms dispatch the three-stage leaf; metadata-bearing T=1 retains the fused route. |
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
reuse, and the loaded hipBLASLt FP8 path advertises one-stage global-read prefetch, so neither is
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

The installed CK Tile and rocWMMA sources do not contain an exact mixed-A8/packed-Q4G64 GEMM that
can replace NInfer's prefill kernel. CK Tile's gfx12 matrix layer has dense signed/signed
`pk_int4_t` K16 and K32 specializations, but both operands are packed INT4. Its group-quant block
GEMMs accept packed INT4 storage only by selecting FP8 or BF8 compute types; they convert the
stored codes before matrix multiplication. That is a different arithmetic profile from NInfer's
A8 low/high-nibble reconstruction followed by direct signed-Q4 integer accumulation and one FP16
activation-scale times FP16 weight-scale application per G64 group. rocWMMA supplies the same
fragment/instruction plumbing and packing transforms, not a fused mixed-width, group-scaled
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
