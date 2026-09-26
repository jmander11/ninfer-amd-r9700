#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::linear {

// Decode-first raw linear contracts. X and Y are token-major: X[T,K], Y[T,N].
// Both paths accumulate each output in FP32 and round once to represented BF16
// at Y. They are standalone R9700 Op groundwork, not an Engine interface.
struct Bf16LinearArgs {
    const hip_bfloat16* input = nullptr;
    const hip_bfloat16* weights = nullptr; // [N,K], row-major
    hip_bfloat16* output = nullptr;
    std::uint32_t tokens = 0; // T in [1,8] for decode; T>=9 for prefill
    std::uint32_t rows = 0;   // N
    std::uint32_t columns = 0; // K
};

// The two pointers are direct views of the candidate converter's W8G32_F16S
// RowSplit planes: signed-INT8 codes [N,padded_columns] and FP16 scale words
// [N,padded_columns/32]. Keeping the planes explicit permits a target binder to
// pass a row slice without repacking the artifact payload. Both byte extents
// must exactly cover the supplied view.
struct W8G32LinearArgs {
    const hip_bfloat16* input = nullptr;
    const std::int8_t* codes = nullptr;
    std::size_t code_bytes = 0;
    const std::uint16_t* scales = nullptr;
    std::size_t scale_bytes = 0;
    hip_bfloat16* output = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
};

// Dynamic signed-A8G32 activation image for persistent W8G32 weights. Codes are direct signed
// bytes [T,padded_K]; scales are represented FP16 [T,padded_K/32]. Canonical K128 padding is
// zero-filled by the codec, including a partial final G32. The status word remains device-owned
// through the ordered quantize/consumer pair so graph capture needs no host readback.
struct A8G32ActivationWorkspace {
    std::int8_t* codes = nullptr;
    std::size_t code_bytes = 0;
    std::uint16_t* scales = nullptr;
    std::size_t scale_bytes = 0;
    std::uint32_t* status = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
};

enum A8G32ActivationStatus : std::uint32_t {
    A8G32ActivationOk = 0U,
    A8G32ActivationNonfinite = 1U << 0U,
    A8G32ActivationScaleOverflow = 1U << 1U,
};

struct A8G32ActivationQuantizeArgs {
    const hip_bfloat16* input = nullptr;
    A8G32ActivationWorkspace workspace{};
};

struct A8W8G32LinearArgs {
    // Packed code-plane bases are 4-byte aligned for native dword fragments.
    const std::int8_t* activation_codes = nullptr;
    std::size_t activation_code_bytes = 0;
    const std::uint16_t* activation_scales = nullptr;
    std::size_t activation_scale_bytes = 0;
    const std::uint32_t* activation_status = nullptr;
    const std::int8_t* weight_codes = nullptr;
    std::size_t weight_code_bytes = 0;
    const std::uint16_t* weight_scales = nullptr;
    std::size_t weight_scale_bytes = 0;
    hip_bfloat16* output = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
};

struct A8W8G32CandidateArgs {
    const hip_bfloat16* input = nullptr;
    const std::int8_t* weight_codes = nullptr;
    std::size_t weight_code_bytes = 0;
    const std::uint16_t* weight_scales = nullptr;
    std::size_t weight_scale_bytes = 0;
    void* activation_workspace = nullptr;
    std::size_t activation_workspace_bytes = 0;
    hip_bfloat16* output = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
};

struct A8W8G32KernelResources {
    int quantize_registers = 0;
    int quantize_static_shared_bytes = 0;
    int quantize_local_bytes = 0;
    int wmma_registers = 0;
    int wmma_static_shared_bytes = 0;
    int wmma_local_bytes = 0;
    int prefill_cta_registers = 0;
    int prefill_cta_static_shared_bytes = 0;
    int prefill_cta_local_bytes = 0;
    int prefill_cta_global_inv_qualification_registers = 0;
    int prefill_cta_global_inv_qualification_static_shared_bytes = 0;
    int prefill_cta_global_inv_qualification_local_bytes = 0;
};

[[nodiscard]] std::size_t a8w8g32_activation_workspace_capacity_bytes(
    std::uint32_t tokens, std::uint32_t columns) noexcept;
[[nodiscard]] hipError_t a8w8g32_bind_activation_workspace(
    void* storage, std::size_t storage_bytes, std::uint32_t tokens,
    std::uint32_t columns, A8G32ActivationWorkspace* out) noexcept;
[[nodiscard]] hipError_t a8g32_quantize_activation(
    const A8G32ActivationQuantizeArgs& args, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t a8w8g32_linear_wmma32(const A8W8G32LinearArgs& args,
                                                hipStream_t stream) noexcept;
[[nodiscard]] hipError_t a8w8g32_linear_prefill_cta(const A8W8G32LinearArgs& args,
                                                    hipStream_t stream) noexcept;
// Qualification-only superseded global-invalidate incumbent retained for direct
// numerical/performance regression against the promoted LDS-scoped production entry.
[[nodiscard]] hipError_t a8w8g32_linear_prefill_cta_global_inv_qualification(
    const A8W8G32LinearArgs& args, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t a8w8g32_linear_candidate(const A8W8G32CandidateArgs& args,
                                                   hipStream_t stream) noexcept;
[[nodiscard]] hipError_t a8w8g32_kernel_resources(
    A8W8G32KernelResources* resources) noexcept;

// Qualified Q4G64 route used by the registered R9700 identities. Activations
// remain row-major packed planes. Persistent W uses the direct gfx1201 layout
// [N/16][K/64][four K16 pairs][16 rows][packed-u64], with FP16 scales in
// [N/16][K/64][16 rows]. K is padded to 128 and padding codes are zero.
// Consumers read this layout directly; no runtime repack or compatibility lane exists.
// The public Tensor dispatch reaches it only with explicit caller-owned workspace; real-model
// PPL, token, and whole-inference gates still decide whether a Q4 identity becomes production.
struct Q4G64ActivationWorkspace {
    std::uint8_t* codes = nullptr;
    std::size_t code_bytes = 0;
    std::uint16_t* scales = nullptr;
    std::size_t scale_bytes = 0;
    std::uint32_t* status = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
};

enum Q4G64ActivationStatus : std::uint32_t {
    Q4G64ActivationOk = 0U,
    Q4G64ActivationNonfinite = 1U << 0U,
    Q4G64ActivationScaleOverflow = 1U << 1U,
};

struct Q4G64ActivationQuantizeArgs {
    const hip_bfloat16* input = nullptr; // represented BF16 [T,K]
    Q4G64ActivationWorkspace workspace{};
};

struct Q4G64LinearArgs {
    // Packed activation code-plane bases are 4-byte aligned; persistent N16/K16
    // weight code-plane bases are 8-byte aligned.
    const std::uint8_t* activation_codes = nullptr;
    std::size_t activation_code_bytes = 0;
    const std::uint16_t* activation_scales = nullptr;
    std::size_t activation_scale_bytes = 0;
    // Ordered after quantization. Any nonzero status makes the consumer publish
    // conspicuous BF16 NaN for every output instead of using invalid codes.
    const std::uint32_t* activation_status = nullptr;
    const std::uint8_t* weight_codes = nullptr;
    std::size_t weight_code_bytes = 0;
    const std::uint16_t* weight_scales = nullptr;
    std::size_t weight_scale_bytes = 0;
    hip_bfloat16* output = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
};

struct Q4G64CandidateArgs {
    const hip_bfloat16* input = nullptr;
    const std::uint8_t* weight_codes = nullptr;
    std::size_t weight_code_bytes = 0;
    const std::uint16_t* weight_scales = nullptr;
    std::size_t weight_scale_bytes = 0;
    void* activation_workspace = nullptr;
    std::size_t activation_workspace_bytes = 0;
    hip_bfloat16* output = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
};

struct Q4G64PrefillQualificationResources {
    int registers = 0;
    int static_shared_bytes = 0;
    int local_bytes = 0;
    int max_threads_per_block = 0;
};

// Higher-precision activation evaluator over the same persistent Q4G64 weight
// planes. A canonical signed A8 code q is stored losslessly as two packed
// nibble planes: unsigned lo=q&15 and signed hi=(q-lo)/16, so q=lo+16*hi.
// The planes together occupy exactly one byte per padded activation element.
struct A8G64ActivationWorkspace {
    std::uint8_t* low_codes = nullptr;
    std::size_t low_code_bytes = 0;
    std::uint8_t* high_codes = nullptr;
    std::size_t high_code_bytes = 0;
    std::uint16_t* scales = nullptr;
    std::size_t scale_bytes = 0;
    std::uint32_t* status = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
};

struct A8G64ActivationQuantizeArgs {
    const hip_bfloat16* input = nullptr; // represented BF16 [T,K]
    A8G64ActivationWorkspace workspace{};
};

struct A8Q4G64LinearArgs {
    // Activation planes are 4-byte aligned; persistent N16/K16 weights are
    // 8-byte aligned for one native packed K16 pair load.
    const std::uint8_t* activation_low_codes = nullptr;
    std::size_t activation_low_code_bytes = 0;
    const std::uint8_t* activation_high_codes = nullptr;
    std::size_t activation_high_code_bytes = 0;
    const std::uint16_t* activation_scales = nullptr;
    std::size_t activation_scale_bytes = 0;
    // Same asynchronous failure contract as the A4 consumer.
    const std::uint32_t* activation_status = nullptr;
    const std::uint8_t* weight_codes = nullptr;
    std::size_t weight_code_bytes = 0;
    const std::uint16_t* weight_scales = nullptr;
    std::size_t weight_scale_bytes = 0;
    hip_bfloat16* output = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
    // Only the M128 prefill route accepts this: publish BF16(output + BF16(projection)) in place
    // (the projected-residual boundary); codec failure still writes BF16 NaN.
    bool accumulate_output = false;
};

struct A8Q4G64CandidateArgs {
    const hip_bfloat16* input = nullptr;
    const std::uint8_t* weight_codes = nullptr;
    std::size_t weight_code_bytes = 0;
    const std::uint16_t* weight_scales = nullptr;
    std::size_t weight_scale_bytes = 0;
    void* activation_workspace = nullptr;
    std::size_t activation_workspace_bytes = 0;
    hip_bfloat16* output = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
};

// Complete private normalized projection; prepared activation planes never cross the Op boundary.
[[nodiscard]] hipError_t a8q4g64_normalized_linear_t1(
    const A8Q4G64CandidateArgs& args, const hip_bfloat16* norm, float eps,
    bool unit_offset, hipStream_t stream) noexcept;
// Batched (T >= 2) form of the T1 normalized projection above for the same K5120 -> N34816
// Q4N16K16 matrix: RMSNorm, the explicit BF16 seam, and the A8G64 codec in one pass (no
// materialized BF16 rows; one wave per row, eight rows per block, a single block clearing its own
// status), then the A8Q4 route the candidate selects for T (small-batch for verify widths, the
// M128xN128 GEMM for prefill chunks). input and norm are 16-byte aligned.
[[nodiscard]] bool a8q4g64_normalized_linear_batched_supported(std::uint32_t tokens) noexcept;
[[nodiscard]] hipError_t a8q4g64_normalized_linear_batched(
    const A8Q4G64CandidateArgs& args, const hip_bfloat16* norm, float eps,
    bool unit_offset, hipStream_t stream) noexcept;

// Two or three A8Q4 projections of one represented BF16 [T,K] input (the Qwen3.8 GDN query-key,
// value and output-gate matrices): the activation is quantized once into the caller workspace and
// every projection reads the same planes through the route the single-projection candidate would
// select, publishing its BF16 [T,rows] output exactly as that route does. Outputs are pairwise
// disjoint.
struct A8Q4G64SharedProjection {
    const std::uint8_t* weight_codes = nullptr;
    std::size_t weight_code_bytes = 0;
    const std::uint16_t* weight_scales = nullptr;
    std::size_t weight_scale_bytes = 0;
    hip_bfloat16* output = nullptr;
    std::uint32_t rows = 0;
};
inline constexpr std::uint32_t kA8Q4G64MaximumSharedProjections = 3;
struct A8Q4G64SharedActivationArgs {
    const hip_bfloat16* input = nullptr;
    void* activation_workspace = nullptr;
    std::size_t activation_workspace_bytes = 0;
    A8Q4G64SharedProjection projections[kA8Q4G64MaximumSharedProjections];
    std::uint32_t projection_count = 0;  // 2 or 3
    std::uint32_t tokens = 0;
    std::uint32_t columns = 0;
};
[[nodiscard]] bool a8q4g64_shared_activation_supported(
    std::uint32_t tokens, std::uint32_t columns, std::uint32_t rows) noexcept;
[[nodiscard]] hipError_t a8q4g64_shared_activation_linear(
    const A8Q4G64SharedActivationArgs& args, hipStream_t stream) noexcept;
// The same projections of activation planes another Op's exact A8G64 codec already prepared
// (with their status word) in activation_workspace for (tokens, columns); input must be null.
[[nodiscard]] hipError_t a8q4g64_prepared_shared_activation_linear(
    const A8Q4G64SharedActivationArgs& args, hipStream_t stream) noexcept;
// Normalized form of the shared projections at K5120: one pass computes the RMSNorm
// BF16(x * rsqrt(mean(x^2) + eps) * (norm + unit_offset)), publishes those BF16 rows to
// `normalized` [T,5120] and quantizes the same values into the A8G64 planes every GEMM reads.
// input, norm and normalized are 16-byte aligned and pairwise disjoint from every output.
// With `side`, the last projection runs on side->stream after `fork` is recorded on `stream`,
// and `join` is recorded on side->stream after it; the caller orders its later work after
// `join` before reading that output or reusing the activation workspace.
struct A8Q4G64SideProjection {
    hipStream_t stream = nullptr;
    hipEvent_t fork = nullptr;
    hipEvent_t join = nullptr;
};
[[nodiscard]] hipError_t a8q4g64_normalized_shared_activation_linear(
    const A8Q4G64SharedActivationArgs& args, const hip_bfloat16* norm, float eps,
    bool unit_offset, hip_bfloat16* normalized, hipStream_t stream,
    const A8Q4G64SideProjection* side = nullptr) noexcept;

// Qwen3.8 GDN output projection: the per-head gated RMSNorm (x * rsqrt(mean_head(x^2) + eps) *
// norm * silu(z), BF16 seam, 48 heads x 128) of input/z [T,6144] feeds the A8G64 codec directly,
// then the A8Q4 GEMM against the Q4N16K16 [5120,6144] weight rounds to BF16 and publishes
// output = BF16(output + projection) in place (the projected-residual boundary). The per-vector
// prepare arithmetic is shared by both extents: T1..8 small-batch widths use one thread per
// 16-byte slot and the small-batch projection; prefill T is a positive multiple of 128 inside the
// qualified M128 inventory. input/gate/norm are 16-byte aligned.
[[nodiscard]] bool a8q4g64_gated_normalized_linear_supported(std::uint32_t tokens) noexcept;
// status_cleared: the caller already zeroed, on this stream after its last use, the status word
// the workspace binds for (tokens, 6144); the small route then skips its reset launch.
[[nodiscard]] hipError_t a8q4g64_gated_normalized_linear(
    const A8Q4G64CandidateArgs& args, const hip_bfloat16* norm, const hip_bfloat16* gate,
    float eps, hipStream_t stream, bool status_cleared = false) noexcept;

// Fixed T1 projected-residual boundary: quantize represented BF16 input with the
// signed-A8G64 codec, evaluate the packed Q4N16K16/G64 projection, round that
// projection to BF16, then publish BF16(residual + projection) in place.
// Domain: T=1, N=5120, K=padded_K in {6144,17408}. The input, both weight
// planes, residual, and caller-owned activation workspace are pairwise disjoint.
// Weight and workspace extents are exact. Any activation codec failure poisons
// every residual element with BF16 NaN. No allocation or host synchronization.
struct A8Q4G64ProjectedResidualT1Args {
    const hip_bfloat16* input = nullptr;
    const std::uint8_t* weight_codes = nullptr;
    std::size_t weight_code_bytes = 0;
    const std::uint16_t* weight_scales = nullptr;
    std::size_t weight_scale_bytes = 0;
    void* activation_workspace = nullptr;
    std::size_t activation_workspace_bytes = 0;
    hip_bfloat16* residual = nullptr;
    std::uint32_t tokens = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
};

[[nodiscard]] hipError_t a8q4g64_projected_residual_t1(
    const A8Q4G64ProjectedResidualT1Args& args, hipStream_t stream) noexcept;

// Exact production GDN T1 projection pair. The represented BF16
// [1,5120] input is quantized once into caller-owned storage, then one combined
// row grid evaluates the N4096 query/key and N12288 value/z matrices. Both
// persistent matrices retain the production Q4N16K16/G64 layout.
struct A8Q4G64GdnPairArgs {
    const hip_bfloat16* input = nullptr;
    const std::uint8_t* weight0_codes = nullptr;
    std::size_t weight0_code_bytes = 0;
    const std::uint16_t* weight0_scales = nullptr;
    std::size_t weight0_scale_bytes = 0;
    hip_bfloat16* output0 = nullptr;
    const std::uint8_t* weight1_codes = nullptr;
    std::size_t weight1_code_bytes = 0;
    const std::uint16_t* weight1_scales = nullptr;
    std::size_t weight1_scale_bytes = 0;
    hip_bfloat16* output1 = nullptr;
    void* activation_workspace = nullptr;
    std::size_t activation_workspace_bytes = 0;
    std::uint32_t tokens = 0;
    std::uint32_t columns = 0;
};

struct A8Q4G64GdnPairResources {
    int registers = 0;
    int static_shared_bytes = 0;
    int local_bytes = 0;
    int max_threads_per_block = 0;
};

// Fixed Text-MLP boundary. The input is the concatenated BF16
// [T,2K] gate/up result. The fused preparation explicitly rounds SiLU(gate)*up
// to BF16 before applying the ordinary signed-A8G64 codec, then invokes the
// unchanged A8Q4 matrix route. Target routing owns its exact admitted profile.
struct FusedSiluA8Q4G64DownArgs {
    const hip_bfloat16* gate_up = nullptr;
    const std::uint8_t* weight_codes = nullptr;
    std::size_t weight_code_bytes = 0;
    const std::uint16_t* weight_scales = nullptr;
    std::size_t weight_scale_bytes = 0;
    void* activation_workspace = nullptr;
    std::size_t activation_workspace_bytes = 0;
    hip_bfloat16* residual = nullptr; // BF16 [T,N], updated in place
    std::uint32_t tokens = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
};

// The whole A8 MLP at small verification widths (T <= 12 inside both routes): the normalized
// gate/up projection of a8q4g64_normalized_linear_batched followed by fused_silu_a8q4g64_down,
// with identical arithmetic. Both stages share one activation workspace base; the small-route
// gate/up prepare also clears the down stage's status word (which must lie beyond the gate/up
// planes), so the fused SiLU quantization needs no separate reset launch. down.gate_up must be
// gate_up.output.
[[nodiscard]] bool a8q4g64_normalized_mlp_supported(std::uint32_t tokens) noexcept;
[[nodiscard]] hipError_t a8q4g64_normalized_mlp(const A8Q4G64CandidateArgs& gate_up,
                                                const hip_bfloat16* norm, float eps,
                                                bool unit_offset,
                                                const FusedSiluA8Q4G64DownArgs& down,
                                                hipStream_t stream) noexcept;

struct A8Q4G64KernelResources {
    int quantize_registers = 0;
    int quantize_static_shared_bytes = 0;
    int quantize_local_bytes = 0;
    int wmma_registers = 0;
    int wmma_static_shared_bytes = 0;
    int wmma_local_bytes = 0;
    int decode_dot8_t1_registers = 0;
    int decode_dot8_t1_static_shared_bytes = 0;
    int decode_dot8_t1_local_bytes = 0;
    int decode_dot8_t1_max_threads_per_block = 0;
    int prefill_cta_registers = 0;
    int prefill_cta_static_shared_bytes = 0;
    int prefill_cta_local_bytes = 0;
    int prefill_cta_global_inv_qualification_registers = 0;
    int prefill_cta_global_inv_qualification_static_shared_bytes = 0;
    int prefill_cta_global_inv_qualification_local_bytes = 0;
    int prefill_cta_m64n64_regression_registers = 0;
    int prefill_cta_m64n64_regression_static_shared_bytes = 0;
    int prefill_cta_m64n64_regression_local_bytes = 0;
    int prefill_cta_m128n128_qualification_registers = 0;
    int prefill_cta_m128n128_qualification_static_shared_bytes = 0;
    int prefill_cta_m128n128_qualification_local_bytes = 0;
    int prefill_cta_m128n128_qualification_max_threads_per_block = 0;
    int prefill_cta_m64n128_regression_registers = 0;
    int prefill_cta_m64n128_regression_static_shared_bytes = 0;
    int prefill_cta_m64n128_regression_local_bytes = 0;
    int prefill_cta_m64n128_regression_max_threads_per_block = 0;
};

// Returns zero for an invalid or overflowing shape. The storage contains the
// packed activation codes, aligned FP16 scale plane, and a device status word;
// it is always wholly caller-owned and stable across graph capture.
[[nodiscard]] std::size_t q4g64_activation_workspace_capacity_bytes(
    std::uint32_t tokens, std::uint32_t columns) noexcept;
[[nodiscard]] hipError_t q4g64_bind_activation_workspace(
    void* storage, std::size_t storage_bytes, std::uint32_t tokens,
    std::uint32_t columns, Q4G64ActivationWorkspace* out) noexcept;
[[nodiscard]] hipError_t q4g64_quantize_activation(
    const Q4G64ActivationQuantizeArgs& args, hipStream_t stream) noexcept;

// Raw complete candidate boundary: canonical BF16->A4G64 quantization followed
// by native gfx1201 signed-INT4 WMMA, FP32 scale composition/accumulation, and
// exactly one BF16 result rounding. It allocates and repacks nothing.
[[nodiscard]] hipError_t q4g64_linear_candidate(const Q4G64CandidateArgs& args,
                                                 hipStream_t stream) noexcept;

// Qualification-visible matrix routes over an already quantized activation.
[[nodiscard]] hipError_t q4g64_linear_wmma32(const Q4G64LinearArgs& args,
                                              hipStream_t stream) noexcept;
[[nodiscard]] hipError_t q4g64_linear_vector(const Q4G64LinearArgs& args,
                                              hipStream_t stream) noexcept;
// Signed-A4G64 x signed-Q4G64 M64xN128 cooperative route. The complete A4
// consumer selects it only for N34816/K5120 prefill; activation precision is unchanged.
[[nodiscard]] hipError_t q4g64_linear_prefill_cta_m64n128(
    const Q4G64LinearArgs& args, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t q4g64_prefill_qualification_resources(
    Q4G64PrefillQualificationResources* resources, bool pingpong = false) noexcept;

[[nodiscard]] std::size_t a8q4g64_activation_workspace_capacity_bytes(
    std::uint32_t tokens, std::uint32_t columns) noexcept;
[[nodiscard]] hipError_t a8q4g64_bind_activation_workspace(
    void* storage, std::size_t storage_bytes, std::uint32_t tokens,
    std::uint32_t columns, A8G64ActivationWorkspace* out) noexcept;
[[nodiscard]] hipError_t a8g64_quantize_activation(
    const A8G64ActivationQuantizeArgs& args, hipStream_t stream) noexcept;

// Exact signed-A8 decomposition over native homogeneous IU4 WMMA. Low-nibble
// products use unsigned A/signed W; high-nibble products use signed A/signed W.
// INT32 is recombined before FP32 scale composition and one BF16 output rounding.
[[nodiscard]] hipError_t a8q4g64_linear_wmma32(const A8Q4G64LinearArgs& args,
                                                hipStream_t stream) noexcept;
// Selected exact-domain T=1 GEMV. Each thread owns one output row and evaluates
// the A8 decomposition with packed gfx1201 mixed-sign dot8. Calls outside the
// seven full-K production tuples are rejected; the candidate boundary retains WMMA.
[[nodiscard]] hipError_t a8q4g64_linear_decode_dot8_t1(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t a8q4g64_gdn_pair_t1(
    const A8Q4G64GdnPairArgs& args, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t a8q4g64_gdn_pair_t1_resources(
    A8Q4G64GdnPairResources* resources) noexcept;
// Qualification-only block-size sweep of the unchanged one-row/thread T=1 kernel.
[[nodiscard]] hipError_t a8q4g64_linear_decode_dot8_block_t1_qualification(
    const A8Q4G64LinearArgs& args, std::uint32_t threads,
    hipStream_t stream) noexcept;
// Direct small-T packed-W4 route retained across the exact N34816/K5120 screen domain. The
// compile-selected candidate boundary calls it only for the eight direct-screen winners; rejected
// widths remain directly callable solely for retained qualification evidence. Whole-DFlash A/B is
// still required before any unconditional product selection.
[[nodiscard]] hipError_t a8q4g64_linear_dflash_small_t_qualification(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t a8q4g64_linear_dflash_mlp_down_t5_qualification(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept;
// Direct 64-token x 128-row ping/pong cooperative-LDS production route. The candidate boundary
// admits it only for the exact qualified tuples/extents; tails use the named regression control.
[[nodiscard]] hipError_t a8q4g64_linear_prefill_cta(const A8Q4G64LinearArgs& args,
                                                    hipStream_t stream) noexcept;
[[nodiscard]] bool a8q4g64_scalar_base_u32_offsets_fit(
    std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t padded_columns) noexcept;
// Superseded single-bank M64xN128 route retained only for direct regression and production-tail
// correctness. No exact qualified product tuple dispatches here.
[[nodiscard]] hipError_t a8q4g64_linear_prefill_cta_m64n128_regression(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept;
// Qualification-only superseded global-invalidate incumbent retained for direct
// numerical/performance regression against the promoted LDS-scoped production entry.
[[nodiscard]] hipError_t a8q4g64_linear_prefill_cta_global_inv_qualification(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept;
// Qualification-only M64xN64 control retained for direct regression against
// the selected M64xN128 production route and for no product dispatch.
[[nodiscard]] hipError_t a8q4g64_linear_prefill_cta_m64n64_regression(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept;
// Qualification-only M128xN128 challenger. Thirty-two waves retain two row
// fragments each while halving persistent-weight rereads across token tiles.
[[nodiscard]] hipError_t a8q4g64_linear_prefill_cta_m128n128_qualification(
    const A8Q4G64LinearArgs& args, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t a8q4g64_linear_candidate(const A8Q4G64CandidateArgs& args,
                                                   hipStream_t stream) noexcept;
// Projected-residual boundary for T >= 2: the signed-A8G64 codec of represented BF16 input
// [T,K], the A8Q4 projection against the Q4N16K16 weight, BF16 rounding of the projection, then
// BF16(residual + projection) published in place into output [T,N]. Served by the small-batch
// N5120 widths and by the M128 prefill inventory (T and N multiples of 128). Codec failure
// writes BF16 NaN.
[[nodiscard]] bool a8q4g64_projected_residual_supported(
    std::uint32_t tokens, std::uint32_t rows, std::uint32_t columns) noexcept;
[[nodiscard]] hipError_t a8q4g64_projected_residual(
    const A8Q4G64CandidateArgs& args, hipStream_t stream) noexcept;
// SiLU(gate)*up of gate_up [T,2K] (BF16 seam), the A8G64 codec, and the N5120/K17408 down
// projection published as the projected-residual boundary BF16(residual + BF16(projection)) in
// place, at every width a8q4g64_projected_residual_supported admits.
[[nodiscard]] hipError_t fused_silu_a8q4g64_down(
    const FusedSiluA8Q4G64DownArgs& args, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t fused_silu_a8g64_prepare(
    const hip_bfloat16* gate_up, const A8G64ActivationWorkspace& workspace,
    hipStream_t stream) noexcept;
[[nodiscard]] hipError_t a8q4g64_kernel_resources(
    A8Q4G64KernelResources* resources) noexcept;

[[nodiscard]] hipError_t bf16_linear(const Bf16LinearArgs& args,
                                     hipStream_t stream) noexcept;
[[nodiscard]] hipError_t w8g32_linear(const W8G32LinearArgs& args,
                                       hipStream_t stream) noexcept;

// Prefill contracts are deliberately separate from decode: they accept the
// every T>=9 and retain FP32 accumulation / one BF16 output rounding. They do
// not alter the decode dispatch above.
[[nodiscard]] hipError_t bf16_linear_prefill(const Bf16LinearArgs& args,
                                             hipStream_t stream) noexcept;
[[nodiscard]] hipError_t w8g32_linear_prefill(const W8G32LinearArgs& args,
                                               hipStream_t stream) noexcept;

// Qualification-visible routes. The product-facing raw entry points above use
// one fixed width dispatch selected from R9700 measurements; these expose the
// alternatives solely so their arithmetic and event timing can be audited.
[[nodiscard]] hipError_t bf16_linear_baseline(const Bf16LinearArgs& args,
                                              hipStream_t stream) noexcept;
[[nodiscard]] hipError_t bf16_linear_wave8(const Bf16LinearArgs& args,
                                           hipStream_t stream) noexcept;
[[nodiscard]] hipError_t w8g32_linear_baseline(const W8G32LinearArgs& args,
                                                hipStream_t stream) noexcept;
[[nodiscard]] hipError_t w8g32_linear_wave8(const W8G32LinearArgs& args,
                                             hipStream_t stream) noexcept;
[[nodiscard]] hipError_t bf16_linear_wave16(const Bf16LinearArgs& args,
                                            hipStream_t stream) noexcept;
[[nodiscard]] hipError_t w8g32_linear_wave16(const W8G32LinearArgs& args,
                                              hipStream_t stream) noexcept;
[[nodiscard]] hipError_t bf16_linear_wmma16(const Bf16LinearArgs& args,
                                             hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::linear
