#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include "core/cache_warm.h"
#include "ninfer/ops/gdn_replay.h"
#include "ops/r9700/linear/fp8lut4_linear.h"
#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/r9700_linear.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::gdn {

// Correctness-first HIP contracts for the Qwen Gated DeltaNet semantic boundary. Every pointer
// is device-resident, feature-fastest, and every nonempty launch has an explicit HIP stream.
// The direct routes deliberately retain represented BF16 projections and FP32 persistent state;
// they do not depend on a retired backend's Tensor/Core abstractions or a separate projection
// implementation.

// Depthwise causal width-four convolution followed by SiLU. `input`/`output` are BF16
// [tokens, channels], `weight` is BF16 [channels, 4], and history is BF16 [channels, 3] ordered
// oldest-to-newest. `history_in` and `history_out` are either disjoint or exactly identical; the
// output state is tail_3(concat(history_in, input)) represented as exact input BF16 bits. The
// convolution sum and SiLU are FP32 before the stated BF16 output rounding.
[[nodiscard]] hipError_t causal_conv1d_silu_bf16(const hip_bfloat16* input,
                                                 const hip_bfloat16* weight,
                                                 const hip_bfloat16* history_in,
                                                 hip_bfloat16* history_out, hip_bfloat16* output,
                                                 std::uint32_t channels, std::uint32_t tokens,
                                                 hipStream_t stream) noexcept;

// Direct regression boundaries for the causal-convolution dispatch. Production selects token
// tile 4 at T>=32 and preserves the serial incumbent below that physical crossover (C=10240:
// serial 16.8/20.6/29.2 us against tile 4 17.9/18.8/20.9 us at T=16/32/64). The
// candidate boundary retains the qualified 4/8/16/32 sweep; the incumbent boundary bypasses
// production selection so later comparisons cannot accidentally time the selected route twice.
[[nodiscard]] hipError_t causal_conv1d_silu_prefill_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, const hip_bfloat16* history_in,
    hip_bfloat16* history_out, hip_bfloat16* output, std::uint32_t channels, std::uint32_t tokens,
    std::uint32_t token_tile, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t causal_conv1d_silu_incumbent_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight, const hip_bfloat16* history_in,
    hip_bfloat16* history_out, hip_bfloat16* output, std::uint32_t channels, std::uint32_t tokens,
    hipStream_t stream) noexcept;

// One-sequence Qwen3.8 prefill projection/convolution handoff at any width T >= 1. The
// represented BF16 projection outputs remain separate: query_key is [T,4096], and token t's
// value projection is the 6144 values at projected_value + t * value_stride, where value_stride
// is 6144 (a separately projected value; the output gate is projected by the caller) or 12288
// (the [value,z] projection rows; z is left to the caller). The operation applies the incumbent
// width-four causal convolution and SiLU formula directly into query [T,2048], key [T,2048], and
// value [T,6144], then publishes the trailing three represented columns of
// concat(history_in, projection) to history_out. history_in/history_out are disjoint or exactly
// identical.
[[nodiscard]] hipError_t projection_conv_prefill_direct_scatter_bf16(
    const hip_bfloat16* query_key, const hip_bfloat16* projected_value, std::uint32_t value_stride,
    const hip_bfloat16* conv_weight, const hip_bfloat16* history_in, hip_bfloat16* history_out,
    hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value, std::uint32_t tokens,
    hipStream_t stream) noexcept;

// A mixed round's GDN convolution in one launch: the prefill owner's direct scatter over its
// leading owner_tokens columns (value_stride 6144, history_in/history_out as above) and
// the recorded convolution of projection_conv_record_bf16 over the trailing width*batch columns of
// the same query_key/projected_value/query/key/value storage (value rows only; the projection
// publishes the output gate). Bitwise those two calls; the owner's
// history and the verify batch's initial state slots are disjoint.
[[nodiscard]] hipError_t projection_conv_mixed_split_bf16(
    const hip_bfloat16* query_key, const hip_bfloat16* projected_value,
    const hip_bfloat16* conv_weight, const hip_bfloat16* history_in, hip_bfloat16* history_out,
    hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value, std::uint32_t owner_tokens,
    const hip_bfloat16* conv_states, const std::int32_t* valid_columns,
    const std::int32_t* initial_state_slots, hip_bfloat16* conv_record, std::uint32_t width,
    std::uint32_t batch, std::uint32_t state_slots, hipStream_t stream) noexcept;

// Qwen3.8-27B verification projection/convolution at the fixed real geometry. `query_key` is
// BF16 [B*W,4096] in column-major Tensor storage, `value_z` is BF16 [B*W,12288] in [value,z]
// row order, convolution weights are BF16 [10240,4], and the state pool is BF16
// [10240,3,state_slots]. Selectors are device I32 [B]. Query/key/value invalid tails are exact
// zero, while z is copied for every physical column. Each valid token publishes its post-token
// width-three history to snapshot_base_slots[b]+token. The caller guarantees distinct destination
// intervals and prevents one row from overwriting another row's initial slot.
[[nodiscard]] hipError_t projection_conv_snapshot_bf16(
    const hip_bfloat16* query_key, const hip_bfloat16* value_z, const hip_bfloat16* conv_weight,
    hip_bfloat16* conv_states, const std::int32_t* valid_columns,
    const std::int32_t* initial_state_slots, const std::int32_t* snapshot_base_slots,
    hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value, hip_bfloat16* z,
    std::uint32_t width, std::uint32_t batch, std::uint32_t state_slots,
    hipStream_t stream) noexcept;

// Replay-record form of the same fixed projection/convolution. `conv_states` is immutable and
// `conv_record` receives the represented BF16 q/k/value projection for every valid token. With a
// null parent tensor, columns are sequential. Otherwise parent_index is I32 [W,B] in column-major
// Tensor storage: a negative parent selects the checkpoint and a nonnegative parent must precede
// the child. Width is 2..16 and batch is 1..8.
[[nodiscard]] hipError_t projection_conv_record_bf16(
    const hip_bfloat16* query_key, const hip_bfloat16* value_z, const hip_bfloat16* conv_weight,
    const hip_bfloat16* conv_states, const std::int32_t* valid_columns,
    const std::int32_t* initial_state_slots, const std::int32_t* parent_index,
    hip_bfloat16* conv_record, hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value,
    hip_bfloat16* z, std::uint32_t width, std::uint32_t batch, std::uint32_t state_slots,
    hipStream_t stream) noexcept;

// One sequence (batch 1) at verification widths 5..8: the Q4N16K16 query-key [4096,5120] and
// value-z [12288,5120] projections of prepared A8G64 `planes` (the small-batch pair arithmetic)
// fused with projection_conv_record_bf16 on their BF16 values; value-z rows 6144.. publish z.
// Bitwise the pair projection followed by projection_conv_record_bf16.
[[nodiscard]] bool gdn_pair_conv_record_supported(std::uint32_t width,
                                                  std::uint32_t batch) noexcept;
// The same fused projection-convolution of one per-token E4M3 image over FP8LUT4 query-key
// [4096, 5120] and value-z [12288, 5120] weights, at batch 1 and widths 4..8 (W4 is the C1
// DFlash K3 chain).
[[nodiscard]] bool gdn_fp8lut4_pair_conv_record_supported(std::uint32_t width,
                                                          std::uint32_t batch) noexcept;
[[nodiscard]] hipError_t gdn_fp8lut4_pair_conv_record_bf16(
    const linear::Fp8ActivationWorkspace& image, const linear::Fp8Lut4Weight& query_key,
    const linear::Fp8Lut4Weight& value_z, const hip_bfloat16* conv_weight,
    const hip_bfloat16* conv_states, const std::int32_t* valid_columns,
    const std::int32_t* initial_state_slots, const std::int32_t* parent_index,
    hip_bfloat16* conv_record, hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value,
    hip_bfloat16* z, std::uint32_t width, std::uint32_t state_slots, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t gdn_pair_conv_record_bf16(
    const linear::A8G64ActivationWorkspace& planes, const std::uint8_t* query_key_codes,
    const std::uint16_t* query_key_scales, const std::uint8_t* value_z_codes,
    const std::uint16_t* value_z_scales, const hip_bfloat16* conv_weight,
    const hip_bfloat16* conv_states, const std::int32_t* valid_columns,
    const std::int32_t* initial_state_slots, const std::int32_t* parent_index,
    hip_bfloat16* conv_record, hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* value,
    hip_bfloat16* z, std::uint32_t width, std::uint32_t state_slots, hipStream_t stream) noexcept;

// Makes the raw FP32 GDN controls. `a`/`b` are represented BF16 [tokens, value_heads], and
// `a_log`/`dt_bias` are FP32 [value_heads]. Outputs g/beta are FP32 [tokens, value_heads]:
// g=-exp(a_log)*softplus(a+dt_bias), beta=sigmoid(b), where softplus(x)=x for x>20 and otherwise
// log1p(exp(x)). All output elements are independently defined.
[[nodiscard]] hipError_t control_gates_bf16(const hip_bfloat16* a, const hip_bfloat16* b,
                                            const float* a_log, const float* dt_bias, float* g,
                                            float* beta, std::uint32_t value_heads,
                                            std::uint32_t tokens, hipStream_t stream) noexcept;

// Two BF16 [48,5120] Qwen3.8 GDN control projections followed by the control formula, T>=1.
// The projection values are rounded to BF16 in registers and are not materialized. This raw
// boundary is also the direct production-symbol qualification seam. T1..24 use one per-token
// arithmetic (identical for every token at every such width); wider launches use a split-K BF16
// WMMA route with FP32 accumulation that reads each hidden column once. Hidden and weight bases
// are 16-byte aligned.
[[nodiscard]] hipError_t bf16_projected_control(const hip_bfloat16* hidden,
                                                const hip_bfloat16* a_weight,
                                                const hip_bfloat16* b_weight, const float* a_log,
                                                const float* dt_bias, float* g, float* beta,
                                                std::uint32_t tokens, hipStream_t stream) noexcept;

// Verification-width GDN front for T1..32: the unit-offset-capable K5120 RMSNorm of residual
// [tokens,5120] (the row-CTA arithmetic of the eager RMSNorm, BF16 seam), the T1 control
// arithmetic of bf16_projected_control (exactly its T1..24 route; above 24 that Op uses its wide
// route, so only the FP64 control oracle relates them) on that seam, and the exact A8G64 codec of
// the same seam into `planes` (bound for tokens x 5120), whose status word is published without a
// reset launch. `hidden`, when non-null, receives the BF16 seam rows; `clear_status`, when
// non-null, is a later Op's status word (disjoint from every operand) that CTA 0 zeroes.
[[nodiscard]] bool bf16_gdn_normalized_front_supported(std::uint32_t tokens) noexcept;
[[nodiscard]] hipError_t bf16_gdn_normalized_front(
    const hip_bfloat16* residual, const hip_bfloat16* norm, float eps, bool unit_offset,
    const hip_bfloat16* a_weight, const hip_bfloat16* b_weight, const float* a_log,
    const float* dt_bias, float* g, float* beta, const linear::A8G64ActivationWorkspace& planes,
    hip_bfloat16* hidden, std::uint32_t* clear_status, hipStream_t stream) noexcept;

// The same front with the seam encoded as a per-token E4M3 image (bound for tokens x 5120) for
// FP8LUT4 projections: each token's owner CTA publishes its codes, scale and status word. CTAs
// past the front grid apply `fold` (when non-null: that layer's deferred replay fold, bitwise
// ops::gdn_replay_fold_layer, whose state it alone touches) and then touch `warm`
// (core/cache_warm.h).
[[nodiscard]] hipError_t
fp8_gdn_normalized_front(const hip_bfloat16* residual, const hip_bfloat16* norm, float eps,
                         bool unit_offset, const hip_bfloat16* a_weight,
                         const hip_bfloat16* b_weight, const float* a_log, const float* dt_bias,
                         float* g, float* beta, const linear::Fp8ActivationWorkspace& image,
                         hipStream_t stream, const CacheWarm& warm = {},
                         const GdnLayerFold* layer_fold = nullptr) noexcept;

// Exact FP32 state movement for transaction/checkpoint publication. Source/destination are
// non-overlapping FP32 elements and count is positive.
[[nodiscard]] hipError_t copy_state_fp32(const float* source, float* destination, std::size_t count,
                                         hipStream_t stream) noexcept;

// Applies the causal normalized Gated DeltaNet recurrence. q/k are represented BF16
// [tokens, qk_heads, key_dim], v/out are BF16 [tokens, value_heads, value_dim], g/beta are FP32
// [tokens, value_heads], and state is FP32 [value_heads, value_dim, key_dim]. For value head h,
// q/k head is floor(h/(value_heads/qk_heads)). For each token: normalize q/k in FP32 using
// x/sqrt(sum(x*x)+epsilon), decay state with exp(g), form beta*(v-decayed_state*k), rank-one
// update, and emit BF16 scale*(updated_state*q). state_in/state_out may be disjoint or exactly
// identical; state_out receives only the final state after all tokens. Inputs/out do not overlap
// state storage. Dimensions are positive, value_heads is divisible by qk_heads, scale/epsilon are
// finite with epsilon positive, and all represented values are finite.
[[nodiscard]] hipError_t
recurrent_bf16_fp32(const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v,
                    const float* g, const float* beta, const float* state_in, float* state_out,
                    hip_bfloat16* output, std::uint32_t key_dim, std::uint32_t value_dim,
                    std::uint32_t qk_heads, std::uint32_t value_heads, std::uint32_t tokens,
                    float scale, float epsilon, hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::gdn
