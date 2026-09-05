#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

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
                                                  hip_bfloat16* history_out,
                                                  hip_bfloat16* output,
                                                  std::uint32_t channels,
                                                  std::uint32_t tokens,
                                                  hipStream_t stream) noexcept;

// Direct regression boundaries for the causal-convolution dispatch. Production selects token
// tile 4 at T>=64 and preserves the serial incumbent below that stable physical crossover. The
// candidate boundary retains the qualified 4/8/16/32 sweep; the incumbent boundary bypasses
// production selection so later comparisons cannot accidentally time the selected route twice.
[[nodiscard]] hipError_t causal_conv1d_silu_prefill_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight,
    const hip_bfloat16* history_in, hip_bfloat16* history_out,
    hip_bfloat16* output, std::uint32_t channels, std::uint32_t tokens,
    std::uint32_t token_tile, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t causal_conv1d_silu_incumbent_qualification(
    const hip_bfloat16* input, const hip_bfloat16* weight,
    const hip_bfloat16* history_in, hip_bfloat16* history_out,
    hip_bfloat16* output, std::uint32_t channels, std::uint32_t tokens,
    hipStream_t stream) noexcept;

// Fixed ordinary-prefill Qwen3.8 projection/convolution handoff. The represented BF16
// projection outputs remain separate: query_key is [2048,4096] and value_z is [2048,12288].
// The operation applies the incumbent width-four causal convolution and SiLU formula directly
// into query [2048,2048], key [2048,2048], and value [2048,6144], copies the upper value_z half
// to z [2048,6144], then publishes the final three represented projection columns to history.
// history_in/history_out are disjoint or exactly identical. This target-specific entry accepts
// exactly 2048 tokens; snapshot/replay and every other ordinary width use their existing paths.
[[nodiscard]] hipError_t projection_conv_prefill_p2048_direct_scatter_bf16(
    const hip_bfloat16* query_key, const hip_bfloat16* value_z,
    const hip_bfloat16* conv_weight, const hip_bfloat16* history_in,
    hip_bfloat16* history_out, hip_bfloat16* query, hip_bfloat16* key,
    hip_bfloat16* value, hip_bfloat16* z, std::uint32_t tokens,
    hipStream_t stream) noexcept;

// Qwen3.8-27B verification projection/convolution at the fixed real geometry. `query_key` is
// BF16 [B*W,4096] in column-major Tensor storage, `value_z` is BF16 [B*W,12288] in [value,z]
// row order, convolution weights are BF16 [10240,4], and the state pool is BF16
// [10240,3,state_slots]. Selectors are device I32 [B]. Query/key/value invalid tails are exact
// zero, while z is copied for every physical column. Each valid token publishes its post-token
// width-three history to snapshot_base_slots[b]+token. The caller guarantees distinct destination
// intervals and prevents one row from overwriting another row's initial slot.
[[nodiscard]] hipError_t projection_conv_snapshot_bf16(
    const hip_bfloat16* query_key, const hip_bfloat16* value_z,
    const hip_bfloat16* conv_weight, hip_bfloat16* conv_states,
    const std::int32_t* valid_columns, const std::int32_t* initial_state_slots,
    const std::int32_t* snapshot_base_slots, hip_bfloat16* query, hip_bfloat16* key,
    hip_bfloat16* value, hip_bfloat16* z, std::uint32_t width, std::uint32_t batch,
    std::uint32_t state_slots, hipStream_t stream) noexcept;

// Replay-record form of the same fixed projection/convolution. `conv_states` is immutable and
// `conv_record` receives the represented BF16 q/k/value projection for every valid token. With a
// null parent tensor, columns are sequential. Otherwise parent_index is I32 [W,B] in column-major
// Tensor storage: a negative parent selects the checkpoint and a nonnegative parent must precede
// the child. Width is 2..16 and batch is 1..4.
[[nodiscard]] hipError_t projection_conv_record_bf16(
    const hip_bfloat16* query_key, const hip_bfloat16* value_z,
    const hip_bfloat16* conv_weight, const hip_bfloat16* conv_states,
    const std::int32_t* valid_columns, const std::int32_t* initial_state_slots,
    const std::int32_t* parent_index, hip_bfloat16* conv_record, hip_bfloat16* query,
    hip_bfloat16* key, hip_bfloat16* value, hip_bfloat16* z, std::uint32_t width,
    std::uint32_t batch, std::uint32_t state_slots, hipStream_t stream) noexcept;

// Makes the raw FP32 GDN controls. `a`/`b` are represented BF16 [tokens, value_heads], and
// `a_log`/`dt_bias` are FP32 [value_heads]. Outputs g/beta are FP32 [tokens, value_heads]:
// g=-exp(a_log)*softplus(a+dt_bias), beta=sigmoid(b), where softplus(x)=x for x>20 and otherwise
// log1p(exp(x)). All output elements are independently defined.
[[nodiscard]] hipError_t control_gates_bf16(const hip_bfloat16* a, const hip_bfloat16* b,
                                            const float* a_log, const float* dt_bias, float* g,
                                            float* beta, std::uint32_t value_heads,
                                            std::uint32_t tokens, hipStream_t stream) noexcept;

// Exact FP32 state movement for transaction/checkpoint publication. Source/destination are
// non-overlapping FP32 elements and count is positive.
[[nodiscard]] hipError_t copy_state_fp32(const float* source, float* destination,
                                         std::size_t count, hipStream_t stream) noexcept;

// Applies the causal normalized Gated DeltaNet recurrence. q/k are represented BF16
// [tokens, qk_heads, key_dim], v/out are BF16 [tokens, value_heads, value_dim], g/beta are FP32
// [tokens, value_heads], and state is FP32 [value_heads, value_dim, key_dim]. For value head h,
// q/k head is floor(h/(value_heads/qk_heads)). For each token: normalize q/k in FP32 using
// x/sqrt(sum(x*x)+epsilon), decay state with exp(g), form beta*(v-decayed_state*k), rank-one
// update, and emit BF16 scale*(updated_state*q). state_in/state_out may be disjoint or exactly
// identical; state_out receives only the final state after all tokens. Inputs/out do not overlap
// state storage. Dimensions are positive, value_heads is divisible by qk_heads, scale/epsilon are
// finite with epsilon positive, and all represented values are finite.
[[nodiscard]] hipError_t recurrent_bf16_fp32(const hip_bfloat16* q, const hip_bfloat16* k,
                                              const hip_bfloat16* v, const float* g,
                                              const float* beta, const float* state_in,
                                              float* state_out, hip_bfloat16* output,
                                              std::uint32_t key_dim, std::uint32_t value_dim,
                                              std::uint32_t qk_heads,
                                              std::uint32_t value_heads,
                                              std::uint32_t tokens, float scale, float epsilon,
                                              hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::gdn
