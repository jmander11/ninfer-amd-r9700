#pragma once

#include "core/fp8_int4_paged_kv_cache.h"

#include <ninfer/targets/qwen3/decoder_state.h>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_27b::detail {

// Target-private bridge from the Qwen sequence publication/cache view into the owned R9700 A3
// Op. Qwen3.8 is presently registered through this package and has one fixed full-attention
// geometry: D256, Hq24, Hkv4, Q:KV=6, any positive T within the target's 262,144-token
// native context capacity, and scale 1/sqrt(256).
struct R9700FullAttentionArgs {
    Tensor query;  // contiguous BF16 [256,24,T]
    qwen3::PagedKVLayerRead cache_read;
    const std::int32_t* row_positions = nullptr; // optional device I32 [T], each in [0,frontier)
    // Optional paired device I32 packed-tree visibility metadata: masks are [T], while prefixes
    // are [T] or one broadcast scalar. The raw Op validates represented values per row; this
    // bridge validates atomic pairing and the explicit zero-or-one prefix stride.
    const std::int32_t* ancestor_masks = nullptr;
    const std::int32_t* prefix_lengths = nullptr;
    std::uint32_t prefix_length_stride = 1; // zero broadcasts one prefix; one is rowwise
    // Optional device I32 scalar. Null activates all T rows; a valid scalar activates its prefix
    // and zeroes the fixed-width tail without reading tail metadata or cache state.
    const std::int32_t* active_query_rows = nullptr;
    // Explicit family-schedule identity: true only inside DFlash target verification. Shape alone
    // must not route an unrelated Text remainder through a qualified DFlash leaf.
    bool dflash_target_verify = false;
    // Caller-owned storage at the planner's stable peak. Dense initial-prefix calls use the
    // rectangular FP32 score plane plus maxima; decode uses the ordinary-WMMA or split arena.
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;
    Tensor output;                             // contiguous FP32 [256,24,T]
};

// Returns the exact caller-owned score storage for one ordinary causal WMMA call, or zero for an
// invalid/overflowing visible context or a compile-time score-streaming control build.
[[nodiscard]] std::size_t r9700_full_attention_score_workspace_capacity_bytes(
    std::size_t visible_context) noexcept;

// Returns the caller-owned peak for every production route reachable within these envelope
// maxima, including an initial-prefix dense P<=4096 score plane. This is suitable for startup
// planning; a concrete call consumes only its exact selected-route prefix of the stable span.
[[nodiscard]] std::size_t r9700_full_attention_workspace_capacity_bytes(
    std::uint32_t query_rows, std::size_t visible_context,
    bool tree_or_device_count, bool dflash_target_verify = false) noexcept;

// Dispatches initial-prefix P128..4096 through staged full-score GQA6 and qualified long-context
// T=1/T=4 through split-512. The default-off parity profile additionally overwrites only P129's
// tail with W1 WMMA and admits chain W5 batched WMMA; every other width retains its existing route.
[[nodiscard]] hipError_t r9700_qwen3_8_27b_full_attention(const R9700FullAttentionArgs& args,
                                                      hipStream_t stream) noexcept;

#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
// Private model-gate route compiled only into an explicitly configured qualification build.
// T>=2 ordinary prefill consumes the B128 XAttention evaluator; the Variant keeps T=1 on its
// existing dense leaf before entering this boundary.
[[nodiscard]] std::size_t r9700_qwen3_8_27b_text_prefill_attention_workspace_capacity_bytes(
    std::uint32_t maximum_query_rows, std::size_t maximum_visible_context) noexcept;
[[nodiscard]] hipError_t r9700_qwen3_8_27b_text_prefill_attention(
    const R9700FullAttentionArgs& args, void* workspace, std::size_t workspace_bytes,
    hipStream_t stream) noexcept;
#endif

} // namespace ninfer::targets::qwen3_8_27b::detail
