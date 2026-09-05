#pragma once

#include "fp8_int4_kv_attention.h"
#include "r9700_xattention_profile.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::kv {

// Caller-owned XAttention scratch. One slot is shared by the 128 causal query rows in a paper
// block for one Q head. keep_pages is emitted in increasing logical-page order, making the
// sparse consumer deterministic and preserving dense token order when tau=1.
struct XAttentionWorkspaceView {
    hip_bfloat16* packed_keys = nullptr;      // [Hkv,context,D], ranker and sparse consumer input
    float* page_mass = nullptr;              // [QH,ceil(T/128),rank_capacity], normalized total
    float* plane_mass = nullptr;              // [QH,ceil(T/128),128/S,rank_capacity]
    std::uint16_t* rank_and_keep = nullptr;  // [QH,ceil(T/128),keep_stride]
    std::uint8_t* selected = nullptr;         // page_mass extent; temporary selection bitmap
    std::int32_t* keep_count = nullptr;       // [QH,ceil(T/128)]; negative means invalid input
    std::uint32_t rank_capacity = 0;          // power of two, >= ceil(context/128)
    std::uint32_t keep_stride = 0;            // ceil(context/64) expanded physical-page IDs
    std::uint32_t query_blocks = 0;
};

// Returns the exact bytes needed by bind_xattention_workspace. Zero means invalid or overflow.
// The caller span starts at a 256-byte-aligned address. Packed logical BF16 K dominates;
// per-query storage is QH * ceil(T/128), with B128 per-plane
// mass/rank storage and an expanded B64 keep list. There is no full-logit or hidden allocation.
[[nodiscard]] std::size_t fp8_int4_kv_xattention_workspace_capacity_bytes(
    std::uint32_t query_heads, std::uint32_t kv_heads, std::uint32_t query_rows,
    std::size_t context) noexcept;

[[nodiscard]] XAttentionWorkspaceView bind_fp8_int4_kv_xattention_workspace(
    void* workspace, std::size_t workspace_bytes, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t query_rows, std::size_t context) noexcept;

struct Fp8Int4KvXAttentionArgs {
    Fp8Int4KvAttentionArgs attention;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;
};

// Paper inverse-stride antidiagonal estimator ranks B128 blocks and expands each retained block
// to the fixed 64-token cache pages consumed by attention.
// This evaluator is causal prefill only: row_positions must be a contiguous increasing suffix,
// tree/device-active metadata is rejected, and ordinary decode/speculative verification retain
// their qualified dense routes. It uses the compile-selected stride/tau profile above.
[[nodiscard]] hipError_t fp8_int4_kv_xattention_prefill(
    const Fp8Int4KvXAttentionArgs& args, hipStream_t stream) noexcept;

// Stage entry points let the qualification executable attribute the composed Op's separate
// launches. The consumer requires a successful rank call over the same arguments and workspace;
// product execution uses only the composed prefill entry point exposed through its target leaf.
[[nodiscard]] hipError_t fp8_int4_kv_xattention_rank_qualification(
    const Fp8Int4KvXAttentionArgs& args, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t fp8_int4_kv_xattention_consumer_qualification(
    const Fp8Int4KvXAttentionArgs& args, hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::kv
