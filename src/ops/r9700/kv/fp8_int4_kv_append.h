#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::kv {

// These two physical plane orders are qualification candidates. FeatureFastest is the typed
// [D,P,Hkv,Nphysical] page-major layout; TokenFastest is [P,D,Nphysical,Hkv] head-major. The
// eventual cache ABI selects one after the fused A3 attention benchmark and PPL gates; no caller
// may reinterpret bytes from one order as the other.
enum class Fp8Int4KvPlaneLayout : std::uint8_t {
    FeatureFastest,
    TokenFastest,
};

enum Fp8Int4KvAppendStatus : std::uint32_t {
    Fp8Int4KvAppendOk = 0,
    Fp8Int4KvAppendNonfiniteInput = 1U << 0U,
    Fp8Int4KvAppendScaleOverflow  = 1U << 1U,
    Fp8Int4KvAppendInvalidAddress = 1U << 2U,
    Fp8Int4KvAppendInvalidPosition = 1U << 3U,
    Fp8Int4KvAppendInvalidCount    = 1U << 4U,
};

// The Qwen3.8 main-text and MTP A2 append boundary. K and V are represented post-projection
// BF16 [tokens][kv_heads][256]. K codes are OCP E4M3FN bytes; V codes are adjacent packed signed
// INT4 bytes and V scales are IEEE binary16 bits. page_table maps logical 64-token pages to
// physical pages. This is a raw R9700-owned device contract, not a public Engine interface.
struct Fp8Int4KvAppendArgs {
    const hip_bfloat16* keys         = nullptr;
    const hip_bfloat16* values       = nullptr;
    const std::uint32_t* positions   = nullptr;
    std::size_t tokens                = 0;
    // Optional device I32 scalar selecting an active prefix in [0,tokens]. This supports MTP
    // alignment/AR panels whose valid count is produced on device. Inactive suffix columns are
    // not read or written. A malformed count sets InvalidCount and rejects the transaction.
    const std::int32_t* token_count   = nullptr;
    // Product appends are one contiguous suffix. The device validates positions[i] exactly
    // equals first_position+i so graph-fed I32 position panels cannot create gaps or aliases.
    std::uint32_t first_position      = 0;
    // A segmented MTP round instead supplies its fixed-address device cursor. Stream ordering
    // advances that cursor between segments; when non-null, first_position is ignored.
    const std::uint32_t* first_position_device = nullptr;
    std::uint32_t kv_heads            = 0;
    const std::uint32_t* page_table   = nullptr;
    std::uint32_t page_table_pages    = 0;
    // Graph-captured compact batches select a live sequence row from the stable pool-wide table.
    // Null keeps page_table as one already-selected row.
    const std::int32_t* page_table_row = nullptr;
    std::uint32_t page_table_row_stride = 0;
    std::uint32_t page_table_row_count = 0;
    std::size_t physical_tokens       = 0;
    std::uint8_t* key_codes           = nullptr;
    std::uint8_t* value_codes         = nullptr;
    std::uint16_t* value_scale_bits   = nullptr;
    // Zeroed by the caller before launch. The kernel atomically ORs Fp8Int4KvAppendStatus bits.
    // A nonzero result invalidates the entire append transaction; callers must not commit it.
    std::uint32_t* status              = nullptr;
    std::uint32_t value_group         = 0;
    Fp8Int4KvPlaneLayout key_layout = Fp8Int4KvPlaneLayout::FeatureFastest;
    Fp8Int4KvPlaneLayout value_layout = Fp8Int4KvPlaneLayout::FeatureFastest;
    Fp8Int4KvPlaneLayout value_scale_layout = Fp8Int4KvPlaneLayout::FeatureFastest;
};

// Launches the exact stored-format A2 append codec. physical_tokens must be a positive multiple
// of 64. page_table_pages must cover every supplied logical position; its page IDs must be in
// [0, physical_tokens / 64). positions must be the exact contiguous suffix
// [first_position, first_position+tokens), so concurrent blocks never write the same V byte or
// scale. The caller owns stream ordering, output-plane non-overlap, allocation-product overflow
// checks, and inspection of status before cache commit.
[[nodiscard]] hipError_t fp8_int4_kv_append(const Fp8Int4KvAppendArgs& args,
                                             hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::kv
