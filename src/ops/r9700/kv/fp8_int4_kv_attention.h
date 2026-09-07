#pragma once

#include "fp8_int4_kv_append.h"

namespace ninfer::ops::r9700::kv {

// First owned A3 QK stage: represented BF16 Q, paged OCP E4M3FN K, FP32 scores. It is a
// vector baseline used to qualify page translation and score numerics before the raw-WMMA
// challenger is admitted. q_to_kv is the fixed Qwen GQA ratio (six for Hq24/Hkv4).
struct Fp8KvQkArgs {
    const hip_bfloat16* query = nullptr; // [query_heads][256]
    const std::uint8_t* key_codes = nullptr;
    const std::uint32_t* page_table = nullptr;
    std::uint32_t page_table_pages = 0;
    const std::int32_t* page_table_row = nullptr;
    std::uint32_t page_table_row_stride = 0;
    std::uint32_t page_table_row_count = 0;
    std::size_t context = 0;
    std::size_t physical_tokens = 0;
    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t q_to_kv = 0;
    float attention_scale = 0.0F;
    Fp8Int4KvPlaneLayout key_layout = Fp8Int4KvPlaneLayout::FeatureFastest;
    float* scores = nullptr; // [query_heads][context]
};

[[nodiscard]] hipError_t fp8_kv_qk_vector(const Fp8KvQkArgs& args, hipStream_t stream) noexcept;

// Raw gfx12 wave32 FP8xFP8-to-FP32 WMMA challenger. It tiles one KV head's six Q heads by sixteen
// context tokens, leaves the unused ten WMMA rows zero, and uses the owned fragment map qualified
// by r9700_qual. Its execution profile explicitly quantizes represented BF16 Q to E4M3FN before
// the dot product; its FP8-Q oracle is therefore distinct from the vector baseline's BF16-Q oracle.
[[nodiscard]] hipError_t fp8_kv_qk_wmma(const Fp8KvQkArgs& args, hipStream_t stream) noexcept;

struct Int4KvPvArgs {
    const float* probabilities = nullptr; // [query_heads][context], FP32
    const std::uint8_t* value_codes = nullptr;
    const std::uint16_t* value_scale_bits = nullptr;
    const std::uint32_t* page_table = nullptr;
    std::uint32_t page_table_pages = 0;
    const std::int32_t* page_table_row = nullptr;
    std::uint32_t page_table_row_stride = 0;
    std::uint32_t page_table_row_count = 0;
    std::size_t context = 0;
    std::size_t physical_tokens = 0;
    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t q_to_kv = 0;
    std::uint32_t value_group = 0;
    Fp8Int4KvPlaneLayout value_layout = Fp8Int4KvPlaneLayout::FeatureFastest;
    Fp8Int4KvPlaneLayout value_scale_layout = Fp8Int4KvPlaneLayout::FeatureFastest;
    float* output = nullptr; // [query_heads][256]
};

[[nodiscard]] hipError_t int4_kv_pv_vector(const Int4KvPvArgs& args, hipStream_t stream) noexcept;

// Correctness-first fused A3 candidate. One wave32 block owns one Q head and streams the paged
// context without materializing scores: BF16-Q/FP8-K scores feed online FP32 softmax, then each
// thread dequantizes its signed INT4 V with its exact binary16 scale and accumulates FP32. This is
// the admissible baseline for later LDS/WMMA tuning because it preserves the public PV arithmetic.
struct Fp8Int4KvAttentionArgs {
    // [query_rows][query_heads][256]. A row is one causal query position. Rows share this cache
    // instance and page table but may have different visible causal frontiers. query_rows is any
    // positive U32 extent; the launcher stages it into legal gfx1201 grid-Y intervals, so the
    // same score-streaming implementation serves decode, verification, and configured prefill.
    const hip_bfloat16* query = nullptr;
    const std::uint8_t* key_codes = nullptr;
    const std::uint8_t* value_codes = nullptr;
    const std::uint16_t* value_scale_bits = nullptr;
    const std::uint32_t* page_table = nullptr;
    std::uint32_t page_table_pages = 0;
    const std::int32_t* page_table_row = nullptr;
    std::uint32_t page_table_row_stride = 0;
    std::uint32_t page_table_row_count = 0;
    std::size_t context = 0;
    std::size_t physical_tokens = 0;
    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t q_to_kv = 0;
    std::uint32_t value_group = 0;
    std::uint32_t query_rows = 1;
    // Optional device I32 array [query_rows] of zero-based causal positions. When null, every row
    // observes `context`; this is the single-row decode fast path. Each supplied position must be
    // in [0, context-1], and its visible context is position+1. The I32 representation is the
    // schedule's public position boundary and avoids a conversion buffer before attention.
    const std::int32_t* row_positions = nullptr;
    // Optional paired device I32 metadata for packed-tree verification. ancestor_masks is
    // [query_rows]; prefix_lengths is either one broadcast scalar or [query_rows]. A logical
    // cache token is visible to row r exactly when it precedes that row's prefix, or when its
    // zero-based suffix offset has the corresponding bit set in ancestor_masks[r]. The visible
    // suffix is bounded to fewer than 32 tokens. Supplying only one array is invalid. Invalid
    // device metadata poisons only the corresponding output row with NaN.
    const std::int32_t* ancestor_masks = nullptr;
    const std::int32_t* prefix_lengths = nullptr;
    // Prefix element stride: zero broadcasts prefix_lengths[0] to every row; one selects the
    // rowwise prefix_lengths[r] form. No other stride is part of this fixed schedule contract.
    std::uint32_t prefix_length_stride = 1;
    // Optional device I32 scalar selecting the active row prefix. Null means all query_rows.
    // A represented value in [0,query_rows] makes rows below it execute normally and makes every
    // later row exact positive zero without reading that row's query, causal/tree metadata, or
    // cache. A negative or over-width value poisons every output row with NaN. This device-side
    // boundary permits one fixed-width captured launch to serve dynamic MTP verification panels.
    const std::int32_t* active_query_rows = nullptr;
    float attention_scale = 0.0F;
    Fp8Int4KvPlaneLayout key_layout = Fp8Int4KvPlaneLayout::FeatureFastest;
    Fp8Int4KvPlaneLayout value_layout = Fp8Int4KvPlaneLayout::FeatureFastest;
    Fp8Int4KvPlaneLayout value_scale_layout = Fp8Int4KvPlaneLayout::FeatureFastest;
    // Optional [query_heads][context] FP32 workspace for the native-WMMA route. The launcher
    // reuses it across query rows on the ordered stream, so its persistent cost is independent
    // of T. The score-streaming baseline ignores this field.
    float* score_workspace = nullptr;
    float* output = nullptr; // [query_rows][query_heads][256]
};

[[nodiscard]] hipError_t fp8_int4_kv_attention_fused(const Fp8Int4KvAttentionArgs& args,
                                                      hipStream_t stream) noexcept;

struct DensePrefillWmmaResources {
    int registers = 0;
    int static_shared_bytes = 0;
    int local_bytes = 0;
};

// Production staged dense initial-prefix route. The caller owns one reusable workspace containing
// the rectangular FP32 score plane and one FP32 maximum per (row, query head). QK uses Bk16 below
// the selected P512 crossover and Bk32 from P512 through P4096 for both G16 and G32.
enum class DensePrefillFullScoreStage : std::uint32_t {
    QkBk16 = 0U,
    Maximum = 1U,
    Pv = 2U,
    QkBk32 = 3U,
};
[[nodiscard]] std::size_t fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(
    std::uint32_t query_rows) noexcept;
[[nodiscard]] hipError_t fp8_int4_kv_attention_dense_prefill_full_score(
    const Fp8Int4KvAttentionArgs& args, void* workspace, std::size_t workspace_bytes,
    hipStream_t stream) noexcept;
[[nodiscard]] hipError_t fp8_int4_kv_attention_dense_prefill_full_score_resources(
    std::uint32_t value_group, DensePrefillFullScoreStage stage,
    DensePrefillWmmaResources* resources) noexcept;

// Native gfx12 decode challenger: raw wave32 FP8-Q/FP8-K WMMA writes one reusable FP32 score workspace,
// followed by a stable FP32 softmax plus exact signed-INT4-times-FP16-scale PV consumer. Q's
// private E4M3 cast is an implementation profile and is checked directly against the same
// represented-input FP64 attention oracle; no FP16 probability or V materialization is introduced.
[[nodiscard]] hipError_t fp8_int4_kv_attention_wmma(const Fp8Int4KvAttentionArgs& args,
                                                     hipStream_t stream) noexcept;

// Qualification-only K4/W5 chain candidate. The caller owns five independent FP32 score planes;
// QK, stable Softmax, and exact INT4/FP16 PV each launch once across all five causal rows.
[[nodiscard]] std::size_t fp8_int4_kv_attention_wmma_batched_w5_workspace_bytes(
    std::size_t context) noexcept;
[[nodiscard]] hipError_t fp8_int4_kv_attention_wmma_batched_w5(
    const Fp8Int4KvAttentionArgs& args, hipStream_t stream) noexcept;

// Production long-context decode leaf for the one selected R9700 cache layout. T=1 keeps the
// qualified FP8-Q/FP8-K WMMA score profile; fixed-width T=4 uses represented-BF16 Q and shares one
// decoded 16-token K tile across all four rows and six query heads of each KV head. Both feed
// 512-token FP32 Softmax/INT4-V partials and one stable FP32 merge. The caller owns the exact,
// fixed-address workspace returned below; the implementation performs no device allocation.
[[nodiscard]] std::size_t fp8_int4_kv_attention_split512_workspace_capacity_bytes(
    std::uint32_t query_rows, std::size_t context) noexcept;
[[nodiscard]] hipError_t fp8_int4_kv_attention_split512(
    const Fp8Int4KvAttentionArgs& args, void* workspace, std::size_t workspace_bytes,
    hipStream_t stream) noexcept;

} // namespace ninfer::ops::r9700::kv
