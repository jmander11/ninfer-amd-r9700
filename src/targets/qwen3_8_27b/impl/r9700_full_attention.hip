#include "r9700_full_attention.h"

#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
#include "ops/r9700/kv/fp8_int4_kv_xattention.h"
#endif
#include "targets/qwen3_8_27b/impl/config.h"

#include <algorithm>
#include <limits>

namespace ninfer::targets::qwen3_8_27b::detail {
namespace {

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kQueryHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kQueriesPerKv = 6;
constexpr std::uint32_t kPageSize = 64;
constexpr std::uint32_t kMaximumRows = kNativeContext;
constexpr float kQwenAttentionScale = 0.0625F;

[[nodiscard]] bool layout_valid(Fp8KInt4VPlaneLayout layout) noexcept {
    return layout == Fp8KInt4VPlaneLayout::FeatureFastestPageMajor ||
           layout == Fp8KInt4VPlaneLayout::TokenFastestHeadMajor;
}

[[nodiscard]] ops::r9700::kv::Fp8Int4KvPlaneLayout
raw_layout(Fp8KInt4VPlaneLayout layout) noexcept {
    return layout == Fp8KInt4VPlaneLayout::TokenFastestHeadMajor
               ? ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest
               : ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
}

[[nodiscard]] bool exact_contiguous_shape(const Tensor& tensor, DType dtype,
                                          std::int32_t d0, std::int32_t d1,
                                          std::int32_t d2 = 1,
                                          std::int32_t d3 = 1) noexcept {
    if (tensor.data == nullptr || tensor.dtype != dtype || tensor.ne[0] != d0 ||
        tensor.ne[1] != d1 || tensor.ne[2] != d2 || tensor.ne[3] != d3) {
        return false;
    }
    const std::int64_t element = static_cast<std::int64_t>(dtype_size(dtype));
    return tensor.nb[0] == element && tensor.nb[1] == element * d0 &&
           tensor.nb[2] == element * d0 * d1 && tensor.nb[3] == element * d0 * d1 * d2;
}

[[nodiscard]] bool contiguous_elements_at_least(const Tensor& tensor, DType dtype,
                                                 std::uint64_t minimum) noexcept {
    if (tensor.data == nullptr || tensor.dtype != dtype) { return false; }
    std::uint64_t elements = 1;
    std::int64_t stride = static_cast<std::int64_t>(dtype_size(dtype));
    for (int axis = 0; axis < 4; ++axis) {
        if (tensor.ne[axis] <= 0 || (tensor.ne[axis] != 1 && tensor.nb[axis] != stride)) {
            return false;
        }
        const auto extent = static_cast<std::uint64_t>(tensor.ne[axis]);
        if (elements > std::numeric_limits<std::uint64_t>::max() / extent) { return false; }
        elements *= extent;
        if (axis != 3 && tensor.ne[axis] != 1) {
            if (stride > std::numeric_limits<std::int64_t>::max() / tensor.ne[axis]) {
                return false;
            }
            stride *= tensor.ne[axis];
        }
    }
    return elements >= minimum;
}

[[nodiscard]] std::uint64_t tensor_elements(const Tensor& tensor) noexcept {
    std::uint64_t elements = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0 || elements > std::numeric_limits<std::uint64_t>::max() /
                                          static_cast<std::uint64_t>(extent)) {
            return 0;
        }
        elements *= static_cast<std::uint64_t>(extent);
    }
    return elements;
}

} // namespace

std::size_t r9700_full_attention_score_workspace_capacity_bytes(
    std::size_t visible_context) noexcept {
    if (!ops::r9700::kv::kFp8QkWmmaDecode) return 0U;
    constexpr std::size_t bytes_per_token = kQueryHeads * sizeof(float);
    if (visible_context == 0U ||
        visible_context > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        return 0U;
    }
    return visible_context * bytes_per_token;
}

std::size_t r9700_full_attention_workspace_capacity_bytes(
    std::uint32_t query_rows, std::size_t visible_context,
    bool tree_or_device_count, bool dflash_target_verify) noexcept {
    std::size_t required = 0U;
    if (ops::r9700::kv::use_split512_attention(
            query_rows, visible_context, tree_or_device_count)) {
        required = ops::r9700::kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(
            query_rows, visible_context);
    } else if (!tree_or_device_count &&
               ops::r9700::kv::use_fp8_qk_wmma(query_rows, visible_context)) {
        required = r9700_full_attention_score_workspace_capacity_bytes(visible_context);
    }
    if (ops::r9700::kv::kAttentionParityCandidate && dflash_target_verify &&
        !tree_or_device_count &&
        query_rows == 5U && visible_context >= ops::r9700::kv::kFp8QkWmmaT1MinimumContext) {
        const std::size_t candidate_context = std::min<std::size_t>(
            visible_context, ops::r9700::kv::kSplit512MinimumContext - 1U);
        required = std::max(
            required,
            ops::r9700::kv::fp8_int4_kv_attention_wmma_batched_w5_workspace_bytes(
                candidate_context));
    }
    // The planner supplies envelope maxima, not necessarily one exact call. Reserve the largest
    // initial-prefix dense score plane reachable anywhere inside that envelope. Runtime dispatch
    // uses only the exact prefix width and therefore retains one stable address within this peak.
    const std::uint32_t dense_rows = static_cast<std::uint32_t>(std::min<std::size_t>(
        std::min<std::size_t>(query_rows, visible_context), 4096U));
    if (dense_rows >= 128U) {
        required = std::max(
            required,
            ops::r9700::kv::fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(
                dense_rows));
    }
    return required;
}

static hipError_t launch_r9700_attention(
    const R9700FullAttentionArgs& a,
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
    bool xattention_prefill, void* xattention_workspace,
    std::size_t xattention_workspace_bytes,
#endif
    hipStream_t stream) noexcept {
    if (!a.cache_read.valid_for(stream) ||
        (a.ancestor_masks == nullptr) != (a.prefix_lengths == nullptr) ||
        (a.ancestor_masks != nullptr && a.row_positions == nullptr) ||
        a.prefix_length_stride > 1U) {
        return hipErrorInvalidValue;
    }
    const auto& cache = a.cache_read.layer();
    const std::uint32_t mapped_pages = a.cache_read.mapped_pages();
    const std::uint32_t visible_frontier = a.cache_read.visible_frontier();
    if (cache.head_dim != static_cast<std::int32_t>(kHeadDim) ||
        cache.num_kv_heads != static_cast<std::int32_t>(kKvHeads) ||
        (cache.value_group != 16 && cache.value_group != 32) ||
        !layout_valid(cache.plane_layouts.key) ||
        !layout_valid(cache.plane_layouts.value) ||
        !layout_valid(cache.plane_layouts.value_scale)) {
        return hipErrorInvalidValue;
    }

    const std::int32_t rows = a.query.ne[2];
    if (rows <= 0 || rows > static_cast<std::int32_t>(kMaximumRows) ||
        !exact_contiguous_shape(a.query, DType::BF16, kHeadDim, kQueryHeads, rows) ||
        !exact_contiguous_shape(a.output, DType::FP32, kHeadDim, kQueryHeads, rows)) {
        return hipErrorInvalidValue;
    }

    const std::uint64_t key_elements = tensor_elements(cache.key_fp8_pages);
    const std::uint64_t value_elements = tensor_elements(cache.value_int4_pages);
    const std::uint64_t scale_elements = tensor_elements(cache.value_scale_fp16_pages);
    constexpr std::uint64_t key_elements_per_token = kHeadDim * kKvHeads;
    if (!contiguous_elements_at_least(cache.key_fp8_pages, DType::FP8_E4M3FN, 1U) ||
        !contiguous_elements_at_least(cache.value_int4_pages, DType::U8, 1U) ||
        !contiguous_elements_at_least(cache.value_scale_fp16_pages, DType::FP16, 1U) ||
        key_elements == 0U || key_elements % key_elements_per_token != 0U) {
        return hipErrorInvalidValue;
    }
    const std::uint64_t physical_tokens = key_elements / key_elements_per_token;
    const std::uint64_t groups_per_token = kHeadDim /
                                            static_cast<std::uint32_t>(cache.value_group);
    if (physical_tokens == 0U || physical_tokens % kPageSize != 0U ||
        value_elements != physical_tokens * kKvHeads * (kHeadDim / 2U) ||
        scale_elements != physical_tokens * kKvHeads * groups_per_token ||
        physical_tokens > std::numeric_limits<std::size_t>::max()) {
        return hipErrorInvalidValue;
    }

    const std::uint64_t table_elements = tensor_elements(cache.block_table);
    const std::uint64_t frontier = visible_frontier;
    if (!contiguous_elements_at_least(cache.block_table, DType::I32, mapped_pages) ||
        mapped_pages > physical_tokens / kPageSize || mapped_pages > table_elements ||
        frontier > static_cast<std::uint64_t>(mapped_pages) * kPageSize ||
        frontier > table_elements * kPageSize ||
        frontier > std::numeric_limits<std::size_t>::max() ||
        frontier > std::numeric_limits<std::uint32_t>::max()) {
        return hipErrorInvalidValue;
    }

    const ops::r9700::kv::Fp8Int4KvAttentionArgs raw{
        .query = static_cast<const hip_bfloat16*>(a.query.data),
        .key_codes = static_cast<const std::uint8_t*>(cache.key_fp8_pages.data),
        .value_codes = static_cast<const std::uint8_t*>(cache.value_int4_pages.data),
        .value_scale_bits =
            static_cast<const std::uint16_t*>(cache.value_scale_fp16_pages.data),
        .page_table = a.cache_read.device_table_row() == nullptr
                          ? static_cast<const std::uint32_t*>(cache.block_table.data)
                          : a.cache_read.pool_block_tables(),
        .page_table_pages = mapped_pages,
        .page_table_row = a.cache_read.device_table_row(),
        .page_table_row_stride = a.cache_read.pool_table_row_stride(),
        .page_table_row_count = a.cache_read.pool_table_row_count(),
        .context = static_cast<std::size_t>(frontier),
        .physical_tokens = static_cast<std::size_t>(physical_tokens),
        .query_heads = kQueryHeads,
        .kv_heads = kKvHeads,
        .q_to_kv = kQueriesPerKv,
        .value_group = static_cast<std::uint32_t>(cache.value_group),
        .query_rows = static_cast<std::uint32_t>(rows),
        .row_positions = a.row_positions,
        .ancestor_masks = a.ancestor_masks,
        .prefix_lengths = a.prefix_lengths,
        .prefix_length_stride = a.prefix_length_stride,
        .active_query_rows = a.active_query_rows,
        .attention_scale = kQwenAttentionScale,
        .key_layout = raw_layout(cache.plane_layouts.key),
        .value_layout = raw_layout(cache.plane_layouts.value),
        .value_scale_layout = raw_layout(cache.plane_layouts.value_scale),
        .output = static_cast<float*>(a.output.data),
    };
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
    if (xattention_prefill) {
        if (a.ancestor_masks != nullptr || a.prefix_lengths != nullptr ||
            a.active_query_rows != nullptr || rows < 2) {
            return hipErrorInvalidValue;
        }
        return ops::r9700::kv::fp8_int4_kv_xattention_prefill(
            ops::r9700::kv::Fp8Int4KvXAttentionArgs{
                raw, xattention_workspace, xattention_workspace_bytes},
            stream);
    }
#endif
    const bool tree_or_device_count =
        a.ancestor_masks != nullptr || a.active_query_rows != nullptr;
    const bool dense_prefill = ops::r9700::kv::use_dense_prefill_attention(
        static_cast<std::uint32_t>(rows), raw.context) &&
        a.ancestor_masks == nullptr && a.prefix_lengths == nullptr &&
        raw.key_layout == ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest &&
        raw.value_layout == ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest &&
        raw.value_scale_layout == ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
    if (dense_prefill) {
        const std::size_t required =
            ops::r9700::kv::fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(
                static_cast<std::uint32_t>(rows));
        if (required == 0U || a.workspace == nullptr || a.workspace_bytes < required) {
            return hipErrorInvalidValue;
        }
        hipError_t error = ops::r9700::kv::fp8_int4_kv_attention_dense_prefill_full_score(
            raw, a.workspace, required, stream);
        if (error != hipSuccess || !ops::r9700::kv::use_text_p129_wmma_tail(
                                       static_cast<std::uint32_t>(rows), raw.context)) {
            return error;
        }
        const std::size_t tail_score_bytes =
            r9700_full_attention_score_workspace_capacity_bytes(raw.context);
        if (tail_score_bytes == 0U || tail_score_bytes > required) {
            return hipErrorInvalidValue;
        }
        // The dense consumers are enqueued first on this same stream, so its workspace front can
        // be reused by the following W1 launches. Overwrite only the final output row, leaving the
        // 128-row prefix byte-for-byte on the original dense route.
        auto tail = raw;
        constexpr std::size_t row_elements = kQueryHeads * kHeadDim;
        tail.query += 128U * row_elements;
        tail.query_rows = 1U;
        tail.row_positions += 128U;
        tail.score_workspace = static_cast<float*>(a.workspace);
        tail.output += 128U * row_elements;
        return ops::r9700::kv::fp8_int4_kv_attention_wmma(tail, stream);
    }
    const bool split512 = ops::r9700::kv::use_split512_attention(
        static_cast<std::uint32_t>(rows), raw.context, tree_or_device_count);
    if (split512) {
        const std::size_t required = r9700_full_attention_workspace_capacity_bytes(
            static_cast<std::uint32_t>(rows), raw.context, tree_or_device_count);
        if (required == 0U || a.workspace == nullptr || a.workspace_bytes < required) {
            return hipErrorInvalidValue;
        }
        return ops::r9700::kv::fp8_int4_kv_attention_split512(
            raw, a.workspace, required, stream);
    }
    if (ops::r9700::kv::use_dflash_w5_batched_wmma(
            static_cast<std::uint32_t>(rows), raw.context, tree_or_device_count,
            a.dflash_target_verify)) {
        const std::size_t required =
            ops::r9700::kv::fp8_int4_kv_attention_wmma_batched_w5_workspace_bytes(raw.context);
        if (required == 0U || a.workspace == nullptr || a.workspace_bytes < required) {
            return hipErrorInvalidValue;
        }
        auto wmma = raw;
        wmma.score_workspace = static_cast<float*>(a.workspace);
        return ops::r9700::kv::fp8_int4_kv_attention_wmma_batched_w5(wmma, stream);
    }
    const bool ordinary_wmma =
        ops::r9700::kv::use_fp8_qk_wmma(static_cast<std::uint32_t>(rows), raw.context) &&
                               a.ancestor_masks == nullptr && a.prefix_lengths == nullptr &&
                               a.active_query_rows == nullptr;
    if (ordinary_wmma) {
        const std::size_t required =
            r9700_full_attention_score_workspace_capacity_bytes(raw.context);
        if (required == 0U || a.workspace == nullptr || a.workspace_bytes < required) {
            return hipErrorInvalidValue;
        }
        auto wmma = raw;
        wmma.score_workspace = static_cast<float*>(a.workspace);
        return ops::r9700::kv::fp8_int4_kv_attention_wmma(wmma, stream);
    }
    return ops::r9700::kv::fp8_int4_kv_attention_fused(raw, stream);
}

hipError_t r9700_qwen3_8_27b_full_attention(const R9700FullAttentionArgs& args,
                                            hipStream_t stream) noexcept {
    return launch_r9700_attention(
        args,
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
        false, nullptr, 0U,
#endif
        stream);
}

#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
std::size_t r9700_qwen3_8_27b_text_prefill_attention_workspace_capacity_bytes(
    std::uint32_t maximum_query_rows, std::size_t maximum_visible_context) noexcept {
    if (maximum_query_rows < 2U) return 0U;
    return ops::r9700::kv::fp8_int4_kv_xattention_workspace_capacity_bytes(
        kQueryHeads, kKvHeads, maximum_query_rows, maximum_visible_context);
}

hipError_t r9700_qwen3_8_27b_text_prefill_attention(
    const R9700FullAttentionArgs& args, void* workspace, std::size_t workspace_bytes,
    hipStream_t stream) noexcept {
    if (args.query.ne[2] < 2) return hipErrorInvalidValue;
    return launch_r9700_attention(args, true, workspace, workspace_bytes, stream);
}
#endif

} // namespace ninfer::targets::qwen3_8_27b::detail
