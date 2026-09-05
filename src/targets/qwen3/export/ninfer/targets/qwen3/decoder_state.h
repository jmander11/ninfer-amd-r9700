#pragma once

#include "core/fp8_int4_paged_kv_cache.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"

#include <hip/hip_bfloat16.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::targets::qwen3 {

// The sole Qwen3.8 growing-cache identity. K is direct OCP E4M3FN, V is canonical signed INT4,
// and each V group owns one FP16 scale. Text and MTP have distinct physical pools/layouts while
// sharing one statically selected V group. There is no dtype, K scale, K mean, sparsity policy,
// keep-fraction, or alternate growing-cache branch.
struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    std::int32_t value_group                = 0;
    Fp8KInt4VPlaneLayouts text_plane_layouts;
    Fp8KInt4VPlaneLayouts mtp_plane_layouts;
    bool enable_mtp                         = false;
    std::int32_t kv_table_rows              = 0;
    std::uint32_t text_physical_page_groups = 0;
    std::uint32_t mtp_physical_page_groups  = 0;
    LinearAttentionStatePoolSpec linear_attention;
};

struct PagedKVCacheLayout {
    Fp8KInt4VPagedKVPoolLayout storage;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;

    [[nodiscard]] std::size_t payload_bytes() const noexcept {
        return storage.storage.payload_bytes();
    }
};

// Per-sequence publication state is separate from the shared physical pool. A failed or abandoned
// launched mutation poisons only this sequence; no cache frontier can advance before device status
// has been copied back and the ordered stream has completed.
struct PagedKVPublication {
    std::uint32_t valid_frontier = 0;
    bool healthy                 = true;
    bool transaction_open        = false;
    std::uint64_t transaction_generation = 0;
};

struct PagedKVTransactionWorkspace {
    // Host-position transactions stage through positions. Device-position transactions consume
    // their caller-owned I32 panel directly and require only status.
    std::uint32_t* positions = nullptr;
    std::size_t position_capacity = 0;
    std::uint32_t* status = nullptr;
    // Fixed-address cumulative frontier for a segmented MTP round. Ordinary one-segment and
    // compaction transactions leave this null.
    std::uint32_t* cursor = nullptr;
};

class PagedKVCache;
class PagedKVTransaction;
class PagedKVLayerRead;

class PagedKVCacheView {
public:
    PagedKVCacheView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }
    [[nodiscard]] std::uint32_t max_context() const noexcept;
    [[nodiscard]] Fp8KInt4VPagedKVLayerView layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCache;
    PagedKVCacheView(const PagedKVCache& cache, const PagedKVAllocation& allocation) noexcept;

    const PagedKVCache* cache_ = nullptr;
    const PagedKVAllocation* allocation_ = nullptr;
};

class PagedKVCache {
public:
    PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout);

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }
    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }
    [[nodiscard]] const Fp8KInt4VPagedKVSpec& spec() const noexcept { return spec_; }
    [[nodiscard]] Fp8KInt4VSemanticFingerprint fingerprint() const noexcept {
        return fp8_k_int4_v_semantic_fingerprint(spec_);
    }
    [[nodiscard]] PagedKVPool& pool() noexcept { return pool_; }
    [[nodiscard]] const PagedKVPool& pool() const noexcept { return pool_; }

    [[nodiscard]] PagedKVCacheView execution_view(const PagedKVAllocation& allocation) const;
    [[nodiscard]] PagedKVLayerRead
    published_layer_read(const PagedKVAllocation& allocation,
                         const PagedKVPublication& publication,
                         std::uint32_t layer) const;
    [[nodiscard]] Fp8KInt4VPagedKVLayerView
    layer_view(std::int32_t layer, const PagedKVAllocation& allocation) const;
    [[nodiscard]] Fp8KInt4VAppendBinding
    bind_append(const PagedKVAllocation& allocation, std::int32_t layer,
                std::span<const std::uint32_t> logical_positions) const;

    [[nodiscard]] PagedKVTransaction
    begin_append(PagedKVAllocation& allocation, PagedKVPublication& publication,
                 std::span<const std::uint32_t> logical_positions,
                 PagedKVTransactionWorkspace workspace);
    [[nodiscard]] PagedKVTransaction
    begin_device_append(PagedKVAllocation& allocation, PagedKVPublication& publication,
                        const std::int32_t* device_positions, std::size_t position_count,
                        PagedKVTransactionWorkspace workspace);
    // Opens a fixed-maximum launch whose active prefix count is produced on device. Pending reads
    // are authorized only up to the conservative maximum and must still consume causal row
    // positions; commit resolves the count and publishes exactly that many tokens, including zero.
    [[nodiscard]] PagedKVTransaction
    begin_device_prefix_append(PagedKVAllocation& allocation, PagedKVPublication& publication,
                               const std::int32_t* device_positions,
                               std::size_t maximum_position_count,
                               const std::int32_t* device_position_count,
                               PagedKVTransactionWorkspace workspace);
    // Opens one graph-capturable MTP round. The device base is copied into workspace.cursor once,
    // then each segment validates against and advances that cursor on the ordered stream. The
    // host publication remains open until one batched post-execution resolution.
    [[nodiscard]] PagedKVTransaction
    begin_device_segmented_append(PagedKVAllocation& allocation,
                                  PagedKVPublication& publication,
                                  const std::int32_t* device_base_frontier,
                                  std::uint32_t minimum_base_frontier,
                                  std::uint32_t maximum_base_frontier,
                                  std::uint32_t visible_frontier_limit,
                                  PagedKVTransactionWorkspace workspace,
                                  const std::int32_t* device_table_row = nullptr);
    [[nodiscard]] PagedKVTransaction
    begin_compact(PagedKVAllocation& allocation, PagedKVPublication& publication,
                  PagedKVTransactionWorkspace workspace);

    // Close a previously published speculative suffix without touching its now-invisible bytes.
    // This is the zero-copy rollback counterpart to begin_compact: callers use begin_compact when
    // accepted tokens must be gathered into a different logical order, and truncate_publication
    // when the retained prefix is already physically in place (including a zero-token rollback).
    void truncate_publication(const PagedKVAllocation& allocation,
                              PagedKVPublication& publication,
                              std::uint32_t retained_frontier) const;

private:
    friend class PagedKVCacheView;
    friend class PagedKVTransaction;

    PagedKVPool pool_;
    Fp8KInt4VPagedKVSpec spec_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
};

// A non-forgeable, ephemeral authorization to consume one typed cache layer. A committed read is
// valid while its publication remains closed at the same generation/frontier. A pending read is
// produced only by the append transaction after that layer's codec launch and is valid only on
// the transaction's ordered stream. Thus a decoder layer may attend its newly appended K/V before
// the all-layer transaction commits without exposing the pending frontier to any other consumer.
class PagedKVLayerRead {
public:
    PagedKVLayerRead() noexcept = default;

    [[nodiscard]] bool valid_for(hipStream_t stream) const noexcept;
    [[nodiscard]] const Fp8KInt4VPagedKVLayerView& layer() const noexcept { return layer_; }
    [[nodiscard]] std::uint32_t visible_frontier() const noexcept {
        return visible_frontier_;
    }
    [[nodiscard]] std::uint32_t mapped_pages() const noexcept { return mapped_pages_; }
    [[nodiscard]] bool pending() const noexcept { return pending_; }
    [[nodiscard]] const std::int32_t* device_table_row() const noexcept {
        return device_table_row_;
    }
    [[nodiscard]] const std::uint32_t* pool_block_tables() const noexcept {
        return pool_block_tables_;
    }
    [[nodiscard]] std::uint32_t pool_table_row_stride() const noexcept {
        return pool_table_row_stride_;
    }
    [[nodiscard]] std::uint32_t pool_table_row_count() const noexcept {
        return pool_table_row_count_;
    }

private:
    friend class PagedKVCache;
    friend class PagedKVTransaction;

    PagedKVLayerRead(Fp8KInt4VPagedKVLayerView layer,
                     const PagedKVAllocation& allocation,
                     const PagedKVPublication& publication,
                     std::uint32_t publication_frontier,
                     std::uint32_t visible_frontier,
                     std::uint32_t mapped_pages,
                     std::int32_t bound_row,
                     std::uint64_t mapping_generation,
                     std::uint64_t generation,
                     hipStream_t ordered_stream,
                     bool pending,
                     const std::int32_t* device_table_row = nullptr,
                     const std::uint32_t* pool_block_tables = nullptr,
                     std::uint32_t pool_table_row_stride = 0,
                     std::uint32_t pool_table_row_count = 0) noexcept;

    Fp8KInt4VPagedKVLayerView layer_;
    const PagedKVAllocation* allocation_ = nullptr;
    const PagedKVPublication* publication_ = nullptr;
    std::uint32_t publication_frontier_ = 0;
    std::uint32_t visible_frontier_ = 0;
    std::uint32_t mapped_pages_ = 0;
    std::int32_t bound_row_ = -1;
    std::uint64_t mapping_generation_ = 0;
    std::uint64_t generation_ = 0;
    hipStream_t ordered_stream_ = nullptr;
    bool pending_ = false;
    const std::int32_t* device_table_row_ = nullptr;
    const std::uint32_t* pool_block_tables_ = nullptr;
    std::uint32_t pool_table_row_stride_ = 0;
    std::uint32_t pool_table_row_count_ = 0;
};

// One transaction covers every layer of one Text or MTP cache. It stages checked host positions
// once or consumes a caller-owned device I32 panel directly, clears one status word once, requires
// every layer exactly once on one stream, and only then publishes the new sequence frontier. The
// represented K/V inputs for each layer are device BF16.
class PagedKVTransaction {
public:
    ~PagedKVTransaction() noexcept;
    PagedKVTransaction(const PagedKVTransaction&) = delete;
    PagedKVTransaction& operator=(const PagedKVTransaction&) = delete;
    PagedKVTransaction(PagedKVTransaction&& other) noexcept;
    PagedKVTransaction& operator=(PagedKVTransaction&& other) noexcept;

    void launch_append_layer(std::uint32_t layer, const hip_bfloat16* keys,
                             const hip_bfloat16* values, hipStream_t stream);
    void launch_compact_layer(std::uint32_t layer, std::uint32_t prefix,
                              std::span<const std::uint32_t> selected_path,
                              hipStream_t stream);
    void begin_device_segment(const std::int32_t* device_positions,
                              std::size_t maximum_position_count,
                              const std::int32_t* device_position_count);
    void end_device_segment(hipStream_t stream);
    // Graph replay executes already-captured codec/cursor nodes. This marks the host authority as
    // in flight so the same one-shot resolution and failure poisoning rules apply.
    void mark_graph_replay(hipStream_t stream);
    void enqueue_segmented_resolution(std::uint32_t* host_status,
                                      std::uint32_t* host_cursor);
    void finish_segmented_resolution(std::uint32_t status, std::uint32_t cursor,
                                     std::uint32_t retained_frontier);
    // Capture records device work but must neither publish the representative frontier nor leave
    // a dummy host authority open after hipStreamEndCapture.
    void close_captured_segmented();
    // For device-prefix transactions this is the conservative launch maximum; the committed
    // count remains device-owned until commit() resolves it.
    [[nodiscard]] std::size_t position_count() const noexcept { return position_count_; }
    [[nodiscard]] PagedKVLayerRead pending_layer_read(std::uint32_t layer) const;
    [[nodiscard]] std::uint32_t commit();
    void abort() noexcept;

private:
    friend class PagedKVCache;
    PagedKVTransaction(PagedKVCache& cache, PagedKVAllocation& allocation,
                       PagedKVPublication& publication,
                       std::span<const std::uint32_t> positions,
                       const std::int32_t* device_positions, std::size_t position_count,
                       const std::int32_t* device_position_count,
                       PagedKVTransactionWorkspace workspace, bool append_mode,
                       const std::int32_t* device_base_frontier = nullptr,
                       std::uint32_t minimum_base_frontier = 0,
                       std::uint32_t maximum_base_frontier = 0,
                       std::uint32_t visible_frontier_limit = 0,
                       const std::int32_t* device_table_row = nullptr);

    void prepare_launch(hipStream_t stream);
    void require_open(const char* operation) const;
    void require_new_layer(std::uint32_t layer, const char* operation) const;
    void finish(bool poison) noexcept;

    PagedKVCache* cache_ = nullptr;
    PagedKVAllocation* allocation_ = nullptr;
    PagedKVPublication* publication_ = nullptr;
    std::span<const std::uint32_t> positions_;
    const std::int32_t* device_positions_ = nullptr;
    std::size_t position_count_ = 0;
    const std::int32_t* device_position_count_ = nullptr;
    const std::int32_t* device_base_frontier_ = nullptr;
    const std::int32_t* device_table_row_ = nullptr;
    PagedKVTransactionWorkspace workspace_;
    hipStream_t stream_ = nullptr;
    std::uint32_t published_frontier_ = 0;
    std::uint32_t initial_frontier_ = 0;
    std::uint32_t minimum_base_frontier_ = 0;
    std::uint32_t maximum_base_frontier_ = 0;
    std::uint32_t visible_frontier_limit_ = 0;
    std::uint32_t compact_prefix_ = 0;
    std::uint64_t layers_launched_ = 0;
    std::uint64_t allocation_mapping_generation_ = 0;
    std::int32_t allocation_bound_row_ = -1;
    bool append_mode_ = false;
    bool segmented_mode_ = false;
    bool segment_open_ = false;
    bool launched_ = false;
    bool positions_uploaded_ = false;
    bool closed_ = false;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;
    LinearAttentionStatePoolLayout linear_attention;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;
    LinearAttentionStatePool linear_attention;

    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::targets::qwen3
