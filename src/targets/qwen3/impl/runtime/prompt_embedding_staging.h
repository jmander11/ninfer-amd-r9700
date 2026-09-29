#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/embedding.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3::detail {

// Prompt rows of the pinned-host token embedding. A prefill chunk knows its token ids on the host,
// so the host gathers the rows of a prompt window into a pinned image (ops::stage_embedding_rows),
// the load stream copies the image into one fixed device region, and the compute stream gathers
// from it with ops::embedding. The Program owns one instance for its lifetime.
//
// Overlap: after a chunk is enqueued, and while it runs, the host stages the next window of the
// same prompt and starts its copy behind the chunk's last read of the region: the chunk gather
// (the copy then overlaps the whole chunk), or with MTP the shifted MTP gather after the Text
// layers (the copy then overlaps the MTP layer). The next chunk reuses
// the image when its window is a prefix of the staged ids (the image is a pure function of the
// ids), and otherwise stages synchronously. Generated tokens are never staged: their ids exist
// only on the device, so decode, verify, drafter and MTP rows read the table in place.
class PromptEmbeddingStaging {
public:
    // Scope of one chunk's reads of the device region. Its end orders the next image copy after
    // every compute-stream read enqueued before it, including on unwinding.
    class Lease {
    public:
        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&&) = delete;
        ~Lease();

        [[nodiscard]] const ops::StagedEmbedding& view() const noexcept { return view_; }
        void end();

    private:
        friend class PromptEmbeddingStaging;
        Lease(PromptEmbeddingStaging& owner, ops::StagedEmbedding view) noexcept
            : owner_(&owner), view_(view) {}

        PromptEmbeddingStaging* owner_ = nullptr;
        ops::StagedEmbedding view_;
    };

    PromptEmbeddingStaging(DeviceContext& device, const Weight& table, DeviceSpan device_image,
                           std::int32_t capacity_ids);
    ~PromptEmbeddingStaging();
    PromptEmbeddingStaging(const PromptEmbeddingStaging&)            = delete;
    PromptEmbeddingStaging& operator=(const PromptEmbeddingStaging&) = delete;

    [[nodiscard]] static std::uint64_t capacity_bytes(QType qtype, std::int32_t features,
                                                      std::int32_t capacity_ids);
    [[nodiscard]] std::int32_t capacity_ids() const noexcept { return capacity_ids_; }

    // Makes the rows of `window` resident, staging them now unless the current image begins with
    // exactly these ids, and orders the compute stream after the image copy. Slot i of the view
    // is window[i]. At most one lease is open.
    [[nodiscard]] Lease acquire(std::span<const std::int32_t> window);
    // Stages `window` and starts its copy on the load stream; the copy waits for the reads of the
    // last lease. Called after a chunk is enqueued, so the host work overlaps its execution.
    void prefetch(std::span<const std::int32_t> window);

private:
    void stage(std::span<const std::int32_t> window);
    void release() noexcept;

    DeviceContext& device_;
    const Weight& table_;
    DeviceSpan device_image_;
    std::int32_t capacity_ids_ = 0;
    PinnedHostBuffer host_image_;
    hipEvent_t copied_   = nullptr;
    hipEvent_t released_ = nullptr;
    bool copy_recorded_     = false;
    bool release_recorded_  = false;
    bool leased_            = false;
    std::vector<std::int32_t> staged_ids_;
    ops::EmbeddingStage stage_{};
};

} // namespace ninfer::targets::qwen3::detail
