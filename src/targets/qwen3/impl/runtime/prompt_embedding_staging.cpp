#include "targets/qwen3/impl/runtime/prompt_embedding_staging.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3::detail {

std::uint64_t PromptEmbeddingStaging::capacity_bytes(QType qtype, std::int32_t features,
                                                     std::int32_t capacity_ids) {
    return ops::embedding_stage_capacity_bytes(qtype, features, capacity_ids);
}

PromptEmbeddingStaging::PromptEmbeddingStaging(DeviceContext& device, const Weight& table,
                                               DeviceSpan device_image,
                                               std::int32_t capacity_ids)
    : device_(device), table_(table), device_image_(device_image), capacity_ids_(capacity_ids),
      host_image_(static_cast<std::size_t>(
          capacity_bytes(table.qtype, table.ndim == 2 ? table.shape[1] : 0, capacity_ids))) {
    if (device_image_.data == nullptr || device_image_.bytes < host_image_.size()) {
        throw std::invalid_argument("prompt embedding staging region is smaller than its image");
    }
    HIP_CHECK(hipEventCreateWithFlags(&copied_, hipEventDisableTiming));
    const hipError_t created = hipEventCreateWithFlags(&released_, hipEventDisableTiming);
    if (created != hipSuccess) {
        (void)hipEventDestroy(copied_);
        HIP_CHECK(created);
    }
}

PromptEmbeddingStaging::~PromptEmbeddingStaging() {
    // The load stream may still read the pinned image.
    if (copy_recorded_) { (void)hipEventSynchronize(copied_); }
    (void)hipEventDestroy(released_);
    (void)hipEventDestroy(copied_);
}

void PromptEmbeddingStaging::stage(std::span<const std::int32_t> window) {
    if (window.empty() || window.size() > static_cast<std::size_t>(capacity_ids_)) {
        throw std::invalid_argument("prompt embedding window is outside the staging capacity");
    }
    // The host image is rewritten only after its previous copy has read it.
    if (copy_recorded_) { HIP_CHECK(hipEventSynchronize(copied_)); }
    staged_ids_.clear();
    stage_ = ops::stage_embedding_rows(
        window, table_,
        std::span<std::byte>(static_cast<std::byte*>(host_image_.data()), host_image_.size()));
    // The device region is rewritten only after the last lease's reads.
    if (release_recorded_) { HIP_CHECK(hipStreamWaitEvent(device_.load_stream, released_, 0)); }
    HIP_CHECK(hipMemcpyAsync(device_image_.data, host_image_.data(),
                             static_cast<std::size_t>(stage_.bytes), hipMemcpyHostToDevice,
                             device_.load_stream));
    HIP_CHECK(hipEventRecord(copied_, device_.load_stream));
    copy_recorded_ = true;
    staged_ids_.assign(window.begin(), window.end());
}

PromptEmbeddingStaging::Lease PromptEmbeddingStaging::acquire(
    std::span<const std::int32_t> window) {
    if (leased_) { throw std::logic_error("prompt embedding staging is already leased"); }
    const bool resident = !window.empty() && window.size() <= staged_ids_.size() &&
                          std::equal(window.begin(), window.end(), staged_ids_.begin());
    if (!resident) { stage(window); }
    HIP_CHECK(hipStreamWaitEvent(device_.stream, copied_, 0));
    ops::StagedEmbedding view = ops::staged_embedding(stage_, table_, device_image_.data);
    view.slots = view.slots.slice(0, 0, static_cast<std::int32_t>(window.size()));
    leased_    = true;
    return Lease(*this, view);
}

void PromptEmbeddingStaging::prefetch(std::span<const std::int32_t> window) {
    if (leased_) { throw std::logic_error("prompt embedding prefetch overlaps a lease"); }
    stage(window);
}

void PromptEmbeddingStaging::release() noexcept {
    // Recording on the compute stream cannot be skipped: without it the next copy could
    // overwrite rows that enqueued gathers have not read yet.
    const hipError_t recorded = hipEventRecord(released_, device_.stream);
    if (recorded != hipSuccess) {
        // The stream is unusable; drain it so no enqueued read outlives the image.
        (void)hipStreamSynchronize(device_.stream);
    }
    // After a failed record the stream is drained, so waiting on a stale event is safe.
    release_recorded_ = true;
    leased_           = false;
}

PromptEmbeddingStaging::Lease::Lease(Lease&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), view_(other.view_) {}

PromptEmbeddingStaging::Lease::~Lease() { end(); }

void PromptEmbeddingStaging::Lease::end() {
    if (owner_ != nullptr) { std::exchange(owner_, nullptr)->release(); }
}

} // namespace ninfer::targets::qwen3::detail
