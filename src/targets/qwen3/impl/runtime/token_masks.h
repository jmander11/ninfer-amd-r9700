#pragma once

#include "core/arena.h"
#include "ninfer/ops/sampling.h"
#include <ninfer/targets/qwen3/frontend.h>

#include <hip/hip_runtime_api.h>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3 {

// Program-owned, startup-sized exchange between speculative verification and the host tool
// grammar. Its device buffers and fine-grained host mailbox stay at fixed addresses for every
// captured graph, and the round's graph contains no host node:
//   publish()  once the verification ids exist: a kernel copies ids/parents/counts into the
//              mailbox and raises its request number;
//   the Program-owned matcher thread, armed for the round, matches the grammar while the target
//              forward runs and posts the configs, the restricted mask rows and the reply number;
//   acquire()  before the first read of the configs (target argmax): a kernel waits for the reply
//              and copies the configs and only the restricted mask rows into device storage.
// Every executed speculative round is bracketed by arm() and, after the round synchronized,
// finish_round(). If no reply arrives within two seconds the round proceeds unmasked and
// finish_round() throws.
class TokenMaskExchange {
public:
    TokenMaskExchange(Tensor masks, Tensor sampling);
    ~TokenMaskExchange();
    TokenMaskExchange(const TokenMaskExchange&)            = delete;
    TokenMaskExchange& operator=(const TokenMaskExchange&) = delete;

    void bind(std::span<const OutputSession* const> outputs,
              std::span<const ops::SamplingConfig> sampling);
    // Ordinary/prefill root sampling; called at a synchronized CPU boundary.
    [[nodiscard]] ops::SamplingConfig root(std::size_t row, hipStream_t stream);
    // A prefilling request's first-token sampling. Its mask occupies the last row, which no
    // decode batch reaches while a request prefills (the batch has at most capacity - 1 rows),
    // and the bound batch is left intact, so this may run inside a mixed round before its
    // work is enqueued.
    [[nodiscard]] ops::SamplingConfig prefill_root(const OutputSession* output,
                                                   ops::SamplingConfig config, hipStream_t stream);
    // Called inside the speculative graph after ids/parents are constructed. The returned
    // configs are target-only: draft proposal sampling keeps its own unmasked configs.
    [[nodiscard]] const ops::SamplingConfig* publish(const Tensor& ids, const Tensor* parents,
                                                     const Tensor& valid_columns,
                                                     hipStream_t stream);
    void acquire(hipStream_t stream);
    void arm();
    void disarm() noexcept;
    // Disarms, then reports a grammar failure or a missing reply of the synchronized round.
    void finish_round();

    struct Mailbox;

private:
    void serve() noexcept;
    void answer(std::uint32_t request) noexcept;
    void fill(bool tree);
    [[nodiscard]] ops::SamplingConfig root_into(const OutputSession* output,
                                                ops::SamplingConfig config, std::size_t row,
                                                hipStream_t stream);
    [[nodiscard]] std::uint32_t* host_mask(std::size_t row) const;
    [[nodiscard]] std::uint32_t* root_mask(std::size_t row) const;
    [[nodiscard]] const std::uint32_t* device_mask(std::size_t row) const;

    Tensor masks_;
    Tensor sampling_;
    const std::size_t width_;
    const std::size_t capacity_;
    PinnedHostBuffer root_masks_;
    // One fine-grained host allocation: header, then the device-published ids/parents/counts,
    // then the host-published configs, per-row restricted mask words and mask rows.
    void* mailbox_allocation_           = nullptr;
    Mailbox* mailbox_                   = nullptr;
    std::int32_t* mail_ids_             = nullptr;
    std::int32_t* mail_parents_         = nullptr;
    std::int32_t* mail_counts_          = nullptr;
    std::uint32_t* mail_mask_words_     = nullptr;
    ops::SamplingConfig* mail_configs_  = nullptr;
    ops::SamplingConfig* mail_fallback_ = nullptr;
    std::uint32_t* mail_masks_          = nullptr;
    std::uint64_t reply_timeout_ticks_  = 0;
    std::vector<const OutputSession*> outputs_;
    std::vector<ops::SamplingConfig> configs_;
    std::mutex mutex_;
    std::condition_variable armed_changed_;
    bool armed_    = false;
    bool stopping_ = false;
    std::exception_ptr error_;
    std::uint32_t answered_ = 0;
    std::thread matcher_;
};

} // namespace ninfer::targets::qwen3
