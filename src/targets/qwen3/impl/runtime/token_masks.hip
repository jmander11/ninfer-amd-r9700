#include "targets/qwen3/impl/runtime/token_masks.h"
#include "core/device.h"
#include "core/roctx.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace ninfer::targets::qwen3 {

// Header of the fine-grained host mailbox. The device writes request/tree/batch/timed_out, the
// matcher thread writes reply; both sides use system-scope atomics for the two sequence words.
struct TokenMaskExchange::Mailbox {
    std::uint32_t request;
    std::uint32_t reply;
    std::uint32_t timed_out;
    std::uint32_t tree;
    std::uint32_t batch;
};

namespace {
constexpr std::size_t kWords            = (kTokenDomain + 31) / 32;
constexpr std::uint32_t kPublishThreads = 256;
constexpr std::uint32_t kAcquireBlocks  = 16;
constexpr std::uint32_t kAcquireThreads = 256;
constexpr auto kReplyTimeout            = std::chrono::seconds(2);
constexpr auto kMatcherPoll             = std::chrono::microseconds(20);

std::size_t align_up(std::size_t value) { return (value + 255U) & ~std::size_t{255U}; }

// Copies the verification ids, optional parents and node counts into the mailbox at its fixed
// row pitch, then raises the request number once every store is visible to the host.
__global__ __launch_bounds__(kPublishThreads) void token_mask_publish_kernel(
    const std::int32_t* ids, std::size_t ids_pitch, const std::int32_t* parents,
    std::size_t parents_pitch, const std::int32_t* valid_columns, std::uint32_t batch,
    std::uint32_t width, std::uint32_t mailbox_width, TokenMaskExchange::Mailbox* mailbox,
    std::int32_t* mail_ids, std::int32_t* mail_parents, std::int32_t* mail_counts) {
    for (std::uint32_t i = threadIdx.x; i < batch * width; i += blockDim.x) {
        const std::uint32_t row = i / width, node = i % width;
        mail_ids[row * mailbox_width + node] = ids[row * ids_pitch + node];
        if (parents != nullptr)
            mail_parents[row * mailbox_width + node] = parents[row * parents_pitch + node];
    }
    for (std::uint32_t row = threadIdx.x; row < batch; row += blockDim.x)
        mail_counts[row] = valid_columns[row];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0U) {
        mailbox->tree  = parents != nullptr ? 1U : 0U;
        mailbox->batch = batch;
        const std::uint32_t request =
            __hip_atomic_load(&mailbox->request, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&mailbox->request, request + 1U, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

// Waits for the matcher's reply to the published request, then copies the reply configs and the
// restricted mask rows into device storage; a restricted row's columns past its node count are
// all-permitted, as the unrestricted sampler would see them. Without a reply in time the bound
// unmasked configs are copied, and every CTA that timed out records it for finish_round().
__global__ __launch_bounds__(kAcquireThreads) void token_mask_acquire_kernel(
    TokenMaskExchange::Mailbox* mailbox, const ops::SamplingConfig* reply_configs,
    const ops::SamplingConfig* fallback_configs, const std::uint32_t* mask_words,
    const std::uint32_t* host_masks, ops::SamplingConfig* device_configs,
    std::uint32_t* device_masks, std::size_t row_words, std::uint64_t timeout_ticks) {
    __shared__ std::uint32_t replied;
    if (threadIdx.x == 0U) {
        const std::uint32_t request =
            __hip_atomic_load(&mailbox->request, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        const std::uint64_t start = wall_clock64();
        std::uint32_t ok          = 1U;
        while (__hip_atomic_load(&mailbox->reply, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) !=
               request) {
            if (wall_clock64() - start > timeout_ticks) {
                ok = 0U;
                break;
            }
            __builtin_amdgcn_s_sleep(32);
        }
        replied = ok;
    }
    __syncthreads();
    const std::uint32_t batch =
        __hip_atomic_load(&mailbox->batch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    constexpr std::size_t config_words = sizeof(ops::SamplingConfig) / sizeof(std::uint32_t);
    static_assert(sizeof(ops::SamplingConfig) % sizeof(std::uint32_t) == 0);
    if (blockIdx.x == 0U) {
        const auto* source = reinterpret_cast<const std::uint32_t*>(
            replied != 0U ? reply_configs : fallback_configs);
        auto* destination = reinterpret_cast<std::uint32_t*>(device_configs);
        for (std::size_t i = threadIdx.x; i < batch * config_words; i += blockDim.x)
            destination[i] = source[i];
    }
    if (replied == 0U) {
        if (threadIdx.x == 0U)
            __hip_atomic_store(&mailbox->timed_out, 1U, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_SYSTEM);
        return;
    }
    const std::size_t thread  = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t threads = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::uint32_t row = 0; row < batch; ++row) {
        const std::size_t words = mask_words[row];
        if (words == 0U) continue;
        const std::size_t base = row * row_words;
        for (std::size_t i = thread; i < words; i += threads)
            device_masks[base + i] = host_masks[base + i];
        for (std::size_t i = words + thread; i < row_words; i += threads)
            device_masks[base + i] = ~std::uint32_t{0};
    }
}

// Whether every node row permits the complete token domain, which the sampler expresses with a
// null mask and no copy.
bool all_permitted(const std::uint32_t* words, std::size_t nodes) {
    constexpr std::size_t full = kTokenDomain / 32;
    constexpr std::uint32_t tail =
        kTokenDomain % 32 == 0 ? 0U : (std::uint32_t{1} << (kTokenDomain % 32)) - 1U;
    for (std::size_t node = 0; node < nodes; ++node) {
        const std::uint32_t* row = words + node * kWords;
        for (std::size_t i = 0; i < full; ++i)
            if (row[i] != ~std::uint32_t{0}) return false;
        if (tail != 0U && (row[full] & tail) != tail) return false;
    }
    return true;
}
} // namespace

TokenMaskExchange::TokenMaskExchange(Tensor masks, Tensor sampling)
    : masks_(masks), sampling_(sampling), width_(masks.ne[1]), capacity_(masks.ne[2]),
      root_masks_(capacity_ * kWords * sizeof(std::uint32_t)) {
    if (masks.dtype != DType::I32 || masks.ne[0] != kWords || !masks.is_contiguous() ||
        width_ == 0 || capacity_ == 0 ||
        sampling.bytes() < capacity_ * sizeof(ops::SamplingConfig)) {
        throw std::invalid_argument("invalid token mask exchange storage");
    }
    outputs_.resize(capacity_, nullptr);
    configs_.resize(capacity_);

    const std::size_t nodes          = width_ * capacity_;
    const std::size_t ids_offset     = align_up(sizeof(Mailbox));
    const std::size_t parents_offset = ids_offset + align_up(nodes * sizeof(std::int32_t));
    const std::size_t counts_offset  = parents_offset + align_up(nodes * sizeof(std::int32_t));
    const std::size_t words_offset   = counts_offset + align_up(capacity_ * sizeof(std::int32_t));
    const std::size_t configs_offset = words_offset + align_up(capacity_ * sizeof(std::uint32_t));
    const std::size_t fallback_offset =
        configs_offset + align_up(capacity_ * sizeof(ops::SamplingConfig));
    const std::size_t masks_offset =
        fallback_offset + align_up(capacity_ * sizeof(ops::SamplingConfig));
    const std::size_t bytes = masks_offset + nodes * kWords * sizeof(std::uint32_t);
    HIP_CHECK(hipHostMalloc(&mailbox_allocation_, bytes, hipHostMallocCoherent));
    std::memset(mailbox_allocation_, 0, bytes);
    auto* base       = static_cast<std::byte*>(mailbox_allocation_);
    mailbox_         = reinterpret_cast<Mailbox*>(base);
    mail_ids_        = reinterpret_cast<std::int32_t*>(base + ids_offset);
    mail_parents_    = reinterpret_cast<std::int32_t*>(base + parents_offset);
    mail_counts_     = reinterpret_cast<std::int32_t*>(base + counts_offset);
    mail_mask_words_ = reinterpret_cast<std::uint32_t*>(base + words_offset);
    mail_configs_    = reinterpret_cast<ops::SamplingConfig*>(base + configs_offset);
    mail_fallback_   = reinterpret_cast<ops::SamplingConfig*>(base + fallback_offset);
    mail_masks_      = reinterpret_cast<std::uint32_t*>(base + masks_offset);

    int device         = 0;
    int wall_clock_khz = 0;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, device));
    reply_timeout_ticks_ =
        static_cast<std::uint64_t>(wall_clock_khz) *
        std::chrono::duration_cast<std::chrono::milliseconds>(kReplyTimeout).count();
    matcher_ = std::thread([this] { serve(); });
}

TokenMaskExchange::~TokenMaskExchange() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    armed_changed_.notify_all();
    matcher_.join();
    (void)hipHostFree(mailbox_allocation_);
}

void TokenMaskExchange::bind(std::span<const OutputSession* const> outputs,
                             std::span<const ops::SamplingConfig> sampling) {
    if (outputs.empty() || outputs.size() > capacity_ || outputs.size() != sampling.size()) {
        throw std::invalid_argument("invalid token mask batch bindings");
    }
    std::lock_guard lock(mutex_);
    outputs_.assign(outputs.begin(), outputs.end());
    configs_.assign(sampling.begin(), sampling.end());
    error_ = {};
}

std::uint32_t* TokenMaskExchange::host_mask(std::size_t row) const {
    return mail_masks_ + row * width_ * kWords;
}

std::uint32_t* TokenMaskExchange::root_mask(std::size_t row) const {
    return static_cast<std::uint32_t*>(root_masks_.data()) + row * kWords;
}

const std::uint32_t* TokenMaskExchange::device_mask(std::size_t row) const {
    return static_cast<const std::uint32_t*>(masks_.data) + row * width_ * kWords;
}

ops::SamplingConfig TokenMaskExchange::root(std::size_t row, hipStream_t stream) {
    return root_into(outputs_.at(row), configs_.at(row), row, stream);
}

ops::SamplingConfig TokenMaskExchange::prefill_root(const OutputSession* output,
                                                    ops::SamplingConfig config,
                                                    hipStream_t stream) {
    return root_into(output, config, capacity_ - 1U, stream);
}

ops::SamplingConfig TokenMaskExchange::root_into(const OutputSession* output,
                                                 ops::SamplingConfig config, std::size_t row,
                                                 hipStream_t stream) {
    config.allowed_token_words         = nullptr;
    config.allowed_token_column_stride = 0;
    if (output && output->has_token_grammar()) {
        const std::array<TokenId, 1> token{0};
        const std::array<std::int32_t, 1> parent{-1};
        output->fill_token_masks(token, parent, {root_mask(row), kWords});
        HIP_CHECK(hipMemcpyAsync(const_cast<std::uint32_t*>(device_mask(row)), root_mask(row),
                                 kWords * sizeof(std::uint32_t), hipMemcpyHostToDevice, stream));
        config.allowed_token_words = device_mask(row);
    }
    return config;
}

const ops::SamplingConfig* TokenMaskExchange::publish(const Tensor& ids, const Tensor* parents,
                                                      const Tensor& valid_columns,
                                                      hipStream_t stream) {
    const auto batch = static_cast<std::size_t>(ids.ne[1]);
    const auto width = static_cast<std::size_t>(ids.ne[0]);
    if (width > width_ || batch > capacity_ || ids.dtype != DType::I32 ||
        valid_columns.dtype != DType::I32 || valid_columns.ne[0] != ids.ne[1] ||
        (parents != nullptr && (parents->dtype != DType::I32 || parents->ne[0] != ids.ne[0] ||
                                parents->ne[1] != ids.ne[1]))) {
        throw std::invalid_argument("invalid speculative token mask inputs");
    }
    hipLaunchKernelGGL(token_mask_publish_kernel, dim3(1), dim3(kPublishThreads), 0, stream,
                       static_cast<const std::int32_t*>(ids.data), ids.nb[1] / sizeof(std::int32_t),
                       parents ? static_cast<const std::int32_t*>(parents->data) : nullptr,
                       parents ? parents->nb[1] / sizeof(std::int32_t) : std::size_t{0},
                       static_cast<const std::int32_t*>(valid_columns.data),
                       static_cast<std::uint32_t>(batch), static_cast<std::uint32_t>(width),
                       static_cast<std::uint32_t>(width_), mailbox_, mail_ids_, mail_parents_,
                       mail_counts_);
    HIP_CHECK(hipGetLastError());
    return static_cast<const ops::SamplingConfig*>(sampling_.data);
}

void TokenMaskExchange::acquire(hipStream_t stream) {
    hipLaunchKernelGGL(token_mask_acquire_kernel, dim3(kAcquireBlocks), dim3(kAcquireThreads), 0,
                       stream, mailbox_, mail_configs_, mail_fallback_, mail_mask_words_,
                       mail_masks_, static_cast<ops::SamplingConfig*>(sampling_.data),
                       static_cast<std::uint32_t*>(masks_.data), width_ * kWords,
                       reply_timeout_ticks_);
    HIP_CHECK(hipGetLastError());
}

void TokenMaskExchange::arm() {
    {
        std::lock_guard lock(mutex_);
        for (std::size_t row = 0; row < capacity_; ++row) {
            mail_fallback_[row] = row < configs_.size() ? configs_[row] : ops::SamplingConfig{};
            mail_fallback_[row].allowed_token_words         = nullptr;
            mail_fallback_[row].allowed_token_column_stride = 0;
        }
        std::atomic_ref(mailbox_->timed_out).store(0U, std::memory_order_relaxed);
        armed_ = true;
    }
    armed_changed_.notify_all();
}

void TokenMaskExchange::disarm() noexcept {
    std::lock_guard lock(mutex_);
    armed_ = false;
}

void TokenMaskExchange::finish_round() {
    std::lock_guard lock(mutex_);
    armed_ = false;
    // The round synchronized, so its last request is final: one it timed out on is never
    // answered later against another round's bindings.
    answered_ = std::atomic_ref(mailbox_->request).load(std::memory_order_acquire);
    if (std::atomic_ref(mailbox_->timed_out).exchange(0U, std::memory_order_relaxed) != 0U) {
        throw std::runtime_error("token grammar mask exchange received no host reply");
    }
    if (error_) { std::rethrow_exception(std::exchange(error_, nullptr)); }
}

// Matcher thread. The mutex is held while answering, so bind/arm/finish at round boundaries
// never overlap a reply; it is released while waiting between polls.
void TokenMaskExchange::serve() noexcept {
    std::unique_lock lock(mutex_);
    for (;;) {
        armed_changed_.wait(lock, [this] { return stopping_ || armed_; });
        if (stopping_) return;
        const std::uint32_t request =
            std::atomic_ref(mailbox_->request).load(std::memory_order_acquire);
        if (request != answered_) {
            answer(request);
            answered_ = request;
            continue;
        }
        armed_changed_.wait_for(lock, kMatcherPoll, [this] { return stopping_ || !armed_; });
    }
}

void TokenMaskExchange::answer(std::uint32_t request) noexcept {
    roctx::ScopedRange range(roctx::Name::TokenMaskMatch, roctx::Category::Control,
                             configs_.size());
    try {
        fill(std::atomic_ref(mailbox_->tree).load(std::memory_order_relaxed) != 0U);
    } catch (...) {
        error_ = std::current_exception();
        // Finish the round safely with nonempty domains, then fail before any candidate can
        // be published.
        for (std::size_t row = 0; row < capacity_; ++row) {
            mail_configs_[row]    = mail_fallback_[row];
            mail_mask_words_[row] = 0U;
        }
    }
    std::atomic_ref(mailbox_->reply).store(request, std::memory_order_release);
}

void TokenMaskExchange::fill(bool tree) {
    // Rows past the bindings (representative warm rounds) carry no grammar.
    const std::size_t batch = std::min<std::size_t>(
        std::atomic_ref(mailbox_->batch).load(std::memory_order_relaxed), capacity_);
    for (std::size_t row = 0; row < batch; ++row) {
        auto& config               = mail_configs_[row];
        config                     = row < configs_.size() ? configs_[row] : ops::SamplingConfig{};
        config.allowed_token_words = nullptr;
        config.allowed_token_column_stride = 0;
        mail_mask_words_[row]              = 0U;
        if (row >= outputs_.size() || !outputs_[row] || !outputs_[row]->has_token_grammar()) {
            continue;
        }
        const std::int32_t count_value = mail_counts_[row];
        if (count_value < 1 || static_cast<std::size_t>(count_value) > width_) {
            throw std::logic_error("invalid token grammar verification node count");
        }
        const auto count      = static_cast<std::size_t>(count_value);
        std::int32_t* parents = mail_parents_ + row * width_;
        if (!tree) {
            for (std::size_t node = 0; node < count; ++node) {
                parents[node] = static_cast<std::int32_t>(node) - 1;
            }
        }
        std::uint32_t* words = host_mask(row);
        std::fill_n(words, count * kWords, ~std::uint32_t{0});
        outputs_[row]->fill_token_masks({mail_ids_ + row * width_, count}, {parents, count},
                                        {words, count * kWords});
        if (all_permitted(words, count)) { continue; }
        // Nodes in one grammar state (free text) share a mask: publish it once, broadcast.
        bool shared = true;
        for (std::size_t node = 1; node < count && shared; ++node) {
            shared = std::equal(words, words + kWords, words + node * kWords);
        }
        config.allowed_token_words         = device_mask(row);
        config.allowed_token_column_stride = shared ? 0 : static_cast<std::int32_t>(kWords);
        mail_mask_words_[row] = static_cast<std::uint32_t>((shared ? 1U : count) * kWords);
    }
}

} // namespace ninfer::targets::qwen3
