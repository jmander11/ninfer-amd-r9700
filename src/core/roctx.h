#pragma once

#include <rocprofiler-sdk-roctx/roctx.h>

#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>

namespace ninfer::roctx {

namespace detail {

inline std::atomic<std::uint32_t> enabled_scopes{0};

} // namespace detail

[[nodiscard]] inline bool ranges_enabled() noexcept {
    return detail::enabled_scopes.load(std::memory_order_relaxed) != 0;
}

// Enabling is reference-counted so independently nested or concurrent owners cannot disable
// ranges while another owner is still active. Ordinary product execution creates no owner.
class ScopedEnable {
public:
    ScopedEnable() noexcept {
        detail::enabled_scopes.fetch_add(1, std::memory_order_relaxed);
    }

    ScopedEnable(const ScopedEnable&)            = delete;
    ScopedEnable& operator=(const ScopedEnable&) = delete;
    ScopedEnable(ScopedEnable&&)                 = delete;
    ScopedEnable& operator=(ScopedEnable&&)      = delete;

    ~ScopedEnable() noexcept {
        detail::enabled_scopes.fetch_sub(1, std::memory_order_relaxed);
    }
};

enum class Category : std::uint32_t {
    Runtime = 1,
    Prefill,
    Decode,
    Mtp,
    DFlash,
    Attention,
    Gdn,
    PostMixer,
    Moe,
    Control,
};

enum class Name : std::size_t {
    Generate,
    Prefill,
    Decode,
    DecodeMtpRound,
    DecodeOrdinaryRound,
    DecodeMtpSubmit,
    DecodeMtpWait,
    DecodeDFlashRound,
    DecodeDFlashSubmit,
    DecodeDFlashWait,
    DecodeOrdinarySubmit,
    DecodeOrdinaryWait,
    PrefillMtpChunk,
    PrefillLayerFull,
    VerifyLayerFull,
    PrefillAttention,
    VerifyAttention,
    PrefillPostMixer,
    VerifyPostMixer,
    PrefillLayerGdn,
    VerifyLayerGdn,
    PrefillGdn,
    VerifyGdn,
    PrefillChunk,
    Count,
};

[[nodiscard]] inline std::uint32_t color(Category category) noexcept {
    switch (category) {
    case Category::Runtime:
        return 0xff4c78a8u;
    case Category::Prefill:
        return 0xff59a14fu;
    case Category::Decode:
        return 0xfff28e2bu;
    case Category::Mtp:
        return 0xffb279a2u;
    case Category::DFlash:
        return 0xffaf7aa1u;
    case Category::Attention:
        return 0xff76b7b2u;
    case Category::Gdn:
        return 0xffe15759u;
    case Category::PostMixer:
        return 0xffedc948u;
    case Category::Moe:
        return 0xffb07aa1u;
    case Category::Control:
        return 0xff9c9c9cu;
    }
    return 0xff9c9c9cu;
}

[[nodiscard]] inline const char* registered_message(Name name) noexcept {
    static constexpr std::array<const char*, static_cast<std::size_t>(Name::Count)> names{
        "generate",
        "prefill",
        "decode",
        "decode.mtp_round",
        "decode.ordinary_round",
        "decode.mtp.submit",
        "decode.mtp.wait",
        "decode.dflash_round",
        "decode.dflash.submit",
        "decode.dflash.wait",
        "decode.ordinary.submit",
        "decode.ordinary.wait",
        "prefill.mtp_chunk",
        "prefill.layer.full",
        "verify.layer.full",
        "prefill.attention",
        "verify.attention",
        "prefill.post_mixer",
        "verify.post_mixer",
        "prefill.layer.gdn",
        "verify.layer.gdn",
        "prefill.gdn",
        "verify.gdn",
        "prefill.chunk",
    };
    const std::size_t index = static_cast<std::size_t>(name);
    return index < names.size() ? names[index] : "unknown";
}

[[nodiscard]] inline const char* category_message(Category category) noexcept {
    switch (category) {
    case Category::Runtime:
        return "runtime";
    case Category::Prefill:
        return "prefill";
    case Category::Decode:
        return "decode";
    case Category::Mtp:
        return "mtp";
    case Category::DFlash:
        return "dflash";
    case Category::Attention:
        return "attention";
    case Category::Gdn:
        return "gdn";
    case Category::PostMixer:
        return "post-mixer";
    case Category::Moe:
        return "moe";
    case Category::Control:
        return "control";
    }
    return "unknown";
}

namespace detail {

// Keep label formatting out of the default-off constructor path. In particular, a ScopedRange in
// an ordinary layer loop must not acquire a 128-byte stack buffer or call formatting/ROCtx code.
[[gnu::noinline]] [[nodiscard]] inline bool push_range(Name name, Category category,
                                                       std::uint64_t payload) noexcept {
    char label[128];
    (void)std::snprintf(label, sizeof(label), "ninfer.%s.%s payload=%llu",
                        category_message(category), registered_message(name),
                        static_cast<unsigned long long>(payload));
    return roctxRangePushA(label) >= 0;
}

} // namespace detail

class ScopedRange {
public:
    explicit ScopedRange(Name name, Category category, std::uint64_t payload = 0) noexcept {
        if (!ranges_enabled()) { return; }
        // ROCtx exposes a string-only nested-range API. Keep the semantic fields visible to
        // rocprofv3 by encoding category and payload into a fixed, allocation-free label.
        active_ = detail::push_range(name, category, payload);
    }

    ScopedRange(const ScopedRange&)            = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
    ScopedRange(ScopedRange&&)                 = delete;
    ScopedRange& operator=(ScopedRange&&)      = delete;

    ~ScopedRange() noexcept {
        if (active_) { (void)roctxRangePop(); }
    }

private:
    bool active_ = false;
};

struct ProfilerControl {
    int (*resume)(roctx_thread_id_t);
    int (*push)(const char*);
    int (*pop)();
    int (*pause)(roctx_thread_id_t);
};

[[nodiscard]] inline ProfilerControl default_profiler_control() noexcept {
    return {roctxProfilerResume, roctxRangePushA, roctxRangePop, roctxProfilerPause};
}

// Owns one process-wide profiler collection interval and its outer thread-local marker. Explicit
// finish reports control failures; exceptional unwinding still balances every successfully
// acquired resource without throwing from the destructor.
class ScopedProfilerRegion {
public:
    explicit ScopedProfilerRegion(const char* label,
                                  ProfilerControl control = default_profiler_control())
        : control_(control) {
        if (control_.resume(0) != 0) {
            throw std::runtime_error("ROCtx profiler resume failed");
        }
        resumed_ = true;
        if (control_.push(label) < 0) {
            const int pause_status = control_.pause(0);
            resumed_               = false;
            if (pause_status != 0) {
                throw std::runtime_error("ROCtx range push and profiler cleanup pause failed");
            }
            throw std::runtime_error("ROCtx range push failed");
        }
        pushed_ = true;
        nested_ranges_.emplace();
    }

    ScopedProfilerRegion(const ScopedProfilerRegion&)            = delete;
    ScopedProfilerRegion& operator=(const ScopedProfilerRegion&) = delete;
    ScopedProfilerRegion(ScopedProfilerRegion&&)                 = delete;
    ScopedProfilerRegion& operator=(ScopedProfilerRegion&&)      = delete;

    ~ScopedProfilerRegion() noexcept { cleanup(); }

    void finish() {
        if (!resumed_ && !pushed_) { return; }
        nested_ranges_.reset();
        const int pop_status = pushed_ ? control_.pop() : 0;
        pushed_              = false;
        const int pause_status = resumed_ ? control_.pause(0) : 0;
        resumed_               = false;
        if (pop_status < 0 && pause_status != 0) {
            throw std::runtime_error("ROCtx range pop and profiler pause failed");
        }
        if (pop_status < 0) { throw std::runtime_error("ROCtx range pop failed"); }
        if (pause_status != 0) { throw std::runtime_error("ROCtx profiler pause failed"); }
    }

private:
    void cleanup() noexcept {
        nested_ranges_.reset();
        if (pushed_) {
            (void)control_.pop();
            pushed_ = false;
        }
        if (resumed_) {
            (void)control_.pause(0);
            resumed_ = false;
        }
    }

    ProfilerControl control_;
    std::optional<ScopedEnable> nested_ranges_;
    bool resumed_ = false;
    bool pushed_  = false;
};

} // namespace ninfer::roctx
