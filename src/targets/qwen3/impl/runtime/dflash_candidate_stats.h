#pragma once

// Opt-in DFlash2 miss-ceiling probe. NINFER_DFLASH_CANDIDATE_STATS=1 copies draft
// logits after propose (sync D2H) and classifies hops / the first reject token.
// NINFER_DFLASH_CANDIDATE_STATS_OUT writes exact machine-readable counters at exit.
// This diagnostic deliberately perturbs execution and must not be used for timing evidence.

#include "core/device.h"
#include "core/tensor.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <vector>

namespace ninfer::targets::qwen3::detail {

inline bool dflash_candidate_stats_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_DFLASH_CANDIDATE_STATS");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

namespace dflash_candidate_stats {
namespace {

constexpr int kMaxDrafts = 11;
constexpr int kMaxWidth  = 16;
constexpr int kVocabCap  = 248320;

struct Probe {
    struct RoundTrace {
        std::vector<std::int32_t> proposal_ids;
        std::vector<std::int32_t> parent_index;
        std::vector<std::int32_t> target_licensed_tokens;
    };

    std::mutex mu;
    int rows   = 0;
    int drafts = 0;
    std::vector<std::uint16_t> logits;
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> token_row;
    std::uint64_t hops                = 0;
    std::uint64_t hits                = 0;
    std::uint64_t in_tree             = 0;
    std::uint64_t in_top16            = 0;
    std::uint64_t in_top64            = 0;
    std::uint64_t in_top256           = 0;
    std::uint64_t in_draft_vocab      = 0;
    std::uint64_t missing_draft_vocab = 0;
    std::uint64_t depth_hops[kMaxDrafts]{};
    std::uint64_t depth_hits[kMaxDrafts]{};
    std::uint64_t depth_top16[kMaxDrafts]{};
    std::uint64_t depth_top256[kMaxDrafts]{};
    std::uint64_t rejects             = 0;
    std::uint64_t reject_in_tree      = 0;
    std::uint64_t reject_top16        = 0;
    std::uint64_t reject_top64        = 0;
    std::uint64_t reject_top256       = 0;
    std::uint64_t reject_in_head      = 0;
    std::uint64_t reject_absent_head  = 0;
    std::uint64_t reject_depth[kMaxDrafts]{};
    std::uint64_t logits_elements = 0;
    std::uint64_t logits_nan      = 0;
    std::uint64_t logits_inf      = 0;
    float logits_finite_min       = std::numeric_limits<float>::infinity();
    float logits_finite_max       = -std::numeric_limits<float>::infinity();
    bool printed       = false;
    bool logged_health = false;
    bool wrote_json    = false;
    std::vector<RoundTrace> trace;
};

inline Probe& probe() {
    static Probe p;
    return p;
}

inline float bf16_f32(std::uint16_t bits) {
    const std::uint32_t s = static_cast<std::uint32_t>(bits) << 16;
    float v;
    std::memcpy(&v, &s, sizeof(v));
    return v;
}

inline void print_report(Probe& p) {
    if (p.printed || p.hops == 0) { return; }
    p.printed          = true;
    const auto pct     = [&](std::uint64_t n) {
        return 100.0 * static_cast<double>(n) / static_cast<double>(p.hops);
    };
    const auto rpct = [&](std::uint64_t n) {
        return p.rejects == 0
                   ? 0.0
                   : 100.0 * static_cast<double>(n) / static_cast<double>(p.rejects);
    };
    std::fprintf(stderr,
                 "dflash_candidate_stats hops=%llu hit=%.1f%% in_tree=%.1f%% top16=%.1f%% "
                 "top64=%.1f%% top256=%.1f%% in_draft_head=%.1f%% absent_from_head=%.1f%%\n",
                 static_cast<unsigned long long>(p.hops), pct(p.hits), pct(p.in_tree),
                 pct(p.in_top16), pct(p.in_top64), pct(p.in_top256), pct(p.in_draft_vocab),
                 pct(p.missing_draft_vocab));
    std::fprintf(stderr,
                 "dflash_candidate_stats REJECT n=%llu in_tree=%.1f%% top16=%.1f%% top64=%.1f%% "
                 "top256=%.1f%% in_head=%.1f%% absent_head=%.1f%%\n",
                 static_cast<unsigned long long>(p.rejects), rpct(p.reject_in_tree),
                 rpct(p.reject_top16), rpct(p.reject_top64), rpct(p.reject_top256),
                 rpct(p.reject_in_head), rpct(p.reject_absent_head));
    std::fprintf(stderr, "dflash_candidate_stats by_depth hops/hit/top16/top256:\n");
    for (int d = 0; d < p.drafts && d < kMaxDrafts; ++d) {
        if (p.depth_hops[d] == 0) { continue; }
        const double h = static_cast<double>(p.depth_hops[d]);
        std::fprintf(stderr, "  d%d n=%llu hit=%.1f%% top16=%.1f%% top256=%.1f%% reject_n=%llu\n",
                     d, static_cast<unsigned long long>(p.depth_hops[d]),
                     100.0 * static_cast<double>(p.depth_hits[d]) / h,
                     100.0 * static_cast<double>(p.depth_top16[d]) / h,
                     100.0 * static_cast<double>(p.depth_top256[d]) / h,
                     static_cast<unsigned long long>(p.reject_depth[d]));
    }
}

inline void write_json_report(Probe& p) {
    if (p.wrote_json) { return; }
    p.wrote_json = true;
    const char* path = std::getenv("NINFER_DFLASH_CANDIDATE_STATS_OUT");
    if (path == nullptr || path[0] == '\0') { return; }
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "dflash_candidate_stats failed to open JSON output: %s\n", path);
        return;
    }
    const auto array = [&](const std::uint64_t* values) {
        out << '[';
        for (int i = 0; i < p.drafts && i < kMaxDrafts; ++i) {
            if (i != 0) { out << ','; }
            out << values[i];
        }
        out << ']';
    };
    out << "{\n"
        << "  \"artifact_type\": \"ninfer_dflash_proposal_selector_raw\",\n"
        << "  \"schema_version\": 2,\n"
        << "  \"diagnostic_only\": true,\n"
        << "  \"timing_eligible\": false,\n"
        << "  \"logits\": {\"rows\": " << p.rows << ", \"drafts\": " << p.drafts
        << ", \"elements\": " << p.logits_elements << ", \"nan\": " << p.logits_nan
        << ", \"inf\": " << p.logits_inf << ", \"finite_min\": ";
    if (std::isfinite(p.logits_finite_min)) {
        out << p.logits_finite_min;
    } else {
        out << "null";
    }
    out << ", \"finite_max\": ";
    if (std::isfinite(p.logits_finite_max)) {
        out << p.logits_finite_max;
    } else {
        out << "null";
    }
    out << "},\n"
        << "  \"proposal\": {\"hops\": " << p.hops << ", \"hits\": " << p.hits
        << ", \"in_tree\": " << p.in_tree << ", \"in_top16\": " << p.in_top16
        << ", \"in_top64\": " << p.in_top64 << ", \"in_top256\": " << p.in_top256
        << ", \"in_draft_head\": " << p.in_draft_vocab
        << ", \"absent_from_draft_head\": " << p.missing_draft_vocab << "},\n"
        << "  \"reject\": {\"count\": " << p.rejects
        << ", \"in_tree\": " << p.reject_in_tree << ", \"in_top16\": "
        << p.reject_top16 << ", \"in_top64\": " << p.reject_top64
        << ", \"in_top256\": " << p.reject_top256 << ", \"in_draft_head\": "
        << p.reject_in_head << ", \"absent_from_draft_head\": "
        << p.reject_absent_head << "},\n"
        << "  \"by_depth\": {\"hops\": ";
    array(p.depth_hops);
    out << ", \"hits\": ";
    array(p.depth_hits);
    out << ", \"top16\": ";
    array(p.depth_top16);
    out << ", \"top256\": ";
    array(p.depth_top256);
    out << ", \"rejects\": ";
    array(p.reject_depth);
    out << "},\n  \"trace\": [\n";
    for (std::size_t i = 0; i < p.trace.size(); ++i) {
        const Probe::RoundTrace& round = p.trace[i];
        const auto i32_array = [&](const std::vector<std::int32_t>& values) {
            out << '[';
            for (std::size_t j = 0; j < values.size(); ++j) {
                if (j != 0) { out << ','; }
                out << values[j];
            }
            out << ']';
        };
        out << "    {\"round_index\": " << i << ", \"proposal_ids\": ";
        i32_array(round.proposal_ids);
        out << ", \"parent_index\": ";
        i32_array(round.parent_index);
        out << ", \"target_licensed_tokens\": ";
        i32_array(round.target_licensed_tokens);
        out << '}' << (i + 1 == p.trace.size() ? "" : ",") << '\n';
    }
    out << "  ]\n}\n";
}

struct PrintOnExit {
    ~PrintOnExit() {
        Probe& p = probe();
        print_report(p);
        write_json_report(p);
    }
};

inline PrintOnExit& printer() {
    static PrintOnExit p;
    return p;
}

inline int rank_in_column(const Probe& p, int token, int depth) {
    if (token < 0 || token >= static_cast<int>(p.token_row.size())) { return -1; }
    const int row = p.token_row[static_cast<std::size_t>(token)];
    if (row < 0) { return -1; }
    const float target =
        bf16_f32(p.logits[static_cast<std::size_t>(depth) * static_cast<std::size_t>(p.rows) +
                          static_cast<std::size_t>(row)]);
    int better               = 0;
    const std::uint16_t* col = p.logits.data() +
                               static_cast<std::size_t>(depth) * static_cast<std::size_t>(p.rows);
    for (int r = 0; r < p.rows; ++r) {
        const float v = bf16_f32(col[r]);
        if (v > target || (v == target && r < row)) { ++better; }
    }
    return better;
}

} // namespace

inline void capture_logits(const Tensor& logits, const Tensor* logit_token_ids, int drafts,
                           hipStream_t stream) {
    if (!dflash_candidate_stats_enabled()) { return; }
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    HIP_CHECK(hipStreamIsCapturing(stream, &capture));
    if (capture != hipStreamCaptureStatusNone) { return; }
    (void)printer();
    if (logits.dtype != DType::BF16) { return; }
    Probe& p = probe();
    std::lock_guard<std::mutex> lock(p.mu);
    p.rows   = logits.ne[0];
    p.drafts = drafts;
    const std::size_t n = static_cast<std::size_t>(p.rows) * static_cast<std::size_t>(drafts);
    p.logits.resize(n);
    HIP_CHECK(hipMemcpyAsync(p.logits.data(), logits.data, n * sizeof(std::uint16_t),
                             hipMemcpyDeviceToHost, stream));
    if (logit_token_ids != nullptr && p.token_ids.empty()) {
        p.token_ids.resize(static_cast<std::size_t>(p.rows));
        HIP_CHECK(hipMemcpyAsync(p.token_ids.data(), logit_token_ids->data,
                                 static_cast<std::size_t>(p.rows) * sizeof(std::int32_t),
                                 hipMemcpyDeviceToHost, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    if (!p.logged_health) {
        p.logged_health = true;
        std::uint64_t nans = 0;
        std::uint64_t infs = 0;
        float mn           = std::numeric_limits<float>::infinity();
        float mx           = -std::numeric_limits<float>::infinity();
        for (std::uint16_t bits : p.logits) {
            const float v = bf16_f32(bits);
            if (!std::isfinite(v)) {
                if (std::isnan(v)) {
                    ++nans;
                } else {
                    ++infs;
                }
                continue;
            }
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        p.logits_elements   = p.logits.size();
        p.logits_nan        = nans;
        p.logits_inf        = infs;
        p.logits_finite_min = mn;
        p.logits_finite_max = mx;
        std::fprintf(stderr,
                     "dflash_candidate_stats logits rows=%d drafts=%d n=%zu nan=%llu inf=%llu "
                     "finite_min=%.4g finite_max=%.4g\n",
                     p.rows, p.drafts, p.logits.size(),
                     static_cast<unsigned long long>(nans),
                     static_cast<unsigned long long>(infs), mn, mx);
    }
    if (p.token_row.empty()) {
        p.token_row.assign(kVocabCap, -1);
        if (p.token_ids.empty()) {
            for (int r = 0; r < p.rows && r < kVocabCap; ++r) {
                p.token_row[static_cast<std::size_t>(r)] = r;
            }
        } else {
            for (int r = 0; r < p.rows; ++r) {
                const int tok = p.token_ids[static_cast<std::size_t>(r)];
                if (tok >= 0 && tok < kVocabCap) {
                    p.token_row[static_cast<std::size_t>(tok)] = r;
                }
            }
        }
    }
}

inline void capture_activation(const char* name, const Tensor& tensor, hipStream_t stream) {
    if (!dflash_candidate_stats_enabled() || tensor.data == nullptr || tensor.dtype != DType::BF16) {
        return;
    }
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    HIP_CHECK(hipStreamIsCapturing(stream, &capture));
    if (capture != hipStreamCaptureStatusNone) { return; }
    const std::size_t n = std::min<std::size_t>(tensor.numel(), 65536);
    std::vector<std::uint16_t> bits(n);
    HIP_CHECK(hipMemcpyAsync(bits.data(), tensor.data, n * sizeof(std::uint16_t),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    std::uint64_t nans = 0;
    std::uint64_t infs = 0;
    std::uint64_t zeros = 0;
    float mn            = std::numeric_limits<float>::infinity();
    float mx            = -std::numeric_limits<float>::infinity();
    float abs_sum       = 0.0f;
    for (std::uint16_t word : bits) {
        const float v = bf16_f32(word);
        if (!std::isfinite(v)) {
            if (std::isnan(v)) {
                ++nans;
            } else {
                ++infs;
            }
            continue;
        }
        if (v == 0.0f) { ++zeros; }
        abs_sum += std::fabs(v);
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    std::fprintf(stderr,
                 "dflash_candidate_stats %s numel=%lld sampled=%zu nan=%llu inf=%llu zero=%llu "
                 "mean_abs=%.4g finite_min=%.4g finite_max=%.4g\n",
                 name, static_cast<long long>(tensor.numel()), n,
                 static_cast<unsigned long long>(nans), static_cast<unsigned long long>(infs),
                 static_cast<unsigned long long>(zeros), abs_sum / static_cast<float>(n), mn, mx);
}

inline void record_round(const std::int32_t* verify_ids, const std::int32_t* parents,
                         const std::int32_t* licensed, int licensed_count, int width,
                         int drafts) {
    if (!dflash_candidate_stats_enabled() || licensed_count <= 0) { return; }
    Probe& p = probe();
    std::lock_guard<std::mutex> lock(p.mu);
    if (p.logits.empty() || p.rows <= 0) { return; }
    const int live_w = width < kMaxWidth ? width : kMaxWidth;
    p.trace.push_back(Probe::RoundTrace{
        .proposal_ids = std::vector<std::int32_t>(verify_ids, verify_ids + live_w),
        .parent_index = std::vector<std::int32_t>(parents, parents + live_w),
        .target_licensed_tokens =
            std::vector<std::int32_t>(licensed, licensed + licensed_count),
    });
    int node         = 0;
    for (int hop = 0; hop < licensed_count; ++hop) {
        const int token = licensed[hop];
        const int depth = hop < drafts ? hop : drafts - 1;
        bool hit        = false;
        bool tree       = false;
        for (int c = 1; c < live_w; ++c) {
            if (verify_ids[c] == token) { tree = true; }
            if (parents[c] == node && verify_ids[c] == token) { hit = true; }
        }
        const int rank = rank_in_column(p, token, depth);
        ++p.hops;
        if (depth >= 0 && depth < kMaxDrafts) { ++p.depth_hops[depth]; }
        if (hit) {
            ++p.hits;
            if (depth >= 0 && depth < kMaxDrafts) { ++p.depth_hits[depth]; }
        }
        if (tree) { ++p.in_tree; }
        if (rank >= 0) {
            ++p.in_draft_vocab;
            if (rank < 16) {
                ++p.in_top16;
                if (depth >= 0 && depth < kMaxDrafts) { ++p.depth_top16[depth]; }
            }
            if (rank < 64) { ++p.in_top64; }
            if (rank < 256) {
                ++p.in_top256;
                if (depth >= 0 && depth < kMaxDrafts) { ++p.depth_top256[depth]; }
            }
        } else {
            ++p.missing_draft_vocab;
        }
        if (!hit) {
            ++p.rejects;
            if (tree) { ++p.reject_in_tree; }
            if (rank >= 0) {
                ++p.reject_in_head;
                if (rank < 16) { ++p.reject_top16; }
                if (rank < 64) { ++p.reject_top64; }
                if (rank < 256) { ++p.reject_top256; }
            } else {
                ++p.reject_absent_head;
            }
            if (depth >= 0 && depth < kMaxDrafts) { ++p.reject_depth[depth]; }
            break;
        }
        for (int c = 1; c < live_w; ++c) {
            if (parents[c] == node && verify_ids[c] == token) {
                node = c;
                break;
            }
        }
    }
}

} // namespace dflash_candidate_stats
} // namespace ninfer::targets::qwen3::detail
