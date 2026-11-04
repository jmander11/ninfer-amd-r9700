#pragma once

// Opt-in eager-only target-decision trace. Setting
// NINFER_DFLASH_DECISION_TRACE_OUT to an explicit path records the represented BF16
// top-two decision at each ordinary decode step and every live DFlash target-verify
// column. The synchronous device copies deliberately perturb execution, so this trace
// is diagnostic-only and rejects Device Graph execution. NINFER_SAMPLING_CAPTURE_PREFIX
// additionally saves complete sampling inputs for watched licensed rounds.

#include "core/device.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3/round_state.h>

#include <hip/hip_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <charconv>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3::detail::decision_trace {

struct RankedBf16 {
    std::int32_t top1_token = -1;
    std::int32_t top2_token = -1;
    std::uint16_t top1_bits = 0;
    std::uint16_t top2_bits = 0;
    float top1_logit        = 0.0F;
    float top2_logit        = 0.0F;
    float margin            = 0.0F;
};

inline float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

inline RankedBf16 rank_bf16(std::span<const std::uint16_t> logits, std::int32_t valid_rows) {
    if (valid_rows < 2 || logits.size() < static_cast<std::size_t>(valid_rows)) {
        throw std::invalid_argument("decision trace requires at least two valid BF16 logits");
    }
    const auto better = [](float value, std::int32_t token, float current,
                           std::int32_t current_token) {
        return value > current || (value == current && token < current_token);
    };
    float first               = -std::numeric_limits<float>::infinity();
    float second              = -std::numeric_limits<float>::infinity();
    std::int32_t first_token  = std::numeric_limits<std::int32_t>::max();
    std::int32_t second_token = std::numeric_limits<std::int32_t>::max();
    std::uint16_t first_bits  = 0;
    std::uint16_t second_bits = 0;
    for (std::int32_t token = 0; token < valid_rows; ++token) {
        const std::uint16_t bits = logits[static_cast<std::size_t>(token)];
        const float value        = bf16_to_float(bits);
        if (!std::isfinite(value)) {
            throw std::runtime_error("decision trace encountered a nonfinite valid logit");
        }
        if (better(value, token, first, first_token)) {
            second       = first;
            second_token = first_token;
            second_bits  = first_bits;
            first        = value;
            first_token  = token;
            first_bits   = bits;
        } else if (token != first_token && better(value, token, second, second_token)) {
            second       = value;
            second_token = token;
            second_bits  = bits;
        }
    }
    if (first_token == std::numeric_limits<std::int32_t>::max() ||
        second_token == std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("decision trace could not resolve two valid logits");
    }
    return {.top1_token = first_token,
            .top2_token = second_token,
            .top1_bits  = first_bits,
            .top2_bits  = second_bits,
            .top1_logit = first,
            .top2_logit = second,
            .margin     = first - second};
}

inline const std::string& output_path() {
    static const std::string path = [] {
        const char* value = std::getenv("NINFER_DFLASH_DECISION_TRACE_OUT");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    return path;
}

inline const std::string& capture_prefix() {
    static const std::string prefix = [] {
        const char* value = std::getenv("NINFER_SAMPLING_CAPTURE_PREFIX");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    return prefix;
}

inline std::vector<std::int32_t> parse_watch_tokens(std::string_view value) {
    std::vector<std::int32_t> result;
    while (!value.empty()) {
        const auto comma   = value.find(',');
        const auto part    = value.substr(0, comma);
        std::int32_t token = -1;
        const auto parsed  = std::from_chars(part.data(), part.data() + part.size(), token);
        if (parsed.ec != std::errc{} || parsed.ptr != part.data() + part.size() || token < 0)
            throw std::invalid_argument(
                "NINFER_SAMPLING_CAPTURE_TOKENS requires nonnegative comma-separated IDs");
        result.push_back(token);
        if (comma == std::string_view::npos) break;
        value.remove_prefix(comma + 1);
        if (value.empty()) throw std::invalid_argument("empty sampling capture token ID");
    }
    if (result.empty()) throw std::invalid_argument("empty sampling capture watch list");
    return result;
}

inline const std::vector<std::int32_t>& watch_tokens() {
    static const auto tokens = [] {
        const char* value = std::getenv("NINFER_SAMPLING_CAPTURE_TOKENS");
        return parse_watch_tokens(value == nullptr ? "248046,248044" : value);
    }();
    return tokens;
}

inline bool watched(std::span<const std::int32_t> tokens) {
    if (capture_prefix().empty()) return false;
    return std::any_of(tokens.begin(), tokens.end(), [](std::int32_t token) {
        return std::find(watch_tokens().begin(), watch_tokens().end(), token) !=
               watch_tokens().end();
    });
}

inline bool enabled() { return !output_path().empty() || !capture_prefix().empty(); }

inline void write_capture_bytes(const std::filesystem::path& path, const void* data,
                                std::size_t bytes) {
    const auto status = std::filesystem::symlink_status(path);
    if (std::filesystem::exists(status) || std::filesystem::is_symlink(status))
        throw std::runtime_error("sampling capture refuses to overwrite " + path.string());
    std::ofstream file;
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.open(path, std::ios::binary);
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

// The caller has synchronized the actual sampling operation and has not released token masks.
// Only watched licensed rounds reach here. Binary rows contain the logical vocabulary only.
inline void capture_sampling(nlohmann::json metadata, const ops::SamplingConfig& cfg,
                             std::span<const std::uint16_t> logits, std::size_t physical_rows,
                             std::int32_t columns, std::int32_t token_domain) {
    static_assert(std::endian::native == std::endian::little);
    static std::mutex mutex;
    static std::uint64_t ordinal = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (columns <= 0 || token_domain <= 0 ||
        physical_rows < static_cast<std::size_t>(token_domain) ||
        logits.size() < physical_rows * static_cast<std::size_t>(columns))
        throw std::logic_error("invalid sampling capture logits");
    const std::string stem                    = capture_prefix() + "-" + std::to_string(ordinal++);
    const std::filesystem::path manifest_path = stem + ".json";
    const std::filesystem::path logits_path   = stem + ".bf16";
    std::vector<std::uint16_t> compact;
    compact.reserve(static_cast<std::size_t>(columns) * static_cast<std::size_t>(token_domain));
    for (std::int32_t col = 0; col < columns; ++col) {
        const auto begin = logits.begin() + static_cast<std::ptrdiff_t>(
                                                static_cast<std::size_t>(col) * physical_rows);
        compact.insert(compact.end(), begin, begin + token_domain);
    }
    write_capture_bytes(logits_path, compact.data(), compact.size() * sizeof(compact[0]));
    metadata["artifact_type"]             = "ninfer_sampling_decision_capture";
    metadata["schema_version"]            = 1;
    metadata["diagnostic_only"]           = true;
    metadata["execution"]                 = "eager";
    metadata["publication"]               = "op_licensed_before_output_session";
    metadata["token_domain"]              = token_domain;
    metadata["columns"]                   = columns;
    metadata["source_physical_rows"]      = physical_rows;
    metadata["source_column_pitch_bytes"] = physical_rows * sizeof(std::uint16_t);
    metadata["logits"]                    = {{"file", logits_path.filename().string()},
                                             {"dtype", "bf16_le"},
                                             {"shape", {columns, token_domain}}};
    metadata["config"] = {{"temperature", cfg.temperature},
                          {"p_less", cfg.p_less},
                          {"seed", cfg.seed},
                          {"typical_exclude", cfg.typical_exclude},
                          {"draft_temperature", cfg.draft_temperature},
                          {"top_k", cfg.top_k},
                          {"top_p", cfg.top_p},
                          {"min_p", cfg.min_p},
                          {"presence_penalty", cfg.presence_penalty},
                          {"frequency_penalty", cfg.frequency_penalty},
                          {"token_counts_present", cfg.token_counts != nullptr},
                          {"allowed_token_column_stride", cfg.allowed_token_column_stride}};
    auto& suppressed = metadata["config"]["suppressed_tokens"] = nlohmann::json::array();
    if (cfg.suppressed_token_count < 0 ||
        cfg.suppressed_token_count > ops::SamplingConfig::kMaximumSuppressedTokens)
        throw std::logic_error("invalid sampling capture suppression count");
    for (std::int32_t i = 0; i < cfg.suppressed_token_count; ++i)
        suppressed.push_back(cfg.suppressed_tokens[i]);
    metadata["allowed_token_words"] = nullptr;
    if (cfg.allowed_token_words != nullptr) {
        const std::size_t words = (static_cast<std::size_t>(token_domain) + 31U) / 32U;
        std::vector<std::uint32_t> masks(words * static_cast<std::size_t>(columns));
        for (std::int32_t col = 0; col < columns; ++col)
            HIP_CHECK(hipMemcpy(masks.data() + static_cast<std::size_t>(col) * words,
                                cfg.allowed_token_words + static_cast<std::int64_t>(col) *
                                                              cfg.allowed_token_column_stride,
                                words * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
        const std::filesystem::path masks_path = stem + ".mask.u32";
        write_capture_bytes(masks_path, masks.data(), masks.size() * sizeof(masks[0]));
        metadata["allowed_token_words"] = {{"file", masks_path.filename().string()},
                                           {"dtype", "u32_le"},
                                           {"shape", {columns, words}}};
    }
    metadata["watch_tokens"]  = watch_tokens();
    const std::string encoded = metadata.dump(2) + "\n";
    write_capture_bytes(manifest_path, encoded.data(), encoded.size());
}

inline std::string ranked_json(const RankedBf16& ranked) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "{\"top1_token\":" << ranked.top1_token << ",\"top1_bf16_bits\":" << ranked.top1_bits
        << ",\"top1_logit\":" << ranked.top1_logit << ",\"top2_token\":" << ranked.top2_token
        << ",\"top2_bf16_bits\":" << ranked.top2_bits << ",\"top2_logit\":" << ranked.top2_logit
        << ",\"margin\":" << ranked.margin << '}';
    return out.str();
}

inline std::int64_t absolute_frontier(std::int32_t input_cache_position) {
    return static_cast<std::int64_t>(input_cache_position) + 1;
}

inline std::int32_t resolved_accepted_column(bool tree_verify, std::int32_t accepted_drafts,
                                             std::int32_t tree_accepted_column) {
    return tree_verify ? tree_accepted_column : accepted_drafts;
}

class Writer {
public:
    explicit Writer(const std::string& path) {
        const std::filesystem::file_status status = std::filesystem::symlink_status(path);
        if (std::filesystem::exists(status) || std::filesystem::is_symlink(status)) {
            throw std::runtime_error("target decision trace refuses to overwrite its output");
        }
        output_.exceptions(std::ios::badbit | std::ios::failbit);
        output_.open(path, std::ios::out);
        output_ << "{\n  \"artifact_type\": \"ninfer_dflash_target_decision_trace\",\n"
                   "  \"schema_version\": 2,\n"
                   "  \"diagnostic_only\": true,\n"
                   "  \"timing_eligible\": false,\n"
                   "  \"production_routing_authorized\": false,\n"
                   "  \"events\": [\n";
        output_.flush();
    }

    Writer(const Writer&)            = delete;
    Writer& operator=(const Writer&) = delete;

    ~Writer() {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            output_ << "\n  ]\n}\n";
            output_.flush();
            // NOLINTNEXTLINE(bugprone-empty-catch): a diagnostic trace must not throw at teardown
        } catch (...) {}
    }

    void append(std::string_view event) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!first_) { output_ << ",\n"; }
        output_ << "    " << event;
        output_.flush();
        first_ = false;
    }

private:
    std::mutex mutex_;
    std::ofstream output_;
    bool first_ = true;
};

inline Writer& writer() {
    static Writer instance(output_path());
    return instance;
}

inline void require_eager(bool use_device_graph) {
    if (!enabled()) { return; }
    if (use_device_graph) {
        throw std::invalid_argument(
            "sampling/target decision diagnostics require eager execution (--no-device-graph)");
    }
    if (!output_path().empty()) (void)writer();
    if (!capture_prefix().empty()) (void)watch_tokens();
}

inline void require_bf16_logits(const Tensor& logits, std::int32_t columns,
                                std::int32_t token_domain) {
    if (logits.dtype != DType::BF16 || logits.data == nullptr || !logits.is_contiguous() ||
        token_domain < 2 || logits.ne[0] < token_domain || columns <= 0 ||
        logits.numel() < static_cast<std::int64_t>(logits.ne[0]) * columns) {
        throw std::logic_error("target decision trace received invalid BF16 logits");
    }
}

inline std::vector<std::uint16_t> copy_logits(const Tensor& logits, std::int32_t columns) {
    const std::size_t elements =
        static_cast<std::size_t>(logits.ne[0]) * static_cast<std::size_t>(columns);
    std::vector<std::uint16_t> host(elements);
    HIP_CHECK(hipMemcpy(host.data(), logits.data, elements * sizeof(std::uint16_t),
                        hipMemcpyDeviceToHost));
    return host;
}

inline void record_ordinary(const Tensor& logits, const qwen3::OrdinaryDecodeIngress& ingress,
                            const qwen3::OrdinaryDecodeEgress& egress, std::int32_t batch_size,
                            std::int32_t token_domain) {
    if (!enabled()) { return; }
    require_bf16_logits(logits, batch_size, token_domain);
    const std::vector<std::uint16_t> host = copy_logits(logits, batch_size);
    const std::size_t stride              = static_cast<std::size_t>(logits.ne[0]);
    for (std::int32_t row = 0; row < batch_size; ++row) {
        const RankedBf16 ranked =
            rank_bf16(std::span<const std::uint16_t>(
                          host.data() + static_cast<std::size_t>(row) * stride, stride),
                      token_domain);
        const std::int32_t input_position = ingress.cache_positions[static_cast<std::size_t>(row)];
        const auto sampled                = egress.sampled_tokens[static_cast<std::size_t>(row)];
        if (watched(std::span<const std::int32_t>(&sampled, 1))) {
            nlohmann::json capture = {{"kind", "ordinary"},
                                      {"lane", ingress.lanes[static_cast<std::size_t>(row)]},
                                      {"logical_position", input_position},
                                      {"rng_purpose", ops::kSamplePurposeDecode},
                                      {"licensed_tokens", {sampled}}};
            capture_sampling(std::move(capture), ingress.sampling[static_cast<std::size_t>(row)],
                             std::span<const std::uint16_t>(
                                 host.data() + static_cast<std::size_t>(row) * stride, stride),
                             stride, 1, token_domain);
        }
        std::ostringstream event;
        event << "{\"kind\":\"ordinary\",\"logits_stage\":\"pre_sample\",\"lane\":"
              << ingress.lanes[static_cast<std::size_t>(row)]
              << ",\"token_domain\":" << token_domain
              << ",\"input_cache_position\":" << input_position
              << ",\"absolute_frontier\":" << absolute_frontier(input_position)
              << ",\"sampled_token\":" << egress.sampled_tokens[static_cast<std::size_t>(row)]
              << ",\"sampled_matches_top1\":"
              << (egress.sampled_tokens[static_cast<std::size_t>(row)] == ranked.top1_token
                      ? "true"
                      : "false")
              << ",\"next_frontier\":" << absolute_frontier(input_position)
              << ",\"decision\":" << ranked_json(ranked) << '}';
        if (!output_path().empty()) writer().append(event.str());
    }
}

inline void record_dflash(const Tensor& logits, const Tensor& target_argmax,
                          const Tensor& cache_positions, const Tensor& verify_ids,
                          const qwen3::DFlashDecodeIngress& ingress,
                          const qwen3::DFlashDecodeEgress& egress,
                          std::span<const std::uint32_t> live_columns, std::int32_t batch_size,
                          std::int32_t verify_width, std::int32_t physical_verify_width,
                          std::int32_t token_domain, bool tree_verify,
                          const ops::SamplingConfig* target_sampling, const Tensor& selector_ids,
                          const Tensor& selector_q) {
    if (!enabled()) { return; }
    if (!capture_prefix().empty() && tree_verify)
        throw std::invalid_argument("sampling decision capture supports chain verification only");
    if (batch_size <= 0 || verify_width <= 0 || physical_verify_width < 2 ||
        physical_verify_width > verify_width ||
        live_columns.size() != static_cast<std::size_t>(batch_size) ||
        target_argmax.dtype != DType::I32 || cache_positions.dtype != DType::I32 ||
        verify_ids.dtype != DType::I32 || target_argmax.data == nullptr ||
        cache_positions.data == nullptr || verify_ids.data == nullptr ||
        !target_argmax.is_contiguous() || !cache_positions.is_contiguous() ||
        !verify_ids.is_contiguous()) {
        throw std::logic_error("target decision trace received invalid DFlash metadata");
    }
    const std::int32_t total_columns = batch_size * verify_width;
    if (target_argmax.numel() < total_columns || cache_positions.numel() < total_columns ||
        verify_ids.numel() < total_columns) {
        throw std::logic_error("target decision trace received short DFlash metadata");
    }
    require_bf16_logits(logits, total_columns, token_domain);
    const std::vector<std::uint16_t> host_logits = copy_logits(logits, total_columns);
    std::vector<std::int32_t> host_argmax(static_cast<std::size_t>(total_columns));
    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(total_columns));
    std::vector<std::int32_t> host_verify_ids(static_cast<std::size_t>(total_columns));
    HIP_CHECK(hipMemcpy(host_argmax.data(), target_argmax.data,
                        host_argmax.size() * sizeof(std::int32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_positions.data(), cache_positions.data,
                        host_positions.size() * sizeof(std::int32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_verify_ids.data(), verify_ids.data,
                        host_verify_ids.size() * sizeof(std::int32_t), hipMemcpyDeviceToHost));
    const std::size_t stride = static_cast<std::size_t>(logits.ne[0]);
    for (std::int32_t row = 0; row < batch_size; ++row) {
        const std::uint32_t count         = live_columns[static_cast<std::size_t>(row)];
        const std::int32_t licensed_count = egress.licensed_counts[static_cast<std::size_t>(row)];
        if (count == 0 || count > static_cast<std::uint32_t>(verify_width) || licensed_count <= 0 ||
            licensed_count > verify_width) {
            throw std::logic_error("target decision trace received invalid DFlash extents");
        }
        const std::int32_t base     = ingress.execution_frontiers[static_cast<std::size_t>(row)];
        const std::int32_t accepted = egress.accepted_drafts[static_cast<std::size_t>(row)];
        const std::int32_t accepted_column = resolved_accepted_column(
            tree_verify, accepted, egress.accepted_column[static_cast<std::size_t>(row)]);
        const std::span<const std::int32_t> licensed(egress.licensed_tokens.data() +
                                                         static_cast<std::size_t>(row) *
                                                             static_cast<std::size_t>(verify_width),
                                                     static_cast<std::size_t>(licensed_count));
        if (watched(licensed)) {
            if (target_sampling == nullptr)
                throw std::logic_error("missing actual target sampling config");
            ops::SamplingConfig cfg;
            HIP_CHECK(hipMemcpy(&cfg, target_sampling + row, sizeof(cfg), hipMemcpyDeviceToHost));
            const auto extent = ingress.proposal_extents[static_cast<std::size_t>(row)];
            if (extent < 0 || count != static_cast<std::uint32_t>(extent + 1))
                throw std::logic_error("sampling capture chain extent mismatch");
            nlohmann::json capture = {
                {"kind", "dflash_chain"},
                {"lane", ingress.lanes[static_cast<std::size_t>(row)]},
                {"base_frontier", base},
                {"proposal_extent", extent},
                {"accepted_drafts", accepted},
                {"licensed_tokens", std::vector<std::int32_t>(licensed.begin(), licensed.end())},
                {"drafts", nlohmann::json::array()},
                {"selector_ids", nlohmann::json::array()},
                {"selector_q", nlohmann::json::array()},
                {"cache_positions", nlohmann::json::array()}};
            const auto flat =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(verify_width);
            for (std::uint32_t col = 0; col < count; ++col) {
                capture["cache_positions"].push_back(host_positions[flat + col]);
                if (col > 0) capture["drafts"].push_back(host_verify_ids[flat + col]);
            }
            if (selector_ids.dtype != DType::I32 || selector_q.dtype != DType::FP32 ||
                selector_ids.ne[0] != selector_q.ne[0] || selector_ids.ne[1] < extent ||
                selector_q.ne[1] < extent || selector_ids.ne[2] <= row || selector_q.ne[2] <= row ||
                selector_ids.data == nullptr || selector_q.data == nullptr)
                throw std::logic_error("invalid sampling capture selector tensors");
            capture["selector_source_strides_bytes"] = {{"ids", selector_ids.nb},
                                                        {"q", selector_q.nb}};
            for (std::int32_t hop = 0; hop < extent; ++hop) {
                std::vector<std::int32_t> ids(static_cast<std::size_t>(selector_ids.ne[0]));
                std::vector<float> probabilities(ids.size());
                const auto* id_source = static_cast<const unsigned char*>(selector_ids.data) +
                                        static_cast<std::size_t>(row) * selector_ids.nb[2] +
                                        static_cast<std::size_t>(hop) * selector_ids.nb[1];
                const auto* q_source  = static_cast<const unsigned char*>(selector_q.data) +
                                        static_cast<std::size_t>(row) * selector_q.nb[2] +
                                        static_cast<std::size_t>(hop) * selector_q.nb[1];
                HIP_CHECK(hipMemcpy(ids.data(), id_source, ids.size() * sizeof(ids[0]),
                                    hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(probabilities.data(), q_source,
                                    probabilities.size() * sizeof(probabilities[0]),
                                    hipMemcpyDeviceToHost));
                capture["selector_ids"].push_back(ids);
                capture["selector_q"].push_back(probabilities);
            }
            capture_sampling(
                std::move(capture), cfg,
                std::span<const std::uint16_t>(host_logits.data() + flat * stride,
                                               static_cast<std::size_t>(count) * stride),
                stride, static_cast<std::int32_t>(count), token_domain);
        }
        std::ostringstream event;
        event << "{\"kind\":\"dflash\",\"lane\":" << ingress.lanes[static_cast<std::size_t>(row)]
              << ",\"token_domain\":" << token_domain << ",\"verify_width\":" << verify_width
              << ",\"physical_verify_width\":" << physical_verify_width
              << ",\"tree_verify\":" << (tree_verify ? "true" : "false")
              << ",\"anchor\":" << ingress.anchors[static_cast<std::size_t>(row)]
              << ",\"base_frontier\":" << base << ",\"context_frontier\":"
              << ingress.context_frontiers[static_cast<std::size_t>(row)]
              << ",\"proposal_extent\":" << ingress.proposal_extents[static_cast<std::size_t>(row)]
              << ",\"accepted_drafts\":" << accepted << ",\"accepted_column\":" << accepted_column
              << ",\"licensed_tokens\":[";
        for (std::int32_t i = 0; i < licensed_count; ++i) {
            if (i != 0) { event << ','; }
            event << egress.licensed_tokens[static_cast<std::size_t>(row) *
                                                static_cast<std::size_t>(verify_width) +
                                            static_cast<std::size_t>(i)];
        }
        event << "],\"next_frontier\":" << (static_cast<std::int64_t>(base) + licensed_count)
              << ",\"columns\":[";
        for (std::uint32_t column = 0; column < count; ++column) {
            if (column != 0) { event << ','; }
            const std::size_t flat =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(verify_width) + column;
            const RankedBf16 ranked = rank_bf16(
                std::span<const std::uint16_t>(host_logits.data() + flat * stride, stride),
                token_domain);
            event << "{\"column\":" << column << ",\"logits_stage\":\"pre_accept_target_verify\""
                  << ",\"verify_token\":" << host_verify_ids[flat]
                  << ",\"input_cache_position\":" << host_positions[flat]
                  << ",\"absolute_frontier\":" << absolute_frontier(host_positions[flat])
                  << ",\"target_argmax\":" << host_argmax[flat]
                  << ",\"target_argmax_matches_top1\":"
                  << (host_argmax[flat] == ranked.top1_token ? "true" : "false")
                  << ",\"decision\":" << ranked_json(ranked) << '}';
        }
        event << "]}";
        if (!output_path().empty()) writer().append(event.str());
    }
}

} // namespace ninfer::targets::qwen3::detail::decision_trace
