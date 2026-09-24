#pragma once

// Opt-in, diagnostic-only snapshot of the final Text prefill boundary at absolute frontier 129.
// NINFER_QWEN3_PREFILL_P129_TRACE=1 enables one snapshot per process and
// NINFER_QWEN3_PREFILL_P129_TRACE_OUT names a new JSON file. The synchronous D2H copies make
// traced runs inadmissible as timing evidence. Ordinary builds take only the disabled getenv
// branch and retain no trace state.

#include "core/tensor.h"
#include "targets/qwen3/impl/runtime/prefill_tail_trace_path.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::prefill_tail_trace {
namespace {

inline float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

struct RankedLogit {
    std::int32_t token = -1;
    std::uint16_t bits = 0;
    float value        = -std::numeric_limits<float>::infinity();
};

inline bool precedes(float value, std::int32_t token, const RankedLogit& incumbent) {
    return value > incumbent.value ||
           (value == incumbent.value && (incumbent.token < 0 || token < incumbent.token));
}

inline void capture_if_enabled(std::uint32_t base, std::uint32_t tokens,
                               const Tensor& tail_hidden, const Tensor& logits,
                               std::int32_t token_domain, hipStream_t stream) {
    const char* enabled = std::getenv("NINFER_QWEN3_PREFILL_P129_TRACE");
    if (enabled == nullptr) { return; }
    if (enabled[0] != '1' || enabled[1] != '\0') {
        throw std::invalid_argument("NINFER_QWEN3_PREFILL_P129_TRACE must be exactly 1");
    }
    if (base + tokens != 129U) { return; }
    if (!((base == 0U && tokens == 129U) || (base == 128U && tokens == 1U))) {
        throw std::logic_error("P129 prefill trace requires exact fresh or one-token append shape");
    }
    if (tokens == 0U || tail_hidden.dtype != DType::BF16 ||
        tail_hidden.ne[0] != 5120 || tail_hidden.ne[1] != 1 ||
        tail_hidden.ne[2] != 1 || tail_hidden.ne[3] != 1 ||
        !tail_hidden.is_contiguous() || tail_hidden.data == nullptr ||
        logits.dtype != DType::BF16 || logits.ne[0] < token_domain || logits.ne[1] != 1 ||
        logits.ne[2] != 1 || logits.ne[3] != 1 || !logits.is_contiguous() ||
        logits.data == nullptr || token_domain <= 1) {
        throw std::logic_error("P129 prefill trace received invalid hidden/logit tensors");
    }
    const char* output_path = std::getenv("NINFER_QWEN3_PREFILL_P129_TRACE_OUT");
    if (output_path == nullptr || output_path[0] == '\0') {
        throw std::invalid_argument("P129 prefill trace requires a nonempty output path");
    }
    ::ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(output_path);

    std::vector<std::uint16_t> hidden(5120U);
    std::vector<std::uint16_t> host_logits(static_cast<std::size_t>(token_domain));
    hipError_t status = hipMemcpyAsync(hidden.data(), tail_hidden.data,
                                       hidden.size() * sizeof(std::uint16_t),
                                       hipMemcpyDeviceToHost, stream);
    if (status == hipSuccess) {
        status = hipMemcpyAsync(host_logits.data(), logits.data,
                                host_logits.size() * sizeof(std::uint16_t),
                                hipMemcpyDeviceToHost, stream);
    }
    if (status == hipSuccess) { status = hipStreamSynchronize(stream); }
    if (status != hipSuccess) {
        throw std::runtime_error(std::string("P129 prefill trace D2H failed: ") +
                                 hipGetErrorString(status));
    }

    RankedLogit first, second;
    for (std::int32_t token = 0; token < token_domain; ++token) {
        const std::uint16_t bits = host_logits[static_cast<std::size_t>(token)];
        const float value        = bf16_to_float(bits);
        if (!std::isfinite(value)) {
            throw std::runtime_error("P129 prefill trace found a nonfinite target logit");
        }
        if (precedes(value, token, first)) {
            second = first;
            first  = RankedLogit{token, bits, value};
        } else if (precedes(value, token, second)) {
            second = RankedLogit{token, bits, value};
        }
    }
    if (first.token < 0 || second.token < 0) {
        throw std::logic_error("P129 prefill trace could not rank two target logits");
    }

    std::ofstream output(output_path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!output) { throw std::runtime_error("P129 prefill trace could not create output"); }
    output << "{\n"
           << "  \"artifact_type\": \"ninfer_qwen3_text_prefill_p129_tail_trace\",\n"
           << "  \"schema_version\": 1,\n"
           << "  \"diagnostic_only\": true,\n"
           << "  \"timing_evidence_eligible\": false,\n"
           << "  \"absolute_frontier\": 129,\n"
           << "  \"prefill_base\": " << base << ",\n"
           << "  \"prefill_tokens\": " << tokens << ",\n"
           << "  \"tail_hidden_kind\": \"final_rmsnorm_bf16\",\n"
           << "  \"tail_hidden_bf16_bits\": [";
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        if (index != 0U) { output << ','; }
        output << hidden[index];
    }
    output << "],\n"
           << "  \"target_logits_kind\": \"full_lm_head_bf16\",\n"
           << "  \"token_domain\": " << token_domain << ",\n"
           << "  \"top1\": {\"token\": " << first.token << ", \"bf16_bits\": "
           << first.bits << ", \"value\": " << std::setprecision(9) << first.value << "},\n"
           << "  \"top2\": {\"token\": " << second.token << ", \"bf16_bits\": "
           << second.bits << ", \"value\": " << std::setprecision(9) << second.value << "},\n"
           << "  \"top1_top2_margin\": " << std::setprecision(9)
           << first.value - second.value << "\n"
           << "}\n";
    output.close();
    if (!output) { throw std::runtime_error("P129 prefill trace output write failed"); }
}

} // namespace
} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::prefill_tail_trace
