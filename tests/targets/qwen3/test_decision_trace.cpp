#include "targets/qwen3/impl/runtime/decision_trace.h"
#include <ninfer/targets/qwen3/frontend.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include <unistd.h>

namespace trace = ninfer::targets::qwen3::detail::decision_trace;

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

template <typename Fn>
void expect_throw(Fn&& fn, std::string_view message) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    expect(false, message);
}

std::uint16_t bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16U);
}

void test_top_two_uses_argmax_tie_order() {
    const std::array<std::uint16_t, 5> logits{
        bf16(1.0F), bf16(3.0F), bf16(3.0F), bf16(2.0F), bf16(-8.0F)};
    const trace::RankedBf16 ranked = trace::rank_bf16(logits, 4);
    expect(ranked.top1_token == 1 && ranked.top2_token == 2,
           "equal maxima select the lowest token first and next-lowest second");
    expect(ranked.top1_bits == bf16(3.0F) && ranked.top2_bits == bf16(3.0F) &&
               ranked.top1_logit == 3.0F && ranked.top2_logit == 3.0F && ranked.margin == 0.0F,
           "trace retains exact represented BF16 top-two values and margin");
    expect(trace::ranked_json(ranked) ==
               "{\"top1_token\":1,\"top1_bf16_bits\":16448,\"top1_logit\":3,"
               "\"top2_token\":2,\"top2_bf16_bits\":16448,\"top2_logit\":3,"
               "\"margin\":0}",
           "ranked decision JSON has the stable machine-readable schema");
}

void test_valid_domain_and_nonfinite_guards() {
    const std::array<std::uint16_t, 4> logits{
        bf16(-1.0F), bf16(2.0F), bf16(9.0F), bf16(8.0F)};
    const trace::RankedBf16 ranked = trace::rank_bf16(logits, 2);
    expect(ranked.top1_token == 1 && ranked.top2_token == 0,
           "physical rows outside the token domain do not participate");
    expect_throw([&] { (void)trace::rank_bf16(logits, 1); },
                 "fewer than two valid logits are rejected");

    auto nonfinite = logits;
    nonfinite[0]   = bf16(std::numeric_limits<float>::infinity());
    expect_throw([&] { (void)trace::rank_bf16(nonfinite, 2); },
                 "nonfinite valid logits are rejected instead of emitting invalid JSON");
}

void test_frontier_and_acceptance_contract() {
    expect(ninfer::targets::qwen3::kTokenDomain == 248077,
           "decision trace ranks the tokenizer domain rather than padded output rows");
    expect(trace::absolute_frontier(128) == 129,
           "decision frontier follows the consumed input cache position");
    expect(trace::resolved_accepted_column(false, 4, 99) == 4,
           "chain verification derives its continuation column from accepted drafts");
    expect(trace::resolved_accepted_column(true, 4, 7) == 7,
           "tree verification retains its explicit accepted column");
}

void test_machine_readable_envelope() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("ninfer-decision-trace-contract-" + std::to_string(static_cast<long long>(getpid())) +
         ".json");
    std::filesystem::remove(path);
    {
        trace::Writer writer(path.string());
        writer.append("{\"kind\":\"ordinary\",\"absolute_frontier\":129}");
    }
    std::ifstream input(path);
    const std::string contents((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
    expect(contents.find("\"artifact_type\": \"ninfer_dflash_target_decision_trace\"") !=
                   std::string::npos &&
               contents.find("\"diagnostic_only\": true") != std::string::npos &&
               contents.find("\"timing_eligible\": false") != std::string::npos &&
               contents.find("\"production_routing_authorized\": false") !=
                   std::string::npos &&
               contents.find("\"events\": [") != std::string::npos &&
               contents.ends_with("\n  ]\n}\n"),
           "trace writer emits a closed diagnostic-only JSON envelope");
    expect_throw([&] { trace::Writer duplicate(path.string()); },
                 "trace writer refuses an existing regular output");
    const std::filesystem::path link = path.string() + ".link";
    std::filesystem::remove(link);
    std::filesystem::create_symlink(path, link);
    expect_throw([&] { trace::Writer symlink(link.string()); },
                 "trace writer refuses an existing output symlink");
    std::filesystem::remove(link);
    std::filesystem::remove(path);
}

void test_graph_execution_is_rejected_when_enabled() {
    if (setenv("NINFER_DFLASH_DECISION_TRACE_OUT", "/tmp/ninfer-unused-decision-trace.json", 1) !=
        0) {
        throw std::runtime_error("setenv failed");
    }
    expect(trace::enabled(), "an explicit nonempty output path enables the decision trace");
    expect_throw([&] { trace::require_eager(true); },
                 "enabled decision tracing rejects Device Graph execution");
}

} // namespace

int main() {
    test_top_two_uses_argmax_tie_order();
    test_valid_domain_and_nonfinite_guards();
    test_frontier_and_acceptance_contract();
    test_machine_readable_envelope();
    test_graph_execution_is_rejected_when_enabled();
    if (failures != 0) {
        std::cerr << failures << " target decision trace checks failed\n";
        return 1;
    }
    std::cout << "Target decision trace checks passed\n";
    return 0;
}
