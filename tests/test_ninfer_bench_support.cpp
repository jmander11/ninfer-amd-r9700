#include "ninfer_bench_support.h"
#include "core/roctx.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"
#include "targets/qwen3/impl/runtime/r9700_cache_profile.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace qb = ninfer::bench;

enum class ProfilerEvent { Resume, Push, Pop, Pause };

struct MockProfilerState {
    std::vector<ProfilerEvent> events;
    int resume_status = 0;
    int push_status   = 0;
    int pop_status    = 0;
    int pause_status  = 0;
};

MockProfilerState* mock_profiler = nullptr;

int mock_resume(roctx_thread_id_t) {
    mock_profiler->events.push_back(ProfilerEvent::Resume);
    return mock_profiler->resume_status;
}

int mock_push(const char*) {
    mock_profiler->events.push_back(ProfilerEvent::Push);
    return mock_profiler->push_status;
}

int mock_pop() {
    mock_profiler->events.push_back(ProfilerEvent::Pop);
    return mock_profiler->pop_status;
}

int mock_pause(roctx_thread_id_t) {
    mock_profiler->events.push_back(ProfilerEvent::Pause);
    return mock_profiler->pause_status;
}

ninfer::roctx::ProfilerControl mock_control(MockProfilerState& state) {
    mock_profiler = &state;
    return {mock_resume, mock_push, mock_pop, mock_pause};
}

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

int expect(bool value, std::string_view label) { return value ? 0 : fail(label); }

int expect_u32(std::uint32_t actual, std::uint32_t expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_string(std::string_view actual, std::string_view expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected `" << expected << "`, got `" << actual << "`\n";
    return 1;
}

int expect_near(double actual, double expected, std::string_view label) {
    if (std::abs(actual - expected) < 1e-6) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, std::string_view label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    return fail(std::string(label) + " did not throw");
}

qb::BenchOptions parse_for_test(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return qb::parse_args(static_cast<int>(argv.size()), argv.data());
}

int test_cli_contract() {
    int failures                  = 0;
    const qb::BenchOptions parsed = parse_for_test({
        "ninfer_bench",
        "--weights",
        "model.ninfer",
        "-p",
        "128,512",
        "-n",
        "64",
        "-pg",
        "2048,128",
        "--whole-pg",
        "4096,64",
        "-r",
        "3",
        "--warmup",
        "2",
        "--max-ctx",
        "4096",
        "--kv-capacity",
        "auto",
        "--prefill-chunk",
        "128",
        "--spec",
        "mtp",
        "--draft-tokens",
        "5",
        "--lm-head-draft",
        "--device",
        "1",
        "--no-device-graph",
        "--profile-measured",
        "--retain-token-ids",
        "--isolate-prompt-decode",
        "--output",
        "json",
        "--output-file",
        "report.json",
    });

    failures += expect_string(parsed.artifact_path, "model.ninfer", "artifact path");
    failures += expect(parsed.n_prompt == std::vector<int>({128, 512}), "prompt list");
    failures += expect(parsed.n_gen == std::vector<int>({64}), "generation list");
    failures += expect(parsed.prompt_gen == std::vector<std::pair<int, int>>({{2048, 128}}),
                       "combined list");
    failures += expect(parsed.whole_inference == std::vector<std::pair<int, int>>({{4096, 64}}),
                       "whole-inference list");
    failures += expect(parsed.repetitions == 3 && parsed.warmup == 2, "repetition settings");
    failures += expect(parsed.max_context == std::optional<std::uint32_t>(4096), "max context");
    failures += expect(parsed.automatic_kv_capacity, "automatic KV capacity");
    failures += expect(parsed.prefill_chunk == 128, "prefill chunk");
    failures += expect(parsed.concurrency == 1, "default concurrency");
    failures += expect(parsed.spec_backend == ninfer::SpeculativeBackend::Mtp, "MTP backend");
    failures += expect(parsed.draft_tokens == 5, "MTP window");
    failures +=
        expect(parsed.proposal_head == ninfer::ProposalHead::Optimized, "optimized proposal head");
    failures += expect(parsed.device == 1 && !parsed.use_device_graph, "device and graph settings");
    failures += expect(parsed.profile_measured, "profile-measured flag");
    failures += expect(parsed.retain_token_ids, "retained token IDs");
    failures += expect(parsed.isolate_prompt_decode, "isolated prompt decode");
    failures +=
        expect(parsed.output == qb::OutputFormat::Json && parsed.output_file == "report.json",
               "output settings");

    const qb::BenchOptions unprofiled =
        parse_for_test({"ninfer_bench", "--weights", "model.ninfer"});
    failures += expect(!unprofiled.profile_measured, "profile-measured default");
    failures += expect(unprofiled.prefill_chunk == ninfer::kDefaultPrefillChunk,
                       "benchmark prefill chunk diverges from the product default");

    const auto defaults = qb::expand_tests(qb::BenchOptions{});
    failures +=
        expect(defaults.size() == 2 && defaults[0].label == "pp512" && defaults[1].label == "tg128",
               "default pp/tg matrix");
    failures += expect(qb::usage_text("ninfer_bench").find("artifact.ninfer") != std::string::npos,
                       "help names native artifact");
    failures += expect(qb::usage_text("ninfer_bench").find("--concurrency <1..4>") !=
                           std::string::npos,
                       "help names the product concurrency range");
    failures += expect(parse_for_test({"ninfer_bench", "--help"}).help_requested, "help flag");

    failures += expect_throws<std::invalid_argument>([] { (void)parse_for_test({"ninfer_bench"}); },
                                                     "missing artifact");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--lm-head-draft"});
        },
        "optimized head without MTP");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test(
                {"ninfer_bench", "--weights", "model.ninfer", "--draft-tokens", "6"});
        },
        "unsupported MTP window");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test(
                {"ninfer_bench", "--weights", "model.ninfer", "--prefill-chunk", "129"});
        },
        "misaligned prefill chunk");
    failures += expect_u32(
        parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--concurrency", "2"})
            .concurrency,
        2, "concurrency 2");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test(
                {"ninfer_bench", "--weights", "model.ninfer", "--concurrency", "5"});
        },
        "unsupported concurrency");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test(
                {"ninfer_bench", "--weights", "model.ninfer", "--kv-capacity", "largest"});
        },
        "invalid KV capacity policy");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test(
                {"ninfer_bench", "--weights", "model.ninfer", "--retain-token-ids"});
        },
        "retained token IDs require JSON");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test({"ninfer_bench", "--weights", "model.ninfer",
                                  "--isolate-prompt-decode"});
        },
        "isolated prompt decode requires prompt-gen");
    return failures;
}

int test_roctx_gate() {
    int failures = 0;
    failures += expect(!ninfer::roctx::ranges_enabled(), "ROCTX ranges default off");
    try {
        ninfer::roctx::ScopedEnable outer;
        failures += expect(ninfer::roctx::ranges_enabled(), "ROCTX outer enable");
        {
            ninfer::roctx::ScopedEnable inner;
            failures += expect(ninfer::roctx::ranges_enabled(), "ROCTX nested enable");
        }
        failures += expect(ninfer::roctx::ranges_enabled(), "ROCTX nested restoration");
        bool worker_observed = false;
        std::thread worker([&worker_observed] {
            worker_observed = ninfer::roctx::ranges_enabled();
        });
        worker.join();
        failures += expect(worker_observed, "ROCTX worker visibility");
        throw std::runtime_error("exercise exceptional restoration");
    } catch (const std::runtime_error&) {}
    failures += expect(!ninfer::roctx::ranges_enabled(), "ROCTX exceptional restoration");

    std::atomic<bool> worker_ready{false};
    std::atomic<bool> release_worker{false};
    std::thread owner([&] {
        ninfer::roctx::ScopedEnable enabled;
        worker_ready.store(true, std::memory_order_release);
        while (!release_worker.load(std::memory_order_acquire)) { std::this_thread::yield(); }
    });
    while (!worker_ready.load(std::memory_order_acquire)) { std::this_thread::yield(); }
    {
        ninfer::roctx::ScopedEnable concurrent;
        failures += expect(ninfer::roctx::ranges_enabled(), "ROCTX concurrent owners enabled");
    }
    failures += expect(ninfer::roctx::ranges_enabled(), "ROCTX remaining concurrent owner");
    release_worker.store(true, std::memory_order_release);
    owner.join();
    failures += expect(!ninfer::roctx::ranges_enabled(), "ROCTX concurrent restoration");
    return failures;
}

int test_roctx_profiler_region() {
    int failures = 0;
    {
        MockProfilerState state;
        ninfer::roctx::ScopedProfilerRegion region("measured", mock_control(state));
        failures += expect(ninfer::roctx::ranges_enabled(), "ROCTX profiler enables ranges");
        region.finish();
        failures += expect(state.events == std::vector<ProfilerEvent>({
                               ProfilerEvent::Resume, ProfilerEvent::Push, ProfilerEvent::Pop,
                               ProfilerEvent::Pause}),
                           "ROCTX profiler success order");
        failures += expect(!ninfer::roctx::ranges_enabled(), "ROCTX profiler finish restoration");
    }
    {
        MockProfilerState state;
        state.resume_status = 1;
        failures += expect_throws<std::runtime_error>(
            [&] { ninfer::roctx::ScopedProfilerRegion region("measured", mock_control(state)); },
            "ROCTX resume failure");
        failures += expect(state.events == std::vector<ProfilerEvent>({ProfilerEvent::Resume}),
                           "ROCTX resume failure order");
    }
    {
        MockProfilerState state;
        state.push_status = -1;
        failures += expect_throws<std::runtime_error>(
            [&] { ninfer::roctx::ScopedProfilerRegion region("measured", mock_control(state)); },
            "ROCTX push failure");
        failures += expect(state.events == std::vector<ProfilerEvent>({
                               ProfilerEvent::Resume, ProfilerEvent::Push, ProfilerEvent::Pause}),
                           "ROCTX push failure cleanup without pop");
        failures += expect(!ninfer::roctx::ranges_enabled(), "ROCTX push failure gate remains off");
    }
    {
        MockProfilerState state;
        state.pop_status   = -1;
        state.pause_status = 1;
        ninfer::roctx::ScopedProfilerRegion region("measured", mock_control(state));
        failures += expect_throws<std::runtime_error>([&] { region.finish(); },
                                                      "ROCTX finish failures");
        failures += expect(state.events == std::vector<ProfilerEvent>({
                               ProfilerEvent::Resume, ProfilerEvent::Push, ProfilerEvent::Pop,
                               ProfilerEvent::Pause}),
                           "ROCTX finish attempts complete cleanup");
        failures += expect(!ninfer::roctx::ranges_enabled(), "ROCTX failed finish restoration");
    }
    {
        MockProfilerState state;
        try {
            ninfer::roctx::ScopedProfilerRegion region("measured", mock_control(state));
            throw std::runtime_error("exercise profiler-region unwinding");
        } catch (const std::runtime_error&) {}
        failures += expect(state.events == std::vector<ProfilerEvent>({
                               ProfilerEvent::Resume, ProfilerEvent::Push, ProfilerEvent::Pop,
                               ProfilerEvent::Pause}),
                           "ROCTX profiler exception cleanup order");
        failures += expect(!ninfer::roctx::ranges_enabled(),
                           "ROCTX profiler exception restoration");
    }
    return failures;
}

int test_measurement_contract() {
    int failures = 0;
    const qb::BenchTest pp{qb::TestKind::Prefill, 512, 0, "pp512"};
    const qb::BenchTest tg{qb::TestKind::Decode, 0, 128, "tg128"};
    const qb::BenchTest combined{qb::TestKind::PrefillDecode, 2048, 128, "pp2048+tg128"};
    const qb::BenchTest whole{qb::TestKind::WholeInference, 2048, 128,
                              "whole-pp2048+tg128"};

    failures += expect_u32(pp.requested_output_tokens(), 1, "pp begin output");
    failures += expect_u32(tg.requested_output_tokens(), 129, "tg begin plus G outputs");
    failures +=
        expect_u32(combined.requested_output_tokens(), 129, "combined begin plus G outputs");
    failures += expect_u32(whole.requested_output_tokens(), 129,
                           "whole-inference begin plus G outputs");
    const ninfer::SpeculativeOptions none{};
    const ninfer::SpeculativeOptions mtp5{.backend      = ninfer::SpeculativeBackend::Mtp,
                                         .draft_tokens = 5};
    failures += expect_u32(pp.required_context(none), 512, "pp context");
    failures += expect_u32(pp.required_context(mtp5), 522, "MTP pp context");
    failures += expect_u32(tg.required_context(none), 129, "tg context");
    failures += expect_u32(tg.required_context(mtp5), 139, "MTP tg context");
    failures += expect_u32(combined.required_context(mtp5), 2186, "MTP combined context");
    failures +=
        expect_u32(qb::decode_graph_prime_output_tokens(mtp5), 13, "MTP graph-prime outputs");
    failures +=
        expect_u32(qb::decode_graph_prime_required_context(mtp5), 23, "MTP graph-prime context");
    const ninfer::SpeculativeOptions dflash4{.backend      = ninfer::SpeculativeBackend::DFlash,
                                             .draft_tokens = 4};
    const ninfer::SpeculativeOptions dflash7{.backend      = ninfer::SpeculativeBackend::DFlash,
                                             .draft_tokens = 7};
    failures += expect_u32(qb::resolved_dflash_verify_width(4, 0), 5, "DFlash k4 chain width");
    failures += expect_u32(qb::resolved_dflash_verify_width(7, 0), 12, "DFlash k7 tree width");
    failures += expect_u32(qb::resolved_dflash_verify_width(7, 9), 9, "explicit DFlash width");
    failures += expect_u32(pp.required_context(dflash4), 522, "DFlash k4 pp context");
    failures += expect_u32(pp.required_context(dflash7), 536, "DFlash k7 pp context");

    const std::vector<qb::BenchTest> matrix = {pp, tg, combined};
    failures +=
        expect_u32(qb::resolve_max_context(matrix, std::nullopt, mtp5, true), 2186, "auto context");
    const std::vector<qb::BenchTest> ordinary_isolated = {
        {qb::TestKind::PrefillDecode, 8192, 256, "pp8192+tg256"}};
    failures += expect_u32(
        qb::resolve_max_context(ordinary_isolated, std::nullopt, none, true), 8448,
        "ordinary prompt-decode context");
    failures += expect_u32(
        qb::resolve_max_context(ordinary_isolated, std::nullopt, none, true, true), 8449,
        "isolated prompt-decode seed context");
    failures +=
        expect_u32(qb::resolve_max_context(matrix, std::optional<std::uint32_t>(4096), mtp5, true),
                   4096, "explicit context");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            (void)qb::resolve_max_context(matrix, std::optional<std::uint32_t>(2048), mtp5, true);
        },
        "undersized context");
    failures += expect_u32(qb::concurrent_kv_capacity_tokens(50134, 2), 100352,
                           "C=2 page-aligned Main KV");
    failures += expect_u32(qb::concurrent_kv_capacity_tokens(64, 1), 64, "exact page C=1");
    return failures;
}

ninfer::GenerationTimings timings(double prepare, double prefill, double decode, double total) {
    return {.prepare_seconds = prepare,
            .vision_seconds  = 0.0,
            .prefill_seconds = prefill,
            .decode_seconds  = decode,
            .total_seconds   = total};
}

ninfer::SpeculativeStats speculative(std::uint64_t rounds, std::uint64_t drafted,
                                     std::uint64_t accepted, std::uint64_t fallback,
                                     std::vector<std::uint64_t> per_position) {
    return {.enabled               = true,
            .draft_window          = 5,
            .rounds                = rounds,
            .drafted_tokens        = drafted,
            .accepted_tokens       = accepted,
            .fallback_steps        = fallback,
            .accepted_per_position = std::move(per_position)};
}

std::vector<qb::TestResult> sample_results() {
    qb::TestResult pp;
    pp.test = {qb::TestKind::Prefill, 512, 0, "pp512"};
    pp.reps = {{timings(0.01, 0.5, 0.0, 0.52), speculative(0, 0, 0, 0, {0, 0, 0, 0, 0}), 1},
               {timings(0.02, 0.25, 0.0, 0.28), speculative(0, 0, 0, 0, {0, 0, 0, 0, 0}), 1}};
    pp.reps[0].generated_token_ids_by_lane = {{10}};
    pp.reps[1].generated_token_ids_by_lane = {{10}};
    pp.workspace_peak_bytes           = 5ULL * 1024ULL * 1024ULL * 1024ULL;
    pp.workspace_allocator_peak_bytes = 4ULL * 1024ULL * 1024ULL;

    qb::TestResult tg;
    tg.test = {qb::TestKind::Decode, 0, 3, "tg3"};
    tg.reps = {{timings(0.01, 0.1, 0.5, 0.62), speculative(1, 5, 5, 0, {1, 1, 1, 1, 1}), 4},
               {timings(0.02, 0.1, 1.0, 1.13), speculative(0, 0, 0, 3, {0, 0, 0, 0, 0}), 4}};
    tg.reps[0].generated_token_ids_by_lane = {{10, 11, 12, 13}};
    tg.reps[1].generated_token_ids_by_lane = {{10, 11, 12, 13}};
    tg.workspace_peak_bytes           = 1024ULL * 1024ULL;
    tg.workspace_allocator_peak_bytes = 512ULL * 1024ULL;
    return {std::move(pp), std::move(tg)};
}

qb::BenchEnvironment sample_environment() {
    qb::BenchEnvironment env;
    env.gpu_name                          = "AMD Radeon AI PRO R9700";
    env.architecture_name                 = "gfx1201";
    env.hip_runtime_version               = "7.15.26333";
    env.hip_driver_version                = "7.15.26333";
    env.device_id                         = 0;
    env.artifact_path                     = "model.ninfer";
    env.artifact_file_size_bytes          = 17500000000ULL;
    env.load                              = {.target               = "qwen3_8_27b_r9700",
                                             .weights_id           = "fixture-weights",
                                             .load_seconds         = 2.5,
                                             .upload_seconds       = 2.0,
                                             .artifact_bytes_read  = 17500000000ULL,
                                             .host_to_device_bytes = 17400000000ULL,
                                             .peak_staging_bytes   = 134217728ULL,
                                             .tensor_count         = 1118,
                                             .resource_count       = 6};
    env.memory.device                     = 0;
    env.memory.max_context                = 4096;
    env.memory.kv_capacity                = 8192;
    env.memory.weights                    = {17400000000ULL, 17400000000ULL, 17400000000ULL};
    env.memory.sequence                   = {2000000000ULL, 1900000000ULL, 1900000000ULL};
    env.memory.workspace                  = {100000000ULL, 0, 0};
    env.memory.request_transient          = {50000000ULL, 0, 40000000ULL};
    env.memory.device_graph_allowance_bytes = 150000000ULL;
    env.memory.kv_payload_bytes           = 123456ULL;
    env.max_context                       = 4096;
    env.prefill_chunk                     = 1024;
    env.speculative_backend               = ninfer::SpeculativeBackend::Mtp;
    env.draft_tokens                      = 5;
    env.proposal_head                     = ninfer::ProposalHead::Optimized;
    env.use_device_graph                  = true;
    env.retain_token_ids                  = true;
    env.decode_graph_primed               = true;
    env.decode_graph_prime_output_tokens  = 13;
    env.repetitions                       = 2;
    env.warmup                            = 1;
    env.corpus_path                       = "bench/fixtures/bench_corpus.ids";
    env.corpus_tokens                     = 65536;
    return env;
}

int test_report_contract() {
    int failures                   = 0;
    const qb::BenchEnvironment env = sample_environment();
    const auto results             = sample_results();
    Json report;
    try {
        report = Json::parse(qb::format_json(
            env, "ninfer_bench --weights model.ninfer --spec mtp --draft-tokens 5", results));
    } catch (const nlohmann::json::exception& error) {
        return fail(std::string("invalid benchmark JSON: ") + error.what());
    }

    failures += expect(report.at("schema_version") == 20, "report schema v20");
    failures += expect(report.at("artifact_type") == "ninfer_bench_report", "report identity");
    failures += expect(report.at("artifact").at("path") == "model.ninfer", "artifact path");
    failures += expect(report.at("load").at("target") == "qwen3_8_27b_r9700", "load target");
    failures += expect(report.at("load").at("weights_id") == "fixture-weights", "load weights id");
    failures +=
        expect(report.at("load").at("host_to_device_bytes") == 17400000000ULL, "load H2D bytes");
    failures += expect(report.at("memory").at("kv_cache_format") == "fp8-k-int4-v",
                       "fixed memory KV format");
    failures += expect(report.at("memory").at("kv_capacity") == 8192, "memory KV capacity");
    failures += expect(report.at("memory").at("workspace").at("capacity_bytes") == 100000000ULL,
                       "workspace capacity");
    failures +=
        expect(report.at("memory").at("request_transient").at("capacity_bytes") == 50000000ULL,
               "request transient capacity");
    failures += expect(report.at("memory").at("device_graph_allowance_bytes") == 150000000ULL,
                       "Device Graph allowance");
    failures += expect(report.at("memory").at("kv_payload_bytes") == 123456ULL, "KV payload");
    failures += expect(report.at("config").at("spec") == "mtp", "spec backend");
    failures += expect(report.at("config").at("pending_timeout_ms") == 0xffffffffU,
                       "benchmark pending timeout");
    failures += expect(report.at("config").at("pending_deadline") == "unbounded",
                       "benchmark pending deadline");
    failures += expect(
        report.at("config").at("kv_value_group") ==
            ninfer::targets::qwen3::detail::kR9700TextKVValueGroup,
        "compiled KV value group");
    failures += expect(
        report.at("config").at("kv_plane_layouts").at("key") ==
                ninfer::targets::qwen3::detail::kR9700TextKVKeyPlaneLayoutName &&
            report.at("config").at("kv_plane_layouts").at("value") ==
                ninfer::targets::qwen3::detail::kR9700TextKVValuePlaneLayoutName &&
            report.at("config").at("kv_plane_layouts").at("value_scale") ==
                ninfer::targets::qwen3::detail::kR9700TextKVValueScalePlaneLayoutName,
        "compiled KV plane layouts");
    failures += expect(report.at("config").at("q4_activation_bits") ==
                           ninfer::ops::r9700::linear::kQ4ActivationBits,
                       "compiled Q4 activation width");
    failures += expect(report.at("config").at("q4_prefill_cta_profile") ==
                           ninfer::ops::r9700::linear::kQ4PrefillCtaProfile,
                       "compiled Q4 prefill CTA profile");
    failures += expect(report.at("config").at("dflash_small_t_candidate") ==
                           ninfer::ops::r9700::linear::kDFlashSmallTCandidateEnabled,
                       "compiled DFlash small-token candidate profile");
    failures += expect(report.at("config").at("dflash_mlp_down_t5_candidate") ==
                           ninfer::ops::r9700::linear::kDFlashMlpDownT5CandidateEnabled,
                       "compiled DFlash MLP-down T5 candidate profile");
    failures += expect(report.at("config").at("w8_activation_bits") ==
                           ninfer::ops::r9700::linear::kW8ActivationBits,
                       "compiled W8 activation width");
    failures += expect(report.at("config").at("fp8_qk_wmma_enabled") ==
                           ninfer::ops::r9700::kv::kFp8QkWmmaDecode,
                       "compiled FP8-Q/K WMMA enablement");
    failures += expect(report.at("config").at("fp8_qk_wmma_profile") ==
                           "t1-ge64-t2-ge320-t3plus-stream-v1" &&
                           report.at("config").at("fp8_qk_wmma_t1_min_context") == 64 &&
                           report.at("config").at("fp8_qk_wmma_t2_min_context") == 320,
                       "compiled FP8-Q/K crossover classifier");
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
    failures += expect(report.at("config").at("xattention_qualification") == true &&
                           report.at("config").at("xattention_profile") ==
                               "b128-s16-tau900" &&
                           report.at("config").at("xattention_find_block") == 128 &&
                           report.at("config").at("xattention_stride") == 16 &&
                           report.at("config").at("xattention_tau_permille") == 900,
                       "compile-bound XAttention qualification profile");
#else
    failures += expect(report.at("config").at("xattention_qualification") == false &&
                           !report.at("config").contains("xattention_profile") &&
                           !report.at("config").contains("xattention_find_block") &&
                           !report.at("config").contains("xattention_stride") &&
                           !report.at("config").contains("xattention_tau_permille"),
                       "compile-bound dense attention profile");
#endif
    failures += expect(report.at("config").at("draft_tokens") == 5, "draft tokens");
    failures += expect(report.at("config").at("dflash_verify_width_requested") == 0,
                       "requested DFlash verify width");
    failures += expect(report.at("config").at("dflash_verify_width") == 0,
                       "resolved DFlash verify width");
    failures += expect(report.at("config").at("proposal_head") == "optimized", "proposal head");
    failures += expect(report.at("config").at("concurrency") == 1, "default concurrency in report");
    failures += expect(report.at("config").at("decode_graph_prime").at("output_tokens") == 13,
                       "graph prime output count");

    const Json& pp = report.at("tests").at(0);
    failures +=
        expect(pp.at("kind") == "pp" && pp.at("requested_output_tokens") == 1, "pp request shape");
    failures += expect_near(pp.at("prefill_tok_s_mean").get<double>(), 1536.0, "pp throughput");
    failures += expect(pp.at("decode_output_tok_s_mean").is_null(), "pp decode is null");
    failures += expect(pp.at("workspace_peak_bytes") == 5ULL * 1024ULL * 1024ULL * 1024ULL,
                       "pp workspace peak");
    failures += expect(pp.at("workspace_allocator_peak_bytes") == 4ULL * 1024ULL * 1024ULL,
                       "pp allocator workspace peak");

    const Json& tg = report.at("tests").at(1);
    failures +=
        expect(tg.at("kind") == "tg" && tg.at("requested_output_tokens") == 4, "tg request shape");
    failures += expect_near(tg.at("decode_output_tok_s_mean").get<double>(), 4.5,
                            "decode output throughput");
    failures += expect_near(tg.at("decode_engine_tok_s_mean").get<double>(), 7.5,
                            "decode engine throughput");

    qb::TestResult concurrent               = sample_results()[1];
    concurrent.concurrency                  = 2;
    const qb::Stats concurrent_output       = qb::compute_stats(qb::decode_output_tok_s_series(concurrent));
    const qb::Stats concurrent_engine       = qb::compute_stats(qb::decode_engine_tok_s_series(concurrent));
    failures += expect_near(concurrent_output.mean, 9.0, "C=2 decode output throughput");
    failures += expect_near(concurrent_engine.mean, 7.5, "C=2 decode engine throughput is not doubled");

    failures += expect(tg.at("speculative").at("rounds") == 1, "speculative rounds");
    failures += expect(tg.at("speculative").at("fallback_steps") == 3, "speculative fallbacks");
    failures += expect_near(tg.at("speculative").at("acceptance_rate").get<double>(), 1.0,
                            "speculative acceptance");
    failures += expect(tg.at("speculative").at("accepted_per_position").size() == 5,
                       "per-position acceptance");
    failures +=
        expect(tg.at("reps").at(0).at("generated_output_tokens") == 4, "rep generated tokens");
    failures += expect(tg.at("reps").at(0).at("generated_token_ids_by_lane") ==
                           Json::array({Json::array({10, 11, 12, 13})}),
                       "rep retained token IDs");
    failures += expect(tg.at("reps").at(0).at("decode_engine_tokens") == 6, "rep engine tokens");
    failures += expect_near(tg.at("reps").at(0).at("timings").at("decode_seconds").get<double>(),
                            0.5, "rep GenerationTimings");
    failures += expect(tg.at("reps").at(0).at("speculative").at("drafted_tokens") == 5,
                       "rep SpeculativeStats");
    return failures;
}

int test_human_and_csv_reports() {
    int failures                   = 0;
    const qb::BenchEnvironment env = sample_environment();
    const auto results             = sample_results();
    const std::string table        = qb::format_table(env, results);
    failures += expect(table.find("qwen3_8_27b_r9700") != std::string::npos, "table target");
    failures += expect(table.find("fixture-weights") != std::string::npos, "table weights id");
    failures += expect(table.find("model.ninfer") != std::string::npos, "table artifact");
    failures +=
        expect(table.find("proposal_head=optimized") != std::string::npos, "table proposal head");
    failures += expect(
        table.find(std::string("q4_prefill_cta_profile=") +
                       std::string(ninfer::ops::r9700::linear::kQ4PrefillCtaProfile)) !=
            std::string::npos,
        "table Q4 prefill CTA profile");
    failures += expect(
        table.find(std::string("dflash_small_t_candidate=") +
                       (ninfer::ops::r9700::linear::kDFlashSmallTCandidateEnabled ? "true" :
                                                                                  "false")) !=
            std::string::npos,
        "table DFlash small-token candidate profile");
    failures += expect(
        table.find(std::string("dflash_mlp_down_t5_candidate=") +
                       (ninfer::ops::r9700::linear::kDFlashMlpDownT5CandidateEnabled ? "true" :
                                                                                     "false")) !=
            std::string::npos,
        "table DFlash MLP-down T5 candidate profile");
    failures +=
        expect(table.find("decode eng t/s") != std::string::npos, "table engine throughput");
    failures += expect(table.find("work peak") != std::string::npos, "table workspace peak");

    const std::string csv = qb::format_csv(env, results);
    failures += expect(csv.starts_with("label,kind,n_prompt,n_gen,target,weights_id"),
                       "CSV identity columns");
    failures += expect(
        csv.find("prefill_chunk,concurrency,spec,draft_tokens,dflash_verify_width_requested,dflash_verify_width,proposal_head") !=
            std::string::npos,
        "CSV concurrency and speculative columns");
    const std::size_t header_end = csv.find('\n');
    const std::size_t row_end    = csv.find('\n', header_end + 1);
    failures += expect(
        header_end != std::string::npos && row_end != std::string::npos &&
            std::count(csv.begin(), csv.begin() + static_cast<std::ptrdiff_t>(header_end), ',') ==
                std::count(csv.begin() + static_cast<std::ptrdiff_t>(header_end + 1),
                           csv.begin() + static_cast<std::ptrdiff_t>(row_end), ','),
        "CSV header and row column counts agree");
    for (const std::string_view field :
         {"proposal_head", "kv_value_group", "kv_key_plane_layout",
          "kv_value_plane_layout", "kv_value_scale_plane_layout", "q4_activation_bits",
          "q4_prefill_cta_profile", "dflash_small_t_candidate",
          "dflash_mlp_down_t5_candidate", "w8_activation_bits",
          "fp8_qk_wmma_enabled", "fp8_qk_wmma_profile",
          "fp8_qk_wmma_t1_min_context", "fp8_qk_wmma_t2_min_context", "kv_payload_bytes",
          "load_host_to_device_bytes", "request_transient_capacity_bytes",
          "device_graph_allowance_bytes", "workspace_peak_bytes",
          "workspace_allocator_peak_bytes", "spec_acceptance_rate",
          "decode_output_tok_s_mean", "decode_engine_tok_s_mean", "total_seconds_mean"}) {
        failures += expect(csv.find(field) != std::string::npos,
                           std::string("CSV field ") + std::string(field));
    }
    failures += expect(std::count(csv.begin(), csv.end(), '\n') == 3, "CSV header plus two rows");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_cli_contract();
    failures += test_roctx_gate();
    failures += test_roctx_profiler_region();
    failures += test_measurement_contract();
    failures += test_report_contract();
    failures += test_human_and_csv_reports();
    return failures == 0 ? 0 : fail("ninfer_bench support contract failed");
}
