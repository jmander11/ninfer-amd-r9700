#include "ninfer_bench_support.h"

#include "core/roctx.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string command_line(int argc, char** argv) {
    std::ostringstream out;
    for (int i = 0; i < argc; ++i) {
        if (i != 0) { out << ' '; }
        out << argv[i];
    }
    return out.str();
}

std::string hip_version_string(int version) {
    if (version <= 0) { return {}; }
    if (version >= 10'000'000) {
        return std::to_string(version / 10'000'000) + "." +
               std::to_string((version / 100'000) % 100) + "." +
               std::to_string(version % 100'000);
    }
    return std::to_string(version);
}

void fill_hip_environment(ninfer::bench::BenchEnvironment& env, int device) {
    env.device_id       = device;
    int runtime_version = 0;
    if (hipRuntimeGetVersion(&runtime_version) == hipSuccess) {
        env.hip_runtime_version = hip_version_string(runtime_version);
    }
    int driver_version = 0;
    if (hipDriverGetVersion(&driver_version) == hipSuccess) {
        env.hip_driver_version = hip_version_string(driver_version);
    }
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, device) == hipSuccess) {
        env.gpu_name          = properties.name;
        env.architecture_name = properties.gcnArchName;
    }
}

void require_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

class ProfileMeasuredRegion {
public:
    explicit ProfileMeasuredRegion(bool enabled) {
        if (!enabled) { return; }
        require_hip(hipDeviceSynchronize(), "profile pre-boundary synchronize");
        region_.emplace("ninfer_bench_measured");
        active_ = true;
    }

    ProfileMeasuredRegion(const ProfileMeasuredRegion&)            = delete;
    ProfileMeasuredRegion& operator=(const ProfileMeasuredRegion&) = delete;

    ~ProfileMeasuredRegion() noexcept {
        if (!active_) { return; }
        (void)hipDeviceSynchronize();
        close();
    }

    void finish() {
        if (!active_) { return; }
        const hipError_t status = hipDeviceSynchronize();
        active_                = false;
        region_->finish();
        require_hip(status, "profile post-boundary synchronize");
    }

private:
    void close() noexcept {
        region_.reset();
        active_ = false;
    }

    std::optional<ninfer::roctx::ScopedProfilerRegion> region_;
    bool active_ = false;
};

bool has_decode_tests(const std::vector<ninfer::bench::BenchTest>& tests) {
    for (const auto& test : tests) {
        if (test.has_decode()) { return true; }
    }
    return false;
}

constexpr auto kBenchmarkPendingDeadline = std::chrono::steady_clock::time_point::max();

ninfer::RequestOptions benchmark_request(const ninfer::bench::BenchTest& test, bool prefix_reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = test.requested_output_tokens();
    options.execution.allow_prefix_reuse      = prefix_reuse;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    options.output.preserve_special_tokens    = true;
    return options;
}

ninfer::GenerationResult consume_generation(ninfer::GenerationResult generated,
                                            const ninfer::bench::BenchTest& test,
                                            std::uint32_t expected, const char* phase) {
    if (generated.generated_token_ids.size() != expected) {
        throw std::runtime_error(test.label + std::string(phase) + " generated " +
                                 std::to_string(generated.generated_token_ids.size()) +
                                 " tokens; expected " + std::to_string(expected));
    }
    if (generated.finish_reason != ninfer::FinishReason::OutputLimit) {
        throw std::runtime_error(test.label + std::string(phase) +
                                 " did not finish at the requested output limit");
    }
    return generated;
}

ninfer::bench::RepTiming run_repetition(ninfer::Engine& engine,
                                        const ninfer::bench::BenchTest& test,
                                        const std::vector<ninfer::TokenId>& corpus,
                                        std::uint32_t concurrency,
                                        bool isolate_prompt_decode = false) {
    const int prompt_tokens = test.kind == ninfer::bench::TestKind::Decode
                                  ? ninfer::bench::kDecodeSeedTokens
                                  : test.n_prompt;
    const std::uint32_t expected = test.requested_output_tokens();
    const bool isolate_batched_decode =
        (concurrency > 1 || isolate_prompt_decode) &&
        test.kind == ninfer::bench::TestKind::PrefillDecode;

    if (concurrency <= 1 && !isolate_batched_decode) {
        const ninfer::RequestOptions request = benchmark_request(test, false);
        auto prompt = engine.prepare_tokens(ninfer::bench::prompt_slice(corpus, prompt_tokens),
                                            false);
        const auto wave_start = std::chrono::steady_clock::now();
        ninfer::GenerationResult result = engine.generate(std::move(prompt), request);
        const auto wave_end             = std::chrono::steady_clock::now();
        std::vector<ninfer::GenerationResult> generated;
        generated.push_back(consume_generation(std::move(result), test, expected, ""));
        ninfer::bench::RepTiming timing = ninfer::bench::fold_lane_results(generated, expected);
        timing.wave_seconds = std::chrono::duration<double>(wave_end - wave_start).count();
        return timing;
    }

    if (isolate_batched_decode) {
        ninfer::RequestOptions seed = benchmark_request(test, true);
        seed.execution.requested_output_tokens = 1;
        std::vector<std::vector<ninfer::TokenId>> histories;
        histories.reserve(concurrency);
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            std::vector<ninfer::TokenId> tokens =
                ninfer::bench::prompt_slice(corpus, prompt_tokens, lane);
            ninfer::GenerationResult seeded =
                consume_generation(engine.generate(engine.prepare_tokens(tokens, true), seed), test,
                                   1, " seed");
            tokens.insert(tokens.end(), seeded.generated_token_ids.begin(),
                          seeded.generated_token_ids.end());
            histories.push_back(std::move(tokens));
        }

        const ninfer::RequestOptions decode = benchmark_request(test, true);
        std::vector<ninfer::PreparedPrompt> prompts;
        prompts.reserve(concurrency);
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            prompts.push_back(engine.prepare_tokens(histories[lane], true));
        }
        std::vector<ninfer::GenerationHandle> handles;
        handles.reserve(concurrency);
        const auto wave_start = std::chrono::steady_clock::now();
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            handles.push_back(engine.submit(std::move(prompts[lane]), decode, ninfer::OutputDelivery::TerminalOnly, kBenchmarkPendingDeadline));
        }
        std::vector<ninfer::GenerationResult> generated;
        generated.reserve(concurrency);
        auto wave_end = wave_start;
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            ninfer::GenerationResult result = handles[lane].wait();
            if (lane + 1 == concurrency) { wave_end = std::chrono::steady_clock::now(); }
            generated.push_back(std::move(result));
        }
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            ninfer::GenerationResult& result = generated[lane];
            result = consume_generation(std::move(result), test, expected, " decode");
            if (result.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
                result.reused_prompt_tokens < static_cast<std::uint32_t>(prompt_tokens)) {
                throw std::runtime_error(
                    test.label + " concurrency-" + std::to_string(concurrency) +
                    " decode isolation missed prefix reuse on lane " + std::to_string(lane) +
                    " (path=" + std::to_string(static_cast<int>(result.prefix_reuse_path)) +
                    " reused=" + std::to_string(result.reused_prompt_tokens) +
                    " prompt=" + std::to_string(prompt_tokens) +
                    " history=" + std::to_string(histories[lane].size()) + ")");
            }
        }
        ninfer::bench::RepTiming timing = ninfer::bench::fold_lane_results(generated, expected);
        timing.timings.prefill_seconds  = 0.0;
        timing.wave_seconds =
            std::chrono::duration<double>(wave_end - wave_start).count();
        return timing;
    }

    const ninfer::RequestOptions request = benchmark_request(test, false);
    const auto whole_start = std::chrono::steady_clock::now();
    std::vector<ninfer::PreparedPrompt> prompts;
    prompts.reserve(concurrency);
    for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
        prompts.push_back(engine.prepare_tokens(
            ninfer::bench::prompt_slice(corpus, prompt_tokens, lane), false));
    }
    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(concurrency);
    const auto wave_start = std::chrono::steady_clock::now();
    for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
        handles.push_back(engine.submit(std::move(prompts[lane]), request, ninfer::OutputDelivery::TerminalOnly, kBenchmarkPendingDeadline));
    }
    std::vector<ninfer::GenerationResult> generated;
    generated.reserve(concurrency);
    auto wave_end = wave_start;
    for (auto& handle : handles) {
        ninfer::GenerationResult result = handle.wait();
        if (&handle == &handles.back()) { wave_end = std::chrono::steady_clock::now(); }
        generated.push_back(std::move(result));
    }
    for (ninfer::GenerationResult& result : generated) {
        result = consume_generation(std::move(result), test, expected, "");
    }
    ninfer::bench::RepTiming timing = ninfer::bench::fold_lane_results(generated, expected);
    if (test.kind == ninfer::bench::TestKind::WholeInference) {
        timing.timings.total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - whole_start).count();
    }
    timing.wave_seconds = std::chrono::duration<double>(wave_end - wave_start).count();
    return timing;
}

void prime_decode_graph(ninfer::Engine& engine, ninfer::bench::BenchEnvironment& env,
                        const std::vector<ninfer::TokenId>& corpus) {
    if (!env.use_device_graph || env.decode_graph_prime_output_tokens == 0) { return; }
    const int decode_tokens = static_cast<int>(env.decode_graph_prime_output_tokens - 1);
    const ninfer::bench::BenchTest prime{ninfer::bench::TestKind::Decode, 0, decode_tokens,
                                         "decode-graph-prime"};
    (void)run_repetition(engine, prime, corpus, 1);
    if (env.concurrency > 1) { (void)run_repetition(engine, prime, corpus, env.concurrency); }
    env.decode_graph_primed = true;
}

struct StatsWindow {
    double seconds                 = 0.0;
    std::uint64_t decode_tokens     = 0;
    std::uint64_t decode_rounds     = 0;
    std::uint64_t decode_row_rounds = 0;
    std::uint64_t prefill_tokens    = 0;
};

StatsWindow stats_window(const ninfer::RuntimeStats& before, const ninfer::RuntimeStats& after,
                         std::chrono::steady_clock::time_point begin,
                         std::chrono::steady_clock::time_point end) {
    return StatsWindow{
        .seconds           = std::chrono::duration<double>(end - begin).count(),
        .decode_tokens     = after.committed_decode_tokens - before.committed_decode_tokens,
        .decode_rounds     = after.decode_rounds - before.decode_rounds,
        .decode_row_rounds = after.decode_row_rounds - before.decode_row_rounds,
        .prefill_tokens    = after.computed_prefill_tokens - before.computed_prefill_tokens,
    };
}

std::string window_json(const char* label, const StatsWindow& w) {
    std::ostringstream out;
    out << "  \"" << label << "\": {\"seconds\": " << w.seconds
        << ", \"decode_tokens\": " << w.decode_tokens << ", \"decode_rounds\": " << w.decode_rounds
        << ", \"decode_row_rounds\": " << w.decode_row_rounds
        << ", \"prefill_tokens\": " << w.prefill_tokens
        << ", \"decode_tok_s\": " << w.decode_tokens / w.seconds
        << ", \"decode_rounds_s\": " << w.decode_rounds / w.seconds
        << ", \"prefill_tok_s\": " << w.prefill_tokens / w.seconds << "}";
    return out.str();
}

// C-1 lanes decode continuously while the remaining lane runs R fresh P-token prefills back to
// back. Decode-only windows before and after bracket acceptance drift of the decoding lanes.
std::string run_contention(ninfer::Engine& engine, const ninfer::bench::BenchOptions& options,
                           const std::vector<ninfer::TokenId>& corpus,
                           std::uint32_t decode_output_tokens) {
    using Clock = std::chrono::steady_clock;
    const auto [prefill_tokens, prefills] = *options.contention;
    const std::uint32_t decode_lanes =
        options.contention_lanes != 0 ? options.contention_lanes : options.concurrency - 1U;
    const int decode_prompt_tokens        = static_cast<int>(options.contention_context);
    constexpr auto kBaselineWindow        = std::chrono::seconds(4);

    ninfer::RequestOptions decode;
    decode.execution.requested_output_tokens = decode_output_tokens;
    decode.execution.allow_prefix_reuse      = false;
    decode.execution.sampling.temperature    = 0.0F;
    decode.stop.include_model_defaults       = false;
    decode.output.raw                        = true;
    decode.output.preserve_special_tokens    = true;
    const std::uint64_t rounds_before = engine.runtime_stats().decode_rounds;
    std::vector<ninfer::GenerationHandle> decoding;
    for (std::uint32_t lane = 0; lane < decode_lanes; ++lane) {
        decoding.push_back(engine.submit(
            engine.prepare_tokens(ninfer::bench::prompt_slice(corpus, decode_prompt_tokens,
                                                              4099U * lane),
                                  false),
            decode, ninfer::OutputDelivery::TerminalOnly, kBenchmarkPendingDeadline));
    }
    // The decode lanes prefill first: allow about 1000 prompt tokens per second per lane.
    const auto ready_deadline =
        Clock::now() + std::chrono::seconds(120 + decode_lanes * options.contention_context / 1000U);
    for (;;) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        if (stats.decode_ready_requests == decode_lanes && stats.prefilling_requests == 0 &&
            stats.decode_rounds >= rounds_before + 16U) {
            break;
        }
        if (Clock::now() > ready_deadline) {
            throw std::runtime_error("contention decode lanes did not become decode-ready");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    const auto baseline = [&] {
        const ninfer::RuntimeStats before = engine.runtime_stats();
        const auto begin                  = Clock::now();
        std::this_thread::sleep_for(kBaselineWindow);
        return stats_window(before, engine.runtime_stats(), begin, Clock::now());
    };
    const StatsWindow before_window = baseline();

    ninfer::RequestOptions prefill = decode;
    prefill.execution.requested_output_tokens = 1;
    std::vector<double> ttft;
    const ninfer::RuntimeStats contention_before = engine.runtime_stats();
    const auto contention_begin                  = Clock::now();
    for (int index = 0; index < prefills; ++index) {
        auto prompt = engine.prepare_tokens(
            ninfer::bench::prompt_slice(corpus, prefill_tokens,
                                        20011U + 7919U * static_cast<std::size_t>(index)),
            false);
        const auto submitted = Clock::now();
        ninfer::GenerationResult result =
            engine.submit(std::move(prompt), prefill, ninfer::OutputDelivery::TerminalOnly,
                          kBenchmarkPendingDeadline)
                .wait();
        ttft.push_back(std::chrono::duration<double>(Clock::now() - submitted).count());
        if (result.generated_token_ids.size() != 1U) {
            throw std::runtime_error("contention prefill did not generate its token");
        }
    }
    const StatsWindow contention = stats_window(contention_before, engine.runtime_stats(),
                                                contention_begin, Clock::now());
    const StatsWindow after_window = baseline();
    const ninfer::RuntimeStats finished = engine.runtime_stats();
    if (finished.decode_ready_requests != decode_lanes) {
        throw std::runtime_error("a decode lane finished before the contention windows closed; "
                                 "raise the decode output budget");
    }
    decoding.clear();

    std::ostringstream out;
    out << "{\n  \"concurrency\": " << options.concurrency
        << ", \"prefill_tokens\": " << prefill_tokens << ", \"prefills\": " << prefills
        << ", \"prefill_chunk\": " << options.prefill_chunk
        << ", \"prefill_slice\": " << options.prefill_slice
        << ", \"prefill_slice_rounds\": " << options.prefill_slice_rounds
        << ", \"decode_context\": " << options.contention_context
        << ", \"decode_lanes\": " << decode_lanes << ",\n  \"ttft_s\": [";
    for (std::size_t i = 0; i < ttft.size(); ++i) { out << (i ? ", " : "") << ttft[i]; }
    out << "],\n" << window_json("decode_only_before", before_window) << ",\n"
        << window_json("contention", contention) << ",\n"
        << window_json("decode_only_after", after_window) << "\n}\n";
    return out.str();
}

// Lane A decodes 4 * G tokens from a 512-token prompt; once it is decode-ready, lane B prefills a
// fresh P-token prompt and generates G tokens. Greedy streams depend on the schedule only through
// numerics, so comparing runs across prefill policies checks the mixed paths end to end.
std::string run_pair_check(ninfer::Engine& engine, const ninfer::bench::BenchOptions& options,
                           const std::vector<ninfer::TokenId>& corpus) {
    const auto [prompt_tokens, generated] = *options.pair_check;
    ninfer::RequestOptions request;
    request.execution.allow_prefix_reuse   = false;
    request.execution.sampling.temperature = 0.0F;
    request.stop.include_model_defaults    = false;
    request.output.raw                     = true;
    request.output.preserve_special_tokens = true;
    ninfer::RequestOptions decode_a = request;
    decode_a.execution.requested_output_tokens = static_cast<std::uint32_t>(4 * generated);
    ninfer::RequestOptions prefill_b = request;
    prefill_b.execution.requested_output_tokens = static_cast<std::uint32_t>(generated);
    const std::uint64_t rounds_before = engine.runtime_stats().decode_rounds;
    ninfer::GenerationHandle a = engine.submit(
        engine.prepare_tokens(ninfer::bench::prompt_slice(corpus, 512, 101U), false), decode_a,
        ninfer::OutputDelivery::TerminalOnly, kBenchmarkPendingDeadline);
    while (engine.runtime_stats().decode_rounds < rounds_before + 4U) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ninfer::GenerationHandle b = engine.submit(
        engine.prepare_tokens(ninfer::bench::prompt_slice(corpus, prompt_tokens, 30011U), false),
        prefill_b, ninfer::OutputDelivery::TerminalOnly, kBenchmarkPendingDeadline);
    const ninfer::GenerationResult rb = b.wait();
    const ninfer::GenerationResult ra = a.wait();
    std::ostringstream out;
    const auto emit = [&](const char* label, const ninfer::GenerationResult& r) {
        out << "  \"" << label << "\": [";
        for (std::size_t i = 0; i < r.generated_token_ids.size(); ++i) {
            out << (i ? "," : "") << r.generated_token_ids[i];
        }
        out << "]";
    };
    out << "{\n  \"prefill_slice\": " << options.prefill_slice
        << ",\n  \"prefill_slice_rounds\": " << options.prefill_slice_rounds << ",\n";
    emit("a", ra);
    out << ",\n";
    emit("b", rb);
    out << "\n}\n";
    return out.str();
}

void write_output(const ninfer::bench::BenchOptions& options, const std::string& text) {
    if (options.output_file.empty()) {
        std::cout << text;
        return;
    }
    const std::filesystem::path path(options.output_file);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open output file: " + options.output_file); }
    output << text;
    std::cout << "wrote " << options.output_file << '\n';
}

} // namespace

int main(int argc, char** argv) {
    ninfer::bench::BenchOptions options;
    try {
        options = ninfer::bench::parse_args(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninfer_bench: " << error.what() << '\n';
        return 2;
    }
    if (options.help_requested) {
        std::cout << ninfer::bench::usage_text(argc > 0 ? argv[0] : "ninfer_bench");
        return 0;
    }

    try {
        const std::vector<ninfer::TokenId> corpus =
            ninfer::bench::load_corpus_ids(options.corpus_path);
        const std::vector<ninfer::bench::BenchTest> tests = ninfer::bench::expand_tests(options);
        if (options.profile_measured && (tests.size() != 1 || options.repetitions != 1)) {
            throw std::invalid_argument(
                "--profile-measured requires exactly one benchmark test and -r 1");
        }
        ninfer::bench::validate_prompt_lengths(tests, corpus.size());
        const ninfer::SpeculativeOptions spec_options{
            options.spec_backend, options.draft_tokens, options.proposal_head,
            options.dflash_verify_width, options.adaptive_draft};
        // C > 1 pp+tg always isolates decode behind a one-token seed, like --isolate-prompt-decode.
        constexpr std::uint32_t kContentionDecodeTokens = 6144;
        const std::uint32_t max_context =
            options.pair_check
                ? static_cast<std::uint32_t>(std::max(options.pair_check->first,
                                                      512 + 4 * options.pair_check->second) +
                                             256)
            : options.contention
                ? options.max_context.value_or(std::max<std::uint32_t>(
                      static_cast<std::uint32_t>(options.contention->first) + 128U,
                      options.contention_context + kContentionDecodeTokens + 128U))
                : ninfer::bench::resolve_max_context(
                      tests, options.max_context, spec_options, options.use_device_graph,
                      options.isolate_prompt_decode || options.concurrency > 1);

        ninfer::EngineOptions engine_options;
        engine_options.artifact_path = options.artifact_path;
        engine_options.device        = options.device;
        engine_options.max_context   = max_context;
        engine_options.max_concurrency = options.concurrency;
        engine_options.max_pending_requests = options.concurrency;
        engine_options.pending_timeout_ms   = ninfer::bench::kBenchmarkPendingTimeoutMs;
        engine_options.kv_capacity = options.automatic_kv_capacity
                                         ? ninfer::KvCapacityPolicy::automatic()
                                         : ninfer::KvCapacityPolicy::explicit_capacity(
                                               ninfer::bench::concurrent_kv_capacity_tokens(
                                                   max_context, options.concurrency));
        engine_options.prefill_chunk = options.prefill_chunk;
        engine_options.prefill_slice = options.prefill_slice;
        engine_options.prefill_slice_rounds = options.prefill_slice_rounds;
        engine_options.speculative.backend       = options.draft_tokens == 0
                                                       ? ninfer::SpeculativeBackend::None
                                                       : options.spec_backend;
        engine_options.speculative.draft_tokens  = options.draft_tokens;
        engine_options.speculative.adaptive_draft = options.adaptive_draft;
        engine_options.speculative.proposal_head = options.proposal_head;
        engine_options.speculative.dflash_verify_width = options.dflash_verify_width;
        engine_options.use_device_graph          = options.use_device_graph;

        ninfer::bench::BenchEnvironment env;
        env.artifact_path            = options.artifact_path;
        env.artifact_file_size_bytes = ninfer::bench::file_size_or_zero(options.artifact_path);
        env.max_context              = max_context;
        env.prefill_chunk            = options.prefill_chunk;
        env.concurrency              = options.concurrency;
        env.pending_timeout_ms       = ninfer::bench::kBenchmarkPendingTimeoutMs;
        env.speculative_backend      = options.draft_tokens == 0
                                           ? ninfer::SpeculativeBackend::None
                                           : options.spec_backend;
        env.draft_tokens             = options.draft_tokens;
        env.dflash_verify_width_requested = options.dflash_verify_width;
        env.dflash_verify_width = options.spec_backend == ninfer::SpeculativeBackend::DFlash
                                      ? ninfer::bench::resolved_dflash_verify_width(
                                            options.draft_tokens, options.dflash_verify_width)
                                      : 0;
        env.proposal_head            = options.proposal_head;
        env.use_device_graph         = options.use_device_graph;
        env.retain_token_ids         = options.retain_token_ids;
        env.isolate_prompt_decode    = options.isolate_prompt_decode;
        env.repetitions              = options.repetitions;
        env.warmup                   = options.warmup;
        env.corpus_path              = options.corpus_path;
        env.corpus_tokens            = corpus.size();
        if (options.use_device_graph && has_decode_tests(tests)) {
            env.decode_graph_prime_output_tokens =
                ninfer::bench::decode_graph_prime_output_tokens(spec_options);
        }

        std::cerr << "[ninfer_bench] loading " << options.artifact_path
                  << " (max_context=" << max_context << ", concurrency=" << options.concurrency
                  << ", kv_format=fp8-k-int4-v)\n";
        ninfer::Engine engine(std::move(engine_options));
        fill_hip_environment(env, options.device);
        env.load   = engine.load_summary();
        env.memory = engine.memory_summary();

        prime_decode_graph(engine, env, corpus);
        if (options.contention) {
            write_output(options, run_contention(engine, options, corpus, kContentionDecodeTokens));
            return 0;
        }
        if (options.pair_check) {
            write_output(options, run_pair_check(engine, options, corpus));
            return 0;
        }

        std::vector<ninfer::bench::TestResult> results;
        results.reserve(tests.size());
        for (std::size_t i = 0; i < tests.size(); ++i) {
            const auto& test = tests[i];
            std::cerr << "[ninfer_bench] test " << (i + 1) << '/' << tests.size() << ' '
                      << test.label << ": warmup=" << options.warmup
                      << " reps=" << options.repetitions << '\n';

            ninfer::bench::TestResult result;
            result.test        = test;
            result.concurrency = options.concurrency;
            engine.reset_memory_peaks();
            for (int warmup = 0; warmup < options.warmup; ++warmup) {
                (void)run_repetition(engine, test, corpus, options.concurrency,
                                     options.isolate_prompt_decode);
            }
            result.reps.reserve(static_cast<std::size_t>(options.repetitions));
            ProfileMeasuredRegion profile_region(options.profile_measured);
            for (int repetition = 0; repetition < options.repetitions; ++repetition) {
                result.reps.push_back(run_repetition(engine, test, corpus, options.concurrency,
                                                     options.isolate_prompt_decode));
            }
            profile_region.finish();
            const ninfer::MemorySummary memory    = engine.memory_summary();
            result.workspace_peak_bytes           = memory.workspace_logical_peak_bytes;
            result.workspace_allocator_peak_bytes = memory.workspace.peak_used_bytes;
            results.push_back(std::move(result));
        }

        std::string report;
        switch (options.output) {
        case ninfer::bench::OutputFormat::Table:
            report = ninfer::bench::format_table(env, results);
            break;
        case ninfer::bench::OutputFormat::Json:
            report = ninfer::bench::format_json(env, command_line(argc, argv), results);
            break;
        case ninfer::bench::OutputFormat::Csv:
            report = ninfer::bench::format_csv(env, results);
            break;
        }
        write_output(options, report);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer_bench: " << error.what() << '\n';
        return 1;
    }
}
