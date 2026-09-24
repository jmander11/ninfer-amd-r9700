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
        const std::uint32_t max_context = ninfer::bench::resolve_max_context(
            tests, options.max_context, spec_options, options.use_device_graph,
            options.isolate_prompt_decode);

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
