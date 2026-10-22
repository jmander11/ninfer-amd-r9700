#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <condition_variable>
#include <mutex>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::RequestOptions request_options(bool reuse = true) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens    = 16;
    options.execution.allow_prefix_reuse         = reuse;
    options.execution.sampling.temperature       = 0.0F;
    options.execution.sampling.top_k             = 0;
    options.execution.sampling.top_p             = 1.0F;
    options.execution.sampling.min_p             = 0.0F;
    options.execution.sampling.presence_penalty  = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.execution.sampling.seed              = 1;
    options.stop.include_model_defaults          = false;
    options.output.raw                           = true;
    options.output.preserve_special_tokens       = true;
    return options;
}

std::vector<ninfer::TokenId> prompt_tokens(ninfer::Engine& engine, std::string text) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    ninfer::ChatMessage message;
    message.parts.push_back(ninfer::MessagePart{.text = std::move(text)});
    input.messages.push_back(std::move(message));
    const auto prepared = engine.prepare(std::move(input));
    return {prepared.token_ids().begin(), prepared.token_ids().end()};
}

std::vector<ninfer::TokenId> continuation(std::vector<ninfer::TokenId> prefix,
                                          const ninfer::GenerationResult& result) {
    prefix.insert(prefix.end(), result.generated_token_ids.begin(),
                  result.generated_token_ids.end());
    return prefix;
}

void require_same_output(const ninfer::GenerationResult& actual,
                         const ninfer::GenerationResult& cold) {
    require(actual.generated_token_ids == cold.generated_token_ids,
            "reused continuation differs from cold greedy token output");
    require(actual.content == cold.content && actual.reasoning == cold.reasoning &&
                actual.finish_reason == cold.finish_reason,
            "reused continuation differs from cold published output");
}

class CancelAfterOutput final : public ninfer::OutputSink {
public:
    void publish(ninfer::OutputDelta delta) override {
        if (!delta.text.empty()) { requested = true; }
    }

    bool requested = false;
};

// C1 plus disposable, non-reusing requests makes each unrelated admission release the old
// allocation. The pool returns its lowest physical pages first, so the short request overwrites
// the beginning of the long request's pages while their unwritten tails retain old contents.
void qualify_recycled_pages(ninfer::Engine& engine) {
    std::vector<ninfer::TokenId> long_prompt(173, 198);
    long_prompt.front() = 1000;
    std::vector<ninfer::TokenId> short_prompt(33, 198);
    short_prompt.front()             = 2000;
    auto disposable                  = request_options(false);
    disposable.execution.cache_write = false;
    // Forty decode tokens cross the short prompt's first page boundary, checking that newly
    // visible slots are overwritten as the frontier advances into recycled pages.
    disposable.execution.requested_output_tokens = 40;
    const auto cold = engine.generate(engine.prepare_tokens(short_prompt), disposable);
    require(cold.generated_token_ids.size() == 40, "short recycle control did not complete");

    const auto before = engine.runtime_stats();
    require(before.gpu_kv_main_capacity_pages == 4,
            "recycle qualifier requires exactly four physical KV pages");
    const auto long_result = engine.generate(engine.prepare_tokens(long_prompt), disposable);
    require(long_result.generated_token_ids.size() == 40, "long recycle request did not complete");
    require(engine.runtime_stats().gpu_kv_main_mapped_pages == 4,
            "long recycle request did not occupy the complete physical KV pool");
    const auto recycled = engine.generate(engine.prepare_tokens(short_prompt), disposable);
    require(engine.runtime_stats().gpu_kv_main_mapped_pages == 2,
            "short recycle request did not replace the long allocation with two pages");
    require(recycled.reused_prompt_tokens == 0,
            "short recycle request unexpectedly reused a prefix");
    require_same_output(recycled, cold);

    CancelAfterOutput sink;
    const ninfer::CancellationView cancellation([&] { return sink.requested; });
    const auto cancelled =
        engine.generate(engine.prepare_tokens(long_prompt), disposable, &sink, cancellation);
    require(sink.requested && cancelled.finish_reason == ninfer::FinishReason::Cancelled &&
                !cancelled.generated_token_ids.empty(),
            "long recycle cancellation was not observed after published output");
    const auto after_cancel = engine.generate(engine.prepare_tokens(short_prompt), disposable);
    require(after_cancel.reused_prompt_tokens == 0,
            "short post-cancel request unexpectedly reused a prefix");
    require_same_output(after_cancel, cold);
    require(engine.runtime_stats().kv_ram_captures == before.kv_ram_captures,
            "disposable recycled-page requests unexpectedly preserved old allocations in RAM");
    std::cout << "engine_recycled_pages: PASS C1 disposable long173->short33 and "
                 "cancelled-long->short; 40-token page-crossing exact cold greedy output; "
                 "cancellation phase not forced\n";
}

int qualify_disk_restart(const std::filesystem::path& artifact,
                         const std::filesystem::path& directory) {
    require(std::filesystem::is_regular_file(artifact), "explicit artifact is not a regular file");
    // The result directory is retained as evidence, including on failure. Never open
    // an existing cache or remove caller-owned data during qualification.
    require(std::filesystem::create_directory(directory),
            "SSD qualification requires a new, nonexistent output directory");
    const auto options = [&] {
        ninfer::EngineOptions value;
        value.artifact_path          = artifact;
        value.max_context            = 256;
        value.kv_capacity            = ninfer::KvCapacityPolicy::explicit_capacity(256);
        value.max_concurrency        = 1;
        value.kv_ram_capacity_bytes  = 8ULL << 30;
        value.kv_disk_capacity_bytes = 1ULL << 30;
        value.kv_disk_location       = directory;
        value.speculative.backend    = ninfer::SpeculativeBackend::None;
        return value;
    };
    std::vector<ninfer::TokenId> continued;
    {
        ninfer::Engine engine(options());
        const auto prompt =
            prompt_tokens(engine, "List the integers from one to twenty, in order.");
        const auto other = prompt_tokens(engine, "Describe how a sailboat moves across a lake.");
        const auto saved = engine.generate(engine.prepare_tokens(prompt), request_options(false));
        require(saved.generated_token_ids.size() == 16, "SSD seed did not produce 16 tokens");
        continued         = continuation(prompt, saved);
        const auto before = engine.runtime_stats();
        (void)engine.generate(engine.prepare_tokens(other), request_options(false));
        require(engine.runtime_stats().kv_ram_captures > before.kv_ram_captures,
                "unrelated request did not capture A into RAM");
        // Public Engine destruction drains the scheduler and persists nondurable RAM
        // plus retained lanes. No private spill API or timing sleep is required.
    }
    {
        ninfer::Engine engine(options());
        // Stats read the disk tier without waiting for its index lock, so a read that races the
        // disk worker keeps the last snapshot; poll briefly for the first one.
        const auto disk_stats = [&](auto satisfied) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            auto stats          = engine.runtime_stats();
            while (!satisfied(stats) && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                stats = engine.runtime_stats();
            }
            return stats;
        };
        const auto before = disk_stats(
            [](const ninfer::RuntimeStats& stats) { return stats.kv_disk_entry_count > 0; });
        require(before.kv_disk_entry_count > 0, "restart did not recover durable SSD entries");
        require(before.kv_ram_restores == 0, "fresh Engine already restored RAM state");
        const auto restored = engine.generate(engine.prepare_tokens(continued), request_options());
        require(restored.prefix_reuse_source == ninfer::PrefixReuseSource::HostDisk &&
                    restored.reused_prompt_tokens > 0,
                "fresh Engine continuation did not exercise SSD restore");
        require(disk_stats([&](const ninfer::RuntimeStats& stats) {
                    return stats.kv_disk_restores > before.kv_disk_restores;
                }).kv_disk_restores > before.kv_disk_restores,
                "SSD restore counter did not advance");
        const auto cold = engine.generate(engine.prepare_tokens(continued), request_options(false));
        require_same_output(restored, cold);
        std::cout << "engine_cache_disk: PASS ordinary C1 greedy; SSD_reused="
                  << restored.reused_prompt_tokens
                  << " output_tokens=" << restored.generated_token_ids.size()
                  << " retained_directory=" << directory.string() << '\n';
    }
    return 0;
}

// Exercise boundary capture during interleaved prefill/decode, cancellation, and subsequent
// reusable restores through the public Engine. The exact GPU cancellation phase is not forced.
int qualify_system_prompts(const std::filesystem::path& artifact, std::string_view backend) {
    require(backend == "none" || backend == "dflash" || backend == "mtp",
            "system cache backend must be none, dflash, or mtp");
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = 8192;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(16384);
    options.max_concurrency       = 2;
    options.prefill_chunk         = 512;
    options.mixed_forward         = 512;
    options.kv_ram_capacity_bytes = 2ULL << 30;
    options.kv_ram_system_prompts = 2;
    options.generation_recovery   = false;
    if (backend != "none") {
        options.speculative.backend       = backend == "dflash" ? ninfer::SpeculativeBackend::DFlash
                                                                : ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens  = backend == "dflash" ? 7 : 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    std::string system = "You are a concise technical assistant. Follow the user's instruction.\n";
    for (unsigned i = 0; i < 80; ++i) {
        system += "Archive entry " + std::to_string(i) +
                  ": cedar trees grow beside the river. This is background context.\n";
    }
    const auto input = [&](std::string user, unsigned system_variant = 0) {
        ninfer::PromptInput value;
        value.options.enable_thinking = false;
        ninfer::ChatMessage instruction;
        instruction.role = ninfer::ChatRole::System;
        instruction.parts.push_back(ninfer::MessagePart{
            .text = (system_variant != 0 ? "Additional instruction " +
                                               std::to_string(system_variant) + ": be precise.\n"
                                         : "") +
                    system});
        value.messages.push_back(std::move(instruction));
        ninfer::ChatMessage message;
        message.parts.push_back(ninfer::MessagePart{.text = std::move(user)});
        value.messages.push_back(std::move(message));
        return value;
    };
    auto request                              = request_options();
    request.execution.cache_write             = true;
    request.execution.requested_output_tokens = 32;
    const std::string question                = "Name the capital of France.";
    ninfer::GenerationResult cold;
    {
        ninfer::Engine engine(options);
        cold = engine.generate(engine.prepare(input(question)), request);
        require(cold.reused_prompt_tokens == 0, "fresh Engine unexpectedly reused a prefix");
    }
    ninfer::Engine engine(options);
    (void)engine.generate(engine.prepare(input("What is three plus five?")), request);
    auto warm = engine.generate(engine.prepare(input(question)), request);
    require_same_output(warm, cold);
    require(warm.system_prompt_cache_hit_tokens > 0 &&
                warm.system_prompt_cache_hit_tokens == warm.reused_prompt_tokens,
            "sequential different-user request did not restore the system snapshot");
    const auto boundary = warm.reused_prompt_tokens;
    // Queue all admissions before the first cold prefill completes. Only the first computes
    // the shared system prefix; later FIFO admissions must refresh their RAM plans.
    auto prompt_a            = engine.prepare(input("What is three plus five?", 1));
    auto prompt_b            = engine.prepare(input(question, 1));
    auto prompt_c            = engine.prepare(input("Give the chemical symbol for oxygen.", 1));
    auto first               = engine.submit(std::move(prompt_a), request);
    auto queued_b            = engine.submit(std::move(prompt_b), request);
    auto queued_c            = engine.submit(std::move(prompt_c), request);
    const auto first_result  = first.wait();
    const auto queued_result = queued_b.wait();
    const auto queued_third  = queued_c.wait();
    require(first_result.system_prompt_cache_hit_tokens == 0 &&
                queued_result.system_prompt_cache_hit_tokens > 0 &&
                queued_third.system_prompt_cache_hit_tokens ==
                    queued_result.system_prompt_cache_hit_tokens,
            "queued first-user variants did not share the newly captured system prefix");
    // Ordinary decode batch shape can change floating-point rounding; queued behavior is
    // checked by exact source/frontier and completion. Exact cache parity is sequential above.
    require(queued_result.prefix_reuse_source == ninfer::PrefixReuseSource::HostRam &&
                queued_third.prefix_reuse_source == ninfer::PrefixReuseSource::HostRam &&
                queued_result.generated_token_ids.size() == 32 &&
                queued_third.generated_token_ids.size() == 32,
            "queued same-system admissions did not complete with RAM reuse");
    const auto captures                           = engine.runtime_stats().kv_ram_captures;
    auto long_decode                              = request;
    long_decode.execution.requested_output_tokens = 128;
    auto decode = engine.submit(engine.prepare(input("Count the integers in order.")), long_decode);
    // This different system is a cold capture. It must stop exactly at the unaligned system
    // boundary while another lane decodes; DFlash takes the mixed-target-forward route.
    auto mixed =
        engine.submit(engine.prepare(input("Give the chemical symbol for oxygen.", 2)), request);
    // The third admission can use the retained snapshot while the earlier chats decode.
    auto active_shared =
        engine.submit(engine.prepare(input("Write the integer after forty-one.")), request);
    const auto mixed_result  = mixed.wait();
    const auto decode_result = decode.wait();
    const auto active_result = active_shared.wait();
    require(active_result.system_prompt_cache_hit_tokens == boundary,
            "new job beside active chats did not reuse the retained system snapshot");
    require(mixed_result.generated_token_ids.size() == 32 &&
                decode_result.generated_token_ids.size() == 128,
            "concurrent system capture failed to complete both requests");
    require(engine.runtime_stats().kv_ram_system_prompt_entries == 2 &&
                engine.runtime_stats().kv_ram_captures > captures,
            "mixed prefill did not retain its second system prefix");
    CancelAfterOutput sink;
    const ninfer::CancellationView cancellation([&] { return sink.requested; });
    const auto cancelled = engine.generate(engine.prepare(input("List the days of the week.")),
                                           long_decode, &sink, cancellation);
    require(sink.requested && cancelled.finish_reason == ninfer::FinishReason::Cancelled,
            "system-cache cancellation was not observed");
    warm = engine.generate(engine.prepare(input(question)), request);
    require_same_output(warm, cold);
    const auto stats = engine.runtime_stats();
    require(stats.kv_ram_system_prompt_entries == 2 && stats.kv_ram_system_prompt_restores >= 3 &&
                stats.kv_cache_fallbacks == 0 &&
                stats.kv_ram_used_bytes <= stats.kv_ram_capacity_bytes,
            "retained system state/budget was lost after concurrent or cancelled work");
    std::cout << "system_prompt_cache: PASS backend=" << backend << " C2 boundary=" << boundary
              << " entries=" << stats.kv_ram_system_prompt_entries
              << " restores=" << stats.kv_ram_system_prompt_restores
              << " bytes=" << stats.kv_ram_system_prompt_used_bytes
              << " cold_ttft_s=" << cold.timings.first_token_seconds
              << " warm_ttft_s=" << warm.timings.first_token_seconds
              << "; queued same-system admissions, active-chat reuse, exact cold greedy output, "
                 "mixed capture, cancellation, no fallback\n";
    return 0;
}

// Deliberately exceeds the initial 8192-token output reservation in two lanes while
// the shared pool fits only one completed context. This is a pressure/lifecycle
// regression, not a token-parity oracle across different compact batch sizes.
int qualify_chunked_pressure(const std::filesystem::path& artifact, std::string_view backend,
                             bool reasoning = false) {
    require(backend == "none" || backend == "dflash", "expected none or dflash backend");
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = 16384;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(18432);
    options.max_concurrency       = 2;
    options.kv_ram_capacity_bytes = 8ULL << 30;
    options.speculative.backend =
        backend == "none" ? ninfer::SpeculativeBackend::None : ninfer::SpeculativeBackend::DFlash;
    if (backend == "dflash") {
        options.speculative.draft_tokens   = 7;
        options.speculative.adaptive_draft = true;
        options.speculative.proposal_head  = ninfer::ProposalHead::Optimized;
    }
    options.generation_recovery = false;
    ninfer::Engine engine(options);
    std::atomic<bool> stalled{false};
    std::jthread observer([&](std::stop_token stop) {
        auto previous = engine.runtime_stats();
        auto changed  = std::chrono::steady_clock::now();
        while (!stop.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            const auto now = engine.runtime_stats();
            if (now.computed_prefill_tokens != previous.computed_prefill_tokens ||
                now.committed_decode_tokens != previous.committed_decode_tokens) {
                changed = std::chrono::steady_clock::now();
            }
            std::cerr << "pressure prefill=" << now.computed_prefill_tokens
                      << " decode=" << now.committed_decode_tokens
                      << " ready=" << now.decode_ready_requests
                      << " blocked=" << now.kv_growth_blocked_requests
                      << " suspended=" << now.kv_suspended_requests
                      << " extensions=" << now.kv_window_extensions
                      << " captures=" << now.kv_live_suspensions
                      << " restores=" << now.kv_live_resumes << std::endl;
            if (std::chrono::steady_clock::now() - changed > std::chrono::seconds(45)) {
                stalled.store(true);
            }
            previous = now;
        }
    });
    const ninfer::CancellationView cancellation([&] { return stalled.load(); });
    auto long_options                              = request_options(false);
    long_options.execution.cache_write             = false;
    long_options.execution.requested_output_tokens = 12000;
    // Positive-temperature penalties exercise live sampler-count preservation and
    // rebinding in addition to counter-based position RNG and target state.
    long_options.execution.sampling.p_less            = false;
    long_options.execution.sampling.temperature       = 0.7F;
    long_options.execution.sampling.presence_penalty  = 0.2F;
    long_options.execution.sampling.frequency_penalty = 0.1F;
    std::vector<ninfer::TokenId> first_prompt(256, 198);
    std::vector<ninfer::TokenId> second_prompt(256, 198);
    first_prompt.front()    = 1000;
    second_prompt.front()   = 2000;
    const auto prepare_long = [&](const std::vector<ninfer::TokenId>& raw) {
        if (!reasoning) { return engine.prepare_tokens(raw); }
        ninfer::PromptInput input;
        input.options.enable_thinking = true;
        ninfer::ChatMessage message;
        message.parts.push_back(ninfer::MessagePart{
            .text = "Derive the sum of fourth powers of integers from 1 to 1000 in two independent "
                    "ways, checking each algebraic identity. Then enumerate the fourth powers "
                    "of the first thousand integers with intermediate arithmetic. Case " +
                    std::to_string(raw.front())});
        input.messages.push_back(std::move(message));
        return engine.prepare(std::move(input));
    };
    if (reasoning) {
        long_options.output.raw                     = false;
        long_options.output.preserve_special_tokens = false;
        long_options.output.thinking_token_budget   = 32;
    }
    auto first               = engine.submit(prepare_long(first_prompt), long_options);
    auto second              = engine.submit(prepare_long(second_prompt), long_options);
    auto replacement_options = request_options(false);
    replacement_options.execution.cache_write = false;
    auto replacement = engine.submit(engine.prepare_tokens({3000, 198, 198}), replacement_options);
    const auto first_result       = first.wait(nullptr, cancellation);
    const auto second_result      = second.wait(nullptr, cancellation);
    const auto replacement_result = replacement.wait(nullptr, cancellation);
    require(!stalled.load(), "pressure made no prefill/decode progress for 45 seconds");
    if (reasoning) {
        for (const auto* result : {&first_result, &second_result}) {
            require(result->reasoning_limit.used_tokens == 32 &&
                        result->reasoning_limit.forced_closes == 1 &&
                        result->reasoning_limit.forced_tokens.size() == 1 &&
                        result->reasoning_limit.retries == 0 && result->recovery.attempts == 0,
                    "real pressure lost the cumulative cap or forced close across suspension");
        }
    }
    require(first_result.generated_token_ids.size() == 12000 &&
                second_result.generated_token_ids.size() == 12000,
            "chunk boundary shortened a request or failed to resume it");
    require(first_result.finish_reason == ninfer::FinishReason::OutputLimit &&
                second_result.finish_reason == ninfer::FinishReason::OutputLimit,
            "chunk pressure changed the lifetime output-limit reason");
    require(replacement_result.generated_token_ids.size() == 16,
            "queued replacement did not complete under chunk pressure");
    auto stats = engine.runtime_stats();
    require(stats.kv_execution_window_tokens == 8192 && stats.kv_window_extensions > 0,
            "RAM-enabled requests did not use chunked output reservations");
    require(stats.kv_live_suspensions > 0 && stats.kv_live_resumes > 0,
            "bounded pool did not exercise active RAM suspension and resume");
    require(stats.running_requests == 0 && stats.kv_suspended_requests == 0 &&
                stats.kv_growth_blocked_requests == 0,
            "pressure run leaked admitted or blocked live requests");
    // A new request after all transfers tests released credit/page ownership.
    const auto after = engine.generate(engine.prepare_tokens(first_prompt), replacement_options);
    require(after.generated_token_ids.size() == 16,
            "post-pressure request could not use released live resources");
    std::cout << "chunked_pressure: PASS backend=" << backend << " reasoning=" << reasoning
              << " C2 pool18432 context16384 output12000x2 plus queued replacement; extensions="
              << stats.kv_window_extensions << " suspensions=" << stats.kv_live_suspensions
              << " resumes=" << stats.kv_live_resumes
              << "; lifecycle/budget check, not cross-batch numerical parity\n";
    return 0;
}

// Timed raw-token scheduling workload: full 32k ceilings stay in accounting,
// while cancellation bounds the experiment independently of generated EOS.
int qualify_chunked_throughput(const std::filesystem::path& artifact, std::string_view backend,
                               bool completion = false) {
    require(backend == "none" || backend == "dflash", "expected none or dflash backend");
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = 262144;
    options.max_concurrency       = 8;
    options.kv_capacity           = ninfer::KvCapacityPolicy::automatic();
    options.kv_ram_capacity_bytes = 12ULL << 30;
    options.kv_ram_system_prompts = 2;
    options.generation_recovery   = false;
    options.pending_timeout_ms    = completion ? 1800000 : 900000;
    options.speculative.backend =
        backend == "none" ? ninfer::SpeculativeBackend::None : ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens   = 7;
    options.speculative.adaptive_draft = true;
    options.speculative.proposal_head  = ninfer::ProposalHead::Optimized;
    ninfer::Engine engine(options);
    std::cerr << "throughput loaded backend=" << backend
              << " pool_tokens=" << engine.memory_summary().kv_capacity << std::endl;
    std::array<ninfer::GenerationHandle, 8> handles;
    std::array<ninfer::GenerationResult, 8> results;
    std::array<std::exception_ptr, 8> errors{};
    auto request                              = request_options(false);
    request.execution.cache_write             = false;
    request.execution.requested_output_tokens = 32768;
    request.execution.sampling.p_less         = true;
    request.execution.sampling.temperature    = 1.5F;
    std::array<ninfer::PreparedPrompt, 8> prompts;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        std::vector<ninfer::TokenId> prompt(60000, 198);
        prompt.front() = static_cast<ninfer::TokenId>(1000 + i);
        prompts[i]     = engine.prepare_tokens(std::move(prompt));
    }
    using Clock         = std::chrono::steady_clock;
    const auto initial  = engine.runtime_stats();
    const auto started  = Clock::now();
    const auto deadline = started + std::chrono::seconds(completion ? 1800 : 300);
    std::mutex client_mutex;
    std::condition_variable client_changed;
    std::size_t client_done = 0;
    std::array<Clock::time_point, 8> client_completed{};
    for (std::size_t i = 0; i < handles.size(); ++i) {
        request.execution.sampling.seed = 17 + i;
        handles[i]                      = engine.submit(std::move(prompts[i]), request);
    }
    std::vector<std::jthread> clients;
    for (std::size_t i = 0; i < handles.size(); ++i) {
        clients.emplace_back([&, i] {
            try {
                results[i] = handles[i].wait(nullptr, ninfer::CancellationView([deadline] {
                                                 return Clock::now() >= deadline;
                                             }));
            } catch (...) { errors[i] = std::current_exception(); }
            {
                const std::lock_guard lock(client_mutex);
                client_completed[i] = Clock::now();
                ++client_done;
            }
            client_changed.notify_one();
        });
    }
    auto midpoint                           = initial;
    double midpoint_seconds                 = 0;
    std::uint32_t sampled_peak_active       = 0;
    std::uint32_t sampled_peak_decode_ready = 0;
    while (Clock::now() < deadline) {
        if (completion) {
            std::unique_lock lock(client_mutex);
            if (client_changed.wait_until(
                    lock, std::min(deadline, Clock::now() + std::chrono::seconds(5)),
                    [&] { return client_done == handles.size(); })) {
                break;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        const auto now            = engine.runtime_stats();
        sampled_peak_active       = std::max(sampled_peak_active, now.running_requests);
        sampled_peak_decode_ready = std::max(sampled_peak_decode_ready, now.decode_ready_requests);
        const double seconds      = std::chrono::duration<double>(Clock::now() - started).count();
        if (midpoint_seconds == 0 && seconds >= 150) {
            midpoint         = now;
            midpoint_seconds = seconds;
        }
        std::cerr << "sample seconds=" << seconds << " prefill=" << now.computed_prefill_tokens
                  << " decode=" << now.committed_decode_tokens << " rounds=" << now.decode_rounds
                  << " rows=" << now.decode_row_rounds << " active=" << now.running_requests
                  << " prefilling=" << now.prefilling_requests
                  << " ready=" << now.decode_ready_requests << " waiting=" << now.waiting_requests
                  << " blocked=" << now.kv_growth_blocked_requests
                  << " suspended=" << now.kv_suspended_requests << " ram=" << now.kv_ram_used_bytes
                  << std::endl;
    }
    const auto sampled           = engine.runtime_stats();
    const double sampled_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    clients.clear();
    const double makespan_seconds =
        std::chrono::duration<double>(
            *std::max_element(client_completed.begin(), client_completed.end()) - started)
            .count();
    auto drained         = engine.runtime_stats();
    const auto quiescent = [](const auto& stats) {
        return stats.running_requests == 0 && stats.prefilling_requests == 0 &&
               stats.decode_ready_requests == 0 && stats.waiting_requests == 0 &&
               stats.kv_suspended_requests == 0 && stats.kv_growth_blocked_requests == 0;
    };
    if (completion) {
        const auto settle_deadline = Clock::now() + std::chrono::seconds(5);
        while (!quiescent(drained) && Clock::now() < settle_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            drained = engine.runtime_stats();
        }
        require(quiescent(drained), "completion workload did not reach quiescent final accounting");
    }
    require(drained.running_requests == 0 && drained.waiting_requests == 0 &&
                drained.kv_suspended_requests == 0 && drained.kv_growth_blocked_requests == 0,
            "timed throughput cancellation did not release all admitted requests");
    const double drain_seconds =
        std::chrono::duration<double>(Clock::now() - started).count() - sampled_seconds;
    for (const auto& error : errors)
        if (error) std::rethrow_exception(error);
    const auto measured      = completion ? drained : sampled;
    const auto decode        = measured.committed_decode_tokens - initial.committed_decode_tokens;
    const auto rounds        = measured.decode_rounds - initial.decode_rounds;
    const auto rows          = measured.decode_row_rounds - initial.decode_row_rounds;
    const auto steady_decode = sampled.committed_decode_tokens - midpoint.committed_decode_tokens;
    const auto steady_rounds = sampled.decode_rounds - midpoint.decode_rounds;
    const auto steady_rows   = sampled.decode_row_rounds - midpoint.decode_row_rounds;
    std::size_t cancelled    = 0;
    std::size_t completed    = 0;
    std::size_t emitted      = 0;
    double ttft              = 0;
    std::size_t started_requests = 0;
    for (const auto& result : results) {
        emitted += result.generated_token_ids.size();
        cancelled += result.finish_reason == ninfer::FinishReason::Cancelled;
        completed += result.finish_reason == ninfer::FinishReason::OutputLimit;
        if (!result.generated_token_ids.empty()) {
            ttft += result.timings.first_token_seconds;
            ++started_requests;
        }
    }
    require(cancelled + completed == handles.size(),
            "timed raw workload ended with an unexpected stop/context outcome");
    if (completion) {
        std::cout << std::setprecision(10) << "chunked_throughput_completion backend=" << backend
                  << " C=8 prompt=60000x8 max_output=32768 safety_deadline_seconds=1800"
                  << " pool_tokens=" << engine.memory_summary().kv_capacity
                  << " window=" << sampled.kv_execution_window_tokens
                  << " makespan_seconds=" << makespan_seconds
                  << " observer_elapsed_seconds=" << sampled_seconds << " total_elapsed_seconds="
                  << std::chrono::duration<double>(Clock::now() - started).count() << " prefill="
                  << drained.computed_prefill_tokens - initial.computed_prefill_tokens
                  << " decode=" << decode
                  << " decode_tok_s=" << static_cast<double>(decode) / makespan_seconds
                  << " returned_tokens=" << emitted
                  << " output_tok_s=" << static_cast<double>(emitted) / makespan_seconds
                  << " avg_decode_batch=" << (rounds ? static_cast<double>(rows) / rounds : 0)
                  << " sampled_peak_active_requests=" << sampled_peak_active
                  << " sampled_peak_decode_ready=" << sampled_peak_decode_ready
                  << " peak_actual_decode_batch=unavailable"
                  << " started=" << started_requests
                  << " avg_ttft=" << (started_requests ? ttft / started_requests : 0)
                  << " cancelled=" << cancelled << " output_limit=" << completed
                  << " drain_seconds=" << drain_seconds
                  << " drained_active=" << drained.running_requests
                  << " extensions=" << drained.kv_window_extensions
                  << " suspensions=" << drained.kv_live_suspensions
                  << " resumes=" << drained.kv_live_resumes
                  << " ram_save_seconds=" << drained.kv_ram_save_seconds
                  << " ram_load_seconds=" << drained.kv_ram_load_seconds
                  << " censored=" << (completed != handles.size() || makespan_seconds >= 1800)
                  << '\n';
        for (std::size_t i = 0; i < results.size(); ++i) {
            std::cout << "client index=" << i << " completed_seconds="
                      << std::chrono::duration<double>(client_completed[i] - started).count()
                      << " generated=" << results[i].generated_token_ids.size()
                      << " ttft=" << results[i].timings.first_token_seconds << '\n';
        }
        require(completed == handles.size() && makespan_seconds < 1800,
                "completion workload failed/censored at safety deadline; no completion throughput "
                "claim");
        for (const auto& result : results) {
            require(result.generated_token_ids.size() == 32768 &&
                        result.finish_reason == ninfer::FinishReason::OutputLimit,
                    "completion workload stopped before the original 32768-token ceiling");
        }
        require(drained.computed_prefill_tokens - initial.computed_prefill_tokens == 8ULL * 60000,
                "completion workload did not evaluate all eight original cold prompts");
        require(decode == 8ULL * (32768 - 1) && emitted == 8ULL * 32768,
                "completion workload committed token accounting differs from exact eight-request "
                "ceiling");
        return 0;
    }

    std::cout << std::setprecision(8) << "chunked_throughput backend=" << backend
              << " C=8 prompt=60000x8 max_output=32768"
              << " pool_tokens=" << engine.memory_summary().kv_capacity
              << " window=" << sampled.kv_execution_window_tokens << " seconds=" << sampled_seconds
              << " decode=" << decode
              << " decode_tok_s=" << static_cast<double>(decode) / sampled_seconds
              << " avg_batch=" << (rounds ? static_cast<double>(rows) / rounds : 0)
              << " second_half_tok_s="
              << static_cast<double>(steady_decode) / (sampled_seconds - midpoint_seconds)
              << " second_half_batch="
              << (steady_rounds ? static_cast<double>(steady_rows) / steady_rounds : 0)
              << " started=" << started_requests
              << " avg_ttft=" << (started_requests ? ttft / started_requests : 0)
              << " returned_tokens=" << emitted << " cancelled=" << cancelled
              << " output_limit=" << completed << " drain_seconds=" << drain_seconds
              << " drained_active=" << drained.running_requests
              << " extensions=" << sampled.kv_window_extensions
              << " suspensions=" << sampled.kv_live_suspensions
              << " resumes=" << sampled.kv_live_resumes << '\n';
    return 0;
}


} // namespace

int main(int argc, char** argv) {
    if ((argc != 2 && argc != 4) || std::string_view(argv[1]) == "--help") {
        std::cout
            << "usage: ninfer_r9700_engine_cache_cancel_qual ARTIFACT.ninfer\n"
               "       ninfer_r9700_engine_cache_cancel_qual ARTIFACT.ninfer --disk-dir "
               "NEW_DIRECTORY\n"
               "       ninfer_r9700_engine_cache_cancel_qual ARTIFACT.ninfer --system-prompts "
               "none|dflash|mtp\n"
               "       ninfer_r9700_engine_cache_cancel_qual ARTIFACT.ninfer --chunked-pressure "
               "none|dflash\n"
               "       ninfer_r9700_engine_cache_cancel_qual ARTIFACT.ninfer "
               "--chunked-reasoning-pressure "
               "none|dflash\n"
               "       ninfer_r9700_engine_cache_cancel_qual ARTIFACT.ninfer --chunked-throughput "
               "none|dflash\n"
               "       ninfer_r9700_engine_cache_cancel_qual ARTIFACT.ninfer "
               "--chunked-throughput-completion "
               "none|dflash\n"
               "Manual real-model check: ordinary greedy C1, 16-token requests, "
               "8 GiB RAM cache.\n"
               "Cancellation starts after the first nonempty stream delta; "
               "its exact GPU phase is not forced.\n"
               "Also checks disposable long-to-short physical-page recycling against cold output.\n"
               "Disk mode instead checks durable restart/HostDisk reuse against cold output;\n"
               "uses 8 GiB RAM + 1 GiB SSD and retains the new directory.\n";
        return argc == 2 ? 0 : 2;
    }
    try {
        if (argc == 4) {
            if (std::string_view(argv[2]) == "--chunked-reasoning-pressure") {
                return qualify_chunked_pressure(argv[1], argv[3], true);
            }
            if (std::string_view(argv[2]) == "--chunked-throughput-completion") {
                return qualify_chunked_throughput(argv[1], argv[3], true);
            }
            if (std::string_view(argv[2]) == "--chunked-throughput") {
                return qualify_chunked_throughput(argv[1], argv[3]);
            }
            if (std::string_view(argv[2]) == "--chunked-pressure") {
                return qualify_chunked_pressure(argv[1], argv[3]);
            }
            if (std::string_view(argv[2]) == "--system-prompts") {
                return qualify_system_prompts(argv[1], argv[3]);
            }
            require(std::string_view(argv[2]) == "--disk-dir", "expected --disk-dir NEW_DIRECTORY");
            return qualify_disk_restart(argv[1], argv[3]);
        }
        ninfer::EngineOptions options;
        options.artifact_path         = argv[1];
        options.max_context           = 256;
        options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(256);
        options.max_concurrency       = 1;
        options.kv_ram_capacity_bytes = 8ULL << 30;
        options.speculative.backend   = ninfer::SpeculativeBackend::None;
        ninfer::Engine engine(std::move(options));
        const auto prompt =
            prompt_tokens(engine, "List the integers from one to twenty, in order.");
        const auto other = prompt_tokens(engine, "Describe how a sailboat moves across a lake.");

        CancelAfterOutput sink;
        const ninfer::CancellationView cancellation([&] { return sink.requested; });
        const auto cancelled =
            engine.generate(engine.prepare_tokens(prompt), request_options(), &sink, cancellation);
        require(sink.requested, "generation did not produce a nonempty stream delta");
        require(cancelled.finish_reason == ninfer::FinishReason::Cancelled,
                "cancellation was not observed before completion; phase coverage is inconclusive");
        require(!cancelled.generated_token_ids.empty(), "cancellation published no token");

        // Resubmit exactly the published prefix. Both a retained boundary and a cleared
        // in-flight ordinary lane must support the same continuation as cold prefill.
        const auto resumed_prompt = continuation(prompt, cancelled);
        const auto resumed =
            engine.generate(engine.prepare_tokens(resumed_prompt), request_options());
        const auto cold_resumed =
            engine.generate(engine.prepare_tokens(resumed_prompt), request_options(false));
        require_same_output(resumed, cold_resumed);
        const auto cold_original =
            engine.generate(engine.prepare_tokens(prompt), request_options(false));
        require(cancelled.generated_token_ids.size() <= cold_original.generated_token_ids.size() &&
                    std::equal(cancelled.generated_token_ids.begin(),
                               cancelled.generated_token_ids.end(),
                               cold_original.generated_token_ids.begin()),
                "cancelled request published a non-prefix of cold greedy output");

        // One lane forces the unrelated request to evict A into RAM. Extending A's
        // committed token ledger then requires a RAM restore, not a resident hit.
        const auto saved = engine.generate(engine.prepare_tokens(prompt), request_options(false));
        const auto ram_prompt = continuation(prompt, saved);
        const auto before     = engine.runtime_stats();
        (void)engine.generate(engine.prepare_tokens(other), request_options(false));
        const auto restored = engine.generate(engine.prepare_tokens(ram_prompt), request_options());
        const auto after    = engine.runtime_stats();
        require(restored.prefix_reuse_source == ninfer::PrefixReuseSource::HostRam &&
                    restored.reused_prompt_tokens > 0,
                "continuation did not exercise host-RAM restore");
        require(after.kv_ram_captures > before.kv_ram_captures &&
                    after.kv_ram_restores > before.kv_ram_restores,
                "RAM capture/restore counters did not advance");
        const auto cold_ram =
            engine.generate(engine.prepare_tokens(ram_prompt), request_options(false));
        require_same_output(restored, cold_ram);
        qualify_recycled_pages(engine);
        std::cout << "engine_cache_cancel: PASS ordinary C1 greedy; cancelled_tokens="
                  << cancelled.generated_token_ids.size()
                  << " cancel_resume_reused=" << resumed.reused_prompt_tokens
                  << " RAM_reused=" << restored.reused_prompt_tokens
                  << "; cancellation phase was not forced\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "engine_cache_cancel: FAIL: " << error.what() << '\n';
        return 1;
    }
}
