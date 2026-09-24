#include "ninfer/engine.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::RequestOptions request_options(bool reuse = true) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 16;
    options.execution.allow_prefix_reuse = reuse;
    options.execution.sampling.temperature = 0.0F;
    options.execution.sampling.top_k = 0;
    options.execution.sampling.top_p = 1.0F;
    options.execution.sampling.min_p = 0.0F;
    options.execution.sampling.presence_penalty = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.execution.sampling.seed = 1;
    options.stop.include_model_defaults = false;
    options.output.raw = true;
    options.output.preserve_special_tokens = true;
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
    prefix.insert(prefix.end(), result.generated_token_ids.begin(), result.generated_token_ids.end());
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

int qualify_disk_restart(const std::filesystem::path& artifact,
                         const std::filesystem::path& directory) {
    require(std::filesystem::is_regular_file(artifact), "explicit artifact is not a regular file");
    // The result directory is retained as evidence, including on failure. Never open
    // an existing cache or remove caller-owned data during qualification.
    require(std::filesystem::create_directory(directory),
            "SSD qualification requires a new, nonexistent output directory");
    const auto options = [&] {
        ninfer::EngineOptions value;
        value.artifact_path = artifact;
        value.max_context = 256;
        value.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256);
        value.max_concurrency = 1;
        value.kv_ram_capacity_bytes = 1ULL << 30;
        value.kv_disk_capacity_bytes = 1ULL << 30;
        value.kv_disk_location = directory;
        value.speculative.backend = ninfer::SpeculativeBackend::None;
        return value;
    };
    std::vector<ninfer::TokenId> continued;
    {
        ninfer::Engine engine(options());
        const auto prompt = prompt_tokens(engine, "List the integers from one to twenty, in order.");
        const auto other = prompt_tokens(engine, "Describe how a sailboat moves across a lake.");
        const auto saved = engine.generate(engine.prepare_tokens(prompt), request_options(false));
        require(saved.generated_token_ids.size() == 16, "SSD seed did not produce 16 tokens");
        continued = continuation(prompt, saved);
        const auto before = engine.runtime_stats();
        (void)engine.generate(engine.prepare_tokens(other), request_options(false));
        require(engine.runtime_stats().kv_ram_captures > before.kv_ram_captures,
                "unrelated request did not capture A into RAM");
        // Public Engine destruction drains the scheduler and persists nondurable RAM
        // plus retained lanes. No private spill API or timing sleep is required.
    }
    {
        ninfer::Engine engine(options());
        const auto before = engine.runtime_stats();
        require(before.kv_disk_entry_count > 0, "restart did not recover durable SSD entries");
        require(before.kv_ram_restores == 0, "fresh Engine already restored RAM state");
        const auto restored = engine.generate(engine.prepare_tokens(continued), request_options());
        require(restored.prefix_reuse_source == ninfer::PrefixReuseSource::HostDisk &&
                    restored.reused_prompt_tokens > 0,
                "fresh Engine continuation did not exercise SSD restore");
        require(engine.runtime_stats().kv_disk_restores > before.kv_disk_restores,
                "SSD restore counter did not advance");
        const auto cold = engine.generate(engine.prepare_tokens(continued), request_options(false));
        require_same_output(restored, cold);
        std::cout << "engine_cache_disk: PASS ordinary C1 greedy; SSD_reused="
                  << restored.reused_prompt_tokens << " output_tokens="
                  << restored.generated_token_ids.size() << " retained_directory="
                  << directory.string() << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if ((argc != 2 && argc != 4) || std::string_view(argv[1]) == "--help") {
        std::cout << "usage: ninfer_r9700_engine_cache_cancel_qual ARTIFACT.ninfer\n"
                     "       ninfer_r9700_engine_cache_cancel_qual ARTIFACT.ninfer --disk-dir NEW_DIRECTORY\n"
                     "Manual real-model check: ordinary greedy C1, 16-token requests, "
                     "1 GiB RAM cache.\n"
                     "Cancellation starts after the first nonempty stream delta; "
                     "its exact GPU phase is not forced.\n"
                     "Disk mode instead checks durable restart/HostDisk reuse against cold output;\n"
                     "uses 1 GiB RAM + 1 GiB SSD and retains the new directory.\n";
        return argc == 2 ? 0 : 2;
    }
    try {
        if (argc == 4) {
            require(std::string_view(argv[2]) == "--disk-dir", "expected --disk-dir NEW_DIRECTORY");
            return qualify_disk_restart(argv[1], argv[3]);
        }
        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.max_context = 256;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256);
        options.max_concurrency = 1;
        options.kv_ram_capacity_bytes = 1ULL << 30;
        options.speculative.backend = ninfer::SpeculativeBackend::None;
        ninfer::Engine engine(std::move(options));
        const auto prompt = prompt_tokens(engine, "List the integers from one to twenty, in order.");
        const auto other = prompt_tokens(engine, "Describe how a sailboat moves across a lake.");

        CancelAfterOutput sink;
        const ninfer::CancellationView cancellation([&] { return sink.requested; });
        const auto cancelled = engine.generate(engine.prepare_tokens(prompt), request_options(),
                                                &sink, cancellation);
        require(sink.requested, "generation did not produce a nonempty stream delta");
        require(cancelled.finish_reason == ninfer::FinishReason::Cancelled,
                "cancellation was not observed before completion; phase coverage is inconclusive");
        require(!cancelled.generated_token_ids.empty(), "cancellation published no token");

        // Resubmit exactly the published prefix. Both a retained boundary and a cleared
        // in-flight ordinary lane must support the same continuation as cold prefill.
        const auto resumed_prompt = continuation(prompt, cancelled);
        const auto resumed = engine.generate(engine.prepare_tokens(resumed_prompt), request_options());
        const auto cold_resumed = engine.generate(engine.prepare_tokens(resumed_prompt),
                                                   request_options(false));
        require_same_output(resumed, cold_resumed);
        const auto cold_original = engine.generate(engine.prepare_tokens(prompt), request_options(false));
        require(cancelled.generated_token_ids.size() <= cold_original.generated_token_ids.size() &&
                    std::equal(cancelled.generated_token_ids.begin(), cancelled.generated_token_ids.end(),
                               cold_original.generated_token_ids.begin()),
                "cancelled request published a non-prefix of cold greedy output");

        // One lane forces the unrelated request to evict A into RAM. Extending A's
        // committed token ledger then requires a RAM restore, not a resident hit.
        const auto saved = engine.generate(engine.prepare_tokens(prompt), request_options(false));
        const auto ram_prompt = continuation(prompt, saved);
        const auto before = engine.runtime_stats();
        (void)engine.generate(engine.prepare_tokens(other), request_options(false));
        const auto restored = engine.generate(engine.prepare_tokens(ram_prompt), request_options());
        const auto after = engine.runtime_stats();
        require(restored.prefix_reuse_source == ninfer::PrefixReuseSource::HostRam &&
                    restored.reused_prompt_tokens > 0,
                "continuation did not exercise host-RAM restore");
        require(after.kv_ram_captures > before.kv_ram_captures &&
                    after.kv_ram_restores > before.kv_ram_restores,
                "RAM capture/restore counters did not advance");
        const auto cold_ram = engine.generate(engine.prepare_tokens(ram_prompt), request_options(false));
        require_same_output(restored, cold_ram);
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
