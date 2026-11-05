#include "ninfer/engine.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

// Manual real-artifact check: the last of twelve seeded requests survives the
// other eleven, exercising compact ingress and deferred-fold record remapping.
int main(int argc, char** argv) {
    try {
        if (argc != 3 ||
            (std::string_view(argv[2]) != "graph" && std::string_view(argv[2]) != "eager"))
            throw std::invalid_argument("usage: dflash_concurrency_qual ARTIFACT graph|eager");
        constexpr std::uint32_t concurrency = 12;
        ninfer::EngineOptions options;
        options.artifact_path              = argv[1];
        options.max_context                = 4096;
        options.kv_capacity                = ninfer::KvCapacityPolicy::automatic();
        options.max_concurrency            = concurrency;
        options.prefill_chunk              = 2048;
        options.mixed_forward              = 0;
        options.use_device_graph           = std::string_view(argv[2]) == "graph";
        options.speculative.backend        = ninfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens   = 7;
        options.speculative.adaptive_draft = false;
        options.speculative.proposal_head  = ninfer::ProposalHead::Optimized;
        ninfer::Engine engine(options);
        ninfer::RequestOptions request;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.requested_output_tokens = 1;
        request.stop.include_model_defaults       = false;
        request.output.raw                        = true;
        request.output.preserve_special_tokens    = true;
        std::vector<ninfer::PreparedPrompt> prompts;
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            std::vector<ninfer::TokenId> ids(512, 198);
            ids.front()     = static_cast<ninfer::TokenId>(1000U + lane);
            const auto seed = engine.generate(engine.prepare_tokens(ids, true), request);
            ids.insert(ids.end(), seed.generated_token_ids.begin(), seed.generated_token_ids.end());
            prompts.push_back(engine.prepare_tokens(std::move(ids), true));
        }
        std::vector<ninfer::GenerationHandle> handles;
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            request.execution.requested_output_tokens = lane + 1U == concurrency ? 512U : 32U;
            handles.push_back(engine.submit(std::move(prompts[lane]), request,
                                            ninfer::OutputDelivery::TerminalOnly));
        }
        const auto full_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        bool full_batch_observed = false;
        while (std::chrono::steady_clock::now() < full_deadline) {
            if (engine.runtime_stats().decode_ready_requests == concurrency) {
                full_batch_observed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!full_batch_observed)
            throw std::runtime_error("twelve decode-ready requests were not observed together");
        nlohmann::json tokens = nlohmann::json::array();
        for (std::uint32_t lane = 0; lane + 1U < concurrency; ++lane) {
            const auto result = handles[lane].wait();
            if (result.generated_token_ids.size() != 32U || result.reused_prompt_tokens < 512U)
                throw std::runtime_error("short lane did not complete its seeded continuation");
            tokens.push_back(result.generated_token_ids);
        }
        // Request completion publication precedes the scheduler's settled boundary.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        auto stats          = engine.runtime_stats();
        while (stats.running_requests != 1U && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            stats = engine.runtime_stats();
        }
        if (stats.running_requests != 1U)
            throw std::runtime_error("last request did not survive the compacted batch alone");
        const auto result = handles.back().wait();
        if (result.generated_token_ids.size() != 512U || result.reused_prompt_tokens < 512U)
            throw std::runtime_error("last lane did not complete its seeded continuation");
        tokens.push_back(result.generated_token_ids);
        std::cout << nlohmann::json({{"status", "PASS"},
                                     {"concurrency", concurrency},
                                     {"mode", argv[2]},
                                     {"fixed_drafts", 7},
                                     {"full_batch_observed", full_batch_observed},
                                     {"generated_token_ids_by_lane", tokens}})
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dflash_concurrency: FAIL: " << error.what() << '\n';
        return 1;
    }
}
