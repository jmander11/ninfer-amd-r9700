#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <array>
#include <barrier>
#include <stdexcept>
#include <string_view>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3 && !(argc == 4 && std::string_view(argv[3]) == "--history")) {
        std::cerr
            << "usage: concurrent_long_prompt_qual ARTIFACT NEW_CACHE_DIRECTORY [--history]\n";
        return 2;
    }
    try {
        if (!std::filesystem::create_directory(argv[2]))
            throw std::runtime_error("fresh cache directory required");
        ninfer::EngineOptions config;
        config.artifact_path              = argv[1];
        config.max_context                = 262144;
        config.max_concurrency            = 6;
        config.kv_capacity                = ninfer::KvCapacityPolicy::automatic();
        config.prefill_chunk              = 2048;
        config.kv_ram_capacity_bytes      = 12ULL << 30;
        config.kv_disk_capacity_bytes     = 25ULL << 30;
        config.kv_disk_location           = argv[2];
        config.pending_timeout_ms         = 900000;
        config.speculative.backend        = ninfer::SpeculativeBackend::DFlash;
        config.speculative.draft_tokens   = 7;
        config.speculative.adaptive_draft = true;
        config.speculative.proposal_head  = ninfer::ProposalHead::Optimized;
        config.generation_recovery        = false;
        std::cerr << "loading\n" << std::flush;
        ninfer::Engine engine(config);
        const auto memory = engine.memory_summary();
        std::cerr << "loaded kv_capacity=" << memory.kv_capacity
                  << " graph_observed=" << memory.device_graph_observed_bytes
                  << " graph_allowance=" << memory.device_graph_allowance_bytes << '\n'
                  << std::flush;
        const bool history = argc == 4;
        std::barrier warm_clients(4);
        std::mutex log_mutex;
        std::atomic<unsigned> finished{0}, errors{0};
        std::atomic<unsigned> reused_requests{0};
        std::atomic<bool> done{false};
        std::jthread observer([&] {
            auto previous     = engine.runtime_stats();
            unsigned stagnant = 0;
            while (!done.load()) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                auto now = engine.runtime_stats();
                stagnant = (now.computed_prefill_tokens == previous.computed_prefill_tokens &&
                            now.committed_decode_tokens == previous.committed_decode_tokens)
                               ? stagnant + 5
                               : 0;
                {
                    std::lock_guard guard(log_mutex);
                    std::cerr << "progress pp=" << now.computed_prefill_tokens
                              << " tg=" << now.committed_decode_tokens
                              << " rounds=" << now.decode_rounds
                              << " rows=" << now.decode_row_rounds
                              << " active=" << now.running_requests
                              << " pf=" << now.prefilling_requests
                              << " dr=" << now.decode_ready_requests
                              << " wait=" << now.waiting_requests
                              << " ram=" << now.kv_ram_used_bytes
                              << " disk=" << now.kv_disk_used_bytes << " stale_s=" << stagnant
                              << '\n'
                              << std::flush;
                }
                if (stagnant >= 45 && now.running_requests) {
                    std::cerr << "STALL\n" << std::flush;
                }
                previous = now;
            }
        });
        constexpr std::array<std::uint32_t, 4> lengths{100739, 117772, 193485, 145000};
        std::vector<std::jthread> clients;
        clients.reserve(lengths.size());
        for (unsigned id = 0; id < lengths.size(); ++id)
            clients.emplace_back([&, id] {
                bool crossed_barrier = false;
                try {
                    ninfer::PromptInput chat;
                    {
                        chat.options.enable_thinking   = history;
                        chat.options.preserve_thinking = true;
                        chat.options.require_tool_call = !history;
                        for (unsigned tool = 0; tool < 9; ++tool)
                            chat.options.tool_jsons.push_back(
                                "{\"type\":\"function\",\"function\":{\"name\":\"inspect_file_" +
                                std::to_string(tool) +
                                "\",\"description\":\"Inspect a local text "
                                "file.\",\"parameters\":{\"type\":\"object\",\"properties\":{"
                                "\"path\":{\"type\":\"string\",\"enum\":[\"/tmp/"
                                "ninfer-check\"]},\"limit\":{\"type\":\"integer\",\"enum\":[1]}},"
                                "\"required\":[\"path\",\"limit\"],\"additionalProperties\":false}}"
                                "}");
                        const std::string line =
                            "The cedar forest follows the river. This archive contains "
                            "observations about trees, water, and the surrounding hills.\n";
                        const auto archive = [&](std::uint32_t lines) {
                            std::string body = "Archive client " + std::to_string(id) + ".\n";
                            for (std::uint32_t n = 0; n < lines; ++n) body += line;
                            return body;
                        };
                        const auto user = [&](std::string text) {
                            ninfer::ChatMessage msg;
                            msg.role = ninfer::ChatRole::User;
                            msg.parts.push_back(ninfer::MessagePart{.text = std::move(text)});
                            chat.messages.push_back(std::move(msg));
                        };
                        if (history) {
                            // Historical assistant boundaries permit a checkpoint around 78% of
                            // each prompt, followed by a large final tool result like the incident.
                            constexpr std::array<unsigned, 4> turns{24, 36, 50, 28};
                            const std::uint32_t total   = lengths[id] / 24;
                            const std::uint32_t earlier = total * 78 / 100;
                            for (unsigned n = 0; n < turns[id]; ++n) {
                                user("Inspect the next archive, then call inspect_file_0 with path "
                                     "/tmp/ninfer-check and limit 1.");
                                ninfer::ChatMessage assistant;
                                assistant.role              = ninfer::ChatRole::Assistant;
                                assistant.reasoning_content = "I will inspect the next archive.";
                                ninfer::ToolCall call;
                                call.id   = "history_" + std::to_string(n);
                                call.name = "inspect_file_0";
                                call.arguments_json =
                                    "{\"path\":\"/tmp/ninfer-check\",\"limit\":1}";
                                assistant.tool_calls.push_back(call);
                                chat.messages.push_back(std::move(assistant));
                                ninfer::ChatMessage result;
                                result.role         = ninfer::ChatRole::Tool;
                                result.tool_call_id = call.id;
                                result.parts.push_back(ninfer::MessagePart{
                                    .text =
                                        archive(n + 1 == turns[id] ? total - earlier
                                                                   : earlier / (turns[id] - 1))});
                                chat.messages.push_back(std::move(result));
                            }
                            user("Call inspect_file_0 with path /tmp/ninfer-check and limit 1 now. "
                                 "Do not repeat the archive.");
                        } else {
                            user(archive(lengths[id] / 24) + "\nCall inspect_file_0 with path "
                                                             "/tmp/ninfer-check and limit 1 now.");
                        }
                    }
                    for (unsigned turn = 0; turn < 3; ++turn) {
                        ninfer::RequestOptions opts;
                        opts.execution.requested_output_tokens    = 65536;
                        opts.execution.sampling.temperature       = 1.5F;
                        opts.execution.sampling.p_less            = true;
                        opts.execution.sampling.seed              = 0x173a + id * 17 + turn;
                        opts.execution.capture_context_checkpoint = true;
                        opts.stop.include_model_defaults          = true;
                        {
                            std::lock_guard guard(log_mutex);
                            std::cerr << "submit client=" << id << " turn=" << turn
                                      << " target_tokens=" << lengths[id] << '\n'
                                      << std::flush;
                        }
                        auto prepared = engine.prepare(chat);
                        if (turn == 2) {
                            // Release warm followups together so short tool calls still exercise
                            // B>1 decode and the return to B1, rather than finishing during
                            // prefill.
                            warm_clients.arrive_and_wait();
                            crossed_barrier = true;
                        }
                        {
                            std::lock_guard guard(log_mutex);
                            std::cerr << "prepared client=" << id << " turn=" << turn
                                      << " actual_tokens=" << prepared.summary().prompt_tokens
                                      << '\n'
                                      << std::flush;
                        }
                        auto r = history ? engine
                                               .submit(std::move(prepared), opts,
                                                       ninfer::OutputDelivery::Streaming)
                                               .wait()
                                         : engine.generate(std::move(prepared), opts);
                        if (r.generated_token_ids.empty())
                            throw std::runtime_error("no generated tokens");
                        if (r.reused_prompt_tokens != 0) ++reused_requests;
                        {
                            std::lock_guard guard(log_mutex);
                            std::cerr << "complete client=" << id << " turn=" << turn
                                      << " output=" << r.generated_token_ids.size()
                                      << " reuse=" << r.reused_prompt_tokens << '\n'
                                      << std::flush;
                        }
                        {
                            if (r.tool_calls.empty())
                                throw std::runtime_error("required tool call missing");
                            ninfer::ChatMessage assistant;
                            assistant.role = ninfer::ChatRole::Assistant;
                            if (!r.content.empty())
                                assistant.parts.push_back(ninfer::MessagePart{.text = r.content});
                            assistant.reasoning_content = r.reasoning;
                            assistant.tool_calls        = r.tool_calls;
                            chat.messages.push_back(std::move(assistant));
                            for (const auto& call : r.tool_calls) {
                                const auto arguments = nlohmann::json::parse(call.arguments_json);
                                bool known_tool      = false;
                                for (unsigned tool = 0; tool < 9; ++tool)
                                    known_tool |=
                                        call.name == "inspect_file_" + std::to_string(tool);
                                if (!known_tool ||
                                    arguments !=
                                        nlohmann::json{{"path", "/tmp/ninfer-check"}, {"limit", 1}})
                                    throw std::runtime_error("invalid generated tool call");
                                ninfer::ChatMessage result;
                                result.role         = ninfer::ChatRole::Tool;
                                result.tool_call_id = call.id;
                                result.parts.push_back(ninfer::MessagePart{
                                    .text = "File contains one line: cedar river."});
                                chat.messages.push_back(std::move(result));
                            }
                            ninfer::ChatMessage user;
                            user.role = ninfer::ChatRole::User;
                            user.parts.push_back(ninfer::MessagePart{
                                .text = "Call inspect_file_0 with path /tmp/ninfer-check and limit "
                                        "1 again."});
                            chat.messages.push_back(std::move(user));
                        }
                    }
                } catch (const std::exception& e) {
                    if (!crossed_barrier) warm_clients.arrive_and_drop();
                    std::lock_guard guard(log_mutex);
                    std::cerr << "client failed " << id << ' ' << e.what() << '\n' << std::flush;
                    ++errors;
                }
                ++finished;
            });
        clients.clear();
        done = true;
        observer.join();
        const auto stats = engine.runtime_stats();
        if (stats.decode_row_rounds <= stats.decode_rounds) {
            std::cerr << "no concurrent decode batch observed\n";
            ++errors;
        }
        if (reused_requests == 0 || stats.kv_ram_captures == 0 || stats.kv_disk_captures == 0 ||
            stats.kv_ram_restores + stats.kv_disk_restores == 0 || stats.kv_cache_fallbacks != 0) {
            std::cerr << "cache capture/restore coverage missing or restore failed\n";
            ++errors;
        }
        std::cout << "finished=" << finished << " errors=" << errors
                  << " rounds=" << stats.decode_rounds << " rows=" << stats.decode_row_rounds
                  << " ram=" << stats.kv_ram_used_bytes << " disk=" << stats.kv_disk_used_bytes
                  << " reused_requests=" << reused_requests
                  << " ram_captures=" << stats.kv_ram_captures
                  << " disk_captures=" << stats.kv_disk_captures
                  << " ram_restores=" << stats.kv_ram_restores
                  << " disk_restores=" << stats.kv_disk_restores << '\n';
        return errors.load() ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
