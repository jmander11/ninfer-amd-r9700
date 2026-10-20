// Saved text-only chat continuations through the public Engine, retaining exact generated IDs
// and published T=1 target logprobs. The separate eager decision trace records actual sampler
// inputs; these logprobs alone are deliberately NOT interpreted as p-less probabilities.
#include "ninfer/engine.h"
#include "product/prompt_input/prompt_input.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;

Json read(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("cannot read " + path.string()); }
    return Json::parse(input);
}

Json records(const std::vector<ninfer::TokenLogprob>& values) {
    Json output = Json::array();
    for (const auto& value : values) {
        Json alternatives = Json::array();
        for (const auto& other : value.alternatives()) {
            alternatives.push_back({{"token", other.token}, {"logprob", other.logprob}});
        }
        output.push_back({{"token", value.token},
                          {"logprob", value.logprob},
                          {"alternatives", std::move(alternatives)}});
    }
    return output;
}

Json recovery_records(const ninfer::GenerationRecoveryStats& stats) {
    return {{"attempts", stats.attempts},
            {"framing_repairs", stats.framing_repairs},
            {"discarded_content_tokens", stats.discarded_content_tokens},
            {"discarded_reasoning_tokens", stats.discarded_reasoning_tokens},
            {"discarded_tool_calls", stats.discarded_tool_calls},
            {"cycle_exclusions", stats.cycle_exclusions}};
}

class TraceSink final : public ninfer::OutputSink {
public:
    Json output   = Json::array();
    Json recovery = Json::array();

    void publish(ninfer::OutputDelta delta) override {
        Json record{{"seconds", elapsed()},
                    {"channel", static_cast<int>(delta.channel)},
                    {"text", delta.text},
                    {"logprobs", records(delta.logprobs)}};
        if (delta.tool_call) {
            const auto& call = *delta.tool_call;
            record["tool"]   = {{"kind", static_cast<int>(call.kind)},
                                {"index", call.index},
                                {"name", call.name},
                                {"arguments", call.arguments}};
        }
        output.push_back(std::move(record));
    }

    void recovery_event(const ninfer::RecoveryEvent& event) override {
        recovery.push_back({{"seconds", elapsed()},
                            {"kind", static_cast<int>(event.kind)},
                            {"cause", event.cause},
                            {"attempts", event.attempts},
                            {"framing_repairs", event.framing_repairs},
                            {"discarded_content_tokens", event.discarded_content_tokens},
                            {"generated_tokens", event.generated_tokens},
                            {"remaining_tokens", event.remaining_tokens}});
        if (event.enable_thinking.has_value()) {
            recovery.back()["enable_thinking"] = *event.enable_thinking;
        }
        if (event.sampling) {
            const auto& sampling        = *event.sampling;
            recovery.back()["sampling"] = {{"p_less", sampling.p_less},
                                           {"temperature", sampling.temperature},
                                           {"top_k", sampling.top_k},
                                           {"top_p", sampling.top_p},
                                           {"min_p", sampling.min_p},
                                           {"presence_penalty", sampling.presence_penalty},
                                           {"frequency_penalty", sampling.frequency_penalty},
                                           {"seed", sampling.seed}};
        }
    }
private:
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();

    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::invalid_argument(
                "usage: ninfer_qwen3_8_27b_generation_trace ARTIFACT MANIFEST.json OUTPUT_DIR");
        }
        const std::filesystem::path manifest_path = std::filesystem::absolute(argv[2]);
        const Json manifest                       = read(manifest_path);
        const std::filesystem::path output_root   = argv[3];
        if (!std::filesystem::create_directory(output_root)) {
            throw std::invalid_argument("output directory must not exist");
        }
        const Json& config = manifest.at("engine");
        ninfer::EngineOptions options;
        options.artifact_path       = argv[1];
        options.max_context         = config.value("max_context", 262144U);
        options.max_concurrency     = config.value("max_concurrency", 6U);
        options.kv_capacity         = ninfer::KvCapacityPolicy::automatic();
        options.prefill_chunk       = config.value("prefill_chunk", 2048U);
        options.use_device_graph    = config.value("device_graph", false);
        options.generation_recovery = config.value("recovery", false);
        const std::string backend   = config.value("backend", "dflash");
        if (backend == "dflash") {
            options.speculative.backend        = ninfer::SpeculativeBackend::DFlash;
            options.speculative.draft_tokens   = config.value("draft_tokens", 7U);
            options.speculative.proposal_head  = ninfer::ProposalHead::Optimized;
            options.speculative.adaptive_draft = config.value("adaptive_draft", false);
            if (config.contains("draft_temperature")) {
                options.speculative.dflash_p_less_draft_temperature =
                    config.at("draft_temperature").get<float>();
            }
        } else if (backend != "none") {
            throw std::invalid_argument("diagnostic backend must be dflash or none");
        }
        ninfer::Engine engine(options);
        std::ofstream(output_root / "manifest.json") << manifest.dump(2) << '\n';
        for (const auto& item : manifest.at("requests")) {
            const std::string label = item.at("label");
            if (label.empty() ||
                label.find_first_not_of(
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") !=
                    std::string::npos) {
                throw std::invalid_argument("request label must be a simple file stem");
            }
            const bool token_prefix = item.contains("token_ids_path");
            if (token_prefix == item.contains("path")) {
                throw std::invalid_argument(
                    "request requires exactly one of path or token_ids_path");
            }
            const auto request_path =
                manifest_path.parent_path() /
                item.at(token_prefix ? "token_ids_path" : "path").get<std::string>();
            Json body   = token_prefix ? Json::object() : read(request_path);
            auto prompt = [&] {
                if (token_prefix) {
                    const auto ids = read(request_path);
                    if (!ids.is_array() || ids.empty()) {
                        throw std::invalid_argument(
                            "token_ids_path must contain a nonempty JSON array");
                    }
                    for (const auto& token : ids) {
                        if (!token.is_number_integer() || token < 0 ||
                            token > std::numeric_limits<ninfer::TokenId>::max()) {
                            throw std::invalid_argument(
                                "token prefix requires nonnegative exact int32 IDs");
                        }
                    }
                    return engine.prepare_tokens(ids.get<std::vector<ninfer::TokenId>>());
                }
                if (body.contains("tool_choice") && body.at("tool_choice") != "auto") {
                    throw std::invalid_argument(
                        "this saved-fixture diagnostic requires auto tool choice");
                }
                auto input = ninfer::product::prompt_from_messages(
                    request_path, body.value("enable_thinking", true), false);
                // These saved HTTP fixtures use explicit function parameter objects and auto
                // tool choice. Match OpenAI parse_tools' absent/null strict normalization;
                // the local messages adapter otherwise preserves the serialized definition.
                // The extra field is visible in the chat template, even when false.
                for (auto& serialized : input.options.tool_jsons) {
                    auto tool = Json::parse(serialized);
                    if (tool.value("type", "") != "function" || !tool.contains("function") ||
                        !tool.at("function").is_object() ||
                        !tool.at("function").contains("parameters") ||
                        !tool.at("function").at("parameters").is_object()) {
                        throw std::invalid_argument(
                            "saved-fixture tools require explicit function parameter objects");
                    }
                    auto& function = tool["function"];
                    if (!function.contains("strict") || function.at("strict").is_null()) {
                        function["strict"] = false;
                    } else if (!function.at("strict").is_boolean()) {
                        throw std::invalid_argument("function strict must be a boolean or null");
                    }
                    serialized = tool.dump();
                }
                input.options.preserve_thinking = body.value("chat_template_kwargs", Json::object())
                                                      .value("preserve_thinking", false);
                return engine.prepare(std::move(input));
            }();
            Json evidence{{"label", label},
                          {"request_path", request_path.string()},
                          {"input_kind", token_prefix ? "token_prefix" : "http_fixture"},
                          {"prompt_tokens", std::vector<ninfer::TokenId>(prompt.token_ids().begin(),
                                                                         prompt.token_ids().end())},
                          {"diagnostic_only", true},
                          {"timing_eligible", false}};
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens =
                item.value("max_tokens", body.value("max_tokens", 8192U));
            request.execution.allow_prefix_reuse = false;
            request.execution.cache_write        = false;
            request.execution.sampling.p_less    = item.value("p_less", true);
            request.execution.sampling.temperature =
                item.value("temperature", body.value("temperature", 1.5F));
            request.execution.sampling.seed =
                item.value("seed", body.value("seed", std::uint64_t{20261020}));
            request.output.raw                     = item.value("raw", false);
            request.stop.include_model_defaults    = item.value("use_model_stops", true);
            evidence["raw"]                        = request.output.raw;
            evidence["use_model_stops"]            = request.stop.include_model_defaults;
            request.output.preserve_special_tokens = true;
            request.output.top_logprobs            = 20;
            request.stop.publish_stop_token        = true;
            std::cerr << "started " << label << " prompt=" << prompt.token_ids().size() << '\n';
            const auto start = std::chrono::steady_clock::now();
            TraceSink sink;
            const bool streaming  = item.value("stream", false);
            evidence["streaming"] = streaming;
            try {
                auto handle = engine.submit(std::move(prompt), request,
                                            streaming ? ninfer::OutputDelivery::Streaming
                                                      : ninfer::OutputDelivery::TerminalOnly);
                const auto& sampling = handle.resolved_sampling();
                evidence["sampling"] = {{"p_less", sampling.p_less},
                                        {"temperature", sampling.temperature},
                                        {"seed", sampling.seed},
                                        {"top_k", sampling.top_k},
                                        {"top_p", sampling.top_p},
                                        {"min_p", sampling.min_p},
                                        {"presence_penalty", sampling.presence_penalty},
                                        {"frequency_penalty", sampling.frequency_penalty}};
                const auto result    = handle.wait(&sink);
                evidence["generated_token_ids"] = result.generated_token_ids;
                evidence["content"]             = result.content;
                evidence["reasoning"]           = result.reasoning;
                evidence["finish_reason"]       = static_cast<int>(result.finish_reason);
                evidence["reasoning_tokens"]    = result.reasoning_tokens;
                evidence["recovery_attempts"]   = result.recovery.attempts;
                evidence["recovery"]            = recovery_records(result.recovery);
                evidence["content_logprobs"]    = records(result.content_logprobs);
                evidence["reasoning_logprobs"]  = records(result.reasoning_logprobs);
                evidence["tool_calls"]          = Json::array();
                for (const auto& call : result.tool_calls) {
                    evidence["tool_calls"].push_back(
                        {{"name", call.name}, {"arguments", call.arguments_json}});
                }
                if (!result.generated_token_ids.empty()) {
                    evidence["terminal_token"] = result.generated_token_ids.back();
                    const auto bytes = engine.token_bytes(result.generated_token_ids.back());
                    evidence["terminal_bytes"] =
                        std::vector<std::uint8_t>(bytes.begin(), bytes.end());
                }
            } catch (const ninfer::RequestError& error) {
                evidence["error"]             = error.what();
                evidence["error_kind"]        = static_cast<int>(error.kind());
                evidence["recovery_attempts"] = error.recovery().attempts;
                evidence["recovery"]          = recovery_records(error.recovery());
            }
            evidence["stream_output"]   = std::move(sink.output);
            evidence["recovery_events"] = std::move(sink.recovery);
            evidence["seconds"] =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::ofstream(output_root / (label + ".json")) << evidence.dump(2) << '\n';
            std::cerr << "completed " << label << " seconds=" << evidence["seconds"] << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generation trace: " << error.what() << '\n';
        return 1;
    }
}
