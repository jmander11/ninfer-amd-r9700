#include "serve/score_schema.h"
#include "serve/openai_schema.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::serve {
namespace {
using Json = nlohmann::json;

[[noreturn]] void invalid_score(std::string message, std::string param) {
    ApiError error;
    error.message = std::move(message);
    error.param = std::move(param);
    error.code = "invalid_score_request";
    throw ApiException(std::move(error));
}
} // namespace

CandidateScoreRequest parse_candidate_score_request(const Json& body) {
    if (!body.is_object()) { invalid_score("score request must be an object", "body"); }
    for (auto it = body.begin(); it != body.end(); ++it) {
        if (it.key() != "model" && it.key() != "messages" && it.key() != "candidates" &&
            it.key() != "enable_thinking" && it.key() != "preserve_thinking" &&
            it.key() != "chat_template_kwargs" && it.key() != "reasoning_effort") {
            invalid_score("unsupported score field: " + it.key(), it.key());
        }
    }
    if (!body.contains("candidates") || !body["candidates"].is_array() ||
        body["candidates"].empty() || body["candidates"].size() > 16) {
        invalid_score("candidates must contain 1..16 nonempty strings", "candidates");
    }
    CandidateScoreRequest request;
    for (const auto& candidate : body["candidates"]) {
        if (!candidate.is_string() || candidate.get_ref<const std::string&>().empty()) {
            invalid_score("candidates must contain 1..16 nonempty strings", "candidates");
        }
        request.candidates.push_back(candidate.get<std::string>());
    }
    Json chat = body;
    chat.erase("candidates");
    request.context = parse_chat_completion_request(chat, {});
    if (request.context.media_item_count() != 0) {
        invalid_score("candidate scoring supports text and tool history, not media", "messages");
    }
    return request;
}

Json make_candidate_score_response(const CandidateScoreRequest& request,
                                   const std::vector<ninfer::ScoreResult>& scores) {
    if (scores.size() != request.candidates.size()) {
        throw std::logic_error("candidate score result count mismatch");
    }
    Json entries = Json::array();
    std::uint64_t total_tokens = 0;
    std::uint64_t total_scored = 0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        const auto& score = scores[i];
        if (score.non_finite != 0 || score.tokens_scored == 0 ||
            !std::isfinite(score.sum_nll) || !std::isfinite(score.mean_nll)) {
            ApiError error;
            error.status = 500;
            error.type = "server_error";
            error.code = "invalid_score_result";
            error.message = "candidate scoring produced no finite complete score";
            throw ApiException(std::move(error));
        }
        entries.push_back(Json{{"index", i}, {"text", request.candidates[i]},
            {"prompt_tokens", score.prompt_tokens}, {"scored_tokens", score.tokens_scored},
            {"sum_nll", score.sum_nll}, {"mean_nll", score.mean_nll},
            {"log_probability", -score.sum_nll}, {"score_seconds", score.score_seconds}});
        total_tokens += score.prompt_tokens;
        total_scored += score.tokens_scored;
    }
    return Json{{"object", "candidate_scores"}, {"model", request.context.model},
                {"scoring", "rendered_assistant"}, {"schedule", "prefill"}, {"data", entries},
                {"usage", {{"prompt_tokens", total_tokens}, {"scored_tokens", total_scored}}}};
}
} // namespace ninfer::serve
