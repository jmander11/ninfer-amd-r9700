#pragma once

#include "serve/request.h"

#include <nlohmann/json.hpp>

namespace ninfer::serve {

struct CandidateScoreRequest {
    GenerationRequest context;
    std::vector<std::string> candidates;
};

CandidateScoreRequest parse_candidate_score_request(const nlohmann::json& body);
nlohmann::json make_candidate_score_response(const CandidateScoreRequest& request,
                                             const std::vector<ninfer::ScoreResult>& scores);

} // namespace ninfer::serve
