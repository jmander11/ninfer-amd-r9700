#include "serve/score_schema.h"

#include <cmath>
#include <iostream>
#include <limits>

using Json = nlohmann::json;
using namespace ninfer::serve;

int main() {
    int failures     = 0;
    const auto check = [&](bool ok, const char* message) {
        if (!ok) {
            ++failures;
            std::cerr << message << '\n';
        }
    };
    const Json body{{"model", "local"},
                    {"messages", Json::array({{{"role", "user"}, {"content", "Pick a color"}}})},
                    {"candidates", {"red", "blue"}},
                    {"preserve_thinking", true}};
    const auto parsed = parse_candidate_score_request(body);
    check(parsed.candidates == std::vector<std::string>{"red", "blue"} &&
              parsed.context.messages.size() == 1 && parsed.context.preserve_thinking == true,
          "candidate order or shared history semantics changed");
    for (const Json& candidates : {Json::array(), Json::array({""}), Json::array({42}), Json("red"),
                                   Json(std::vector<std::string>(17, "x"))}) {
        auto invalid          = body;
        invalid["candidates"] = candidates;
        bool rejected         = false;
        try {
            (void)parse_candidate_score_request(invalid);
        } catch (const ApiException& error) { rejected = error.error().param == "candidates"; }
        check(rejected, "invalid candidate list admitted");
    }
    for (const auto* field : {"stream", "temperature", "tools", "response_format", "max_tokens"}) {
        auto invalid   = body;
        invalid[field] = true;
        bool rejected  = false;
        try {
            (void)parse_candidate_score_request(invalid);
        } catch (const ApiException&) { rejected = true; }
        check(rejected, "generation control admitted to score endpoint");
    }
    auto media                      = body;
    media["messages"][0]["content"] = Json::array(
        {Json{{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,AA=="}}}}});
    bool rejected_media = false;
    try {
        (void)parse_candidate_score_request(media);
    } catch (const ApiException&) { rejected_media = true; }
    check(rejected_media, "media admitted to text scoring");

    std::vector<ninfer::ScoreResult> scores(2);
    scores[0].prompt_tokens = 7;
    scores[0].tokens_scored = 2;
    scores[0].sum_nll       = 3;
    scores[0].mean_nll      = 1.5;
    scores[1].prompt_tokens = 8;
    scores[1].tokens_scored = 3;
    scores[1].sum_nll       = 3.75;
    scores[1].mean_nll      = 1.25;
    const auto response     = make_candidate_score_response(parsed, scores);
    check(response["object"] == "candidate_scores" && response["model"] == "local" &&
              response["scoring"] == "rendered_assistant" && response["schedule"] == "prefill",
          "score response identity is incorrect");
    check(response["data"][0]["log_probability"] == -3 && response["data"][1]["mean_nll"] == 1.25 &&
              response["data"][1]["index"] == 1 && response["data"][1]["text"] == "blue" &&
              response["usage"]["prompt_tokens"] == 15 && response["usage"]["scored_tokens"] == 5,
          "score sign, denominator, order or aggregate usage changed");
    for (int mode = 0; mode < 3; ++mode) {
        auto bad = scores;
        if (mode == 0) { bad[1].non_finite = 1; }
        if (mode == 1) { bad[1].tokens_scored = 0; }
        if (mode == 2) { bad[1].sum_nll = std::numeric_limits<double>::infinity(); }
        bool rejected = false;
        try {
            (void)make_candidate_score_response(parsed, bad);
        } catch (const ApiException& error) { rejected = error.error().status == 500; }
        check(rejected, "nonfinite or empty candidate score silently serialized");
    }
    if (!failures) { std::cout << "ok\n"; }
    return failures ? 1 : 0;
}
