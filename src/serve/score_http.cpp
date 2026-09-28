#include "serve/http_server.h"
#include "serve/openai_schema.h"
#include "serve/score_schema.h"

namespace ninfer::serve {

void HttpServer::handle_score(const httplib::Request& request, httplib::Response& response) {
    try {
        nlohmann::json body;
        try { body = nlohmann::json::parse(request.body); }
        catch (const nlohmann::json::exception&) {
            ApiError error;
            error.message = "request body is not valid JSON";
            throw ApiException(std::move(error));
        }
        const auto parsed = parse_candidate_score_request(body);
        auto scores = service_->score_candidates(parsed, [&request] {
            return request.is_connection_alive && !request.is_connection_alive();
        });
        response.set_content(make_candidate_score_response(parsed, scores).dump(), "application/json");
    } catch (const ApiException& exception) { emit_openai_error(response, exception.error()); }
}

} // namespace ninfer::serve
