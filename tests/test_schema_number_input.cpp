#include "serve/json_input.h"
#include "serve/generation_service.h"
#include "serve/request.h"
#include "serve/anthropic_schema.h"
#include "serve/openai_schema.h"
#include "serve/responses_schema.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        for (const auto& [prefix, suffix] :
             {std::pair{R"({"response_format":{"json_schema":{"schema":)", "}}}"},
              std::pair{R"({"text":{"format":{"schema":)", "}}}"},
              std::pair{R"({"output_config":{"format":{"schema":)", "}}}"},
              std::pair{R"({"tools":[{"function":{"parameters":)", "}}]}"},
              std::pair{R"({"tools":[{"parameters":)", "}]}"},
              std::pair{R"({"tools":[{"input_schema":)", "}]}"}}) {
            const auto bad = std::string(prefix) + R"({"const":9007199254740993.0})" + suffix;
            bool rejected  = false;
            try {
                (void)ninfer::serve::parse_request_body(bad);
            } catch (const ninfer::serve::ApiException& error) {
                rejected = error.error().status == 400 && error.error().param.ends_with("/const") &&
                           error.error().message.find(error.error().param) != std::string::npos &&
                           error.error().code == (std::string_view(prefix).starts_with("{\"tools\"")
                                                      ? "invalid_tool_schema"
                                                      : "invalid_output_constraint");
            }
            if (!rejected) throw std::runtime_error("request schema silently rounded: " + bad);
            const auto exact = std::string(prefix) + R"({"const":9007199254740993})" + suffix;
            (void)ninfer::serve::parse_request_body(exact);
        }
        (void)ninfer::serve::parse_request_body(R"({"temperature":1.00000000000000001})");
        const std::string schema =
            R"({"type":"object","properties":{"title":{"type":"string"},"tags":{"type":"array"},"people":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}}},"required":["title","tags","people"]})";
        const auto chat = ninfer::serve::parse_chat_completion_request(
            ninfer::serve::parse_request_body(
                R"({"model":"local","messages":[{"role":"user","content":"hello"}],"response_format":{"type":"json_schema","json_schema":{"name":"result","schema":)" +
                schema + "}}}"),
            {});
        if (!chat.output_constraint || chat.output_constraint->source != schema) {
            throw std::runtime_error("Chat response schema lost property order");
        }
        const auto responses = ninfer::serve::parse_responses_request(
            ninfer::serve::parse_request_body(
                R"({"model":"local","input":"hello","text":{"format":{"type":"json_schema","name":"result","schema":)" +
                schema + "}}}"),
            {});
        if (!responses.generation.output_constraint ||
            responses.generation.output_constraint->source != schema ||
            responses.text_format.at("schema").dump() != schema) {
            throw std::runtime_error("Responses schema or echoed format lost property order");
        }
        const auto response =
            ninfer::serve::make_response_object("resp_order", 123, responses, {}, {});
        if (response.body.at("text").at("format").at("schema").dump() != schema) {
            throw std::runtime_error("Responses output envelope lost schema property order");
        }
        const auto response_tools = ninfer::serve::parse_responses_request(
            ninfer::serve::parse_request_body(
                R"({"model":"local","input":"hello","tools":[{"type":"function","name":"f","parameters":)" +
                schema + "}]}"),
            {});
        if (response_tools.generation.tools.size() != 1 ||
            response_tools.generation.tools[0].parameters_json != schema ||
            response_tools.tools.at(0).at("parameters").dump() != schema) {
            throw std::runtime_error("Responses tool schema lost property order");
        }
        const auto tools = ninfer::serve::parse_chat_completion_request(
            ninfer::serve::parse_request_body(
                R"({"model":"local","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"f","parameters":)" +
                schema + "}}]}"),
            {});
        if (tools.tools.size() != 1 || tools.tools[0].parameters_json != schema) {
            throw std::runtime_error("Chat tool schema lost property order");
        }
        const auto anthropic = ninfer::serve::parse_messages_request(
            ninfer::serve::parse_request_body(
                R"({"model":"local","max_tokens":16,"messages":[{"role":"user","content":"hello"}],"tools":[{"name":"f","input_schema":)" +
                schema + "}]}"),
            {});
        if (anthropic.tools.size() != 1 || anthropic.tools[0].parameters_json != schema) {
            throw std::runtime_error("Anthropic tool schema lost property order");
        }
        std::cout << "PASS: HTTP schema numeric-source boundaries and property order\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
