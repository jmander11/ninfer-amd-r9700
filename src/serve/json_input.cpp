#include "serve/json_input.h"

#include "serve/request.h"
#include "text/json_input.h"

#include <string>

namespace ninfer::serve {

nlohmann::ordered_json parse_request_body(std::string_view source) {
    const auto parsed = text::parse_json_numbers(source);
    const auto check  = [&](const std::string& path, bool tool = false,
                            const std::string& name = {}) {
        if (const auto location = text::inexact_schema_number(parsed, path)) {
            ApiError error;
            error.status = 400;
            error.message =
                (name.empty() ? "" : "tool " + name + ": ") +
                "schema numeric value cannot be preserved by the JSON representation at " +
                *location;
            error.param = *location;
            error.code  = tool ? "invalid_tool_schema" : "invalid_output_constraint";
            throw ApiException(std::move(error));
        }
    };
    for (const char* path : {"/response_format/json_schema/schema", "/text/format/schema",
                             "/output_config/format/schema"}) {
        check(path);
    }
    const auto tools = [&](auto&& self, const nlohmann::ordered_json& values,
                           const std::string& path) -> void {
        if (!values.is_array()) return;
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto& tool = values[index];
            if (!tool.is_object()) continue;
            const auto prefix = path + '/' + std::to_string(index);
            if (tool.contains("tools") && tool.contains("type") && tool["type"] == "namespace") {
                self(self, tool["tools"], prefix + "/tools");
            } else if (tool.contains("function") && tool["function"].is_object()) {
                const auto& function = tool["function"];
                const auto name      = function.contains("name") && function["name"].is_string()
                                           ? function["name"].get<std::string>()
                                           : std::string{};
                check(prefix + "/function/parameters", true, name);
            } else {
                const auto name = tool.contains("name") && tool["name"].is_string()
                                      ? tool["name"].get<std::string>()
                                      : std::string{};
                check(prefix + "/parameters", true, name);
                check(prefix + "/input_schema", true, name);
            }
        }
    };
    if (parsed.value.is_object() && parsed.value.contains("tools")) {
        tools(tools, parsed.value["tools"], "/tools");
    }
    return parsed.value;
}

} // namespace ninfer::serve
