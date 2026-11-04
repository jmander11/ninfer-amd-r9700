#include "text/json_input.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>

int main() {
    try {
        for (const auto& [source, expected] :
             {std::pair{R"({"const":9007199254740993.0})", "/const"},
              std::pair{R"({"minimum":9007199254740993})", ""}, std::pair{R"({"const":0.1})", ""},
              std::pair{R"({"const":100e-2})", ""}, std::pair{R"({"const":-0.0})", ""},
              std::pair{R"({"const":1.00000000000000001})", "/const"},
              std::pair{R"({"const":1e-324})", "/const"},
              std::pair{R"({"description":1.00000000000000001})", ""},
              std::pair{R"({"const":1.00000000000000001,"const":1})", ""},
              std::pair{R"({"const":{"a":[1.00000000000000001]}})", "/const/a/0"},
              std::pair{R"({"properties":{"a/b~":{"minimum":1.00000000000000001}}})",
                        "/properties/a~1b~0/minimum"},
              std::pair{R"({"prefixItems":[{"enum":[1.00000000000000001]}]})",
                        "/prefixItems/0/enum/0"}}) {
            const auto parsed   = ninfer::text::parse_json_numbers(source);
            const auto location = ninfer::text::inexact_schema_number(parsed);
            if (location.value_or("") != expected) {
                throw std::runtime_error(std::string(source) + ": wrong precision disposition");
            }
        }
        const auto request = ninfer::text::parse_json_numbers(
            R"({"temperature":1.00000000000000001,"schema":{"const":1}})");
        if (ninfer::text::inexact_schema_number(request, "/schema")) {
            throw std::runtime_error("ordinary request number was treated as a schema assertion");
        }
        using Json      = nlohmann::ordered_json;
        const auto same = [](std::string_view a, std::string_view b) {
            return ninfer::text::represented_json_equal(Json::parse(a), Json::parse(b));
        };
        for (const auto& [a, b] : {std::pair{"1", "1.0"}, std::pair{"0", "-0.0"},
                                   std::pair{R"({"a":1,"b":[2,3.0]})", R"({"b":[2.0,3],"a":1.0})"},
                                   std::pair{"9223372036854775807", "9223372036854775807"},
                                   std::pair{"18446744073709551615", "18446744073709551615"}}) {
            if (!same(a, b) || !same(b, a))
                throw std::runtime_error("equal represented JSON values were distinguished");
        }
        for (const auto& [a, b] :
             {std::pair{"9007199254740993", "9007199254740992.0"},
              std::pair{"9223372036854775807", "9223372036854775808.0"},
              std::pair{"18446744073709551615", "18446744073709551616.0"},
              std::pair{"-1", "18446744073709551615"},
              std::pair{R"({"a":[1,2]})", R"({"a":[2,1]})"}, std::pair{"true", "1"}}) {
            if (same(a, b) || same(b, a))
                throw std::runtime_error("distinct represented JSON values compared equal");
        }
        // The vendor may reorder object keys and serialize 1.0 as 1. Source identity must
        // still select the second tool without rounding the first tool's larger integer.
        const std::vector<Json> tool_schemas{
            Json::parse(
                R"({"type":"object","properties":{"n":{"type":"integer","minimum":9007199254740993},"v":{"const":1.0}}})"),
            Json::parse(
                R"({"type":"object","properties":{"n":{"type":"integer","minimum":9007199254740992.0},"v":{"const":1.0}}})")};
        const auto vendor_schema = Json::parse(
            R"({"properties":{"v":{"const":1},"n":{"minimum":9007199254740992,"type":"integer"}},"type":"object"})");
        const auto selected =
            std::find_if(tool_schemas.begin(), tool_schemas.end(), [&](const auto& candidate) {
                return ninfer::text::represented_json_equal(vendor_schema, candidate);
            });
        if (selected != tool_schemas.begin() + 1)
            throw std::runtime_error("schema diagnostic identity selected the wrong tool");
        std::cout << "PASS: decimal schema source values and paths\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
