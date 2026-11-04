#include "text/json_schema.h"

#include <xgrammar/xgrammar.h>

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void language(const std::string& schema,
              const std::vector<std::pair<std::string, bool>>& instances) {
    xgrammar::GrammarCompiler compiler(xgrammar::TokenizerInfo({}, xgrammar::VocabType::RAW, 0));
    const auto normalized = ninfer::text::prepare_json_schema(schema);
    const auto grammar    = compiler.CompileJSONSchema(
        normalized.json, false, std::nullopt, std::pair<std::string, std::string>{",", ":"}, false);
    for (const auto& [instance, expected] : instances) {
        xgrammar::GrammarMatcher matcher(grammar, std::nullopt, true);
        const bool actual = matcher.AcceptString(instance) && matcher.IsCompleted();
        if (actual != expected)
            throw std::runtime_error(std::string("schema intersection mismatch: ")
                                         .append(schema)
                                         .append(" on ")
                                         .append(instance));
    }
}
} // namespace

int main() {
    try {
        language(
            R"({"allOf":[{"type":"string","pattern":"^[a-z]+$"},{"minLength":2,"maxLength":3}]})",
            {{R"("ab")", true},
             {R"("abc")", true},
             {R"("a")", false},
             {R"("abcd")", false},
             {R"("a1")", false}});
        language(R"({"type":"string","minLength":2,"anyOf":[{"const":"a"},{"const":"bb"}]})",
                 {{R"("a")", false}, {R"("bb")", true}, {R"("bc")", false}});
        language(R"({"$defs":{"v":{"type":"integer","minimum":1}},"$ref":"#/$defs/v","maximum":3})",
                 {{"0", false}, {"1", true}, {"3", true}, {"4", false}});
        language(
            R"({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"additionalProperties":false,"allOf":[{"minProperties":1},{"maxProperties":2}]})",
            {{"{}", false},
             {R"({"a":1})", true},
             {R"({"a":1,"b":2})", true},
             {R"({"c":1})", false}});
        language(
            R"({"allOf":[{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":false},{"type":"object","properties":{"name":{"type":"string"}}}]})",
            {{R"({"id":1})", true}, {R"({"id":1,"name":"x"})", false}, {"{}", false}});
        language(
            R"({"type":"array","prefixItems":[{"allOf":[{"type":"string","minLength":1},{"maxLength":2}]}],"items":false})",
            {{"[]", true},
             {R"(["a"])", true},
             {R"(["ab"])", true},
             {R"([""])", false},
             {R"(["abc"])", false},
             {R"(["a",1])", false}});
        language(R"({"type":"integer","minimum":1.0,"maximum":2.0})",
                 {{"0", false}, {"1", true}, {"2", true}, {"3", false}});
        language(R"({"type":"array","prefixItems":[false]})", {{"[]", true}, {"[1]", false}});
        language(R"({"type":"array","prefixItems":[true]})",
                 {{"[]", true}, {"[1]", true}, {"[1,2]", true}});
        language(
            R"({"type":"array","prefixItems":[{"type":"integer"},{"type":"string"}],"maxItems":1})",
            {{"[]", true}, {"[1]", true}, {"[\"x\"]", false}, {"[1,\"x\"]", false}});
        for (const std::string schema :
             {R"({"type":"number","minimum":9007199254740993})", R"({"minimum":9007199254740993})",
              R"({"type":["number","string"],"maximum":-9007199254740993})"}) {
            bool rejected = false;
            try {
                (void)ninfer::text::prepare_json_schema(schema);
            } catch (const ninfer::text::SchemaError& error) {
                rejected = error.kind() == ninfer::text::SchemaErrorKind::UnsupportedJsonSchema &&
                           (error.pointer() == "/minimum" || error.pointer() == "/maximum");
            }
            if (!rejected) throw std::runtime_error("number bound rounded an exact source integer");
        }
        language(R"({"type":"integer","minimum":9007199254740993,"maximum":9007199254740994})",
                 {{"9007199254740992", false},
                  {"9007199254740993", true},
                  {"9007199254740994", true},
                  {"9007199254740995", false}});
        language(
            R"({"allOf":[{"type":"integer","minimum":9007199254740992.0},{"type":"integer","minimum":9007199254740993}]})",
            {{"9007199254740992", false}, {"9007199254740993", true}});
        language(
            R"({"allOf":[{"type":"integer","maximum":9007199254740994.0},{"type":"integer","maximum":9007199254740993}]})",
            {{"9007199254740993", true}, {"9007199254740994", false}});
        {
            const auto prepared = ninfer::text::prepare_json_schema(
                R"({"allOf":[{"type":"integer","minimum":9007199254740992.0},{"type":"integer","minimum":9007199254740993}]})");
            const auto normalized = nlohmann::ordered_json::parse(prepared.json);
            const auto generated  = normalized.at("$ref").get<std::string>().substr(1) + "/minimum";
            if (prepared.source_pointer(generated) != "/allOf/1/minimum")
                throw std::runtime_error(
                    "stronger exact bound inherited the rounded source location");
        }
        {
            const auto prepared = ninfer::text::prepare_json_schema(
                R"({"$defs":{"range":{"type":"number","minimum":0.0000001,"maximum":0.0000002}},"type":"object","properties":{"value":{"$ref":"#/$defs/range"}},"required":["value"],"allOf":[{"type":"object"}]})");
            bool located = false;
            try {
                (void)xgrammar::Grammar::FromJSONSchema(
                    prepared.json, false, std::nullopt,
                    std::pair<std::string, std::string>{",", ":"}, false);
            } catch (const xgrammar::JSONSchemaCompileError& error) {
                located = prepared.source_pointer(error.pointer) == "/$defs/range" &&
                          !error.source_schema.empty();
            }
            if (!located)
                throw std::runtime_error(
                    "composition lost the original referenced schema location");
        }
        language(
            R"({"type":"integer","$defs":{"n":{"const":1}},"oneOf":[{"$ref":"#/$defs/n"},{"const":2}]})",
            {{"0", false}, {"1", true}, {"2", true}, {"3", false}});
        std::ifstream oracle_file(std::string(NINFER_SOURCE_DIR) +
                                  "/tests/fixtures/schema_composition_oracle.json");
        if (!oracle_file) throw std::runtime_error("missing independent schema oracle fixture");
        const auto oracle = nlohmann::ordered_json::parse(oracle_file);
        for (const auto& item : oracle.at("cases")) {
            if (!item.value("admitted", true)) {
                bool rejected = false;
                try {
                    language(item.at("schema").dump(), {});
                } catch (const ninfer::text::SchemaError& error) {
                    rejected = error.kind() == ninfer::text::SchemaErrorKind::UnsupportedJsonSchema;
                } catch (const xgrammar::JSONSchemaCompileError& error) {
                    rejected = error.kind == xgrammar::SchemaErrorType::kUnsupportedSchema;
                }
                if (!rejected)
                    throw std::runtime_error(
                        "schema outside the supported proof subset was not rejected");
                continue;
            }
            std::vector<std::pair<std::string, bool>> instances;
            for (const auto& sample : item.at("instances"))
                instances.emplace_back(sample.at("value").dump(), sample.at("valid").get<bool>());
            language(item.at("schema").dump(), instances);
        }
        std::cout << "PASS: composed schema logical languages and preserved native subset\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
