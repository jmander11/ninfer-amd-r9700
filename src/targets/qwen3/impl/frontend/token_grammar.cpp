#include "targets/qwen3/impl/frontend/token_grammar.h"
#include "text/json_input.h"
#include "text/json_schema.h"
#include "ninfer/targets/qwen3/frontend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <cctype>
#include <set>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#if defined(__GLIBC__)
#    include <malloc.h>
#endif

namespace ninfer::targets::qwen3::frontend_internal {
namespace {
using Json = nlohmann::ordered_json;

bool annotation(std::string_view key) {
    return key == "$schema" || key == "$id" || key == "title" || key == "description" ||
           key == "default" || key == "examples" || key == "deprecated" || key == "readOnly" ||
           key == "writeOnly" || key == "$comment";
}

// Rewrites a client schema into one the grammar compiler enforces, accepting a superset of the
// client's instances so a valid call is never masked out; the client still validates the
// arguments it receives. `format` and the content keywords are annotations in JSON Schema 2020-12.
// Assertions that only narrow the instance set are dropped, `oneOf` widens to `anyOf`, and a
// single-branch `allOf` with annotation-only siblings is inlined. Keywords that grant instances
// (`patternProperties`) stay and are checked by the shared schema
// validator.
void relax_schema(Json& schema) {
    if (!schema.is_object()) { return; }
    // Inline first so the inlined keywords are relaxed below.
    if (schema.contains("allOf") && schema["allOf"].is_array() && schema["allOf"].size() == 1 &&
        schema["allOf"][0].is_object()) {
        const bool annotation_siblings =
            std::all_of(schema.items().begin(), schema.items().end(), [](const auto& item) {
                return item.key() == "allOf" || annotation(item.key());
            });
        if (annotation_siblings) {
            Json inner = std::move(schema["allOf"][0]);
            schema.erase("allOf");
            for (auto it = inner.begin(); it != inner.end(); ++it) {
                schema[it.key()] = it.value();
            }
        }
    }
    for (const char* key : {"format", "contentEncoding", "contentMediaType", "contentSchema",
                            "uniqueItems", "multipleOf", "propertyNames", "not", "contains",
                            "minContains", "maxContains", "dependentRequired", "dependentSchemas",
                            "if", "then", "else", "unevaluatedProperties", "unevaluatedItems"}) {
        schema.erase(key);
    }
    if (schema.contains("oneOf") && !schema.contains("anyOf")) {
        schema["anyOf"] = std::move(schema["oneOf"]);
        schema.erase("oneOf");
    }
    for (const char* map : {"properties", "$defs", "definitions"}) {
        if (schema.contains(map) && schema[map].is_object()) {
            for (auto& child : schema[map]) { relax_schema(child); }
        }
    }
    for (const char* child : {"items", "additionalProperties"}) {
        if (schema.contains(child)) { relax_schema(schema[child]); }
    }
    for (const char* list : {"prefixItems", "anyOf", "allOf"}) {
        if (schema.contains(list) && schema[list].is_array()) {
            for (auto& child : schema[list]) { relax_schema(child); }
        }
    }
}

} // namespace

TokenGrammarState::TokenGrammarState(std::shared_ptr<const TokenGrammarData> grammar,
                                     std::vector<TokenId> ignored_model_stops)
    : grammar_(std::move(grammar)), ignored_model_stops_(std::move(ignored_model_stops)),
      committed_(grammar_->compiled), work_(grammar_->constraint) {}

bool TokenGrammarState::accept(xgrammar::GrammarMatcher& matcher, TokenId token) const {
    if (std::find(ignored_model_stops_.begin(), ignored_model_stops_.end(), token) !=
        ignored_model_stops_.end()) {
        // Preserve the caller's disabled model-stop semantics. Such a token is
        // still forbidden inside an incomplete envelope, but an eligible EOS
        // must not permanently terminate the grammar while generation continues.
        auto probe = matcher.Fork();
        return probe.AcceptToken(token);
    }
    return matcher.AcceptToken(token);
}

void TokenGrammarState::preview(std::span<const TokenId> tokens,
                                std::span<const std::uint8_t> ignored_framing) {
    if (preview_) { throw std::logic_error("token grammar already has a preview"); }
    if (!ignored_framing.empty() && ignored_framing.size() != tokens.size()) {
        throw std::invalid_argument("token grammar framing shape mismatch");
    }
    auto candidate = committed_.Fork();
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (!ignored_framing.empty() && ignored_framing[i]) { continue; }
        if (!accept(candidate, tokens[i])) {
            throw std::logic_error("generated token violates the declared token grammar");
        }
    }
    preview_.emplace(std::move(candidate));
}

void TokenGrammarState::commit_preview() noexcept {
    if (!preview_) { std::terminate(); }
    committed_ = std::move(*preview_);
    preview_.reset();
}

void TokenGrammarState::discard_preview() noexcept { preview_.reset(); }

ConstraintStats TokenGrammarState::constraint_stats() const noexcept {
    auto result       = work_;
    result.complete   = committed_.IsCompleted();
    result.terminated = committed_.IsTerminated();
    return result;
}

void TokenGrammarState::record_upload(std::size_t bytes) const noexcept {
    work_.mask_upload_bytes += bytes;
}

void TokenGrammarState::fill_masks(std::span<const TokenId> tokens,
                                   std::span<const std::int32_t> parents,
                                   std::span<std::uint32_t> words, TokenId replay_close,
                                   std::span<const std::uint8_t> replay_close_allowed) const {
    if (tokens.empty() || tokens.size() != parents.size() ||
        words.size() != tokens.size() * mask_words) {
        throw std::invalid_argument("invalid token grammar verification mask shape");
    }
    if (!replay_close_allowed.empty() &&
        (replay_close_allowed.size() != tokens.size() || replay_close < 0 ||
         static_cast<std::size_t>(replay_close) >= mask_words * 32)) {
        throw std::invalid_argument("invalid replay framing mask shape");
    }
    const auto started      = std::chrono::steady_clock::now();
    std::uint64_t evaluated = 0;
    std::vector<std::optional<xgrammar::GrammarMatcher>> nodes(tokens.size());
    nodes[0].emplace(committed_.Fork());
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        if (node != 0) {
            const auto parent = parents[node];
            if (parent < 0 || static_cast<std::size_t>(parent) >= node) {
                throw std::invalid_argument("token grammar tree parent must precede child");
            }
            const auto& parent_node = nodes[static_cast<std::size_t>(parent)];
            if (parent_node) {
                auto child         = parent_node->Fork();
                const bool framing = !replay_close_allowed.empty() &&
                                     replay_close_allowed[parent] && tokens[node] == replay_close;
                if ((framing || accept(child, tokens[node])) && !child.IsTerminated()) {
                    nodes[node].emplace(std::move(child));
                }
            }
        }
        auto output = words.subspan(node * mask_words, mask_words);
        if (!nodes[node]) {
            std::fill(output.begin(), output.end(), ~std::uint32_t{0});
            continue;
        }
        std::int64_t shape = mask_words;
        DLTensor mask{.data        = output.data(),
                      .device      = {kDLCPU, 0},
                      .ndim        = 1,
                      .dtype       = {kDLInt, 32, 1},
                      .shape       = &shape,
                      .strides     = nullptr,
                      .byte_offset = 0};
        (void)nodes[node]->FillNextTokenBitmask(&mask);
        if (!replay_close_allowed.empty() && replay_close_allowed[node]) {
            output[static_cast<std::size_t>(replay_close) / 32] |=
                std::uint32_t{1} << (static_cast<unsigned>(replay_close) % 32);
        }
        ++evaluated;
    }
    ++work_.mask_calls;
    work_.mask_nodes += evaluated;
    work_.mask_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

TokenGrammarCompiler::TokenGrammarCompiler(std::shared_ptr<const Tokenizer> tokenizer)
    : tokenizer_(std::move(tokenizer)) {}

std::shared_ptr<const TokenGrammarData>
TokenGrammarCompiler::compile(std::span<const std::string> tools, bool starts_in_reasoning,
                              bool require_tool_call,
                              const std::optional<OutputConstraint>& constraint) {
    if (tools.empty() && !constraint) {
        if (require_tool_call) {
            throw RequestError(RequestErrorKind::InvalidToolSchema,
                               "a required tool call needs declared tools");
        }
        return {};
    }
    if ((!tools.empty() || require_tool_call) && constraint) {
        throw RequestError(RequestErrorKind::InvalidOutputConstraint,
                           "output constraints cannot be combined with tools");
    }
    const auto compile_started = std::chrono::steady_clock::now();
    std::vector<TokenGrammarData::Definition> definitions;
    std::vector<text::PreparedJsonSchema> prepared_tools;
    std::optional<text::PreparedJsonSchema> prepared_constraint;
    auto tags = Json::array();
    std::set<std::string> names;
    for (std::size_t tool_index = 0; tool_index < tools.size(); ++tool_index) {
        const auto& text = tools[tool_index];
        try {
            const auto parsed_tool = text::parse_json_numbers(text);
            if (const auto location =
                    text::inexact_schema_number(parsed_tool, "/function/parameters")) {
                throw text::SchemaError(
                    text::SchemaErrorKind::UnsupportedJsonSchema,
                    "tool schema numeric value cannot be preserved at " + *location,
                    location->substr(std::string_view("/function/parameters").size()));
            }
            const auto& tool     = parsed_tool.value;
            const auto& function = tool.at("function");
            const auto name      = function.at("name").get<std::string>();
            if (name.empty() || name.find_first_of("<>\r\n") != std::string::npos ||
                !names.insert(name).second) {
                throw std::invalid_argument("invalid or duplicate tool name: " + name);
            }
            Json schema = function.value("parameters", Json{{"type", "object"}});
            relax_schema(schema);
            prepared_tools.push_back(text::prepare_json_schema(schema.dump(), false));
            schema = Json::parse(prepared_tools.back().json);
            definitions.push_back({name, schema, std::nullopt});
            tags.push_back(Json{{"type", "tag"},
                                {"begin", "<tool_call>\n<function=" + name + ">\n"},
                                {"content", Json{{"type", "json_schema"},
                                                 {"style", "qwen_xml"},
                                                 {"any_order", true},
                                                 {"json_schema", schema}}},
                                {"end", "</function>\n</tool_call>"}});
        } catch (const text::SchemaError& error) {
            throw RequestError(RequestErrorKind::InvalidToolSchema,
                               std::string("invalid or unsupported tool schema: ") + error.what(),
                               {}, RequestErrorLocation{error.pointer(), tool_index});
        } catch (const Json::exception& error) {
            throw RequestError(RequestErrorKind::InvalidToolSchema,
                               std::string("invalid tool declaration: ") + error.what());
        } catch (const std::invalid_argument& error) {
            throw RequestError(RequestErrorKind::InvalidToolSchema,
                               std::string("invalid or unsupported tool schema: ") + error.what());
        }
    }
    // After a complete call, triggered_tags returns to free text. Reserve
    // protocol prefixes, not only complete tags: otherwise malformed variants
    // such as </invoke\n or </function = null> escape as "prose". Exclusions apply only to
    // the dispatch/free-text region, not to schema-constrained argument values
    // or to a call's actual closing frame. The captured legacy </invoke> tail
    // is also protocol debris, not a supported Qwen invocation envelope.
    Json format{{"type", "triggered_tags"},
                {"triggers", {"<tool_call"}},
                {"tags", tags},
                {"excludes",
                 {"</invoke", "</parameter", "</function", "</tool_call", "<invoke", "<parameter",
                  "<function", "</think>"}}};
    if (require_tool_call) {
        // The first content item must be a declared call, not merely a promise
        // in the prompt. Subsequent calls/prose retain the normal dispatch grammar.
        // Whitespace accommodates the thinking-close boundary without permitting
        // an arbitrary free-text answer or an early model stop.
        format =
            Json{{"type", "sequence"},
                 {"elements", Json::array({Json{{"type", "regex"}, {"pattern", R"([ \t\r\n]*)"}},
                                           Json{{"type", "or"}, {"elements", tags}}, format})}};
    }
    if (constraint) {
        try {
            if (constraint->kind != OutputConstraintKind::Choice && !constraint->choices.empty()) {
                throw std::invalid_argument("literal alternatives require a choice constraint");
            }
            if (constraint->kind == OutputConstraintKind::Choice) {
                if (!constraint->source.empty() || constraint->choices.empty()) {
                    throw std::invalid_argument(
                        "choice requires nonempty literal alternatives and no source");
                }
                auto alternatives = Json::array();
                for (const auto& value : constraint->choices) {
                    // Structural literals own escaping, including quotes and backslashes.
                    alternatives.push_back(Json{{"type", "const_string"}, {"value", value}});
                }
                format = Json{{"type", "or"}, {"elements", std::move(alternatives)}};
            } else if (constraint->kind == OutputConstraintKind::Regex) {
                format = Json{{"type", "regex"}, {"pattern", constraint->source}};
            } else if (constraint->kind == OutputConstraintKind::Grammar) {
                if (constraint->source.empty()) {
                    throw std::invalid_argument("grammar must not be empty");
                }
                format = Json{{"type", "grammar"}, {"grammar", constraint->source}};
            } else {
                Json schema =
                    constraint->kind == OutputConstraintKind::JsonObject
                        ? Json{{"type", "object"}, {"additionalProperties", true}}
                        : [&] {
                              const auto parsed = text::parse_json_numbers(constraint->source);
                              if (const auto location = text::inexact_schema_number(parsed)) {
                                  throw text::SchemaError(
                                      text::SchemaErrorKind::UnsupportedJsonSchema,
                                      "schema numeric value cannot be preserved at " + *location,
                                      *location);
                              }
                              return parsed.value;
                          }();
                // Response constraints enforce assertions; tool-schema relaxation does not apply.
                prepared_constraint = text::prepare_json_schema(schema.dump());
                schema              = Json::parse(prepared_constraint->json);
                format =
                    Json{{"type", "json_schema"}, {"json_schema", schema}, {"any_order", false}};
            }
        } catch (const text::SchemaError& error) {
            throw RequestError(RequestErrorKind::InvalidOutputConstraint, error.what(), {},
                               RequestErrorLocation{error.pointer(), std::nullopt});
        } catch (const Json::exception& error) {
            throw RequestError(RequestErrorKind::InvalidOutputConstraint, error.what());
        } catch (const std::invalid_argument& error) {
            throw RequestError(RequestErrorKind::InvalidOutputConstraint, error.what());
        }
    }
    if (starts_in_reasoning) {
        format = Json{
            {"type", "sequence"},
            {"elements",
             Json::array(
                 {// A tool envelope belongs to the content/call phase. Without this
                  // exclusion, the model can rehearse complete calls indefinitely
                  // inside reasoning, where they must never be published as calls.
                  Json{{"type", "tag"},
                       {"begin", ""},
                       {"content", Json{{"type", "any_text"},
                                        {"excludes", constraint ? Json::array()
                                                                : Json::array({"<tool_call"})}}},
                       {"end", "</think>"}},
                  format})}};
    }
    const auto wait_started = std::chrono::steady_clock::now();
    std::scoped_lock lock(mutex_);
    const double compile_wait_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_started).count();

    // Compile temporaries are freed before this returns, including a rejected
    // schema. malloc_trim releases the free space at the top of each glibc
    // arena and leaves the compiler cache in place.
    struct ReleaseCompileHeap {
        ~ReleaseCompileHeap() noexcept {
#if defined(__GLIBC__)
            ::malloc_trim(0);
#endif
        }
    } release_compile_heap;

    std::future<std::vector<TokenGrammarData::Definition>> arguments;
    if (!definitions.empty()) {
        if (!arguments_compiler_) {
            arguments_compiler_.emplace(xgrammar::TokenizerInfo({}, xgrammar::VocabType::RAW, 0), 1,
                                        true, 32 * 1024 * 1024);
        }
        // The byte validators and vocabulary masks have independent compilers.
        // One joined worker overlaps their cold builds; the outer lock bounds
        // compilation to one request and protects both caches until it joins.
        arguments = std::async(std::launch::async, [this, definitions = std::move(definitions),
                                                    prepared_tools]() mutable {
            for (std::size_t index = 0; index < definitions.size(); ++index) {
                auto& definition = definitions[index];
                try {
                    definition.arguments_validator = arguments_compiler_->CompileJSONSchema(
                        definition.parameters.dump(), true, std::nullopt, std::nullopt, false,
                        std::nullopt, true);
                } catch (const xgrammar::JSONSchemaCompileError& error) {
                    const auto pointer = prepared_tools[index].source_pointer(error.pointer);
                    throw RequestError(RequestErrorKind::InvalidToolSchema,
                                       "invalid schema for tool " + definition.name + ": " +
                                           error.what() + " at " +
                                           (pointer.empty() ? "/" : pointer),
                                       {}, RequestErrorLocation{pointer, index});
                }
            }
            return std::move(definitions);
        });
    }

    if (!compiler_) {
        std::vector<std::string> vocab(kTokenDomain);
        for (std::size_t id = 0; id < kTokenDomain; ++id) {
            if (tokenizer_->is_valid_token(static_cast<int>(id))) {
                vocab[id] = tokenizer_->decode_token_bytes(static_cast<int>(id));
            }
        }
        xgrammar::TokenizerInfo info(vocab, xgrammar::VocabType::RAW, kTokenDomain,
                                     tokenizer_->default_stop_token_ids());
        // Cold vocabulary preprocessing uses the target host's sixteen physical cores.
        compiler_.emplace(info, 16, true, 128 * 1024 * 1024);
    }
    try {
        const auto structural_grammar = [](const Json& content) {
            auto result = xgrammar::Grammar::FromStructuralTag(
                Json{{"type", "structural_tag"}, {"format", content}}.dump());
            if (std::holds_alternative<xgrammar::StructuralTagError>(result)) {
                std::visit([](const auto& error) { throw std::runtime_error(error.what()); },
                           std::get<xgrammar::StructuralTagError>(result));
            }
            return std::get<xgrammar::Grammar>(std::move(result));
        };
        xgrammar::CompilationCacheAccess access;
        // The complete framing, reasoning mode, schema ordering, and literal sources form the key.
        auto compiled = compiler_->CompileCachedGrammar(
            format.dump(),
            [&] {
                if (!constraint) return structural_grammar(format);
                const auto& content = starts_in_reasoning ? format.at("elements").at(1) : format;
                auto grammar        = [&] {
                    if (constraint->kind == OutputConstraintKind::Grammar)
                        return xgrammar::Grammar::FromEBNF(constraint->source);
                    if (constraint->kind == OutputConstraintKind::Regex)
                        return xgrammar::Grammar::FromRegex(constraint->source);
                    if (constraint->kind == OutputConstraintKind::Choice)
                        return structural_grammar(content);
                    return xgrammar::Grammar::FromJSONSchema(
                        content.at("json_schema").dump(), false, std::nullopt,
                        std::pair<std::string, std::string>{",", ":"}, false);
                }();
                if (starts_in_reasoning)
                    grammar = xgrammar::Grammar::Concat(
                        {structural_grammar(format.at("elements").at(0)), grammar});
                return grammar;
            },
            &access);
        auto completed_definitions =
            arguments.valid() ? arguments.get() : std::vector<TokenGrammarData::Definition>{};
        ConstraintStats stats;
        stats.active = true;
        stats.branch = starts_in_reasoning
                           ? ConstraintBranch::Reasoning
                           : (constraint ? ConstraintBranch::Answer : ConstraintBranch::ToolCall);
        switch (access) {
        case xgrammar::CompilationCacheAccess::kHit:
            stats.compile_cache = ConstraintCompileCache::Hit;
            break;
        case xgrammar::CompilationCacheAccess::kBuilt:
            stats.compile_cache = ConstraintCompileCache::Built;
            break;
        case xgrammar::CompilationCacheAccess::kWaited:
            stats.compile_cache = ConstraintCompileCache::Waited;
            break;
        }
        stats.compile_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - compile_started)
                .count();
        stats.compile_wait_seconds = compile_wait_seconds;
        return std::make_shared<const TokenGrammarData>(
            TokenGrammarData{std::move(completed_definitions), std::move(compiled), stats});
    } catch (const xgrammar::JSONSchemaCompileError& error) {
        if (constraint && prepared_constraint) {
            const auto pointer = prepared_constraint->source_pointer(error.pointer);
            throw RequestError(RequestErrorKind::InvalidOutputConstraint,
                               std::string(error.what()) + " at " +
                                   (pointer.empty() ? "/" : pointer),
                               {}, RequestErrorLocation{pointer, std::nullopt});
        }
        for (std::size_t index = 0; index < prepared_tools.size(); ++index) {
            // Structural conversion can reserialize the schema; compare represented DOM values.
            if (!error.source_schema.empty() &&
                !text::represented_json_equal(Json::parse(error.source_schema),
                                              Json::parse(prepared_tools[index].json)))
                continue;
            const auto pointer = prepared_tools[index].source_pointer(error.pointer);
            throw RequestError(RequestErrorKind::InvalidToolSchema,
                               std::string(error.what()) + " at " +
                                   (pointer.empty() ? "/" : pointer),
                               {}, RequestErrorLocation{pointer, index});
        }
        throw RequestError(constraint ? RequestErrorKind::InvalidOutputConstraint
                                      : RequestErrorKind::InvalidToolSchema,
                           error.what());
    } catch (const std::runtime_error& error) {
        // This pinned compiler reports malformed patterns/references through
        // runtime_error, including its non-public LogFatalError. Translate only
        // the client-schema compilation boundary; allocation failures and
        // runtime matcher/initialization failures retain their original type.
        throw RequestError(constraint ? RequestErrorKind::InvalidOutputConstraint
                                      : RequestErrorKind::InvalidToolSchema,
                           std::string("invalid or unsupported ") +
                               (constraint ? "output constraint: " : "tool schema: ") +
                               error.what());
    }
}

} // namespace ninfer::targets::qwen3::frontend_internal

namespace ninfer::targets::qwen3 {
namespace {
using Json = nlohmann::ordered_json;

std::string_view trim_parameter(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.remove_suffix(1);
    }
    return text;
}

Json parse_parameter_json(std::string_view text) {
    // nlohmann otherwise overwrites duplicate object keys before the grammar
    // sees the serialized value. Preserve the raw-string alternative, but do
    // not let this lossy parse turn invalid nested JSON into a valid argument.
    std::vector<std::set<std::string>> objects;
    bool duplicate    = false;
    const auto result = Json::parse(
        text,
        [&](int, Json::parse_event_t event, Json& value) {
            if (event == Json::parse_event_t::object_start) { objects.emplace_back(); }
            if (event == Json::parse_event_t::key &&
                !objects.back().insert(value.get<std::string>()).second) {
                duplicate = true;
            }
            if (event == Json::parse_event_t::object_end) { objects.pop_back(); }
            return true;
        },
        false);
    return duplicate ? Json(Json::value_t::discarded) : result;
}

} // namespace

std::optional<ToolCall> TokenGrammarData::decode_call(std::string_view text) const {
    auto consume = [&](std::string_view literal) {
        if (!text.starts_with(literal)) { return false; }
        text.remove_prefix(literal.size());
        return true;
    };
    text = trim_parameter(text);
    if (!consume("<tool_call>")) { return std::nullopt; }
    text = trim_parameter(text);
    if (!consume("<function=")) { return std::nullopt; }
    auto end = text.find('>');
    if (end == std::string_view::npos) { return std::nullopt; }
    const std::string name(text.substr(0, end));
    auto definition = std::find_if(definitions.begin(), definitions.end(),
                                   [&](const Definition& item) { return item.name == name; });
    if (definition == definitions.end()) { return std::nullopt; }
    text.remove_prefix(end + 1);
    // Qwen strings are raw; other values use JSON. The compiled argument
    // grammar resolves types and ambiguous literal delimiters together. A raw
    // string may contain "</parameter>", so a lexical first-close parser is
    // insufficient even when generation obeyed its declared schema.
    xgrammar::GrammarMatcher validator(*definition->arguments_validator, std::nullopt, true);
    if (!validator.AcceptString("{")) { return std::nullopt; }
    std::set<std::string> keys;
    auto decode = [&](auto&& self, std::string_view remaining, xgrammar::GrammarMatcher matcher,
                      std::string serialized, std::size_t count) -> std::optional<ToolCall> {
        remaining                                 = trim_parameter(remaining);
        constexpr std::string_view function_close = "</function>";
        constexpr std::string_view tool_close     = "</tool_call>";
        if (remaining.starts_with(function_close)) {
            remaining = trim_parameter(remaining.substr(function_close.size()));
            if (!remaining.starts_with(tool_close) ||
                !trim_parameter(remaining.substr(tool_close.size())).empty() ||
                !matcher.AcceptString("}") || !matcher.IsCompleted()) {
                return std::nullopt;
            }
            serialized += '}';
            return ToolCall{.id = {}, .name = name, .arguments_json = std::move(serialized)};
        }
        constexpr std::string_view parameter_open  = "<parameter=";
        constexpr std::string_view parameter_close = "</parameter>";
        if (!remaining.starts_with(parameter_open)) { return std::nullopt; }
        remaining.remove_prefix(parameter_open.size());
        const auto key_end = remaining.find('>');
        if (key_end == 0 || key_end == std::string_view::npos) { return std::nullopt; }
        const std::string key(remaining.substr(0, key_end));
        if (keys.contains(key)) { return std::nullopt; }
        remaining.remove_prefix(key_end + 1);
        const auto prefix = (count ? "," : "") + Json(key).dump() + ":";
        if (!matcher.AcceptString(prefix)) { return std::nullopt; }
        serialized += prefix;
        keys.insert(key);
        for (auto close = remaining.find(parameter_close); close != std::string_view::npos;
             close      = remaining.find(parameter_close, close + parameter_close.size())) {
            auto raw = remaining.substr(0, close);
            // Only the two canonical framing LFs are removed; string whitespace
            // and embedded XML-looking text remain data.
            if (raw.size() < 2 || raw.front() != '\n' || raw.back() != '\n') { continue; }
            raw.remove_prefix(1);
            raw.remove_suffix(1);
            const auto parsed = parse_parameter_json(raw);
            for (int interpretation = 0; interpretation < 2; ++interpretation) {
                if (interpretation && (parsed.is_discarded() || parsed == Json(raw))) { continue; }
                const auto value = interpretation ? parsed.dump() : Json(raw).dump();
                auto candidate   = matcher.Fork();
                if (candidate.AcceptString(value)) {
                    if (auto call = self(self, remaining.substr(close + parameter_close.size()),
                                         std::move(candidate), serialized + value, count + 1)) {
                        return call;
                    }
                }
            }
        }
        keys.erase(key);
        return std::nullopt;
    };
    return decode(decode, text, std::move(validator), "{", 0);
}

} // namespace ninfer::targets::qwen3
