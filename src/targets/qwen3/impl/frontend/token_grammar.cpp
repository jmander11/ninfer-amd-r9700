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
      committed_(grammar_->compiled), committed_complete_(committed_.IsCompleted()),
      committed_terminated_(committed_.IsTerminated()), work_(grammar_->constraint) {}

bool TokenGrammarState::advance(xgrammar::GrammarMatcher& matcher, TokenId token,
                                int& history) const {
    if (!matcher.AcceptToken(token)) { return false; }
    if (std::find(ignored_model_stops_.begin(), ignored_model_stops_.end(), token) !=
        ignored_model_stops_.end()) {
        // Preserve the caller's disabled model-stop semantics. Such a token is
        // still forbidden inside an incomplete envelope, but an eligible EOS
        // must not permanently terminate the grammar while generation continues.
        matcher.Rollback(1);
        return true;
    }
    ++history;
    return true;
}

namespace {

// Rolls the walked matcher back to the state its scope started from, also on unwinding.
struct MatcherRollback {
    xgrammar::GrammarMatcher& matcher;
    int tokens = 0;

    ~MatcherRollback() {
        if (tokens != 0) { matcher.Rollback(tokens); }
    }
};

} // namespace

void TokenGrammarState::preview(std::span<const TokenId> tokens,
                                std::span<const std::uint8_t> ignored_framing,
                                std::span<const std::uint8_t> reasoning_closes) {
    if (preview_) { throw std::logic_error("token grammar already has a preview"); }
    if (!ignored_framing.empty() && ignored_framing.size() != tokens.size()) {
        throw std::invalid_argument("token grammar framing shape mismatch");
    }
    if (!reasoning_closes.empty() && reasoning_closes.size() != tokens.size()) {
        throw std::invalid_argument("token grammar reasoning-close shape mismatch");
    }
    Preview result;
    MatcherRollback validated{committed_};
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (!ignored_framing.empty() && ignored_framing[i]) { continue; }
        if (!reasoning_closes.empty() && reasoning_closes[i]) {
            if (!grammar_->starts_in_reasoning) {
                throw std::logic_error("reasoning close requires an initial reasoning grammar");
            }
            // The already qualified any_text body may end in a byte/marker prefix.
            // Finish that semantic phase with the actual committed close token.
            result.restarted.emplace(grammar_->compiled);
            result.tokens.clear();
        }
        int history = 0;
        if (!advance(result.restarted ? *result.restarted : committed_, tokens[i], history)) {
            throw std::logic_error("generated token violates the declared token grammar");
        }
        if (!result.restarted) {
            validated.tokens += history;
            result.tokens.push_back(tokens[i]);
        }
    }
    preview_.emplace(std::move(result));
}

void TokenGrammarState::commit_preview() noexcept {
    if (!preview_) { std::terminate(); }
    if (preview_->restarted) {
        committed_ = std::move(*preview_->restarted);
    } else {
        for (const auto token : preview_->tokens) {
            int history = 0;
            if (!advance(committed_, token, history)) { std::terminate(); }
        }
    }
    committed_complete_   = committed_.IsCompleted();
    committed_terminated_ = committed_.IsTerminated();
    preview_.reset();
}

void TokenGrammarState::discard_preview() noexcept { preview_.reset(); }

ConstraintStats TokenGrammarState::constraint_stats() const noexcept {
    auto result       = work_;
    result.complete   = committed_complete_;
    result.terminated = committed_terminated_;
    return result;
}

void TokenGrammarState::record_upload(std::size_t bytes) const noexcept {
    work_.mask_upload_bytes += bytes;
}

void TokenGrammarState::fill_masks(std::span<const TokenId> tokens,
                                   std::span<const std::int32_t> parents,
                                   std::span<std::uint32_t> words, TokenId replay_close,
                                   std::span<const std::uint8_t> replay_close_allowed,
                                   std::span<const std::uint8_t> reasoning_close_allowed) {
    if (tokens.empty() || tokens.size() != parents.size() ||
        words.size() != tokens.size() * mask_words) {
        throw std::invalid_argument("invalid token grammar verification mask shape");
    }
    if (!replay_close_allowed.empty() &&
        (replay_close_allowed.size() != tokens.size() || replay_close < 0 ||
         static_cast<std::size_t>(replay_close) >= mask_words * 32)) {
        throw std::invalid_argument("invalid replay framing mask shape");
    }
    if (!reasoning_close_allowed.empty() &&
        (reasoning_close_allowed.size() != tokens.size() || replay_close < 0 ||
         static_cast<std::size_t>(replay_close) >= mask_words * 32 ||
         !grammar_->starts_in_reasoning)) {
        throw std::invalid_argument("invalid semantic reasoning-close mask shape");
    }
    std::vector<std::vector<std::size_t>> children(tokens.size());
    for (std::size_t node = 1; node < tokens.size(); ++node) {
        const auto parent = parents[node];
        if (parent < 0 || static_cast<std::size_t>(parent) >= node) {
            throw std::invalid_argument("token grammar tree parent must precede child");
        }
        children[static_cast<std::size_t>(parent)].push_back(node);
    }
    const auto started      = std::chrono::steady_clock::now();
    std::uint64_t evaluated = 0;
    std::vector<std::uint8_t> reached(tokens.size());
    bool close_checked = false;
    const auto fill    = [&](std::size_t node, xgrammar::GrammarMatcher& matcher) {
        auto output        = words.subspan(node * mask_words, mask_words);
        std::int64_t shape = mask_words;
        DLTensor mask{.data        = output.data(),
                      .device      = {kDLCPU, 0},
                      .ndim        = 1,
                      .dtype       = {kDLInt, 32, 1},
                      .shape       = &shape,
                      .strides     = nullptr,
                      .byte_offset = 0};
        (void)matcher.FillNextTokenBitmask(&mask);
        if (!reasoning_close_allowed.empty() && reasoning_close_allowed[node]) {
            if (!close_checked) {
                auto closed = xgrammar::GrammarMatcher(grammar_->compiled);
                if (!closed.AcceptToken(replay_close)) {
                    throw std::logic_error("initial reasoning grammar rejects its canonical close");
                }
                close_checked = true;
            }
            output[static_cast<std::size_t>(replay_close) / 32] |=
                std::uint32_t{1} << (static_cast<unsigned>(replay_close) % 32);
        }
        if (!replay_close_allowed.empty() && replay_close_allowed[node]) {
            output[static_cast<std::size_t>(replay_close) / 32] |=
                std::uint32_t{1} << (static_cast<unsigned>(replay_close) % 32);
        }
        reached[node] = 1;
        ++evaluated;
    };
    // `matcher` is at the state after `node`'s token and is returned in that state.
    const auto visit = [&](const auto& self, std::size_t node,
                           xgrammar::GrammarMatcher& matcher) -> void {
        fill(node, matcher);
        for (const auto child : children[node]) {
            const bool framing = !replay_close_allowed.empty() && replay_close_allowed[node] &&
                                 tokens[child] == replay_close;
            int history        = 0;
            if (!reasoning_close_allowed.empty() && reasoning_close_allowed[node] &&
                tokens[child] == replay_close) {
                xgrammar::GrammarMatcher restarted(grammar_->compiled);
                if ((framing || advance(restarted, tokens[child], history)) &&
                    !restarted.IsTerminated()) {
                    self(self, child, restarted);
                }
                continue;
            }
            if (framing) {
                self(self, child, matcher);
                continue;
            }
            if (!advance(matcher, tokens[child], history)) { continue; }
            const MatcherRollback rollback{matcher, history};
            if (!matcher.IsTerminated()) { self(self, child, matcher); }
        }
    };
    visit(visit, 0, committed_);
    for (std::size_t node = 0; node < tokens.size(); ++node) {
        if (!reached[node]) {
            std::fill_n(words.begin() + node * mask_words, mask_words, ~std::uint32_t{0});
        }
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
                    // JSON whitespace is syntax, not a schema restriction. In particular,
                    // preserve space-plus-quote tokens at free-form string boundaries rather
                    // than forcing the model onto low-probability compact spellings.
                    auto json = xgrammar::Grammar::FromJSONSchema(
                        content.at("json_schema").dump(), true, std::nullopt, std::nullopt, false);
                    const auto whitespace = xgrammar::Grammar::FromEBNF(R"(root ::= [ \t\r\n]*)");
                    return xgrammar::Grammar::Concat({whitespace, json, whitespace});
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
        return std::make_shared<const TokenGrammarData>(TokenGrammarData{
            std::move(completed_definitions), std::move(compiled), stats, starts_in_reasoning});
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
    // The search advances one validator and rolls it back; a failed AcceptString leaves it
    // unchanged. Successive close candidates extend the same raw string, so its
    // JSON-string interpretation is accepted incrementally and decoding stays linear.
    xgrammar::GrammarMatcher validator(*definition->arguments_validator, std::nullopt, true);
    if (!validator.AcceptString("{")) { return std::nullopt; }
    std::set<std::string> keys;
    // Whether the text after a value can start the next level: the close or an unused key.
    const auto continues = [&](std::string_view rest) {
        constexpr std::string_view parameter_open = "<parameter=";
        if (rest.starts_with("</function>")) { return true; }
        if (!rest.starts_with(parameter_open)) { return false; }
        rest.remove_prefix(parameter_open.size());
        const auto key_end = rest.find('>');
        return key_end != 0 && key_end != std::string_view::npos &&
               !keys.contains(std::string(rest.substr(0, key_end)));
    };
    auto decode = [&](auto&& self, xgrammar::GrammarMatcher& matcher, std::string_view remaining,
                      std::string serialized, std::size_t count) -> std::optional<ToolCall> {
        remaining                                 = trim_parameter(remaining);
        constexpr std::string_view function_close = "</function>";
        constexpr std::string_view tool_close     = "</tool_call>";
        if (remaining.starts_with(function_close)) {
            remaining = trim_parameter(remaining.substr(function_close.size()));
            if (!remaining.starts_with(tool_close) ||
                !trim_parameter(remaining.substr(tool_close.size())).empty() ||
                !matcher.AcceptString("}")) {
                return std::nullopt;
            }
            if (!matcher.IsCompleted()) {
                matcher.Rollback(1);
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
        // The grammar opens a value with [ \n\r\t]* "\n" and closes it with "\n</parameter>".
        // Any LF of the opening blank run can be the framing LF, the first one first; string
        // whitespace and embedded XML-looking text remain data.
        const auto blanks = std::min(remaining.find_first_not_of(" \t\r\n"), remaining.size());
        if (remaining.substr(0, blanks).find('\n') == std::string_view::npos) {
            return std::nullopt;
        }
        const auto prefix = (count ? "," : "") + Json(key).dump() + ":";
        if (!matcher.AcceptString(prefix)) { return std::nullopt; }
        serialized += prefix;
        keys.insert(key);
        std::optional<xgrammar::GrammarMatcher> json_matcher;
        for (auto opening = remaining.find('\n'); opening < blanks;
             opening      = remaining.find('\n', opening + 1)) {
            const auto value_begin = opening + 1;
            int string_rows        = 0;
            bool string_viable     = matcher.AcceptString("\"");
            string_rows += string_viable ? 1 : 0;
            std::string escaped;
            auto escaped_end = value_begin;
            for (auto close = remaining.find(parameter_close); close != std::string_view::npos;
                 close      = remaining.find(parameter_close, close + parameter_close.size())) {
                if (close < value_begin + 1 || remaining[close - 1] != '\n') { continue; }
                const auto rest = remaining.substr(close + parameter_close.size());
                if (!continues(trim_parameter(rest))) { continue; }
                const auto raw    = remaining.substr(value_begin, close - 1 - value_begin);
                const auto parsed = parse_parameter_json(raw);
                if (string_viable) {
                    auto chunk    = Json(std::string(raw.substr(escaped_end - value_begin))).dump();
                    chunk         = chunk.substr(1, chunk.size() - 2);
                    string_viable = matcher.AcceptString(chunk);
                    if (string_viable) {
                        ++string_rows;
                        escaped += chunk;
                        escaped_end = close - 1;
                        if (matcher.AcceptString("\"")) {
                            if (auto call = self(self, matcher, rest,
                                                 serialized + '"' + escaped + '"', count + 1)) {
                                return call;
                            }
                            matcher.Rollback(1);
                        }
                    }
                }
                if (parsed.is_discarded() || parsed == Json(raw)) { continue; }
                if (!json_matcher) {
                    json_matcher.emplace(matcher.Fork());
                    json_matcher->Rollback(string_rows);
                }
                const auto value = parsed.dump();
                if (json_matcher->AcceptString(value)) {
                    if (auto call =
                            self(self, *json_matcher, rest, serialized + value, count + 1)) {
                        return call;
                    }
                    json_matcher->Rollback(1);
                }
            }
            if (string_rows != 0) { matcher.Rollback(string_rows); }
        }
        keys.erase(key);
        matcher.Rollback(1);
        return std::nullopt;
    };
    return decode(decode, validator, text, "{", 0);
}

} // namespace ninfer::targets::qwen3
