// Per-round tool-grammar masks after a long free-text output: the rolled-back single matcher
// must produce the same masks and acceptance as forking the committed parser per node, and its
// per-round host time must not grow with the committed output. `--bench` prints the timings.
#include "targets/qwen3/impl/frontend/token_grammar.h"
#include "targets/qwen3/official_tokenizer_dir.h"

#include <ninfer/targets/qwen3/frontend.h>
#include <ninfer/targets/qwen3/frontend_resources.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <array>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace fi = ninfer::targets::qwen3::frontend_internal;
using ninfer::TokenId;

namespace {

constexpr std::size_t kWords = fi::TokenGrammarState::mask_words;

// The forking implementation: the committed matcher is forked for every preview and for every
// draft-tree node. It is the acceptance and mask oracle.
class ForkingState {
public:
    ForkingState(std::shared_ptr<const ninfer::targets::qwen3::TokenGrammarData> grammar,
                 std::vector<TokenId> ignored)
        : grammar_(std::move(grammar)), ignored_(std::move(ignored)),
          committed_(grammar_->compiled) {}

    bool accept(xgrammar::GrammarMatcher& matcher, TokenId token) const {
        if (std::find(ignored_.begin(), ignored_.end(), token) != ignored_.end()) {
            auto probe = matcher.Fork();
            return probe.AcceptToken(token);
        }
        return matcher.AcceptToken(token);
    }

    bool preview(std::span<const TokenId> tokens) {
        auto candidate = committed_.Fork();
        for (const auto token : tokens) {
            if (!accept(candidate, token)) { return false; }
        }
        preview_.emplace(std::move(candidate));
        return true;
    }

    void commit() {
        committed_ = std::move(*preview_);
        preview_.reset();
    }

    void fill_masks(std::span<const TokenId> tokens, std::span<const std::int32_t> parents,
                    std::span<std::uint32_t> words, TokenId replay_close,
                    std::span<const std::uint8_t> replay_close_allowed,
                    std::span<const std::uint8_t> reasoning_close_allowed) const {
        std::vector<std::optional<xgrammar::GrammarMatcher>> nodes(tokens.size());
        nodes[0].emplace(committed_.Fork());
        for (std::size_t node = 0; node < nodes.size(); ++node) {
            if (node != 0) {
                const auto parent = parents[node];
                if (const auto& parent_node = nodes[static_cast<std::size_t>(parent)]) {
                    auto child         = parent_node->Fork();
                    const bool framing = !replay_close_allowed.empty() &&
                                         replay_close_allowed[parent] &&
                                         tokens[node] == replay_close;
                    if (!reasoning_close_allowed.empty() && reasoning_close_allowed[parent] &&
                        tokens[node] == replay_close) {
                        child = xgrammar::GrammarMatcher(grammar_->compiled);
                    }
                    if ((framing || accept(child, tokens[node])) && !child.IsTerminated()) {
                        nodes[node].emplace(std::move(child));
                    }
                }
            }
            auto output = words.subspan(node * kWords, kWords);
            if (!nodes[node]) {
                std::fill(output.begin(), output.end(), ~std::uint32_t{0});
                continue;
            }
            std::int64_t shape = kWords;
            DLTensor mask{.data        = output.data(),
                          .device      = {kDLCPU, 0},
                          .ndim        = 1,
                          .dtype       = {kDLInt, 32, 1},
                          .shape       = &shape,
                          .strides     = nullptr,
                          .byte_offset = 0};
            (void)nodes[node]->FillNextTokenBitmask(&mask);
            const auto close_bit = [&] {
                output[static_cast<std::size_t>(replay_close) / 32] |=
                    std::uint32_t{1} << (static_cast<unsigned>(replay_close) % 32);
            };
            if (!reasoning_close_allowed.empty() && reasoning_close_allowed[node]) close_bit();
            if (!replay_close_allowed.empty() && replay_close_allowed[node]) close_bit();
        }
    }

private:
    std::shared_ptr<const ninfer::targets::qwen3::TokenGrammarData> grammar_;
    std::vector<TokenId> ignored_;
    xgrammar::GrammarMatcher committed_;
    std::optional<xgrammar::GrammarMatcher> preview_;
};

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot read " + path); }
    return {std::istreambuf_iterator<char>(stream), {}};
}

// Prose with near-misses of the trigger, the excludes and the reasoning close, none complete.
std::string free_text(std::size_t bytes, std::mt19937& random) {
    static const std::vector<std::string> words{
        "the",       "parser",     "keeps",      "state",  "for",     "every",    "byte",
        "<tool",     "</thin",     "</func",     "<param", "</invok", "<functio", "tool_call",
        "function",  "\n",         "\n\n",       "value",  "=",       "{",        "}",
        "const",     "int",        "x;",         "return", "Ünïcode", "日本語",   "-",
        "<tool_cal", "</tool_cal", "</paramete", "think>", "</",      "<",        ">"};
    std::string text;
    while (text.size() < bytes) {
        text += words[random() % words.size()];
        text += ' ';
    }
    return text;
}

struct Fixture {
    std::shared_ptr<const fi::Tokenizer> tokenizer;
    std::unique_ptr<fi::TokenGrammarCompiler> compiler;
    std::shared_ptr<const ninfer::targets::qwen3::TokenGrammarData> grammar;
    TokenId think_close = -1;
};

Fixture make_fixture(const std::string& directory) {
    const auto tokenizer_json  = read_file(directory + "/tokenizer.json");
    const auto config_json     = read_file(directory + "/tokenizer_config.json");
    const auto generation_json = read_file(directory + "/generation_config.json");
    Fixture fixture;
    fixture.tokenizer = std::make_shared<const fi::Tokenizer>(
        fi::TokenizerResources{tokenizer_json, config_json, generation_json});
    fixture.compiler = std::make_unique<fi::TokenGrammarCompiler>(fixture.tokenizer);
    const std::vector<std::string> tools{
        R"({"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"filePath":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":64}},"required":["filePath","limit"],"additionalProperties":false}}})",
        R"({"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}})",
        R"({"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"filePath":{"type":"string"},"content":{"type":"string"}},"required":["filePath","content"],"additionalProperties":false}}})",
        R"({"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"filePath":{"type":"string"},"oldString":{"type":"string"},"newString":{"type":"string"},"replaceAll":{"type":"boolean"}},"required":["filePath","oldString","newString"],"additionalProperties":false}}})",
        R"({"type":"function","function":{"name":"grep","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"include":{"type":"string"}},"required":["pattern"],"additionalProperties":false}}})",
        R"({"type":"function","function":{"name":"pick","parameters":{"type":"object","properties":{"mode":{"enum":["a","b"]},"n":{"anyOf":[{"type":"integer"},{"type":"string","maxLength":5}]},"flag":{"type":"boolean"}},"required":["mode"],"additionalProperties":false}}})",
        R"({"type":"function","function":{"name":"todowrite","parameters":{"type":"object","properties":{"todos":{"type":"array","items":{"type":"object","properties":{"content":{"type":"string"},"status":{"type":"string","enum":["pending","in_progress","completed","cancelled"]},"priority":{"type":"string","enum":["high","medium","low"]},"id":{"type":"string"}},"required":["content","status","priority","id"],"additionalProperties":false}}},"required":["todos"],"additionalProperties":false}}})"};
    fixture.grammar     = fixture.compiler->compile(tools, true);
    fixture.think_close = fixture.tokenizer->encode("</think>").at(0);
    return fixture;
}

struct Tree {
    std::vector<TokenId> tokens;
    std::vector<std::int32_t> parents;
};

// The committed continuation as a chain plus off-path branches: unrelated tokens, the trigger,
// the reasoning close, a model stop, and children of nodes the grammar may reject.
Tree draft_tree(const Fixture& fixture, std::span<const TokenId> next,
                std::span<const TokenId> pool, std::mt19937& random) {
    Tree tree{{0}, {-1}};
    const auto add = [&](TokenId token, std::int32_t parent) {
        tree.tokens.push_back(token);
        tree.parents.push_back(parent);
        return static_cast<std::int32_t>(tree.tokens.size() - 1);
    };
    std::int32_t parent = 0;
    std::vector<std::int32_t> chain{0};
    for (const auto token : next) {
        parent = add(token, parent);
        chain.push_back(parent);
    }
    const auto any      = [&] { return pool[random() % pool.size()]; };
    const auto on_chain = [&] { return chain[random() % chain.size()]; };
    const auto stray    = add(any(), on_chain());
    add(fixture.tokenizer->encode("<tool_call>").at(0), on_chain());
    add(fixture.think_close, on_chain());
    add(fixture.tokenizer->default_stop_token_ids().at(0), on_chain());
    add(next.empty() ? any() : next[0], stray);
    add(any(), on_chain());
    return tree;
}

int check_equivalence(const Fixture& fixture, bool ignore_stops) {
    std::mt19937 random(ignore_stops ? 7 : 3);
    const std::string call = "<tool_call>\n<function=bash>\n<parameter=command>\n"
                             "echo '</parameter>' && ls\n</parameter>\n</function>\n</tool_call>";
    const std::string text =
        free_text(20000, random) + "</think>\n\n" + free_text(20000, random) + "\n\n" + call;
    auto ids                  = fixture.tokenizer->encode(text);
    const auto content_tokens = ids.size() - fixture.tokenizer->encode(call).size();
    ids.push_back(fixture.tokenizer->default_stop_token_ids().at(0));
    std::vector<TokenId> ignored;
    if (ignore_stops) { ignored = fixture.tokenizer->default_stop_token_ids(); }
    fi::TokenGrammarState state(fixture.grammar, ignored);
    ForkingState oracle(fixture.grammar, ignored);
    std::vector<TokenId> pool(ids.begin(),
                              ids.begin() + static_cast<std::ptrdiff_t>(content_tokens));
    int failures = 0, sampled = 0;
    std::vector<std::uint32_t> expected, actual;
    const auto compare = [&](const Tree& tree, bool flags, std::size_t position) {
        std::vector<std::uint8_t> replay(tree.tokens.size());
        if (flags) {
            for (auto& bit : replay) bit = random() % 3 == 0;
        }
        expected.assign(tree.tokens.size() * kWords, 0);
        actual.assign(tree.tokens.size() * kWords, 0x5a5a5a5a);
        const std::span<const std::uint8_t> replay_span =
            flags ? std::span<const std::uint8_t>(replay) : std::span<const std::uint8_t>{};
        oracle.fill_masks(tree.tokens, tree.parents, expected, fixture.think_close, replay_span,
                          {});
        state.fill_masks(tree.tokens, tree.parents, actual, fixture.think_close, replay_span, {});
        ++sampled;
        if (expected != actual) {
            std::cerr << "mask mismatch at token " << position << (flags ? " (replay)" : "")
                      << '\n';
            ++failures;
        }
    };
    std::size_t position = 0;
    std::size_t round    = 0;
    while (position < ids.size()) {
        const std::size_t take = std::min<std::size_t>(1 + random() % 4, ids.size() - position);
        const bool in_call     = position + 4 >= content_tokens;
        if (in_call || round % 61 == 0) {
            const std::span<const TokenId> next(ids.data() + position,
                                                std::min<std::size_t>(3, ids.size() - position));
            const auto tree = draft_tree(fixture, next, pool, random);
            compare(tree, false, position);
            if (round % 2 == 0) compare(tree, true, position);
            // A rejected preview and a discarded preview leave the committed state unchanged.
            std::vector<TokenId> bad(ids.begin() + static_cast<std::ptrdiff_t>(position),
                                     ids.begin() + static_cast<std::ptrdiff_t>(position + take));
            bad.push_back(fixture.tokenizer->encode("</parameter>").at(0));
            bool threw = false;
            try {
                state.preview(bad);
            } catch (const std::logic_error&) { threw = true; }
            ForkingState probe = oracle;
            if (threw == probe.preview(bad)) {
                std::cerr << "preview acceptance differs at token " << position << '\n';
                ++failures;
            }
            if (!threw) state.discard_preview();
            compare(tree, false, position);
        }
        const std::span<const TokenId> accepted(ids.data() + position, take);
        if (!oracle.preview(accepted)) {
            std::cerr << "oracle rejected the generated text at token " << position << '\n';
            return failures + 1;
        }
        oracle.commit();
        state.preview(accepted);
        state.commit_preview();
        position += take;
        ++round;
    }
    if (state.constraint_stats().complete != true) {
        std::cerr << "completed call is not complete\n";
        ++failures;
    }
    std::cout << (ignore_stops ? "ignored stops: " : "default stops: ") << sampled
              << " sampled mask trees over " << ids.size() << " tokens, " << failures
              << " failures\n";
    return failures;
}

// Reasoning-close restarts (a reasoning budget forcing `</think>`) through preview and the tree.
int check_reasoning_close(const Fixture& fixture) {
    std::mt19937 random(11);
    const auto ids = fixture.tokenizer->encode(free_text(4000, random));
    fi::TokenGrammarState state(fixture.grammar);
    ForkingState oracle(fixture.grammar, {});
    oracle.preview(ids);
    oracle.commit();
    state.preview(ids);
    state.commit_preview();
    Tree tree{{0, fixture.think_close, ids[0], fixture.think_close, ids[1]}, {-1, 0, 1, 0, 3}};
    const std::vector<std::uint8_t> closes{1, 0, 0, 0, 0};
    std::vector<std::uint32_t> expected(tree.tokens.size() * kWords),
        actual(tree.tokens.size() * kWords);
    oracle.fill_masks(tree.tokens, tree.parents, expected, fixture.think_close, {}, closes);
    state.fill_masks(tree.tokens, tree.parents, actual, fixture.think_close, {}, closes);
    int failures = expected != actual;
    const std::vector<TokenId> forced{ids[0], fixture.think_close, ids[1]};
    const std::vector<std::uint8_t> forced_closes{0, 1, 0};
    state.preview(forced, {}, forced_closes);
    state.commit_preview();
    xgrammar::GrammarMatcher restarted(fixture.grammar->compiled);
    failures += !restarted.AcceptToken(fixture.think_close) || !restarted.AcceptToken(ids[1]);
    const Tree root{{0}, {-1}};
    std::vector<std::uint32_t> after(kWords), reference(kWords);
    state.fill_masks(root.tokens, root.parents, after);
    std::int64_t shape = kWords;
    DLTensor mask{.data        = reference.data(),
                  .device      = {kDLCPU, 0},
                  .ndim        = 1,
                  .dtype       = {kDLInt, 32, 1},
                  .shape       = &shape,
                  .strides     = nullptr,
                  .byte_offset = 0};
    (void)restarted.FillNextTokenBitmask(&mask);
    failures += after != reference;
    if (failures != 0) std::cerr << "reasoning-close restart differs from the forking oracle\n";
    return failures;
}

template <typename Round>
double median_ms(int rounds, Round&& round) {
    std::vector<double> samples;
    for (int i = 0; i < rounds; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        round(i);
        samples.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count());
    }
    std::nth_element(samples.begin(), samples.begin() + samples.size() / 2, samples.end());
    return samples[samples.size() / 2];
}

// One decode round per lane at --draft-tokens 3: commit one token, then fill a 4-node chain.
void bench(const Fixture& fixture, bool in_call) {
    std::mt19937 random(5);
    std::printf("| %s before round | forking ms/round | rollback ms/round |\n",
                in_call ? "reasoning + tool-call string" : "free text");
    std::printf("|---|---|---|\n");
    const std::string open = "</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\n";
    for (const std::size_t bytes : {1024u, 8192u, 32768u}) {
        const auto ids     = fixture.tokenizer->encode(in_call ? free_text(256, random) + open +
                                                                     free_text(bytes + 4096, random)
                                                               : free_text(bytes + 4096, random));
        std::size_t prefix = 0, prefix_bytes = 0;
        while (prefix_bytes < bytes + (in_call ? 256 + open.size() : 0)) {
            prefix_bytes += fixture.tokenizer->decode_token_bytes(ids[prefix]).size();
            ++prefix;
        }
        fi::TokenGrammarState state(fixture.grammar);
        ForkingState oracle(fixture.grammar, {});
        const std::span<const TokenId> head(ids.data(), prefix);
        oracle.preview(head);
        oracle.commit();
        state.preview(head);
        state.commit_preview();
        const std::vector<std::int32_t> parents{-1, 0, 1, 2};
        std::vector<std::uint32_t> words(parents.size() * kWords);
        constexpr int kRounds = 64;
        const double forking  = median_ms(kRounds, [&](int i) {
            const std::span<const TokenId> one(ids.data() + prefix + i, 1);
            oracle.preview(one);
            oracle.commit();
            const std::span<const TokenId> tree(ids.data() + prefix + i, 4);
            oracle.fill_masks(tree, parents, words, -1, {}, {});
        });
        const double rollback = median_ms(kRounds, [&](int i) {
            const std::span<const TokenId> one(ids.data() + prefix + kRounds + i, 1);
            state.preview(one);
            state.commit_preview();
            const std::span<const TokenId> tree(ids.data() + prefix + kRounds + i, 4);
            state.fill_masks(tree, parents, words);
        });
        std::printf("| %zu B | %.3f | %.3f |\n", prefix_bytes, forking, rollback);
    }
}

using Json = nlohmann::ordered_json;

std::string_view trim(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.remove_suffix(1);
    }
    return text;
}

Json parse_json_value(std::string_view text) {
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

// The forking decoder: every close candidate of every parameter forks the validator and
// accepts the whole value. It is the oracle for TokenGrammarData::decode_call on calls whose
// values open with exactly one LF.
std::optional<ninfer::ToolCall>
forking_decode(const ninfer::targets::qwen3::TokenGrammarData& grammar, std::string_view text) {
    auto consume = [&](std::string_view literal) {
        if (!text.starts_with(literal)) { return false; }
        text.remove_prefix(literal.size());
        return true;
    };
    text = trim(text);
    if (!consume("<tool_call>")) { return std::nullopt; }
    text = trim(text);
    if (!consume("<function=")) { return std::nullopt; }
    auto end = text.find('>');
    if (end == std::string_view::npos) { return std::nullopt; }
    const std::string name(text.substr(0, end));
    auto definition = std::find_if(grammar.definitions.begin(), grammar.definitions.end(),
                                   [&](const auto& item) { return item.name == name; });
    if (definition == grammar.definitions.end()) { return std::nullopt; }
    text.remove_prefix(end + 1);
    xgrammar::GrammarMatcher validator(*definition->arguments_validator, std::nullopt, true);
    if (!validator.AcceptString("{")) { return std::nullopt; }
    std::set<std::string> keys;
    auto decode = [&](auto&& self, std::string_view remaining, xgrammar::GrammarMatcher matcher,
                      std::string serialized,
                      std::size_t count) -> std::optional<ninfer::ToolCall> {
        remaining = trim(remaining);
        if (remaining.starts_with("</function>")) {
            remaining = trim(remaining.substr(std::string_view("</function>").size()));
            if (!remaining.starts_with("</tool_call>") ||
                !trim(remaining.substr(std::string_view("</tool_call>").size())).empty() ||
                !matcher.AcceptString("}") || !matcher.IsCompleted()) {
                return std::nullopt;
            }
            serialized += '}';
            return ninfer::ToolCall{.id = {}, .name = name, .arguments_json = serialized};
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
            if (raw.size() < 2 || raw.front() != '\n' || raw.back() != '\n') { continue; }
            raw.remove_prefix(1);
            raw.remove_suffix(1);
            const auto parsed = parse_json_value(raw);
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

struct GeneratedCall {
    std::string text;
    // The same call with every value opened by exactly one LF.
    std::string canonical;
    std::size_t parameters = 0;
};

// Calls whose values embed close-tag lookalikes, so the decoder backtracks over many
// candidates, and whose openings use the blanks the grammar admits before the LF.
GeneratedCall generate_call(std::mt19937& random) {
    static const std::vector<std::pair<std::string, std::vector<std::string>>> tools{
        {"write", {"content", "filePath"}},
        {"edit", {"filePath", "oldString", "newString", "replaceAll"}},
        {"read", {"filePath", "limit"}},
        {"pick", {"mode", "n", "flag"}},
        {"bash", {"command"}}};
    static const std::vector<std::string> fragments{"x",
                                                    "a b",
                                                    "\n",
                                                    "</parameter>",
                                                    "\n</parameter> ",
                                                    "\n</parameter> <parameter=filePath>\n",
                                                    "\n</parameter> <parameter=newString>\n",
                                                    "</function>",
                                                    "</tool_call>",
                                                    "<parameter=k>",
                                                    "12",
                                                    "true",
                                                    "{\"a\":1}",
                                                    "[1,2]",
                                                    "null",
                                                    "\"s\"",
                                                    "b",
                                                    "日本",
                                                    "\t"};
    static const std::vector<std::string> openings{"\n", "\n", "\n\n", " \n", "\r\n", "\t \n"};
    const auto& [name, keys] = tools[random() % tools.size()];
    std::vector<std::string> order(keys);
    std::shuffle(order.begin(), order.end(), random);
    order.resize(1 + random() % order.size());
    GeneratedCall call{"<tool_call>\n<function=" + name + ">\n", {}};
    call.canonical  = call.text;
    call.parameters = order.size();
    for (const auto& key : order) {
        std::string value;
        for (int i = 0, n = static_cast<int>(random() % 6); i < n; ++i) {
            value += fragments[random() % fragments.size()];
        }
        if (random() % 3 == 0)
            value = std::vector<std::string>{"12", "true", "a", "b", "abc"}[random() % 5];
        const auto& opening = openings[random() % openings.size()];
        const auto body     = value + "\n</parameter>\n";
        call.text += "<parameter=" + key + ">" + opening + body;
        call.canonical += "<parameter=" + key + ">" + opening.substr(opening.find('\n')) + body;
    }
    call.text += "</function>\n</tool_call>";
    call.canonical += "</function>\n</tool_call>";
    return call;
}

bool same_call(const std::optional<ninfer::ToolCall>& a, const std::optional<ninfer::ToolCall>& b) {
    return a.has_value() == b.has_value() &&
           (!a || (a->name == b->name && a->arguments_json == b->arguments_json));
}

bool blank_opened(std::string_view text) {
    for (auto open = text.find("<parameter="); open != std::string_view::npos;
         open      = text.find("<parameter=", open + 1)) {
        const auto end = text.find('>', open);
        if (end != std::string_view::npos && end + 1 < text.size() &&
            std::string_view(" \t\r").find(text[end + 1]) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

bool grammar_completes(const Fixture& fixture, std::string_view call) {
    xgrammar::GrammarMatcher generation(fixture.grammar->compiled);
    return generation.AcceptString("</think>\n\n" + std::string(call)) && generation.IsCompleted();
}

// decode_call equals the forking decoder at every `</tool_call>` prefix, as the tool filter calls
// it, when no value opens with blanks. A value opened by the blanks the grammar admits before
// its LF decodes like its single-LF form.
int check_decode(const Fixture& fixture) {
    std::mt19937 random(13);
    int failures = 0, compared = 0, decoded = 0, blank_compared = 0, accepted = 0, widened = 0;
    for (int i = 0; i < 4000; ++i) {
        const auto call = generate_call(random);
        if (!blank_opened(call.canonical)) {
            const std::string_view text(call.canonical);
            for (auto close = text.find("</tool_call>"); close != std::string_view::npos;
                 close      = text.find("</tool_call>", close + 1)) {
                const auto prefix = text.substr(0, close + std::string_view("</tool_call>").size());
                const auto actual = fixture.grammar->decode_call(prefix);
                const auto expected = forking_decode(*fixture.grammar, prefix);
                ++compared;
                decoded += actual.has_value();
                // Beyond the forking decoder, only calls the grammar completes are decoded: a
                // value framed by a later LF of its opening run.
                if (!expected && actual && grammar_completes(fixture, prefix)) {
                    ++widened;
                    continue;
                }
                if (!same_call(expected, actual)) {
                    ++failures;
                    std::cerr << "decode differs: " << Json(std::string(prefix)).dump() << '\n';
                }
            }
        }
        // Every call the tool grammar completes is published. Enum literals are excluded: their
        // grammar admits blanks inside the value that the decoder keeps as data.
        if (!call.text.starts_with("<tool_call>\n<function=pick>") &&
            grammar_completes(fixture, call.text)) {
            ++accepted;
            if (!fixture.grammar->decode_call(call.text)) {
                ++failures;
                std::cerr << "grammar-accepted call not decoded: " << Json(call.text).dump()
                          << '\n';
            }
        }
        // Blank openings only at the generated parameters, so the single-LF form is the oracle.
        std::size_t opens = 0;
        for (auto at = call.canonical.find("<parameter="); at != std::string::npos;
             at      = call.canonical.find("<parameter=", at + 1)) {
            ++opens;
        }
        if (call.text == call.canonical || blank_opened(call.canonical) ||
            opens != call.parameters) {
            continue;
        }
        const auto expected = forking_decode(*fixture.grammar, call.canonical);
        // A rejected value can absorb a later opening, whose blanks are then string data.
        if (expected && expected->arguments_json.find("<parameter=") != std::string::npos) {
            continue;
        }
        ++blank_compared;
        const auto actual = fixture.grammar->decode_call(call.text);
        if (!expected && actual && grammar_completes(fixture, call.text)) {
            ++widened;
            continue;
        }
        if (!same_call(expected, actual)) {
            ++failures;
            std::cerr << "blank opening decodes differently: " << Json(call.text).dump() << '\n';
        }
    }
    std::cout << "decode: " << compared << " prefixes compared, " << decoded << " decoded, "
              << blank_compared << " blank-opened calls, " << accepted
              << " grammar-accepted calls, " << widened << " decoded beyond the forking decoder, "
              << failures << " failures\n";
    return failures;
}

// One round of the tool filter while a call is pending: before, every `</tool_call>` candidate of
// the call was decoded again each round; now only a new candidate is decoded, once.
void bench_decode(const Fixture& fixture) {
    std::printf(
        "| call bytes | candidates | forking: decode all, ms/round | new: decode newest, ms |\n");
    std::printf("|---|---|---|---|\n");
    const std::string unit = "y = 2\n</parameter> z\n# </tool_call>\n";
    for (const std::size_t bytes : {1024u, 4096u, 8192u, 32768u}) {
        std::string content;
        while (content.size() < bytes) content += unit;
        const auto text = "<tool_call>\n<function=write>\n<parameter=content>\n" + content;
        std::vector<std::size_t> ends;
        for (auto close = text.find("</tool_call>"); close != std::string::npos;
             close      = text.find("</tool_call>", close + 1)) {
            ends.push_back(close + std::string_view("</tool_call>").size());
        }
        double forking = -1;
        if (bytes <= 8192) {
            const auto begin = std::chrono::steady_clock::now();
            for (const auto end : ends) (void)forking_decode(*fixture.grammar, text.substr(0, end));
            forking =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                    .count();
        }
        const auto begin = std::chrono::steady_clock::now();
        (void)fixture.grammar->decode_call(std::string_view(text).substr(0, ends.back()));
        const double newest =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count();
        if (forking < 0) {
            std::printf("| %zu | %zu | not run | %.3f |\n", text.size(), ends.size(), newest);
        } else {
            std::printf("| %zu | %zu | %.1f | %.3f |\n", text.size(), ends.size(), forking, newest);
        }
    }
}

// An OpenCode-shaped request: six tools, thinking on.
struct Session {
    ninfer::targets::qwen3::Frontend frontend;
    ninfer::targets::qwen3::PreparedPrompt prompt;
};

Session make_session(const std::string& directory) {
    ninfer::targets::qwen3::FrontendResources resources;
    resources.tokenizer_json           = read_file(directory + "/tokenizer.json");
    resources.tokenizer_config_json    = read_file(directory + "/tokenizer_config.json");
    resources.chat_template_jinja      = read_file(directory + "/chat_template.jinja");
    resources.generation_config_json   = read_file(directory + "/generation_config.json");
    resources.preprocessor_config_json = read_file(directory + "/preprocessor_config.json");
    resources.video_preprocessor_config_json =
        read_file(directory + "/video_preprocessor_config.json");
    auto frontend = ninfer::targets::qwen3::make_frontend(resources, false);
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "solve"});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = true;
    const std::string schema =
        R"("additionalProperties":false,"$schema":"http://json-schema.org/draft-07/schema#")";
    for (
        const auto& [name, properties, required] : std::vector<std::array<std::string, 3>>{
            {"bash",
             R"("command":{"type":"string"},"timeout":{"type":"number"},"description":{"type":"string"})",
             R"("command","description")"},
            {"read",
             R"("filePath":{"type":"string"},"offset":{"type":"number"},"limit":{"type":"number"})",
             R"("filePath")"},
            {"write", R"("content":{"type":"string"},"filePath":{"type":"string"})",
             R"("content","filePath")"},
            {"edit",
             R"("filePath":{"type":"string"},"oldString":{"type":"string"},"newString":{"type":"string"},"replaceAll":{"type":"boolean"})",
             R"("filePath","oldString","newString")"},
            {"glob", R"("pattern":{"type":"string"},"path":{"type":"string"})", R"("pattern")"},
            {"grep",
             R"("pattern":{"type":"string"},"path":{"type":"string"},"include":{"type":"string"})",
             R"("pattern")"}}) {
        input.options.tool_jsons.push_back(R"({"type":"function","function":{"name":")" + name +
                                           R"(","parameters":{"type":"object","properties":{)" +
                                           properties + "},\"required\":[" + required + "]," +
                                           schema + "}}}");
    }
    auto prompt = frontend.prepare(std::move(input));
    return Session{std::move(frontend), std::move(prompt)};
}

// A write call whose content repeats `unit` up to `bytes`, as generated text after reasoning.
std::string write_call(const std::string& unit, std::size_t bytes) {
    std::string content;
    while (content.size() < bytes) content += unit;
    return "Writing it.\n</think>\n\n<tool_call>\n<function=write>\n<parameter=content>\n" +
           content +
           "\n</parameter>\n<parameter=filePath>\n/scratch/solver.py\n</parameter>\n</function>\n"
           "</tool_call>";
}

struct StreamResult {
    std::vector<double> round_ms;
    std::vector<ninfer::ToolCall> calls;
};

// Commits the text three tokens per round, as at --draft-tokens 3 with full acceptance.
StreamResult stream(const Session& session, const Fixture& fixture, const std::string& text) {
    const auto ids = fixture.tokenizer->encode(text);
    auto output    = session.frontend.make_output_session(session.prompt,
                                                          session.frontend.default_stop_policy());
    StreamResult result;
    for (std::size_t position = 0; position < ids.size();) {
        const auto take     = std::min<std::size_t>(3, ids.size() - position);
        const auto begin    = std::chrono::steady_clock::now();
        const auto decision = output.preview(std::span<const TokenId>(ids.data() + position, take),
                                             1u << 20, ninfer::FinishReason::OutputLimit);
        (void)output.commit_preview();
        result.round_ms.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count());
        position += decision.accepted_tokens;
        if (decision.finished()) break;
    }
    result.calls.assign(output.tool_calls().begin(), output.tool_calls().end());
    return result;
}

// A call whose content holds `</tool_call>` and `</parameter>` lookalikes is published once,
// with the arguments decode_call gives for the complete call.
int check_tool_stream(const Session& session, const Fixture& fixture) {
    const auto text     = write_call("y = 2\n</parameter> z\n# </tool_call> <parameter=k>\n", 2048);
    const auto result   = stream(session, fixture, text);
    const auto expected = fixture.grammar->decode_call(text.substr(text.find("<tool_call>")));
    const bool ok       = result.calls.size() == 1 && expected &&
                          result.calls[0].arguments_json == expected->arguments_json;
    if (!ok) std::cerr << "tool stream did not publish the decoded call once\n";
    return ok ? 0 : 1;
}

// Host time of one output round while a long write call streams.
void bench_tool_stream(const Session& session, const Fixture& fixture) {
    std::printf("| write content | rounds | mean ms/round, last 10%% | max ms/round |\n");
    std::printf("|---|---|---|---|\n");
    for (const auto& [label, unit] : std::vector<std::pair<std::string, std::string>>{
             {"code", "def f(x):\n    return [b ^ 0xae for b in x]  # xor chain\n"},
             {"code + </tool_call>", "x = 1  # </tool_call>\n"},
             {"code + </parameter> + </tool_call>", "y = 2\n</parameter> z\n# </tool_call>\n"}}) {
        for (const std::size_t bytes : {8192u, 32768u}) {
            const auto result = stream(session, fixture, write_call(unit, bytes));
            const auto& ms    = result.round_ms;
            const auto tail   = ms.size() / 10;
            double mean       = 0;
            for (auto i = ms.size() - tail - 1; i + 1 < ms.size(); ++i) mean += ms[i];
            std::printf("| %s, %zu B | %zu | %.3f | %.3f |\n", label.c_str(), bytes, ms.size(),
                        mean / static_cast<double>(tail), *std::max_element(ms.begin(), ms.end()));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (skip_without_official_tokenizer("token grammar rounds")) { return 0; }
    const auto fixture = make_fixture(*official_tokenizer_dir());
    if (argc == 2 && std::string_view(argv[1]) == "--bench") {
        bench(fixture, false);
        bench(fixture, true);
        bench_decode(fixture);
        bench_tool_stream(make_session(*official_tokenizer_dir()), fixture);
        return 0;
    }
    int failures = check_equivalence(fixture, false);
    failures += check_equivalence(fixture, true);
    failures += check_reasoning_close(fixture);
    failures += check_decode(fixture);
    failures += check_tool_stream(make_session(*official_tokenizer_dir()), fixture);
    if (failures != 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "token grammar rounds: PASS\n";
    return 0;
}
