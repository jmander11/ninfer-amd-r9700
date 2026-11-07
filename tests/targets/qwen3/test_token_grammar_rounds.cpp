// Per-round tool-grammar masks after a long free-text output: the rolled-back single matcher
// must produce the same masks and acceptance as forking the committed parser per node, and its
// per-round host time must not grow with the committed output. `--bench` prints the timings.
#include "targets/qwen3/impl/frontend/token_grammar.h"
#include "targets/qwen3/official_tokenizer_dir.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
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

} // namespace

int main(int argc, char** argv) {
    if (skip_without_official_tokenizer("token grammar rounds")) { return 0; }
    const auto fixture = make_fixture(*official_tokenizer_dir());
    if (argc == 2 && std::string_view(argv[1]) == "--bench") {
        bench(fixture, false);
        bench(fixture, true);
        return 0;
    }
    int failures = check_equivalence(fixture, false);
    failures += check_equivalence(fixture, true);
    failures += check_reasoning_close(fixture);
    if (failures != 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "token grammar rounds: PASS\n";
    return 0;
}
