#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
// Independent language oracle: can this byte prefix complete one compact JSON string
// with at most maximum Unicode scalars? It never invokes grammar/compiler machinery.
bool json_string_prefix(std::string_view text, int maximum) {
    if (text.empty()) return true;
    if (text.front() != '"') return false;
    std::size_t index = 1;
    int count         = 0;
    const auto hex    = [](unsigned char byte) -> int {
        if (byte >= '0' && byte <= '9') return byte - '0';
        if (byte >= 'a' && byte <= 'f') return byte - 'a' + 10;
        if (byte >= 'A' && byte <= 'F') return byte - 'A' + 10;
        return -1;
    };
    while (index < text.size()) {
        unsigned char byte = static_cast<unsigned char>(text[index++]);
        if (byte == '"') return index == text.size();
        if (count == maximum || byte < 0x20) return false;
        if (byte == '\\') {
            if (index == text.size()) return true;
            byte = static_cast<unsigned char>(text[index++]);
            if (std::string_view("\"\\/bfnrt").find(static_cast<char>(byte)) !=
                std::string_view::npos) {
                ++count;
                continue;
            }
            if (byte != 'u') return false;
            int code = 0;
            for (int digit = 0; digit < 4; ++digit) {
                if (index == text.size()) return true;
                const int nibble = hex(static_cast<unsigned char>(text[index++]));
                if (nibble < 0) return false;
                code            = code * 16 + nibble;
                const int shift = 4 * (3 - digit);
                const int low = code << shift, high = low + (1 << shift) - 1;
                if (low >= 0xdc00 && high <= 0xdfff) return false;
            }
            if (code >= 0xd800 && code <= 0xdbff) {
                for (char marker : std::string_view("\\u")) {
                    if (index == text.size()) return true;
                    if (text[index++] != marker) return false;
                }
                code = 0;
                for (int digit = 0; digit < 4; ++digit) {
                    if (index == text.size()) return true;
                    const int nibble = hex(static_cast<unsigned char>(text[index++]));
                    if (nibble < 0) return false;
                    code            = code * 16 + nibble;
                    const int shift = 4 * (3 - digit);
                    const int low = code << shift, high = low + (1 << shift) - 1;
                    if (low > 0xdfff || high < 0xdc00) return false;
                }
            }
        } else if (byte >= 0x80) {
            const int length = byte >= 0xc2 && byte <= 0xdf   ? 2
                               : byte >= 0xe0 && byte <= 0xef ? 3
                               : byte >= 0xf0 && byte <= 0xf4 ? 4
                                                              : 0;
            if (length == 0) return false;
            const unsigned char lead = byte;
            for (int position = 1; position < length; ++position) {
                if (index == text.size()) return true;
                byte = static_cast<unsigned char>(text[index++]);
                if (byte < 0x80 || byte > 0xbf ||
                    (position == 1 &&
                     ((lead == 0xe0 && byte < 0xa0) || (lead == 0xed && byte > 0x9f) ||
                      (lead == 0xf0 && byte < 0x90) || (lead == 0xf4 && byte > 0x8f))))
                    return false;
            }
        }
        ++count;
    }
    return true;
}

std::vector<std::uint32_t> mask(xgrammar::GrammarMatcher& matcher, std::size_t domain) {
    std::vector<std::uint32_t> words((domain + 31) / 32);
    std::int64_t shape = static_cast<std::int64_t>(words.size());
    DLTensor view{.data        = words.data(),
                  .device      = {kDLCPU, 0},
                  .ndim        = 1,
                  .dtype       = {kDLInt, 32, 1},
                  .shape       = &shape,
                  .strides     = nullptr,
                  .byte_offset = 0};
    (void)matcher.FillNextTokenBitmask(&view);
    return words;
}

bool allowed(const std::vector<std::uint32_t>& words, int token) {
    return (words[static_cast<std::size_t>(token) / 32] & (1u << (token % 32))) != 0;
}
} // namespace

int main() {
    try {
        const std::vector<std::string> units{"a",   "b",       "你",   "😀",
                                             "\\n", "\\u0000", "\\\"", "\\\\"};
        std::vector<std::string> vocabulary{
            "a",        "b",        "ab",   "aaa",  "\"",    "\"!",  "a\"!",     "ab\"!",
            "!",        "你",       "😀",   "\xe4", "\xbd",  "\xa0", "\xe4\xbd", "\xbd\xa0",
            "\xf0\x9f", "\x98\x80", "a你",  "\\",   "n",     "u",    "0",        "\\n",
            "\\u00",    "00",       "\\\"", "\\\\", "\xf0",  "\x9f", "\x98",     "\x80",
            "A",        "B",        "\"A",  "\"B",  "ab\"A", "a\"B"};
        const int a_close = static_cast<int>(vocabulary.size());
        vocabulary.emplace_back("a\"");
        // More than 1000 accepted interiors force AdaptiveTokenMask's dense bitset route.
        // Repeated spellings are distinct token IDs and every bit still has the byte oracle.
        for (int i = 0; i < 1100; ++i)
            vocabulary.push_back(units[static_cast<std::size_t>(i) % units.size()]);
        for (int i = 0; i < 1100; ++i) vocabulary.emplace_back("REJECT");
        const int eos = static_cast<int>(vocabulary.size());
        vocabulary.emplace_back();
        vocabulary.emplace_back();
        xgrammar::GrammarCompiler compiler(xgrammar::TokenizerInfo(
            vocabulary, xgrammar::VocabType::RAW, static_cast<int>(vocabulary.size()),
            std::vector<std::int32_t>{eos}));

        struct Bound {
            int minimum;
            int maximum;
            std::string suffix;
        };

        for (const auto& branches : std::vector<std::vector<Bound>>{
                 {{0, 2, "!"}}, {{1, 2, "!"}}, {{2, 2, "!"}}, {{0, 1, "A"}, {2, 2, "B"}}}) {
            std::string source = "root ::= ";
            std::set<std::string> language, prefixes;
            for (const auto& bound : branches) {
                if (source != "root ::= ") source += " | ";
                source += "\"\\\"\" unit{" + std::to_string(bound.minimum) + "," +
                          std::to_string(bound.maximum) + "} \"\\\"" + bound.suffix + "\"";
                const std::string end = "\"" + bound.suffix;
                if (bound.minimum == 0) language.insert("\"" + end);
                for (const auto& a : units) {
                    if (bound.minimum <= 1)
                        language.insert(std::string("\"").append(a).append(end));
                    if (bound.maximum == 2)
                        for (const auto& b : units)
                            language.insert(std::string("\"").append(a).append(b).append(end));
                }
            }
            source += R"(
unit ::= "a" | "b" | "你" | "😀" | "\\n" | "\\u0000" | "\\\"" | "\\\\"
)";
            for (const auto& word : language)
                for (std::size_t n = 0; n <= word.size(); ++n) prefixes.insert(word.substr(0, n));
            const auto compiled = compiler.CompileGrammar(source);
            for (const auto& prefix : prefixes) {
                xgrammar::GrammarMatcher matcher(compiled);
                if (!matcher.AcceptString(prefix))
                    throw std::runtime_error("finite-language prefix rejected");
                const auto before = mask(matcher, vocabulary.size());
                for (int token = 0; token < static_cast<int>(before.size() * 32); ++token) {
                    const bool expected = token == eos
                                              ? language.contains(prefix)
                                              : token < static_cast<int>(vocabulary.size()) &&
                                                    !vocabulary[token].empty() &&
                                                    prefixes.contains(prefix + vocabulary[token]);
                    if (allowed(before, token) != expected)
                        throw std::runtime_error(
                            "counted mask differs from independently enumerated byte language");
                    if (!expected) continue;
                    if (!matcher.AcceptToken(token))
                        throw std::runtime_error("mask licensed a rejected token");
                    matcher.Rollback(1);
                    if (mask(matcher, vocabulary.size()) != before)
                        throw std::runtime_error("counted rollback changed committed mask");
                }
            }
        }
        for (int upper : {0, 1, 127, 128, 129, 200}) {
            const auto compiled = compiler.CompileJSONSchema(
                "{\"type\":\"string\",\"maxLength\":" + std::to_string(upper) + "}");
            for (int count : std::set<int>{0, std::max(0, upper - 1), upper}) {
                xgrammar::GrammarMatcher matcher(compiled);
                if (!matcher.AcceptString("\"" + std::string(count, 'a')))
                    throw std::runtime_error("bounded JSON prefix rejected");
                const auto words = mask(matcher, vocabulary.size());
                if (allowed(words, 0) != (count < upper) ||
                    allowed(words, 2) != (count + 2 <= upper) ||
                    allowed(words, a_close) != (count < upper) || !allowed(words, 4))
                    throw std::runtime_error("JSON scalar count boundary mask is wrong");
            }
        }
        // The production JSON scalar grammar must count every represented escape as one,
        // including a surrogate pair split between tokens, and reject isolated surrogates.
        const auto json_one = compiler.CompileJSONSchema(R"({"type":"string","maxLength":1})");
        for (const auto& [represented, scalar] :
             std::vector<std::pair<std::string, bool>>{{"你", true},
                                                       {"😀", true},
                                                       {"\\n", true},
                                                       {"\\u0000", true},
                                                       {"\\uD83D\\uDE00", true},
                                                       {"\\uD83D", false},
                                                       {"\\uDE00", false}}) {
            const std::string word = "\"" + represented + "\"";
            for (std::size_t length = 1; length < word.size(); ++length) {
                xgrammar::GrammarMatcher matcher(json_one);
                const auto prefix = word.substr(0, length);
                if (!matcher.AcceptString(prefix)) {
                    if (scalar) throw std::runtime_error("JSON scalar prefix rejected");
                    continue;
                }
                const auto actual = mask(matcher, vocabulary.size());
                // Enumerated JSON spellings of this scalar are enough to check transitions
                // along this represented word; full complete-string checks establish validity.
                for (int token : {0, 2, 4, 9, 10, 19, 20, 21, 22, 23, 24, 25}) {
                    auto probe = matcher.Fork();
                    if (allowed(actual, token) != probe.AcceptToken(token))
                        throw std::runtime_error("JSON scalar mask licensed a rejected fragment");
                }
            }
            xgrammar::GrammarMatcher completed(json_one);
            if ((completed.AcceptString(word) && completed.IsCompleted()) != scalar)
                throw std::runtime_error("JSON scalar escape language is wrong");
        }
        for (int upper : {0, 1, 2, 127, 128, 129, 200}) {
            const auto bounded = compiler.CompileJSONSchema(
                "{\"type\":\"string\",\"maxLength\":" + std::to_string(upper) + "}");
            std::set<std::string> states{"", "\"", "\"" + std::string(upper, 'a'),
                                         "\"" + std::string(upper, 'a') + "\""};
            if (upper > 0) {
                states.insert("\"" + std::string(upper - 1, 'a'));
                for (const std::string scalar : {"你", "😀", "\\n", "\\u0000", "\\uD83D\\uDE00"})
                    for (std::size_t n = 1; n < scalar.size(); ++n)
                        states.insert("\"" + scalar.substr(0, n));
            }
            for (const auto& prefix : states) {
                if (!json_string_prefix(prefix, upper))
                    throw std::runtime_error("invalid independent JSON oracle fixture");
                xgrammar::GrammarMatcher matcher(bounded);
                if (!matcher.AcceptString(prefix))
                    throw std::runtime_error("JSON oracle prefix rejected");
                const auto actual = mask(matcher, vocabulary.size());
                for (int token = 0; token < static_cast<int>(actual.size() * 32); ++token) {
                    const bool expected =
                        token == eos
                            ? (!prefix.empty() && prefix.back() == '"' && prefix.size() > 1)
                            : token < static_cast<int>(vocabulary.size()) &&
                                  !vocabulary[token].empty() &&
                                  json_string_prefix(prefix + vocabulary[token], upper);
                    if (allowed(actual, token) != expected)
                        throw std::runtime_error("bounded JSON dense mask disagrees with "
                                                 "independent scalar-prefix oracle at bound " +
                                                 std::to_string(upper) + " token " +
                                                 std::to_string(token));
                }
            }
        }
        std::cout
            << "PASS: counted scalar masks, UTF-8/escape fragments, continuation and rollback\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
