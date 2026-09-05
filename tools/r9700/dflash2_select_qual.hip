#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/dflash2_path_select.h"
#include "ninfer/ops/linear.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kV = ninfer::ops::kDflash2PathSelectCodebookRows;
constexpr int kT = 2;
constexpr int kB = 2;
constexpr int kH = ninfer::ops::kDflash2PathSelectHidden;
constexpr int kR = ninfer::ops::kDflash2PathSelectRank;
constexpr int kTop = ninfer::ops::kDflash2PathSelectTopK;
constexpr int kW = 5;
constexpr int kW8Group = 32;
constexpr int kQ4Group = 64;

enum class ProjectionKind : std::uint8_t { BF16, W8, Q4 };

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

template <class T>
ninfer::DeviceBuffer upload(const std::vector<T>& host) {
    ninfer::DeviceBuffer result(host.size() * sizeof(T));
    result.copy_from_host(host.data(), result.size());
    return result;
}

template <class T>
std::vector<T> download(const ninfer::DeviceBuffer& device, std::size_t count) {
    std::vector<T> host(count);
    device.copy_to_host(host.data(), count * sizeof(T));
    return host;
}

std::uint16_t half_bits(float value) {
    return std::bit_cast<std::uint16_t>(static_cast<_Float16>(value));
}

double half_value(std::uint16_t bits) {
    return static_cast<double>(std::bit_cast<_Float16>(bits));
}

double bf16_value(hip_bfloat16 value) { return static_cast<double>(static_cast<float>(value)); }

std::size_t column_index(int token, int batch) {
    return static_cast<std::size_t>(batch) * kT + token;
}

struct ProjectionFixture {
    std::vector<hip_bfloat16> bf16;
    std::vector<std::int8_t> codes;
    std::vector<std::uint16_t> scales;
    std::vector<std::uint8_t> q4_codes;
    std::vector<std::uint16_t> q4_scales;
    std::vector<std::uint8_t> q4_codes_row;
    std::vector<std::uint16_t> q4_scales_row;
    ninfer::DeviceBuffer d_bf16;
    ninfer::DeviceBuffer d_codes;
    ninfer::DeviceBuffer d_scales;
    ninfer::DeviceBuffer d_q4_codes;
    ninfer::DeviceBuffer d_q4_scales;
    ninfer::Weight bf16_view;
    ninfer::Weight w8_view;
    ninfer::Weight q4_view;

    [[nodiscard]] double bf16_at(int row, int column) const {
        return bf16_value(bf16[static_cast<std::size_t>(row) * kH + column]);
    }

    [[nodiscard]] double w8_at(int row, int column) const {
        const auto code = codes[static_cast<std::size_t>(row) * kH + column];
        const auto scale = scales[static_cast<std::size_t>(row) * (kH / kW8Group) +
                                  column / kW8Group];
        return static_cast<double>(code) * half_value(scale);
    }

    [[nodiscard]] double q4_at(int row, int column) const {
        const std::size_t index = static_cast<std::size_t>(row) * kH + column;
        const unsigned nibble =
            (q4_codes_row[index >> 1U] >> ((index & 1U) * 4U)) & 0x0fU;
        const int code = nibble >= 8U ? static_cast<int>(nibble) - 16
                                      : static_cast<int>(nibble);
        const auto scale = q4_scales_row[static_cast<std::size_t>(row) * (kH / kQ4Group) +
                                         column / kQ4Group];
        return static_cast<double>(code) * half_value(scale);
    }
};

void set_nibble(std::vector<std::uint8_t>& packed, std::size_t index, int code) {
    packed[index >> 1U] |=
        static_cast<std::uint8_t>((code & 0x0f) << ((index & 1U) * 4U));
}

ProjectionFixture make_projection() {
    ProjectionFixture result;
    result.bf16.resize(static_cast<std::size_t>(kR) * kH);
    result.codes.resize(static_cast<std::size_t>(kR) * kH);
    result.scales.resize(static_cast<std::size_t>(kR) * (kH / kW8Group));
    result.q4_codes.resize(static_cast<std::size_t>(kR) * kH / 2U);
    result.q4_scales.resize(static_cast<std::size_t>(kR) * (kH / kQ4Group));
    result.q4_codes_row.resize(result.q4_codes.size());
    result.q4_scales_row.resize(result.q4_scales.size());
    for (int row = 0; row < kR; ++row) {
        for (int group = 0; group < kH / kW8Group; ++group) {
            const float scale = static_cast<float>(1 + (row + 3 * group) % 5) / 2048.0F;
            result.scales[static_cast<std::size_t>(row) * (kH / kW8Group) + group] =
                half_bits(scale);
            for (int lane = 0; lane < kW8Group; ++lane) {
                const int column = group * kW8Group + lane;
                const int code = (row * 13 + group * 7 + lane * 5) % 31 - 15;
                result.codes[static_cast<std::size_t>(row) * kH + column] =
                    static_cast<std::int8_t>(code);
                result.bf16[static_cast<std::size_t>(row) * kH + column] =
                    hip_bfloat16(static_cast<float>(code) * scale);
            }
        }
    }
    for (int row = 0; row < kR; ++row) {
        for (int group = 0; group < kH / kQ4Group; ++group) {
            const float scale = static_cast<float>(1 + (row + 3 * group) % 5) / 1024.0F;
            constexpr std::size_t groups = kH / kQ4Group;
            const std::size_t logical_scale = static_cast<std::size_t>(row) * groups + group;
            const std::size_t n16_scale =
                ((static_cast<std::size_t>(row) >> 4U) * groups + group) * 16U +
                (static_cast<std::size_t>(row) & 15U);
            result.q4_scales_row[logical_scale] = half_bits(scale);
            result.q4_scales[n16_scale] = half_bits(scale);
            for (int lane = 0; lane < kQ4Group; ++lane) {
                const int code = (row * 13 + group * 7 + lane * 5) % 15 - 7;
                const std::size_t logical =
                    static_cast<std::size_t>(row) * kH + group * kQ4Group + lane;
                set_nibble(result.q4_codes_row, logical, code);
                const std::size_t pair = static_cast<std::size_t>(lane) / 16U;
                const std::size_t within = static_cast<std::size_t>(lane) % 16U;
                const std::size_t word =
                    (((static_cast<std::size_t>(row) >> 4U) * groups + group) * 4U + pair) *
                        16U +
                    (static_cast<std::size_t>(row) & 15U);
                set_nibble(result.q4_codes, word * 16U + within, code);
            }
        }
    }
    result.d_bf16 = upload(result.bf16);
    result.d_codes = upload(result.codes);
    result.d_scales = upload(result.scales);
    result.d_q4_codes = upload(result.q4_codes);
    result.d_q4_scales = upload(result.q4_scales);

    result.bf16_view.payload = result.d_bf16.data();
    result.bf16_view.payload_bytes = result.d_bf16.size();
    result.bf16_view.qdata = result.d_bf16.data();
    result.bf16_view.qdata_bytes = result.d_bf16.size();
    result.bf16_view.qtype = ninfer::QType::BF16_CTRL;
    result.bf16_view.layout = ninfer::QuantLayout::Contiguous;

    result.w8_view.payload = result.d_codes.data();
    result.w8_view.payload_bytes = result.d_codes.size() + result.d_scales.size();
    result.w8_view.qdata = result.d_codes.data();
    result.w8_view.qdata_bytes = result.d_codes.size();
    result.w8_view.scales = result.d_scales.data();
    result.w8_view.scale_bytes = result.d_scales.size();
    result.w8_view.qtype = ninfer::QType::W8G32_F16S;
    result.w8_view.group_size = kW8Group;
    result.w8_view.group = kW8Group;
    result.w8_view.layout = ninfer::QuantLayout::RowSplit;
    result.w8_view.scale_dtype = ninfer::DType::FP16;
    result.w8_view.scale_ne[0] = kH / kW8Group;
    result.w8_view.scale_ne[1] = kR;
    result.w8_view.scale_nb[0] = sizeof(std::uint16_t);
    result.w8_view.scale_nb[1] = (kH / kW8Group) * sizeof(std::uint16_t);

    result.q4_view.payload = result.d_q4_codes.data();
    result.q4_view.payload_bytes = result.d_q4_codes.size() + result.d_q4_scales.size();
    result.q4_view.qdata = result.d_q4_codes.data();
    result.q4_view.qdata_bytes = result.d_q4_codes.size();
    result.q4_view.scales = result.d_q4_scales.data();
    result.q4_view.scale_bytes = result.d_q4_scales.size();
    result.q4_view.qtype = ninfer::QType::Q4G64_F16S;
    result.q4_view.group_size = kQ4Group;
    result.q4_view.group = kQ4Group;
    result.q4_view.layout = ninfer::QuantLayout::Q4N16K16;
    result.q4_view.scale_dtype = ninfer::DType::FP16;
    result.q4_view.scale_ne[0] = kH / kQ4Group;
    result.q4_view.scale_ne[1] = kR;
    result.q4_view.scale_nb[0] = sizeof(std::uint16_t);
    result.q4_view.scale_nb[1] = (kH / kQ4Group) * sizeof(std::uint16_t);

    for (ninfer::Weight* weight : {&result.bf16_view, &result.w8_view, &result.q4_view}) {
        weight->ndim = 2;
        weight->n = kR;
        weight->k = kH;
        weight->shape[0] = kR;
        weight->shape[1] = kH;
        weight->padded_shape[0] = kR;
        weight->padded_shape[1] = kH;
    }
    return result;
}

struct Corpus {
    std::vector<hip_bfloat16> logits;
    std::vector<hip_bfloat16> hidden;
    std::vector<hip_bfloat16> predecessor;
    std::vector<hip_bfloat16> successor;
    std::vector<std::int32_t> anchors{17, 29};
    std::vector<std::int32_t> positions{101, 4093};
    std::vector<std::int32_t> mapped_ids;
    std::array<int, 18> special_rows{
        7, 8192, 31, 65537, 1024, 131071, 16384, 248319, 4097,
        98303, 2, 196607, 32768, 77, 225279, 123457, 2048, 147455};
};

void fill_code(std::vector<hip_bfloat16>& book, int token, int salt) {
    for (int rank = 0; rank < kR; ++rank) {
        const int raw = (token * (salt + 3) + rank * (salt + 11) + salt * 19) % 29 - 14;
        book[static_cast<std::size_t>(token) * kR + rank] =
            hip_bfloat16(static_cast<float>(raw) / 64.0F);
    }
}

Corpus make_corpus() {
    Corpus result;
    result.logits.assign(static_cast<std::size_t>(kV) * kT * kB, hip_bfloat16(-6.0F));
    result.hidden.resize(static_cast<std::size_t>(kH) * kT * kB);
    result.predecessor.assign(static_cast<std::size_t>(kR) * kV, hip_bfloat16(0.0F));
    result.successor.assign(static_cast<std::size_t>(kR) * kV, hip_bfloat16(0.0F));
    result.mapped_ids.resize(kV);
    for (int token = 0; token < kV; ++token) result.mapped_ids[token] = token;

    for (int batch = 0; batch < kB; ++batch) {
        for (int token = 0; token < kT; ++token) {
            const std::size_t column = column_index(token, batch);
            for (int feature = 0; feature < kH; ++feature) {
                const int raw = (feature * 17 + token * 13 + batch * 23) % 41 - 20;
                result.hidden[column * kH + feature] =
                    hip_bfloat16(static_cast<float>(raw) / 128.0F);
            }
            for (int candidate = 0; candidate < 18; ++candidate) {
                int level = candidate;
                if (candidate == 5) level = 4;
                if (candidate == 11) level = 10;
                const float value = 4.0F - static_cast<float>(level) * 0.25F +
                                    static_cast<float>(token + 2 * batch) / 64.0F;
                result.logits[column * kV + result.special_rows[candidate]] =
                    hip_bfloat16(value);
            }
        }
    }

    std::vector<int> code_tokens(result.anchors.begin(), result.anchors.end());
    for (int index = 0; index < 18; ++index) {
        const int row = result.special_rows[index];
        const int mapped = 2000 + (index * 7) % 19;
        result.mapped_ids[row] = mapped;
        code_tokens.push_back(row);
        code_tokens.push_back(mapped);
    }
    std::sort(code_tokens.begin(), code_tokens.end());
    code_tokens.erase(std::unique(code_tokens.begin(), code_tokens.end()), code_tokens.end());
    for (int token : code_tokens) {
        fill_code(result.predecessor, token, 5);
        fill_code(result.successor, token, 13);
    }
    return result;
}

struct Candidate { double unary; int row; int token; };

bool topk_better(const Candidate& lhs, const Candidate& rhs) {
    return lhs.unary > rhs.unary || (lhs.unary == rhs.unary && lhs.row < rhs.row);
}

std::vector<Candidate> topk(const Corpus& corpus, int token, int batch, bool mapped) {
    const std::size_t column = column_index(token, batch);
    std::vector<Candidate> values;
    values.reserve(kV);
    for (int row = 0; row < kV; ++row) {
        const double value = bf16_value(corpus.logits[column * kV + row]);
        if (!std::isfinite(value)) continue;
        values.push_back({value, row, mapped ? corpus.mapped_ids[row] : row});
    }
    std::partial_sort(values.begin(), values.begin() + kTop, values.end(), topk_better);
    values.resize(kTop);
    return values;
}

std::vector<double> q4_activations(const Corpus& corpus) {
    std::vector<double> result(static_cast<std::size_t>(kH) * kT * kB);
    for (std::size_t column = 0; column < static_cast<std::size_t>(kT * kB); ++column) {
        for (int group = 0; group < kH / kQ4Group; ++group) {
            float maximum = 0.0F;
            for (int lane = 0; lane < kQ4Group; ++lane) {
                maximum = std::max(maximum, std::abs(static_cast<float>(
                    corpus.hidden[column * kH + group * kQ4Group + lane])));
            }
            std::uint16_t bits = maximum == 0.0F ? 0U : half_bits(maximum / 127.0F);
            if (maximum != 0.0F && bits == 0U) bits = 1U;
            const double scale = half_value(bits);
            for (int lane = 0; lane < kQ4Group; ++lane) {
                const std::size_t index = column * kH + group * kQ4Group + lane;
                const double value = bf16_value(corpus.hidden[index]);
                const int code = bits == 0U
                                     ? 0
                                     : std::clamp(static_cast<int>(std::nearbyint(value / scale)),
                                                  -127, 127);
                result[index] = static_cast<double>(code) * scale;
            }
        }
    }
    return result;
}

std::vector<hip_bfloat16> project_hidden(const Corpus& corpus,
                                         const ProjectionFixture& projection,
                                         ProjectionKind kind) {
    std::vector<hip_bfloat16> result(static_cast<std::size_t>(kR) * kT * kB);
    const std::vector<double> q4_input =
        kind == ProjectionKind::Q4 ? q4_activations(corpus) : std::vector<double>{};
    for (int batch = 0; batch < kB; ++batch) {
        for (int token = 0; token < kT; ++token) {
            const std::size_t column = column_index(token, batch);
            for (int row = 0; row < kR; ++row) {
                double sum = 0.0;
                for (int feature = 0; feature < kH; ++feature) {
                    const double input = kind == ProjectionKind::Q4
                                             ? q4_input[column * kH + feature]
                                             : bf16_value(corpus.hidden[column * kH + feature]);
                    const double weight = kind == ProjectionKind::W8
                                              ? projection.w8_at(row, feature)
                                              : kind == ProjectionKind::Q4
                                                    ? projection.q4_at(row, feature)
                                                    : projection.bf16_at(row, feature);
                    sum = std::fma(input, weight, sum);
                }
                result[column * kR + row] = hip_bfloat16(static_cast<float>(sum));
            }
        }
    }
    return result;
}

double markov_score(const Corpus& corpus, const std::vector<hip_bfloat16>& projected,
                    const Candidate& candidate, int previous, int token, int batch) {
    const std::size_t column = column_index(token, batch);
    double score = candidate.unary;
    for (int rank = 0; rank < kR; ++rank) {
        const double h = bf16_value(projected[column * kR + rank]);
        const double pred = bf16_value(
            corpus.predecessor[static_cast<std::size_t>(previous) * kR + rank]);
        const double succ = bf16_value(
            corpus.successor[static_cast<std::size_t>(candidate.token) * kR + rank]);
        score = std::fma(pred * h, succ, score);
    }
    return score;
}

struct ChainOracle {
    std::vector<std::int32_t> path;
    std::vector<std::int32_t> selector_ids;
    std::vector<float> selector_q;
};

ChainOracle chain_oracle(const Corpus& corpus, const std::vector<hip_bfloat16>& projected,
                         bool mapped) {
    ChainOracle out;
    out.path.resize(kT * kB);
    out.selector_ids.resize(kTop * kT * kB);
    out.selector_q.assign(kTop * kT * kB, 0.0F);
    for (int batch = 0; batch < kB; ++batch) {
        int previous = corpus.anchors[batch];
        for (int token = 0; token < kT; ++token) {
            const auto candidates = topk(corpus, token, batch, mapped);
            int pick = 0;
            double best = markov_score(corpus, projected, candidates[0], previous, token, batch);
            for (int index = 1; index < kTop; ++index) {
                const double score = markov_score(
                    corpus, projected, candidates[index], previous, token, batch);
                if (score > best ||
                    (score == best && candidates[index].token < candidates[pick].token)) {
                    best = score;
                    pick = index;
                }
            }
            const std::size_t column = column_index(token, batch);
            out.path[column] = candidates[pick].token;
            for (int index = 0; index < kTop; ++index) {
                out.selector_ids[column * kTop + index] = candidates[index].token;
            }
            out.selector_q[column * kTop + pick] = 1.0F;
            previous = candidates[pick].token;
        }
    }
    return out;
}

struct TreeOracle {
    std::vector<std::int32_t> verify, parent, cache, rope, mask, live;
};

TreeOracle tree_oracle(const Corpus& corpus, const std::vector<hip_bfloat16>& projected,
                       bool mapped) {
    TreeOracle out;
    out.verify.resize(kW * kB);
    out.parent.resize(kW * kB);
    out.cache.resize(kW * kB);
    out.rope.resize(kW * kB);
    out.mask.resize(kW * kB);
    out.live.resize(kB);
    for (int batch = 0; batch < kB; ++batch) {
        struct Node { int id; int parent; int depth; double score; };
        std::vector<Node> nodes{{corpus.anchors[batch], -1, 0, 0.0}};
        std::vector<int> frontier{0};
        for (int token = 0; token < kT; ++token) {
            const auto candidates = topk(corpus, token, batch, mapped);
            struct Expansion { double score; int parent; int candidate; };
            std::vector<Expansion> expansions;
            for (int parent : frontier) {
                for (const Candidate& candidate : candidates) {
                    expansions.push_back({nodes[parent].score + markov_score(
                                              corpus, projected, candidate, nodes[parent].id,
                                              token, batch),
                                          parent, candidate.token});
                }
            }
            std::sort(expansions.begin(), expansions.end(), [](const Expansion& lhs,
                                                               const Expansion& rhs) {
                if (lhs.score != rhs.score) return lhs.score > rhs.score;
                if (lhs.candidate != rhs.candidate) return lhs.candidate < rhs.candidate;
                return lhs.parent < rhs.parent;
            });
            frontier.clear();
            for (int beam = 0; beam < 2 && beam < static_cast<int>(expansions.size()); ++beam) {
                const Expansion& item = expansions[beam];
                frontier.push_back(static_cast<int>(nodes.size()));
                nodes.push_back({item.candidate, item.parent, token + 1, item.score});
            }
        }
        const int count = std::min(kW, static_cast<int>(nodes.size()));
        out.live[batch] = count;
        for (int index = 0; index < count; ++index) {
            const std::size_t dst = static_cast<std::size_t>(batch) * kW + index;
            out.verify[dst] = nodes[index].id;
            out.parent[dst] = index == 0 ? -1 : nodes[index].parent;
            out.cache[dst] = corpus.positions[batch] + index;
            out.rope[dst] = corpus.positions[batch] + nodes[index].depth;
            int mask = 0;
            for (int current = index; current >= 0; current = nodes[current].parent) {
                mask |= 1 << current;
            }
            out.mask[dst] = mask;
        }
        for (int index = count; index < kW; ++index) {
            const std::size_t dst = static_cast<std::size_t>(batch) * kW + index;
            const std::size_t last = static_cast<std::size_t>(batch) * kW + count - 1;
            out.verify[dst] = out.verify[last];
            out.parent[dst] = out.parent[last];
            out.cache[dst] = corpus.positions[batch] + index;
            out.rope[dst] = out.rope[last];
            out.mask[dst] = out.mask[last];
        }
    }
    return out;
}

struct DeviceFixture {
    ninfer::DeviceBuffer logits, hidden, predecessor, successor, anchors, positions, mapped_ids,
        configs;
    ninfer::DeviceBuffer path{static_cast<std::size_t>(kT * kB) * sizeof(std::int32_t)};
    ninfer::DeviceBuffer selector_ids{
        static_cast<std::size_t>(kTop * kT * kB) * sizeof(std::int32_t)};
    ninfer::DeviceBuffer selector_q{static_cast<std::size_t>(kTop * kT * kB) * sizeof(float)};

    explicit DeviceFixture(const Corpus& corpus)
        : logits(upload(corpus.logits)), hidden(upload(corpus.hidden)),
          predecessor(upload(corpus.predecessor)), successor(upload(corpus.successor)),
          anchors(upload(corpus.anchors)), positions(upload(corpus.positions)),
          mapped_ids(upload(corpus.mapped_ids)),
          configs(upload(std::vector<ninfer::ops::SamplingConfig>(kB))) {}
};

template <class T>
void require_equal(const std::vector<T>& actual, const std::vector<T>& expected,
                   const std::string& label) {
    if (actual.size() != expected.size()) fail(label + ": extent mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            fail(label + ": mismatch at " + std::to_string(index));
        }
    }
}

void require_workspace_peak(const ninfer::WorkspaceArena& workspace, ninfer::QType qtype,
                            const char* label) {
    constexpr std::size_t columns = static_cast<std::size_t>(kT * kB);
    constexpr std::size_t split = columns * 32U * kTop;
    constexpr std::size_t merged = columns * kTop;
    constexpr std::size_t topk_bytes =
        (split + merged) * (sizeof(float) + sizeof(std::int32_t));
    const std::size_t expected = static_cast<std::size_t>(kR) * kT * kB * sizeof(std::uint16_t) +
        std::max(topk_bytes,
                 ninfer::ops::linear_workspace_capacity_bytes(qtype, kT * kB, kH));
    if (workspace.peak_used() != expected || workspace.peak_used() > workspace.capacity()) {
        fail(std::string(label) + " workspace query/execution mismatch: peak=" +
             std::to_string(workspace.peak_used()) + " expected=" +
             std::to_string(expected) + " capacity=" +
             std::to_string(workspace.capacity()));
    }
}

void run_chain(ninfer::DeviceContext& device, const Corpus& corpus,
               const ProjectionFixture& projection, DeviceFixture& data, ProjectionKind kind,
               bool mapped) {
    ninfer::Tensor logits(data.logits.data(), ninfer::DType::BF16, {kV, kT, kB});
    ninfer::Tensor hidden(data.hidden.data(), ninfer::DType::BF16, {kH, kT, kB});
    ninfer::Tensor pred(data.predecessor.data(), ninfer::DType::BF16, {kR, kV});
    ninfer::Tensor succ(data.successor.data(), ninfer::DType::BF16, {kR, kV});
    ninfer::Tensor anchors(data.anchors.data(), ninfer::DType::I32, {kB});
    ninfer::Tensor positions(data.positions.data(), ninfer::DType::I32, {kB});
    ninfer::Tensor mapped_ids(data.mapped_ids.data(), ninfer::DType::I32, {kV});
    ninfer::Tensor path(data.path.data(), ninfer::DType::I32, {kT, kB});
    ninfer::Tensor selector_ids(data.selector_ids.data(), ninfer::DType::I32, {kTop, kT, kB});
    ninfer::Tensor selector_q(data.selector_q.data(), ninfer::DType::FP32, {kTop, kT, kB});
    const ninfer::Weight& weight = kind == ProjectionKind::BF16
                                       ? projection.bf16_view
                                       : kind == ProjectionKind::W8 ? projection.w8_view
                                                                    : projection.q4_view;
    const auto projected = project_hidden(corpus, projection, kind);
    const ChainOracle expected = chain_oracle(corpus, projected, mapped);
    ninfer::WorkspaceArena workspace(ninfer::ops::dflash2_path_select_workspace_capacity_bytes(
        weight.qtype, kT, kT, kB));
    ninfer::ops::dflash2_path_select(
        logits, hidden, weight, pred, succ, anchors, positions,
        static_cast<const ninfer::ops::SamplingConfig*>(data.configs.data()), path, workspace,
        device.stream, mapped ? &mapped_ids : nullptr, &selector_ids, &selector_q);
    device.synchronize();
    require_equal(download<std::int32_t>(data.path, expected.path.size()), expected.path,
                  "chain path");
    require_equal(download<std::int32_t>(data.selector_ids, expected.selector_ids.size()),
                  expected.selector_ids, "chain top-k support");
    require_equal(download<float>(data.selector_q, expected.selector_q.size()),
                  expected.selector_q, "chain selector q");
    require_workspace_peak(workspace, weight.qtype, "chain");
}

void run_tree(ninfer::DeviceContext& device, const Corpus& corpus,
              const ProjectionFixture& projection, DeviceFixture& data, ProjectionKind kind) {
    ninfer::Tensor logits(data.logits.data(), ninfer::DType::BF16, {kV, kT, kB});
    ninfer::Tensor hidden(data.hidden.data(), ninfer::DType::BF16, {kH, kT, kB});
    ninfer::Tensor pred(data.predecessor.data(), ninfer::DType::BF16, {kR, kV});
    ninfer::Tensor succ(data.successor.data(), ninfer::DType::BF16, {kR, kV});
    ninfer::Tensor anchors(data.anchors.data(), ninfer::DType::I32, {kB});
    ninfer::Tensor frontiers(data.positions.data(), ninfer::DType::I32, {kB});
    ninfer::Tensor mapped_ids(data.mapped_ids.data(), ninfer::DType::I32, {kV});
    ninfer::DeviceBuffer d_verify(kW * kB * sizeof(std::int32_t));
    ninfer::DeviceBuffer d_parent(kW * kB * sizeof(std::int32_t));
    ninfer::DeviceBuffer d_cache(kW * kB * sizeof(std::int32_t));
    ninfer::DeviceBuffer d_rope(kW * kB * sizeof(std::int32_t));
    ninfer::DeviceBuffer d_mask(kW * kB * sizeof(std::int32_t));
    ninfer::DeviceBuffer d_live(kB * sizeof(std::int32_t));
    ninfer::Tensor verify(d_verify.data(), ninfer::DType::I32, {kW, kB});
    ninfer::Tensor parent(d_parent.data(), ninfer::DType::I32, {kW, kB});
    ninfer::Tensor cache(d_cache.data(), ninfer::DType::I32, {kW, kB});
    ninfer::Tensor rope(d_rope.data(), ninfer::DType::I32, {kW, kB});
    ninfer::Tensor mask(d_mask.data(), ninfer::DType::I32, {kW, kB});
    ninfer::Tensor live(d_live.data(), ninfer::DType::I32, {kB});
    const ninfer::Weight& weight = kind == ProjectionKind::W8 ? projection.w8_view
                                                              : projection.q4_view;
    const auto projected = project_hidden(corpus, projection, kind);
    const TreeOracle expected = tree_oracle(corpus, projected, true);
    ninfer::WorkspaceArena workspace(ninfer::ops::dflash2_path_select_workspace_capacity_bytes(
        weight.qtype, kT, kT, kB));
    ninfer::ops::dflash2_tree_select(logits, hidden, weight, pred, succ, anchors, frontiers,
                                     verify, parent, cache, rope, mask, live, workspace,
                                     device.stream, &mapped_ids);
    device.synchronize();
    require_equal(download<std::int32_t>(d_verify, expected.verify.size()), expected.verify,
                  "tree verify ids");
    require_equal(download<std::int32_t>(d_parent, expected.parent.size()), expected.parent,
                  "tree parent index");
    require_equal(download<std::int32_t>(d_cache, expected.cache.size()), expected.cache,
                  "tree cache positions");
    require_equal(download<std::int32_t>(d_rope, expected.rope.size()), expected.rope,
                  "tree rope positions");
    require_equal(download<std::int32_t>(d_mask, expected.mask.size()), expected.mask,
                  "tree ancestor mask");
    require_equal(download<std::int32_t>(d_live, expected.live.size()), expected.live,
                  "tree live columns");
    require_workspace_peak(workspace, weight.qtype, "tree");
}

template <class Function>
void expect_invalid(Function&& function, const char* label) {
    try {
        function();
        fail(std::string(label) + ": malformed weight was accepted");
    } catch (const std::invalid_argument&) {
    }
}

void run_malformed_planes(ninfer::DeviceContext& device, const ProjectionFixture& projection,
                          DeviceFixture& data) {
    ninfer::Tensor logits(data.logits.data(), ninfer::DType::BF16, {kV, kT, kB});
    ninfer::Tensor hidden(data.hidden.data(), ninfer::DType::BF16, {kH, kT, kB});
    ninfer::Tensor pred(data.predecessor.data(), ninfer::DType::BF16, {kR, kV});
    ninfer::Tensor succ(data.successor.data(), ninfer::DType::BF16, {kR, kV});
    ninfer::Tensor anchors(data.anchors.data(), ninfer::DType::I32, {kB});
    ninfer::Tensor positions(data.positions.data(), ninfer::DType::I32, {kB});
    ninfer::Tensor path(data.path.data(), ninfer::DType::I32, {kT, kB});
    ninfer::WorkspaceArena workspace(ninfer::ops::dflash2_path_select_workspace_capacity_bytes(
        projection.w8_view.qtype, kT, kT, kB));
    auto invoke = [&](const ninfer::Weight& weight) {
        workspace.reset();
        ninfer::ops::dflash2_path_select(
            logits, hidden, weight, pred, succ, anchors, positions,
            static_cast<const ninfer::ops::SamplingConfig*>(data.configs.data()), path, workspace,
            device.stream);
    };
    ninfer::Weight short_codes = projection.w8_view;
    --short_codes.qdata_bytes;
    expect_invalid([&] { invoke(short_codes); }, "short W8 code plane");
    ninfer::Weight short_scales = projection.w8_view;
    short_scales.scale_bytes -= sizeof(std::uint16_t);
    expect_invalid([&] { invoke(short_scales); }, "short W8 scale plane");
    ninfer::Weight overlap = projection.w8_view;
    overlap.scales = overlap.qdata;
    expect_invalid([&] { invoke(overlap); }, "overlapping W8 planes");
    ninfer::Weight high_plane = projection.w8_view;
    high_plane.qhigh = high_plane.qdata;
    high_plane.high_plane_bytes = 1;
    expect_invalid([&] { invoke(high_plane); }, "unexpected W8 high plane");

    ninfer::WorkspaceArena q4_workspace(
        ninfer::ops::dflash2_path_select_workspace_capacity_bytes(
            projection.q4_view.qtype, kT, kT, kB));
    auto invoke_q4 = [&](const ninfer::Weight& weight) {
        q4_workspace.reset();
        ninfer::ops::dflash2_path_select(
            logits, hidden, weight, pred, succ, anchors, positions,
            static_cast<const ninfer::ops::SamplingConfig*>(data.configs.data()), path,
            q4_workspace, device.stream);
    };
    ninfer::Weight short_q4_codes = projection.q4_view;
    --short_q4_codes.qdata_bytes;
    expect_invalid([&] { invoke_q4(short_q4_codes); }, "short Q4 code plane");
    ninfer::Weight short_q4_scales = projection.q4_view;
    short_q4_scales.scale_bytes -= sizeof(std::uint16_t);
    expect_invalid([&] { invoke_q4(short_q4_scales); }, "short Q4 scale plane");
    ninfer::Weight overlap_q4 = projection.q4_view;
    overlap_q4.scales = overlap_q4.qdata;
    expect_invalid([&] { invoke_q4(overlap_q4); }, "overlapping Q4 planes");
    ninfer::Weight retired_q4_layout = projection.q4_view;
    retired_q4_layout.layout = ninfer::QuantLayout::RowSplit;
    expect_invalid([&] { invoke_q4(retired_q4_layout); }, "retired Q4 RowSplit layout");
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device;
        const Corpus corpus = make_corpus();
        const ProjectionFixture projection = make_projection();
        DeviceFixture data(corpus);
        run_chain(device, corpus, projection, data, ProjectionKind::BF16, false);
        run_chain(device, corpus, projection, data, ProjectionKind::W8, false);
        run_chain(device, corpus, projection, data, ProjectionKind::W8, true);
        run_chain(device, corpus, projection, data, ProjectionKind::Q4, false);
        run_chain(device, corpus, projection, data, ProjectionKind::Q4, true);
        run_tree(device, corpus, projection, data, ProjectionKind::W8);
        run_tree(device, corpus, projection, data, ProjectionKind::Q4);
        run_malformed_planes(device, projection, data);
        std::cout << "dflash2_select: PASS gfx1201 BF16+W8G32+Q4G64 T=2 B=2 "
                     "independent-FP64 identity/mapped top-k/ties chain+tree "
                     "caller-owned-arena malformed-planes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dflash2_select: FAIL: " << error.what() << '\n';
        return 1;
    }
}
