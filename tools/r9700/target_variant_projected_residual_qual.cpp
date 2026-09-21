#include "targets/qwen3_8_27b/impl/variant.h"
#include "ninfer/ops/projected_residual.h"

#include <stdexcept>
#include <cstdio>
#include <limits>

using ninfer::QType;
using ninfer::targets::qwen3::TextPhase;
using Variant = ninfer::targets::qwen3_8_27b::detail::Variant;

constexpr bool selected(bool candidate = true, std::uint32_t activation_bits = 8U,
                        bool inventory = true, TextPhase phase = TextPhase::Verify,
                        bool base_text = true, std::uint32_t tokens = 1U,
                        std::uint32_t rows = 5120U, std::uint32_t columns = 6144U,
                        QType weight = QType::Q4G64_F16S) {
    return Variant::ExecutionState::projected_residual_t1_selected(
        candidate, activation_bits, inventory, phase, base_text, tokens, rows, columns, weight);
}

static_assert(selected());
static_assert(selected(true, 8U, true, TextPhase::Verify, true, 1U, 5120U, 17408U));
static_assert(!selected(false));
static_assert(!selected(true, 4U));
static_assert(!selected(true, 8U, false));
static_assert(!selected(true, 8U, true, TextPhase::Prefill));
static_assert(!selected(true, 8U, true, TextPhase::Verify, false));
static_assert(!selected(true, 8U, true, TextPhase::Verify, true, 2U));
static_assert(!selected(true, 8U, true, TextPhase::Verify, true, 1U, 4096U));
static_assert(!selected(true, 8U, true, TextPhase::Verify, true, 1U, 5120U, 5120U));
static_assert(!selected(true, 8U, true, TextPhase::Verify, true, 1U, 5120U, 6144U,
                        QType::W8G32_F16S));

namespace {

// Only malformed calls reach the public Op: these synthetic non-overlapping addresses let
// the rejection regression run on a host with no GPU and no device allocations.
struct Binding {
    ninfer::Tensor input;
    ninfer::Weight weight;
    ninfer::Tensor residual;
    ninfer::DeviceSpan workspace;

    explicit Binding(std::int32_t columns)
        : input(reinterpret_cast<void*>(0x10000000U), ninfer::DType::BF16, {columns, 1}),
          residual(reinterpret_cast<void*>(0x40000000U), ninfer::DType::BF16, {5120, 1}),
          workspace{reinterpret_cast<void*>(0x50000000U),
                    ninfer::ops::projected_residual_t1_workspace_capacity_bytes(columns)} {
        weight.qtype = QType::Q4G64_F16S;
        weight.ndim = 2;
        weight.n = weight.shape[0] = weight.padded_shape[0] = 5120;
        weight.k = weight.shape[1] = weight.padded_shape[1] = columns;
        weight.group_size = weight.group = 64;
        weight.layout = ninfer::QuantLayout::Q4N16K16;
        weight.scale_dtype = ninfer::DType::FP16;
        weight.qdata = reinterpret_cast<void*>(0x20000000U);
        weight.scales = reinterpret_cast<void*>(0x30000000U);
        weight.qdata_bytes = std::size_t{5120} * columns / 2U;
        weight.scale_bytes = std::size_t{5120} * (columns / 64U) * sizeof(std::uint16_t);
    }

    void* pointer(unsigned plane) const {
        const void* pointers[]{input.data, weight.qdata, weight.scales, residual.data,
                               workspace.data};
        return const_cast<void*>(pointers[plane]);
    }

    void set_pointer(unsigned plane, void* value) {
        switch (plane) {
        case 0: input.data = value; break;
        case 1: weight.qdata = value; break;
        case 2: weight.scales = value; break;
        case 3: residual.data = value; break;
        case 4: workspace.data = value; break;
        }
    }
};

template<class Mutate>
void reject(std::int32_t columns, const char* label, Mutate mutate) {
    Binding binding(columns);
    mutate(binding);
    try {
        ninfer::ops::projected_residual_t1(binding.input, binding.weight, binding.residual,
                                          binding.workspace, nullptr);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(label);
}

void malformed_bindings(std::int32_t columns) {
    reject(columns, "short codes", [](auto& b) { --b.weight.qdata_bytes; });
    reject(columns, "oversized codes", [](auto& b) { ++b.weight.qdata_bytes; });
    reject(columns, "short scales", [](auto& b) { --b.weight.scale_bytes; });
    reject(columns, "oversized scales", [](auto& b) { ++b.weight.scale_bytes; });
    reject(columns, "short workspace", [](auto& b) { --b.workspace.bytes; });
    reject(columns, "oversized workspace", [](auto& b) { ++b.workspace.bytes; });
    reject(columns, "wrong input dtype", [](auto& b) { b.input.dtype = ninfer::DType::FP16; });
    reject(columns, "wrong residual dtype", [](auto& b) { b.residual.dtype = ninfer::DType::FP16; });
    reject(columns, "strided input", [](auto& b) { b.input.nb[0] *= 2; });
    reject(columns, "strided residual", [](auto& b) { b.residual.nb[0] *= 2; });
    reject(columns, "T2", [](auto& b) { b.input.ne[1] = 2; });
    reject(columns, "N4096", [](auto& b) { b.residual.ne[0] = 4096; });
    reject(columns, "K5120", [](auto& b) { b.input.ne[0] = b.weight.k =
        b.weight.shape[1] = b.weight.padded_shape[1] = 5120; });
    reject(columns, "wrong weight rank", [](auto& b) { b.weight.ndim = 3; });
    reject(columns, "wrong weight layout", [](auto& b) { b.weight.layout = ninfer::QuantLayout::RowSplit; });
    reject(columns, "wrong weight type", [](auto& b) { b.weight.qtype = QType::W8G32_F16S; });
    reject(columns, "wrong group", [](auto& b) { b.weight.group = 32; });
    reject(columns, "wrong scale type", [](auto& b) { b.weight.scale_dtype = ninfer::DType::BF16; });
    reject(columns, "padded width", [](auto& b) { b.weight.padded_shape[1] += 128; });
    reject(columns, "high plane", [](auto& b) { b.weight.qhigh = b.weight.qdata; });
    for (unsigned plane = 0; plane < 5; ++plane) {
        reject(columns, "null plane", [=](auto& b) { b.set_pointer(plane, nullptr); });
        reject(columns, "unaligned plane", [=](auto& b) {
            b.set_pointer(plane, reinterpret_cast<void*>(
                reinterpret_cast<std::uintptr_t>(b.pointer(plane)) + 1U));
        });
        reject(columns, "overflowing range", [=](auto& b) {
            b.set_pointer(plane, reinterpret_cast<void*>(
                std::numeric_limits<std::uintptr_t>::max() - 255U));
        });
        for (unsigned other = plane + 1; other < 5; ++other) {
            reject(columns, "aliased planes", [=](auto& b) {
                b.set_pointer(other, b.pointer(plane));
            });
            reject(columns, "partially overlapping planes", [=](auto& b) {
                b.set_pointer(other, reinterpret_cast<void*>(
                    reinterpret_cast<std::uintptr_t>(b.pointer(plane)) + 8U));
            });
        }
    }
}

} // namespace

int main() {
    try {
        malformed_bindings(6144);
        malformed_bindings(17408);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "projected residual host rejection failed: %s\n", error.what());
        return 1;
    }
    if (ninfer::ops::projected_residual_t1_workspace_capacity_bytes(6144) == 0U ||
        ninfer::ops::projected_residual_t1_workspace_capacity_bytes(17408) == 0U) {
        return 1;
    }
    try {
        (void)ninfer::ops::projected_residual_t1_workspace_capacity_bytes(5120);
    } catch (const std::invalid_argument&) {
        return 0;
    }
    return 1;
}
