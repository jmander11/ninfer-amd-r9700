#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include <hip/hip_runtime.h>

namespace ninfer::ops::r9700::linear {

// Exact host ABI recovered from the retained, emitted qualification object.
// This declaration does not provide a second implementation: it only permits
// the disconnected harness to call the preserved code object byte-for-byte.
struct Fp8GateUpM128N256QualificationArgs {
    const std::uint8_t* activation_codes;
    std::size_t activation_code_count;
    const float* token_scales;
    std::size_t token_scale_bytes;
    const std::uint32_t* status;
    std::uint32_t tokens;
    std::uint32_t activation_columns;
    std::uint32_t weight_columns;
    const std::uint8_t* weight_codes;
    std::size_t weight_code_count;
    const float* weight_scales;
    std::size_t weight_scale_bytes;
    std::uint16_t* output_bf16;
    std::size_t output_element_count;
};

struct Fp8GateUpM128N256QualificationResources {
    int registers;
    int static_lds_bytes;
    int private_bytes;
    int maximum_threads;
    int active_blocks_per_wgp;
};

hipError_t fp8_gate_up_m128n256_qualification(
    const Fp8GateUpM128N256QualificationArgs&, hipStream_t);
Fp8GateUpM128N256QualificationResources
fp8_gate_up_m128n256_qualification_resources();

static_assert(sizeof(Fp8GateUpM128N256QualificationArgs) == 104);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, activation_codes) == 0);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, activation_code_count) == 8);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, token_scales) == 16);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, token_scale_bytes) == 24);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, status) == 32);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, tokens) == 40);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, activation_columns) == 44);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, weight_columns) == 48);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, weight_codes) == 56);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, weight_code_count) == 64);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, weight_scales) == 72);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, weight_scale_bytes) == 80);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, output_bf16) == 88);
static_assert(offsetof(Fp8GateUpM128N256QualificationArgs, output_element_count) == 96);

} // namespace ninfer::ops::r9700::linear

namespace ninfer::qualification::fp8_gate_up_m128n256_retained {

inline constexpr std::uint32_t kTokens = 2048;
inline constexpr std::uint32_t kRows = 34816;
inline constexpr std::uint32_t kColumns = 5120;
inline constexpr std::uint32_t kTileM = 128;
inline constexpr std::uint32_t kTileN = 256;
inline constexpr std::uint32_t kTileK = 32;
inline constexpr std::uint32_t kThreads = 1024;

struct Arguments {
    const std::uint8_t* activation_codes = nullptr;
    const float* token_scales = nullptr;
    const std::uint8_t* weight_codes = nullptr;
    const float* weight_scales = nullptr;
    std::uint16_t* output_bf16 = nullptr;
    const std::uint32_t* status = nullptr;
    std::uint32_t tokens = kTokens;
    std::uint32_t rows = kRows;
    std::uint32_t columns = kColumns;
};

struct Resources {
    int registers = 0;
    int static_lds_bytes = 0;
    int private_bytes = 0;
    int maximum_threads = 0;
    int active_blocks_per_wgp = 0;
};

inline bool disjoint(const void* first, std::size_t first_bytes,
                     const void* second, std::size_t second_bytes) {
    const auto a = reinterpret_cast<std::uintptr_t>(first);
    const auto b = reinterpret_cast<std::uintptr_t>(second);
    if (a > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
        b > std::numeric_limits<std::uintptr_t>::max() - second_bytes) return false;
    return a + first_bytes <= b || b + second_bytes <= a;
}

inline hipError_t launch(const Arguments& a, hipStream_t stream) {
    constexpr std::size_t activation_count =
        static_cast<std::size_t>(kTokens) * kColumns;
    constexpr std::size_t weight_count = static_cast<std::size_t>(kRows) * kColumns;
    constexpr std::size_t output_count = static_cast<std::size_t>(kTokens) * kRows;
    constexpr std::size_t token_scale_bytes = kTokens * sizeof(float);
    constexpr std::size_t weight_scale_bytes = kRows * sizeof(float);
    constexpr std::size_t output_bytes = output_count * sizeof(std::uint16_t);
    if (a.tokens != kTokens || a.rows != kRows || a.columns != kColumns ||
        a.activation_codes == nullptr || a.token_scales == nullptr ||
        a.weight_codes == nullptr || a.weight_scales == nullptr ||
        a.output_bf16 == nullptr || a.status == nullptr ||
        (reinterpret_cast<std::uintptr_t>(a.activation_codes) & 15U) != 0U ||
        (reinterpret_cast<std::uintptr_t>(a.weight_codes) & 15U) != 0U ||
        !disjoint(a.output_bf16, output_bytes, a.activation_codes, activation_count) ||
        !disjoint(a.output_bf16, output_bytes, a.weight_codes, weight_count) ||
        !disjoint(a.output_bf16, output_bytes, a.token_scales, token_scale_bytes) ||
        !disjoint(a.output_bf16, output_bytes, a.weight_scales, weight_scale_bytes)) {
        return hipErrorInvalidValue;
    }
    const ninfer::ops::r9700::linear::Fp8GateUpM128N256QualificationArgs retained{
        a.activation_codes, activation_count, a.token_scales, token_scale_bytes,
        a.status, kTokens, kColumns, kColumns, a.weight_codes, weight_count,
        a.weight_scales, weight_scale_bytes, a.output_bf16, output_count};
    return ninfer::ops::r9700::linear::fp8_gate_up_m128n256_qualification(retained, stream);
}

inline hipError_t resources(Resources* output) {
    if (output == nullptr) return hipErrorInvalidValue;
    const auto retained =
        ninfer::ops::r9700::linear::fp8_gate_up_m128n256_qualification_resources();
    *output = {retained.registers, retained.static_lds_bytes, retained.private_bytes,
               retained.maximum_threads, retained.active_blocks_per_wgp};
    return hipSuccess;
}

} // namespace ninfer::qualification::fp8_gate_up_m128n256_retained
