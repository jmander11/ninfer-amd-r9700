#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"

#include "ops/r9700/eager/eager_ops.h"

#include <functional>
#include <map>
#include <regex>
#include <unistd.h>

namespace {

namespace eager = ninfer::ops::r9700::eager;

constexpr std::uint32_t kRows = 5120U;
constexpr std::uint32_t kGroupCountFactor = 64U;
constexpr std::size_t kCopies = 3U;
constexpr std::size_t kResidualTrials = 12U;
constexpr std::size_t kSamplesPerCopy = kResidualTrials / kCopies;
constexpr std::size_t kScrubBytes = 80U * 1024U * 1024U;
constexpr double kRequiredSavingMs = 0.2;
constexpr std::array<bool, kResidualTrials> kIncumbentFirst{
    true, true, true, false, false, false, false, false, false, true, true, true};

struct ResidualShape {
    std::uint32_t columns;
    const char* role;
};

constexpr std::array<ResidualShape, 2> kResidualShapes{{
    {6144U, "attention_and_gdn_output"},
    {17408U, "mlp_down"},
}};

struct CandidateArgs {
    const hip_bfloat16* input = nullptr;
    const std::uint8_t* weight_codes = nullptr;
    std::size_t weight_code_bytes = 0U;
    const std::uint16_t* weight_scales = nullptr;
    std::size_t weight_scale_bytes = 0U;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0U;
    hip_bfloat16* residual = nullptr;
    std::uint32_t tokens = 1U;
    std::uint32_t rows = kRows;
    std::uint32_t columns = 0U;
    std::uint32_t padded_columns = 0U;
};

struct DeviceWeights {
    std::unique_ptr<DeviceBuffer<std::uint8_t>> codes;
    std::unique_ptr<DeviceBuffer<std::uint16_t>> scales;
};

struct CaseResult {
    ResidualShape shape{};
    std::uint64_t maximum_delta_bf16_steps = 0U;
    std::uint64_t maximum_residual_bf16_steps = 0U;
    std::array<double, kResidualTrials> incumbent_samples{};
    std::array<double, kResidualTrials> candidate_samples{};
    std::array<double, kCopies> incumbent_copy_medians{};
    std::array<double, kCopies> candidate_copy_medians{};
    double incumbent_median_ms = 0.0;
    double candidate_median_ms = 0.0;
};

constexpr std::size_t packed_pair_index(std::uint32_t row, std::uint32_t group,
                                        std::uint32_t pair, std::uint32_t groups) {
    return (((static_cast<std::size_t>(row >> 4U) * groups + group) * 4U + pair) *
            16U + (row & 15U));
}

constexpr std::size_t packed_scale_index(std::uint32_t row, std::uint32_t group,
                                         std::uint32_t groups) {
    return ((static_cast<std::size_t>(row >> 4U) * groups + group) * 16U +
            (row & 15U));
}

int oracle_weight_code(const DecodeDot8Weights& weights, std::uint32_t row,
                       std::uint32_t group, std::uint32_t lane,
                       std::uint32_t groups) {
    // Independently decode Q4N16K16: [N/16,G,K/16,N16], with eight packed
    // nibbles in each K/16 element. Do not share the candidate address helpers.
    const std::size_t n_block = row / 16U;
    const std::size_t n_lane = row % 16U;
    const std::size_t k_block = lane / 16U;
    const std::size_t packed_word =
        n_block * static_cast<std::size_t>(groups) * 4U * 16U +
        static_cast<std::size_t>(group) * 4U * 16U + k_block * 16U + n_lane;
    const std::size_t byte = packed_word * sizeof(std::uint64_t) +
                             (lane % 16U) / 2U;
    const unsigned shift = (lane & 1U) * 4U;
    const unsigned code = (weights.codes[byte] >> shift) & 15U;
    return code >= 8U ? static_cast<int>(code) - 16 : static_cast<int>(code);
}

float oracle_weight_scale(const DecodeDot8Weights& weights, std::uint32_t row,
                          std::uint32_t group, std::uint32_t groups) {
    // Canonical scale layout is [N/16,G,N16], independently from the kernel.
    const std::size_t offset =
        (row / 16U) * static_cast<std::size_t>(groups) * 16U +
        static_cast<std::size_t>(group) * 16U + row % 16U;
    return half_value(weights.scales[offset]);
}

void distinguish_weights(DecodeDot8Weights& weights, std::uint32_t columns) {
    const std::uint8_t salt = columns == 6144U ? 0x31U : 0x8cU;
    for (std::size_t index = 0U; index < weights.codes.size(); ++index) {
        weights.codes[index] ^= static_cast<std::uint8_t>(salt + (index & 3U));
    }
    for (std::size_t index = 0U; index < weights.scales.size(); ++index) {
        const float factor = columns == 6144U ? 1.03125F : 0.96875F;
        weights.scales[index] = half_bits(half_value(weights.scales[index]) * factor);
    }
}

std::vector<hip_bfloat16> make_residual(std::uint32_t columns) {
    std::vector<hip_bfloat16> values(kRows);
    for (std::size_t index = 0U; index < values.size(); ++index) {
        const int centered = static_cast<int>((index * 29U + columns / 128U) % 257U) - 128;
        values[index] = hip_bfloat16(static_cast<float>(centered) / 37.0F);
    }
    return values;
}

__device__ __forceinline__ int load_q4x8_residual(const std::uint8_t* codes,
                                                   std::size_t element) {
    return *reinterpret_cast<const int*>(codes + element / 2U);
}

__device__ __forceinline__ int load_weight_x8_residual(
    const std::uint8_t* codes, std::uint32_t row, std::uint32_t group,
    std::uint32_t word, std::uint32_t groups) {
    const std::uint64_t pair = reinterpret_cast<const std::uint64_t*>(codes)[
        packed_pair_index(row, group, word >> 1U, groups)];
    return static_cast<int>((word & 1U) != 0U ? pair >> 32U : pair);
}

__device__ hip_bfloat16 conspicuous_nan() {
    hip_bfloat16 value;
    value.data = 0x7fc1U;
    return value;
}

__global__ __launch_bounds__(256) void a8q4g64_projected_residual_t1_kernel(
    const std::uint8_t* activation_low, const std::uint8_t* activation_high,
    const std::uint16_t* activation_scales, const std::uint32_t* activation_status,
    const std::uint8_t* weight_codes, const std::uint16_t* weight_scales,
    hip_bfloat16* residual, std::uint32_t columns) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= kRows) return;
    if (*activation_status != 0U) {
        residual[row] = conspicuous_nan();
        return;
    }
    const std::uint32_t groups = columns / kGroupCountFactor;
    float total = 0.0F;
    for (std::uint32_t group = 0U; group < groups; ++group) {
        int low_dot = 0;
        int high_dot = 0;
#pragma unroll
        for (std::uint32_t word = 0U; word < 8U; ++word) {
            const std::size_t element =
                static_cast<std::size_t>(group) * kGroupCountFactor + word * 8U;
            const int low = load_q4x8_residual(activation_low, element);
            const int high = load_q4x8_residual(activation_high, element);
            const int weight =
                load_weight_x8_residual(weight_codes, row, group, word, groups);
            low_dot = __builtin_amdgcn_sudot8(false, low, true, weight, low_dot, false);
            high_dot = __builtin_amdgcn_sudot8(true, high, true, weight, high_dot, false);
        }
        const int combined = low_dot + 16 * high_dot;
        const float activation_scale =
            __half2float(__ushort_as_half(activation_scales[group]));
        const float weight_scale = __half2float(__ushort_as_half(
            weight_scales[packed_scale_index(row, group, groups)]));
        total = fmaf(static_cast<float>(combined),
                     activation_scale * weight_scale, total);
    }
    const hip_bfloat16 delta = static_cast<hip_bfloat16>(total);
    residual[row] = hip_bfloat16(static_cast<float>(residual[row]) +
                                 static_cast<float>(delta));
#else
    (void)activation_low; (void)activation_high; (void)activation_scales;
    (void)activation_status; (void)weight_codes; (void)weight_scales;
    (void)residual; (void)columns;
#endif
}

__global__ void projected_residual_cache_scrub_kernel(std::uint32_t* values,
                                                       std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = values[index] * 1664525U + 1013904223U;
}

bool overlap(const void* lhs, std::size_t lhs_bytes,
             const void* rhs, std::size_t rhs_bytes) {
    if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0U || rhs_bytes == 0U) return false;
    const auto l = reinterpret_cast<std::uintptr_t>(lhs);
    const auto r = reinterpret_cast<std::uintptr_t>(rhs);
    return l < r + rhs_bytes && r < l + lhs_bytes;
}

bool valid_candidate(const CandidateArgs& args) {
    const bool shape = args.tokens == 1U && args.rows == kRows &&
        (args.columns == 6144U || args.columns == 17408U) &&
        args.padded_columns == args.columns;
    const std::uint32_t groups = args.columns / kGroupCountFactor;
    const std::size_t input_bytes = static_cast<std::size_t>(args.columns) * 2U;
    const std::size_t code_bytes = static_cast<std::size_t>(kRows) * args.columns / 2U;
    const std::size_t scale_bytes = static_cast<std::size_t>(kRows) * groups * 2U;
    const std::size_t residual_bytes = static_cast<std::size_t>(kRows) * 2U;
    const std::size_t workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(1U, args.columns);
    if (!shape || args.input == nullptr || args.weight_codes == nullptr ||
        args.weight_scales == nullptr || args.workspace == nullptr || args.residual == nullptr ||
        args.weight_code_bytes != code_bytes || args.weight_scale_bytes != scale_bytes ||
        args.workspace_bytes != workspace_bytes ||
        (reinterpret_cast<std::uintptr_t>(args.weight_codes) & 7U) != 0U) return false;
    const std::array<std::pair<const void*, std::size_t>, 4> regions{{
        {args.input, input_bytes}, {args.weight_codes, code_bytes},
        {args.weight_scales, scale_bytes}, {args.residual, residual_bytes}}};
    for (std::size_t i = 0U; i < regions.size(); ++i) {
        if (overlap(args.workspace, workspace_bytes, regions[i].first, regions[i].second))
            return false;
        for (std::size_t j = i + 1U; j < regions.size(); ++j) {
            if (overlap(regions[i].first, regions[i].second,
                        regions[j].first, regions[j].second)) return false;
        }
    }
    return true;
}

hipError_t launch_candidate(const CandidateArgs& args, hipStream_t stream = nullptr) {
    if (!valid_candidate(args)) return hipErrorInvalidValue;
    linear::A8G64ActivationWorkspace workspace{};
    if (linear::a8q4g64_bind_activation_workspace(
            args.workspace, args.workspace_bytes, 1U, args.columns,
            &workspace) != hipSuccess) return hipErrorInvalidValue;
    hipError_t status = linear::a8g64_quantize_activation({args.input, workspace}, stream);
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(a8q4g64_projected_residual_t1_kernel,
                       dim3((kRows + 255U) / 256U), dim3(256U), 0, stream,
                       workspace.low_codes, workspace.high_codes, workspace.scales,
                       workspace.status, args.weight_codes, args.weight_scales,
                       args.residual, args.columns);
    return hipGetLastError();
}

linear::A8Q4G64CandidateArgs incumbent_args(
    const CandidateArgs& args, hip_bfloat16* delta) {
    return {args.input, args.weight_codes, args.weight_code_bytes,
            args.weight_scales, args.weight_scale_bytes, args.workspace,
            args.workspace_bytes, delta, 1U, kRows, args.columns,
            args.padded_columns};
}

hipError_t launch_incumbent(const CandidateArgs& args, hip_bfloat16* delta,
                            hipStream_t stream = nullptr) {
    if (!valid_candidate(args) || overlap(delta, kRows * 2U, args.residual, kRows * 2U))
        return hipErrorInvalidValue;
    hipError_t status = linear::a8q4g64_linear_candidate(
        incumbent_args(args, delta), stream);
    if (status != hipSuccess) return status;
    return eager::residual_add_bf16(delta, args.residual, kRows, stream);
}

void scrub(DeviceBuffer<std::uint32_t>& buffer) {
    constexpr std::uint32_t threads = 256U;
    const std::size_t count = buffer.bytes() / sizeof(std::uint32_t);
    hipLaunchKernelGGL(projected_residual_cache_scrub_kernel,
                       dim3(static_cast<unsigned>((count + threads - 1U) / threads)),
                       dim3(threads), 0, nullptr, buffer.get(), count);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

double timed(const std::function<hipError_t()>& launch) {
    Event begin;
    Event end;
    HIP_CHECK(hipEventRecord(begin.get()));
    HIP_CHECK(launch());
    HIP_CHECK(hipEventRecord(end.get()));
    HIP_CHECK(hipEventSynchronize(end.get()));
    float elapsed = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&elapsed, begin.get(), end.get()));
    return elapsed;
}

template <std::size_t N> double even_median(std::array<double, N> values) {
    std::sort(values.begin(), values.end());
    return 0.5 * (values[N / 2U - 1U] + values[N / 2U]);
}

void fill_guarded(DeviceBuffer<hip_bfloat16>& buffer,
                  const std::vector<hip_bfloat16>& payload_values,
                  std::uint16_t guard) {
    std::vector<hip_bfloat16> host(payload_values.size() + 2U);
    host.front().data = guard;
    host.back().data = guard;
    std::copy(payload_values.begin(), payload_values.end(), host.begin() + 1);
    HIP_CHECK(hipMemcpy(buffer.get(), host.data(), buffer.bytes(), hipMemcpyHostToDevice));
}

std::vector<hip_bfloat16> download_guarded(const DeviceBuffer<hip_bfloat16>& buffer,
                                           std::uint16_t guard, const char* label) {
    std::vector<hip_bfloat16> host(kRows + 2U);
    HIP_CHECK(hipMemcpy(host.data(), buffer.get(), buffer.bytes(), hipMemcpyDeviceToHost));
    if (host.front().data != guard || host.back().data != guard)
        fail(std::string(label) + " changed a canary");
    return {host.begin() + 1, host.end() - 1};
}

CaseResult run_case(const ResidualShape& shape) {
    const std::uint32_t groups = shape.columns / kGroupCountFactor;
    const auto input = make_decode_dot8_input(shape.columns);
    const auto activation = quantize_host(input, 1U, shape.columns);
    auto weights = make_decode_dot8_weights(kRows, shape.columns);
    distinguish_weights(weights, shape.columns);
    const auto initial_residual = make_residual(shape.columns);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));
    std::array<DeviceWeights, kCopies> copies;
    for (auto& copy : copies) {
        copy.codes = std::make_unique<DeviceBuffer<std::uint8_t>>(weights.codes.size());
        copy.scales = std::make_unique<DeviceBuffer<std::uint16_t>>(weights.scales.size());
        HIP_CHECK(hipMemcpy(copy.codes->get(), weights.codes.data(), copy.codes->bytes(),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(copy.scales->get(), weights.scales.data(), copy.scales->bytes(),
                            hipMemcpyHostToDevice));
    }
    const std::size_t workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(1U, shape.columns);
    DeviceBuffer<std::uint8_t> workspace(workspace_bytes);
    DeviceBuffer<hip_bfloat16> delta(kRows + 2U);
    DeviceBuffer<hip_bfloat16> incumbent(kRows + 2U);
    DeviceBuffer<hip_bfloat16> candidate(kRows + 2U);
    DeviceBuffer<std::uint32_t> scrub_buffer(kScrubBytes / sizeof(std::uint32_t));
    HIP_CHECK(hipMemset(scrub_buffer.get(), 0x5a, scrub_buffer.bytes()));
    auto args_for = [&](std::size_t copy, hip_bfloat16* residual) {
        return CandidateArgs{d_input.get(), copies[copy].codes->get(), copies[copy].codes->bytes(),
            copies[copy].scales->get(), copies[copy].scales->bytes(), workspace.get(),
            workspace_bytes, residual, 1U, kRows, shape.columns, shape.columns};
    };
    auto valid = args_for(0U, candidate.get() + 1U);
    auto malformed = valid;
    malformed.tokens = 2U;
    if (launch_candidate(malformed) != hipErrorInvalidValue) fail("projected residual accepted T2");
    malformed = valid; malformed.rows = 4096U;
    if (launch_candidate(malformed) != hipErrorInvalidValue) fail("projected residual accepted N4096");
    malformed = valid; malformed.columns = 5120U; malformed.padded_columns = 5120U;
    if (launch_candidate(malformed) != hipErrorInvalidValue) fail("projected residual accepted K5120");
    malformed = valid; --malformed.weight_code_bytes;
    if (launch_candidate(malformed) != hipErrorInvalidValue) fail("projected residual accepted short codes");
    malformed = valid; ++malformed.weight_codes;
    if (launch_candidate(malformed) != hipErrorInvalidValue) fail("projected residual accepted misalignment");
    malformed = valid; malformed.workspace = malformed.residual;
    if (launch_candidate(malformed) != hipErrorInvalidValue) fail("projected residual accepted workspace alias");
    malformed = valid; malformed.input = reinterpret_cast<const hip_bfloat16*>(malformed.weight_codes);
    if (launch_candidate(malformed) != hipErrorInvalidValue) fail("projected residual accepted input alias");
    malformed = valid; malformed.residual = const_cast<hip_bfloat16*>(malformed.input);
    if (launch_candidate(malformed) != hipErrorInvalidValue) fail("projected residual accepted residual alias");

    fill_guarded(incumbent, initial_residual, 0x7fc2U);
    fill_guarded(candidate, initial_residual, 0x7fc3U);
    std::vector<hip_bfloat16> delta_seed(kRows);
    for (auto& value : delta_seed) value.data = 0x7fc4U;
    fill_guarded(delta, delta_seed, 0x7fc4U);
    HIP_CHECK(launch_incumbent(args_for(0U, incumbent.get() + 1U), delta.get() + 1U));
    HIP_CHECK(launch_candidate(args_for(0U, candidate.get() + 1U)));
    HIP_CHECK(hipDeviceSynchronize());
    const auto incumbent_host = download_guarded(incumbent, 0x7fc2U, "incumbent residual");
    const auto candidate_host = download_guarded(candidate, 0x7fc3U, "candidate residual");
    const auto delta_host = download_guarded(delta, 0x7fc4U, "incumbent delta");
    if (!std::equal(incumbent_host.begin(), incumbent_host.end(), candidate_host.begin(),
                    [](hip_bfloat16 lhs, hip_bfloat16 rhs) { return lhs.data == rhs.data; }))
        fail("projected residual is not bit-exact to Linear plus residual-add");

    std::uint64_t maximum_delta_steps = 0U;
    std::uint64_t maximum_residual_steps = 0U;
    for (std::uint32_t row = 0U; row < kRows; ++row) {
        double sum = 0.0;
        for (std::uint32_t group = 0U; group < groups; ++group) {
            std::int32_t dot = 0;
            for (std::uint32_t lane = 0U; lane < kGroupCountFactor; ++lane) {
                const std::size_t index = static_cast<std::size_t>(group) * kGroupCountFactor + lane;
                const int a = unsigned_nibble(activation.low, index) +
                              16 * signed_nibble(activation.high, index);
                dot += a * oracle_weight_code(weights, row, group, lane, groups);
            }
            sum = std::fma(static_cast<double>(dot),
                static_cast<double>(half_value(activation.scales[group])) *
                    oracle_weight_scale(weights, row, group, groups), sum);
        }
        const hip_bfloat16 oracle_delta(static_cast<float>(sum));
        const hip_bfloat16 oracle_residual(
            static_cast<float>(initial_residual[row]) + static_cast<float>(oracle_delta));
        maximum_delta_steps = std::max(maximum_delta_steps,
                                        bf16_steps(delta_host[row], oracle_delta));
        maximum_residual_steps = std::max(maximum_residual_steps,
                                           bf16_steps(candidate_host[row], oracle_residual));
        if (bf16_steps(delta_host[row], oracle_delta) > 2U ||
            bf16_steps(candidate_host[row], oracle_residual) > 2U)
            fail("projected residual differs from independent represented-weight FP64 oracle");
    }

    hip_bfloat16 nan{};
    nan.data = 0x7fc1U;
    HIP_CHECK(hipMemcpy(d_input.get(), &nan, sizeof(nan), hipMemcpyHostToDevice));
    fill_guarded(candidate, initial_residual, 0x7fc5U);
    HIP_CHECK(launch_candidate(args_for(0U, candidate.get() + 1U)));
    HIP_CHECK(hipDeviceSynchronize());
    const auto poisoned = download_guarded(candidate, 0x7fc5U, "status-poison residual");
    if (!std::all_of(poisoned.begin(), poisoned.end(), [](hip_bfloat16 value) {
            return value.data == 0x7fc1U;
        })) fail("projected residual did not poison every output");
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));

    CaseResult result{shape, maximum_delta_steps, maximum_residual_steps};
    std::array<std::array<double, kSamplesPerCopy>, kCopies> incumbent_by_copy{};
    std::array<std::array<double, kSamplesPerCopy>, kCopies> candidate_by_copy{};
    std::array<std::size_t, kCopies> incumbent_count{}, candidate_count{};
    std::array<std::size_t, kCopies> incumbent_first_count{}, candidate_first_count{};
    for (std::size_t warm = 0U; warm < 6U; ++warm) {
        fill_guarded(incumbent, initial_residual, 0x7fc2U);
        HIP_CHECK(launch_incumbent(args_for(warm % kCopies, incumbent.get() + 1U),
                                   delta.get() + 1U));
        fill_guarded(candidate, initial_residual, 0x7fc3U);
        HIP_CHECK(launch_candidate(args_for(warm % kCopies, candidate.get() + 1U)));
    }
    HIP_CHECK(hipDeviceSynchronize());
    for (std::size_t trial = 0U; trial < kResidualTrials; ++trial) {
        const std::size_t incumbent_copy = trial % kCopies;
        const std::size_t candidate_copy = incumbent_copy;
        if (kIncumbentFirst[trial]) ++incumbent_first_count[incumbent_copy];
        else ++candidate_first_count[candidate_copy];
        const auto run_incumbent = [&] {
            scrub(scrub_buffer);
            fill_guarded(incumbent, initial_residual, 0x7fc2U);
            const double value = timed([&] { return launch_incumbent(
                args_for(incumbent_copy, incumbent.get() + 1U), delta.get() + 1U); });
            result.incumbent_samples[trial] = value;
            incumbent_by_copy[incumbent_copy][incumbent_count[incumbent_copy]++] = value;
        };
        const auto run_candidate = [&] {
            scrub(scrub_buffer);
            fill_guarded(candidate, initial_residual, 0x7fc3U);
            const double value = timed([&] { return launch_candidate(
                args_for(candidate_copy, candidate.get() + 1U)); });
            result.candidate_samples[trial] = value;
            candidate_by_copy[candidate_copy][candidate_count[candidate_copy]++] = value;
        };
        if (kIncumbentFirst[trial]) { run_incumbent(); run_candidate(); }
        else { run_candidate(); run_incumbent(); }
    }
    result.incumbent_median_ms = even_median(result.incumbent_samples);
    result.candidate_median_ms = even_median(result.candidate_samples);
    for (std::size_t copy = 0U; copy < kCopies; ++copy) {
        if (incumbent_count[copy] != kSamplesPerCopy ||
            candidate_count[copy] != kSamplesPerCopy ||
            incumbent_first_count[copy] != kSamplesPerCopy / 2U ||
            candidate_first_count[copy] != kSamplesPerCopy / 2U)
            fail("projected residual timing is not jointly allocation/order balanced");
        result.incumbent_copy_medians[copy] = even_median(incumbent_by_copy[copy]);
        result.candidate_copy_medians[copy] = even_median(candidate_by_copy[copy]);
        if (result.candidate_copy_medians[copy] >
            result.incumbent_copy_medians[copy] * 1.01)
            fail("projected residual regressed on a cold allocation");
    }
    return result;
}

void write_array(std::ostream& out, const auto& values) {
    out << '[';
    for (std::size_t index = 0U; index < values.size(); ++index)
        out << (index == 0U ? "" : ",") << values[index];
    out << ']';
}

std::string read_all_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot read retained receipt " + path.string());
    const std::string value((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    if (input.bad() || value.empty()) fail("retained receipt is empty " + path.string());
    return value;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 9 || std::string_view(argv[1]) != "--out-json" ||
            std::string_view(argv[3]) != "--assembly" ||
            std::string_view(argv[5]) != "--static-receipt" ||
            std::string_view(argv[7]) != "--compile-receipt") {
            std::cerr << "usage: a8q4_projected_residual_t1_qual --out-json NEW.json "
                         "--assembly FILE.s --static-receipt RECEIPT.txt "
                         "--compile-receipt RECEIPT.json\n";
            return 2;
        }
        const std::filesystem::path output = argv[2];
        const std::filesystem::path assembly = std::filesystem::canonical(argv[4]);
        const std::filesystem::path static_receipt = std::filesystem::canonical(argv[6]);
        const std::filesystem::path compile_receipt = std::filesystem::canonical(argv[8]);
        const std::string compile_receipt_text = read_all_text(compile_receipt);
        require_fresh_output(output);
        const std::filesystem::path root = std::filesystem::canonical(NINFER_SOURCE_DIR);
        const std::filesystem::path checker = std::filesystem::canonical(
            root / "tools/r9700/check_a8q4_projected_residual_t1_static.py");
        const std::string static_command = "python3 " + shell_quote(checker.string()) + " " +
            shell_quote(assembly.string()) + " --expect-receipt " +
            shell_quote(static_receipt.string());
        if (std::system(static_command.c_str()) != 0) fail("projected residual static gate failed");
        const std::string static_text = read_text(static_receipt);
        std::smatch static_match;
        const std::regex static_expression(
            "a8q4_projected_residual_t1_static: PASS .*vgpr=([0-9]+) sgpr=([0-9]+) "
            "wave32=true lds=0 scratch=0 spills=0");
        if (!std::regex_match(static_text, static_match, static_expression))
            fail("projected residual static receipt differs");
        HIP_CHECK(hipSetDevice(0));
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, 0));
        if (std::string_view(properties.name) != "AMD Radeon AI PRO R9700" ||
            std::string_view(properties.gcnArchName) != "gfx1201" || properties.warpSize != 32)
            fail("projected residual qualification requires R9700 gfx1201 wave32");
        char pci_bus_id[32]{};
        HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
        std::string pci = pci_bus_id;
        if (pci.size() == 7U) pci = "0000:" + pci;
        std::transform(pci.begin(), pci.end(), pci.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
        const std::filesystem::path power = std::filesystem::path("/sys/bus/pci/devices") /
            pci / "power_dpm_force_performance_level";
        const std::string power_before = read_text(power);
        if (power_before != "auto") fail("projected residual qualification requires auto power");
        int runtime_version = 0;
        int driver_version = 0;
        HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
        HIP_CHECK(hipDriverGetVersion(&driver_version));

        std::array<CaseResult, kResidualShapes.size()> results;
        for (std::size_t index = 0U; index < kResidualShapes.size(); ++index)
            results[index] = run_case(kResidualShapes[index]);
        const std::string power_after = read_text(power);
        if (power_after != "auto") fail("power changed during projected residual qualification");
        double saved_ms = 0.0;
        for (const auto& result : results)
            saved_ms += 64.0 * (result.incumbent_median_ms - result.candidate_median_ms);
        if (saved_ms < kRequiredSavingMs)
            fail("projected residual misses the 0.2 ms/token aggregate admission bound");

        const std::filesystem::path executable = std::filesystem::canonical("/proc/self/exe");
        const std::array<std::pair<const char*, const char*>, 12> source_specs{{
            {"qualifier", "tools/r9700/a8q4_projected_residual_t1_qual.hip"},
            {"shared_qualifier", "tools/r9700/a8q4_shape_sweep_qual.hip"},
            {"static_checker", "tools/r9700/check_a8q4_projected_residual_t1_static.py"},
            {"linear_contract", "src/ops/r9700/linear/r9700_linear.h"},
            {"linear_implementation", "src/ops/r9700/linear/r9700_linear.hip"},
            {"residual_contract", "src/ops/r9700/eager/eager_ops.h"},
            {"residual_implementation", "src/ops/r9700/eager/eager_ops.hip"},
            {"plan", "profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919/plan.json"},
            {"commands", "profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919/commands.sh"},
            {"preflight", "profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919/preflight.py"},
            {"measure", "profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919/measure.py"},
            {"layer_trace", "profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/analysis-v2.json"},
        }};
        std::ostringstream report;
        report << std::setprecision(17)
               << "{\n  \"schema\":\"ninfer.r9700.a8q4-projected-residual-t1-qualification.v1\",\n"
               << "  \"status\":\"qualified_for_whole_ab_only\",\n"
               << "  \"production_routing_authorized\":false,\n"
               << "  \"invocation\":{\"argv\":[\"" << json_escape(executable.string())
               << "\",\"--out-json\",\"" << json_escape(output.string())
               << "\",\"--assembly\",\"" << json_escape(assembly.string())
               << "\",\"--static-receipt\",\"" << json_escape(static_receipt.string())
               << "\",\"--compile-receipt\",\"" << json_escape(compile_receipt.string())
               << "\"]},\n"
               << "  \"process_receipt\":{\"pid\":" << static_cast<long long>(::getpid())
               << ",\"executable_sha256\":\"" << sha256_file(executable)
               << "\",\"exact_argv_recorded\":true},\n"
               << "  \"compile_receipt\":{\"path\":\""
               << json_escape(compile_receipt.string()) << "\",\"bytes\":"
               << std::filesystem::file_size(compile_receipt) << ",\"sha256\":\""
               << sha256_file(compile_receipt) << "\",\"text\":\""
               << json_escape(compile_receipt_text) << "\"},\n"
               << "  \"hardware\":{\"device\":0,\"name\":\""
               << json_escape(properties.name) << "\",\"architecture\":\""
               << json_escape(properties.gcnArchName) << "\",\"pci\":\""
               << json_escape(pci) << "\",\"wave_size\":" << properties.warpSize
               << ",\"hip_runtime_version\":" << runtime_version
               << ",\"hip_driver_version\":" << driver_version << "},\n"
               << "  \"power_profile\":{\"path\":\"" << json_escape(power.string())
               << "\",\"required\":\"auto\",\"before\":\""
               << json_escape(power_before) << "\",\"after\":\""
               << json_escape(power_after) << "\"},\n"
               << "  \"executable\":{\"path\":\"" << json_escape(executable.string())
               << "\",\"bytes\":" << std::filesystem::file_size(executable)
               << ",\"sha256\":\"" << sha256_file(executable) << "\"},\n"
               << "  \"sources\":[\n";
        for (std::size_t index = 0U; index < source_specs.size(); ++index) {
            const auto path = std::filesystem::canonical(root / source_specs[index].second);
            report << "    {\"role\":\"" << source_specs[index].first << "\",\"path\":\""
                   << json_escape(path.string()) << "\",\"bytes\":"
                   << std::filesystem::file_size(path) << ",\"sha256\":\""
                   << sha256_file(path) << "\"}"
                   << (index + 1U == source_specs.size() ? "\n" : ",\n");
        }
        report << "  ],\n  \"static\":{\"status\":\"passed\",\"native_iu4_dot8\":true,"
                  "\"wave_size\":32,\"vgpr\":" << static_match[1].str()
               << ",\"sgpr\":" << static_match[2].str()
               << ",\"lds_bytes\":0,\"scratch_bytes\":0,\"spill_count\":0,"
                  "\"assembly_sha256\":\"" << sha256_file(assembly)
               << "\",\"receipt_sha256\":\"" << sha256_file(static_receipt)
               << "\",\"checker_sha256\":\"" << sha256_file(checker) << "\"},\n"
               << "  \"bound_evidence\":{\"trace_path\":\""
               << json_escape((root / "profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/analysis-v2.json").string())
               << "\",\"trace_sha256\":\""
               << sha256_file(root / "profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/analysis-v2.json")
               << "\",\"attention_and_gdn_output_calls\":64,\"mlp_down_calls\":64},\n"
               << "  \"correctness\":{\"exact_complete_residual_bits\":true,"
                  "\"independent_fp64_represented_weight_oracle\":true,"
                  "\"status_poisoning\":true,\"canaries\":true,"
                  "\"malformed_alignment_alias_rejection\":true},\n"
               << "  \"timing\":{\"calls_per_shape_per_token\":64,"
                  "\"trials_per_arm\":12,\"disjoint_weight_copies\":3,"
                  "\"cache_scrub_bytes_before_each_arm\":" << kScrubBytes
               << ",\"arm_order_balanced\":true,\"allocation_balanced\":true,"
                  "\"joint_copy_order_balanced\":true,\"cases\":[\n";
        for (std::size_t index = 0U; index < results.size(); ++index) {
            const auto& result = results[index];
            report << "    {\"role\":\"" << result.shape.role << "\",\"rows\":" << kRows
                   << ",\"columns\":" << result.shape.columns
                   << ",\"maximum_delta_bf16_steps\":" << result.maximum_delta_bf16_steps
                   << ",\"maximum_residual_bf16_steps\":" << result.maximum_residual_bf16_steps
                   << ",\"incumbent_samples_ms\":";
            write_array(report, result.incumbent_samples);
            report << ",\"candidate_samples_ms\":"; write_array(report, result.candidate_samples);
            report << ",\"incumbent_copy_medians_ms\":";
            write_array(report, result.incumbent_copy_medians);
            report << ",\"candidate_copy_medians_ms\":";
            write_array(report, result.candidate_copy_medians);
            report << ",\"incumbent_median_ms\":" << result.incumbent_median_ms
                   << ",\"candidate_median_ms\":" << result.candidate_median_ms << "}"
                   << (index + 1U == results.size() ? "\n" : ",\n");
        }
        report << "  ]},\n  \"aggregate_bound\":{\"formula\":"
                  "\"64*sum(incumbent_median_ms-candidate_median_ms)\","
                  "\"saved_ms_per_token\":" << saved_ms
               << ",\"minimum_ms_per_token\":" << kRequiredSavingMs
               << ",\"passed\":true},\n"
               << "  \"decision\":\"prepare_selector_off_on_whole_c1_ab\"\n}\n";
        const std::string payload = report.str();
        const int descriptor = ::open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                                      0644);
        if (descriptor < 0) fail("cannot create projected residual report exclusively");
        FILE* stream = fdopen(descriptor, "wb");
        if (stream == nullptr) {
            (void)::close(descriptor);
            (void)::unlink(output.c_str());
            fail("cannot bind projected residual report stream");
        }
        const bool wrote = fwrite(payload.data(), 1U, payload.size(), stream) == payload.size();
        const bool closed = fclose(stream) == 0;
        if (!wrote || !closed) {
            (void)::unlink(output.c_str());
            fail("cannot finish projected residual report");
        }
        std::cout << "a8q4_projected_residual_t1: PASS saved_ms_per_token="
                  << saved_ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "A8Q4 projected residual T1 qualification: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
