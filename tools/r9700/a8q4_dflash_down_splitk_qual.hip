#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"

#include "ops/r9700/linear/r9700_q4_activation_profile.h"

#include <cctype>

namespace {

// DFlash verify down-GEMM split-K challenger. Exact N5120/K17408 at the DFlash
// verify widths T5/T6. Splits the K reduction across S z-blocks to raise wave
// count for the low-occupancy shape; a reduce kernel combines the S partial-sum
// planes into the BF16 output.
constexpr std::uint32_t kSplitkRows = 5120U;
constexpr std::uint32_t kSplitkColumns = 17408U;
constexpr std::array<std::uint32_t, 2> kSplitkWidths{5U, 6U};
// Bounded split-factor sweep. The production route fixes S at
// linear::kDFlashDownSplitkFactor; this sweep selects the factor.
constexpr std::array<std::uint32_t, 3> kSplitkFactors{2U, 4U, 8U};

std::size_t splitk_partial_sum_bytes(std::uint32_t S, std::uint32_t tokens,
                                     std::uint32_t rows) {
    return static_cast<std::size_t>(S) * tokens * rows * sizeof(float);
}

// Compile-time split-factor dispatch. The launch is a template on S, so the sweep
// must select the factor at compile time; this helper instantiates the bounded set.
template <std::uint32_t S>
hipError_t call_splitk(const linear::A8Q4G64LinearArgs& base_args, float* partial_sums,
                       hipStream_t stream = nullptr) {
    linear::A8Q4G64LinearArgs args = base_args;
    args.partial_sums = partial_sums;
    args.partial_sum_bytes = splitk_partial_sum_bytes(S, base_args.tokens, base_args.rows);
    return linear::a8q4g64_linear_dflash_down_splitk_t<S>(args, stream);
}

hipError_t call_splitk_dispatch(std::uint32_t S, const linear::A8Q4G64LinearArgs& base_args,
                                float* partial_sums, hipStream_t stream = nullptr) {
    switch (S) {
        case 2U: return call_splitk<2U>(base_args, partial_sums, stream);
        case 4U: return call_splitk<4U>(base_args, partial_sums, stream);
        case 8U: return call_splitk<8U>(base_args, partial_sums, stream);
        default: return hipErrorInvalidValue;
    }
}

struct DeviceIdentity {
    hipDeviceProp_t properties{};
    std::string pci_bus_id;
    std::string power_path;
};

std::string normalize_pci(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    if (value.size() == 7U && value[2] == ':' && value[5] == '.') value = "0000:" + value;
    return value;
}

DeviceIdentity select_one_r9700_auto() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    if (count <= 0) fail("HIP device 0 is unavailable");
    HIP_CHECK(hipSetDevice(0));
    DeviceIdentity identity;
    HIP_CHECK(hipGetDeviceProperties(&identity.properties, 0));
    char pci[32]{};
    HIP_CHECK(hipDeviceGetPCIBusId(pci, sizeof(pci), 0));
    identity.pci_bus_id = normalize_pci(pci);
    const std::string sysfs = "/sys/bus/pci/devices/" + identity.pci_bus_id;
    identity.power_path = sysfs + "/power_dpm_force_performance_level";
    if (std::string_view(identity.properties.name) != "AMD Radeon AI PRO R9700" ||
        std::string_view(identity.properties.gcnArchName) != "gfx1201" ||
        identity.properties.integrated != 0 || identity.properties.warpSize != 32 ||
        read_text(sysfs + "/vendor") != "0x1002" ||
        read_text(sysfs + "/device") != "0x7551") {
        fail("device 0 is not the discrete 1002:7551 R9700 gfx1201 wave32 target");
    }
    if (read_text(identity.power_path) != "auto") {
        fail("split-K timing requires device-0 DPM profile auto before allocation");
    }
    return identity;
}

bool run_case(std::uint32_t tokens, const std::filesystem::path& output_path,
              const DeviceIdentity& device) {
    const std::uint32_t rows = kSplitkRows;
    const std::uint32_t columns = kSplitkColumns;
    if (std::find(kSplitkWidths.begin(), kSplitkWidths.end(), tokens) ==
        kSplitkWidths.end()) {
        fail("split-K qualifier received an unqualified width");
    }
    const std::uint32_t padded = (columns + 127U) / 128U * 128U;
    const std::uint32_t groups = padded / kGroup;
    const std::vector<hip_bfloat16> input = make_input(columns, tokens);
    const HostActivation represented = quantize_host(input, tokens, columns);
    const DecodeDot8Weights weights = make_decode_dot8_weights(rows, padded);

    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<std::uint8_t> d_weight_codes(weights.codes.size());
    DeviceBuffer<std::uint16_t> d_weight_scales(weights.scales.size());
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(tokens) * rows + 2U);
    const std::size_t workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(tokens, columns);
    DeviceBuffer<std::uint8_t> d_workspace(workspace_bytes);
    // Largest partial-sum plane across the bounded sweep; reused for every S.
    const std::uint32_t max_S = *std::max_element(kSplitkFactors.begin(),
                                                   kSplitkFactors.end());
    const std::size_t partial_elements =
        splitk_partial_sum_bytes(max_S, tokens, rows) / sizeof(float);
    DeviceBuffer<float> d_partial_sums(partial_elements + 2U);
    constexpr std::uint32_t kPartialGuardBits = 0x7fa12345U;
    const float partial_guard = std::bit_cast<float>(kPartialGuardBits);
    HIP_CHECK(hipMemcpy(d_partial_sums.get(), &partial_guard, sizeof(partial_guard),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_partial_sums.get() + partial_elements + 1U, &partial_guard,
                        sizeof(partial_guard), hipMemcpyHostToDevice));
    float* const partial_payload = d_partial_sums.get() + 1U;
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight_codes.get(), weights.codes.data(), d_weight_codes.bytes(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight_scales.get(), weights.scales.data(), d_weight_scales.bytes(),
                        hipMemcpyHostToDevice));
    linear::A8G64ActivationWorkspace workspace{};
    HIP_CHECK(linear::a8q4g64_bind_activation_workspace(
        d_workspace.get(), workspace_bytes, tokens, columns, &workspace));
    HIP_CHECK(linear::a8g64_quantize_activation({d_input.get(), workspace}, nullptr));
    const linear::A8Q4G64LinearArgs base_args{
        workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
        workspace.high_code_bytes, workspace.scales, workspace.scale_bytes,
        workspace.status, d_weight_codes.get(), d_weight_codes.bytes(),
        d_weight_scales.get(), d_weight_scales.bytes(), d_output.get() + 1U,
        tokens, rows, columns, padded};
    const auto incumbent = [&] { return linear::a8q4g64_linear_wmma32(base_args, nullptr); };
    const auto candidate = [&](std::uint32_t S) {
        return call_splitk_dispatch(S, base_args, partial_payload);
    };

    // Routing/shape guards: the production route must reject non-verify shapes and
    // widths, and the split-K launch must reject a missing or undersized plane.
    {
        linear::A8Q4G64LinearArgs invalid = base_args;
        invalid.partial_sums = partial_payload;
        invalid.partial_sum_bytes = splitk_partial_sum_bytes(max_S, tokens, rows);
        invalid.rows = 4096U;
        if (linear::a8q4g64_linear_dflash_down_splitk(invalid, nullptr) !=
            hipErrorInvalidValue) {
            fail("split-K candidate accepted a non-verify matrix shape");
        }
        invalid = base_args;
        invalid.partial_sums = partial_payload;
        invalid.partial_sum_bytes = splitk_partial_sum_bytes(max_S, tokens, rows);
        invalid.tokens = 7U;
        if (linear::a8q4g64_linear_dflash_down_splitk(invalid, nullptr) !=
            hipErrorInvalidValue) {
            fail("split-K candidate accepted T=7");
        }
        invalid = base_args;
        invalid.partial_sums = nullptr;
        if (linear::a8q4g64_linear_dflash_down_splitk(invalid, nullptr) !=
            hipErrorInvalidValue) {
            fail("split-K candidate accepted a missing partial-sum plane");
        }
        invalid = base_args;
        invalid.partial_sum_bytes = 4U;
        if (linear::a8q4g64_linear_dflash_down_splitk(invalid, nullptr) !=
            hipErrorInvalidValue) {
            fail("split-K candidate accepted an undersized partial-sum plane");
        }
        invalid = base_args;
        invalid.partial_sums = reinterpret_cast<float*>(d_output.get());
        invalid.partial_sum_bytes = splitk_partial_sum_bytes(max_S, tokens, rows);
        if (linear::a8q4g64_linear_dflash_down_splitk(invalid, nullptr) !=
            hipErrorInvalidValue) {
            fail("split-K candidate accepted partial sums aliasing output");
        }
        invalid = base_args;
        invalid.partial_sums = reinterpret_cast<float*>(d_weight_codes.get());
        invalid.partial_sum_bytes = splitk_partial_sum_bytes(max_S, tokens, rows);
        if (linear::a8q4g64_linear_dflash_down_splitk(invalid, nullptr) !=
            hipErrorInvalidValue) {
            fail("split-K candidate accepted partial sums aliasing weights");
        }
    }

    std::vector<hip_bfloat16> guarded(static_cast<std::size_t>(tokens) * rows + 2U);
    const auto initialize = [&](std::uint16_t bits) {
        for (hip_bfloat16& value : guarded) value.data = bits;
        HIP_CHECK(hipMemcpy(d_output.get(), guarded.data(), d_output.bytes(),
                            hipMemcpyHostToDevice));
    };
    const auto download = [&](std::uint16_t canary) {
        HIP_CHECK(hipMemcpy(guarded.data(), d_output.get(), d_output.bytes(),
                            hipMemcpyDeviceToHost));
        if (guarded.front().data != canary || guarded.back().data != canary) {
            fail("split-K candidate changed an output guard");
        }
        return std::vector<hip_bfloat16>(guarded.begin() + 1, guarded.end() - 1);
    };

    // Poisoned-status canary: a nonzero activation status must poison every output.
    {
        const std::uint32_t bad_status = linear::Q4G64ActivationNonfinite;
        HIP_CHECK(hipMemcpy(workspace.status, &bad_status, sizeof(bad_status),
                            hipMemcpyHostToDevice));
        initialize(0x7fc2U);
        HIP_CHECK(candidate(linear::kDFlashDownSplitkFactor));
        HIP_CHECK(hipDeviceSynchronize());
        const auto poisoned = download(0x7fc2U);
        if (!std::all_of(poisoned.begin(), poisoned.end(), [](hip_bfloat16 value) {
                return value.data == 0x7fc1U;
            })) {
            fail("split-K candidate did not poison every output for nonzero status");
        }
        HIP_CHECK(linear::a8g64_quantize_activation({d_input.get(), workspace}, nullptr));
    }

    // Full-rewrite canary plus the independent FP64 represented oracle at the
    // production split factor.
    const std::array<std::uint32_t, 24> sampled_rows{
        0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U,
        16U, 31U, 32U, rows / 3U, rows / 2U, rows - 17U, rows - 16U, rows - 1U};
    const auto oracle_steps = [&](const std::vector<hip_bfloat16>& values,
                                  const char* route) {
        std::uint64_t maximum_steps = 0U;
        for (std::uint32_t token = 0; token < tokens; ++token) {
            for (const std::uint32_t row : sampled_rows) {
                double sum = 0.0;
                for (std::uint32_t group = 0; group < groups; ++group) {
                    std::int32_t integer_dot = 0;
                    for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                        const std::size_t index =
                            static_cast<std::size_t>(token) * padded + group * kGroup + lane;
                        const int activation = unsigned_nibble(represented.low, index) +
                                               16 * signed_nibble(represented.high, index);
                        integer_dot += activation * decode_dot8_weight_code(row, group, lane);
                    }
                    sum = std::fma(
                        static_cast<double>(integer_dot),
                        static_cast<double>(half_value(represented.scales[
                            static_cast<std::size_t>(token) * groups + group])) *
                            half_value(decode_dot8_weight_scale_bits(row, group)), sum);
                }
                const hip_bfloat16 oracle(static_cast<float>(sum));
                const hip_bfloat16 actual =
                    values[static_cast<std::size_t>(token) * rows + row];
                maximum_steps = std::max(maximum_steps, bf16_steps(actual, oracle));
                if (bf16_steps(actual, oracle) > 2U) {
                    fail(std::string(route) +
                         " differs from the independent FP64 represented oracle");
                }
            }
        }
        return maximum_steps;
    };

    initialize(0x7fc2U);
    HIP_CHECK(candidate(linear::kDFlashDownSplitkFactor));
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<hip_bfloat16> candidate_output = download(0x7fc2U);
    if (std::any_of(candidate_output.begin(), candidate_output.end(),
                    [](hip_bfloat16 value) { return value.data == 0x7fc2U; })) {
        fail("split-K candidate did not rewrite every output");
    }
    const std::uint64_t candidate_maximum_steps =
        oracle_steps(candidate_output, "split-K candidate");

    initialize(0x7fc3U);
    HIP_CHECK(incumbent());
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<hip_bfloat16> incumbent_output = download(0x7fc3U);
    if (std::any_of(incumbent_output.begin(), incumbent_output.end(),
                    [](hip_bfloat16 value) { return value.data == 0x7fc3U; })) {
        fail("WMMA incumbent did not rewrite every output");
    }
    const std::uint64_t incumbent_maximum_steps =
        oracle_steps(incumbent_output, "WMMA incumbent");
    const bool pairwise_bit_exact =
        std::equal(candidate_output.begin(), candidate_output.end(),
                   incumbent_output.begin(),
                   [](hip_bfloat16 lhs, hip_bfloat16 rhs) { return lhs.data == rhs.data; });

    // Capture and replay the production factor at stable addresses, then require the
    // complete BF16 result to match eager execution.
    hipStream_t graph_stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&graph_stream, hipStreamNonBlocking));
    initialize(0x7fc5U);
    HIP_CHECK(hipStreamBeginCapture(graph_stream, hipStreamCaptureModeGlobal));
    HIP_CHECK(call_splitk_dispatch(linear::kDFlashDownSplitkFactor, base_args,
                                   partial_payload, graph_stream));
    HIP_CHECK(hipStreamEndCapture(graph_stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
        HIP_CHECK(hipGraphLaunch(graph_exec, graph_stream));
    }
    HIP_CHECK(hipStreamSynchronize(graph_stream));
    const std::vector<hip_bfloat16> graph_output = download(0x7fc5U);
    if (!std::equal(graph_output.begin(), graph_output.end(), candidate_output.begin(),
                    [](hip_bfloat16 lhs, hip_bfloat16 rhs) {
                        return lhs.data == rhs.data;
                    })) {
        fail("split-K graph replay differs from eager output");
    }
    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(graph_stream));

    // Bounded split-factor sweep: paired unprofiled timing of each S against the
    // WMMA incumbent, plus a per-S oracle check.
    struct FactorResult {
        std::uint32_t S;
        double incumbent_median_ms;
        double median_ms;
        double ratio_mean;
        double ratio_upper;
        std::uint64_t max_steps;
        bool accepted;
    };
    std::vector<FactorResult> factor_results;
    for (const std::uint32_t S : kSplitkFactors) {
        initialize(0x7fc4U);
        HIP_CHECK(candidate(S));
        HIP_CHECK(hipDeviceSynchronize());
        const std::vector<hip_bfloat16> sweep_output = download(0x7fc4U);
        const std::uint64_t sweep_steps = oracle_steps(sweep_output, "split-K sweep");

        std::array<double, kTrials> incumbent_forward{};
        std::array<double, kTrials> candidate_forward{};
        std::array<double, kTrials> incumbent_reverse{};
        std::array<double, kTrials> candidate_reverse{};
        std::array<double, kTrials> incumbent_balanced{};
        std::array<double, kTrials> candidate_balanced{};
        std::array<double, kTrials> forward_ratios{};
        std::array<double, kTrials> reverse_ratios{};
        for (std::size_t trial = 0; trial < kTrials; ++trial) {
            incumbent_forward[trial] = event_ms(10, incumbent);
            candidate_forward[trial] = event_ms(10, [&] { return candidate(S); });
            candidate_reverse[trial] = event_ms(10, [&] { return candidate(S); });
            incumbent_reverse[trial] = event_ms(10, incumbent);
            incumbent_balanced[trial] =
                (incumbent_forward[trial] + incumbent_reverse[trial]) * 0.5;
            candidate_balanced[trial] =
                (candidate_forward[trial] + candidate_reverse[trial]) * 0.5;
            forward_ratios[trial] = candidate_forward[trial] / incumbent_forward[trial];
            reverse_ratios[trial] = candidate_reverse[trial] / incumbent_reverse[trial];
        }
        const double incumbent_median = median(incumbent_balanced);
        const double candidate_median = median(candidate_balanced);
        double ratio_sum = 0.0;
        for (const double ratio : forward_ratios) ratio_sum += ratio;
        for (const double ratio : reverse_ratios) ratio_sum += ratio;
        constexpr double ratio_count = 2.0 * static_cast<double>(kTrials);
        const double ratio_mean = ratio_sum / ratio_count;
        double ratio_squared_deviation = 0.0;
        for (const double ratio : forward_ratios) {
            ratio_squared_deviation += (ratio - ratio_mean) * (ratio - ratio_mean);
        }
        for (const double ratio : reverse_ratios) {
            ratio_squared_deviation += (ratio - ratio_mean) * (ratio - ratio_mean);
        }
        const double ratio_standard_deviation =
            std::sqrt(ratio_squared_deviation / (ratio_count - 1.0));
        const double ratio_upper =
            ratio_mean + 2.0 * ratio_standard_deviation / std::sqrt(ratio_count);
        const bool accepted = candidate_median < incumbent_median && ratio_upper < 1.0;
        factor_results.push_back({S, incumbent_median, candidate_median, ratio_mean,
                                  ratio_upper, sweep_steps, accepted});
    }
    if (read_text(device.power_path) != "auto") {
        fail("split-K timing DPM profile changed before publication");
    }
    std::array<float, 2> partial_guards{};
    HIP_CHECK(hipMemcpy(&partial_guards[0], d_partial_sums.get(), sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&partial_guards[1], d_partial_sums.get() + partial_elements + 1U,
                        sizeof(float), hipMemcpyDeviceToHost));
    if (std::bit_cast<std::uint32_t>(partial_guards[0]) != kPartialGuardBits ||
        std::bit_cast<std::uint32_t>(partial_guards[1]) != kPartialGuardBits) {
        fail("split-K candidate changed a partial-sum guard");
    }

    // Select the fastest accepted factor; the production constant must match it.
    const FactorResult* best = nullptr;
    for (const FactorResult& result : factor_results) {
        if (!result.accepted) continue;
        if (best == nullptr || result.median_ms < best->median_ms) best = &result;
    }
    const bool production_matches_best =
        best != nullptr && best->S == linear::kDFlashDownSplitkFactor;
    const bool all_factors_pass_oracle = std::all_of(
        factor_results.begin(), factor_results.end(),
        [](const FactorResult& result) { return result.max_steps <= 2U; });
    const bool accepted =
        all_factors_pass_oracle && best != nullptr && production_matches_best;

    const std::filesystem::path executable = std::filesystem::canonical("/proc/self/exe");
    const std::filesystem::path source_root = std::filesystem::canonical(NINFER_SOURCE_DIR);
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\n  \"schema\": \"ninfer.r9700.a8q4-dflash-down-splitk-cell.v1\",\n"
        << "  \"status\": \"" << (accepted ? "passed" : "rejected") << "\",\n"
        << "  \"production_dispatch_changed\": false,\n"
        << "  \"scope\": \"standalone Q4 down N5120/K17408 at exact DFlash T5/T6; "
           "bounded split-K sweep; no production routing\",\n"
        << "  \"hardware\": {\"device\": \"" << json_escape(device.properties.name)
        << "\", \"architecture\": \"" << json_escape(device.properties.gcnArchName)
        << "\", \"pci_bus_id\": \"" << json_escape(device.pci_bus_id)
        << "\", \"pci_vendor_device\": \"1002:7551\", \"integrated\": false, "
           "\"wavefront_width\": 32, \"power_profile_before_after\": \"auto\"},\n"
        << "  \"shape\": {\"rows\": " << rows << ", \"columns\": " << columns
        << ", \"tokens\": " << tokens << "},\n"
        << "  \"numeric\": {\"oracle\": \"independent FP64 represented A8G64 x Q4G64 formula\", "
           "\"maximum_bf16_steps_allowed\": 2, \"candidate_maximum_bf16_steps\": "
        << candidate_maximum_steps << ", \"incumbent_maximum_bf16_steps\": "
        << incumbent_maximum_steps << ", \"pairwise_bit_exact_diagnostic\": "
        << (pairwise_bit_exact ? "true" : "false")
        << ", \"graph_replays\": 2, \"graph_bit_exact_to_eager\": true, "
           "\"output_canaries\": true, \"partial_sum_canaries\": true, "
           "\"alias_rejection\": true},\n"
        << "  \"sweep\": {\"factors\": [";
    for (std::size_t index = 0; index < factor_results.size(); ++index) {
        const FactorResult& result = factor_results[index];
        out << (index == 0 ? "" : ", ") << "{\"S\": " << result.S
            << ", \"incumbent_median_ms\": " << result.incumbent_median_ms
            << ", \"median_ms\": " << result.median_ms
            << ", \"ratio_mean\": " << result.ratio_mean
            << ", \"ratio_upper\": " << result.ratio_upper
            << ", \"max_bf16_steps\": " << result.max_steps
            << ", \"accepted\": " << (result.accepted ? "true" : "false") << "}";
    }
    out << "], \"best_S\": " << (best != nullptr ? std::to_string(best->S) : "null")
        << ", \"production_S\": " << linear::kDFlashDownSplitkFactor
        << ", \"production_matches_best\": " << (production_matches_best ? "true" : "false")
        << ", \"all_factors_pass_oracle\": " << (all_factors_pass_oracle ? "true" : "false")
        << "},\n"
        << "  \"executable\": {\"path\": \"" << json_escape(executable.string())
        << "\", \"sha256\": \"" << sha256_file(executable) << "\"},\n"
        << "  \"sources\": {\"kernel\": \""
        << sha256_file(source_root / "src/ops/r9700/linear/r9700_linear.hip")
        << "\", \"contract\": \""
        << sha256_file(source_root / "src/ops/r9700/linear/r9700_linear.h")
        << "\", \"qualifier\": \""
        << sha256_file(source_root / "tools/r9700/a8q4_dflash_down_splitk_qual.hip")
        << "\"}\n}\n";
    const std::string report = out.str();
    const int descriptor =
        ::open(output_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) fail("cannot exclusively create split-K report");
    std::size_t offset = 0U;
    while (offset < report.size()) {
        const ssize_t written =
            ::write(descriptor, report.data() + offset, report.size() - offset);
        if (written <= 0) {
            (void)::close(descriptor);
            (void)::unlink(output_path.c_str());
            fail("cannot write split-K report");
        }
        offset += static_cast<std::size_t>(written);
    }
    if (::close(descriptor) != 0) {
        (void)::unlink(output_path.c_str());
        fail("cannot close split-K report");
    }
    std::cout << "A8Q4 DFlash down split-K qualification: "
              << (accepted ? "PASS" : "REJECTED")
              << " rows=" << rows << " columns=" << columns << " tokens=" << tokens
              << " best_S=" << (best != nullptr ? std::to_string(best->S) : "none")
              << " production_S=" << linear::kDFlashDownSplitkFactor << '\n';
    return accepted;
}

std::uint32_t parse_u32(const char* value, const char* name) {
    std::size_t consumed = 0U;
    const unsigned long parsed = std::stoul(value, &consumed);
    if (value[consumed] != '\0' || parsed > std::numeric_limits<std::uint32_t>::max()) {
        fail(std::string("invalid ") + name);
    }
    return static_cast<std::uint32_t>(parsed);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 || std::string_view(argv[2]) != "--out-json") {
            std::cerr << "usage: a8q4_dflash_down_splitk_qual TOKENS --out-json FRESH.json\n";
            return 2;
        }
        const std::filesystem::path output_path = argv[3];
        require_fresh_output(output_path);
        const DeviceIdentity device = select_one_r9700_auto();
        return run_case(parse_u32(argv[1], "tokens"), output_path, device) ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "A8Q4 DFlash down split-K qualification: FAIL: " << error.what()
                  << '\n';
        return 1;
    }
}
