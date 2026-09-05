#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"

namespace {

struct HostA4Activation {
    std::vector<std::uint8_t> codes;
    std::vector<std::uint16_t> scales;
};

HostA4Activation quantize_a4_host(const std::vector<hip_bfloat16>& input,
                                  std::uint32_t tokens, std::uint32_t columns) {
    const std::uint32_t padded = (columns + 127U) / 128U * 128U;
    const std::uint32_t groups = padded / kGroup;
    HostA4Activation result{
        std::vector<std::uint8_t>(static_cast<std::size_t>(tokens) * padded / 2U),
        std::vector<std::uint16_t>(static_cast<std::size_t>(tokens) * groups)};
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t group = 0; group < groups; ++group) {
            float maximum = 0.0F;
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const std::uint32_t column = group * kGroup + lane;
                const float value = column < columns
                    ? static_cast<float>(input[static_cast<std::size_t>(token) * columns + column])
                    : 0.0F;
                maximum = std::max(maximum, std::abs(value));
            }
            std::uint16_t bits = maximum == 0.0F ? 0U : half_bits(maximum / 7.0F);
            if (maximum != 0.0F && bits == 0U) bits = 1U;
            const float scale = half_value(bits);
            result.scales[static_cast<std::size_t>(token) * groups + group] = bits;
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                const std::uint32_t column = group * kGroup + lane;
                const float value = column < columns
                    ? static_cast<float>(input[static_cast<std::size_t>(token) * columns + column])
                    : 0.0F;
                const int code = bits == 0U ? 0 : std::clamp(
                    static_cast<int>(std::nearbyint(value / scale)), -8, 7);
                set_nibble(result.codes,
                           static_cast<std::size_t>(token) * padded + group * kGroup + lane,
                           code);
            }
        }
    }
    return result;
}

struct A4Timing {
    Shape shape{};
    std::uint32_t tokens = 0;
    std::array<double, kTrials> incumbent{};
    std::array<double, kTrials> challenger{};
    double incumbent_median = 0.0;
    double challenger_median = 0.0;
    std::uint64_t maximum_bf16_steps = 0;
};

A4Timing run_a4_case(Shape shape, std::uint32_t tokens, bool timing) {
    const std::uint32_t padded = (shape.columns + 127U) / 128U * 128U;
    const std::uint32_t groups = padded / kGroup;
    const std::vector<hip_bfloat16> input = make_input(shape.columns, tokens);
    const HostA4Activation expected = quantize_a4_host(input, tokens, shape.columns);
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(shape.rows) * padded / 2U);
    std::vector<std::uint16_t> weight_scales(static_cast<std::size_t>(shape.rows) * groups);
    for (std::uint32_t row = 0; row < shape.rows; ++row) {
        for (std::uint32_t group = 0; group < groups; ++group) {
            weight_scales[static_cast<std::size_t>(row) * groups + group] =
                prefill_weight_scale_bits(row & 3U, group);
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                set_nibble(weights, static_cast<std::size_t>(row) * padded +
                           group * kGroup + lane, prefill_weight_code(row & 15U, group * kGroup + lane));
            }
        }
    }
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<std::uint8_t> d_weights(weights.size());
    DeviceBuffer<std::uint16_t> d_weight_scales(weight_scales.size());
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(tokens) * shape.rows);
    const std::size_t a4_bytes = linear::q4g64_activation_workspace_capacity_bytes(tokens, shape.columns);
    const std::size_t a8_bytes = linear::a8q4g64_activation_workspace_capacity_bytes(tokens, shape.columns);
    DeviceBuffer<std::uint8_t> d_a4(a4_bytes), d_a8(a8_bytes);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights.get(), weights.data(), weights.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight_scales.get(), weight_scales.data(),
                        weight_scales.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
    linear::Q4G64ActivationWorkspace a4{};
    linear::A8G64ActivationWorkspace a8{};
    HIP_CHECK(linear::q4g64_bind_activation_workspace(d_a4.get(), a4_bytes, tokens,
                                                       shape.columns, &a4));
    HIP_CHECK(linear::a8q4g64_bind_activation_workspace(d_a8.get(), a8_bytes, tokens,
                                                         shape.columns, &a8));
    const linear::Q4G64LinearArgs a4_args{
        a4.codes, a4.code_bytes, a4.scales, a4.scale_bytes, a4.status,
        d_weights.get(), d_weights.bytes(), d_weight_scales.get(), d_weight_scales.bytes(),
        d_output.get(), tokens, shape.rows, shape.columns, padded};
    const linear::A8Q4G64CandidateArgs incumbent{
        d_input.get(), d_weights.get(), d_weights.bytes(), d_weight_scales.get(),
        d_weight_scales.bytes(), d_a8.get(), a8_bytes, d_output.get(), tokens,
        shape.rows, shape.columns, padded};
    const auto launch_a4 = [&] {
        hipError_t status = linear::q4g64_quantize_activation({d_input.get(), a4}, nullptr);
        return status == hipSuccess
            ? linear::q4g64_linear_prefill_cta_m64n128_qualification(a4_args, nullptr) : status;
    };
    const auto launch_incumbent = [&] { return linear::a8q4g64_linear_candidate(incumbent, nullptr); };

    if (shape.rows == 129U && shape.columns == 193U && tokens == 193U) {
        linear::Q4G64LinearArgs malformed = a4_args;
        ++malformed.activation_codes;
        if (linear::q4g64_linear_prefill_cta_m64n128_qualification(malformed, nullptr) !=
            hipErrorInvalidValue) {
            fail("A4 M64N128 accepted a non-dword-aligned activation plane");
        }
        const std::uint32_t bad_status = linear::Q4G64ActivationNonfinite;
        HIP_CHECK(hipMemcpy(a4.status, &bad_status, sizeof(bad_status), hipMemcpyHostToDevice));
        HIP_CHECK(linear::q4g64_linear_prefill_cta_m64n128_qualification(a4_args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<hip_bfloat16> poisoned(static_cast<std::size_t>(tokens) * shape.rows);
        HIP_CHECK(hipMemcpy(poisoned.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
        if (!std::all_of(poisoned.begin(), poisoned.end(), [](hip_bfloat16 value) {
                return value.data == 0x7fc1U;
            })) {
            fail("A4 M64N128 did not poison every irregular-tail output");
        }
    }

    HIP_CHECK(linear::q4g64_quantize_activation({d_input.get(), a4}, nullptr));
    std::vector<std::uint8_t> actual_codes(a4.code_bytes);
    std::vector<std::uint16_t> actual_scales(a4.scale_bytes / 2U);
    HIP_CHECK(hipMemcpy(actual_codes.data(), a4.codes, a4.code_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual_scales.data(), a4.scales, a4.scale_bytes, hipMemcpyDeviceToHost));
    if (actual_codes != expected.codes || actual_scales != expected.scales) {
        fail("A4 codec differs from independent host image");
    }
    HIP_CHECK(linear::q4g64_linear_wmma32(a4_args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<hip_bfloat16> reference(static_cast<std::size_t>(tokens) * shape.rows);
    HIP_CHECK(hipMemcpy(reference.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
    HIP_CHECK(linear::q4g64_linear_prefill_cta_m64n128_qualification(a4_args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<hip_bfloat16> actual(reference.size());
    HIP_CHECK(hipMemcpy(actual.data(), d_output.get(), d_output.bytes(), hipMemcpyDeviceToHost));
    if (actual != reference) fail("A4 M64N128 is not bit-exact with the signed-A4 WMMA reference");

    const std::array<std::uint32_t, 8> sample_rows{0U, 1U, 15U, 16U,
        shape.rows / 3U, shape.rows / 2U, shape.rows - 2U, shape.rows - 1U};
    const std::array<std::uint32_t, 8> sample_tokens{0U, 1U, 15U, 16U,
        tokens / 3U, tokens / 2U, tokens - 2U, tokens - 1U};
    std::uint64_t maximum_steps = 0;
    const auto check_oracle = [&](std::uint32_t token, std::uint32_t row) {
        double sum = 0.0;
        for (std::uint32_t group = 0; group < groups; ++group) {
            std::int32_t dot = 0;
            for (std::uint32_t lane = 0; lane < kGroup; ++lane) {
                dot += signed_nibble(expected.codes, static_cast<std::size_t>(token) * padded +
                                     group * kGroup + lane) *
                       signed_nibble(weights, static_cast<std::size_t>(row) * padded +
                                     group * kGroup + lane);
            }
            sum = std::fma(static_cast<double>(dot),
                static_cast<double>(half_value(expected.scales[static_cast<std::size_t>(token) * groups + group])) *
                half_value(weight_scales[static_cast<std::size_t>(row) * groups + group]), sum);
        }
        const hip_bfloat16 oracle(static_cast<float>(sum));
        maximum_steps = std::max(maximum_steps, bf16_steps(actual[static_cast<std::size_t>(token) * shape.rows + row], oracle));
    };
    if (shape.rows <= 129U) {
        for (std::uint32_t token = 0; token < tokens; ++token)
            for (std::uint32_t row = 0; row < shape.rows; ++row) check_oracle(token, row);
    } else {
        for (std::uint32_t token : sample_tokens)
            for (std::uint32_t row : sample_rows) check_oracle(token, row);
    }
    if (maximum_steps > 2U) fail("A4 independent FP64 represented formula exceeds two BF16 steps");
    A4Timing result{shape, tokens, {}, {}, 0.0, 0.0, maximum_steps};
    if (timing) {
        HIP_CHECK(launch_incumbent()); HIP_CHECK(launch_a4()); HIP_CHECK(hipDeviceSynchronize());
        for (std::size_t trial = 0; trial < kTrials; ++trial) {
            if ((trial & 1U) == 0U) {
                result.incumbent[trial] = event_ms(1U, launch_incumbent, 0U);
                result.challenger[trial] = event_ms(1U, launch_a4, 0U);
            } else {
                result.challenger[trial] = event_ms(1U, launch_a4, 0U);
                result.incumbent[trial] = event_ms(1U, launch_incumbent, 0U);
            }
        }
        result.incumbent_median = median(result.incumbent);
        result.challenger_median = median(result.challenger);
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    const bool regression = argc == 2 && std::string_view(argv[1]) == "--regression";
    const bool benchmark = argc == 4 && std::string_view(argv[1]) == "--benchmark" &&
                           std::string_view(argv[2]) == "--out-json";
    if (!regression && !benchmark) {
        std::cerr << "usage: a4q4_prefill_qual --regression | "
                     "--benchmark --out-json FRESH.json\n";
        return 2;
    }
    try {
        linear::Q4G64PrefillQualificationResources resources{};
        HIP_CHECK(linear::q4g64_prefill_qualification_resources(&resources));
        if (resources.registers > 84 || resources.static_shared_bytes != 6528 ||
            resources.local_bytes != 0 || resources.max_threads_per_block < 512) {
            fail("A4Q4 M64N128 resource gate failed");
        }
        (void)run_a4_case(Shape{128, 256, 0, 0}, 192, false);
        (void)run_a4_case(Shape{129, 193, 0, 0}, 193, false);
        const std::array<Shape, 8> shapes{{{7168,5120,0,0},{4096,5120,0,0},
            {12288,5120,0,0},{5120,6144,0,0},{34816,5120,0,0},
            {5120,17408,0,0},{5120,10240,0,0},{1024,5120,0,0}}};
        std::vector<A4Timing> evidence;
        for (Shape shape : shapes) for (std::uint32_t tokens : {1024U,2048U,4096U,8192U}) {
            const PrefillResult a8_oracle = run_prefill_case(shape, tokens, false, false);
            if (a8_oracle.maximum_bf16_steps > 2U) fail("A8 independent oracle failed");
            evidence.push_back(run_a4_case(shape, tokens, benchmark));
        }
        if (benchmark) {
            const std::filesystem::path output = argv[3];
            require_fresh_output(output);
            double weighted_incumbent = 0.0, weighted_challenger = 0.0;
            constexpr std::array<std::uint32_t, 8> counts{32,48,48,65,65,65,1,0};
            bool no_regression = true;
            for (std::size_t i = 0; i < evidence.size(); ++i) {
                if (evidence[i].tokens == 2048U) {
                    const std::size_t shape_index = i / 4U;
                    weighted_incumbent += counts[shape_index] * evidence[i].incumbent_median;
                    weighted_challenger += counts[shape_index] * evidence[i].challenger_median;
                }
                no_regression = no_regression &&
                    evidence[i].challenger_median <= evidence[i].incumbent_median;
            }
            const bool operator_gate = no_regression &&
                weighted_challenger * 1.5 <= weighted_incumbent;
            std::ostringstream json;
            json << std::setprecision(17)
                 << "{\n  \"schema_version\": 1,\n"
                 << "  \"artifact_type\": \"ninfer_r9700_a4q4_prefill_m64n128_vs_a8q4_production_ab\",\n"
                 << "  \"production_state\": {\"dispatch_changed\": false, \"selected_activation\": \"A8G64\"},\n"
                 << "  \"numerical\": {\"a4_reference_bit_exact\": true, "
                    "\"a4_and_a8_independent_fp64_oracles\": true, "
                    "\"maximum_bf16_steps_allowed\": 2},\n"
                 << "  \"resources\": {\"registers\": " << resources.registers
                 << ", \"lds_bytes\": " << resources.static_shared_bytes
                 << ", \"local_bytes\": " << resources.local_bytes
                 << ", \"max_threads_per_block\": " << resources.max_threads_per_block << "},\n"
                 << "  \"rows\": [\n";
            for (std::size_t i = 0; i < evidence.size(); ++i) {
                const A4Timing& row = evidence[i];
                json << "    {\"shape\": [" << row.shape.rows << ',' << row.shape.columns
                     << "], \"tokens\": " << row.tokens
                     << ", \"incumbent_complete_ms\": " << row.incumbent_median
                     << ", \"challenger_complete_ms\": " << row.challenger_median
                     << ", \"maximum_a4_bf16_steps\": " << row.maximum_bf16_steps
                     << ", \"incumbent_samples_ms\": [";
                for (std::size_t j = 0; j < kTrials; ++j) json << (j ? "," : "") << row.incumbent[j];
                json << "], \"challenger_samples_ms\": [";
                for (std::size_t j = 0; j < kTrials; ++j) json << (j ? "," : "") << row.challenger[j];
                json << "]}" << (i + 1U == evidence.size() ? "\n" : ",\n");
            }
            json << "  ],\n  \"decision\": {\"weighted_incumbent_ms\": "
                 << weighted_incumbent << ", \"weighted_challenger_ms\": "
                 << weighted_challenger << ", \"no_regressing_tuple\": "
                 << (no_regression ? "true" : "false")
                 << ", \"minimum_operator_speedup\": 1.5, \"operator_gate_passed\": "
                 << (operator_gate ? "true" : "false")
                 << ", \"model_quality_and_whole_gates_required\": true}\n}\n";
            const std::string text = json.str();
            const int fd = ::open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
            if (fd < 0) fail("cannot create fresh A4Q4 report");
            const ssize_t written = ::write(fd, text.data(), text.size());
            if (::close(fd) != 0 || written != static_cast<ssize_t>(text.size()))
                fail("cannot write complete A4Q4 report");
            std::cout << "A4Q4 M64N128 physical A/B: "
                      << (operator_gate ? "OPERATOR PASS, MODEL GATES PENDING\n" : "REJECTED\n");
            return operator_gate ? 0 : 1;
        }
        std::cout << "A4Q4 M64N128 codec/reference/FP64/resource regression: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "A4Q4 M64N128 regression: FAIL: " << error.what() << '\n';
        return 1;
    }
}
