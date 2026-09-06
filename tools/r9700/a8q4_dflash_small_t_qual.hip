#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"

#include <cctype>

namespace {

#ifndef NINFER_DFLASH_SMALL_T_ROWS
#define NINFER_DFLASH_SMALL_T_ROWS 34816U
#define NINFER_DFLASH_SMALL_T_COLUMNS 5120U
#define NINFER_DFLASH_SMALL_T_WIDTHS 4U, 5U, 6U, 8U, 10U, 12U, 15U, 16U, 18U, 20U, 24U
#define NINFER_DFLASH_SMALL_T_CALL(args) \
    linear::a8q4g64_linear_dflash_small_t_qualification((args), nullptr)
#define NINFER_DFLASH_SMALL_T_SCOPE \
    "N34816/K5120 flattened K4/W5 and K5/W6 widths through C4 only"
#define NINFER_DFLASH_SMALL_T_SCHEMA "ninfer.r9700.a8q4-dflash-small-t-cell.v3"
#define NINFER_DFLASH_SMALL_T_LABEL "small-T"
#define NINFER_DFLASH_SMALL_T_QUALIFIER_SOURCE "tools/r9700/a8q4_dflash_small_t_qual.hip"
#endif

constexpr std::array<Shape, 1> kDFlashSmallTShapes{{
    {NINFER_DFLASH_SMALL_T_ROWS, NINFER_DFLASH_SMALL_T_COLUMNS, 0, 0},
}};

constexpr std::array kDFlashSmallTWidths{NINFER_DFLASH_SMALL_T_WIDTHS};

bool exact_shape(Shape shape) {
    return std::any_of(kDFlashSmallTShapes.begin(), kDFlashSmallTShapes.end(),
                       [&](Shape item) {
                           return item.rows == shape.rows && item.columns == shape.columns;
                       });
}

bool exact_width(std::uint32_t tokens) {
    return std::find(kDFlashSmallTWidths.begin(), kDFlashSmallTWidths.end(), tokens) !=
           kDFlashSmallTWidths.end();
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
        fail("small-T timing requires device-0 DPM profile auto before allocation");
    }
    return identity;
}

bool run_case(Shape shape, std::uint32_t tokens, const std::filesystem::path& output_path,
              const DeviceIdentity& device) {
    if (!exact_shape(shape) || !exact_width(tokens)) {
        fail(NINFER_DFLASH_SMALL_T_LABEL " qualifier received an unqualified shape/width");
    }
    const std::uint32_t padded = (shape.columns + 127U) / 128U * 128U;
    const std::uint32_t groups = padded / kGroup;
    const std::vector<hip_bfloat16> input = make_input(shape.columns, tokens);
    const HostActivation represented = quantize_host(input, tokens, shape.columns);
    const DecodeDot8Weights weights = make_decode_dot8_weights(shape.rows, padded);

    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<std::uint8_t> d_weight_codes(weights.codes.size());
    DeviceBuffer<std::uint16_t> d_weight_scales(weights.scales.size());
    DeviceBuffer<hip_bfloat16> d_output(static_cast<std::size_t>(tokens) * shape.rows + 2U);
    const std::size_t workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(tokens, shape.columns);
    DeviceBuffer<std::uint8_t> d_workspace(workspace_bytes);
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight_codes.get(), weights.codes.data(), d_weight_codes.bytes(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight_scales.get(), weights.scales.data(), d_weight_scales.bytes(),
                        hipMemcpyHostToDevice));
    linear::A8G64ActivationWorkspace workspace{};
    HIP_CHECK(linear::a8q4g64_bind_activation_workspace(
        d_workspace.get(), workspace_bytes, tokens, shape.columns, &workspace));
    HIP_CHECK(linear::a8g64_quantize_activation({d_input.get(), workspace}, nullptr));
    const linear::A8Q4G64LinearArgs args{
        workspace.low_codes, workspace.low_code_bytes, workspace.high_codes,
        workspace.high_code_bytes, workspace.scales, workspace.scale_bytes,
        workspace.status, d_weight_codes.get(), d_weight_codes.bytes(),
        d_weight_scales.get(), d_weight_scales.bytes(), d_output.get() + 1U,
        tokens, shape.rows, shape.columns, padded};
    const auto candidate = [&] {
        return NINFER_DFLASH_SMALL_T_CALL(args);
    };
    const auto incumbent = [&] { return linear::a8q4g64_linear_wmma32(args, nullptr); };

    linear::A8Q4G64LinearArgs invalid = args;
    invalid.rows = 4096U;
    if (NINFER_DFLASH_SMALL_T_CALL(invalid) != hipErrorInvalidValue) {
        fail("small-T candidate accepted a C1-rejected matrix shape");
    }
    invalid = args;
    invalid.tokens = 3U;
    if (NINFER_DFLASH_SMALL_T_CALL(invalid) != hipErrorInvalidValue) {
        fail("small-T candidate accepted T=3");
    }
    invalid = args;
    invalid.tokens = 7U;
    if (NINFER_DFLASH_SMALL_T_CALL(invalid) != hipErrorInvalidValue) {
        fail("small-T candidate accepted T=7");
    }
#ifdef NINFER_DFLASH_MLP_DOWN_T5_ONLY
    invalid = args;
    invalid.tokens = 6U;
    if (NINFER_DFLASH_SMALL_T_CALL(invalid) != hipErrorInvalidValue) {
        fail("MLP-down small-T candidate accepted spill-rejected T=6");
    }
#endif
    invalid = args;
    invalid.tokens = 9U;
    if (NINFER_DFLASH_SMALL_T_CALL(invalid) != hipErrorInvalidValue) {
        fail("small-T candidate accepted T=9");
    }
    invalid = args;
    ++invalid.activation_low_codes;
    if (NINFER_DFLASH_SMALL_T_CALL(invalid) != hipErrorInvalidValue) {
        fail("small-T candidate accepted a misaligned activation plane");
    }
    invalid = args;
    --invalid.weight_code_bytes;
    if (NINFER_DFLASH_SMALL_T_CALL(invalid) != hipErrorInvalidValue) {
        fail("small-T candidate accepted an undersized weight plane");
    }
    invalid = args;
    invalid.output = reinterpret_cast<hip_bfloat16*>(
        const_cast<std::uint8_t*>(invalid.activation_low_codes));
    if (NINFER_DFLASH_SMALL_T_CALL(invalid) != hipErrorInvalidValue) {
        fail("small-T candidate accepted output/input aliasing");
    }

    std::vector<hip_bfloat16> guarded(static_cast<std::size_t>(tokens) * shape.rows + 2U);
    const auto initialize = [&](std::uint16_t bits) {
        for (hip_bfloat16& value : guarded) value.data = bits;
        HIP_CHECK(hipMemcpy(d_output.get(), guarded.data(), d_output.bytes(),
                            hipMemcpyHostToDevice));
    };
    const auto download = [&](std::uint16_t canary) {
        HIP_CHECK(hipMemcpy(guarded.data(), d_output.get(), d_output.bytes(),
                            hipMemcpyDeviceToHost));
        if (guarded.front().data != canary || guarded.back().data != canary) {
            fail("small-T candidate changed an output guard");
        }
        return std::vector<hip_bfloat16>(guarded.begin() + 1, guarded.end() - 1);
    };

    const std::uint32_t bad_status = linear::Q4G64ActivationNonfinite;
    HIP_CHECK(hipMemcpy(workspace.status, &bad_status, sizeof(bad_status),
                        hipMemcpyHostToDevice));
    initialize(0x7fc2U);
    HIP_CHECK(candidate());
    HIP_CHECK(hipDeviceSynchronize());
    const auto poisoned = download(0x7fc2U);
    if (!std::all_of(poisoned.begin(), poisoned.end(), [](hip_bfloat16 value) {
            return value.data == 0x7fc1U;
        })) {
        fail("small-T candidate did not poison every output for nonzero status");
    }
    HIP_CHECK(linear::a8g64_quantize_activation({d_input.get(), workspace}, nullptr));

    initialize(0x7fc2U);
    HIP_CHECK(candidate());
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<hip_bfloat16> candidate_output = download(0x7fc2U);
    if (std::any_of(candidate_output.begin(), candidate_output.end(), [](hip_bfloat16 value) {
            return value.data == 0x7fc2U;
        })) {
        fail("small-T candidate did not rewrite every output");
    }
    const std::array<std::uint32_t, 24> sampled_rows{
        0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U,
        16U, 31U, 32U, shape.rows / 3U, shape.rows / 2U, shape.rows - 17U,
        shape.rows - 16U, shape.rows - 1U};
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
                    values[static_cast<std::size_t>(token) * shape.rows + row];
                maximum_steps = std::max(maximum_steps, bf16_steps(actual, oracle));
                if (bf16_steps(actual, oracle) > 2U) {
                    fail(std::string(route) +
                         " differs from the independent FP64 represented oracle");
                }
            }
        }
        return maximum_steps;
    };
    const std::uint64_t candidate_maximum_steps =
        oracle_steps(candidate_output, "small-T candidate");

    initialize(0x7fc3U);
    HIP_CHECK(incumbent());
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<hip_bfloat16> incumbent_output = download(0x7fc3U);
    if (std::any_of(incumbent_output.begin(), incumbent_output.end(), [](hip_bfloat16 value) {
            return value.data == 0x7fc3U;
        })) {
        fail("WMMA incumbent did not rewrite every output");
    }
    const std::uint64_t incumbent_maximum_steps =
        oracle_steps(incumbent_output, "WMMA incumbent");
    const bool pairwise_bit_exact =
        std::equal(candidate_output.begin(), candidate_output.end(), incumbent_output.begin(),
                   [](hip_bfloat16 lhs, hip_bfloat16 rhs) { return lhs.data == rhs.data; });

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
        candidate_forward[trial] = event_ms(10, candidate);
        candidate_reverse[trial] = event_ms(10, candidate);
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
    const double incumbent_forward_median = median(incumbent_forward);
    const double candidate_forward_median = median(candidate_forward);
    const double incumbent_reverse_median = median(incumbent_reverse);
    const double candidate_reverse_median = median(candidate_reverse);
    const double forward_ratio_median = median(forward_ratios);
    const double reverse_ratio_median = median(reverse_ratios);
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
    const double order_ratio_delta = std::abs(forward_ratio_median - reverse_ratio_median);
    constexpr double kMaximumOrderRatioDelta = 0.02;
    const bool accepted = candidate_median < incumbent_median &&
                          candidate_forward_median < incumbent_forward_median &&
                          candidate_reverse_median < incumbent_reverse_median &&
                          forward_ratio_median < 1.0 && reverse_ratio_median < 1.0 &&
                          ratio_upper < 1.0 && order_ratio_delta <= kMaximumOrderRatioDelta;
    if (read_text(device.power_path) != "auto") {
        fail("small-T timing DPM profile changed before publication");
    }

    const std::filesystem::path executable = std::filesystem::canonical("/proc/self/exe");
    const std::filesystem::path source_root = std::filesystem::canonical(NINFER_SOURCE_DIR);
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\n  \"schema\": \"" NINFER_DFLASH_SMALL_T_SCHEMA "\",\n"
        << "  \"status\": \"" << (accepted ? "passed" : "rejected") << "\",\n"
        << "  \"production_dispatch_changed\": false,\n"
        << "  \"scope\": \"" NINFER_DFLASH_SMALL_T_SCOPE "\",\n"
        << "  \"hardware\": {\"device\": \"" << json_escape(device.properties.name)
        << "\", \"architecture\": \"" << json_escape(device.properties.gcnArchName)
        << "\", \"pci_bus_id\": \"" << json_escape(device.pci_bus_id)
        << "\", \"pci_vendor_device\": \"1002:7551\", \"integrated\": false, "
           "\"wavefront_width\": 32, \"power_profile_before_after\": \"auto\"},\n"
        << "  \"shape\": {\"rows\": " << shape.rows << ", \"columns\": "
        << shape.columns << ", \"tokens\": " << tokens << "},\n"
        << "  \"numeric\": {\"oracle\": \"independent FP64 represented A8G64 x Q4G64 formula\", "
           "\"maximum_bf16_steps_allowed\": 2, \"candidate_maximum_bf16_steps\": "
        << candidate_maximum_steps << ", \"incumbent_maximum_bf16_steps\": "
        << incumbent_maximum_steps << ", \"pairwise_bit_exact_diagnostic\": "
        << (pairwise_bit_exact ? "true" : "false") << "},\n"
        << "  \"timing\": {\"method\": \"unprofiled HIP events\", "
           "\"iterations_per_sample\": 10, \"forward_reverse_pairs\": 7,\n";
    const auto emit = [&](const char* name, const auto& values, bool comma = true) {
        out << "    \"" << name << "\": ";
        write_samples(out, values);
        out << (comma ? ",\n" : "\n");
    };
    emit("incumbent_forward_ms", incumbent_forward);
    emit("candidate_forward_ms", candidate_forward);
    emit("candidate_reverse_ms", candidate_reverse);
    emit("incumbent_reverse_ms", incumbent_reverse);
    emit("incumbent_balanced_ms", incumbent_balanced);
    emit("candidate_balanced_ms", candidate_balanced);
    emit("forward_candidate_over_incumbent", forward_ratios);
    emit("reverse_candidate_over_incumbent", reverse_ratios, false);
    out << "  },\n  \"decision\": {\"criterion\": \"both launch-order medians faster, two-standard-error paired ratio upper below one, order ratio delta at most 0.02\", "
        << "\"incumbent_median_ms\": " << incumbent_median
        << ", \"candidate_median_ms\": " << candidate_median
        << ", \"candidate_over_incumbent\": " << candidate_median / incumbent_median
        << ", \"incumbent_forward_median_ms\": " << incumbent_forward_median
        << ", \"candidate_forward_median_ms\": " << candidate_forward_median
        << ", \"incumbent_reverse_median_ms\": " << incumbent_reverse_median
        << ", \"candidate_reverse_median_ms\": " << candidate_reverse_median
        << ", \"forward_ratio_median\": " << forward_ratio_median
        << ", \"reverse_ratio_median\": " << reverse_ratio_median
        << ", \"paired_ratio_mean\": " << ratio_mean
        << ", \"paired_ratio_standard_deviation\": " << ratio_standard_deviation
        << ", \"paired_ratio_upper\": " << ratio_upper
        << ", \"order_ratio_delta\": " << order_ratio_delta
        << ", \"maximum_order_ratio_delta\": " << kMaximumOrderRatioDelta
        << ", \"accepted\": " << (accepted ? "true" : "false") << "},\n"
        << "  \"executable\": {\"path\": \"" << json_escape(executable.string())
        << "\", \"sha256\": \"" << sha256_file(executable) << "\"},\n"
        << "  \"sources\": {\"kernel\": \""
        << sha256_file(source_root / "src/ops/r9700/linear/r9700_linear.hip")
        << "\", \"contract\": \""
        << sha256_file(source_root / "src/ops/r9700/linear/r9700_linear.h")
        << "\", \"qualifier\": \""
        << sha256_file(source_root / NINFER_DFLASH_SMALL_T_QUALIFIER_SOURCE)
#ifdef NINFER_DFLASH_SMALL_T_SHARED_HARNESS_SOURCE
        << "\", \"shared_harness\": \""
        << sha256_file(source_root / NINFER_DFLASH_SMALL_T_SHARED_HARNESS_SOURCE)
#endif
        << "\"}\n}\n";
    const std::string report = out.str();
    const int descriptor =
        ::open(output_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) fail("cannot exclusively create small-T report");
    std::size_t offset = 0U;
    while (offset < report.size()) {
        const ssize_t written =
            ::write(descriptor, report.data() + offset, report.size() - offset);
        if (written <= 0) {
            (void)::close(descriptor);
            (void)::unlink(output_path.c_str());
            fail("cannot write small-T report");
        }
        offset += static_cast<std::size_t>(written);
    }
    if (::close(descriptor) != 0) {
        (void)::unlink(output_path.c_str());
        fail("cannot close small-T report");
    }
    std::cout << "A8Q4 DFlash small-T qualification: "
              << (accepted ? "PASS" : "REJECTED") << " rows=" << shape.rows
              << " columns=" << shape.columns << " tokens=" << tokens
              << " incumbent_ms=" << incumbent_median
              << " candidate_ms=" << candidate_median << '\n';
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
        if (argc != 6 || std::string_view(argv[4]) != "--out-json") {
            std::cerr << "usage: a8q4_dflash_small_t_qual ROWS COLUMNS TOKENS "
                         "--out-json FRESH.json\n";
            return 2;
        }
        const std::filesystem::path output_path = argv[5];
        require_fresh_output(output_path);
        const DeviceIdentity device = select_one_r9700_auto();
        return run_case(
                   Shape{parse_u32(argv[1], "rows"), parse_u32(argv[2], "columns"), 0, 0},
                   parse_u32(argv[3], "tokens"), output_path, device)
                   ? 0
                   : 1;
    } catch (const std::exception& error) {
        std::cerr << "A8Q4 DFlash small-T qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
