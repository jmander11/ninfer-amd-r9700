#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"

#include <functional>

namespace {

constexpr std::uint32_t kColumns = 5120U;
constexpr std::uint32_t kRows0 = 4096U;
constexpr std::uint32_t kRows1 = 12288U;
constexpr std::uint32_t kGroups = kColumns / kGroup;
constexpr std::size_t kCopies = 3U;
constexpr std::size_t kPairTrials = 12U;
constexpr std::size_t kSamplesPerCopy = kPairTrials / kCopies;
constexpr std::array<bool, kPairTrials> kSerialFirst{
    true, true, false, true, false, true, false, false, true, false, true, false};

struct DeviceWeights {
    std::unique_ptr<DeviceBuffer<std::uint8_t>> codes0;
    std::unique_ptr<DeviceBuffer<std::uint16_t>> scales0;
    std::unique_ptr<DeviceBuffer<std::uint8_t>> codes1;
    std::unique_ptr<DeviceBuffer<std::uint16_t>> scales1;
};

struct GuardedOutput {
    DeviceBuffer<hip_bfloat16> storage;
    std::uint32_t rows;
    explicit GuardedOutput(std::uint32_t count) : storage(count + 2U), rows(count) {}
    hip_bfloat16* payload() const { return storage.get() + 1U; }
};

void initialize_guard(GuardedOutput& output, std::uint16_t bits) {
    std::vector<hip_bfloat16> host(static_cast<std::size_t>(output.rows) + 2U);
    for (auto& value : host) value.data = bits;
    HIP_CHECK(hipMemcpy(output.storage.get(), host.data(), output.storage.bytes(),
                        hipMemcpyHostToDevice));
}

std::vector<hip_bfloat16> download_guarded(const GuardedOutput& output,
                                            std::uint16_t bits) {
    std::vector<hip_bfloat16> host(static_cast<std::size_t>(output.rows) + 2U);
    HIP_CHECK(hipMemcpy(host.data(), output.storage.get(), output.storage.bytes(),
                        hipMemcpyDeviceToHost));
    if (host.front().data != bits || host.back().data != bits) {
        fail("paired GDN projection changed an output canary");
    }
    return {host.begin() + 1, host.end() - 1};
}

linear::A8Q4G64CandidateArgs serial_args(
    const hip_bfloat16* input, const DeviceWeights& weights, bool second,
    void* workspace, std::size_t workspace_bytes, hip_bfloat16* output) {
    const std::uint32_t rows = second ? kRows1 : kRows0;
    const auto* codes = second ? weights.codes1->get() : weights.codes0->get();
    const auto* scales = second ? weights.scales1->get() : weights.scales0->get();
    const std::size_t code_bytes = second ? weights.codes1->bytes() : weights.codes0->bytes();
    const std::size_t scale_bytes = second ? weights.scales1->bytes() : weights.scales0->bytes();
    return {input, codes, code_bytes, scales, scale_bytes, workspace,
            workspace_bytes, output, 1U, rows, kColumns, kColumns};
}

linear::A8Q4G64GdnPairArgs pair_args(
    const hip_bfloat16* input, const DeviceWeights& weights, void* workspace,
    std::size_t workspace_bytes, hip_bfloat16* output0, hip_bfloat16* output1) {
    return {input,
            weights.codes0->get(), weights.codes0->bytes(),
            weights.scales0->get(), weights.scales0->bytes(), output0,
            weights.codes1->get(), weights.codes1->bytes(),
            weights.scales1->get(), weights.scales1->bytes(), output1,
            workspace, workspace_bytes, 1U, kColumns};
}

hipError_t launch_serial(const hip_bfloat16* input, const DeviceWeights& weights,
                         void* workspace, std::size_t workspace_bytes,
                         hip_bfloat16* output0, hip_bfloat16* output1) {
    hipError_t status = linear::a8q4g64_linear_candidate(
        serial_args(input, weights, false, workspace, workspace_bytes, output0), nullptr);
    if (status != hipSuccess) return status;
    return linear::a8q4g64_linear_candidate(
        serial_args(input, weights, true, workspace, workspace_bytes, output1), nullptr);
}

std::uint64_t check_fp64_output(const std::vector<hip_bfloat16>& actual,
                                const HostActivation& activation,
                                const DecodeDot8Weights& weights,
                                std::uint32_t rows) {
    std::uint64_t maximum_steps = 0U;
    for (std::uint32_t row = 0U; row < rows; ++row) {
        double sum = 0.0;
        for (std::uint32_t group = 0U; group < kGroups; ++group) {
            std::int32_t integer_dot = 0;
            for (std::uint32_t lane = 0U; lane < kGroup; ++lane) {
                const std::size_t index = static_cast<std::size_t>(group) * kGroup + lane;
                const int activation_code = unsigned_nibble(activation.low, index) +
                                            16 * signed_nibble(activation.high, index);
                integer_dot += activation_code * decode_dot8_weight_code(row, group, lane);
            }
            const std::size_t scale_index =
                ((static_cast<std::size_t>(row) >> 4U) * kGroups + group) * 16U +
                (row & 15U);
            sum = std::fma(static_cast<double>(integer_dot),
                           static_cast<double>(half_value(activation.scales[group])) *
                               half_value(weights.scales[scale_index]),
                           sum);
        }
        const hip_bfloat16 oracle(static_cast<float>(sum));
        const std::uint64_t steps = bf16_steps(actual[row], oracle);
        maximum_steps = std::max(maximum_steps, steps);
        if (steps > 2U) fail("paired GDN projection differs from FP64 represented oracle");
    }
    return maximum_steps;
}

double time_once(const std::function<hipError_t()>& launch) {
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

template <std::size_t N>
double pair_median(std::array<double, N> samples) {
    std::sort(samples.begin(), samples.end());
    if constexpr ((N & 1U) != 0U) return samples[N / 2U];
    return 0.5 * (samples[N / 2U - 1U] + samples[N / 2U]);
}

void require_invalid(linear::A8Q4G64GdnPairArgs args, const char* label) {
    if (linear::a8q4g64_gdn_pair_t1(args, nullptr) != hipErrorInvalidValue) {
        fail(std::string("paired GDN projection accepted ") + label);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5 || std::string_view(argv[1]) != "--out-json" ||
            std::string_view(argv[3]) != "--assembly") {
            std::cerr << "usage: a8q4_gdn_pair_t1_qual --out-json FRESH.json "
                         "--assembly a8q4_gdn_pair_t1.s\n";
            return 2;
        }
        const std::filesystem::path output_path = argv[2];
        const std::filesystem::path assembly = std::filesystem::canonical(argv[4]);
        require_fresh_output(output_path);
        const std::filesystem::path source_root =
            std::filesystem::canonical(NINFER_SOURCE_DIR);
        const std::filesystem::path executable =
            std::filesystem::canonical("/proc/self/exe");
        const std::filesystem::path checker = std::filesystem::canonical(
            source_root / "tools/r9700/check_a8q4_gdn_pair_t1_static.py");
        const std::string static_command = "python3 " + shell_quote(checker.string()) +
                                           " " + shell_quote(assembly.string());
        if (std::system(static_command.c_str()) != 0) {
            fail("paired GDN static receipt failed");
        }
        int count = 0;
        HIP_CHECK(hipGetDeviceCount(&count));
        if (count <= 0) fail("paired GDN qualification requires device 0");
        HIP_CHECK(hipSetDevice(0));
        char pci_bus_id[32]{};
        HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
        std::string pci = pci_bus_id;
        if (pci.size() == 7U) pci = "0000:" + pci;
        std::transform(pci.begin(), pci.end(), pci.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        const std::filesystem::path power =
            std::filesystem::path("/sys/bus/pci/devices") / pci /
            "power_dpm_force_performance_level";
        if (read_text(power) != "auto") fail("paired GDN qualification requires auto power");
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, 0));
        if (std::string_view(properties.name) != "AMD Radeon AI PRO R9700" ||
            std::string_view(properties.gcnArchName) != "gfx1201" ||
            properties.warpSize != 32) {
            fail("paired GDN qualification requires the R9700 gfx1201 wave32 target");
        }

        const std::vector<hip_bfloat16> input = make_decode_dot8_input(kColumns);
        const HostActivation activation = quantize_host(input, 1U, kColumns);
        const DecodeDot8Weights host0 = make_decode_dot8_weights(kRows0, kColumns);
        const DecodeDot8Weights host1 = make_decode_dot8_weights(kRows1, kColumns);
        DeviceBuffer<hip_bfloat16> d_input(input.size());
        HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));
        std::array<DeviceWeights, kCopies> copies;
        for (DeviceWeights& copy : copies) {
            copy.codes0 = std::make_unique<DeviceBuffer<std::uint8_t>>(host0.codes.size());
            copy.scales0 = std::make_unique<DeviceBuffer<std::uint16_t>>(host0.scales.size());
            copy.codes1 = std::make_unique<DeviceBuffer<std::uint8_t>>(host1.codes.size());
            copy.scales1 = std::make_unique<DeviceBuffer<std::uint16_t>>(host1.scales.size());
            HIP_CHECK(hipMemcpy(copy.codes0->get(), host0.codes.data(), copy.codes0->bytes(),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(copy.scales0->get(), host0.scales.data(), copy.scales0->bytes(),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(copy.codes1->get(), host1.codes.data(), copy.codes1->bytes(),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(copy.scales1->get(), host1.scales.data(), copy.scales1->bytes(),
                                hipMemcpyHostToDevice));
        }
        const std::size_t workspace_bytes =
            linear::a8q4g64_activation_workspace_capacity_bytes(1U, kColumns);
        DeviceBuffer<std::uint8_t> workspace(workspace_bytes);
        GuardedOutput serial0(kRows0), serial1(kRows1), paired0(kRows0), paired1(kRows1);

        auto args = pair_args(d_input.get(), copies[0], workspace.get(), workspace_bytes,
                              paired0.payload(), paired1.payload());
        auto malformed = args;
        malformed.tokens = 2U;
        require_invalid(malformed, "T2");
        malformed = args;
        malformed.columns = 4992U;
        require_invalid(malformed, "K4992");
        malformed = args;
        --malformed.weight1_code_bytes;
        require_invalid(malformed, "short second code plane");
        malformed = args;
        ++malformed.weight0_codes;
        require_invalid(malformed, "misaligned first code plane");
        malformed = args;
        malformed.output1 = malformed.output0;
        require_invalid(malformed, "aliased outputs");
        malformed = args;
        malformed.output0 = reinterpret_cast<hip_bfloat16*>(workspace.get());
        require_invalid(malformed, "workspace/output alias");
        malformed = args;
        malformed.input = reinterpret_cast<const hip_bfloat16*>(malformed.weight0_codes);
        require_invalid(malformed, "input/weight alias");

        initialize_guard(serial0, 0x7fc2U);
        initialize_guard(serial1, 0x7fc2U);
        initialize_guard(paired0, 0x7fc3U);
        initialize_guard(paired1, 0x7fc3U);
        HIP_CHECK(launch_serial(d_input.get(), copies[0], workspace.get(), workspace_bytes,
                                serial0.payload(), serial1.payload()));
        HIP_CHECK(linear::a8q4g64_gdn_pair_t1(args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        const auto serial_host0 = download_guarded(serial0, 0x7fc2U);
        const auto serial_host1 = download_guarded(serial1, 0x7fc2U);
        const auto paired_host0 = download_guarded(paired0, 0x7fc3U);
        const auto paired_host1 = download_guarded(paired1, 0x7fc3U);
        if (!std::equal(serial_host0.begin(), serial_host0.end(), paired_host0.begin(),
                        [](hip_bfloat16 a, hip_bfloat16 b) { return a.data == b.data; }) ||
            !std::equal(serial_host1.begin(), serial_host1.end(), paired_host1.begin(),
                        [](hip_bfloat16 a, hip_bfloat16 b) { return a.data == b.data; })) {
            fail("paired GDN projection is not bit-exact to two serial production calls");
        }
        const std::uint64_t max_steps = std::max(
            check_fp64_output(paired_host0, activation, host0, kRows0),
            check_fp64_output(paired_host1, activation, host1, kRows1));

        hip_bfloat16 nonfinite{};
        nonfinite.data = 0x7fc1U;
        HIP_CHECK(hipMemcpy(d_input.get(), &nonfinite, sizeof(nonfinite),
                            hipMemcpyHostToDevice));
        initialize_guard(paired0, 0x7fc4U);
        initialize_guard(paired1, 0x7fc4U);
        HIP_CHECK(linear::a8q4g64_gdn_pair_t1(
            pair_args(d_input.get(), copies[0], workspace.get(), workspace_bytes,
                      paired0.payload(), paired1.payload()), nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        const auto poison_host0 = download_guarded(paired0, 0x7fc4U);
        const auto poison_host1 = download_guarded(paired1, 0x7fc4U);
        if (!std::all_of(poison_host0.begin(), poison_host0.end(), [](hip_bfloat16 v) {
                return v.data == 0x7fc1U;
            }) || !std::all_of(poison_host1.begin(), poison_host1.end(), [](hip_bfloat16 v) {
                return v.data == 0x7fc1U;
            })) fail("paired GDN kernel status contract did not poison both outputs");
        HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(),
                            hipMemcpyHostToDevice));

        // Warm both arms while traversing all three disjoint 42.5-MiB pairs. The
        // measured launch sequence continues the 0,1,2 cycle, so at least two
        // complete pairs (>85 MiB) intervene before any allocation is revisited.
        for (std::size_t launch = 0U; launch < 6U; ++launch) {
            const DeviceWeights& w = copies[launch % kCopies];
            if ((launch & 1U) == 0U) {
                HIP_CHECK(launch_serial(d_input.get(), w, workspace.get(), workspace_bytes,
                                        serial0.payload(), serial1.payload()));
            } else {
                HIP_CHECK(linear::a8q4g64_gdn_pair_t1(
                    pair_args(d_input.get(), w, workspace.get(), workspace_bytes,
                              paired0.payload(), paired1.payload()), nullptr));
            }
        }
        HIP_CHECK(hipDeviceSynchronize());
        std::array<double, kPairTrials> serial_samples{};
        std::array<double, kPairTrials> paired_samples{};
        std::array<std::array<double, kSamplesPerCopy>, kCopies> serial_by_copy{};
        std::array<std::array<double, kSamplesPerCopy>, kCopies> paired_by_copy{};
        std::array<std::size_t, kCopies> serial_counts{};
        std::array<std::size_t, kCopies> paired_counts{};
        for (std::size_t trial = 0U; trial < kPairTrials; ++trial) {
            const std::size_t first_copy = (2U * trial) % kCopies;
            const std::size_t second_copy = (first_copy + 1U) % kCopies;
            const std::size_t serial_copy = kSerialFirst[trial] ? first_copy : second_copy;
            const std::size_t paired_copy = kSerialFirst[trial] ? second_copy : first_copy;
            const auto serial_launch = [&, serial_copy] {
                const DeviceWeights& w = copies[serial_copy];
                return launch_serial(d_input.get(), w, workspace.get(), workspace_bytes,
                                     serial0.payload(), serial1.payload());
            };
            const auto paired_launch = [&, paired_copy] {
                const DeviceWeights& w = copies[paired_copy];
                return linear::a8q4g64_gdn_pair_t1(
                    pair_args(d_input.get(), w, workspace.get(), workspace_bytes,
                              paired0.payload(), paired1.payload()), nullptr);
            };
            if (kSerialFirst[trial]) {
                serial_samples[trial] = time_once(serial_launch);
                paired_samples[trial] = time_once(paired_launch);
            } else {
                paired_samples[trial] = time_once(paired_launch);
                serial_samples[trial] = time_once(serial_launch);
            }
            serial_by_copy[serial_copy][serial_counts[serial_copy]++] = serial_samples[trial];
            paired_by_copy[paired_copy][paired_counts[paired_copy]++] = paired_samples[trial];
        }
        if (!std::all_of(serial_counts.begin(), serial_counts.end(), [](std::size_t count) {
                return count == kSamplesPerCopy;
            }) || !std::all_of(paired_counts.begin(), paired_counts.end(), [](std::size_t count) {
                return count == kSamplesPerCopy;
            })) fail("paired GDN timing schedule is not allocation-balanced");
        const double serial_ms = pair_median(serial_samples);
        const double paired_ms = pair_median(paired_samples);
        std::array<double, kCopies> serial_copy_medians{};
        std::array<double, kCopies> paired_copy_medians{};
        for (std::size_t copy = 0U; copy < kCopies; ++copy) {
            serial_copy_medians[copy] = pair_median(serial_by_copy[copy]);
            paired_copy_medians[copy] = pair_median(paired_by_copy[copy]);
            if (paired_copy_medians[copy] > serial_copy_medians[copy] * 0.99) {
                fail("paired GDN route did not clear the direct gate on every allocation");
            }
        }
        linear::A8Q4G64GdnPairResources resources{};
        HIP_CHECK(linear::a8q4g64_gdn_pair_t1_resources(&resources));
        if (resources.local_bytes != 0 || resources.static_shared_bytes != 0 ||
            resources.max_threads_per_block < 256) {
            fail("paired GDN kernel resource envelope is unsuitable");
        }
        const std::size_t pair_weight_bytes = copies[0].codes0->bytes() +
            copies[0].scales0->bytes() + copies[0].codes1->bytes() +
            copies[0].scales1->bytes();
        if (read_text(power) != "auto") fail("power changed during paired GDN qualification");
        int runtime_version = 0;
        int driver_version = 0;
        HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
        HIP_CHECK(hipDriverGetVersion(&driver_version));
        const std::array<std::pair<const char*, const char*>, 4> source_specs{{
            {"qualifier", "tools/r9700/a8q4_gdn_pair_t1_qual.hip"},
            {"shared_qualifier", "tools/r9700/a8q4_shape_sweep_qual.hip"},
            {"contract", "src/ops/r9700/linear/r9700_linear.h"},
            {"implementation", "src/ops/r9700/linear/r9700_linear.hip"},
        }};
        std::ostringstream report;
        report << std::setprecision(17)
               << "{\n  \"schema\": \"ninfer.r9700.a8q4-gdn-pair-t1-production-regression.v1\",\n"
               << "  \"status\": \"production_route_direct_regression_passed\",\n"
               << "  \"production_routing_selected\": true,\n"
               << "  \"command\": \"";
        for (int index = 0; index < argc; ++index) {
            if (index != 0) report << ' ';
            report << json_escape(argv[index]);
        }
        report << "\",\n  \"executable\": {\"path\": \""
               << json_escape(executable.string()) << "\", \"bytes\": "
               << std::filesystem::file_size(executable) << ", \"sha256\": \""
               << sha256_file(executable) << "\"},\n  \"sources\": [\n";
        for (std::size_t index = 0; index < source_specs.size(); ++index) {
            const auto path = std::filesystem::canonical(source_root / source_specs[index].second);
            report << "    {\"role\": \"" << source_specs[index].first
                   << "\", \"path\": \"" << json_escape(path.string())
                   << "\", \"bytes\": " << std::filesystem::file_size(path)
                   << ", \"sha256\": \"" << sha256_file(path) << "\"}"
                   << (index + 1U == source_specs.size() ? "\n" : ",\n");
        }
        report << "  ],\n  \"device\": {\"name\": \"" << json_escape(properties.name)
               << "\", \"architecture\": \"" << json_escape(properties.gcnArchName)
               << "\", \"pci_bus_id\": \"" << pci
               << "\", \"wave_size\": " << properties.warpSize
               << ", \"hip_runtime_version\": " << runtime_version
               << ", \"hip_driver_version\": " << driver_version << "},\n"
               << "  \"power_profile_before\": \"auto\",\n"
               << "  \"power_profile_after\": \"auto\",\n"
               << "  \"static_receipt\": {\"status\": \"passed\", \"command\": \""
               << json_escape(static_command) << "\", \"checker_sha256\": \""
               << sha256_file(checker) << "\", \"assembly_sha256\": \""
               << sha256_file(assembly) << "\", \"exact_route_symbol\": true, "
                  "\"native_iu4_dot8\": true, \"wave_size\": 32, \"vgpr\": "
               << resources.registers << ", \"sgpr\": 44, \"lds_bytes\": 0, "
                  "\"scratch_bytes\": 0, \"spill_count\": 0},\n"
               << "  \"correctness\": {\"both_complete_outputs_bit_exact_to_serial\": true, "
                  "\"independent_fp64_represented_oracle\": true, "
                  "\"maximum_bf16_steps\": " << max_steps
               << ", \"status_poisoning\": true, \"canaries\": true, "
                  "\"malformed_extent_alignment_alias_rejection\": true},\n"
               << "  \"timing\": {\"scope\": \"complete serial quant+dot twice versus "
                  "quant once+combined dot\", \"trials_per_arm\": " << kPairTrials
               << ", \"disjoint_weight_copies\": " << kCopies
               << ", \"weight_pair_bytes\": " << pair_weight_bytes
               << ", \"intervening_weight_bytes\": " << 2U * pair_weight_bytes
               << ", \"arm_order_balanced\": true, \"allocation_balanced\": true,\n"
               << "    \"serial_samples_ms\": [";
        for (std::size_t i = 0; i < kPairTrials; ++i) report << (i ? "," : "") << serial_samples[i];
        report << "],\n    \"paired_samples_ms\": [";
        for (std::size_t i = 0; i < kPairTrials; ++i) report << (i ? "," : "") << paired_samples[i];
        report << "],\n    \"serial_copy_medians_ms\": [";
        for (std::size_t i = 0; i < kCopies; ++i) report << (i ? "," : "") << serial_copy_medians[i];
        report << "],\n    \"paired_copy_medians_ms\": [";
        for (std::size_t i = 0; i < kCopies; ++i) report << (i ? "," : "") << paired_copy_medians[i];
        report << "],\n    \"serial_median_ms\": " << serial_ms
               << ", \"paired_median_ms\": " << paired_ms
               << ", \"speedup\": " << serial_ms / paired_ms
               << ", \"all_copy_medians_clear_one_percent\": true},\n"
               << "  \"decision\": \"production_direct_regression_passed\"\n}\n";
        std::ofstream output(output_path, std::ios::binary | std::ios::out);
        if (!output) fail("cannot open paired GDN qualification report");
        output << report.str();
        output.close();
        if (!output) fail("cannot finish paired GDN qualification report");
        std::cout << std::fixed << std::setprecision(6)
                  << "gdn-pair-t1 qualification=PASS bit_exact=true fp64_max_bf16_steps="
                  << max_steps << " serial_complete_ms=" << serial_ms
                  << " paired_complete_ms=" << paired_ms
                  << " speedup=" << serial_ms / paired_ms
                  << " weight_pair_bytes=" << pair_weight_bytes
                  << " disjoint_weight_copies=" << kCopies
                  << " intervening_weight_bytes=" << 2U * pair_weight_bytes
                  << " serial_copy_medians_ms=" << serial_copy_medians[0] << ','
                  << serial_copy_medians[1] << ',' << serial_copy_medians[2]
                  << " paired_copy_medians_ms=" << paired_copy_medians[0] << ','
                  << paired_copy_medians[1] << ',' << paired_copy_medians[2]
                  << " vgpr=" << resources.registers
                  << " lds=" << resources.static_shared_bytes
                  << " scratch=" << resources.local_bytes << '\n';
        return paired_ms <= serial_ms * 0.99 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "A8Q4 GDN paired T1 qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
