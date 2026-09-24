#define NINFER_A8Q4_QUAL_NO_MAIN
#include "a8q4_shape_sweep_qual.hip"

#if defined(NINFER_ATTENTION_PROJECTION_PRODUCTION_QUAL)
#include "ops/r9700/attention_projection/attention_projection.h"
#endif

#include <functional>
#include <regex>

namespace {

constexpr std::uint32_t kColumns = 5120U;
constexpr std::uint32_t kRows = 7168U;
constexpr std::uint32_t kQueryRows = 6144U;
constexpr std::uint32_t kKeyRows = 1024U;
constexpr std::uint32_t kGroups = kColumns / kGroup;
constexpr std::size_t kCopies = 3U;
constexpr std::size_t kPairTrials = 12U;
constexpr std::size_t kSamplesPerCopy = kPairTrials / kCopies;
constexpr std::uint32_t kAttentionLayers = 16U;
constexpr double kBaseDecodeMs = 1000.0 / 27.0281083;
constexpr std::array<bool, kPairTrials> kSerialFirst{
    true, true, false, true, false, true, false, false, true, false, true, false};

struct DeviceWeights {
    std::unique_ptr<DeviceBuffer<std::uint8_t>> codes0;
    std::unique_ptr<DeviceBuffer<std::uint16_t>> scales0;
    std::unique_ptr<DeviceBuffer<std::uint8_t>> codes1;
    std::unique_ptr<DeviceBuffer<std::uint16_t>> scales1;
};

struct Outputs {
    DeviceBuffer<hip_bfloat16> query{kQueryRows + 2U};
    DeviceBuffer<hip_bfloat16> key{kKeyRows + 2U};
    DeviceBuffer<hip_bfloat16> gate{kQueryRows + 2U};
    DeviceBuffer<hip_bfloat16> value{kKeyRows + 2U};
};

struct PairArgs {
    const hip_bfloat16* input = nullptr;
    const std::uint8_t* weight0_codes = nullptr;
    std::size_t weight0_code_bytes = 0U;
    const std::uint16_t* weight0_scales = nullptr;
    std::size_t weight0_scale_bytes = 0U;
    const std::uint8_t* weight1_codes = nullptr;
    std::size_t weight1_code_bytes = 0U;
    const std::uint16_t* weight1_scales = nullptr;
    std::size_t weight1_scale_bytes = 0U;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0U;
    hip_bfloat16* query = nullptr;
    hip_bfloat16* key = nullptr;
    hip_bfloat16* gate = nullptr;
    hip_bfloat16* value = nullptr;
    std::uint32_t tokens = 1U;
    std::uint32_t columns = kColumns;
};

constexpr std::size_t q4_pair_index(std::uint32_t row, std::uint32_t group,
                                    std::uint32_t pair) {
    return (((static_cast<std::size_t>(row >> 4U) * kGroups + group) * 4U + pair) *
            16U + (row & 15U));
}

constexpr std::size_t q4_scale_index(std::uint32_t row, std::uint32_t group) {
    return ((static_cast<std::size_t>(row >> 4U) * kGroups + group) * 16U +
            (row & 15U));
}

int represented_weight_code(const DecodeDot8Weights& weights, std::uint32_t row,
                            std::uint32_t group, std::uint32_t lane) {
    const std::size_t word = q4_pair_index(row, group, lane >> 4U);
    const std::size_t byte = word * sizeof(std::uint64_t) + ((lane & 15U) >> 1U);
    const unsigned shift = (lane & 1U) * 4U;
    const unsigned code = (weights.codes[byte] >> shift) & 15U;
    return code >= 8U ? static_cast<int>(code) - 16 : static_cast<int>(code);
}

__device__ __forceinline__ int load_q4x8_local(const std::uint8_t* codes,
                                                std::size_t element) {
    return *reinterpret_cast<const int*>(codes + element / 2U);
}

__device__ __forceinline__ int load_weight_x8_local(
    const std::uint8_t* codes, std::uint32_t row, std::uint32_t group,
    std::uint32_t word) {
    const std::uint64_t pair =
        reinterpret_cast<const std::uint64_t*>(codes)[q4_pair_index(row, group, word >> 1U)];
    return static_cast<int>((word & 1U) != 0U ? pair >> 32U : pair);
}

__device__ hip_bfloat16 poison_bf16() {
    hip_bfloat16 value;
    value.data = 0x7fc1U;
    return value;
}

__global__ __launch_bounds__(256) void a8q4g64_attention_pair_t1_direct_kernel(
    const std::uint8_t* activation_low, const std::uint8_t* activation_high,
    const std::uint16_t* activation_scales, const std::uint32_t* activation_status,
    const std::uint8_t* weight0_codes, const std::uint16_t* weight0_scales,
    const std::uint8_t* weight1_codes, const std::uint16_t* weight1_scales,
    hip_bfloat16* query, hip_bfloat16* key, hip_bfloat16* gate, hip_bfloat16* value) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const std::uint32_t combined_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (combined_row >= 2U * kRows) return;
    const bool second = combined_row >= kRows;
    const std::uint32_t row = second ? combined_row - kRows : combined_row;
    hip_bfloat16* output = nullptr;
    std::uint32_t output_row = row;
    if (!second) {
        if (row < kQueryRows) output = query;
        else { output = key; output_row -= kQueryRows; }
    } else {
        if (row < kQueryRows) output = gate;
        else { output = value; output_row -= kQueryRows; }
    }
    if (*activation_status != 0U) {
        output[output_row] = poison_bf16();
        return;
    }
    const std::uint8_t* codes = second ? weight1_codes : weight0_codes;
    const std::uint16_t* scales = second ? weight1_scales : weight0_scales;
    float total = 0.0F;
    for (std::uint32_t group = 0U; group < kGroups; ++group) {
        int low_dot = 0;
        int high_dot = 0;
#pragma unroll
        for (std::uint32_t word = 0U; word < 8U; ++word) {
            const std::size_t element = static_cast<std::size_t>(group) * kGroup + word * 8U;
            const int low = load_q4x8_local(activation_low, element);
            const int high = load_q4x8_local(activation_high, element);
            const int weight = load_weight_x8_local(codes, row, group, word);
            low_dot = __builtin_amdgcn_sudot8(false, low, true, weight, low_dot, false);
            high_dot = __builtin_amdgcn_sudot8(true, high, true, weight, high_dot, false);
        }
        const int combined = low_dot + 16 * high_dot;
        const float activation_scale = __half2float(__ushort_as_half(activation_scales[group]));
        const float weight_scale = __half2float(__ushort_as_half(scales[q4_scale_index(row, group)]));
        total = fmaf(static_cast<float>(combined), activation_scale * weight_scale, total);
    }
    output[output_row] = static_cast<hip_bfloat16>(total);
#else
    (void)activation_low; (void)activation_high; (void)activation_scales;
    (void)activation_status; (void)weight0_codes; (void)weight0_scales;
    (void)weight1_codes; (void)weight1_scales; (void)query; (void)key;
    (void)gate; (void)value;
#endif
}

__global__ void attention_extract_kernel(const hip_bfloat16* source,
                                         hip_bfloat16* destination,
                                         std::uint32_t source_row,
                                         std::uint32_t rows) {
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) destination[row] = source[source_row + row];
}

bool overlap(const void* a, std::size_t as, const void* b, std::size_t bs) {
    if (a == nullptr || b == nullptr || as == 0U || bs == 0U) return false;
    const auto ab = reinterpret_cast<std::uintptr_t>(a);
    const auto bb = reinterpret_cast<std::uintptr_t>(b);
    return ab < bb + bs && bb < ab + as;
}

hipError_t launch_pair(const PairArgs& args, hipStream_t stream = nullptr) {
#if defined(NINFER_ATTENTION_PROJECTION_PRODUCTION_QUAL)
    return ninfer::ops::r9700::attention_projection::launch(
        {.hidden = args.input,
         .query_key_codes = args.weight0_codes,
         .query_key_code_bytes = args.weight0_code_bytes,
         .query_key_scales = args.weight0_scales,
         .query_key_scale_bytes = args.weight0_scale_bytes,
         .gate_value_codes = args.weight1_codes,
         .gate_value_code_bytes = args.weight1_code_bytes,
         .gate_value_scales = args.weight1_scales,
         .gate_value_scale_bytes = args.weight1_scale_bytes,
         .activation_workspace = args.workspace,
         .activation_workspace_bytes = args.workspace_bytes,
         .query = args.query,
         .key = args.key,
         .gate = args.gate,
         .value = args.value,
         .tokens = args.tokens,
         .columns = args.columns}, stream);
#else
    const std::size_t code_bytes = static_cast<std::size_t>(kRows) * kColumns / 2U;
    const std::size_t scale_bytes = static_cast<std::size_t>(kRows) * kGroups * 2U;
    const std::size_t workspace_bytes =
        linear::a8q4g64_activation_workspace_capacity_bytes(1U, kColumns);
    if (args.input == nullptr || args.weight0_codes == nullptr || args.weight0_scales == nullptr ||
        args.weight1_codes == nullptr || args.weight1_scales == nullptr ||
        args.workspace == nullptr || args.query == nullptr || args.key == nullptr ||
        args.gate == nullptr || args.value == nullptr || args.tokens != 1U ||
        args.columns != kColumns || args.weight0_code_bytes != code_bytes ||
        args.weight1_code_bytes != code_bytes || args.weight0_scale_bytes != scale_bytes ||
        args.weight1_scale_bytes != scale_bytes || args.workspace_bytes != workspace_bytes ||
        (reinterpret_cast<std::uintptr_t>(args.weight0_codes) & 7U) != 0U ||
        (reinterpret_cast<std::uintptr_t>(args.weight1_codes) & 7U) != 0U) return hipErrorInvalidValue;
    const std::array<std::pair<const void*, std::size_t>, 9> planes{{
        {args.input, kColumns * sizeof(hip_bfloat16)},
        {args.weight0_codes, code_bytes}, {args.weight0_scales, scale_bytes},
        {args.weight1_codes, code_bytes}, {args.weight1_scales, scale_bytes},
        {args.query, kQueryRows * sizeof(hip_bfloat16)},
        {args.key, kKeyRows * sizeof(hip_bfloat16)},
        {args.gate, kQueryRows * sizeof(hip_bfloat16)},
        {args.value, kKeyRows * sizeof(hip_bfloat16)}}};
    for (std::size_t i = 0; i < planes.size(); ++i) {
        if (overlap(args.workspace, workspace_bytes, planes[i].first, planes[i].second))
            return hipErrorInvalidValue;
        for (std::size_t j = i + 1; j < planes.size(); ++j) {
            if (overlap(planes[i].first, planes[i].second, planes[j].first, planes[j].second))
                return hipErrorInvalidValue;
        }
    }
    linear::A8G64ActivationWorkspace workspace{};
    if (linear::a8q4g64_bind_activation_workspace(args.workspace, args.workspace_bytes,
                                                   1U, kColumns, &workspace) != hipSuccess)
        return hipErrorInvalidValue;
    hipError_t status = linear::a8g64_quantize_activation({args.input, workspace}, stream);
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(a8q4g64_attention_pair_t1_direct_kernel,
                       dim3((2U * kRows + 255U) / 256U), dim3(256U), 0, stream,
                       workspace.low_codes, workspace.high_codes, workspace.scales,
                       workspace.status, args.weight0_codes, args.weight0_scales,
                       args.weight1_codes, args.weight1_scales, args.query, args.key,
                       args.gate, args.value);
    return hipGetLastError();
#endif
}

linear::A8Q4G64CandidateArgs serial_args(const hip_bfloat16* input,
                                         const std::uint8_t* codes, std::size_t code_bytes,
                                         const std::uint16_t* scales, std::size_t scale_bytes,
                                         void* workspace, std::size_t workspace_bytes,
                                         hip_bfloat16* output) {
    return {input, codes, code_bytes, scales, scale_bytes, workspace, workspace_bytes,
            output, 1U, kRows, kColumns, kColumns};
}

hipError_t extract(const hip_bfloat16* source, hip_bfloat16* destination,
                   std::uint32_t source_row, std::uint32_t rows,
                   hipStream_t stream = nullptr) {
    hipLaunchKernelGGL(attention_extract_kernel, dim3((rows + 255U) / 256U),
                       dim3(256U), 0, stream, source, destination, source_row, rows);
    return hipGetLastError();
}

hipError_t launch_serial(const hip_bfloat16* input, const DeviceWeights& weights,
                         void* workspace, std::size_t workspace_bytes,
                         hip_bfloat16* temp0, hip_bfloat16* temp1, Outputs& outputs) {
    hipError_t status = linear::a8q4g64_linear_candidate(
        serial_args(input, weights.codes0->get(), weights.codes0->bytes(),
                    weights.scales0->get(), weights.scales0->bytes(), workspace,
                    workspace_bytes, temp0), nullptr);
    if (status != hipSuccess) return status;
    status = extract(temp0, outputs.query.get() + 1U, 0U, kQueryRows);
    if (status != hipSuccess) return status;
    status = extract(temp0, outputs.key.get() + 1U, kQueryRows, kKeyRows);
    if (status != hipSuccess) return status;
    status = linear::a8q4g64_linear_candidate(
        serial_args(input, weights.codes1->get(), weights.codes1->bytes(),
                    weights.scales1->get(), weights.scales1->bytes(), workspace,
                    workspace_bytes, temp1), nullptr);
    if (status != hipSuccess) return status;
    status = extract(temp1, outputs.gate.get() + 1U, 0U, kQueryRows);
    if (status != hipSuccess) return status;
    return extract(temp1, outputs.value.get() + 1U, kQueryRows, kKeyRows);
}

PairArgs pair_args(const hip_bfloat16* input, const DeviceWeights& weights,
                   void* workspace, std::size_t workspace_bytes, Outputs& outputs) {
    return {input, weights.codes0->get(), weights.codes0->bytes(), weights.scales0->get(),
            weights.scales0->bytes(), weights.codes1->get(), weights.codes1->bytes(),
            weights.scales1->get(), weights.scales1->bytes(), workspace, workspace_bytes,
            outputs.query.get() + 1U, outputs.key.get() + 1U, outputs.gate.get() + 1U,
            outputs.value.get() + 1U, 1U, kColumns};
}

void fill_guards(Outputs& outputs, std::uint16_t bits) {
    for (auto [buffer, rows] : std::array<std::pair<hip_bfloat16*, std::size_t>, 4>{{
             {outputs.query.get(), kQueryRows}, {outputs.key.get(), kKeyRows},
             {outputs.gate.get(), kQueryRows}, {outputs.value.get(), kKeyRows}}}) {
        std::vector<hip_bfloat16> host(rows + 2U);
        for (auto& value : host) value.data = bits;
        HIP_CHECK(hipMemcpy(buffer, host.data(), (rows + 2U) * sizeof(hip_bfloat16),
                            hipMemcpyHostToDevice));
    }
}

std::array<std::vector<hip_bfloat16>, 4> download(const Outputs& outputs,
                                                   std::uint16_t guard) {
    std::array<std::vector<hip_bfloat16>, 4> result{
        std::vector<hip_bfloat16>(kQueryRows), std::vector<hip_bfloat16>(kKeyRows),
        std::vector<hip_bfloat16>(kQueryRows), std::vector<hip_bfloat16>(kKeyRows)};
    const std::array<std::pair<const hip_bfloat16*, std::size_t>, 4> source{{
        {outputs.query.get(), kQueryRows}, {outputs.key.get(), kKeyRows},
        {outputs.gate.get(), kQueryRows}, {outputs.value.get(), kKeyRows}}};
    for (std::size_t i = 0U; i < source.size(); ++i) {
        std::vector<hip_bfloat16> host(source[i].second + 2U);
        HIP_CHECK(hipMemcpy(host.data(), source[i].first, host.size() * sizeof(hip_bfloat16),
                            hipMemcpyDeviceToHost));
        if (host.front().data != guard || host.back().data != guard)
            fail("attention paired projection changed an output canary");
        std::copy(host.begin() + 1, host.end() - 1, result[i].begin());
    }
    return result;
}

std::uint64_t oracle_steps(const std::array<std::vector<hip_bfloat16>, 4>& actual,
                           const HostActivation& activation,
                           const DecodeDot8Weights& weights0,
                           const DecodeDot8Weights& weights1) {
    std::uint64_t maximum = 0U;
    for (std::uint32_t matrix = 0U; matrix < 2U; ++matrix) {
        const DecodeDot8Weights& weights = matrix == 0U ? weights0 : weights1;
        for (std::uint32_t row = 0U; row < kRows; ++row) {
            double sum = 0.0;
            for (std::uint32_t group = 0U; group < kGroups; ++group) {
                std::int32_t dot = 0;
                for (std::uint32_t lane = 0U; lane < kGroup; ++lane) {
                    const std::size_t index = static_cast<std::size_t>(group) * kGroup + lane;
                    const int a = unsigned_nibble(activation.low, index) +
                                  16 * signed_nibble(activation.high, index);
                    dot += a * represented_weight_code(weights, row, group, lane);
                }
                sum = std::fma(static_cast<double>(dot),
                               static_cast<double>(half_value(activation.scales[group])) *
                                   half_value(weights.scales[q4_scale_index(row, group)]), sum);
            }
            const std::uint32_t part = matrix * 2U + (row >= kQueryRows ? 1U : 0U);
            const std::uint32_t part_row = row < kQueryRows ? row : row - kQueryRows;
            const std::uint64_t steps = bf16_steps(actual[part][part_row],
                                                    hip_bfloat16(static_cast<float>(sum)));
            maximum = std::max(maximum, steps);
            if (steps > 2U) fail("attention paired output differs from FP64 represented oracle");
        }
    }
    return maximum;
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

template <std::size_t N> double median(std::array<double, N> values) {
    std::sort(values.begin(), values.end());
    return 0.5 * (values[N / 2U - 1U] + values[N / 2U]);
}

void require_invalid(PairArgs args, const char* label) {
    if (launch_pair(args) != hipErrorInvalidValue)
        fail(std::string("attention pair accepted ") + label);
}

} // namespace

int main(int argc, char** argv) {
    try {
#if defined(NINFER_ATTENTION_PROJECTION_PRODUCTION_QUAL)
        if (argc != 7 || std::string_view(argv[1]) != "--out-json" ||
            std::string_view(argv[3]) != "--assembly" ||
            std::string_view(argv[5]) != "--static-receipt") {
            std::cerr << "usage: attention_projection_t1_op_qual --out-json NEW.json "
                         "--assembly FILE.s --static-receipt RECEIPT.txt\n";
            return 2;
        }
#else
        if (argc != 5 || std::string_view(argv[1]) != "--out-json" ||
            std::string_view(argv[3]) != "--assembly") {
            std::cerr << "usage: a8q4_attention_pair_t1_qual --out-json NEW.json "
                         "--assembly FILE.s\n";
            return 2;
        }
#endif
        const std::filesystem::path output_path = argv[2];
        const std::filesystem::path assembly = std::filesystem::canonical(argv[4]);
#if defined(NINFER_ATTENTION_PROJECTION_PRODUCTION_QUAL)
        const std::filesystem::path static_receipt = std::filesystem::canonical(argv[6]);
#endif
        require_fresh_output(output_path);
        const std::filesystem::path root = std::filesystem::canonical(NINFER_SOURCE_DIR);
        const std::filesystem::path executable = std::filesystem::canonical("/proc/self/exe");
#if defined(NINFER_ATTENTION_PROJECTION_PRODUCTION_QUAL)
        const std::filesystem::path checker = std::filesystem::canonical(
            root / "tools/r9700/check_attention_projection_t1_op_static.py");
#else
        const std::filesystem::path checker = std::filesystem::canonical(
            root / "tools/r9700/check_a8q4_attention_pair_t1_static.py");
#endif
        const std::string static_command = "python3 " + shell_quote(checker.string()) + " " +
                                           shell_quote(assembly.string())
#if defined(NINFER_ATTENTION_PROJECTION_PRODUCTION_QUAL)
                                           + " --expect-receipt " +
                                           shell_quote(static_receipt.string())
#endif
                                           ;
        if (std::system(static_command.c_str()) != 0) fail("attention pair static gate failed");
#if defined(NINFER_ATTENTION_PROJECTION_PRODUCTION_QUAL)
        const std::string static_text = read_text(static_receipt);
        std::smatch static_match;
        if (!std::regex_match(static_text, static_match, std::regex(
                "attention_projection_t1_op_static: PASS .*vgpr=([0-9]+) sgpr=([0-9]+) "
                "wave32=true lds=0 scratch=0 spills=0 native_mixed_iu4_dot8=true"))) {
            fail("production attention projection static receipt differs");
        }
#endif

        HIP_CHECK(hipSetDevice(0));
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, 0));
        if (std::string_view(properties.name) != "AMD Radeon AI PRO R9700" ||
            std::string_view(properties.gcnArchName) != "gfx1201" || properties.warpSize != 32)
            fail("attention pair qualification requires R9700 gfx1201 wave32");
        char pci_bus_id[32]{};
        HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
        std::string pci = pci_bus_id;
        if (pci.size() == 7U) pci = "0000:" + pci;
        std::transform(pci.begin(), pci.end(), pci.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        const std::filesystem::path power = std::filesystem::path("/sys/bus/pci/devices") /
                                             pci / "power_dpm_force_performance_level";
        if (read_text(power) != "auto") fail("attention pair qualification requires auto power");

        const auto input = make_decode_dot8_input(kColumns);
        const auto activation = quantize_host(input, 1U, kColumns);
        const auto host0 = make_decode_dot8_weights(kRows, kColumns);
        auto host1 = make_decode_dot8_weights(kRows, kColumns);
        for (std::size_t index = 0U; index < host1.codes.size(); ++index) {
            const std::uint8_t mask = (index & 1U) != 0U ? 0x31U : 0x8cU;
            host1.codes[index] ^= mask;
        }
        DeviceBuffer<hip_bfloat16> d_input(input.size());
        HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));
        std::array<DeviceWeights, kCopies> copies;
        for (auto& copy : copies) {
            copy.codes0 = std::make_unique<DeviceBuffer<std::uint8_t>>(host0.codes.size());
            copy.scales0 = std::make_unique<DeviceBuffer<std::uint16_t>>(host0.scales.size());
            copy.codes1 = std::make_unique<DeviceBuffer<std::uint8_t>>(host1.codes.size());
            copy.scales1 = std::make_unique<DeviceBuffer<std::uint16_t>>(host1.scales.size());
            HIP_CHECK(hipMemcpy(copy.codes0->get(), host0.codes.data(), copy.codes0->bytes(), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(copy.scales0->get(), host0.scales.data(), copy.scales0->bytes(), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(copy.codes1->get(), host1.codes.data(), copy.codes1->bytes(), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(copy.scales1->get(), host1.scales.data(), copy.scales1->bytes(), hipMemcpyHostToDevice));
        }
        const std::size_t workspace_bytes = linear::a8q4g64_activation_workspace_capacity_bytes(1U, kColumns);
        DeviceBuffer<std::uint8_t> workspace(workspace_bytes);
        DeviceBuffer<hip_bfloat16> temp0(kRows), temp1(kRows);
        Outputs serial_outputs;
        Outputs paired_outputs;
        auto valid = pair_args(d_input.get(), copies[0], workspace.get(), workspace_bytes, paired_outputs);
        auto malformed = valid; malformed.tokens = 2U; require_invalid(malformed, "T2");
        malformed = valid; malformed.columns = 4992U; require_invalid(malformed, "K4992");
        malformed = valid; --malformed.weight1_code_bytes; require_invalid(malformed, "short code plane");
        malformed = valid; ++malformed.weight0_codes; require_invalid(malformed, "misaligned weights");
        malformed = valid; malformed.key = malformed.query; require_invalid(malformed, "aliased outputs");
        malformed = valid; malformed.workspace = malformed.query; require_invalid(malformed, "workspace/output alias");
        malformed = valid; malformed.input = reinterpret_cast<const hip_bfloat16*>(malformed.weight0_codes);
        require_invalid(malformed, "input/weight alias");

        fill_guards(serial_outputs, 0x7fc2U);
        fill_guards(paired_outputs, 0x7fc3U);
        HIP_CHECK(launch_serial(d_input.get(), copies[0], workspace.get(), workspace_bytes,
                                temp0.get(), temp1.get(), serial_outputs));
        HIP_CHECK(launch_pair(valid));
        HIP_CHECK(hipDeviceSynchronize());
        const auto serial_host = download(serial_outputs, 0x7fc2U);
        const auto paired_host = download(paired_outputs, 0x7fc3U);
        for (std::size_t part = 0U; part < 4U; ++part) {
            if (!std::equal(serial_host[part].begin(), serial_host[part].end(),
                            paired_host[part].begin(), [](hip_bfloat16 a, hip_bfloat16 b) {
                                return a.data == b.data;
                            })) fail("attention pair is not bit exact to serial plus extracts");
        }
        const std::uint64_t max_steps = oracle_steps(paired_host, activation, host0, host1);

        hip_bfloat16 nonfinite{}; nonfinite.data = 0x7fc1U;
        HIP_CHECK(hipMemcpy(d_input.get(), &nonfinite, sizeof(nonfinite), hipMemcpyHostToDevice));
        fill_guards(paired_outputs, 0x7fc4U);
        HIP_CHECK(launch_pair(pair_args(d_input.get(), copies[0], workspace.get(), workspace_bytes,
                                        paired_outputs)));
        HIP_CHECK(hipDeviceSynchronize());
        const auto poisoned = download(paired_outputs, 0x7fc4U);
        for (const auto& part : poisoned)
            if (!std::all_of(part.begin(), part.end(), [](hip_bfloat16 v) { return v.data == 0x7fc1U; }))
                fail("attention pair status did not poison all direct outputs");
        HIP_CHECK(hipMemcpy(d_input.get(), input.data(), d_input.bytes(), hipMemcpyHostToDevice));

        for (std::size_t launch = 0U; launch < 6U; ++launch) {
            const auto& w = copies[launch % kCopies];
            if ((launch & 1U) == 0U)
                HIP_CHECK(launch_serial(d_input.get(), w, workspace.get(), workspace_bytes,
                                        temp0.get(), temp1.get(), serial_outputs));
            else HIP_CHECK(launch_pair(pair_args(d_input.get(), w, workspace.get(), workspace_bytes,
                                                 paired_outputs)));
        }
        HIP_CHECK(hipDeviceSynchronize());
        std::array<double, kPairTrials> serial_samples{}, paired_samples{};
        std::array<std::array<double, kSamplesPerCopy>, kCopies> serial_by_copy{}, paired_by_copy{};
        std::array<std::size_t, kCopies> serial_counts{}, paired_counts{};
        for (std::size_t trial = 0U; trial < kPairTrials; ++trial) {
            const std::size_t first = (2U * trial) % kCopies;
            const std::size_t second = (first + 1U) % kCopies;
            const std::size_t si = kSerialFirst[trial] ? first : second;
            const std::size_t pi = kSerialFirst[trial] ? second : first;
            const auto serial = [&] { return launch_serial(d_input.get(), copies[si], workspace.get(),
                workspace_bytes, temp0.get(), temp1.get(), serial_outputs); };
            const auto paired = [&] { return launch_pair(pair_args(d_input.get(), copies[pi],
                workspace.get(), workspace_bytes, paired_outputs)); };
            if (kSerialFirst[trial]) { serial_samples[trial] = time_once(serial); paired_samples[trial] = time_once(paired); }
            else { paired_samples[trial] = time_once(paired); serial_samples[trial] = time_once(serial); }
            serial_by_copy[si][serial_counts[si]++] = serial_samples[trial];
            paired_by_copy[pi][paired_counts[pi]++] = paired_samples[trial];
        }
        std::array<double, kCopies> serial_copy{}, paired_copy{};
        for (std::size_t i = 0U; i < kCopies; ++i) {
            if (serial_counts[i] != kSamplesPerCopy || paired_counts[i] != kSamplesPerCopy)
                fail("attention pair schedule is not allocation balanced");
            serial_copy[i] = median(serial_by_copy[i]);
            paired_copy[i] = median(paired_by_copy[i]);
            if (paired_copy[i] > serial_copy[i] * 0.99)
                fail("attention pair did not clear direct gate on every allocation");
        }
        const double serial_ms = median(serial_samples);
        const double paired_ms = median(paired_samples);
        const double saved_per_layer_ms = serial_ms - paired_ms;
        const double saved_per_token_ms = saved_per_layer_ms * kAttentionLayers;
        const double whole_percent = 100.0 * saved_per_token_ms / kBaseDecodeMs;
        if (whole_percent < 1.0) fail("attention pair cannot credibly clear whole-decode admission");
        if (read_text(power) != "auto") fail("power changed during attention pair qualification");

        const std::size_t pair_weight_bytes = copies[0].codes0->bytes() + copies[0].scales0->bytes() +
                                              copies[0].codes1->bytes() + copies[0].scales1->bytes();
        int runtime_version = 0;
        int driver_version = 0;
        HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
        HIP_CHECK(hipDriverGetVersion(&driver_version));
        std::ostringstream report;
#if defined(NINFER_ATTENTION_PROJECTION_PRODUCTION_QUAL)
        const std::array<std::pair<const char*, const char*>, 8> source_specs{{
            {"qualifier_entry", "tools/r9700/attention_projection_t1_op_qual.hip"},
            {"qualifier_body", "tools/r9700/a8q4_attention_pair_t1_qual.hip"},
            {"shared_qualifier", "tools/r9700/a8q4_shape_sweep_qual.hip"},
            {"op_contract", "include/ninfer/ops/attention_projection.h"},
            {"raw_contract", "src/ops/r9700/attention_projection/attention_projection.h"},
            {"op_implementation", "src/ops/r9700/attention_projection/attention_projection.hip"},
            {"static_checker", "tools/r9700/check_attention_projection_t1_op_static.py"},
            {"linear_primitive", "src/ops/r9700/linear/r9700_linear.hip"},
        }};
#else
        const std::array<std::pair<const char*, const char*>, 4> source_specs{{
            {"qualifier", "tools/r9700/a8q4_attention_pair_t1_qual.hip"},
            {"shared_qualifier", "tools/r9700/a8q4_shape_sweep_qual.hip"},
            {"linear_contract", "src/ops/r9700/linear/r9700_linear.h"},
            {"linear_implementation", "src/ops/r9700/linear/r9700_linear.hip"},
        }};
#endif
        report << std::setprecision(17)
#if defined(NINFER_ATTENTION_PROJECTION_PRODUCTION_QUAL)
               << "{\n  \"schema\": \"ninfer.r9700.attention-projection-t1-production-qualification.v1\",\n"
#else
               << "{\n  \"schema\": \"ninfer.r9700.a8q4-attention-pair-t1-qualification.v1\",\n"
#endif
               << "  \"status\": \"qualified_for_whole_inference_ab\",\n"
               << "  \"command\": \"";
        for (int index = 0; index < argc; ++index) {
            if (index != 0) report << ' ';
            report << json_escape(argv[index]);
        }
        report << "\",\n"
               << "  \"route\": \"two_q4_n7168_k5120_to_q6144_k1024_gate6144_v1024\",\n"
               << "  \"executable\": {\"path\": \"" << json_escape(executable.string())
               << "\", \"bytes\": " << std::filesystem::file_size(executable)
               << ", \"sha256\": \"" << sha256_file(executable) << "\"},\n"
               << "  \"sources\": [\n";
        for (std::size_t i = 0U; i < source_specs.size(); ++i) {
            const auto path = std::filesystem::canonical(root / source_specs[i].second);
            report << "    {\"role\": \"" << source_specs[i].first << "\", \"path\": \""
                   << json_escape(path.string()) << "\", \"bytes\": "
                   << std::filesystem::file_size(path) << ", \"sha256\": \""
                   << sha256_file(path) << "\"}" << (i + 1U == source_specs.size() ? "\n" : ",\n");
        }
        report << "  ],\n"
               << "  \"device\": {\"index\": 0, \"name\": \"" << json_escape(properties.name)
               << "\", \"architecture\": \"" << json_escape(properties.gcnArchName)
               << "\", \"pci_bus_id\": \"" << json_escape(pci)
               << "\", \"wave_size\": " << properties.warpSize
               << ", \"hip_runtime_version\": " << runtime_version
               << ", \"hip_driver_version\": " << driver_version << "},\n"
               << "  \"power_profile\": {\"path\": \"" << json_escape(power.string())
               << "\", \"required\": \"auto\", \"before\": \"auto\", \"after\": \"auto\"},\n"
               << "  \"static\": {\"status\": \"passed\", \"assembly_sha256\": \""
               << sha256_file(assembly) << "\", \"checker_sha256\": \"" << sha256_file(checker)
               << "\", \"native_iu4_dot8\": true, \"wave32\": true, "
#if defined(NINFER_ATTENTION_PROJECTION_PRODUCTION_QUAL)
               << "\"vgpr\": " << static_match[1].str()
               << ", \"sgpr\": " << static_match[2].str()
               << ", \"receipt_sha256\": \"" << sha256_file(static_receipt)
               << "\", \"receipt\": \"" << json_escape(static_text) << "\", "
#endif
               << "\"lds_bytes\": 0, \"scratch_bytes\": 0, \"spill_count\": 0},\n"
               << "  \"correctness\": {\"complete_outputs_bit_exact_to_serial_plus_four_extracts\": true, "
                  "\"independent_fp64_represented_weight_oracle\": true, \"maximum_bf16_steps\": "
               << max_steps << ", \"canaries\": true, \"status_poisoning\": true, "
                  "\"malformed_extent_alignment_alias_rejection\": true},\n"
               << "  \"timing\": {\"scope\": \"two complete production linears plus four extracts versus "
                  "quantize-once combined direct scatter\", \"trials_per_arm\": " << kPairTrials
               << ", \"disjoint_weight_copies\": " << kCopies << ", \"weight_pair_bytes\": "
               << pair_weight_bytes << ", \"intervening_weight_bytes\": " << 2U * pair_weight_bytes
               << ", \"arm_order_balanced\": true, \"allocation_balanced\": true,\n"
               << "    \"serial_samples_ms\": [";
        for (std::size_t i = 0; i < kPairTrials; ++i) report << (i ? "," : "") << serial_samples[i];
        report << "],\n    \"paired_samples_ms\": [";
        for (std::size_t i = 0; i < kPairTrials; ++i) report << (i ? "," : "") << paired_samples[i];
        report << "],\n    \"serial_copy_medians_ms\": [";
        for (std::size_t i = 0; i < kCopies; ++i) report << (i ? "," : "") << serial_copy[i];
        report << "], \"paired_copy_medians_ms\": [";
        for (std::size_t i = 0; i < kCopies; ++i) report << (i ? "," : "") << paired_copy[i];
        report << "], \"serial_median_ms\": " << serial_ms << ", \"paired_median_ms\": "
               << paired_ms << ", \"speedup\": " << serial_ms / paired_ms << "},\n"
               << "  \"whole_bound\": {\"all_q4_attention_layers\": " << kAttentionLayers
               << ", \"base_decode_ms_per_token\": " << kBaseDecodeMs
               << ", \"saved_ms_per_token\": " << saved_per_token_ms
               << ", \"credible_whole_percent\": " << whole_percent
               << ", \"admission_minimum_percent\": 1.0, \"passed\": true},\n"
               << "  \"decision\": \"prepare_selector_off_on_whole_inference_ab\"\n}\n";
        const std::string payload = report.str();
        const int descriptor = ::open(output_path.c_str(),
                                      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (descriptor < 0) fail("cannot create attention pair report exclusively");
        FILE* output = fdopen(descriptor, "wb");
        if (output == nullptr) {
            (void)::close(descriptor);
            (void)::unlink(output_path.c_str());
            fail("cannot bind attention pair report stream");
        }
        const bool wrote = fwrite(payload.data(), 1U, payload.size(), output) == payload.size();
        const bool closed = fclose(output) == 0;
        if (!wrote || !closed) {
            (void)::unlink(output_path.c_str());
            fail("cannot finish attention pair report");
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "attention-pair qualification=PASS serial_ms=" << serial_ms
                  << " paired_ms=" << paired_ms << " speedup=" << serial_ms / paired_ms
                  << " saved_ms_per_token=" << saved_per_token_ms
                  << " credible_whole_percent=" << whole_percent
                  << " fp64_max_bf16_steps=" << max_steps << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "A8Q4 attention pair T1 qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
