#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

void require_lt(hipblasStatus_t status, const char* operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + ": status " + std::to_string(static_cast<int>(status)));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
        require_hip(hipMalloc(reinterpret_cast<void**>(&data_), elements * sizeof(T)), "hipMalloc");
    }
    ~DeviceBuffer() {
        if (data_ != nullptr) (void)hipFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] T* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return elements_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t elements_ = 0;
};

class Event {
public:
    Event() { require_hip(hipEventCreate(&event_), "hipEventCreate"); }
    ~Event() {
        if (event_ != nullptr) (void)hipEventDestroy(event_);
    }
    [[nodiscard]] hipEvent_t get() const noexcept { return event_; }

private:
    hipEvent_t event_ = nullptr;
};

class LtContext {
public:
    LtContext() {
        constexpr std::uint64_t tokens = 2048;
        constexpr std::uint64_t rows = 34816;
        constexpr std::uint64_t columns = 5120;
        require_lt(hipblasLtCreate(&handle), "hipblasLtCreate");
        require_lt(hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                   "hipblasLtMatmulDescCreate");
        constexpr hipblasOperation_t transpose = HIPBLAS_OP_T;
        require_lt(hipblasLtMatmulDescSetAttribute(
                       operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)),
                   "set TRANSA");
        constexpr hipblasLtMatmulMatrixScale_t outer =
            HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        require_lt(hipblasLtMatmulDescSetAttribute(
                       operation, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &outer, sizeof(outer)),
                   "set A scale mode");
        require_lt(hipblasLtMatmulDescSetAttribute(
                       operation, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &outer, sizeof(outer)),
                   "set B scale mode");
        const void* a_scale = row_scales.get();
        const void* b_scale = token_scales.get();
        require_lt(hipblasLtMatmulDescSetAttribute(
                       operation, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                       &a_scale, sizeof(a_scale)),
                   "set A scale pointer");
        require_lt(hipblasLtMatmulDescSetAttribute(
                       operation, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                       &b_scale, sizeof(b_scale)),
                   "set B scale pointer");
        require_lt(hipblasLtMatrixLayoutCreate(
                       &weight, HIP_R_8F_E4M3, columns, rows, columns),
                   "create weight layout");
        require_lt(hipblasLtMatrixLayoutCreate(
                       &activation, HIP_R_8F_E4M3, columns, tokens, columns),
                   "create activation layout");
        require_lt(hipblasLtMatrixLayoutCreate(
                       &output, HIP_R_16BF, rows, tokens, rows),
                   "create output layout");
    }

    ~LtContext() {
        if (output != nullptr) (void)hipblasLtMatrixLayoutDestroy(output);
        if (activation != nullptr) (void)hipblasLtMatrixLayoutDestroy(activation);
        if (weight != nullptr) (void)hipblasLtMatrixLayoutDestroy(weight);
        if (operation != nullptr) (void)hipblasLtMatmulDescDestroy(operation);
        if (handle != nullptr) (void)hipblasLtDestroy(handle);
    }

    DeviceBuffer<float> row_scales{34816};
    DeviceBuffer<float> token_scales{2048};
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t operation = nullptr;
    hipblasLtMatrixLayout_t weight = nullptr;
    hipblasLtMatrixLayout_t activation = nullptr;
    hipblasLtMatrixLayout_t output = nullptr;
};

} // namespace

int main() try {
    require_hip(hipSetDevice(0), "hipSetDevice");
    LtContext context;
    std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
    require_lt(hipblaslt_ext::getAllAlgos(
                   context.handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                   HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_8F_E4M3, HIP_R_8F_E4M3,
                   HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F, algorithms),
               "getAllAlgos");
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    struct Supported {
        int index = -1;
        std::size_t workspace = 0;
        hipblasLtMatmulAlgo_t algorithm{};
    };
    std::vector<Supported> supported_algorithms;
    std::size_t maximum_workspace = 0;
    std::cout << "catalog=" << algorithms.size() << '\n';
    for (std::size_t rank = 0; rank < algorithms.size(); ++rank) {
        auto algorithm = algorithms[rank].algo;
        std::size_t workspace = 0;
        const hipblasStatus_t status = hipblaslt_ext::matmulIsAlgoSupported(
            context.handle, context.operation, &alpha, context.weight, context.activation,
            &beta, context.output, context.output, algorithm, workspace);
        if (status != HIPBLAS_STATUS_SUCCESS) continue;
        supported_algorithms.push_back(
            Supported{hipblaslt_ext::getIndexFromAlgo(algorithm), workspace, algorithm});
        maximum_workspace = std::max(maximum_workspace, workspace);
        std::cout << "supported catalog_rank=" << rank
                  << " index=" << supported_algorithms.back().index
                  << " workspace=" << workspace
                  << " kernel=" << hipblaslt_ext::getKernelNameFromAlgo(context.handle, algorithm)
                  << '\n';
    }
    std::cout << "supported=" << supported_algorithms.size()
              << " maximum_workspace=" << maximum_workspace << '\n';

    constexpr std::size_t tokens = 2048;
    constexpr std::size_t rows = 34816;
    constexpr std::size_t columns = 5120;
    DeviceBuffer<std::uint8_t> weight(rows * columns);
    DeviceBuffer<std::uint8_t> activation(tokens * columns);
    DeviceBuffer<std::uint16_t> output(tokens * rows);
    require_hip(hipMemset(weight.get(), 0x38, weight.bytes()), "initialize weight");
    require_hip(hipMemset(activation.get(), 0x38, activation.bytes()), "initialize activation");
    require_hip(hipMemset(output.get(), 0, output.bytes()), "initialize output");
    std::vector<float> ones_rows(rows, 1.0F);
    std::vector<float> ones_tokens(tokens, 1.0F);
    require_hip(hipMemcpy(context.row_scales.get(), ones_rows.data(),
                          rows * sizeof(float), hipMemcpyHostToDevice),
                "initialize row scales");
    require_hip(hipMemcpy(context.token_scales.get(), ones_tokens.data(),
                          tokens * sizeof(float), hipMemcpyHostToDevice),
                "initialize token scales");
    Event begin;
    Event end;
    constexpr std::array<int, 3> screen_indices{123104, 123096, 123097};
    for (const int index : screen_indices) {
        const auto found = std::find_if(
            supported_algorithms.begin(), supported_algorithms.end(),
            [index](const Supported& candidate) { return candidate.index == index; });
        if (found == supported_algorithms.end()) {
            std::cout << "screen index=" << index << " unsupported\n";
            continue;
        }
        for (int warmup = 0; warmup < 2; ++warmup) {
            require_lt(hipblasLtMatmul(
                           context.handle, context.operation, &alpha, weight.get(), context.weight,
                           activation.get(), context.activation, &beta, output.get(), context.output,
                           output.get(), context.output, &found->algorithm, nullptr, 0, nullptr),
                       "warmup hipblasLtMatmul");
        }
        require_hip(hipDeviceSynchronize(), "warmup synchronize");
        std::array<float, 7> samples{};
        for (float& sample : samples) {
            require_hip(hipEventRecord(begin.get()), "record begin");
            require_lt(hipblasLtMatmul(
                           context.handle, context.operation, &alpha, weight.get(), context.weight,
                           activation.get(), context.activation, &beta, output.get(), context.output,
                           output.get(), context.output, &found->algorithm, nullptr, 0, nullptr),
                       "timed hipblasLtMatmul");
            require_hip(hipEventRecord(end.get()), "record end");
            require_hip(hipEventSynchronize(end.get()), "synchronize end");
            require_hip(hipEventElapsedTime(&sample, begin.get(), end.get()), "elapsed time");
        }
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        std::cout << "screen index=" << index << " median_ms=" << sorted[3] << " samples=";
        for (const float sample : samples) std::cout << sample << ',';
        std::cout << '\n';
    }
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << "fp8_gate_up_catalog_qual: " << error.what() << '\n';
    return EXIT_FAILURE;
}
