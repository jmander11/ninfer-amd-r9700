// Bounded gfx1201 feasibility probe for a directly bound E4M3 prefill GEMM.
// This is not a product route: it measures installed hipBLASLt algorithms at one
// dominant Qwen3.8-27B projection shape before an artifact/runtime implementation.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

#define HIP_CHECK(expr) do { \
    const hipError_t error__ = (expr); \
    if (error__ != hipSuccess) throw std::runtime_error(hipGetErrorString(error__)); \
} while (false)

#define LT_CHECK(expr) do { \
    const hipblasStatus_t status__ = (expr); \
    if (status__ != HIPBLAS_STATUS_SUCCESS) { \
        std::ostringstream stream__; \
        stream__ << #expr << " failed with hipBLAS status " << static_cast<int>(status__); \
        throw std::runtime_error(stream__.str()); \
    } \
} while (false)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_ != nullptr) (void)hipFree(data_); }
    T* get() const noexcept { return data_; }
private:
    T* data_ = nullptr;
};

class Event {
public:
    Event() { HIP_CHECK(hipEventCreate(&event_)); }
    ~Event() { (void)hipEventDestroy(event_); }
    hipEvent_t get() const noexcept { return event_; }
private:
    hipEvent_t event_{};
};

float time_algorithm(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t operation,
                     const void* a, hipblasLtMatrixLayout_t a_layout,
                     const void* b, hipblasLtMatrixLayout_t b_layout,
                     void* output, hipblasLtMatrixLayout_t output_layout,
                     const hipblasLtMatmulAlgo_t& algorithm, void* workspace,
                     std::size_t workspace_bytes, hipStream_t stream) {
    constexpr int warmups = 0;
    constexpr int iterations = 1;
    const float alpha = 1.0F;
    const float beta = 0.0F;
    for (int index = 0; index < warmups; ++index) {
        LT_CHECK(hipblasLtMatmul(handle, operation, &alpha, a, a_layout, b, b_layout,
                                &beta, output, output_layout, output, output_layout,
                                &algorithm, workspace, workspace_bytes, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    Event begin;
    Event end;
    HIP_CHECK(hipEventRecord(begin.get(), stream));
    for (int index = 0; index < iterations; ++index) {
        LT_CHECK(hipblasLtMatmul(handle, operation, &alpha, a, a_layout, b, b_layout,
                                &beta, output, output_layout, output, output_layout,
                                &algorithm, workspace, workspace_bytes, stream));
    }
    HIP_CHECK(hipEventRecord(end.get(), stream));
    HIP_CHECK(hipEventSynchronize(end.get()));
    float milliseconds = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, begin.get(), end.get()));
    return milliseconds / iterations;
}

} // namespace

int main() try {
    constexpr std::int64_t m = 2048;
    constexpr std::int64_t n = 34816;
    constexpr std::int64_t k = 5120;
    constexpr std::size_t maximum_workspace = std::size_t{512} << 20;

    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    if (std::string_view(properties.gcnArchName).substr(0, 7) != "gfx1201" ||
        properties.warpSize != 32) {
        throw std::runtime_error("fp8 probe requires the gfx1201 wave32 target");
    }

    DeviceBuffer<std::uint8_t> a(static_cast<std::size_t>(m * k));
    DeviceBuffer<std::uint8_t> b(static_cast<std::size_t>(n * k));
    DeviceBuffer<std::uint16_t> output(static_cast<std::size_t>(m * n));
    DeviceBuffer<std::uint8_t> workspace(maximum_workspace);
    DeviceBuffer<float> a_scale(static_cast<std::size_t>(n));
    DeviceBuffer<float> b_scale(static_cast<std::size_t>(m));
    HIP_CHECK(hipMemset(a.get(), 0, static_cast<std::size_t>(m * k)));
    HIP_CHECK(hipMemset(b.get(), 0, static_cast<std::size_t>(n * k)));
    HIP_CHECK(hipMemset(output.get(), 0, static_cast<std::size_t>(m * n * 2)));
    const std::vector<float> host_a_scale(static_cast<std::size_t>(n), 1.0F);
    const std::vector<float> host_b_scale(static_cast<std::size_t>(m), 1.0F);
    HIP_CHECK(hipMemcpy(a_scale.get(), host_a_scale.data(),
                        host_a_scale.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_scale.get(), host_b_scale.data(),
                        host_b_scale.size() * sizeof(float), hipMemcpyHostToDevice));

    hipblasLtHandle_t handle{};
    hipblasLtMatmulDesc_t operation{};
    hipblasLtMatrixLayout_t a_layout{};
    hipblasLtMatrixLayout_t b_layout{};
    hipblasLtMatrixLayout_t output_layout{};
    hipblasLtMatmulPreference_t preference{};
    LT_CHECK(hipblasLtCreate(&handle));
    LT_CHECK(hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    // Native column-major mapping of row-major engine storage:
    //   weight[N,K] memory == A_col[K,N], activation[M,K] == B_col[K,M],
    //   output[M,N] memory == D_col[N,M].  Compute A^T * B -> D.
    const hipblasOperation_t trans_a = HIPBLAS_OP_T;
    LT_CHECK(hipblasLtMatmulDescSetAttribute(
        operation, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    const void* a_scale_pointer = a_scale.get();
    const void* b_scale_pointer = b_scale.get();
    LT_CHECK(hipblasLtMatmulDescSetAttribute(
        operation, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &a_scale_pointer, sizeof(a_scale_pointer)));
    LT_CHECK(hipblasLtMatmulDescSetAttribute(
        operation, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &b_scale_pointer, sizeof(b_scale_pointer)));
    const hipblasLtMatmulMatrixScale_t outer_scale =
        HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    LT_CHECK(hipblasLtMatmulDescSetAttribute(
        operation, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
        &outer_scale, sizeof(outer_scale)));
    LT_CHECK(hipblasLtMatmulDescSetAttribute(
        operation, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
        &outer_scale, sizeof(outer_scale)));

    LT_CHECK(hipblasLtMatrixLayoutCreate(&a_layout, HIP_R_8F_E4M3, k, n, k));
    LT_CHECK(hipblasLtMatrixLayoutCreate(&b_layout, HIP_R_8F_E4M3, k, m, k));
    LT_CHECK(hipblasLtMatrixLayoutCreate(&output_layout, HIP_R_16BF, n, m, n));

    LT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    const std::uint64_t workspace_limit = maximum_workspace;
    LT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_limit, sizeof(workspace_limit)));

    // The heuristic list is ordered by estimated time. Keep this a bounded
    // feasibility test rather than an exhaustive library tuning campaign.
    std::array<hipblasLtMatmulHeuristicResult_t, 1> results{};
    int result_count = 0;
    std::cerr << "querying one hipBLASLt heuristic\n";
    LT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle, operation, a_layout, b_layout, output_layout, output_layout,
        preference, static_cast<int>(results.size()), results.data(), &result_count));
    if (result_count == 0) throw std::runtime_error("no installed gfx1201 E4M3 algorithm");
    std::cerr << "timing one hipBLASLt algorithm\n";

    hipStream_t stream{};
    HIP_CHECK(hipStreamCreate(&stream));
    std::vector<float> timings;
    timings.reserve(static_cast<std::size_t>(result_count));
    for (int index = 0; index < result_count; ++index) {
        if (results[index].state != HIPBLAS_STATUS_SUCCESS ||
            results[index].workspaceSize > maximum_workspace) continue;
        try {
            timings.push_back(time_algorithm(
                handle, operation, b.get(), a_layout, a.get(), b_layout,
                output.get(), output_layout, results[index].algo, workspace.get(),
                results[index].workspaceSize, stream));
        } catch (const std::exception&) {
            // A failing heuristic cannot be selected; retain successful candidates.
        }
    }
    HIP_CHECK(hipStreamDestroy(stream));
    if (timings.empty()) throw std::runtime_error("all returned E4M3 algorithms failed");
    std::sort(timings.begin(), timings.end());
    std::array<std::uint16_t, 2> output_samples{};
    HIP_CHECK(hipMemcpy(&output_samples[0], output.get(), sizeof(std::uint16_t),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&output_samples[1], output.get() + (m * n - 1),
                        sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    if (output_samples[0] != 0U || output_samples[1] != 0U) {
        throw std::runtime_error("zero-input E4M3 GEMM did not produce zero BF16 endpoints");
    }
    const double operations = 2.0 * static_cast<double>(m) * n * k;
    const double best_ms = timings.front();
    const double tflops = operations / (best_ms * 1.0e9);
    std::cout << std::fixed << std::setprecision(6)
              << "{\n  \"shape\": {\"m\": " << m << ", \"n\": " << n
              << ", \"k\": " << k << "},\n"
              << "  \"heuristic_count\": " << result_count << ",\n"
              << "  \"successful_count\": " << timings.size() << ",\n"
              << "  \"best_ms\": " << best_ms << ",\n"
              << "  \"best_tflops\": " << tflops << "\n}\n";

    (void)hipblasLtMatmulPreferenceDestroy(preference);
    (void)hipblasLtMatrixLayoutDestroy(output_layout);
    (void)hipblasLtMatrixLayoutDestroy(b_layout);
    (void)hipblasLtMatrixLayoutDestroy(a_layout);
    (void)hipblasLtMatmulDescDestroy(operation);
    (void)hipblasLtDestroy(handle);
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << "fp8_hipblaslt_probe: " << error.what() << '\n';
    return EXIT_FAILURE;
}
