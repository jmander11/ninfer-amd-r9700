#pragma once

#include "core/tensor.h"
#include "ops/r9700/linear/fp8_activation.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::ops {

// Explicit device-bound library lifetime shared by serialized prepared Linears.
// Construct before the caller's final device-memory capacity snapshot. The
// context must outlive all borrowing executions and their submitted device work.
class LinearExecutionContext final {
public:
    LinearExecutionContext();
    ~LinearExecutionContext();
    LinearExecutionContext(const LinearExecutionContext&) = delete;
    LinearExecutionContext& operator=(const LinearExecutionContext&) = delete;
    LinearExecutionContext(LinearExecutionContext&&) = delete;
    LinearExecutionContext& operator=(LinearExecutionContext&&) = delete;

private:
    friend class LinearExecution;
    hipblasLtHandle_t handle_ = nullptr;
};

// Prepared row-scaled-E4M3 Linear implementation profile. N and K are fixed by
// the directly bound Weight; each T must be prepared explicitly before run().
// Library state is borrowed; descriptors and per-width algorithms are owned.
class LinearExecution final {
public:
    static constexpr std::size_t kMaximumMatmulWorkspaceBytes = std::size_t{512} << 20U;

    struct LaunchStatus {
        hipError_t hip            = hipSuccess;
        hipblasStatus_t hipblaslt = HIPBLAS_STATUS_SUCCESS;

        [[nodiscard]] bool ok() const noexcept {
            return hip == hipSuccess && hipblaslt == HIPBLAS_STATUS_SUCCESS;
        }
    };

    struct AlgorithmProfile {
        std::size_t matmul_workspace_bytes = 0;
        std::size_t algorithm_max_workspace_bytes = 0;
        float waves_count = 0.0F;
        int heuristic_rank = -1;
        std::array<std::uint8_t, 16> fingerprint{};
    };

    struct PreparedProfile {
        std::uint32_t tokens = 0;
        std::size_t activation_workspace_bytes = 0;
        std::size_t selected_matmul_workspace_bytes = 0;
        std::size_t selected_algorithm_max_workspace_bytes = 0;
        float selected_waves_count = 0.0F;
        int selected_heuristic_rank = -1;
        int heuristic_result_count = 0;
        std::array<std::uint8_t, 16> selected_algorithm_fingerprint{};
        std::vector<AlgorithmProfile> viable_algorithms;
    };

    // Preparation owns no activation buffer: immutable weight/width descriptors
    // and library allocations can become resident before capacity resolution.
    // A zero matmul limit restricts preparation to zero-workspace algorithms.
    LinearExecution(LinearExecutionContext& context, const Weight& weight,
                    std::size_t activation_storage_capacity_bytes,
                    std::size_t matmul_workspace_bytes);
    ~LinearExecution();

    LinearExecution(const LinearExecution&)            = delete;
    LinearExecution& operator=(const LinearExecution&) = delete;
    LinearExecution(LinearExecution&&) noexcept;
    LinearExecution& operator=(LinearExecution&&) noexcept;

    // Bind once, after the caller's arena is allocated and before any launch.
    // Regions must be disjoint from weights and each other and stable until all
    // submitted work completes. Borrowers of one context/region are serialized.
    // This updates descriptor pointers only; it performs no heuristic search.
    void bind_storage(void* activation_storage, std::size_t activation_storage_capacity_bytes,
                      void* matmul_workspace, std::size_t matmul_workspace_bytes);

    [[nodiscard]] static std::size_t activation_workspace_capacity_bytes(
        std::uint32_t tokens, std::uint32_t columns) noexcept;

    // Idempotently creates immutable descriptors and enumerates the returned
    // supported heuristics fitting the retained matmul workspace. The default
    // build selects the first; an exact-shape qualification build may require
    // and select a predeclared common identity.
    [[nodiscard]] const PreparedProfile& prepare(std::uint32_t tokens);
    [[nodiscard]] const PreparedProfile* prepared_profile(
        std::uint32_t tokens) const noexcept;
    [[nodiscard]] const r9700::linear::Fp8ActivationWorkspace* activation_workspace(
        std::uint32_t tokens) const noexcept;

    [[nodiscard]] std::uint32_t rows() const noexcept;
    [[nodiscard]] std::uint32_t columns() const noexcept;
    [[nodiscard]] int hipblaslt_version() const noexcept;

    // Enqueues BF16->E4M3 quantization, prepared E4M3 GEMM, and a status
    // consumer that poisons the complete BF16 output on nonfinite input.
    [[nodiscard]] LaunchStatus run(std::uint32_t tokens, const hip_bfloat16* input,
                                   hip_bfloat16* output, hipStream_t stream) noexcept;

    // Same GEMM and status consumer as run(), consuming the E4M3 activation that the immediately
    // preceding run() of another execution bound to the same activation storage wrote for the
    // same width and column count (one quantization shared by projections of one input).
    [[nodiscard]] LaunchStatus run_quantized(std::uint32_t tokens, hip_bfloat16* output,
                                             hipStream_t stream) noexcept;

    // Qualification-only control using the first viable heuristic while preserving the same
    // activation quantizer, descriptors, weight bytes, and output/status boundary.
    [[nodiscard]] LaunchStatus run_default_heuristic(
        std::uint32_t tokens, const hip_bfloat16* input, hip_bfloat16* output,
        hipStream_t stream) noexcept;

    // Qualification-only execution of one exact fingerprint from the fixed
    // returned viable-heuristic catalog for this prepared width.
    [[nodiscard]] LaunchStatus run_qualified_algorithm(
        std::uint32_t tokens, const std::array<std::uint8_t, 16>& fingerprint,
        const hip_bfloat16* input, hip_bfloat16* output, hipStream_t stream) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::ops
