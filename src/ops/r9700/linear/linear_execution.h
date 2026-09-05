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

namespace ninfer::ops {

// Prepared row-scaled-E4M3 Linear implementation profile. N and K are fixed by
// the directly bound Weight; each T must be prepared explicitly before run().
// This owner remains qualification-only until the target Program owns and
// threads it through selected role calls.
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

    struct PreparedProfile {
        std::uint32_t tokens = 0;
        std::size_t activation_workspace_bytes = 0;
        std::size_t selected_matmul_workspace_bytes = 0;
        std::size_t selected_algorithm_max_workspace_bytes = 0;
        float selected_waves_count = 0.0F;
        int selected_heuristic_rank = -1;
        int heuristic_result_count = 0;
        std::array<std::uint8_t, 16> selected_algorithm_fingerprint{};
    };

    // The weight and caller-owned activation region, plus any non-empty matmul
    // workspace, must be mutually disjoint and remain stable until submitted work completes.
    // A null/zero matmul workspace explicitly restricts preparation to zero-workspace algorithms.
    // Multiple owners may reference the same externally serialized workspace
    // regions; one owner itself must not be used concurrently.
    LinearExecution(const Weight& weight, void* activation_storage,
                    std::size_t activation_storage_capacity_bytes,
                    void* matmul_workspace, std::size_t matmul_workspace_bytes);
    ~LinearExecution();

    LinearExecution(const LinearExecution&)            = delete;
    LinearExecution& operator=(const LinearExecution&) = delete;
    LinearExecution(LinearExecution&&) noexcept;
    LinearExecution& operator=(LinearExecution&&) noexcept;

    [[nodiscard]] static std::size_t activation_workspace_capacity_bytes(
        std::uint32_t tokens, std::uint32_t columns) noexcept;

    // Idempotently creates immutable descriptors and selects the first
    // supported heuristic fitting the retained matmul workspace.
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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::ops
