#pragma once

#include "core/tensor.h"
#include "ops/r9700/linear/fp8_activation.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::ops {

// Row-scaled-E4M3 Linear bound to one directly bound Weight: per-token E4M3 quantization of the
// represented BF16 input followed by fp8_small_t_linear (T <= 32) or
// fp8_row_scaled_prefill_linear (T > 32). N and K must be multiples of 128 (every Qwen3.8
// protected projection); other weights are rejected at construction. The activation image lives
// in one caller-owned region bound once before any launch.
class LinearExecution final {
public:
    LinearExecution(const Weight& weight, std::size_t activation_storage_capacity_bytes);
    ~LinearExecution();

    LinearExecution(const LinearExecution&)            = delete;
    LinearExecution& operator=(const LinearExecution&) = delete;
    LinearExecution(LinearExecution&&) noexcept;
    LinearExecution& operator=(LinearExecution&&) noexcept;

    // Bind once, after the caller's arena is allocated and before any launch. The region must be
    // disjoint from the weight and stable until all submitted work completes. Borrowers of one
    // region are serialized.
    void bind_storage(void* activation_storage, std::size_t activation_storage_capacity_bytes);

    [[nodiscard]] static std::size_t
    activation_workspace_capacity_bytes(std::uint32_t tokens, std::uint32_t columns) noexcept;

    // The E4M3 activation image of a T-token call in the bound region; empty before binding or
    // when T exceeds the region.
    [[nodiscard]] std::optional<r9700::linear::Fp8ActivationWorkspace>
    activation_workspace(std::uint32_t tokens) const noexcept;

    [[nodiscard]] std::uint32_t rows() const noexcept;
    [[nodiscard]] std::uint32_t columns() const noexcept;

    // Enqueues BF16->E4M3 quantization and the projection; every output element of a token whose
    // input is nonfinite is the canonical BF16 quiet NaN.
    [[nodiscard]] hipError_t run(std::uint32_t tokens, const hip_bfloat16* input,
                                 hip_bfloat16* output, hipStream_t stream) noexcept;

    // Same projection, consuming the E4M3 activation that the immediately preceding run() of
    // another execution bound to the same activation storage wrote for the same width and column
    // count (one quantization shared by projections of one input).
    [[nodiscard]] hipError_t run_quantized(std::uint32_t tokens, hip_bfloat16* output,
                                           hipStream_t stream) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::ops
