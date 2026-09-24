// Implements: include/ninfer/ops/sampling.h
// Match: validated contiguous BF16/I32 tensors and a shared-layout workspace.
// Algorithm assumptions: launcher and kernels use sampler_multiblock_ok() from
// the same layout authority. Small/fallback shapes use the per-row kernel; the
// partial/group/mass pipeline owns every multiblock mode.
#include "ninfer/ops/sampling.h"

#include "ops/common/math.h"
#include "ops/r9700/sampling/sampling_kernels.h"
#include "core/device.h"

namespace ninfer::ops::detail {

std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain, std::int32_t columns) {
    return make_sampling_workspace_layout(token_domain, columns).bytes;
}

void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, hipStream_t stream) {
    const std::int32_t physical_rows     = logits.ne[0];
    const std::int32_t batch             = logits.ne[1];
    const auto* positions                = static_cast<const std::int32_t*>(logical_positions.data);
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, batch);
    if (!layout.multiblock) {
        sample_row_kernel<<<static_cast<unsigned int>(batch), kSamplerBlock, 0, stream>>>(
            static_cast<const hip_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            configs, positions, purpose, token_domain, physical_rows);
        HIP_CHECK(hipGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(batch));
    sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const hip_bfloat16*>(logits.data), configs, token_domain, physical_rows,
        scratch);
    HIP_CHECK(hipGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(batch));
    sampling_group_finalize_sample_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
        partial_blocks, groups, scratch);
    HIP_CHECK(hipGetLastError());
    const dim3 p_less_grid(
        static_cast<unsigned int>(sampler_p_less_workers_per_column(partial_blocks, batch)),
        static_cast<unsigned int>(batch));
    sampling_p_less_mass_sample_kernel<<<p_less_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const hip_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
        configs, positions, purpose, token_domain, physical_rows, partial_blocks, scratch);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail

#include "ninfer/types.h"
#include "ninfer/ops/argmax.h"
#include "ops/r9700/sampling/argmax_kernels.h"
#include <algorithm>
#include <stdexcept>
namespace ninfer::ops {
namespace { constexpr int kMaximumRequestRows=static_cast<int>(kMaximumConcurrency); }
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
            const SamplingConfig* configs, std::int32_t columns_per_config, hipStream_t stream) {
    if (stream == nullptr || configs == nullptr || columns_per_config <= 0 ||
        logits.dtype != DType::BF16 || out.dtype != DType::I32 ||
        logits.ne[0] <= 0 || logits.ne[1] <= 0 || logits.ne[2] != 1 || logits.ne[3] != 1 ||
        out.ne[0] != logits.ne[1] || out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1 ||
        valid_rows <= 0 || valid_rows > logits.ne[0] || logits.ne[1] % columns_per_config != 0 ||
        logits.ne[1] / columns_per_config > kMaximumRequestRows ||
        !logits.is_contiguous() || !out.is_contiguous() || !logits.data || !out.data ||
        logits.data == out.data) {
        throw std::invalid_argument("argmax: invalid configured vocabulary/column geometry");
    }
    argmax_kernel<<<logits.ne[1], kArgmaxBlock, 0, stream>>>(
        static_cast<const hip_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
        valid_rows, logits.ne[0], configs, columns_per_config);
    HIP_CHECK(hipGetLastError());
}
std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain,
                                              std::int32_t min_lanes,
                                              std::int32_t max_lanes) {
    if (token_domain <= 0 || min_lanes <= 0 || max_lanes < min_lanes ||
        max_lanes > kMaximumRequestRows) {
        throw std::invalid_argument("sampling workspace: invalid profile or lane interval");
    }
    return make_sampling_workspace_layout(token_domain, max_lanes).bytes;
}

void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions,
            std::int32_t purpose, WorkspaceArena& workspace, hipStream_t stream) {
    if (stream == nullptr) { throw std::invalid_argument("sample: stream must be non-null"); }
    if (logits.dtype != DType::BF16) { throw std::invalid_argument("sample: logits must be BF16"); }
    if (out.dtype != DType::I32) { throw std::invalid_argument("sample: out must be I32"); }
    if (logits.ne[2] != 1 || logits.ne[3] != 1 || logits.ne[0] <= 0 || logits.ne[1] <= 0 ||
        logits.ne[1] > kMaximumRequestRows) {
        throw std::invalid_argument("sample: logits must be rank-2 [physical_rows,B], B in [1,4]");
    }
    if (out.ne[0] != logits.ne[1] || out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("sample: out must be I32 [B]");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument("sample: token_domain must be in [1,physical_rows]");
    }
    if (logical_positions.dtype != DType::I32 || logical_positions.ne[0] != logits.ne[1] ||
        logical_positions.ne[1] != 1 || logical_positions.ne[2] != 1 ||
        logical_positions.ne[3] != 1) {
        throw std::invalid_argument("sample: logical_positions must be I32 [B]");
    }
    if (!logits.is_contiguous() || !out.is_contiguous() || !logical_positions.is_contiguous()) {
        throw std::invalid_argument("sample: tensors must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr || logical_positions.data == nullptr ||
        configs == nullptr) {
        throw std::invalid_argument("sample: device pointers must be non-null");
    }

    auto scope = workspace.scope();
    const auto bytes = make_sampling_workspace_layout(token_domain, logits.ne[1]).bytes;
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    detail::sample_batch_launch(logits, out, token_domain, configs, logical_positions, purpose, scratch, stream);
}
} // namespace ninfer::ops
