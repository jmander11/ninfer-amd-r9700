#include "ops/r9700/dflash/bidirectional_gqa_attention_launch.h"

#include "core/device.h"
#include "ops/r9700/dflash/bidirectional_gqa_attention_kernels.h"

namespace ninfer::ops::detail {

void bidirectional_gqa_attention_launch(
    const Tensor& q, const Tensor& query_k, const Tensor& query_v,
    const Tensor& context_lengths, const Tensor& valid_columns, const Tensor& table_rows,
    float scale, const BidirectionalGqaBF16ContextView& context,
    GqaContextExecutionEnvelope envelope, Tensor& out, hipStream_t stream) {
    const dim3 grid(r9700::kBidirectionalKvHeads, static_cast<unsigned int>(q.ne[2]),
                    static_cast<unsigned int>(q.ne[3]));
    r9700::bidirectional_gqa_bf16_kernel<<<grid, r9700::kBidirectionalBlock, 0, stream>>>(
        static_cast<const hip_bfloat16*>(q.data),
        static_cast<const hip_bfloat16*>(query_k.data),
        static_cast<const hip_bfloat16*>(query_v.data),
        static_cast<const std::int32_t*>(context_lengths.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<const hip_bfloat16*>(context.key_pages.data),
        static_cast<const hip_bfloat16*>(context.value_pages.data),
        static_cast<const std::int32_t*>(context.block_tables.data),
        static_cast<hip_bfloat16*>(out.data), q.ne[2], context.key_pages.ne[2],
        context.block_tables.ne[0], context.block_tables.ne[1],
        static_cast<int>(envelope.min_context), static_cast<int>(envelope.max_context), scale);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail
