#pragma once

#include "ninfer/ops/bidirectional_gqa_attention.h"

namespace ninfer::ops::detail {

void bidirectional_gqa_attention_launch(
    const Tensor& q, const Tensor& query_k, const Tensor& query_v,
    const Tensor& context_lengths, const Tensor& valid_columns, const Tensor& table_rows,
    float scale, const BidirectionalGqaBF16ContextView& context,
    GqaContextExecutionEnvelope envelope, Tensor& out, hipStream_t stream);

} // namespace ninfer::ops::detail
