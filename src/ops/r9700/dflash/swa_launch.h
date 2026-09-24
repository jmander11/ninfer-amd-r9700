#pragma once

#include "ninfer/ops/swa.h"

namespace ninfer::ops::detail {

struct SwaLaunchPlan {
    bool direct = true;
    std::int32_t splits = 1;
};

void swa_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                const Tensor& positions, const Tensor& valid_columns, const Tensor& lanes,
                float scale, const CyclicKVCacheLayerView& context,
                SwaContextExecutionEnvelope envelope, SwaLaunchPlan plan,
                const Tensor& partial_acc, const Tensor& partial_m, const Tensor& partial_l,
                Tensor& out, hipStream_t stream);

} // namespace ninfer::ops::detail
