#pragma once

// Rejected timing prototype, included inside the attention implementation's
// anonymous namespace. Launch 256 threads and grid (48, ceil(query_rows / 16)):
// x / 2 is the query head, x % 2 is the 128-feature half. No new workspace.
struct DensePvHiloCandidateShared {
    float probability[16][16];
    std::uint16_t probability_hi[16][16];
    std::uint16_t probability_lo[16][16];
    std::uint16_t value_hi[16][128];
    std::uint16_t value_lo[16][128];
    float denominator[16];
    std::uint32_t row_context[16];
    std::uint32_t invalid_rows;
    std::uint32_t inactive_rows;
    std::uint32_t maximum_context;
    std::size_t physical_base;
    std::size_t table_offset;
    std::uint32_t page_valid;
};

template <std::uint32_t Group, bool DeviceActiveRows>
__global__ __launch_bounds__(kDensePrefillThreads)
void dense_full_score_pv_hilo_candidate_kernel(
    const float* scores, const float* maxima, const std::uint8_t* values,
    const std::uint16_t* scales, const std::uint32_t* pages,
    const std::int32_t* page_table_row, std::uint32_t page_table_row_stride,
    std::uint32_t page_table_row_count, std::size_t context, std::size_t physical_tokens,
    const std::int32_t* row_positions, const std::int32_t* active_query_rows,
    std::uint32_t total_query_rows, std::uint32_t panel_origin,
    std::uint32_t whole_rows, float* output) {
    static_assert(Group == 16U, "The PV prototype qualifies only the production Group16 cache");
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    __shared__ DensePvHiloCandidateShared shared;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t axis = lane & 15U;
    const std::uint32_t half = lane >> 4U;
    const std::uint32_t wave = tid >> 5U;
    const std::uint32_t query_head = blockIdx.x / 2U;
    const std::uint32_t kv_head = query_head / 6U;
    const std::uint32_t feature_base = (blockIdx.x % 2U) * 128U;
    const std::uint32_t query_base = blockIdx.y * 16U;
    const std::uint32_t tile_rows = min(16U, total_query_rows - query_base);
    if (tid == 0U) {
        const std::int32_t active = DeviceActiveRows
            ? dense_panel_active_rows(active_query_rows, total_query_rows, panel_origin, whole_rows)
            : static_cast<std::int32_t>(total_query_rows);
        shared.inactive_rows = 0U;
        shared.invalid_rows = 0U;
        shared.maximum_context = 0U;
        shared.table_offset = 0U;
        const bool invalid_active = active < 0 ||
            static_cast<std::uint32_t>(active) > total_query_rows;
        for (std::uint32_t row = 0U; row < 16U; ++row) {
            shared.row_context[row] = 0U;
            shared.denominator[row] = 0.0F;
            if (row >= tile_rows) continue;
            const std::uint32_t bit = 1U << row;
            const std::uint32_t global_row = query_base + row;
            if (invalid_active) shared.invalid_rows |= bit;
            else if (global_row >= static_cast<std::uint32_t>(active))
                shared.inactive_rows |= bit;
            else {
                const std::int32_t position = row_positions[global_row];
                if (position < 0 || static_cast<std::size_t>(position) >= context)
                    shared.invalid_rows |= bit;
                else {
                    shared.row_context[row] = static_cast<std::uint32_t>(position) + 1U;
                    shared.maximum_context = max(shared.maximum_context, shared.row_context[row]);
                }
            }
        }
        if (!invalid_active && shared.maximum_context != 0U && page_table_row != nullptr) {
            const std::int32_t table_row = page_table_row[0];
            if (table_row < 0 || static_cast<std::uint32_t>(table_row) >= page_table_row_count) {
                shared.invalid_rows |= ((1U << tile_rows) - 1U) & ~shared.inactive_rows;
                shared.maximum_context = 0U;
            } else shared.table_offset = static_cast<std::size_t>(table_row) * page_table_row_stride;
        }
    }
    F32x8 numerator{};
    __syncthreads();
    for (std::uint32_t key_base = 0U; key_base < shared.maximum_context; key_base += 16U) {
        if (tid == 0U) {
            const std::size_t physical_page = pages[shared.table_offset + key_base / kPageSize];
            shared.page_valid = physical_page < physical_tokens / kPageSize ? 1U : 0U;
            shared.physical_base = physical_page * kPageSize + key_base % kPageSize;
            if (shared.page_valid == 0U)
                for (std::uint32_t row = 0U; row < tile_rows; ++row)
                    if (shared.row_context[row] > key_base) shared.invalid_rows |= 1U << row;
        }
        __syncthreads();
        // All 256 lanes load one probability. Read the previous invalid mask
        // before any lane publishes this block's nonfinite-score failures.
        const std::uint32_t row = tid / 16U;
        const std::uint32_t token = tid % 16U;
        const bool row_valid = (shared.invalid_rows & (1U << row)) == 0U;
        float probability = 0.0F;
        bool invalid_score = false;
        if (row < tile_rows && key_base + token < shared.row_context[row] &&
            shared.page_valid != 0U && row_valid) {
            const std::size_t vector = static_cast<std::size_t>(query_base + row) * 24U + query_head;
            const float maximum = maxima[vector];
            const float score = scores[vector * context + key_base + token];
            invalid_score = !isfinite(maximum) || !isfinite(score);
            if (!invalid_score) probability = __expf(score - maximum);
        }
        __syncthreads();
        if (invalid_score) atomicOr(&shared.invalid_rows, 1U << row);
        const hip_bfloat16 probability_hi(probability);
        const hip_bfloat16 probability_lo(probability - static_cast<float>(probability_hi));
        shared.probability[row][token] = probability;
        shared.probability_hi[row][token] = probability_hi.data;
        shared.probability_lo[row][token] = probability_lo.data;
        for (std::uint32_t flat = tid; flat < 16U * 128U; flat += 256U) {
            const std::uint32_t key = flat / 128U;
            const std::uint32_t local_feature = flat % 128U;
            const std::uint32_t feature = feature_base + local_feature;
            float value = 0.0F;
            if (shared.page_valid != 0U && key_base + key < shared.maximum_context) {
                const std::uint8_t packed = values[int4_offset<Group, false, false>(
                    shared.physical_base + key, kv_head, feature & ~1U,
                    4U, physical_tokens, false)];
                int code = static_cast<int>((packed >> ((feature & 1U) * 4U)) & 15U);
                if (code >= 8) code -= 16;
                const std::uint16_t scale = scales[int4_offset<Group, false, false>(
                    shared.physical_base + key, kv_head, (feature / Group) * Group,
                    4U, physical_tokens, true)];
                value = static_cast<float>(code) * __half2float(__ushort_as_half(scale));
            }
            // Signed INT4 times finite FP16 has <=14 significant bits, so this
            // BF16 pair reconstructs the represented public V exactly.
            const hip_bfloat16 value_hi(value);
            const hip_bfloat16 value_lo(value - static_cast<float>(value_hi));
            shared.value_hi[key][local_feature] = value_hi.data;
            shared.value_lo[key][local_feature] = value_lo.data;
        }
        __syncthreads();
        if (tid < 16U) {
            float sum = 0.0F;
            for (std::uint32_t key = 0U; key < 16U; ++key)
                sum += shared.probability[tid][key];
            shared.denominator[tid] += sum;
        }
        I16x8 p_hi{}, p_lo{}, v_hi{}, v_lo{};
#pragma unroll
        for (std::uint32_t item = 0U; item < 8U; ++item) {
            const std::uint32_t key = half * 8U + item;
            const std::uint32_t feature = wave * 16U + axis;
            p_hi[item] = static_cast<short>(shared.probability_hi[axis][key]);
            p_lo[item] = static_cast<short>(shared.probability_lo[axis][key]);
            v_hi[item] = static_cast<short>(shared.value_hi[key][feature]);
            v_lo[item] = static_cast<short>(shared.value_lo[key][feature]);
        }
        numerator = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(p_hi, v_hi, numerator);
        numerator = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(p_hi, v_lo, numerator);
        numerator = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(p_lo, v_hi, numerator);
        numerator = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(p_lo, v_lo, numerator);
        __syncthreads();
    }
#pragma unroll
    for (std::uint32_t item = 0U; item < 8U; ++item) {
        const std::uint32_t row = half * 8U + item;
        if (row >= tile_rows) continue;
        const std::uint32_t bit = 1U << row;
        const std::size_t index =
            (static_cast<std::size_t>(query_base + row) * 24U + query_head) * kHeadDim +
            feature_base + wave * 16U + axis;
        if ((shared.inactive_rows & bit) != 0U) output[index] = 0.0F;
        else if ((shared.invalid_rows & bit) != 0U || shared.denominator[row] == 0.0F)
            output[index] = __builtin_nanf("");
        else output[index] = numerator[item] / shared.denominator[row];
    }
#else
    (void)scores; (void)maxima; (void)values; (void)scales; (void)pages;
    (void)page_table_row; (void)page_table_row_stride; (void)page_table_row_count;
    (void)context; (void)physical_tokens; (void)row_positions; (void)active_query_rows;
    (void)total_query_rows; (void)panel_origin; (void)whole_rows; (void)output;
#endif
}
