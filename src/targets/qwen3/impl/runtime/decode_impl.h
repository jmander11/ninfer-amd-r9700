#include "targets/qwen3/impl/runtime/instance.h"
#include "targets/qwen3/impl/runtime/schedule.h"

#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"

#include <stdexcept>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::schedule {
namespace {

auto ordinary_batch_body(OrdinaryBatchContext& state, std::int32_t batch_size) {
    return [&state, batch_size] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3::OrdinaryDecodeState& ordinary = state.frame;
        HIP_CHECK(hipMemcpyAsync(ordinary.ingress.data, &state.host_ingress,
                                 sizeof(qwen3::OrdinaryDecodeIngress), hipMemcpyHostToDevice,
                                 state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.model,
                         state.execution.linear_execution, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);
        if (!state.text_kv_transactions.empty()) {
            card.set_text_kv_transactions(state.text_kv_transactions);
        }

        Tensor tokens          = ordinary.tokens.slice(0, 0, batch_size);
        Tensor cache_positions = ordinary.cache_positions.slice(0, 0, batch_size);
        Tensor rope_positions  = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows         = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes           = ordinary.lanes.slice(0, 0, batch_size);
        Tensor hidden          = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits          = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled         = ordinary.sampled_tokens.slice(0, 0, batch_size);

        if (state.text_kv_transactions.size() != static_cast<std::size_t>(batch_size)) {
            throw std::logic_error("ordinary decode batch has incomplete Text KV authorities");
        }
        const auto* all_positions = static_cast<const std::int32_t*>(cache_positions.data);
        for (std::int32_t row = 0; row < batch_size; ++row) {
            state.text_kv_transactions[static_cast<std::size_t>(row)]->begin_device_segment(
                all_positions + row, 1U, nullptr);
        }
        card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows, lanes,
                                   hidden, logits);
        for (qwen3::PagedKVTransaction* transaction : state.text_kv_transactions) {
            transaction->end_device_segment(state.execution.device.stream);
        }
        ops::scatter(hidden, lanes, state.continuation_hidden_store, state.execution.device.stream);
        ops::sample(logits, sampled, TextConfig::token_domain, ordinary.sampling, cache_positions,
                    ops::kSamplePurposeDecode, state.execution.work, state.execution.device.stream);
        HIP_CHECK(hipMemcpyAsync(&state.host_egress, ordinary.egress.data,
                                 sizeof(qwen3::OrdinaryDecodeEgress), hipMemcpyDeviceToHost,
                                 state.execution.device.stream));
    };
}

} // namespace

void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   DecodeGraphDefinition& definition) {
    auto body = ordinary_batch_body(state, batch_size);
    capture_graph(state, definition, body);
}

void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::schedule
