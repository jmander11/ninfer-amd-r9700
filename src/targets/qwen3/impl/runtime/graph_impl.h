#pragma once

#include "targets/qwen3/impl/runtime/instance.h"
#include "targets/qwen3/impl/runtime/schedule.h"

#include "ninfer/ops/persistent_decode.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::schedule {

template <class Context, class Body>
void run_prepared(Context& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        executable->launch(state.execution.device.stream);
    } else {
        body();
    }
}

// Captures one decode graph definition. A single-sequence definition is lowered to persistent
// decode kernels (ops/persistent_decode.h): bitwise the captured launches, measured faster at
// C1; multi-sequence definitions keep their kernel nodes (lowered, C4 DFlash decoded 2.7% slower
// even with its wide phases hosted; docs/performance.md).
template <class Context, class Body>
void capture_graph(Context& state, DecodeGraphDefinition& definition, std::int32_t batch_size,
                   Body&& body) {
    state.execution.work.reset();
    definition.capture(state.execution.device.stream, body);
    if (batch_size == 1) definition.rewrite(ops::persistent_decode_lower);
}

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::schedule
