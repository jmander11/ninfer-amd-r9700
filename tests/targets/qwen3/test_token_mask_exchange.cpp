#include "core/device.h"
#include "core/decode_graph.h"
#include "targets/qwen3/impl/runtime/token_masks.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>

int main() {
    using namespace ninfer;
    namespace q3 = ninfer::targets::qwen3;
    DeviceContext device;
    constexpr std::int32_t words = (q3::kTokenDomain + 31) / 32;
    DeviceBuffer masks(words * sizeof(std::uint32_t));
    DeviceBuffer sampling(sizeof(ops::SamplingConfig));
    DeviceBuffer ids(sizeof(std::int32_t));
    DeviceBuffer counts(sizeof(std::int32_t));
    const std::int32_t token = 0, count = 1;
    ids.copy_from_host(&token, sizeof(token));
    counts.copy_from_host(&count, sizeof(count));
    q3::TokenMaskExchange exchange(
        Tensor(masks.data(), DType::I32, {words, 1, 1}),
        Tensor(sampling.data(), DType::U8, {static_cast<std::int32_t>(sampling.size())}));
    const std::array<const q3::OutputSession*, 1> outputs{nullptr};
    std::array<ops::SamplingConfig, 1> configs{};
    configs[0].temperature = 0.75F;
    configs[0].seed        = 1234;
    const Tensor id_view(ids.data(), DType::I32, {1, 1});
    const Tensor count_view(counts.data(), DType::I32, {1});
    DecodeGraphDefinition graph;
    graph.capture(device.stream, [&] {
        (void)exchange.publish(id_view, nullptr, count_view, device.stream);
        exchange.acquire(device.stream);
    });
    DecodeGraphExecutable executable;
    executable.instantiate(graph);
    executable.upload(device.stream);
    device.synchronize();
    for (int round = 0; round < 3; ++round) {
        configs[0].seed += 1;
        exchange.bind(outputs, configs);
        if (round != 0) exchange.arm();
        executable.launch(device.stream);
        if (round == 0) {
            // Withhold host service beyond the old two-second deadline after submitting
            // the tiny graph to an otherwise idle, synchronized stream.
            std::this_thread::sleep_for(std::chrono::milliseconds(2300));
            exchange.arm();
        }
        device.synchronize();
        exchange.finish_round();
        ops::SamplingConfig actual;
        sampling.copy_to_host(&actual, sizeof(actual));
        if (actual.temperature != configs[0].temperature || actual.seed != configs[0].seed ||
            actual.allowed_token_words != nullptr || actual.allowed_token_column_stride != 0) {
            std::cerr << "delayed/repeated graph exchange returned incorrect sampling config\n";
            return 1;
        }
    }
    std::cout << "delayed and repeated graph mask exchange passed\n";
}
