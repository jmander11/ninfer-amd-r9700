#include "../../src/targets/qwen3_8_27b/impl/variant.h"

#define NINFER_QWEN3_VARIANT ::ninfer::targets::qwen3_8_27b::detail::Variant
#define NINFER_QWEN3_RUNTIME_NS qwen3_8_27b_runtime
#include "../../src/targets/qwen3/impl/runtime/dflash_context_impl.h"

#include "../../src/core/arena.h"
#include "../../src/core/device.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    try {
        namespace runtime =
            ninfer::targets::qwen3::detail::qwen3_8_27b_runtime;
        using Config = runtime::DFlashConfig;

        ninfer::DeviceContext device(0);
        require(device.is_gfx1201() && device.props.warpSize == 32,
                "DFlash qualifier did not select wave32 gfx1201");

        ninfer::LayoutBuilder builder;
        runtime::DFlashPersistentLayout layout;
        layout.local = ninfer::plan_cyclic_kv_cache(builder, Config::local_layers,
                                                    Config::local_capacity, Config::kv_heads,
                                                    Config::head_dim, 2);
        layout.rewrite_checkpoint_local = ninfer::plan_cyclic_kv_cache(
            builder, Config::local_layers, Config::local_capacity, Config::kv_heads,
            Config::head_dim, 2);
        layout.staging_local = ninfer::plan_cyclic_kv_cache(
            builder, Config::local_layers, Config::local_capacity, Config::kv_heads,
            Config::head_dim, 1);
        layout.prefill_features = builder.add_tensor(
            ninfer::DType::BF16, {Config::feature_rows, 1}, 256, "DFlash prefill features");
        layout.prefill_positions =
            builder.add_tensor(ninfer::DType::I32, {1}, 256, "DFlash prefill positions");
        layout.pending_features = builder.add_tensor(
            ninfer::DType::BF16, {Config::feature_rows, 1, 2}, 256, "DFlash pending features");

        require(!layout.full.has_value(),
                "Qwen3.8 DFlash2 unexpectedly planned a growing Full cache");
        require(layout.local.payload_bytes() != 0 &&
                    layout.kv_payload_bytes() == layout.local.payload_bytes() +
                                                     layout.rewrite_checkpoint_local.payload_bytes() +
                                                     layout.staging_local.payload_bytes(),
                "DFlash payload accounting does not describe the three BF16 cyclic states");

        ninfer::DeviceBuffer backing(builder.finish(256, "DFlash persistent state"));
        runtime::DFlashPersistentState state({backing.data(), backing.size()}, layout);
        require(!state.full.has_value() && state.local.layer_count() == Config::local_layers &&
                    state.local.capacity() == Config::local_capacity &&
                    state.local.num_kv_heads() == Config::kv_heads &&
                    state.local.head_dim() == Config::head_dim &&
                    state.local.lane_capacity() == 2,
                "DFlash2 did not bind its fixed BF16 cyclic geometry");

        constexpr std::int32_t lane = 1;
        for (std::uint32_t layer = 0; layer < state.local.layer_count(); ++layer) {
            const ninfer::CyclicKVCacheLayerView view = state.local.layer_view(layer);
            const ninfer::Tensor key = view.k.slice(3, lane, 1);
            const ninfer::Tensor value = view.v.slice(3, lane, 1);
            HIP_CHECK(hipMemsetAsync(key.data, 0x35 + static_cast<int>(layer), key.bytes(),
                                     device.stream));
            HIP_CHECK(hipMemsetAsync(value.data, 0x65 + static_cast<int>(layer), value.bytes(),
                                     device.stream));
        }
        state.save_rewrite_checkpoint(lane, device.stream);
        device.synchronize();

        std::vector<std::byte> expected(state.rewrite_checkpoint_local.lane_host_bytes());
        state.rewrite_checkpoint_local.copy_lane_to_host(lane, expected.data(), device.stream);
        device.synchronize();
        require(std::any_of(expected.begin(), expected.end(), [](std::byte byte) {
                    return byte != std::byte{0};
                }),
                "DFlash checkpoint image remained zero after BF16 state capture");

        for (std::uint32_t layer = 0; layer < state.local.layer_count(); ++layer) {
            const ninfer::CyclicKVCacheLayerView view = state.local.layer_view(layer);
            HIP_CHECK(hipMemsetAsync(view.k.slice(3, lane, 1).data, 0,
                                     view.k.slice(3, lane, 1).bytes(), device.stream));
            HIP_CHECK(hipMemsetAsync(view.v.slice(3, lane, 1).data, 0,
                                     view.v.slice(3, lane, 1).bytes(), device.stream));
        }
        state.restore_rewrite_checkpoint(lane, device.stream);
        std::vector<std::byte> restored(state.local.lane_host_bytes());
        state.local.copy_lane_to_host(lane, restored.data(), device.stream);
        device.synchronize();
        require(restored == expected,
                "DFlash HIP rewrite-checkpoint restore changed the BF16 cyclic image");

        std::cout << "dflash_state: PASS cyclic_bytes=" << layout.kv_payload_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dflash_state: FAIL: " << error.what() << '\n';
        return 1;
    }
}
