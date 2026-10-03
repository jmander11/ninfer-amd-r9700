#include "../../src/targets/qwen3_8_27b/impl/variant.h"

#define NINFER_QWEN3_VARIANT    ::ninfer::targets::qwen3_8_27b::detail::Variant
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
        namespace runtime = ninfer::targets::qwen3::detail::qwen3_8_27b_runtime;
        using Config      = runtime::DFlashConfig;

        ninfer::DeviceContext device(0);
        require(device.is_gfx1201() && device.props.warpSize == 32,
                "DFlash qualifier did not select wave32 gfx1201");

        ninfer::LayoutBuilder builder;
        runtime::DFlashPersistentLayout layout;
        layout.local =
            ninfer::plan_cyclic_kv_cache(builder, Config::local_layers, Config::local_capacity,
                                         Config::kv_heads, Config::head_dim, 2);
        layout.staging_local =
            ninfer::plan_cyclic_kv_cache(builder, Config::local_layers, Config::local_capacity,
                                         Config::kv_heads, Config::head_dim, 1);
        layout.prefill_features = builder.add_tensor(ninfer::DType::BF16, {Config::feature_rows, 1},
                                                     256, "DFlash prefill features");
        layout.prefill_positions =
            builder.add_tensor(ninfer::DType::I32, {1}, 256, "DFlash prefill positions");
        layout.pending_features = builder.add_tensor(
            ninfer::DType::BF16, {Config::feature_rows, 1, 2}, 256, "DFlash pending features");

        require(!layout.full.has_value(),
                "Qwen3.8 DFlash2 unexpectedly planned a growing Full cache");
        require(layout.local.payload_bytes() != 0 &&
                    layout.kv_payload_bytes() ==
                        layout.local.payload_bytes() + layout.staging_local.payload_bytes(),
                "DFlash payload accounting does not describe the two BF16 cyclic states");

        ninfer::DeviceBuffer backing(builder.finish(256, "DFlash persistent state"));
        runtime::DFlashPersistentState state({backing.data(), backing.size()}, layout);
        require(!state.full.has_value() && state.local.layer_count() == Config::local_layers &&
                    state.local.capacity() == Config::local_capacity &&
                    state.local.num_kv_heads() == Config::kv_heads &&
                    state.local.head_dim() == Config::head_dim && state.local.lane_capacity() == 2,
                "DFlash2 did not bind its fixed BF16 cyclic geometry");

        // Exact byte oracle for the lane-owned pinned rewrite image: distinct patterns per lane,
        // copied with the cyclic primitives the Program's capture uses (lane -> staging lane ->
        // pinned image on the copy stream) and both restores (image -> lane, staging -> lane).
        constexpr std::int32_t lane  = 1;
        const std::size_t lane_bytes = state.local.lane_host_bytes();
        require(lane_bytes == state.staging_local.lane_host_bytes(),
                "DFlash staging lane image differs from the per-lane image");
        const auto pattern = [lane_bytes](std::uint32_t seed) {
            std::vector<std::byte> bytes(lane_bytes);
            std::uint32_t x = seed;
            for (std::byte& byte : bytes) {
                x    = x * 1664525U + 1013904223U;
                byte = static_cast<std::byte>(x >> 24);
            }
            return bytes;
        };
        const std::vector<std::byte> expected = pattern(0x9e3779b9U);
        const std::vector<std::byte> other    = pattern(0x7f4a7c15U);
        state.local.copy_lane_from_host(expected.data(), lane, device.stream);
        state.local.copy_lane_from_host(other.data(), 0, device.stream);

        ninfer::PinnedHostBuffer image(lane_bytes);
        std::fill_n(static_cast<unsigned char*>(image.data()), lane_bytes, 0);
        hipEvent_t snapshot_done = nullptr;
        HIP_CHECK(hipEventCreateWithFlags(&snapshot_done, hipEventDisableTiming));
        state.staging_local.copy_lane_from(state.local, lane, 0, device.stream);
        HIP_CHECK(hipEventRecord(snapshot_done, device.stream));
        HIP_CHECK(hipStreamWaitEvent(device.copy_stream, snapshot_done, 0));
        state.staging_local.copy_lane_to_host(0, image.data(), device.copy_stream);
        HIP_CHECK(hipStreamSynchronize(device.copy_stream));
        HIP_CHECK(hipEventDestroy(snapshot_done));
        require(std::equal(expected.begin(), expected.end(),
                           static_cast<const std::byte*>(image.data())),
                "DFlash rewrite capture into the pinned image changed the lane bytes");

        const auto clear_lane = [&] {
            for (std::uint32_t layer = 0; layer < state.local.layer_count(); ++layer) {
                const ninfer::CyclicKVCacheLayerView view = state.local.layer_view(layer);
                HIP_CHECK(hipMemsetAsync(view.k.slice(3, lane, 1).data, 0,
                                         view.k.slice(3, lane, 1).bytes(), device.stream));
                HIP_CHECK(hipMemsetAsync(view.v.slice(3, lane, 1).data, 0,
                                         view.v.slice(3, lane, 1).bytes(), device.stream));
            }
        };
        const auto require_lanes = [&](const char* message) {
            std::vector<std::byte> restored(lane_bytes);
            std::vector<std::byte> neighbour(lane_bytes);
            state.local.copy_lane_to_host(lane, restored.data(), device.stream);
            state.local.copy_lane_to_host(0, neighbour.data(), device.stream);
            device.synchronize();
            require(restored == expected && neighbour == other, message);
        };
        clear_lane();
        state.local.copy_lane_from_host(image.data(), lane, device.stream);
        require_lanes("DFlash restore from the pinned rewrite image is not byte exact");
        clear_lane();
        state.local.copy_lane_from(state.staging_local, 0, lane, device.stream);
        require_lanes("DFlash restore from the staging lane is not byte exact");

        std::cout << "dflash_state: PASS cyclic_bytes=" << layout.kv_payload_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dflash_state: FAIL: " << error.what() << '\n';
        return 1;
    }
}
