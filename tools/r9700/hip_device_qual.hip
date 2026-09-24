#include "../../src/core/device.h"
#include "../../src/core/arena.h"
#include "../../src/core/decode_graph.h"

#include <array>
#include <iostream>

int main() {
    try {
        ninfer::DeviceContext device(0);
        if (!device.is_gfx1201() || device.props.warpSize != 32) {
            std::cerr << "selected HIP device is not wave32 gfx1201\n";
            return 1;
        }
        ninfer::DeviceEventTimer timer(device);
        timer.start();
        const float elapsed = timer.stop_ms();
        const std::array<std::uint32_t, 4> source = {0x12345678U, 0x9abcdef0U, 7U, 11U};
        std::array<std::uint32_t, 4> copied{};
        ninfer::DeviceBuffer buffer(sizeof(source));
        buffer.copy_from_host(source.data(), sizeof(source));
        buffer.copy_to_host(copied.data(), sizeof(copied));
        if (copied != source) {
            std::cerr << "HIP buffer copy did not preserve bytes\n";
            return 1;
        }
        bool rejected_buffer_range = false;
        try {
            buffer.copy_from_host(source.data(), 1U, buffer.size());
        } catch (const std::out_of_range&) {
            rejected_buffer_range = true;
        }
        if (!rejected_buffer_range) {
            std::cerr << "HIP buffer accepted an out-of-range transfer\n";
            return 1;
        }
        copied.fill(0U);
        buffer.copy_from_host_async(source.data(), sizeof(source), device.stream);
        device.order_copy_after_compute();
        buffer.copy_to_host_async(copied.data(), sizeof(copied), device.copy_stream);
        device.synchronize_all();
        if (copied != source) {
            std::cerr << "ordered HIP asynchronous buffer copy did not preserve bytes\n";
            return 1;
        }
        ninfer::DeviceArena arena(1024);
        const auto first = arena.alloc_bytes(17, 256);
        {
            const auto scope = arena.scope();
            (void)scope;
            const auto second = arena.alloc_bytes(300, 512);
            if (reinterpret_cast<std::uintptr_t>(second.data) % 512U != 0U) {
                std::cerr << "HIP arena did not honor alignment\n";
                return 1;
            }
        }
        if (arena.used() != static_cast<std::size_t>(
                                reinterpret_cast<std::uintptr_t>(first.data) -
                                reinterpret_cast<std::uintptr_t>(arena.base())) + first.bytes) {
            std::cerr << "HIP arena scope did not restore offset\n";
            return 1;
        }
        arena.reset_peak();
        if (arena.peak_used() != arena.used()) {
            std::cerr << "HIP arena did not reset its peak accounting\n";
            return 1;
        }
        const ninfer::Tensor tensor = arena.alloc(ninfer::DType::FP32, {2, 4}, 256);
        if (tensor.data == nullptr || tensor.bytes() != 8U * sizeof(float) ||
            reinterpret_cast<std::uintptr_t>(tensor.data) % 256U != 0U) {
            std::cerr << "HIP arena did not preserve Tensor allocation semantics\n";
            return 1;
        }
        ninfer::PinnedHostBuffer pinned(4096);
        if (pinned.data() == nullptr || pinned.size() != 4096U) {
            std::cerr << "HIP pinned staging allocation is invalid\n";
            return 1;
        }
        ninfer::HostPinnedArena pinned_arena(1024);
        void* pinned_first = pinned_arena.try_alloc(17, 256);
        void* pinned_second = pinned_arena.try_alloc(300, 512);
        if (pinned_first == nullptr || pinned_second == nullptr ||
            reinterpret_cast<std::uintptr_t>(pinned_second) % 512U != 0U) {
            std::cerr << "HIP pinned arena allocation is invalid\n";
            return 1;
        }
        pinned_arena.free(pinned_first);
        pinned_arena.free(pinned_second);
        if (pinned_arena.used() != 0U) {
            std::cerr << "HIP pinned arena did not release its allocation\n";
            return 1;
        }
        ninfer::DecodeGraphDefinition graph;
        graph.capture(device.stream, [&] { HIP_CHECK(hipMemsetAsync(buffer.data(), 0x4c, buffer.size(), device.stream)); });
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(graph);
        executable.upload(device.stream);
        executable.launch(device.stream);
        device.synchronize();
        buffer.copy_to_host(copied.data(), sizeof(copied));
        for (const auto word : copied) {
            if (word != 0x4c4c4c4cU) { std::cerr << "HIP graph replay wrote wrong bytes\n"; return 1; }
        }
        ninfer::DecodeGraphDefinition updated_graph;
        updated_graph.capture(device.stream, [&] {
            HIP_CHECK(hipMemsetAsync(buffer.data(), 0x5a, buffer.size(), device.stream));
        });
        executable.update(updated_graph);
        executable.launch(device.stream);
        device.synchronize();
        buffer.copy_to_host(copied.data(), sizeof(copied));
        for (const auto word : copied) {
            if (word != 0x5a5a5a5aU) {
                std::cerr << "HIP graph update wrote wrong bytes\n";
                return 1;
            }
        }
        device.order_copy_after_compute();
        device.synchronize_all();

        std::cout << "hip_core_device: PASS arch=" << device.props.gcnArchName
                  << " wave=" << device.props.warpSize
                  << " vram_mib=" << device.total_vram() / (1024 * 1024)
                  << " event_ms=" << elapsed << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hip_core_device: FAIL: " << error.what() << '\n';
        return 1;
    }
}
