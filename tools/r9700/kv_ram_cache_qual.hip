#include "core/arena.h"
#include "core/device.h"
#include "core/fp8_int4_paged_kv_cache.h"
#include "targets/qwen3/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3/impl/runtime/prefix_identity.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace q3 = ninfer::targets::qwen3;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

q3::PreparedPromptData prompt_data(std::int32_t first_token) {
    q3::PreparedPromptData prompt;
    for (std::int32_t i = 0; i < 7; ++i) { prompt.token_ids.push_back(first_token + i); }
    prompt.token_types.assign(prompt.token_ids.size(), 0);
    prompt.positions.resize(prompt.token_ids.size() * 3U);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t token = 0; token < prompt.token_ids.size(); ++token) {
            prompt.positions[axis * prompt.token_ids.size() + token] =
                static_cast<std::int32_t>(token);
        }
    }
    return prompt;
}

void paint_allocation(const ninfer::PagedKVPool& pool,
                      const ninfer::PagedKVAllocation& allocation, std::uint8_t seed,
                      hipStream_t stream) {
    for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
        const ninfer::Tensor& plane = pool.plane(plane_index);
        std::vector<std::uint8_t> host(plane.bytes(), 0xE1U);
        for (std::size_t logical = 0; logical < allocation.page_ids().size(); ++logical) {
            const std::int32_t physical = allocation.page_ids()[logical];
            for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                const std::uint8_t value = static_cast<std::uint8_t>(
                    seed + 13U * plane_index + 3U * logical + static_cast<std::size_t>(head));
                if (pool.plane_order(plane_index) == ninfer::PagedKVPlaneOrder::PageMajor) {
                    const std::size_t begin = static_cast<std::size_t>(physical) * plane.nb[3];
                    std::fill_n(host.begin() + static_cast<std::ptrdiff_t>(begin),
                                static_cast<std::size_t>(plane.nb[3]), value);
                    break;
                }
                const std::size_t begin = static_cast<std::size_t>(head) * plane.nb[3] +
                                          static_cast<std::size_t>(physical) * plane.nb[2];
                std::fill_n(host.begin() + static_cast<std::ptrdiff_t>(begin),
                            static_cast<std::size_t>(plane.nb[2]), value);
            }
        }
        HIP_CHECK(hipMemcpyAsync(plane.data, host.data(), host.size(), hipMemcpyHostToDevice,
                                 stream));
    }
}

void verify_allocation(const ninfer::PagedKVPool& pool,
                       const ninfer::PagedKVAllocation& allocation, std::uint8_t seed,
                       const char* label) {
    for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
        const ninfer::Tensor& plane = pool.plane(plane_index);
        std::vector<std::uint8_t> host(plane.bytes());
        HIP_CHECK(hipMemcpy(host.data(), plane.data, host.size(), hipMemcpyDeviceToHost));
        for (std::size_t logical = 0; logical < allocation.page_ids().size(); ++logical) {
            const std::int32_t physical = allocation.page_ids()[logical];
            for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                const std::uint8_t expected = static_cast<std::uint8_t>(
                    seed + 13U * plane_index + 3U * logical + static_cast<std::size_t>(head));
                const std::size_t begin =
                    pool.plane_order(plane_index) == ninfer::PagedKVPlaneOrder::PageMajor
                        ? static_cast<std::size_t>(physical) * plane.nb[3]
                        : static_cast<std::size_t>(head) * plane.nb[3] +
                              static_cast<std::size_t>(physical) * plane.nb[2];
                const std::size_t bytes =
                    pool.plane_order(plane_index) == ninfer::PagedKVPlaneOrder::PageMajor
                        ? static_cast<std::size_t>(plane.nb[3])
                        : static_cast<std::size_t>(plane.nb[2]);
                for (std::size_t byte = 0; byte < bytes; ++byte) {
                    if (host[begin + byte] != expected) {
                        throw std::runtime_error(std::string(label) +
                                                 " restored logical-page byte mismatch");
                    }
                }
                if (pool.plane_order(plane_index) == ninfer::PagedKVPlaneOrder::PageMajor) {
                    break;
                }
            }
        }
    }
}

bool all_bytes_equal(const ninfer::DeviceBuffer& buffer, std::uint8_t value) {
    std::vector<std::uint8_t> host(buffer.size());
    buffer.copy_to_host(host.data(), host.size());
    return std::all_of(host.begin(), host.end(), [value](std::uint8_t byte) {
        return byte == value;
    });
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(0);
        require(device.is_gfx1201() && device.props.warpSize == 32,
                "KV RAM qualifier requires wave32 gfx1201");

        const ninfer::Fp8KInt4VPagedKVSpec text_spec{
            .page_group_count = 10,
            .logical_page_capacity = 8,
            .table_rows = 4,
            .layer_count = 2,
            .head_dim = 256,
            .num_kv_heads = 4,
            .value_group = 16,
            .plane_layouts = {.key = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                              .value = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                              .value_scale =
                                  ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor},
        };
        const ninfer::Fp8KInt4VPagedKVSpec mtp_spec{
            .page_group_count = 8,
            .logical_page_capacity = 8,
            .table_rows = 4,
            .layer_count = 1,
            .head_dim = 256,
            .num_kv_heads = 4,
            .value_group = 16,
            .plane_layouts = {.key = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                              .value = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                              .value_scale =
                                  ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor},
        };
        const auto text_semantics = ninfer::fp8_k_int4_v_semantic_fingerprint(text_spec);
        const auto mtp_semantics = ninfer::fp8_k_int4_v_semantic_fingerprint(mtp_spec);

        ninfer::LayoutBuilder text_source_builder;
        const auto text_source_layout =
            ninfer::plan_fp8_k_int4_v_paged_kv_pool(text_source_builder, text_spec);
        ninfer::DeviceBuffer text_source_backing(text_source_builder.finish(256));
        ninfer::PagedKVPool text_source_pool(
            {text_source_backing.data(), text_source_backing.size()}, text_source_layout.storage);
        auto text_source = text_source_pool.reserve(3);
        text_source.materialize_pages(1, device.stream);
        auto text_hole = text_source_pool.reserve(1);
        text_hole.materialize_pages(1, device.stream);
        text_source.materialize_pages(3, device.stream);

        ninfer::LayoutBuilder mtp_source_builder;
        const auto mtp_source_layout =
            ninfer::plan_fp8_k_int4_v_paged_kv_pool(mtp_source_builder, mtp_spec);
        ninfer::DeviceBuffer mtp_source_backing(mtp_source_builder.finish(256));
        ninfer::PagedKVPool mtp_source_pool(
            {mtp_source_backing.data(), mtp_source_backing.size()}, mtp_source_layout.storage);
        auto mtp_source = mtp_source_pool.reserve(3);
        mtp_source.materialize_pages(1, device.stream);
        auto mtp_hole = mtp_source_pool.reserve(1);
        mtp_hole.materialize_pages(1, device.stream);
        mtp_source.materialize_pages(3, device.stream);

        require(text_source_pool.plane_count() == 6 && mtp_source_pool.plane_count() == 3,
                "typed Text/MTP pools do not own exactly three planes per layer");
        require(text_source.page_ids()[1] != text_source.page_ids()[0] + 1 &&
                    mtp_source.page_ids()[1] != mtp_source.page_ids()[0] + 1,
                "source page maps were not fragmented");
        paint_allocation(text_source_pool, text_source, 0x21U, device.stream);
        paint_allocation(mtp_source_pool, mtp_source, 0x71U, device.stream);
        device.order_copy_after_compute();

        q3::PreparedPromptData prompt = prompt_data(101);
        q3::detail::ResidentPrefixIdentity identity;
        identity.assign(prompt);
        q3::detail::RamCaptureSource source;
        source.execution_frontier = 6;
        source.ledger_frontier = 7;
        source.text_kv_valid = 6;
        source.mtp_kv_valid = 5;
        source.ledger = prompt.token_ids;
        source.identity = &identity;
        source.hash_f = q3::detail::prefix_hash_at(prompt.token_ids, identity, 6);
        source.text = &text_source;
        source.text_pool = &text_source_pool;
        source.text_semantics = text_semantics;
        source.backend = &mtp_source;
        source.backend_pool = &mtp_source_pool;
        source.backend_semantics = mtp_semantics;
        source.stream = device.copy_stream;

        q3::detail::KVRamCache cache(8U << 20);
        require(cache.capture(source), "Text+MTP KV RAM capture failed");
        const auto match = cache.plan_match(prompt, q3::detail::prefix_hash_chain(prompt));
        require(match.has_value(), "captured KV RAM entry did not match its exact identity");

        ninfer::LayoutBuilder text_dest_builder;
        const auto text_dest_layout =
            ninfer::plan_fp8_k_int4_v_paged_kv_pool(text_dest_builder, text_spec);
        ninfer::DeviceBuffer text_dest_backing(text_dest_builder.finish(256));
        ninfer::PagedKVPool text_dest_pool({text_dest_backing.data(), text_dest_backing.size()},
                                           text_dest_layout.storage);
        auto text_dest = text_dest_pool.reserve(3);
        text_dest.materialize_pages(1, device.stream);
        auto text_dest_hole = text_dest_pool.reserve(1);
        text_dest_hole.materialize_pages(1, device.stream);
        text_dest.materialize_pages(3, device.stream);

        ninfer::LayoutBuilder mtp_dest_builder;
        const auto mtp_dest_layout =
            ninfer::plan_fp8_k_int4_v_paged_kv_pool(mtp_dest_builder, mtp_spec);
        ninfer::DeviceBuffer mtp_dest_backing(mtp_dest_builder.finish(256));
        ninfer::PagedKVPool mtp_dest_pool({mtp_dest_backing.data(), mtp_dest_backing.size()},
                                          mtp_dest_layout.storage);
        auto mtp_dest = mtp_dest_pool.reserve(3);
        mtp_dest.materialize_pages(1, device.stream);
        auto mtp_dest_hole = mtp_dest_pool.reserve(1);
        mtp_dest_hole.materialize_pages(1, device.stream);
        mtp_dest.materialize_pages(3, device.stream);
        HIP_CHECK(hipStreamSynchronize(device.stream));
        text_dest_backing.fill(0xA5);
        mtp_dest_backing.fill(0xA5);

        q3::detail::RamRestoreTarget target;
        target.text_dst_pages = 3;
        target.backend_dst_pages = 3;
        target.text = &text_dest;
        target.text_pool = &text_dest_pool;
        target.text_semantics = text_semantics;
        target.backend = &mtp_dest;
        target.backend_pool = &mtp_dest_pool;
        target.backend_semantics = mtp_semantics;
        target.stream = device.stream;

        auto wrong = target;
        wrong.text_semantics.value_group = 32;
        bool rejected_mismatch = false;
        try {
            (void)cache.unpack_device(match->entry_id, wrong);
        } catch (const std::invalid_argument&) {
            rejected_mismatch = true;
        }
        HIP_CHECK(hipStreamSynchronize(device.stream));
        require(rejected_mismatch, "semantic fingerprint mismatch was accepted");
        require(all_bytes_equal(text_dest_backing, 0xA5) &&
                    all_bytes_equal(mtp_dest_backing, 0xA5),
                "semantic mismatch modified a destination byte");

        auto wrong_backend = target;
        wrong_backend.backend_semantics->value_group = 32;
        rejected_mismatch = false;
        try {
            (void)cache.unpack_device(match->entry_id, wrong_backend);
        } catch (const std::invalid_argument&) {
            rejected_mismatch = true;
        }
        HIP_CHECK(hipStreamSynchronize(device.stream));
        require(rejected_mismatch, "backend semantic fingerprint mismatch was accepted");
        require(all_bytes_equal(text_dest_backing, 0xA5) &&
                    all_bytes_equal(mtp_dest_backing, 0xA5),
                "backend semantic mismatch modified a destination byte");

        const q3::detail::RamRestoredHost restored =
            cache.unpack_device(match->entry_id, target);
        HIP_CHECK(hipStreamSynchronize(device.stream));
        require(restored.text_kv_valid == 6 && restored.mtp_kv_valid == 5,
                "restored Text/MTP frontiers changed");
        verify_allocation(text_dest_pool, text_dest, 0x21U, "Text");
        verify_allocation(mtp_dest_pool, mtp_dest, 0x71U, "MTP");
        const auto copy_seconds = cache.harvest_copy_seconds();
        require(copy_seconds.save >= 0.0 && copy_seconds.load >= 0.0,
                "copy timing accounting is invalid");

        const std::size_t entry_bytes = cache.snapshot().used_bytes;
        require(entry_bytes != 0, "captured KV RAM entry has no storage");
        q3::detail::KVRamCache eviction_cache(entry_bytes + 255U);
        require(eviction_cache.capture(source), "eviction probe first capture failed");
        require(eviction_cache.capture(source), "eviction probe replacement capture failed");
        const auto eviction = eviction_cache.snapshot();
        require(eviction.entry_count == 1 && eviction.captures == 2 && eviction.evictions == 1,
                "unpinned FIFO entry was not evicted for the replacement image");

        std::cout << "kv_ram_cache: PASS text_planes=" << text_source_pool.plane_count()
                  << " mtp_planes=" << mtp_source_pool.plane_count()
                  << " entry_bytes=" << entry_bytes << " evictions=" << eviction.evictions
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "kv_ram_cache: FAIL: " << error.what() << '\n';
        return 1;
    }
}
