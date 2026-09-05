#include "../../src/core/arena.h"
#include "../../src/core/cyclic_kv_cache.h"
#include "../../src/core/decode_graph.h"
#include "../../src/core/device.h"
#include "../../src/core/fp8_int4_paged_kv_cache.h"
#include "../../src/core/layout.h"
#include "../../src/core/linear_attention_state.h"
#include "../../src/ops/r9700/kv/fp8_int4_kv_append.h"
#include "../../src/runtime/engine/request_memory.h"
#include "../../src/targets/qwen3/impl/runtime/r9700_cache_profile.h"
#include <ninfer/targets/qwen3/decoder_state.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void synchronize(hipStream_t stream) { HIP_CHECK(hipStreamSynchronize(stream)); }

std::vector<std::byte> copy_cyclic_lane(const ninfer::CyclicKVCache& cache, std::int32_t lane,
                                        hipStream_t stream) {
    std::vector<std::byte> image(cache.lane_host_bytes());
    cache.copy_lane_to_host(lane, image.data(), stream);
    synchronize(stream);
    return image;
}

std::vector<std::byte> copy_paged_allocation(const ninfer::PagedKVAllocation& allocation,
                                             const ninfer::PagedKVPool& pool,
                                             hipStream_t stream) {
    std::vector<std::byte> image(
        ninfer::paged_kv_host_image_bytes(pool, allocation.mapped_page_count()));
    ninfer::pack_paged_kv_allocation_to_host(allocation, pool, image.data(), stream);
    synchronize(stream);
    return image;
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(0);
        require(device.is_gfx1201() && device.props.warpSize == 32,
                "state qualifier did not select wave32 gfx1201");

        ninfer::runtime::RequestMemory request_memory(device, 4096);
        request_memory.activate(1024, 256);
        const auto active = request_memory.region();
        require(active && active.size == 1024 && active.alignment == 256 &&
                    reinterpret_cast<std::uintptr_t>(active.data) % 256U == 0U,
                "request memory activation did not retain its frozen device allocation");
        bool rejected_growth = false;
        try {
            request_memory.activate(4097, 256);
        } catch (const std::invalid_argument&) {
            rejected_growth = true;
        }
        require(rejected_growth, "request memory permitted a request-time backing allocation");
        request_memory.deactivate();
        require(!request_memory.region() && request_memory.summary().capacity_bytes == 4096,
                "request memory deactivation changed frozen backing ownership");

        ninfer::LayoutBuilder paged_builder;
        const ninfer::Fp8KInt4VPagedKVSpec paged_spec{
            .page_group_count = 4,
            .logical_page_capacity = 4,
            .table_rows = 2,
            .layer_count = 2,
            .head_dim = 32,
            .num_kv_heads = 1,
            .value_group = 16,
        };
        const auto paged_layout =
            ninfer::plan_fp8_k_int4_v_paged_kv_pool(paged_builder, paged_spec);
        ninfer::DeviceBuffer paged_backing(paged_builder.finish(256, "typed paged cache"));
        ninfer::PagedKVPool paged_pool({paged_backing.data(), paged_backing.size()},
                                       paged_layout.storage);
        auto allocation = paged_pool.reserve(2);
        allocation.materialize_tokens(65, device.stream);
        allocation.bind_row(1, device.stream);
        synchronize(device.stream);
        require(allocation.mapped_page_count() == 2 && allocation.bound_row() == 1 &&
                    paged_pool.free_pages() == 2,
                "typed paged allocation did not retain its mapped-page lifetime");
        std::int32_t table[4] = {-1, -1, -1, -1};
        HIP_CHECK(hipMemcpy(table, allocation.block_table().data, sizeof(table),
                            hipMemcpyDeviceToHost));
        require(table[0] == allocation.page_ids()[0] && table[1] == allocation.page_ids()[1],
                "typed paged block-table mapping was not published on the HIP stream");

        for (std::size_t plane = 0; plane < paged_pool.plane_count(); ++plane) {
            HIP_CHECK(hipMemsetAsync(paged_pool.plane(plane).data,
                                     static_cast<int>(0x20U + plane),
                                     paged_pool.plane(plane).bytes(), device.stream));
        }
        const std::size_t paged_image_bytes =
            ninfer::paged_kv_host_image_bytes(paged_pool, allocation.mapped_page_count());
        std::vector<std::byte> paged_image(paged_image_bytes);
        ninfer::pack_paged_kv_allocation_to_host(allocation, paged_pool, paged_image.data(),
                                                 device.stream);
        synchronize(device.stream);
        paged_pool.zero_pages(allocation.page_ids(), device.stream);
        ninfer::unpack_paged_kv_allocation_from_host(allocation, paged_pool, paged_image.data(),
                                                     allocation.mapped_page_count(),
                                                     allocation.mapped_page_count(), device.stream);
        std::vector<std::byte> paged_round_trip(paged_image_bytes);
        ninfer::pack_paged_kv_allocation_to_host(allocation, paged_pool, paged_round_trip.data(),
                                                 device.stream);
        synchronize(device.stream);
        require(paged_round_trip == paged_image,
                "typed FP8-K/INT4-V page spill/restore changed persistent bytes");
        allocation.release();
        require(paged_pool.free_pages() == paged_pool.page_group_count() &&
                    paged_pool.mapped_pages() == 0,
                "typed paged allocation release leaked physical pages");

        // The canonical Qwen decoder owner has only the typed three-plane cache. Its transaction
        // binds one allocation/publication pair, launches every layer on one stream, and makes
        // the frontier visible only after the shared device status word is read successfully.
        namespace q3 = ninfer::targets::qwen3;
        ninfer::LayoutBuilder decoder_builder;
        const q3::DecoderStateSpec decoder_spec{
            .full_attention_layers = 2,
            .mtp_layers = 1,
            .capacity = 128,
            .kv_heads = 1,
            .attention_head_dim = 256,
            .value_group = ninfer::targets::qwen3::detail::kR9700TextKVValueGroup,
            .text_plane_layouts =
                ninfer::targets::qwen3::detail::kR9700TextKVPlaneLayouts,
            .mtp_plane_layouts = {
                .key = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                .value = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                .value_scale = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
            },
            .enable_mtp = true,
            .kv_table_rows = 2,
            .text_physical_page_groups = 3,
            .mtp_physical_page_groups = 2,
            .linear_attention = {
                .layers = 1,
                .conv_channels = 4,
                .conv_width = 3,
                .value_heads = 1,
                .value_head_dim = 4,
                .key_head_dim = 4,
                .slot_count = 1,
                .conv_dtype = ninfer::DType::BF16,
            },
        };
        const auto decoder_layout = q3::plan_decoder_state(decoder_builder, decoder_spec);
        ninfer::DeviceBuffer decoder_backing(decoder_builder.finish(256, "Qwen decoder state"));
        q3::DecoderState decoder({decoder_backing.data(), decoder_backing.size()}, decoder_layout);
        require(decoder.text_kv.pool().plane_count() == 6 && decoder.mtp_cache() != nullptr &&
                    decoder.mtp_cache()->pool().plane_count() == 3,
                "canonical Qwen decoder state is not exactly three planes per layer");
        require(decoder.text_kv.spec().value_group ==
                        ninfer::targets::qwen3::detail::kR9700TextKVValueGroup &&
                    decoder.text_kv.spec().plane_layouts ==
                        ninfer::targets::qwen3::detail::kR9700TextKVPlaneLayouts,
                "Qwen decoder state does not match the fixed R9700 schedule profile");
        const auto decoder_fingerprint = decoder.text_kv.fingerprint();
        ninfer::require_fp8_k_int4_v_semantic_fingerprint(decoder_fingerprint,
                                                           decoder_fingerprint);
        bool rejected_decoder_fingerprint = false;
        try {
            auto incompatible = decoder_fingerprint;
            incompatible.value_group = decoder_fingerprint.value_group == 16 ? 32 : 16;
            ninfer::require_fp8_k_int4_v_semantic_fingerprint(incompatible,
                                                               decoder_fingerprint);
        } catch (const std::invalid_argument&) {
            rejected_decoder_fingerprint = true;
        }
        require(rejected_decoder_fingerprint,
                "canonical Qwen decoder cache accepted an incompatible RAM fingerprint");
        auto decoder_allocation = decoder.text_kv.pool().reserve(2);
        decoder_allocation.materialize_tokens(66, device.stream);
        decoder_allocation.bind_row(0, device.stream);
        synchronize(device.stream);

        constexpr std::size_t decoder_tokens = 66;
        constexpr std::size_t decoder_elements = decoder_tokens * 256;
        std::vector<hip_bfloat16> decoder_keys(decoder_elements);
        std::vector<hip_bfloat16> decoder_values(decoder_elements);
        for (std::size_t i = 0; i < decoder_elements; ++i) {
            decoder_keys[i] = hip_bfloat16(static_cast<float>(static_cast<int>(i % 29U) - 14) /
                                           16.0F);
            decoder_values[i] = hip_bfloat16(static_cast<float>(static_cast<int>(i % 17U) - 8) /
                                             8.0F);
        }
        ninfer::DeviceBuffer decoder_keys_device(decoder_elements * sizeof(hip_bfloat16));
        ninfer::DeviceBuffer decoder_values_device(decoder_elements * sizeof(hip_bfloat16));
        decoder_keys_device.copy_from_host(decoder_keys.data(), decoder_keys_device.size());
        decoder_values_device.copy_from_host(decoder_values.data(), decoder_values_device.size());
        std::vector<std::uint32_t> decoder_positions(decoder_tokens);
        for (std::uint32_t i = 0; i < decoder_positions.size(); ++i) { decoder_positions[i] = i; }
        ninfer::DeviceBuffer transaction_positions(decoder_tokens * sizeof(std::uint32_t));
        ninfer::DeviceBuffer transaction_status(sizeof(std::uint32_t));
        const q3::PagedKVTransactionWorkspace transaction_workspace{
            .positions = static_cast<std::uint32_t*>(transaction_positions.data()),
            .position_capacity = decoder_tokens,
            .status = static_cast<std::uint32_t*>(transaction_status.data()),
        };

        q3::PagedKVPublication publication;
        q3::PagedKVLayerRead pending_layer_zero;
        {
            const std::uint64_t generation = publication.transaction_generation;
            auto transaction = decoder.text_kv.begin_append(
                decoder_allocation, publication, decoder_positions, transaction_workspace);
            require(publication.transaction_open &&
                        publication.transaction_generation == generation + 1U,
                    "Qwen append did not open a fresh publication generation");
            bool rejected_pending_before_launch = false;
            try {
                (void)transaction.pending_layer_read(0);
            } catch (const std::logic_error&) {
                rejected_pending_before_launch = true;
            }
            require(rejected_pending_before_launch,
                    "Qwen append exposed a pending layer before its codec launch");
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            pending_layer_zero = transaction.pending_layer_read(0);
            require(pending_layer_zero.valid_for(device.stream) &&
                        pending_layer_zero.pending() &&
                        pending_layer_zero.visible_frontier() == decoder_tokens,
                    "Qwen append did not authorize its launched layer on the ordered stream");
            bool rejected_unlaunched_layer = false;
            try {
                (void)transaction.pending_layer_read(1);
            } catch (const std::logic_error&) {
                rejected_unlaunched_layer = true;
            }
            require(rejected_unlaunched_layer,
                    "Qwen append authorized an unlaunched pending layer");
            transaction.launch_append_layer(
                1, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            require(transaction.pending_layer_read(1).valid_for(device.stream),
                    "Qwen append did not authorize its second launched layer");
            require(transaction.commit() == decoder_tokens &&
                        publication.valid_frontier == decoder_tokens && publication.healthy &&
                        !publication.transaction_open &&
                        !pending_layer_zero.valid_for(device.stream),
                    "Qwen append published its frontier before a successful all-layer commit");
        }
        const q3::PagedKVLayerRead committed_layer_zero =
            decoder.text_kv.published_layer_read(decoder_allocation, publication, 0);
        require(committed_layer_zero.valid_for(device.stream) &&
                    !committed_layer_zero.pending() &&
                    committed_layer_zero.visible_frontier() == decoder_tokens,
                "Qwen committed layer authorization is unavailable after append");
        decoder_allocation.unbind_row();
        decoder_allocation.bind_row(1, device.stream);
        synchronize(device.stream);
        require(!committed_layer_zero.valid_for(device.stream),
                "Qwen committed read survived allocation block-table rebinding");
        const q3::PagedKVLayerRead row_one_read =
            decoder.text_kv.published_layer_read(decoder_allocation, publication, 0);
        require(row_one_read.valid_for(device.stream),
                "Qwen committed read was unavailable after block-table rebinding");
        decoder_allocation.unbind_row();
        decoder_allocation.bind_row(0, device.stream);
        synchronize(device.stream);
        require(!row_one_read.valid_for(device.stream) &&
                    !committed_layer_zero.valid_for(device.stream),
                "Qwen stale committed read revived after a second block-table binding");
        const q3::PagedKVLayerRead rebound_committed =
            decoder.text_kv.published_layer_read(decoder_allocation, publication, 0);
        require(rebound_committed.valid_for(device.stream),
                "Qwen committed read was not refreshed for the rebound allocation");
        {
            constexpr std::array<std::uint32_t, 3> selected = {0, 3, 8};
            auto transaction = decoder.text_kv.begin_compact(
                decoder_allocation, publication, transaction_workspace);
            require(!rebound_committed.valid_for(device.stream),
                    "Qwen committed read survived an open compaction generation");
            bool rejected_compaction_read = false;
            try {
                (void)transaction.pending_layer_read(0);
            } catch (const std::logic_error&) {
                rejected_compaction_read = true;
            }
            require(rejected_compaction_read,
                    "Qwen compaction exposed an append-only pending read");
            transaction.launch_compact_layer(0, 55, selected, device.stream);
            transaction.launch_compact_layer(1, 55, selected, device.stream);
            require(transaction.commit() == 58 && publication.valid_frontier == 58 &&
                        publication.healthy &&
                        !rebound_committed.valid_for(device.stream),
                    "Qwen compaction did not publish the selected all-layer frontier");
        }
        require(decoder.text_kv
                    .published_layer_read(decoder_allocation, publication, 0)
                    .valid_for(device.stream),
                "Qwen committed layer authorization was not refreshed after compaction");
        bool rejected_noncontiguous = false;
        try {
            constexpr std::array<std::uint32_t, 1> gap = {59};
            (void)decoder.text_kv.begin_append(
                decoder_allocation, publication, gap, transaction_workspace);
        } catch (const std::invalid_argument&) {
            rejected_noncontiguous = true;
        }
        require(rejected_noncontiguous && publication.valid_frontier == 58,
                "Qwen append accepted a gap beyond the committed frontier");

        // The schedule already owns an I32 position panel on device. Consume that panel directly,
        // without a host conversion/staging buffer, and publish only after both cache layers and
        // the shared device status word have completed successfully.
        constexpr std::array<std::int32_t, 2> device_suffix = {58, 59};
        ninfer::DeviceBuffer device_suffix_positions(device_suffix.size() * sizeof(std::int32_t));
        device_suffix_positions.copy_from_host(device_suffix.data(),
                                               device_suffix_positions.size());
        const q3::PagedKVTransactionWorkspace device_transaction_workspace{
            .positions = nullptr,
            .position_capacity = 0,
            .status = static_cast<std::uint32_t*>(transaction_status.data()),
        };
        {
            const std::uint64_t generation = publication.transaction_generation;
            auto transaction = decoder.text_kv.begin_device_append(
                decoder_allocation, publication,
                static_cast<const std::int32_t*>(device_suffix_positions.data()),
                device_suffix.size(), device_transaction_workspace);
            require(publication.transaction_open &&
                        publication.transaction_generation == generation + 1U,
                    "Qwen device append did not open a fresh publication generation");
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            require(transaction.pending_layer_read(0).valid_for(device.stream) &&
                        transaction.pending_layer_read(0).visible_frontier() == 60,
                    "Qwen device append did not authorize its pending suffix");
            transaction.launch_append_layer(
                1, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            require(transaction.commit() == 60 && publication.valid_frontier == 60 &&
                        publication.healthy && !publication.transaction_open,
                    "Qwen device append did not atomically publish its valid suffix");
        }

        // Speculative cancellation/identity-prefix resolution is a publication-only rollback:
        // bytes beyond the retained frontier remain physically allocated but become unreadable,
        // and every authorization from the superseded generation is invalidated.
        const auto pre_truncate_read =
            decoder.text_kv.published_layer_read(decoder_allocation, publication, 0);
        const std::uint64_t pre_truncate_generation = publication.transaction_generation;
        decoder.text_kv.truncate_publication(decoder_allocation, publication, 58);
        require(publication.valid_frontier == 58 && publication.healthy &&
                    !publication.transaction_open &&
                    publication.transaction_generation == pre_truncate_generation + 1U &&
                    !pre_truncate_read.valid_for(device.stream),
                "Qwen publication truncation did not invalidate the speculative suffix");
        bool rejected_truncate_growth = false;
        try {
            decoder.text_kv.truncate_publication(decoder_allocation, publication, 59);
        } catch (const std::invalid_argument&) {
            rejected_truncate_growth = true;
        }
        require(rejected_truncate_growth && publication.valid_frontier == 58,
                "Qwen publication truncation accepted frontier growth");
        {
            auto transaction = decoder.text_kv.begin_device_append(
                decoder_allocation, publication,
                static_cast<const std::int32_t*>(device_suffix_positions.data()),
                device_suffix.size(), device_transaction_workspace);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            transaction.launch_append_layer(
                1, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            require(transaction.commit() == 60 && publication.valid_frontier == 60,
                    "Qwen append could not reuse a truncated physical suffix");
        }

        // One physical MTP round first aligns the licensed target prefix, then appends active AR
        // inputs one at a time. Device counts make inactive panel tails inert. Resolution retains
        // only the licensed alignment frontier; speculative AR bytes stay physically present but
        // become unreadable through the generation-checked publication.
        auto& mtp_cache = *decoder.mtp_cache();
        auto mtp_allocation = mtp_cache.pool().reserve(1);
        mtp_allocation.materialize_tokens(8, device.stream);
        mtp_allocation.bind_row(0, device.stream);
        mtp_cache.pool().zero_pages(mtp_allocation.page_ids(), device.stream);
        constexpr std::array<std::int32_t, 4> dynamic_positions = {0, 1, -1, 117};
        ninfer::DeviceBuffer dynamic_positions_device(dynamic_positions.size() *
                                                       sizeof(std::int32_t));
        dynamic_positions_device.copy_from_host(dynamic_positions.data(),
                                                dynamic_positions_device.size());
        ninfer::DeviceBuffer dynamic_count_device(sizeof(std::int32_t));
        const std::int32_t dynamic_count = 2;
        dynamic_count_device.copy_from_host(&dynamic_count, sizeof(dynamic_count));
        synchronize(device.stream);
        const auto mtp_zero_image =
            copy_paged_allocation(mtp_allocation, mtp_cache.pool(), device.stream);
        q3::PagedKVPublication mtp_publication;
        {
            auto transaction = mtp_cache.begin_device_prefix_append(
                mtp_allocation, mtp_publication,
                static_cast<const std::int32_t*>(dynamic_positions_device.data()),
                3U,
                static_cast<const std::int32_t*>(dynamic_count_device.data()),
                device_transaction_workspace);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            require(transaction.pending_layer_read(0).visible_frontier() == 3,
                    "MTP alignment did not authorize its smaller per-row maximum");
            require(transaction.commit() == 2 && mtp_publication.valid_frontier == 2 &&
                        mtp_publication.healthy && !mtp_publication.transaction_open,
                    "MTP alignment did not publish its licensed device prefix");
        }
        const std::uint32_t licensed_alignment_frontier = mtp_publication.valid_frontier;
        const auto mtp_alignment_image =
            copy_paged_allocation(mtp_allocation, mtp_cache.pool(), device.stream);
        require(mtp_alignment_image != mtp_zero_image,
                "MTP alignment did not change its represented cache bytes");

        // An exact two-token append on a different physical page is the byte oracle for the
        // device-prefix alignment. Equality proves that both invalid inactive positions were
        // ignored: one is inside the per-row maximum, and one is outside it in the wider panel.
        auto mtp_control_allocation = mtp_cache.pool().reserve(1);
        mtp_control_allocation.materialize_tokens(8, device.stream);
        mtp_control_allocation.bind_row(1, device.stream);
        mtp_cache.pool().zero_pages(mtp_control_allocation.page_ids(), device.stream);
        synchronize(device.stream);
        constexpr std::array<std::int32_t, 3> control_positions = {0, 1, 2};
        ninfer::DeviceBuffer control_positions_device(control_positions.size() *
                                                       sizeof(std::int32_t));
        control_positions_device.copy_from_host(control_positions.data(),
                                                control_positions_device.size());
        q3::PagedKVPublication mtp_control_publication;
        {
            auto transaction = mtp_cache.begin_device_append(
                mtp_control_allocation, mtp_control_publication,
                static_cast<const std::int32_t*>(control_positions_device.data()), 2U,
                device_transaction_workspace);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            require(transaction.commit() == 2,
                    "MTP alignment byte oracle did not publish two tokens");
        }
        require(copy_paged_allocation(mtp_control_allocation, mtp_cache.pool(), device.stream) ==
                    mtp_alignment_image,
                "MTP inactive alignment tail changed physical cache bytes");

        constexpr std::array<std::int32_t, 2> ar_positions = {2, -1};
        ninfer::DeviceBuffer ar_positions_device(ar_positions.size() * sizeof(std::int32_t));
        ar_positions_device.copy_from_host(ar_positions.data(), ar_positions_device.size());
        const std::int32_t one_dynamic_count = 1;
        dynamic_count_device.copy_from_host(&one_dynamic_count, sizeof(one_dynamic_count));
        synchronize(device.stream);
        {
            auto transaction = mtp_cache.begin_device_prefix_append(
                mtp_allocation, mtp_publication,
                static_cast<const std::int32_t*>(ar_positions_device.data()), 1U,
                static_cast<const std::int32_t*>(dynamic_count_device.data()),
                device_transaction_workspace);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()) + 2U * 256U,
                static_cast<const hip_bfloat16*>(decoder_values_device.data()) + 2U * 256U,
                device.stream);
            require(transaction.commit() == 3 && mtp_publication.valid_frontier == 3 &&
                        mtp_publication.healthy && !mtp_publication.transaction_open,
                    "active MTP AR append did not advance one token");
        }
        const auto mtp_active_ar_image =
            copy_paged_allocation(mtp_allocation, mtp_cache.pool(), device.stream);
        require(mtp_active_ar_image != mtp_alignment_image,
                "active MTP AR append did not change physical cache bytes");

        // Extend the control by the same third represented token. The exact image match proves
        // that the split alignment/AR lifecycle is physically identical to one contiguous append.
        {
            auto transaction = mtp_cache.begin_device_append(
                mtp_control_allocation, mtp_control_publication,
                static_cast<const std::int32_t*>(control_positions_device.data()) + 2, 1U,
                device_transaction_workspace);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()) + 2U * 256U,
                static_cast<const hip_bfloat16*>(decoder_values_device.data()) + 2U * 256U,
                device.stream);
            require(transaction.commit() == 3,
                    "MTP contiguous byte oracle did not publish its third token");
        }
        require(copy_paged_allocation(mtp_control_allocation, mtp_cache.pool(), device.stream) ==
                    mtp_active_ar_image,
                "split MTP alignment/AR writes differ from contiguous represented bytes");

        const std::int32_t zero_dynamic_count = 0;
        dynamic_count_device.copy_from_host(&zero_dynamic_count, sizeof(zero_dynamic_count));
        synchronize(device.stream);
        {
            auto transaction = mtp_cache.begin_device_prefix_append(
                mtp_allocation, mtp_publication,
                static_cast<const std::int32_t*>(ar_positions_device.data()) + 1, 1U,
                static_cast<const std::int32_t*>(dynamic_count_device.data()),
                device_transaction_workspace);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            require(transaction.commit() == 3 && mtp_publication.valid_frontier == 3 &&
                        mtp_publication.healthy && !mtp_publication.transaction_open,
                    "zero-count MTP AR transaction changed the published frontier");
        }
        require(copy_paged_allocation(mtp_allocation, mtp_cache.pool(), device.stream) ==
                    mtp_active_ar_image,
                "zero-count MTP AR transaction inspected its invalid position or source");

        const auto speculative_read =
            mtp_cache.published_layer_read(mtp_allocation, mtp_publication, 0);
        const std::uint64_t before_alignment_resolution =
            mtp_publication.transaction_generation;
        mtp_cache.truncate_publication(mtp_allocation, mtp_publication,
                                       licensed_alignment_frontier);
        require(mtp_publication.valid_frontier == licensed_alignment_frontier &&
                    mtp_publication.healthy && !mtp_publication.transaction_open &&
                    mtp_publication.transaction_generation == before_alignment_resolution + 1U &&
                    !speculative_read.valid_for(device.stream) &&
                    mtp_cache.published_layer_read(mtp_allocation, mtp_publication, 0)
                            .visible_frontier() == licensed_alignment_frontier,
                "MTP round resolution did not retain exactly the licensed alignment frontier");
        require(copy_paged_allocation(mtp_allocation, mtp_cache.pool(), device.stream) ==
                    mtp_active_ar_image,
                "MTP round resolution rewrote bytes while truncating publication");

        // A two-token speculative suffix can resolve partially, then a later suffix can be wholly
        // cancelled. Both paths are publication-only: retained represented bytes are exact and the
        // sequence stays healthy and reusable.
        constexpr std::array<std::int32_t, 2> partial_positions = {2, 3};
        ninfer::DeviceBuffer partial_positions_device(partial_positions.size() *
                                                       sizeof(std::int32_t));
        partial_positions_device.copy_from_host(partial_positions.data(),
                                                partial_positions_device.size());
        synchronize(device.stream);
        {
            auto transaction = mtp_cache.begin_device_append(
                mtp_allocation, mtp_publication,
                static_cast<const std::int32_t*>(partial_positions_device.data()),
                partial_positions.size(), device_transaction_workspace);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()) + 2U * 256U,
                static_cast<const hip_bfloat16*>(decoder_values_device.data()) + 2U * 256U,
                device.stream);
            require(transaction.commit() == 4,
                    "MTP partial-resolution setup did not publish its suffix");
        }
        const auto before_partial_truncate =
            copy_paged_allocation(mtp_allocation, mtp_cache.pool(), device.stream);
        mtp_cache.truncate_publication(mtp_allocation, mtp_publication, 3);
        require(mtp_publication.valid_frontier == 3 && mtp_publication.healthy &&
                    !mtp_publication.transaction_open &&
                    copy_paged_allocation(mtp_allocation, mtp_cache.pool(), device.stream) ==
                        before_partial_truncate,
                "MTP partial resolution did not retain exact physical bytes and frontier");

        constexpr std::array<std::int32_t, 2> cancelled_positions = {3, 4};
        ninfer::DeviceBuffer cancelled_positions_device(cancelled_positions.size() *
                                                         sizeof(std::int32_t));
        cancelled_positions_device.copy_from_host(cancelled_positions.data(),
                                                  cancelled_positions_device.size());
        synchronize(device.stream);
        const std::uint32_t cancellation_base = mtp_publication.valid_frontier;
        {
            auto transaction = mtp_cache.begin_device_append(
                mtp_allocation, mtp_publication,
                static_cast<const std::int32_t*>(cancelled_positions_device.data()),
                cancelled_positions.size(), device_transaction_workspace);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()) + 3U * 256U,
                static_cast<const hip_bfloat16*>(decoder_values_device.data()) + 3U * 256U,
                device.stream);
            require(transaction.commit() == 5,
                    "MTP cancellation setup did not publish its speculative suffix");
        }
        const auto before_cancel_truncate =
            copy_paged_allocation(mtp_allocation, mtp_cache.pool(), device.stream);
        mtp_cache.truncate_publication(mtp_allocation, mtp_publication, cancellation_base);
        require(mtp_publication.valid_frontier == cancellation_base &&
                    mtp_publication.healthy && !mtp_publication.transaction_open &&
                    copy_paged_allocation(mtp_allocation, mtp_cache.pool(), device.stream) ==
                        before_cancel_truncate,
                "MTP cancellation did not restore its base frontier without rewriting bytes");
        mtp_control_allocation.release();
        mtp_allocation.release();

        // A round-scoped segmented transaction performs alignment and every AR append under one
        // generation and one device cursor. It resolves status/cursor once, publishes only the
        // licensed alignment prefix, and records the identical fixed-address sequence in a HIP
        // graph. Exact page images provide the independent stored-byte oracle for eager/replay.
        auto segmented_allocation = mtp_cache.pool().reserve(1);
        auto segmented_control = mtp_cache.pool().reserve(1);
        segmented_allocation.materialize_tokens(8, device.stream);
        segmented_control.materialize_tokens(8, device.stream);
        segmented_allocation.bind_row(0, device.stream);
        segmented_control.bind_row(1, device.stream);
        mtp_cache.pool().zero_pages(segmented_allocation.page_ids(), device.stream);
        mtp_cache.pool().zero_pages(segmented_control.page_ids(), device.stream);
        constexpr std::array<std::int32_t, 4> segmented_positions = {0, 1, 2, -1};
        ninfer::DeviceBuffer segmented_positions_device(segmented_positions.size() *
                                                         sizeof(std::int32_t));
        segmented_positions_device.copy_from_host(segmented_positions.data(),
                                                  segmented_positions_device.size());
        ninfer::DeviceBuffer segmented_base_device(sizeof(std::int32_t));
        ninfer::DeviceBuffer segmented_count_device(3U * sizeof(std::int32_t));
        constexpr std::int32_t segmented_base = 0;
        constexpr std::array<std::int32_t, 3> segmented_counts = {2, 1, 0};
        segmented_base_device.copy_from_host(&segmented_base, sizeof(segmented_base));
        segmented_count_device.copy_from_host(segmented_counts.data(),
                                              segmented_count_device.size());
        ninfer::DeviceBuffer segmented_status_device(sizeof(std::uint32_t));
        ninfer::DeviceBuffer segmented_cursor_device(sizeof(std::uint32_t));
        ninfer::DeviceBuffer segmented_table_row_device(sizeof(std::int32_t));
        constexpr std::int32_t segmented_table_row = 0;
        segmented_table_row_device.copy_from_host(&segmented_table_row,
                                                  sizeof(segmented_table_row));
        const q3::PagedKVTransactionWorkspace segmented_workspace{
            .status = static_cast<std::uint32_t*>(segmented_status_device.data()),
            .cursor = static_cast<std::uint32_t*>(segmented_cursor_device.data()),
        };
        const auto launch_segmented_round = [&](q3::PagedKVTransaction& transaction) {
            const auto* positions =
                static_cast<const std::int32_t*>(segmented_positions_device.data());
            const auto* counts = static_cast<const std::int32_t*>(segmented_count_device.data());
            transaction.begin_device_segment(positions, 2U, counts);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            transaction.end_device_segment(device.stream);
            transaction.begin_device_segment(positions + 2, 1U, counts + 1);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()) + 2U * 256U,
                static_cast<const hip_bfloat16*>(decoder_values_device.data()) + 2U * 256U,
                device.stream);
            transaction.end_device_segment(device.stream);
            transaction.begin_device_segment(positions + 3, 1U, counts + 2);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            transaction.end_device_segment(device.stream);
        };

        q3::PagedKVPublication segmented_publication;
        {
            auto transaction = mtp_cache.begin_device_segmented_append(
                segmented_allocation, segmented_publication,
                static_cast<const std::int32_t*>(segmented_base_device.data()), 0U, 0U, 8U,
                segmented_workspace,
                static_cast<const std::int32_t*>(segmented_table_row_device.data()));
            launch_segmented_round(transaction);
            std::uint32_t status_word = 0;
            std::uint32_t cursor_word = 0;
            transaction.enqueue_segmented_resolution(&status_word, &cursor_word);
            synchronize(device.stream);
            require(status_word == 0U && cursor_word == 3U,
                    "segmented MTP eager round produced a wrong device resolution");
            transaction.finish_segmented_resolution(status_word, cursor_word, 2U);
        }
        require(segmented_publication.valid_frontier == 2U &&
                    segmented_publication.healthy && !segmented_publication.transaction_open,
                "segmented MTP eager round published a wrong licensed frontier");
        const auto segmented_eager_image =
            copy_paged_allocation(segmented_allocation, mtp_cache.pool(), device.stream);

        constexpr std::array<std::int32_t, 3> segmented_control_positions = {0, 1, 2};
        ninfer::DeviceBuffer segmented_control_positions_device(
            segmented_control_positions.size() * sizeof(std::int32_t));
        segmented_control_positions_device.copy_from_host(segmented_control_positions.data(),
                                                          segmented_control_positions_device.size());
        q3::PagedKVPublication segmented_control_publication;
        {
            auto transaction = mtp_cache.begin_device_append(
                segmented_control, segmented_control_publication,
                static_cast<const std::int32_t*>(segmented_control_positions_device.data()), 3U,
                device_transaction_workspace);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            require(transaction.commit() == 3U,
                    "segmented MTP byte oracle failed its contiguous append");
        }
        mtp_cache.truncate_publication(segmented_control, segmented_control_publication, 2U);
        require(copy_paged_allocation(segmented_control, mtp_cache.pool(), device.stream) ==
                    segmented_eager_image,
                "segmented MTP eager bytes differ from the contiguous codec oracle");

        mtp_cache.truncate_publication(segmented_allocation, segmented_publication, 0U);
        mtp_cache.pool().zero_pages(segmented_allocation.page_ids(), device.stream);
        synchronize(device.stream);
        ninfer::DecodeGraphDefinition segmented_graph;
        {
            auto captured = mtp_cache.begin_device_segmented_append(
                segmented_allocation, segmented_publication,
                static_cast<const std::int32_t*>(segmented_base_device.data()), 0U, 0U, 8U,
                segmented_workspace,
                static_cast<const std::int32_t*>(segmented_table_row_device.data()));
            segmented_graph.capture(device.stream, [&] { launch_segmented_round(captured); });
            captured.close_captured_segmented();
        }
        ninfer::DecodeGraphExecutable segmented_executable;
        segmented_executable.instantiate(segmented_graph);
        {
            auto replay = mtp_cache.begin_device_segmented_append(
                segmented_allocation, segmented_publication,
                static_cast<const std::int32_t*>(segmented_base_device.data()), 0U, 0U, 8U,
                segmented_workspace,
                static_cast<const std::int32_t*>(segmented_table_row_device.data()));
            replay.mark_graph_replay(device.stream);
            segmented_executable.launch(device.stream);
            std::uint32_t status_word = 0;
            std::uint32_t cursor_word = 0;
            replay.enqueue_segmented_resolution(&status_word, &cursor_word);
            synchronize(device.stream);
            require(status_word == 0U && cursor_word == 3U,
                    "segmented MTP Device Graph produced a wrong device resolution");
            replay.finish_segmented_resolution(status_word, cursor_word, 2U);
        }
        require(segmented_publication.valid_frontier == 2U &&
                    copy_paged_allocation(segmented_allocation, mtp_cache.pool(), device.stream) ==
                        segmented_eager_image,
                "segmented MTP Device Graph differs from eager state or exact cache bytes");

        // Ordinary decode uses one segment while the captured graph reads the live compact row
        // selector. Replay against a different allocation must mutate only that row and remain
        // byte-exact with an independently executed eager transaction.
        mtp_cache.truncate_publication(segmented_allocation, segmented_publication, 0U);
        mtp_cache.truncate_publication(segmented_control, segmented_control_publication, 0U);
        mtp_cache.pool().zero_pages(segmented_allocation.page_ids(), device.stream);
        mtp_cache.pool().zero_pages(segmented_control.page_ids(), device.stream);
        constexpr std::int32_t ordinary_capture_table_row = 0;
        segmented_table_row_device.copy_from_host(&ordinary_capture_table_row,
                                                  sizeof(ordinary_capture_table_row));
        synchronize(device.stream);
        const auto launch_ordinary_segment = [&](q3::PagedKVTransaction& transaction) {
            transaction.begin_device_segment(
                static_cast<const std::int32_t*>(segmented_positions_device.data()), 1U, nullptr);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            transaction.end_device_segment(device.stream);
        };
        {
            auto eager = mtp_cache.begin_device_segmented_append(
                segmented_allocation, segmented_publication,
                static_cast<const std::int32_t*>(segmented_base_device.data()), 0U, 0U, 8U,
                segmented_workspace,
                static_cast<const std::int32_t*>(segmented_table_row_device.data()));
            launch_ordinary_segment(eager);
            std::uint32_t status_word = 0;
            std::uint32_t cursor_word = 0;
            eager.enqueue_segmented_resolution(&status_word, &cursor_word);
            synchronize(device.stream);
            require(status_word == 0U && cursor_word == 1U,
                    "ordinary-style eager append produced a wrong device resolution");
            eager.finish_segmented_resolution(status_word, cursor_word, 1U);
        }
        const auto ordinary_eager_image =
            copy_paged_allocation(segmented_allocation, mtp_cache.pool(), device.stream);
        mtp_cache.truncate_publication(segmented_allocation, segmented_publication, 0U);
        mtp_cache.pool().zero_pages(segmented_allocation.page_ids(), device.stream);
        synchronize(device.stream);
        ninfer::DecodeGraphDefinition ordinary_graph;
        {
            auto captured = mtp_cache.begin_device_segmented_append(
                segmented_allocation, segmented_publication,
                static_cast<const std::int32_t*>(segmented_base_device.data()), 0U, 0U, 8U,
                segmented_workspace,
                static_cast<const std::int32_t*>(segmented_table_row_device.data()));
            ordinary_graph.capture(device.stream, [&] { launch_ordinary_segment(captured); });
            captured.close_captured_segmented();
        }
        ninfer::DecodeGraphExecutable ordinary_executable;
        ordinary_executable.instantiate(ordinary_graph);
        constexpr std::int32_t ordinary_replay_table_row = 1;
        segmented_table_row_device.copy_from_host(&ordinary_replay_table_row,
                                                  sizeof(ordinary_replay_table_row));
        synchronize(device.stream);
        {
            auto replay = mtp_cache.begin_device_segmented_append(
                segmented_control, segmented_control_publication,
                static_cast<const std::int32_t*>(segmented_base_device.data()), 0U, 0U, 8U,
                segmented_workspace,
                static_cast<const std::int32_t*>(segmented_table_row_device.data()));
            replay.mark_graph_replay(device.stream);
            ordinary_executable.launch(device.stream);
            std::uint32_t status_word = 0;
            std::uint32_t cursor_word = 0;
            replay.enqueue_segmented_resolution(&status_word, &cursor_word);
            synchronize(device.stream);
            require(status_word == 0U && cursor_word == 1U,
                    "ordinary-style graph replay produced a wrong device resolution");
            replay.finish_segmented_resolution(status_word, cursor_word, 1U);
        }
        const std::vector<std::byte> zero_segmented_image(ordinary_eager_image.size());
        require(segmented_control_publication.valid_frontier == 1U &&
                    copy_paged_allocation(segmented_control, mtp_cache.pool(), device.stream) ==
                        ordinary_eager_image &&
                    copy_paged_allocation(segmented_allocation, mtp_cache.pool(), device.stream) ==
                        zero_segmented_image,
                "ordinary-style graph replay ignored its live row or differs from eager bytes");
        segmented_control.release();
        segmented_allocation.release();

        auto invalid_count_allocation = mtp_cache.pool().reserve(1);
        invalid_count_allocation.materialize_tokens(4, device.stream);
        invalid_count_allocation.bind_row(1, device.stream);
        const std::int32_t invalid_dynamic_count = 5;
        dynamic_count_device.copy_from_host(&invalid_dynamic_count,
                                            sizeof(invalid_dynamic_count));
        synchronize(device.stream);
        q3::PagedKVPublication invalid_count_publication;
        bool rejected_dynamic_count = false;
        try {
            auto transaction = mtp_cache.begin_device_prefix_append(
                invalid_count_allocation, invalid_count_publication,
                static_cast<const std::int32_t*>(dynamic_positions_device.data()),
                dynamic_positions.size(),
                static_cast<const std::int32_t*>(dynamic_count_device.data()),
                device_transaction_workspace);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            (void)transaction.commit();
        } catch (const std::runtime_error&) {
            rejected_dynamic_count = true;
        }
        std::uint32_t invalid_count_status = 0;
        HIP_CHECK(hipMemcpy(&invalid_count_status, transaction_status.data(),
                            sizeof(invalid_count_status), hipMemcpyDeviceToHost));
        require(rejected_dynamic_count &&
                    (invalid_count_status &
                     ninfer::ops::r9700::kv::Fp8Int4KvAppendInvalidCount) != 0U &&
                    !invalid_count_publication.healthy &&
                    invalid_count_publication.valid_frontier == 0 &&
                    !invalid_count_publication.transaction_open,
                "invalid device-prefix count did not poison only its publication");
        invalid_count_allocation.release();

        // A negative I32 position reaches the raw device validator as 0xffffffff. Rejecting this
        // separate allocation proves both old-frontier preservation and publication-local poison:
        // the valid sequence above must remain healthy and readable at frontier 60.
        auto invalid_position_allocation = decoder.text_kv.pool().reserve(1);
        invalid_position_allocation.materialize_tokens(1, device.stream);
        invalid_position_allocation.bind_row(1, device.stream);
        synchronize(device.stream);
        q3::PagedKVPublication invalid_position_publication;
        constexpr std::array<std::int32_t, 1> negative_position = {-1};
        ninfer::DeviceBuffer negative_position_device(sizeof(std::int32_t));
        negative_position_device.copy_from_host(negative_position.data(),
                                                negative_position_device.size());
        bool rejected_device_position = false;
        try {
            auto transaction = decoder.text_kv.begin_device_append(
                invalid_position_allocation, invalid_position_publication,
                static_cast<const std::int32_t*>(negative_position_device.data()),
                negative_position.size(), device_transaction_workspace);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            transaction.launch_append_layer(
                1, static_cast<const hip_bfloat16*>(decoder_keys_device.data()),
                static_cast<const hip_bfloat16*>(decoder_values_device.data()), device.stream);
            (void)transaction.commit();
        } catch (const std::runtime_error&) {
            rejected_device_position = true;
        }
        std::uint32_t invalid_position_status = 0;
        HIP_CHECK(hipMemcpy(&invalid_position_status, transaction_status.data(),
                            sizeof(invalid_position_status), hipMemcpyDeviceToHost));
        require(rejected_device_position &&
                    (invalid_position_status &
                     ninfer::ops::r9700::kv::Fp8Int4KvAppendInvalidPosition) != 0U &&
                    !invalid_position_publication.healthy &&
                    invalid_position_publication.valid_frontier == 0 &&
                    !invalid_position_publication.transaction_open && publication.healthy &&
                    publication.valid_frontier == 60 &&
                    decoder.text_kv
                        .published_layer_read(decoder_allocation, publication, 0)
                        .valid_for(device.stream),
                "bad device positions changed a frontier or poisoned another publication");
        invalid_position_allocation.release();

        // A launched nonfinite mutation must leave the old frontier unchanged and poison only
        // this sequence publication record, making its partially written planes unreadable.
        const hip_bfloat16 nonfinite = hip_bfloat16(std::numeric_limits<float>::quiet_NaN());
        ninfer::DeviceBuffer nonfinite_keys(256 * sizeof(hip_bfloat16));
        ninfer::DeviceBuffer finite_values(256 * sizeof(hip_bfloat16));
        std::vector<hip_bfloat16> failed_keys(256, hip_bfloat16(0.25F));
        std::vector<hip_bfloat16> failed_values(256, hip_bfloat16(-0.5F));
        failed_keys[13] = nonfinite;
        nonfinite_keys.copy_from_host(failed_keys.data(), nonfinite_keys.size());
        finite_values.copy_from_host(failed_values.data(), finite_values.size());
        bool rejected_nonfinite = false;
        try {
            constexpr std::array<std::uint32_t, 1> next = {60};
            auto transaction = decoder.text_kv.begin_append(
                decoder_allocation, publication, next, transaction_workspace);
            transaction.launch_append_layer(
                0, static_cast<const hip_bfloat16*>(nonfinite_keys.data()),
                static_cast<const hip_bfloat16*>(finite_values.data()), device.stream);
            transaction.launch_append_layer(
                1, static_cast<const hip_bfloat16*>(nonfinite_keys.data()),
                static_cast<const hip_bfloat16*>(finite_values.data()), device.stream);
            (void)transaction.commit();
        } catch (const std::runtime_error&) {
            rejected_nonfinite = true;
        }
        require(rejected_nonfinite && !publication.healthy &&
                    publication.valid_frontier == 60 && !publication.transaction_open,
                "failed Qwen append published or failed to poison its sequence state");
        bool rejected_poisoned = false;
        try {
            constexpr std::array<std::uint32_t, 1> next = {60};
            (void)decoder.text_kv.begin_append(
                decoder_allocation, publication, next, transaction_workspace);
        } catch (const std::logic_error&) {
            rejected_poisoned = true;
        }
        require(rejected_poisoned,
                "Qwen decoder cache accepted a transaction after sequence poisoning");

        ninfer::LayoutBuilder cyclic_builder;
        const auto cyclic_layout = ninfer::plan_cyclic_kv_cache(cyclic_builder, 2, 5, 1, 4, 2);
        ninfer::DeviceBuffer cyclic_backing(cyclic_builder.finish(256, "cyclic state"));
        ninfer::CyclicKVCache cyclic({cyclic_backing.data(), cyclic_backing.size()}, cyclic_layout);
        for (std::uint32_t layer = 0; layer < cyclic.layer_count(); ++layer) {
            const auto view = cyclic.layer_view(layer);
            HIP_CHECK(hipMemsetAsync(view.k.data, 0x31 + static_cast<int>(layer), view.k.bytes(),
                                     device.stream));
            HIP_CHECK(hipMemsetAsync(view.v.data, 0x51 + static_cast<int>(layer), view.v.bytes(),
                                     device.stream));
        }
        cyclic.copy_lane_from(cyclic, 0, 1, device.stream);
        const auto cyclic_source = copy_cyclic_lane(cyclic, 0, device.stream);
        const auto cyclic_copy = copy_cyclic_lane(cyclic, 1, device.stream);
        require(cyclic_copy == cyclic_source, "cyclic BF16 D2D lane copy changed bytes");
        std::vector<std::byte> cyclic_replacement(cyclic.lane_host_bytes(), std::byte{0x7a});
        cyclic.copy_lane_from_host(cyclic_replacement.data(), 1, device.stream);
        require(copy_cyclic_lane(cyclic, 1, device.stream) == cyclic_replacement,
                "cyclic BF16 host spill/restore changed bytes");

        ninfer::LayoutBuilder linear_builder;
        const ninfer::LinearAttentionStatePoolSpec linear_spec{
            .layers = 2,
            .conv_channels = 4,
            .conv_width = 3,
            .value_heads = 2,
            .value_head_dim = 4,
            .key_head_dim = 3,
            .slot_count = 2,
            .conv_dtype = ninfer::DType::BF16,
        };
        const auto linear_layout =
            ninfer::plan_linear_attention_state_pool(linear_builder, linear_spec);
        ninfer::DeviceBuffer linear_backing(linear_builder.finish(256, "linear state"));
        ninfer::LinearAttentionStatePool linear({linear_backing.data(), linear_backing.size()},
                                                linear_layout);
        std::vector<std::byte> conv(linear.conv_host_image_bytes(), std::byte{0x3c});
        std::vector<std::byte> recurrent(linear.recurrent_host_image_bytes(), std::byte{0x5d});
        linear.unpack_slot_from_host(0, conv.data(), recurrent.data(), device.stream);
        linear.copy_slot(0, 1, device.stream);
        std::vector<std::byte> conv_out(conv.size());
        std::vector<std::byte> recurrent_out(recurrent.size());
        linear.pack_slot_to_host(1, conv_out.data(), recurrent_out.data(), device.stream);
        synchronize(device.stream);
        require(conv_out == conv && recurrent_out == recurrent,
                "BF16/FP32 linear-attention state copy/spill changed bytes");
        linear.zero_slot(1, device.stream);
        linear.pack_slot_to_host(1, conv_out.data(), recurrent_out.data(), device.stream);
        synchronize(device.stream);
        require(std::all_of(conv_out.begin(), conv_out.end(),
                            [](std::byte value) { return value == std::byte{0}; }) &&
                    std::all_of(recurrent_out.begin(), recurrent_out.end(),
                                [](std::byte value) { return value == std::byte{0}; }),
                "linear-attention state zero did not retain ordered HIP semantics");

        std::cout << "hip_persistent_state: PASS pages=" << paged_image_bytes
                  << " cyclic_bytes=" << cyclic.lane_host_bytes()
                  << " linear_bytes=" << conv.size() + recurrent.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hip_persistent_state: FAIL: " << error.what() << '\n';
        return 1;
    }
}
