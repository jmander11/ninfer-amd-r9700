#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/fp8_int4_paged_kv_cache.h"
#include "core/linear_attention_state.h"
#include "targets/qwen3/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3/impl/runtime/prefix_identity.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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

void paint_allocation(const ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                      std::uint8_t seed, hipStream_t stream) {
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
        HIP_CHECK(
            hipMemcpyAsync(plane.data, host.data(), host.size(), hipMemcpyHostToDevice, stream));
    }
}

void verify_allocation(const ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                       std::uint8_t seed, const char* label) {
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
    return std::all_of(host.begin(), host.end(),
                       [value](std::uint8_t byte) { return byte == value; });
}

// Blocks a stream behind a host callback until released, so copies enqueued after it stay pending.
struct StreamGate {
    std::mutex mutex;
    std::condition_variable cv;
    bool released = false;
    bool finished = false;
    bool launched = false;

    static void callback(void* pointer) {
        auto& gate = *static_cast<StreamGate*>(pointer);
        std::unique_lock lock(gate.mutex);
        gate.cv.wait(lock, [&] { return gate.released; });
        gate.finished = true;
        gate.cv.notify_all();
    }

    void launch(hipStream_t stream) {
        HIP_CHECK(hipLaunchHostFunc(stream, callback, this));
        launched = true;
    }

    void release() {
        std::lock_guard lock(mutex);
        released = true;
        cv.notify_all();
    }

    ~StreamGate() {
        release();
        if (launched) {
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return finished; });
        }
    }
};

// One-page Text pool plus a retained-lane capture source with a tail-hidden image.
struct CopyFixture {
    ninfer::Fp8KInt4VPagedKVSpec spec;
    ninfer::Fp8KInt4VSemanticFingerprint semantics;
    ninfer::Fp8KInt4VPagedKVPoolLayout layout;
    std::unique_ptr<ninfer::DeviceBuffer> backing;
    std::unique_ptr<ninfer::PagedKVPool> pool;

    explicit CopyFixture(const ninfer::Fp8KInt4VPagedKVSpec& text_spec) : spec(text_spec) {
        semantics = ninfer::fp8_k_int4_v_semantic_fingerprint(spec);
        ninfer::LayoutBuilder builder;
        layout  = ninfer::plan_fp8_k_int4_v_paged_kv_pool(builder, spec);
        backing = std::make_unique<ninfer::DeviceBuffer>(builder.finish(256));
        pool    = std::make_unique<ninfer::PagedKVPool>(
            ninfer::DeviceSpan{backing->data(), backing->size()}, layout.storage);
    }
};

struct HiddenCapture {
    q3::PreparedPromptData prompt;
    q3::detail::ResidentPrefixIdentity identity;
    q3::detail::RamCaptureSource source;

    HiddenCapture(CopyFixture& fixture, const ninfer::PagedKVAllocation& allocation,
                  std::int32_t first_token, const ninfer::Tensor& hidden, hipStream_t stream)
        : prompt(prompt_data(first_token)) {
        identity.assign(prompt);
        source.execution_frontier = 6;
        source.ledger_frontier    = 7;
        source.text_kv_valid      = 6;
        source.tail_hidden_valid  = true;
        source.ledger             = prompt.token_ids;
        source.identity           = &identity;
        source.hash_f             = q3::detail::prefix_hash_at(prompt.token_ids, identity, 6);
        source.text               = &allocation;
        source.text_pool          = fixture.pool.get();
        source.text_semantics     = fixture.semantics;
        source.tail_hidden        = &hidden;
        source.stream             = stream;
    }

    HiddenCapture(const HiddenCapture&)            = delete;
    HiddenCapture& operator=(const HiddenCapture&) = delete;
};

// Two restores in one round each own an in-flight H2D. Consuming the first must account its copy
// as a load, not orphan it as save time.
void two_pending_loads_fold_as_loads(ninfer::DeviceContext& device,
                                     const ninfer::Fp8KInt4VPagedKVSpec& text_spec) {
    CopyFixture fixture(text_spec);
    constexpr std::size_t kBulkBytes = 8ULL << 20;
    ninfer::DeviceBuffer bulk(kBulkBytes);
    bulk.fill(0x3c);
    ninfer::Tensor hidden(bulk.data(), ninfer::DType::U8, {static_cast<std::int32_t>(kBulkBytes)});
    q3::detail::KVRamCache cache(64ULL << 20);
    std::vector<std::uint64_t> ids;
    for (const std::int32_t first : {701, 801}) {
        auto source = fixture.pool->reserve(1);
        source.materialize_pages(1, device.stream);
        device.order_copy_after_compute();
        HiddenCapture capture(fixture, source, first, hidden, device.copy_stream);
        const auto result = cache.capture(capture.source);
        device.synchronize_all();
        source.release();
        require(result.status == q3::detail::RamCaptureStatus::Captured, "two-load capture failed");
        ids.push_back(result.entry_id);
    }
    (void)cache.harvest_copy_seconds();
    auto dest_a = fixture.pool->reserve(1);
    auto dest_b = fixture.pool->reserve(1);
    dest_a.materialize_pages(1, device.stream);
    dest_b.materialize_pages(1, device.stream);
    HIP_CHECK(hipStreamSynchronize(device.stream));
    ninfer::DeviceBuffer out_a(kBulkBytes);
    ninfer::DeviceBuffer out_b(kBulkBytes);
    ninfer::Tensor out_a_t(out_a.data(), ninfer::DType::U8,
                           {static_cast<std::int32_t>(kBulkBytes)});
    ninfer::Tensor out_b_t(out_b.data(), ninfer::DType::U8,
                           {static_cast<std::int32_t>(kBulkBytes)});
    const auto restore = [&](std::uint64_t id, ninfer::PagedKVAllocation& dest,
                             ninfer::Tensor& out) {
        q3::detail::RamRestoreTarget target;
        target.text           = &dest;
        target.text_pool      = fixture.pool.get();
        target.text_semantics = fixture.semantics;
        target.text_dst_pages = 1;
        target.tail_hidden    = &out;
        target.stream         = device.copy_stream;
        cache.claim(id);
        (void)cache.unpack_device(id, target);
    };
    restore(ids[0], dest_a, out_a_t);
    restore(ids[1], dest_b, out_b_t);
    cache.consume(ids[0]);
    cache.consume(ids[1]);
    const auto loaded = cache.harvest_copy_seconds();
    device.synchronize_all();
    dest_a.release();
    dest_b.release();
    if (loaded.save != 0.0 || loaded.load <= 0.0) {
        std::cerr << "two pending loads harvested save=" << loaded.save << " load=" << loaded.load
                  << '\n';
    }
    require(loaded.save == 0.0 && loaded.load > 0.0,
            "a second restore orphaned the first restore's load as save time");
}

// A stream-ordered wait must order the consumer stream without blocking the host.
void stream_wait_does_not_block_host(ninfer::DeviceContext& device,
                                     const ninfer::Fp8KInt4VPagedKVSpec& text_spec) {
    CopyFixture fixture(text_spec);
    q3::detail::KVRamCache cache(64ULL << 20);
    auto source = fixture.pool->reserve(1);
    source.materialize_pages(1, device.stream);
    HIP_CHECK(hipStreamSynchronize(device.stream));
    ninfer::DeviceBuffer bulk(1ULL << 20);
    ninfer::Tensor hidden(bulk.data(), ninfer::DType::U8, {1 << 20});
    bool captured = false;
    bool prompt   = false;
    {
        StreamGate gate;
        gate.launch(device.copy_stream);
        HiddenCapture capture(fixture, source, 901, hidden, device.copy_stream);
        captured = cache.capture(capture.source).status == q3::detail::RamCaptureStatus::Captured;
        if (captured) {
            std::atomic<bool> returned{false};
            std::thread waiter([&] {
                try {
                    cache.wait_pending_copies_on_stream(device.stream);
                } catch (...) {}
                returned.store(true);
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
            while (!returned.load() && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            prompt = returned.load();
            gate.release();
            waiter.join();
        }
    }
    device.synchronize_all();
    source.release();
    require(captured, "stream-wait capture failed");
    require(prompt, "stream-ordered RAM wait blocked the host on a gated copy");
}

// A restore that fails validation after enqueueing H2D reads must still fence the record;
// otherwise its host block looks idle while a copy reads it.
void failed_unpack_fences_enqueued_reads(ninfer::DeviceContext& device,
                                         const ninfer::Fp8KInt4VPagedKVSpec& text_spec) {
    CopyFixture fixture(text_spec);
    q3::detail::KVRamCache cache(64ULL << 20);
    ninfer::DeviceBuffer bulk(1ULL << 20);
    ninfer::Tensor hidden(bulk.data(), ninfer::DType::U8, {1 << 20});
    std::uint64_t entry_id = 0;
    {
        auto source = fixture.pool->reserve(1);
        source.materialize_pages(1, device.stream);
        HIP_CHECK(hipStreamSynchronize(device.stream));
        HiddenCapture capture(fixture, source, 1001, hidden, device.copy_stream);
        const auto result = cache.capture(capture.source);
        device.synchronize_all();
        source.release();
        require(result.status == q3::detail::RamCaptureStatus::Captured,
                "failed-unpack capture failed");
        entry_id = result.entry_id;
    }
    auto dest = fixture.pool->reserve(1);
    dest.materialize_pages(1, device.stream);
    HIP_CHECK(hipStreamSynchronize(device.stream));
    q3::detail::RamRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = fixture.pool.get();
    target.text_semantics = fixture.semantics;
    target.text_dst_pages = 1;
    target.stream         = device.copy_stream;
    // A context-checkpoint restore naming a frontier with no stored head fails after the text
    // pages are already enqueued.
    target.reuse      = ninfer::PrefixReusePath::RestoreContextCheckpoint;
    target.reuse_base = 3;
    cache.claim(entry_id);
    bool threw  = false;
    bool fenced = false;
    {
        StreamGate gate;
        gate.launch(device.copy_stream);
        try {
            (void)cache.unpack_device(entry_id, target);
        } catch (const std::logic_error&) { threw = true; }
        fenced = threw && !cache.copies_ready(entry_id);
        gate.release();
    }
    device.synchronize_all();
    cache.release(entry_id);
    dest.release();
    require(threw, "failed-unpack fixture did not reject the missing head");
    require(fenced, "a failed restore left its enqueued H2D reads unfenced");
}

// Rewrite checkpoints live in lane-owned pinned host images. The RAM tier copies those images
// host-to-host on capture and restore, beside the device GDN/DFlash current state. Exact bytes.
void rewrite_images_round_trip(ninfer::DeviceContext& device,
                               const ninfer::Fp8KInt4VPagedKVSpec& text_spec) {
    CopyFixture fixture(text_spec);
    ninfer::LayoutBuilder state_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        state_builder, ninfer::LinearAttentionStatePoolSpec{.layers         = 3,
                                                            .conv_channels  = 40,
                                                            .conv_width     = 3,
                                                            .value_heads    = 2,
                                                            .value_head_dim = 8,
                                                            .key_head_dim   = 8,
                                                            .slot_count     = 2});
    const auto cyclic_layout = ninfer::plan_cyclic_kv_cache(state_builder, 2, 128, 2, 16, 2);
    ninfer::DeviceBuffer state_backing(state_builder.finish(256));
    const ninfer::DeviceSpan state_span{state_backing.data(), state_backing.size()};
    ninfer::LinearAttentionStatePool gdn(state_span, gdn_layout);
    ninfer::CyclicKVCache cyclic(state_span, cyclic_layout);
    constexpr std::int32_t lane = 1;

    const auto pattern = [](std::size_t bytes, std::uint32_t seed) {
        std::vector<std::uint8_t> out(bytes);
        for (std::uint8_t& byte : out) {
            seed = seed * 1664525U + 1013904223U;
            byte = static_cast<std::uint8_t>(seed >> 24);
        }
        return out;
    };
    const std::size_t conv_bytes   = gdn.conv_host_image_bytes();
    const std::size_t rec_bytes    = gdn.recurrent_host_image_bytes();
    const std::size_t cyclic_bytes = cyclic.lane_host_bytes();
    const auto current_conv        = pattern(conv_bytes, 11);
    const auto current_rec         = pattern(rec_bytes, 12);
    const auto current_cyclic      = pattern(cyclic_bytes, 13);
    const auto rewrite_conv        = pattern(conv_bytes, 21);
    const auto rewrite_rec         = pattern(rec_bytes, 22);
    const auto rewrite_cyclic      = pattern(cyclic_bytes, 23);
    gdn.unpack_slot_from_host(lane, current_conv.data(), current_rec.data(), device.stream);
    cyclic.copy_lane_from_host(current_cyclic.data(), lane, device.stream);
    ninfer::DeviceBuffer hidden_bytes(2 * 1024);
    hidden_bytes.fill(0x5a);
    ninfer::Tensor tail_hidden(hidden_bytes.data(), ninfer::DType::U8, {1024});
    ninfer::Tensor rewrite_hidden(static_cast<std::uint8_t*>(hidden_bytes.data()) + 1024,
                                  ninfer::DType::U8, {1024});

    q3::detail::KVRamCache cache(64ULL << 20);
    std::uint64_t entry_id = 0;
    {
        auto source = fixture.pool->reserve(1);
        source.materialize_pages(1, device.stream);
        device.order_copy_after_compute();
        HiddenCapture capture(fixture, source, 1101, tail_hidden, device.copy_stream);
        capture.source.gdn                       = &gdn;
        capture.source.gdn_current_slot          = lane;
        capture.source.dflash_local              = &cyclic;
        capture.source.dflash_lane               = lane;
        capture.source.rewrite_valid             = true;
        capture.source.rewrite_frontier          = 4;
        capture.source.rewrite_checkpoint_hidden = &rewrite_hidden;
        capture.source.rewrite_state             = {rewrite_conv.data(), rewrite_rec.data(),
                                                    rewrite_cyclic.data()};
        const auto result                        = cache.capture(capture.source);
        device.synchronize_all();
        source.release();
        require(result.status == q3::detail::RamCaptureStatus::Captured,
                "rewrite-image capture failed");
        entry_id = result.entry_id;
    }

    gdn.zero_slot(lane, device.stream);
    std::vector<std::uint8_t> zero_cyclic(cyclic_bytes, 0);
    cyclic.copy_lane_from_host(zero_cyclic.data(), lane, device.stream);
    auto dest = fixture.pool->reserve(1);
    dest.materialize_pages(1, device.stream);
    HIP_CHECK(hipStreamSynchronize(device.stream));
    std::vector<std::uint8_t> out_conv(conv_bytes, 0);
    std::vector<std::uint8_t> out_rec(rec_bytes, 0);
    std::vector<std::uint8_t> out_cyclic(cyclic_bytes, 0);
    q3::detail::RamRestoreTarget target;
    target.text                      = &dest;
    target.text_pool                 = fixture.pool.get();
    target.text_semantics            = fixture.semantics;
    target.text_dst_pages            = 1;
    target.gdn                       = &gdn;
    target.gdn_current_slot          = lane;
    target.rewrite_state             = {out_conv.data(), out_rec.data(), out_cyclic.data()};
    target.tail_hidden               = &tail_hidden;
    target.rewrite_checkpoint_hidden = &rewrite_hidden;
    target.dflash_local              = &cyclic;
    target.dflash_lane               = lane;
    target.stream                    = device.copy_stream;
    cache.claim(entry_id);
    const q3::detail::RamRestoredHost restored = cache.unpack_device(entry_id, target);
    cache.consume(entry_id);
    device.synchronize_all();
    std::vector<std::uint8_t> back_conv(conv_bytes);
    std::vector<std::uint8_t> back_rec(rec_bytes);
    std::vector<std::uint8_t> back_cyclic(cyclic_bytes);
    gdn.pack_slot_to_host(lane, back_conv.data(), back_rec.data(), device.stream);
    cyclic.copy_lane_to_host(lane, back_cyclic.data(), device.stream);
    HIP_CHECK(hipStreamSynchronize(device.stream));
    dest.release();
    require(restored.rewrite_valid && restored.rewrite_frontier == 4,
            "RAM restore lost the rewrite checkpoint metadata");
    require(out_conv == rewrite_conv && out_rec == rewrite_rec && out_cyclic == rewrite_cyclic,
            "RAM restore did not reproduce the rewrite host images byte-exactly");
    require(back_conv == current_conv && back_rec == current_rec && back_cyclic == current_cyclic,
            "RAM restore did not reproduce the current GDN/DFlash lane byte-exactly");
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(0);
        require(device.is_gfx1201() && device.props.warpSize == 32,
                "KV RAM qualifier requires wave32 gfx1201");

        const ninfer::Fp8KInt4VPagedKVSpec text_spec{
            .page_group_count      = 10,
            .logical_page_capacity = 8,
            .table_rows            = 4,
            .layer_count           = 2,
            .head_dim              = 256,
            .num_kv_heads          = 4,
            .value_group           = 16,
            .plane_layouts = {.key         = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                              .value       = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                              .value_scale = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor},
        };
        const ninfer::Fp8KInt4VPagedKVSpec mtp_spec{
            .page_group_count      = 8,
            .logical_page_capacity = 8,
            .table_rows            = 4,
            .layer_count           = 1,
            .head_dim              = 256,
            .num_kv_heads          = 4,
            .value_group           = 16,
            .plane_layouts = {.key         = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                              .value       = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                              .value_scale = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor},
        };
        const auto text_semantics = ninfer::fp8_k_int4_v_semantic_fingerprint(text_spec);
        const auto mtp_semantics  = ninfer::fp8_k_int4_v_semantic_fingerprint(mtp_spec);

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
        ninfer::PagedKVPool mtp_source_pool({mtp_source_backing.data(), mtp_source_backing.size()},
                                            mtp_source_layout.storage);
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
        source.ledger_frontier    = 7;
        source.text_kv_valid      = 6;
        source.mtp_kv_valid       = 5;
        source.ledger             = prompt.token_ids;
        source.identity           = &identity;
        source.hash_f             = q3::detail::prefix_hash_at(prompt.token_ids, identity, 6);
        source.text               = &text_source;
        source.text_pool          = &text_source_pool;
        source.text_semantics     = text_semantics;
        source.backend            = &mtp_source;
        source.backend_pool       = &mtp_source_pool;
        source.backend_semantics  = mtp_semantics;
        source.stream             = device.copy_stream;

        q3::detail::KVRamCache cache(8U << 20);
        require(cache.capture(source).status == q3::detail::RamCaptureStatus::Captured,
                "Text+MTP KV RAM capture failed");
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
        target.text_dst_pages    = 3;
        target.backend_dst_pages = 3;
        target.text              = &text_dest;
        target.text_pool         = &text_dest_pool;
        target.text_semantics    = text_semantics;
        target.backend           = &mtp_dest;
        target.backend_pool      = &mtp_dest_pool;
        target.backend_semantics = mtp_semantics;
        target.stream            = device.stream;

        auto wrong                       = target;
        wrong.text_semantics.value_group = 32;
        bool rejected_mismatch           = false;
        try {
            (void)cache.unpack_device(match->entry_id, wrong);
        } catch (const std::invalid_argument&) { rejected_mismatch = true; }
        HIP_CHECK(hipStreamSynchronize(device.stream));
        require(rejected_mismatch, "semantic fingerprint mismatch was accepted");
        require(all_bytes_equal(text_dest_backing, 0xA5) && all_bytes_equal(mtp_dest_backing, 0xA5),
                "semantic mismatch modified a destination byte");

        auto wrong_backend                           = target;
        wrong_backend.backend_semantics->value_group = 32;
        rejected_mismatch                            = false;
        try {
            (void)cache.unpack_device(match->entry_id, wrong_backend);
        } catch (const std::invalid_argument&) { rejected_mismatch = true; }
        HIP_CHECK(hipStreamSynchronize(device.stream));
        require(rejected_mismatch, "backend semantic fingerprint mismatch was accepted");
        require(all_bytes_equal(text_dest_backing, 0xA5) && all_bytes_equal(mtp_dest_backing, 0xA5),
                "backend semantic mismatch modified a destination byte");

        const q3::detail::RamRestoredHost restored = cache.unpack_device(match->entry_id, target);
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
        require(eviction_cache.capture(source).status == q3::detail::RamCaptureStatus::Captured,
                "eviction probe first capture failed");
        require(eviction_cache.capture(source).status ==
                    q3::detail::RamCaptureStatus::NeedsEviction,
                "full RAM cache did not request eviction");
        const auto victim = eviction_cache.peek_oldest_unpinned();
        require(victim && eviction_cache.evict_one_unpinned(*victim),
                "eviction probe could not release the oldest entry");
        require(eviction_cache.capture(source).status == q3::detail::RamCaptureStatus::Captured,
                "eviction probe replacement capture failed");
        const auto eviction = eviction_cache.snapshot();
        require(eviction.entry_count == 1 && eviction.captures == 2 && eviction.evictions == 1,
                "unpinned FIFO entry was not evicted for the replacement image");

        two_pending_loads_fold_as_loads(device, text_spec);
        stream_wait_does_not_block_host(device, text_spec);
        failed_unpack_fences_enqueued_reads(device, text_spec);
        rewrite_images_round_trip(device, text_spec);

        std::cout << "kv_ram_cache: PASS text_planes=" << text_source_pool.plane_count()
                  << " mtp_planes=" << mtp_source_pool.plane_count()
                  << " entry_bytes=" << entry_bytes << " evictions=" << eviction.evictions << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "kv_ram_cache: FAIL: " << error.what() << '\n';
        return 1;
    }
}
