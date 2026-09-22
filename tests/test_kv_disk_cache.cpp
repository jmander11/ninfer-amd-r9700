#include "core/device.h"
#include "core/fp8_int4_paged_kv_cache.h"
#include "runtime/contract/types.h"
#include "targets/qwen3/impl/runtime/kv_disk_cache.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
namespace q3 = ninfer::targets::qwen3;
namespace cache = q3::detail;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        std::array<char, 40> pattern{};
        std::strcpy(pattern.data(), "/tmp/ninfer-fixed-kv-XXXXXX");
        const char* created = ::mkdtemp(pattern.data());
        if (!created) { throw std::runtime_error("mkdtemp failed"); }
        path = created;
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

struct Pool {
    ninfer::Fp8KInt4VPagedKVSpec spec;
    ninfer::Fp8KInt4VPagedKVPoolLayout layout;
    std::unique_ptr<ninfer::DeviceArena> arena;
    std::unique_ptr<ninfer::PagedKVPool> storage;
    explicit Pool(std::uint32_t pages) {
        spec = {.page_group_count = pages, .logical_page_capacity = 4, .table_rows = 2,
                .layer_count = 2, .head_dim = 128, .num_kv_heads = 2, .value_group = 32,
                .plane_layouts = {
                    .key = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                    .value = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                    .value_scale = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor}};
        ninfer::LayoutBuilder builder;
        layout = ninfer::plan_fp8_k_int4_v_paged_kv_pool(builder, spec);
        arena = std::make_unique<ninfer::DeviceArena>(builder.finish(256));
        storage = std::make_unique<ninfer::PagedKVPool>(
            ninfer::DeviceSpan{arena->base(), arena->capacity()}, layout.storage);
    }
    auto semantics() const { return ninfer::fp8_k_int4_v_semantic_fingerprint(spec); }
};

q3::PreparedPromptData prompt(std::size_t tokens = 130) {
    q3::PreparedPromptData result;
    result.token_ids.resize(tokens);
    result.token_types.assign(tokens, 0);
    result.positions.resize(tokens * 3);
    for (std::size_t i = 0; i < tokens; ++i) {
        result.token_ids[i] = static_cast<ninfer::TokenId>(100 + i);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            result.positions[axis * tokens + i] = static_cast<std::int32_t>(i);
        }
    }
    q3::VisionItem image;
    image.content_digest[0] = 42;
    image.token_spans.push_back({8, 8});
    result.vision_items.push_back(image);
    std::fill(result.token_types.begin() + 8, result.token_types.begin() + 16, 1);
    return result;
}

cache::DiskOpenConfig config(const std::filesystem::path& path, Pool& pool,
                              cache::KVRamCache& ram, ninfer::KvDiskCompress compression) {
    cache::DiskOpenConfig result;
    result.location = path;
    result.capacity_bytes = 32ULL << 20;
    result.compress = compression;
    result.max_context = 256;
    result.ram = &ram;
    result.text_pool = pool.storage.get();
    result.logical_page_bytes = ninfer::paged_kv_logical_page_bytes(*pool.storage);
    result.gdn_staging_bytes = 256;
    result.fingerprint = cache::make_disk_fingerprint(
        "qwen3.8-27b", "r9700-int-candidate", "fixed-cache-test-artifact",
        ninfer::SpeculativeBackend::None, *pool.storage, nullptr, pool.semantics(),
        std::nullopt, nullptr, nullptr);
    return result;
}

void verify_crc() {
    std::vector<std::uint8_t> bytes(8193);
    for (std::size_t i = 0; i < bytes.size(); ++i) { bytes[i] = static_cast<std::uint8_t>(i * 73); }
    for (std::size_t length : {0U, 9U, 257U, 8193U}) {
        std::uint32_t oracle = ~0U;
        for (std::size_t i = 0; i < length; ++i) {
            oracle ^= bytes[i];
            for (int bit = 0; bit < 8; ++bit) {
                oracle = (oracle >> 1) ^ ((oracle & 1U) ? 0x82f63b78U : 0U);
            }
        }
        require(cache::KVDiskCache::test_crc32c(std::span(bytes).first(length)) == ~oracle,
                "CRC32C differs from independent bitwise oracle");
    }
}

void run_roundtrip(ninfer::DeviceContext& device, ninfer::KvDiskCompress compression) {
    TemporaryDirectory directory;
    Pool source_pool(8);
    cache::KVRamCache ram(8ULL << 20);
    auto allocation = source_pool.storage->reserve(3);
    allocation.materialize_pages(3, device.stream);
    const auto page_bytes = ninfer::paged_kv_logical_page_bytes(*source_pool.storage);
    ninfer::PinnedHostBuffer page(page_bytes);
    std::vector<std::vector<std::uint8_t>> expected(3, std::vector<std::uint8_t>(page_bytes));
    for (std::size_t index = 0; index < expected.size(); ++index) {
        for (std::size_t offset = 0; offset < page_bytes; ++offset) {
            expected[index][offset] = static_cast<std::uint8_t>((offset / 64 + index * 29) % 251);
        }
        std::memcpy(page.data(), expected[index].data(), page_bytes);
        ninfer::unpack_paged_kv_logical_page_from_host(allocation, *source_pool.storage,
            page.data(), static_cast<std::uint32_t>(index), device.copy_stream);
        device.synchronize_all();
    }

    auto retained = prompt();
    cache::ResidentPrefixIdentity identity;
    identity.assign(retained);
    cache::RamCaptureSource capture;
    capture.execution_frontier = 129;
    capture.ledger_frontier = 130;
    capture.text_kv_valid = 129;
    capture.tail_hidden_valid = false;
    capture.ledger = retained.token_ids;
    capture.identity = &identity;
    capture.hash_f = cache::prefix_hash_at(retained.token_ids, identity, 129);
    capture.text = &allocation;
    capture.text_pool = source_pool.storage.get();
    capture.text_semantics = source_pool.semantics();
    capture.stream = device.copy_stream;

    ram.test_fail_next_capture_metadata_allocation();
    require(ram.capture(capture).status == cache::RamCaptureStatus::Dropped,
            "optional RAM metadata allocation failure did not drop capture");
    const auto saved = ram.capture(capture);
    require(saved.status == cache::RamCaptureStatus::Captured, "RAM capture failed");
    ram.wait_pending_copies();
    const auto image = ram.host_kv(saved.entry_id);
    for (std::uint32_t index = 0; index < 3; ++index) {
        ninfer::gather_logical_page_from_host_image(image.text, *source_pool.storage,
                                                   3, index, page.data());
        require(std::memcmp(page.data(), expected[index].data(), page_bytes) == 0,
                "RAM logical page gather changed fixed-codec bytes");
    }
    {
        auto ram_destination = source_pool.storage->reserve(3);
        ram_destination.materialize_pages(3, device.stream);
        device.synchronize_all();
        cache::RamRestoreTarget target;
        target.text = &ram_destination;
        target.text_pool = source_pool.storage.get();
        target.text_semantics = source_pool.semantics();
        target.text_dst_pages = 3;
        target.stream = device.copy_stream;
        ram.claim(saved.entry_id);
        const auto restored = ram.unpack_device(saved.entry_id, target);
        ram.wait_pending_copies();
        require(restored.ledger == retained.token_ids && restored.identity.matches(retained, 129),
                "RAM restore changed represented token/vision identity");
        for (std::uint32_t index = 0; index < 3; ++index) {
            ninfer::pack_paged_kv_logical_page_to_host(ram_destination, *source_pool.storage,
                                                      index, page.data(), device.copy_stream);
            device.synchronize_all();
            require(std::memcmp(page.data(), expected[index].data(), page_bytes) == 0,
                    "RAM restore changed fixed-codec bytes");
        }
        ram.release(saved.entry_id);
    }
    {
        cache::KVDiskCache disk(config(directory.path, source_pool, ram, compression));
        disk.note_ram_resident(saved.entry_id, 0);
        require(disk.emergency_spill_ram(saved.entry_id), "SSD spill failed");
        disk.wait_idle_and_fsync();
        require(disk.ram_is_durable(saved.entry_id), "spill did not publish a durable RAM ticket");
    }
    require(ram.evict_one_unpinned(saved.entry_id), "durable RAM entry could not be evicted");

    // Startup uses a different physical page capacity, as enabling Vision can change VRAM.
    // Logical codec identity and packed disk bytes must remain reusable.
    Pool destination_pool(12);
    cache::KVDiskCache reopened(config(directory.path, destination_pool, ram, compression));
    const auto chain = cache::prefix_hash_chain(retained);
    const auto match = reopened.plan_match(retained, chain);
    require(match && match->reuse_base == 129, "durable vision prefix missing after reopen");
    auto different_media = retained;
    different_media.vision_items[0].content_digest[0] ^= 1;
    require(!reopened.plan_match(different_media, cache::prefix_hash_chain(different_media)),
            "changed Vision digest incorrectly reused disk state");
    require(reopened.claim(match->entry_id, match->hash_f, match->execution_frontier,
                            match->reuse_base, match->reuse, match->committed_generation),
            "durable generation could not be claimed");
    auto destination = destination_pool.storage->reserve(3);
    destination.materialize_pages(3, device.stream);
    device.synchronize_all();
    cache::DiskRestoreTarget target;
    target.text = &destination;
    target.text_pool = destination_pool.storage.get();
    target.text_semantics = destination_pool.semantics();
    target.text_dst_pages = 3;
    target.reuse = match->reuse;
    target.reuse_base = match->reuse_base;
    target.stream = device.copy_stream;
    const auto ticket = reopened.restore_device(match->entry_id, target);
    reopened.wait_copies(ticket);
    require(!reopened.restore_failed(), "disk restore failed");
    device.synchronize_all();
    reopened.release_restore_ticket(ticket);
    reopened.release(match->entry_id);
    for (std::uint32_t index = 0; index < 3; ++index) {
        ninfer::pack_paged_kv_logical_page_to_host(destination, *destination_pool.storage,
                                                  index, page.data(), device.copy_stream);
        device.synchronize_all();
        require(std::memcmp(page.data(), expected[index].data(), page_bytes) == 0,
                "SSD scatter/restore changed fixed-codec bytes");
    }

    const auto objects = reopened.test_main_page_ids(match->entry_id);
    require(!objects.empty(), "restored entry has no persistent pages");
    reopened.test_break_object(objects.front(), cache::DiskObjectKind::Main);
    require(reopened.claim(match->entry_id), "corruption probe could not claim entry");
    bool rejected = false;
    std::uint64_t corrupt_ticket = 0;
    try {
        corrupt_ticket = reopened.restore_device(match->entry_id, target);
        reopened.wait_copies(corrupt_ticket);
    } catch (const ninfer::runtime::CacheRestoreFailure&) {
        rejected = true;
    }
    reopened.cancel_restore();
    device.synchronize_all();
    if (corrupt_ticket != 0) { reopened.release_restore_ticket(corrupt_ticket); }
    reopened.release(match->entry_id);
    require(rejected, "corrupted SSD page was not rejected by restore CRC validation");
    reopened.invalidate_entry(match->entry_id);
    require(!reopened.plan_match(retained, chain), "invalidated corrupt prefix remained reusable");
}
} // namespace

int main(int argc, char** argv) {
    try {
        verify_crc();
        if (argc == 2 && std::string_view(argv[1]) == "--crc-only") {
            std::cout << "fixed cache CRC32C oracle: PASS\n";
            return 0;
        }
        ninfer::DeviceContext device;
        run_roundtrip(device, ninfer::KvDiskCompress::Off);
        run_roundtrip(device, ninfer::KvDiskCompress::Zstd);
        std::cout << "fixed RAM/SSD cache exact bytes, restart, Vision identity, CRC and invalidation: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fixed cache: FAIL: " << error.what() << '\n';
        return 1;
    }
}
