#include "core/device.h"
#include "core/fp8_int4_paged_kv_cache.h"
#include "runtime/contract/types.h"
#include "targets/qwen3/impl/runtime/kv_disk_cache.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
namespace q3    = ninfer::targets::qwen3;
namespace cache = q3::detail;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

struct TemporaryDirectory {
    std::filesystem::path path;

    TemporaryDirectory() {
        // temp_directory_path honors TMPDIR, which the unit-test runner points at the build
        // tree so fsync-heavy disk-tier scratch stays off a container overlay.
        std::string pattern =
            (std::filesystem::temp_directory_path() / "ninfer-fixed-kv-XXXXXX").string();
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
        spec = {
            .page_group_count      = pages,
            .logical_page_capacity = 4,
            .table_rows            = 2,
            .layer_count           = 2,
            .head_dim              = 128,
            .num_kv_heads          = 2,
            .value_group           = 32,
            .plane_layouts = {.key         = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                              .value       = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                              .value_scale = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor}};
        ninfer::LayoutBuilder builder;
        layout  = ninfer::plan_fp8_k_int4_v_paged_kv_pool(builder, spec);
        arena   = std::make_unique<ninfer::DeviceArena>(builder.finish(256));
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

cache::DiskOpenConfig config(const std::filesystem::path& path, Pool& pool, cache::KVRamCache& ram,
                             ninfer::KvDiskCompress compression,
                             std::size_t capacity_bytes = 32ULL << 20) {
    cache::DiskOpenConfig result;
    result.location           = path;
    result.capacity_bytes     = capacity_bytes;
    result.compress           = compression;
    result.max_context        = 256;
    result.ram                = &ram;
    result.text_pool          = pool.storage.get();
    result.logical_page_bytes = ninfer::paged_kv_logical_page_bytes(*pool.storage);
    result.fingerprint        = cache::make_disk_fingerprint(
        "qwen3.8-27b", "r9700-int-candidate", "fixed-cache-test-artifact",
        ninfer::SpeculativeBackend::None, *pool.storage, nullptr, pool.semantics(), std::nullopt,
        nullptr, nullptr);
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
        ninfer::unpack_paged_kv_logical_page_from_host(
            allocation, *source_pool.storage, page.data(), static_cast<std::uint32_t>(index),
            device.copy_stream);
        device.synchronize_all();
    }

    auto retained = prompt();
    cache::ResidentPrefixIdentity identity;
    identity.assign(retained);
    cache::RamCaptureSource capture;
    capture.execution_frontier = 129;
    capture.ledger_frontier    = 130;
    capture.text_kv_valid      = 129;
    capture.tail_hidden_valid  = false;
    capture.ledger             = retained.token_ids;
    capture.identity           = &identity;
    capture.hash_f             = cache::prefix_hash_at(retained.token_ids, identity, 129);
    capture.text               = &allocation;
    capture.text_pool          = source_pool.storage.get();
    capture.text_semantics     = source_pool.semantics();
    capture.stream             = device.copy_stream;

    ram.test_fail_next_capture_metadata_allocation();
    require(ram.capture(capture).status == cache::RamCaptureStatus::Dropped,
            "optional RAM metadata allocation failure did not drop capture");
    const auto saved = ram.capture(capture);
    require(saved.status == cache::RamCaptureStatus::Captured, "RAM capture failed");
    ram.wait_pending_copies();
    const auto image = ram.host_kv(saved.entry_id);
    for (std::uint32_t index = 0; index < 3; ++index) {
        ninfer::gather_logical_page_from_host_image(image.text, *source_pool.storage, 3, index,
                                                    page.data());
        require(std::memcmp(page.data(), expected[index].data(), page_bytes) == 0,
                "RAM logical page gather changed fixed-codec bytes");
    }
    {
        auto ram_destination = source_pool.storage->reserve(3);
        ram_destination.materialize_pages(3, device.stream);
        device.synchronize_all();
        cache::RamRestoreTarget target;
        target.text           = &ram_destination;
        target.text_pool      = source_pool.storage.get();
        target.text_semantics = source_pool.semantics();
        target.text_dst_pages = 3;
        target.stream         = device.copy_stream;
        ram.claim(saved.entry_id);
        const auto restored = ram.unpack_device(saved.entry_id, target);
        ram.wait_pending_copies();
        require(restored.ledger == retained.token_ids && restored.identity.matches(retained, 129),
                "RAM restore changed represented token/vision identity");
        for (std::uint32_t index = 0; index < 3; ++index) {
            ninfer::pack_paged_kv_logical_page_to_host(ram_destination, *source_pool.storage, index,
                                                       page.data(), device.copy_stream);
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
    target.text           = &destination;
    target.text_pool      = destination_pool.storage.get();
    target.text_semantics = destination_pool.semantics();
    target.text_dst_pages = 3;
    target.reuse          = match->reuse;
    target.reuse_base     = match->reuse_base;
    target.stream         = device.copy_stream;
    const auto ticket     = reopened.restore_device(match->entry_id, target);
    reopened.wait_copies(ticket);
    require(!reopened.restore_failed(), "disk restore failed");
    device.synchronize_all();
    reopened.release_restore_ticket(ticket);
    reopened.release(match->entry_id);
    for (std::uint32_t index = 0; index < 3; ++index) {
        ninfer::pack_paged_kv_logical_page_to_host(destination, *destination_pool.storage, index,
                                                   page.data(), device.copy_stream);
        device.synchronize_all();
        require(std::memcmp(page.data(), expected[index].data(), page_bytes) == 0,
                "SSD scatter/restore changed fixed-codec bytes");
    }

    const auto objects = reopened.test_main_page_ids(match->entry_id);
    require(!objects.empty(), "restored entry has no persistent pages");
    reopened.test_break_object(objects.front(), cache::DiskObjectKind::Main);
    require(reopened.claim(match->entry_id), "corruption probe could not claim entry");
    bool rejected                = false;
    std::uint64_t corrupt_ticket = 0;
    try {
        corrupt_ticket = reopened.restore_device(match->entry_id, target);
        reopened.wait_copies(corrupt_ticket);
    } catch (const ninfer::runtime::CacheRestoreFailure&) { rejected = true; }
    reopened.cancel_restore();
    device.synchronize_all();
    if (corrupt_ticket != 0) { reopened.release_restore_ticket(corrupt_ticket); }
    reopened.release(match->entry_id);
    require(rejected, "corrupted SSD page was not rejected by restore CRC validation");
    reopened.invalidate_entry(match->entry_id);
    require(!reopened.plan_match(retained, chain), "invalidated corrupt prefix remained reusable");
}

// ---- Disk I/O stays off the decode and admission paths ----

template <class Predicate>
bool wait_pred(Predicate predicate, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!predicate() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return predicate();
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

q3::PreparedPromptData text_prompt(const std::vector<ninfer::TokenId>& tokens) {
    q3::PreparedPromptData result;
    result.token_ids = tokens;
    result.token_types.assign(tokens.size(), 0);
    result.positions.resize(tokens.size() * 3);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            result.positions[axis * tokens.size() + i] = static_cast<std::int32_t>(i);
        }
    }
    return result;
}

// A Text-only retained lane whose ledger is `tokens` plus one uncommitted token, so a lookup of
// `retained` reuses the whole `tokens` prefix without requiring tail hidden state.
struct TextCapture {
    q3::PreparedPromptData retained;
    cache::ResidentPrefixIdentity identity;
    cache::RamCaptureSource source;

    TextCapture(const std::vector<ninfer::TokenId>& tokens, ninfer::PagedKVAllocation& allocation,
                Pool& pool, hipStream_t stream) {
        auto ledger = tokens;
        ledger.push_back(0);
        retained = text_prompt(ledger);
        identity.assign(retained);
        const auto frontier       = static_cast<std::uint32_t>(tokens.size());
        source.execution_frontier = frontier;
        source.ledger_frontier    = frontier + 1;
        source.text_kv_valid      = frontier;
        source.ledger             = retained.token_ids;
        source.identity           = &identity;
        source.hash_f             = cache::prefix_hash_at(retained.token_ids, identity, frontier);
        source.text               = &allocation;
        source.text_pool          = pool.storage.get();
        source.text_semantics     = pool.semantics();
        source.stream             = stream;
    }

    TextCapture(const TextCapture&)            = delete;
    TextCapture& operator=(const TextCapture&) = delete;
};

std::uint64_t capture_tokens(cache::KVRamCache& ram, Pool& pool,
                             ninfer::PagedKVAllocation& allocation, ninfer::DeviceContext& device,
                             const std::vector<ninfer::TokenId>& tokens) {
    TextCapture capture(tokens, allocation, pool, device.copy_stream);
    const auto result = ram.capture(capture.source);
    require(result.status == cache::RamCaptureStatus::Captured, "stall fixture RAM capture failed");
    device.synchronize_all();
    ram.wait_pending_copies();
    return result.entry_id;
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// An idle prepare that makes capacity releases the mutex for the eviction's MANIFEST write. A
// cancellation in that window must stop the spill before it installs; otherwise
// cancel_idle_spill waits for the whole idle image write.
void idle_cancel_during_capacity_eviction(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    auto options = config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20);
    const auto first =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 21));
    std::size_t used = 0;
    {
        cache::KVDiskCache disk(options);
        disk.note_ram_resident(first, 0);
        require(disk.emergency_spill_ram(first), "idle-cancel-evict first spill failed");
        disk.wait_idle_and_fsync();
        used = disk.snapshot().used_bytes;
    }
    options.capacity_bytes = used;
    cache::KVDiskCache disk(options);
    const auto second =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 22));
    disk.note_ram_resident(second, 0);
    // The eviction's MANIFEST write is held until the cancellation has registered, so the
    // cancellation always lands inside the window regardless of host scheduling.
    disk.test_hold_manifest_io(true);
    disk.test_set_payload_io_stall_ms(2000);
    disk.request_idle_spill();
    if (!wait_pred([&] { return disk.test_manifest_io_entered(); }, std::chrono::seconds(5))) {
        disk.test_hold_manifest_io(false);
        disk.test_set_payload_io_stall_ms(0);
        disk.cancel_idle_spill();
        require(false, "idle-cancel-evict idle prepare did not reach capacity eviction");
    }
    const std::uint64_t epoch = disk.test_idle_cancel_epoch();
    std::chrono::steady_clock::time_point released;
    std::chrono::steady_clock::time_point finished;
    std::thread canceller([&] {
        disk.cancel_idle_spill();
        finished = std::chrono::steady_clock::now();
    });
    const bool registered =
        wait_pred([&] { return disk.test_idle_cancel_epoch() != epoch; }, std::chrono::seconds(5));
    released = std::chrono::steady_clock::now();
    disk.test_hold_manifest_io(false);
    canceller.join();
    disk.test_set_payload_io_stall_ms(0);
    const bool durable = disk.ram_is_durable(second);
    disk.wait_idle_and_fsync();
    require(registered,
            "idle-cancel-evict cancellation did not register during the MANIFEST write");
    require(finished - released <= std::chrono::milliseconds(1200),
            "idle cancel waited for a spill installed after cancellation");
    require(!durable, "idle spill committed after cancellation");
}

// Pinning an entry for a disk spill waits for that entry's own device copies, not every pending
// RAM copy: an unrelated gated capture must not stall it.
void spill_pin_waits_only_its_entry(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    HIP_CHECK(hipStreamSynchronize(device.stream));
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20));
    hipStream_t side = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking));
    constexpr std::size_t kDelayBytes = 256ULL << 20;
    ninfer::DeviceBuffer delay_source(kDelayBytes);
    ninfer::PinnedHostBuffer delay_destination(kDelayBytes);
    bool prompt  = false;
    bool spilled = false;
    {
        StreamGate gate;
        gate.launch(side);
        TextCapture unrelated(std::vector<ninfer::TokenId>(64, 23), allocation, pool, side);
        const bool gated =
            ram.capture(unrelated.source).status == cache::RamCaptureStatus::Captured;
        // Keep the spilled entry's own D2H in flight briefly behind a large copy.
        HIP_CHECK(hipMemcpyAsync(delay_destination.data(), delay_source.data(), kDelayBytes,
                                 hipMemcpyDeviceToHost, device.copy_stream));
        TextCapture own(std::vector<ninfer::TokenId>(64, 24), allocation, pool, device.copy_stream);
        const auto captured = ram.capture(own.source);
        if (gated && captured.status == cache::RamCaptureStatus::Captured) {
            disk.note_ram_resident(captured.entry_id, 0);
            std::atomic<bool> done{false};
            std::atomic<bool> result{false};
            std::thread spiller([&] {
                try {
                    result.store(disk.emergency_spill_ram(captured.entry_id));
                } catch (...) {}
                done.store(true);
            });
            prompt = wait_pred([&] { return done.load(); }, std::chrono::milliseconds(1500));
            gate.release();
            spiller.join();
            spilled = result.load();
        }
        gate.release();
        require(gated && captured.status == cache::RamCaptureStatus::Captured,
                "spill-own-copy fixture capture failed");
    }
    device.synchronize_all();
    HIP_CHECK(hipStreamSynchronize(side));
    HIP_CHECK(hipStreamDestroy(side));
    disk.wait_idle_and_fsync();
    require(prompt, "disk spill pin waited for an unrelated gated RAM copy");
    require(spilled, "spill-own-copy spill failed");
}

// Compaction copies the whole live pack store. Lookups on the scheduler thread must not wait for
// that copy, and the worker still publishes the new generation.
void plan_match_during_compaction(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    const auto options =
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20);
    const std::vector<ninfer::TokenId> tokens_a(64, 25);
    const std::vector<ninfer::TokenId> tokens_b(128, 26);
    {
        cache::KVDiskCache disk(options);
        for (const auto* tokens : {&tokens_a, &tokens_b}) {
            const auto ram_id = capture_tokens(ram, pool, allocation, device, *tokens);
            disk.note_ram_resident(ram_id, 0);
            require(disk.emergency_spill_ram(ram_id), "compaction-lock fixture spill failed");
        }
        require(disk.test_fifo_evict_one(), "compaction-lock fixture eviction failed");
    }
    const auto before = read_bytes(directory.path / "PACKSET");
    cache::KVDiskCache disk(options);
    disk.test_set_compaction_copy_stall_ms(300);
    std::thread maintenance([&] { disk.wait_idle_and_fsync(); });
    if (!wait_pred([&] { return disk.test_compaction_copy_entered(); }, std::chrono::seconds(5))) {
        disk.test_set_compaction_copy_stall_ms(0);
        maintenance.join();
        require(false, "compaction-lock maintenance never started copying");
    }
    auto ledger = tokens_b;
    ledger.push_back(0);
    const auto prompt  = text_prompt(ledger);
    const auto started = std::chrono::steady_clock::now();
    const auto match   = disk.plan_match(prompt, cache::prefix_hash_chain(prompt));
    const auto elapsed = std::chrono::steady_clock::now() - started;
    disk.test_set_compaction_copy_stall_ms(0);
    maintenance.join();
    const auto after = read_bytes(directory.path / "PACKSET");
    require(elapsed <= std::chrono::milliseconds(150), "plan_match waited for the compaction copy");
    require(match.has_value(), "compaction-lock lost the retained entry");
    require(before.size() == 32 && after.size() == 32 &&
                std::memcmp(before.data() + 12, after.data() + 12, sizeof(std::uint64_t)) != 0,
            "compaction-lock maintenance did not publish a new generation");
}

// Retries a non-blocking reclaim until it evicts. Reclaim has side effects, so it must not be a
// wait_pred predicate, which evaluates once more after succeeding.
bool reclaim_until_evicted(cache::KVDiskCache& disk, cache::RamReclaim& result) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (result != cache::RamReclaim::Evicted && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        result = disk.reclaim_ram_entry(false);
    }
    return result == cache::RamReclaim::Evicted;
}

// Reclaiming RAM for a capture while other lanes decode must not wait on a disk write: a
// disk-durable entry is evicted first, and without one the oldest entry is spilled on the disk
// worker and evicted once durable, never dropped unsaved and never spilled synchronously.
void ram_reclaim_never_blocks_on_disk(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20));
    const std::vector<ninfer::TokenId> older_tokens(64, 27);
    const auto older = capture_tokens(ram, pool, allocation, device, older_tokens);
    disk.note_ram_resident(older, 0);
    const auto newer =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 28));
    disk.note_ram_resident(newer, 0);
    require(disk.emergency_spill_ram(newer), "ram-reclaim fixture spill failed");
    disk.wait_idle_and_fsync();
    // Any disk write from here on would take seconds.
    disk.test_set_payload_io_stall_ms(2000);
    const auto resident = [&](std::uint64_t id) {
        const auto ids = ram.fifo_ids();
        return std::find(ids.begin(), ids.end(), id) != ids.end();
    };
    auto started = std::chrono::steady_clock::now();
    auto result  = disk.reclaim_ram_entry(false);
    auto elapsed = std::chrono::steady_clock::now() - started;
    const bool durable_first =
        result == cache::RamReclaim::Evicted && !resident(newer) && resident(older);
    const bool durable_prompt = elapsed <= std::chrono::milliseconds(500);
    const auto drops          = disk.snapshot().drops;
    started                   = std::chrono::steady_clock::now();
    result                    = disk.reclaim_ram_entry(false);
    elapsed                   = std::chrono::steady_clock::now() - started;
    const bool unsaved_kept   = result == cache::RamReclaim::Pending && resident(older);
    const bool unsaved_prompt = elapsed <= std::chrono::milliseconds(500);
    disk.test_set_payload_io_stall_ms(0);
    const bool evicted_once_durable = reclaim_until_evicted(disk, result);
    auto ledger                     = older_tokens;
    ledger.push_back(0);
    const auto prompt  = text_prompt(ledger);
    const bool on_disk = disk.plan_match(prompt, cache::prefix_hash_chain(prompt)).has_value();
    require(durable_first, "non-blocking reclaim did not evict the disk-durable entry first");
    require(durable_prompt, "reclaim of a durable entry waited on disk I/O");
    require(unsaved_kept,
            "non-blocking reclaim did not defer the unsaved oldest entry to its spill");
    require(unsaved_prompt, "non-blocking reclaim spilled synchronously");
    require(evicted_once_durable && !resident(older),
            "non-blocking reclaim never evicted the spilled entry");
    require(disk.snapshot().drops == drops, "non-blocking reclaim dropped an unsaved RAM entry");
    require(on_disk, "reclaimed RAM entry is not on disk");
}

// Under sustained load write-behind is still spilling the oldest entry when RAM fills. That entry
// is I/O-pinned, so it is not an eviction candidate; reclaim must finish its spill and evict it
// rather than drop a newer entry.
void ram_reclaim_waits_for_inflight_spill(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20));
    const auto older =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 37));
    disk.note_ram_resident(older, 0);
    const auto newer =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 38));
    disk.note_ram_resident(newer, 0);
    disk.test_set_payload_io_stall_ms(300);
    disk.request_idle_spill();
    if (!wait_pred([&] { return ram.test_io_pins(older) != 0; }, std::chrono::seconds(5))) {
        disk.test_set_payload_io_stall_ms(0);
        require(false, "ram-reclaim-inflight write-behind never started");
    }
    const auto resident = [&](std::uint64_t id) {
        const auto ids = ram.fifo_ids();
        return std::find(ids.begin(), ids.end(), id) != ids.end();
    };
    const auto drops      = disk.snapshot().drops;
    const auto started    = std::chrono::steady_clock::now();
    auto result           = disk.reclaim_ram_entry(false);
    const auto elapsed    = std::chrono::steady_clock::now() - started;
    const bool newer_kept = result != cache::RamReclaim::Evicted && resident(newer);
    const bool prompt     = elapsed <= std::chrono::milliseconds(200);
    const bool evicted    = reclaim_until_evicted(disk, result);
    disk.test_set_payload_io_stall_ms(0);
    require(newer_kept, "reclaim dropped a newer entry while the oldest was spilling");
    require(prompt, "non-blocking reclaim waited for the in-flight spill");
    require(evicted && !resident(older) && resident(newer),
            "reclaim did not evict the spilled oldest entry");
    require(disk.snapshot().drops == drops, "reclaim dropped an unsaved RAM entry");
}

// A non-blocking reclaim for an admission's second capture must not target the entry its first
// capture just made: the deferral would roll that entry back, the retry would recapture and
// target it again, and admission would defer until no other lane decodes. With every older entry
// unsavable this generation, the oldest one is dropped unsaved instead; with nothing else left
// the capture is refused rather than deferred.
void ram_reclaim_skips_attempt_captures(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20));
    const auto older =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 47));
    disk.note_ram_resident(older, 0);
    disk.test_arm_fail_prepare_spill();
    require(!disk.emergency_spill_ram(older), "ram-reclaim-attempt fixture spill did not fail");
    const auto attempt =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 48));
    disk.note_ram_resident(attempt, 0);
    // Any disk write from here on would take seconds.
    disk.test_set_payload_io_stall_ms(2000);
    const auto resident = [&](std::uint64_t id) {
        const auto ids = ram.fifo_ids();
        return std::find(ids.begin(), ids.end(), id) != ids.end();
    };
    const std::array<std::uint64_t, 1> keep{attempt};
    const auto drops = disk.snapshot().drops;
    const auto unsaved =
        disk.snapshot()
            .drop_reasons[static_cast<std::size_t>(ninfer::KvDiskDropReason::ReclaimUnsaved)];
    const auto started       = std::chrono::steady_clock::now();
    const auto first         = disk.reclaim_ram_entry(false, keep);
    const bool dropped_older = first == cache::RamReclaim::Evicted && !resident(older) &&
                               resident(attempt) && !disk.ram_reclaim_pending();
    const bool drop_counted =
        disk.snapshot().drops == drops + 1 &&
        disk.snapshot()
                .drop_reasons[static_cast<std::size_t>(ninfer::KvDiskDropReason::ReclaimUnsaved)] ==
            unsaved + 1;
    const auto second = disk.reclaim_ram_entry(false, keep);
    const bool refused =
        second == cache::RamReclaim::NoVictim && resident(attempt) && !disk.ram_reclaim_pending();
    const bool prompt =
        std::chrono::steady_clock::now() - started <= std::chrono::milliseconds(500);
    disk.test_set_payload_io_stall_ms(0);
    require(
        dropped_older,
        "reclaim targeted this attempt's capture instead of dropping the unsavable older entry");
    require(drop_counted, "dropping the unsavable older entry was not counted");
    require(refused, "reclaim with only this attempt's capture left did not refuse the capture");
    require(prompt, "reclaim excluding this attempt's capture waited on disk I/O");
}

// restore_device waits for another entry's in-flight window reads. The readiness probe lets
// admission keep decoding instead of taking that wait.
void restore_setup_ready_tracks_window_reads(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(32ULL << 20);
    auto allocation = pool.storage->reserve(2);
    allocation.materialize_pages(1, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 32ULL << 20));

    struct BarrierGuard {
        cache::KVDiskCache& disk;

        ~BarrierGuard() { disk.test_release_page_read_barrier(); }
    } barrier{disk};

    const std::vector<ninfer::TokenId> tokens_a{12, 13, 14, 15};
    const std::vector<ninfer::TokenId> tokens_b{16, 17, 18, 19};
    for (const auto* tokens : {&tokens_a, &tokens_b}) {
        const auto ram_id = capture_tokens(ram, pool, allocation, device, *tokens);
        disk.note_ram_resident(ram_id, 0);
        require(disk.emergency_spill_ram(ram_id), "restore-ready fixture spill failed");
    }
    auto ledger_a = tokens_a;
    ledger_a.push_back(0);
    auto ledger_b = tokens_b;
    ledger_b.push_back(0);
    const auto prompt_a = text_prompt(ledger_a);
    const auto prompt_b = text_prompt(ledger_b);
    const auto match_a  = disk.plan_match(prompt_a, cache::prefix_hash_chain(prompt_a));
    const auto match_b  = disk.plan_match(prompt_b, cache::prefix_hash_chain(prompt_b));
    require(match_a && match_b && disk.claim(match_b->entry_id),
            "restore-ready fixture could not claim the prefetched entry");
    const bool idle_ready = disk.restore_setup_ready(match_a->entry_id);
    disk.test_arm_page_read_barrier();
    disk.prefetch_window(match_b->entry_id, 1, 0);
    if (!wait_pred([&] { return disk.test_page_read_entered(); }, std::chrono::seconds(2))) {
        disk.test_release_page_read_barrier();
        disk.release(match_b->entry_id);
        require(false, "restore-ready prefetch did not enter its page read");
    }
    const bool busy_ready = disk.restore_setup_ready(match_a->entry_id);
    disk.test_release_page_read_barrier();
    const bool drained_ready = wait_pred(
        [&] { return disk.restore_setup_ready(match_a->entry_id); }, std::chrono::seconds(2));
    disk.release(match_b->entry_id);
    require(idle_ready, "restore setup reported busy with no window reads");
    require(!busy_ready, "restore setup reported ready while another entry's read was in flight");
    require(drained_ready, "restore setup stayed busy after the window read finished");
}

void run_stall_cases(ninfer::DeviceContext& device) {
    idle_cancel_during_capacity_eviction(device);
    spill_pin_waits_only_its_entry(device);
    plan_match_during_compaction(device);
    ram_reclaim_never_blocks_on_disk(device);
    ram_reclaim_waits_for_inflight_spill(device);
    ram_reclaim_skips_attempt_captures(device);
    restore_setup_ready_tracks_window_reads(device);
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
        if (argc == 3 && std::string_view(argv[1]) == "--case" &&
            std::string_view(argv[2]) == "stalls") {
            run_stall_cases(device);
            std::cout << "disk tier stays off decode and admission paths: PASS\n";
            return 0;
        }
        run_roundtrip(device, ninfer::KvDiskCompress::Off);
        run_roundtrip(device, ninfer::KvDiskCompress::Zstd);
        run_stall_cases(device);
        std::cout << "fixed RAM/SSD cache exact bytes, restart, Vision identity, CRC and "
                     "invalidation: PASS\n";
        std::cout << "disk tier stays off decode and admission paths: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fixed cache: FAIL: " << error.what() << '\n';
        return 1;
    }
}
