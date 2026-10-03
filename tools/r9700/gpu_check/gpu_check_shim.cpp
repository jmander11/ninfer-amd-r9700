// LD_PRELOAD host shim for the gfx1201 device checks (the R9700 counterpart of compute-sanitizer).
//
// NINFER_GPU_CHECK is a comma-separated set of:
//   memcheck   every hipMalloc is placed so its 256-byte-rounded end abuts an unmapped guard page
//              (and a guard page precedes it); an out-of-bounds device access faults the process.
//   initcheck  every hipMalloc is filled with NINFER_GPU_CHECK_POISON (default 0xff, the NaN/-1
//              pattern), so a read of unwritten device memory perturbs the oracle comparison.
//   racecheck  maps the LDS race checker's shadow at its fixed address (before the first hipMalloc)
//              for binaries built through hip_racecheck_launcher.py and reports hazards at exit
//              (exit status 86; 87 when the checker itself fails).
// The memcheck and initcheck modes need no rebuild; racecheck needs the instrumented build.

#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <climits>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "lds_race_layout.h"

namespace rc = ninfer::gpu_check;

namespace {

constexpr std::size_t kAlignment = 256;
constexpr int kRaceExitStatus    = 86; // hazards found
constexpr int kToolExitStatus    = 87; // the checker itself could not run

struct Config {
    bool memcheck  = false;
    bool initcheck = false;
    bool racecheck = false;
    int poison     = 0xff;
};

Config read_config() {
    Config c;
    const char* modes = std::getenv("NINFER_GPU_CHECK");
    if (modes != nullptr) {
        const std::string text = modes;
        c.memcheck             = text.find("memcheck") != std::string::npos;
        c.initcheck            = text.find("initcheck") != std::string::npos;
        c.racecheck            = text.find("racecheck") != std::string::npos;
    }
    if (const char* poison = std::getenv("NINFER_GPU_CHECK_POISON"); poison != nullptr) {
        c.poison = static_cast<int>(std::strtol(poison, nullptr, 0)) & 0xff;
    }
    return c;
}

const Config& config() {
    static const Config c = read_config();
    return c;
}

[[noreturn]] void die(const char* what, hipError_t error) {
    std::fprintf(stderr, "ninfer gpu_check: %s failed: %s\n", what, hipGetErrorString(error));
    std::_Exit(kToolExitStatus);
}

#define GC_CHECK(call)                                                                             \
    do {                                                                                           \
        const hipError_t e_ = (call);                                                              \
        if (e_ != hipSuccess) { die(#call, e_); }                                                  \
    } while (0)

std::size_t round_up(std::size_t v, std::size_t m) { return (v + m - 1) / m * m; }

hipMemAllocationProp device_prop() {
    int device = 0;
    GC_CHECK(hipGetDevice(&device));
    hipMemAllocationProp prop{};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = device;
    return prop;
}

std::size_t granularity(const hipMemAllocationProp& prop) {
    std::size_t g = 0;
    GC_CHECK(hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum));
    return g;
}

// ---------------------------------------------------------------------------------------------
// memcheck: guarded allocations

struct Guarded {
    void* reserve;
    std::size_t reserve_bytes;
    void* mapped;
    std::size_t mapped_bytes;
    hipMemGenericAllocationHandle_t handle;
};

std::mutex g_mutex;

std::map<void*, Guarded>& guarded() {
    static auto* map = new std::map<void*, Guarded>();
    return *map;
}

hipError_t guarded_malloc(void** ptr, std::size_t size) {
    if (size == 0) { // hipMalloc's contract: success with a null pointer
        *ptr = nullptr;
        return hipSuccess;
    }
    const hipMemAllocationProp prop = device_prop();
    const std::size_t g             = granularity(prop);
    const std::size_t payload       = round_up(size, kAlignment);
    const std::size_t mapped_bytes  = round_up(payload, g);
    const std::size_t reserve_bytes = mapped_bytes + 2 * g;
    void* reserve                   = nullptr;
    hipError_t e                    = hipMemAddressReserve(&reserve, reserve_bytes, g, nullptr, 0);
    if (e != hipSuccess) { return e; }
    hipMemGenericAllocationHandle_t handle{};
    if ((e = hipMemCreate(&handle, mapped_bytes, &prop, 0)) != hipSuccess) {
        (void)hipMemAddressFree(reserve, reserve_bytes);
        return e;
    }
    auto* mapped = static_cast<char*>(reserve) + g;
    hipMemAccessDesc access{};
    access.location = prop.location;
    access.flags    = hipMemAccessFlagsProtReadWrite;
    if ((e = hipMemMap(mapped, mapped_bytes, 0, handle, 0)) != hipSuccess ||
        (e = hipMemSetAccess(mapped, mapped_bytes, &access, 1)) != hipSuccess) {
        (void)hipMemRelease(handle);
        (void)hipMemAddressFree(reserve, reserve_bytes);
        return e;
    }
    void* user = mapped + mapped_bytes - payload;
    {
        std::lock_guard lock(g_mutex);
        guarded()[user] = Guarded{reserve, reserve_bytes, mapped, mapped_bytes, handle};
    }
    *ptr = user;
    return hipSuccess;
}

bool guarded_free(void* ptr, hipError_t* result) {
    Guarded entry{};
    {
        std::lock_guard lock(g_mutex);
        const auto it = guarded().find(ptr);
        if (it == guarded().end()) { return false; }
        entry = it->second;
        guarded().erase(it);
    }
    // hipFree synchronizes the device before releasing memory; keep that contract.
    hipError_t e = hipDeviceSynchronize();
    if (e == hipSuccess) { e = hipMemUnmap(entry.mapped, entry.mapped_bytes); }
    if (e == hipSuccess) { e = hipMemRelease(entry.handle); }
    if (e == hipSuccess) { e = hipMemAddressFree(entry.reserve, entry.reserve_bytes); }
    *result = e;
    return true;
}

// ---------------------------------------------------------------------------------------------
// racecheck: shadow mapping and the exit report

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
    const char* v = std::getenv(name);
    return v == nullptr ? fallback : static_cast<std::uint32_t>(std::strtoul(v, nullptr, 0));
}

const char* kind_name(std::uint32_t kind) {
    switch (kind) {
    case rc::kReadAfterWrite:
        return "read-after-write";
    case rc::kWriteAfterRead:
        return "write-after-read";
    case rc::kWriteAfterWrite:
        return "write-after-write";
    case rc::kOutOfBounds:
        return "LDS access beyond the launch's group segment";
    case rc::kSlotTooSmall:
        return "launch group segment larger than NINFER_RACECHECK_LDS_BYTES (excess not checked)";
    default:
        return "unknown";
    }
}

void race_report() {
    if (hipDeviceSynchronize() != hipSuccess) {
        std::fprintf(stderr, "ninfer racecheck: device failed before the report\n");
        return;
    }
    rc::Header header{};
    GC_CHECK(hipMemcpy(&header, reinterpret_cast<void*>(rc::kShadowBase), sizeof header,
                       hipMemcpyDeviceToHost));
    const std::uint32_t stored =
        header.record_count < rc::kMaxRecords ? header.record_count : rc::kMaxRecords;
    std::vector<rc::Record> records(stored);
    std::vector<std::uint32_t> counts(rc::kSiteTableSize);
    if (stored != 0) {
        GC_CHECK(hipMemcpy(records.data(),
                           reinterpret_cast<void*>(rc::kShadowBase + rc::kRecordsOffset),
                           stored * sizeof(rc::Record), hipMemcpyDeviceToHost));
        GC_CHECK(hipMemcpy(counts.data(),
                           reinterpret_cast<void*>(rc::kShadowBase + rc::kSiteCountsOffset),
                           counts.size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    }
    // One record per (site, hazard kind), stored by the first hazard there.
    for (const rc::Record& r : records) {
        if (r.valid == 0) { continue; }
        const std::string site(r.site, strnlen(r.site, sizeof r.site));
        const std::string other = r.other_wave == UINT32_MAX
                                      ? std::string("another wave")
                                      : "wave " + std::to_string(r.other_wave);
        std::fprintf(stderr,
                     "ninfer racecheck: %s in %s (%u hazards); first: LDS byte %u, wave %u vs %s, "
                     "barrier interval %u, block (%u,%u,%u) thread %u\n",
                     kind_name(r.kind), site.c_str(), counts[r.site_slot % rc::kSiteTableSize],
                     r.lds_offset, r.wave, other.c_str(), r.epoch, r.block[0], r.block[1],
                     r.block[2], r.thread);
    }
    if (header.record_count > rc::kMaxRecords) {
        std::fprintf(stderr, "ninfer racecheck: %u further hazard sites not recorded\n",
                     header.record_count - rc::kMaxRecords);
    }
    std::fprintf(stderr, "ninfer racecheck: %u workgroups checked, %u hazards\n", header.nonce,
                 header.hazard_count);
    if (header.hazard_count != 0) {
        std::fflush(nullptr);
        std::_Exit(kRaceExitStatus);
    }
}

void install_racecheck() {
    const std::uint32_t slot_count  = env_u32("NINFER_RACECHECK_SLOTS", 2048);
    const std::uint32_t slot_words  = env_u32("NINFER_RACECHECK_LDS_BYTES", 65536) / 4;
    const hipMemAllocationProp prop = device_prop();
    const std::size_t g             = granularity(prop);
    const std::size_t bytes =
        round_up(rc::kSlotsOffset + std::size_t{slot_count} * slot_words * 8, g);
    void* want = reinterpret_cast<void*>(rc::kShadowBase);
    void* got  = nullptr;
    GC_CHECK(hipMemAddressReserve(&got, bytes, g, want, 0));
    if (got != want) {
        std::fprintf(stderr, "ninfer racecheck: shadow reserved at %p instead of %p\n", got, want);
        std::_Exit(kToolExitStatus);
    }
    hipMemGenericAllocationHandle_t handle{};
    GC_CHECK(hipMemCreate(&handle, bytes, &prop, 0));
    GC_CHECK(hipMemMap(got, bytes, 0, handle, 0));
    hipMemAccessDesc access{};
    access.location = prop.location;
    access.flags    = hipMemAccessFlagsProtReadWrite;
    GC_CHECK(hipMemSetAccess(got, bytes, &access, 1));
    GC_CHECK(hipMemset(got, 0, rc::kSlotsOffset));
    const rc::Header header{rc::kHeaderMagic, slot_count, slot_words, 0, 0, 0};
    GC_CHECK(hipMemcpy(got, &header, sizeof header, hipMemcpyHostToDevice));
    std::atexit(race_report);
}

// The shadow is installed before the process's first device allocation, which precedes any
// kernel launch in NInfer programs. Processes that never allocate (ctest itself) never touch HIP.
void ensure_racecheck() {
    static std::once_flag once;
    if (config().racecheck) { std::call_once(once, install_racecheck); }
}

template <class F>
F real(const char* name) {
    static_assert(sizeof(F) == sizeof(void*));
    void* symbol = dlsym(RTLD_NEXT, name);
    if (symbol == nullptr) {
        std::fprintf(stderr, "ninfer gpu_check: %s not found\n", name);
        std::_Exit(kToolExitStatus);
    }
    F f;
    std::memcpy(&f, &symbol, sizeof f);
    return f;
}

} // namespace

extern "C" hipError_t hipMalloc(void** ptr, size_t size) {
    static const auto next = real<hipError_t (*)(void**, size_t)>("hipMalloc");
    static std::once_flag banner;
    std::call_once(banner, [] {
        // One line per checked process, so a log shows which tests ran under the check.
        std::fprintf(stderr, "ninfer gpu_check: %s%s%s active\n", config().memcheck ? "memcheck " : "",
                     config().initcheck ? "initcheck " : "", config().racecheck ? "racecheck" : "");
    });
    ensure_racecheck();
    const hipError_t e = config().memcheck ? guarded_malloc(ptr, size) : next(ptr, size);
    if (e == hipSuccess && config().initcheck && size != 0) {
        GC_CHECK(hipMemset(*ptr, config().poison, size));
        GC_CHECK(hipDeviceSynchronize());
    }
    return e;
}

extern "C" hipError_t hipFree(void* ptr) {
    static const auto next = real<hipError_t (*)(void*)>("hipFree");
    hipError_t result      = hipSuccess;
    if (ptr != nullptr && config().memcheck && guarded_free(ptr, &result)) { return result; }
    return next(ptr);
}
