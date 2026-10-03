// Shared host/device layout of the LDS race checker's shadow region. The shadow lives at a fixed
// device virtual address so instrumented code objects need no per-module registration.
#pragma once

#include <cstdint>

namespace ninfer::gpu_check {

inline constexpr unsigned long long kShadowBase    = 0x7e0000000000ull;
inline constexpr unsigned long long kRecordsOffset = 4096;
inline constexpr unsigned kMaxRecords              = 1024;
// Open-addressed (site, kind) table: one stored record per distinct hazard site, plus its count.
inline constexpr unsigned long long kSiteKeysOffset   = kRecordsOffset + kMaxRecords * 128ull;
inline constexpr unsigned kSiteTableSize              = 4096;
inline constexpr unsigned long long kSiteCountsOffset = kSiteKeysOffset + kSiteTableSize * 8ull;
inline constexpr unsigned long long kSlotsOffset      = kSiteCountsOffset + kSiteTableSize * 4ull;

inline constexpr std::uint32_t kHeaderMagic = 0x4e524331u; // "NRC1"

// Size of the per-wave NinferRcState the instrumenter allocates in each kernel's frame.
inline constexpr unsigned kStateBytes = 32;

enum HazardKind : unsigned {
    kReadAfterWrite  = 1,
    kWriteAfterRead  = 2,
    kWriteAfterWrite = 3,
    kOutOfBounds     = 4,
    kSlotTooSmall    = 5,
};

struct Header {
    std::uint32_t magic;
    std::uint32_t slot_count;
    std::uint32_t slot_words;
    std::uint32_t nonce;
    std::uint32_t hazard_count;
    std::uint32_t record_count;
};

struct Record {
    std::uint32_t valid;
    std::uint32_t kind;
    std::uint32_t wave;
    std::uint32_t other_wave;
    std::uint32_t lds_offset;
    std::uint32_t epoch;
    std::uint32_t block[3];
    std::uint32_t thread;
    std::uint32_t site_slot;
    char site[84];
};

static_assert(sizeof(Record) == 128);

// One 64-bit shadow word per 4-byte LDS word. A tag mismatch or an older epoch means the entry
// belongs to another workgroup instance or an earlier barrier interval and starts fresh.
inline constexpr unsigned kTagMask   = 0xffffu;
inline constexpr unsigned kEpochMask = 0xffffffu;

struct Entry {
    unsigned tag          = 0; // 16 bits, 0 = empty
    unsigned epoch        = 0; // 24 bits
    unsigned writer       = 0; // 6 bits, 1-based wave, 0 = none
    unsigned write_mask   = 0; // 4 bits
    unsigned reader       = 0; // 6 bits, first reading wave
    unsigned multi_reader = 0; // 1 bit, a second distinct wave read
    unsigned read_mask    = 0; // 4 bits

#if defined(__HIP_DEVICE_COMPILE__)
    __device__
#endif
        static Entry decode(unsigned long long v) {
        Entry e;
        e.tag          = static_cast<unsigned>(v & 0xffffu);
        e.epoch        = static_cast<unsigned>((v >> 16) & 0xffffffu);
        e.writer       = static_cast<unsigned>((v >> 40) & 0x3fu);
        e.write_mask   = static_cast<unsigned>((v >> 46) & 0xfu);
        e.reader       = static_cast<unsigned>((v >> 50) & 0x3fu);
        e.multi_reader = static_cast<unsigned>((v >> 56) & 0x1u);
        e.read_mask    = static_cast<unsigned>((v >> 57) & 0xfu);
        return e;
    }

#if defined(__HIP_DEVICE_COMPILE__)
    __device__
#endif
        unsigned long long encode() const {
        return static_cast<unsigned long long>(tag & 0xffffu) |
               (static_cast<unsigned long long>(epoch & 0xffffffu) << 16) |
               (static_cast<unsigned long long>(writer & 0x3fu) << 40) |
               (static_cast<unsigned long long>(write_mask & 0xfu) << 46) |
               (static_cast<unsigned long long>(reader & 0x3fu) << 50) |
               (static_cast<unsigned long long>(multi_reader & 0x1u) << 56) |
               (static_cast<unsigned long long>(read_mask & 0xfu) << 57);
    }
};

} // namespace ninfer::gpu_check
