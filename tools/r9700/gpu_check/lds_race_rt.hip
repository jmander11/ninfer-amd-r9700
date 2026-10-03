// Device half of the gfx1201 LDS race checker. lds_race_instrument links this bitcode into every
// instrumented code object; it is never compiled into the product. The host shim
// (gpu_check_shim.cpp) maps the shadow at kShadowBase before any kernel runs.

#include <hip/hip_runtime.h>

#include "lds_race_layout.h"

namespace rc = ninfer::gpu_check;

struct NinferRcState {
    unsigned long long slot; // shadow word array of this workgroup instance
    unsigned tag;
    unsigned epoch;
    unsigned wave; // 1-based wave index in the workgroup
    unsigned enabled;
    unsigned words; // LDS words of this launch's group segment (static plus dynamic)
};

static_assert(sizeof(NinferRcState) <= ninfer::gpu_check::kStateBytes &&
              alignof(NinferRcState) <= 8);

namespace {

__device__ inline rc::Header* header() { return reinterpret_cast<rc::Header*>(rc::kShadowBase); }

__device__ inline unsigned linear_thread() {
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

__device__ inline unsigned group_segment_bytes() {
    // hsa_kernel_dispatch_packet_t::group_segment_size: static plus dynamic LDS of this launch.
    const auto* packet = static_cast<const unsigned char*>(__builtin_amdgcn_dispatch_ptr());
    return *reinterpret_cast<const unsigned*>(packet + 28);
}

__device__ void report(const NinferRcState* s, unsigned kind, unsigned other_wave, unsigned offset,
                       const char* site) {
    rc::Header* h = header();
    atomicAdd(&h->hazard_count, 1u);
    auto* keys   = reinterpret_cast<unsigned long long*>(rc::kShadowBase + rc::kSiteKeysOffset);
    auto* counts = reinterpret_cast<unsigned*>(rc::kShadowBase + rc::kSiteCountsOffset);
    const unsigned long long key = (reinterpret_cast<unsigned long long>(site) << 3) | kind;
    unsigned index =
        static_cast<unsigned>((key * 0x9e3779b97f4a7c15ull) >> 52) % rc::kSiteTableSize;
    bool inserted = false;
    for (unsigned probe = 0; probe < rc::kSiteTableSize;
         ++probe, index = (index + 1) % rc::kSiteTableSize) {
        const unsigned long long seen = atomicCAS(&keys[index], 0ull, key);
        if (seen == key) {
            atomicAdd(&counts[index], 1u);
            return;
        }
        if (seen == 0ull) {
            inserted = true;
            break;
        }
    }
    if (!inserted) {
        return; // site table full: the hazard is counted in hazard_count only
    }
    atomicAdd(&counts[index], 1u);
    const unsigned slot = atomicAdd(&h->record_count, 1u);
    if (slot >= rc::kMaxRecords) { return; }
    rc::Record* r = reinterpret_cast<rc::Record*>(rc::kShadowBase + rc::kRecordsOffset) + slot;
    r->kind       = kind;
    r->wave       = s->wave - 1;
    r->other_wave = other_wave - 1;
    r->lds_offset = offset;
    r->epoch      = s->epoch;
    r->block[0]   = blockIdx.x;
    r->block[1]   = blockIdx.y;
    r->block[2]   = blockIdx.z;
    r->thread     = linear_thread();
    r->site_slot  = index;
    unsigned i    = 0;
    for (; i + 1 < sizeof(r->site) && site[i] != '\0'; ++i) { r->site[i] = site[i]; }
    r->site[i] = '\0';
    __threadfence_system();
    r->valid = 1;
}

__device__ void access_word(NinferRcState* s, unsigned word, unsigned mask, bool write,
                            const char* site) {
    if (word >= s->words) {
        report(s, rc::kOutOfBounds, s->wave, word * 4u, site);
        return;
    }
    auto* entry            = reinterpret_cast<unsigned long long*>(s->slot) + word;
    unsigned long long old = __hip_atomic_load(entry, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    for (;;) {
        rc::Entry e = rc::Entry::decode(old);
        if (e.tag != s->tag || e.epoch != (s->epoch & rc::kEpochMask)) {
            e       = rc::Entry{};
            e.tag   = s->tag;
            e.epoch = s->epoch & rc::kEpochMask;
        }
        unsigned hazard = 0;
        unsigned other  = 0;
        if (write) {
            if (e.writer != 0 && e.writer != s->wave && (e.write_mask & mask) != 0) {
                hazard = rc::kWriteAfterWrite;
                other  = e.writer;
            } else if ((e.read_mask & mask) != 0 &&
                       (e.multi_reader || (e.reader != 0 && e.reader != s->wave))) {
                hazard = rc::kWriteAfterRead;
                other  = e.reader != s->wave ? e.reader : 0;
            }
            if (e.writer == s->wave) {
                e.write_mask |= mask;
            } else {
                e.writer     = s->wave;
                e.write_mask = mask;
            }
        } else {
            if (e.writer != 0 && e.writer != s->wave && (e.write_mask & mask) != 0) {
                hazard = rc::kReadAfterWrite;
                other  = e.writer;
            }
            if (e.reader == 0) {
                e.reader = s->wave;
            } else if (e.reader != s->wave) {
                e.multi_reader = 1;
            }
            e.read_mask |= mask;
        }
        const unsigned long long desired = e.encode();
        if (desired == old) {
            if (hazard != 0) { report(s, hazard, other, word * 4u, site); }
            return;
        }
        const unsigned long long seen = atomicCAS(entry, old, desired);
        if (seen == old) {
            if (hazard != 0) { report(s, hazard, other, word * 4u, site); }
            return;
        }
        old = seen;
    }
}

__device__ void access_range(NinferRcState* s, unsigned offset, unsigned bytes, bool write,
                             const char* site) {
    if (s->enabled == 0 || bytes == 0) { return; }
    const unsigned end = offset + bytes;
    for (unsigned word = offset / 4u; word * 4u < end; ++word) {
        const unsigned lo   = max(offset, word * 4u) - word * 4u;
        const unsigned hi   = min(end, word * 4u + 4u) - word * 4u;
        const unsigned mask = ((1u << hi) - 1u) & ~((1u << lo) - 1u);
        access_word(s, word, mask, write, site);
    }
}

} // namespace

extern "C" {

// Every workgroup instance gets a fresh nonce through `nonce_lds` (one extra LDS word the
// instrumenter adds to each kernel), clears its shadow slot, and starts at epoch 0.
__device__ __attribute__((noinline)) void
__ninfer_rc_enter(NinferRcState* s, __attribute__((address_space(3))) unsigned* nonce_lds) {
    rc::Header* h      = header();
    const unsigned tid = linear_thread();
    if (tid == 0) { *nonce_lds = atomicAdd(&h->nonce, 1u); }
    __syncthreads();
    const unsigned nonce      = *nonce_lds;
    const unsigned slot_words = h->slot_words;
    s->slot    = rc::kShadowBase + rc::kSlotsOffset +
                 static_cast<unsigned long long>(nonce % h->slot_count) * slot_words * 8ull;
    s->tag     = (nonce % rc::kTagMask) + 1u;
    s->epoch   = 0;
    s->wave    = tid / warpSize + 1u;
    s->enabled = 1;
    const unsigned segment_words = (group_segment_bytes() + 3u) / 4u;
    if (segment_words > slot_words) {
        // The shadow slot is smaller than this launch's LDS: the excess is reported, not shadowed.
        if (tid == 0) {
            report(s, rc::kSlotTooSmall, s->wave, segment_words * 4u, "launch group segment");
        }
    }
    const unsigned used    = min(segment_words, slot_words);
    s->words               = used;
    auto* slot             = reinterpret_cast<unsigned long long*>(s->slot);
    const unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    for (unsigned i = tid; i < used; i += threads) { slot[i] = 0; }
    __threadfence();
    __syncthreads();
}

__device__ __attribute__((noinline)) void __ninfer_rc_barrier(NinferRcState* s) { ++s->epoch; }

__device__ __attribute__((noinline)) void __ninfer_rc_lds(NinferRcState* s,
                                                          __attribute__((address_space(3))) void* p,
                                                          unsigned bytes, unsigned write,
                                                          const char* site) {
    const auto offset = static_cast<unsigned>(reinterpret_cast<unsigned long long>(p));
    access_range(s, offset, bytes, write != 0, site);
}

__device__ __attribute__((noinline)) void
__ninfer_rc_flat(NinferRcState* s, void* p, unsigned bytes, unsigned write, const char* site) {
    if (!__builtin_amdgcn_is_shared(p)) { return; }
    // The shared aperture occupies the high half of a flat LDS address; the low half is the offset.
    access_range(s, static_cast<unsigned>(reinterpret_cast<unsigned long long>(p)), bytes,
                 write != 0, site);
}

} // extern "C"
