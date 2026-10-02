// Persistent decode kernel (ops::persistent_decode_lower): one resident grid of
// kBlockThreads-thread blocks, two per WGP, executes a run of captured launches as phases separated by grid barriers.
// A phase's virtual CTAs run the launch's kernel body (persistent_cta.h; or an entry's re-tiled
// body of the same Op) on a group of whole waves: kBlockThreads / Threads groups per block for a
// Threads-wide body, the whole block for a kBlockThreads-wide one; group g of block b runs
// virtual CTAs g * gridDim.x + b, ... (a partial last round spreads over all blocks). Each hosted kernel is an Entry: its kernel, launch width, body width,
// LDS, body and the weight bytes its virtual CTAs stream. While a block waits at a grid barrier
// it prefetches the next phase's instructions and its own first virtual CTAs' weight lines.
#include "ninfer/ops/persistent_decode.h"

#include "core/device.h"
#include "ops/r9700/gdn/gdn_front_kernels.h"
#include "ops/r9700/gdn_recurrence/gdn_record_kernel.h"
#include "ops/r9700/linear/fp8_activation_kernels.h"
#include "ops/r9700/linear/fp8_small_t_kernel.h"
#include "ops/r9700/linear/fp8lut4_small_t_kernel.h"
#include "ops/r9700/persistent/persistent_cta.h"
#include "ops/r9700/rope/rope_kernels.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::ops {
namespace {

namespace persistent = r9700::persistent;
namespace linear = r9700::linear;
namespace gdn_front = r9700::gdn::front_kernels;
namespace recurrence = detail::gated_delta_net::recurrence;

// Two 384-thread blocks per WGP: a 128-thread FP8LUT4 GEMM phase ran as fast as 128-thread
// persistent blocks and ~0.8 us faster per boundary than one 768-thread block per WGP.
constexpr std::uint32_t kBlockThreads = 384U;
constexpr std::uint32_t kBlocksPerWgp = 2U;
// LDS of one block: its share of the WGP's 64 KiB in 512-byte allocation granules.
constexpr std::size_t kBlockLds = 64U * 1024U / kBlocksPerWgp / 512U * 512U;
// Waves per SIMD of the resident blocks (the register budget: 1536 / waves, in 24-register granules).
constexpr std::uint32_t kWavesPerSimd = kBlocksPerWgp * kBlockThreads / 32U / 4U;
constexpr std::uint32_t kWaveSize = 32U;

struct NoShared {};

// A launch's kernel arguments, laid out identically on host and device.
template <std::size_t I, class T>
struct Slot {
    T value;
};
template <class Indices, class... A>
struct PackImpl;
template <std::size_t... I, class... A>
struct PackImpl<std::index_sequence<I...>, A...> : Slot<I, A>... {
    static constexpr std::size_t size = sizeof...(A);
    template <std::size_t J>
    using Type = std::tuple_element_t<J, std::tuple<A...>>;
};
template <class... A>
using Pack = PackImpl<std::index_sequence_for<A...>, A...>;

template <std::size_t J, class P>
__host__ __device__ constexpr const auto& slot_value(const P& pack) {
    return static_cast<const Slot<J, typename P::template Type<J>>&>(pack).value;
}
template <std::size_t J, class P>
constexpr auto& slot_value(P& pack) {
    return static_cast<Slot<J, typename P::template Type<J>>&>(pack).value;
}

#if defined(__HIPCC__)
// A body argument. Every captured pointer argument addresses global memory, but a body inlined
// into the persistent kernel reads it from LDS as a generic pointer; routing it through the global
// address space lets address-space inference emit global instead of FLAT accesses, as in the
// body's own kernel.
template <class T>
__device__ __forceinline__ decltype(auto) body_arg(const T& value) {
    if constexpr (std::is_pointer_v<T>) {
        // Rebuilt from its block-uniform address words as a global pointer, so the conversion
        // back to the body's generic type is not folded away and the address stays scalar.
        const auto address = reinterpret_cast<std::uintptr_t>(value);
        const auto word = [](std::uintptr_t bits) {
            return static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                __builtin_amdgcn_readfirstlane(static_cast<std::uint32_t>(bits))));
        };
        const std::uint64_t words = word(address) | word(address >> 32U) << 32U;
        using Pointee = std::remove_pointer_t<T>;
        return (T)((persistent::Global<Pointee>*)(words));  // global to generic
    } else {
        return (value);
    }
}
#endif

template <class Kernel>
struct KernelParams;
template <class... A>
struct KernelParams<void (*)(A...)> {
    using type = Pack<A...>;
};

// Default: the virtual grid is the captured one, and a group's consecutive virtual CTAs are
// separated by a group barrier (lds_reuse_safe: the body writes LDS before its first barrier only
// where its previous CTA no longer reads after its last one, so no barrier is needed).
struct CapturedGrid {
    static constexpr bool lds_reuse_safe = false;
    template <class Params>
    static uint3 grid(uint3 captured, const Params&) {
        return captured;
    }
};

// The weight bytes each virtual CTA of a phase streams, for the prefetch of the phase at the
// grid barrier before it: CTA c < split reads parts[0] (c - split: parts[1]), each part up to two
// contiguous ranges of `bytes` bytes at base + c' * bytes.
// (Trivial: staged in __shared__ phase slots; value-initialize on the host.)
struct StreamRange {
    std::uint64_t base;
    std::uint64_t bytes;  // per CTA; 0: none
};
struct PhaseStreams {
    StreamRange parts[2][2];
    std::uint32_t split;
    std::uint32_t groups;  // virtual CTAs per block; 0: the phase streams nothing
    std::uint32_t total;   // virtual CTAs of the phase
    std::uint32_t reserved;
};

// ---- Entries.
// Row block c of an FP8LUT4 weight: its contiguous N16 code tiles and group codes.
inline PhaseStreams fp8lut4_streams(const linear::Fp8Lut4Weight& first,
                                    const linear::Fp8Lut4Weight& second) {
    const auto part = [](const linear::Fp8Lut4Weight& weight) {
        return std::array<StreamRange, 2>{
            StreamRange{reinterpret_cast<std::uint64_t>(weight.codes),
                        static_cast<std::uint64_t>(weight.columns) * 8U},
            StreamRange{reinterpret_cast<std::uint64_t>(weight.groups),
                        static_cast<std::uint64_t>(weight.columns) / 2U}};
    };
    PhaseStreams streams{};
    const auto a = part(first);
    streams.parts[0][0] = a[0];
    streams.parts[0][1] = a[1];
    if (second.codes != nullptr) {
        const auto b = part(second);
        streams.parts[1][0] = b[0];
        streams.parts[1][1] = b[1];
    }
    streams.split = first.rows / 16U;
    return streams;
}

// T <= 16 (one token tile): a single-sequence verification width is at most 8.
template <bool kAccumulate, bool kSiluPair>
struct Fp8Lut4SmallT {
    static constexpr std::uint32_t kTiles = 1U;
    // Before its first barrier the body writes only the codebook, which the previous CTA reads
    // only before its K-split barrier.
    static constexpr bool lds_reuse_safe = true;
    static constexpr auto kernel = &linear::fp8lut4_kernels::fp8lut4_small_t_kernel<
        kAccumulate, kSiluPair, linear::fp8lut4::kVerifySplit, linear::fp8lut4::kVerifySteps,
        kTiles>;
    static constexpr std::uint32_t launch_threads = 32U * linear::fp8lut4::kVerifySplit;
    static constexpr std::uint32_t threads = launch_threads;
    using Shared =
        linear::fp8lut4_kernels::SmallTShared<kSiluPair, linear::fp8lut4::kVerifySplit, kTiles>;
    template <class Params>
    static uint3 grid(uint3 captured, const Params&) {
        return captured;
    }
    template <class Params>
    static PhaseStreams streams(const Params& params) {
        const linear::fp8lut4_kernels::KernelArgs& a = slot_value<0>(params);
        return fp8lut4_streams(a.weight, a.second);
    }
    template <class Cta>
    __device__ static void run(const Cta& cta, Shared& shared,
                               const linear::fp8lut4_kernels::KernelArgs& a) {
        linear::fp8lut4_kernels::small_t_body<kAccumulate, kSiluPair,
                                              linear::fp8lut4::kVerifySplit,
                                              linear::fp8lut4::kVerifySteps, kTiles>(
            cta, shared, a);
    }
};

template <linear::fp8_small_t_kernels::Epilogue Kind>
struct Fp8SmallT : CapturedGrid {
    static constexpr std::uint32_t kTiles = 1U;
    static constexpr auto kernel = &linear::fp8_small_t_kernels::fp8_small_t_kernel<Kind, kTiles>;
    static constexpr std::uint32_t launch_threads = 32U * linear::fp8_small_t_kernels::kSplit;
    static constexpr std::uint32_t threads = launch_threads;
    using Shared = linear::fp8_small_t_kernels::SmallTShared<kTiles>;
    template <class Params>
    static PhaseStreams streams(const Params& params) {
        const linear::fp8_small_t_kernels::KernelArgs& a = slot_value<0>(params);
        // Row block c: sixteen contiguous E4M3 rows.
        PhaseStreams streams{};
        const std::uint64_t bytes = 16U * static_cast<std::uint64_t>(a.columns);
        streams.parts[0][0] = {reinterpret_cast<std::uint64_t>(a.first.codes), bytes};
        if (Kind == linear::fp8_small_t_kernels::Epilogue::PairSplit)
            streams.parts[1][0] = {reinterpret_cast<std::uint64_t>(a.second.codes), bytes};
        streams.split = a.first.rows / 16U;
        return streams;
    }
    template <class Cta>
    __device__ static void run(const Cta& cta, Shared& shared,
                               const linear::fp8_small_t_kernels::KernelArgs& a) {
        linear::fp8_small_t_kernels::small_t_body<Kind, kTiles>(cta, shared, a);
    }
};

// A launch sized as its work CTAs followed by warm CTAs (core/cache_warm.h): the virtual grid is
// the work CTAs (argument `WorkSlot`). The warm CTAs are dropped: the persistent kernel prefetches
// each block's weights of the next phase at the grid barrier instead (PhaseStreams).
template <std::size_t WorkSlot>
struct WorkGrid {
    static constexpr bool lds_reuse_safe = false;
    template <class Params>
    static uint3 grid(uint3, const Params& params) {
        return uint3{static_cast<std::uint32_t>(slot_value<WorkSlot>(params)), 1U, 1U};
    }
};

struct QuantizeActivation : WorkGrid<8> {
    static constexpr auto kernel = &linear::fp8_activation_kernels::fp8_quantize_activation_kernel;
    static constexpr std::uint32_t launch_threads = linear::fp8_activation_kernels::kBlockSize;
    static constexpr std::uint32_t threads = kBlockThreads;
    using Shared = linear::fp8_activation_kernels::QuantizeShared;
    template <class Cta, class... A>
    __device__ static void run(const Cta& cta, Shared& shared, const A&... a) {
        linear::fp8_activation_kernels::quantize_activation_body(cta, shared, a...);
    }
};

struct QuantizeNormalized : WorkGrid<9> {
    static constexpr auto kernel = &linear::fp8_activation_kernels::fp8_quantize_normalized_kernel;
    static constexpr std::uint32_t launch_threads = linear::fp8_activation_kernels::kNormThreads;
    static constexpr std::uint32_t threads = kBlockThreads;
    using Shared = linear::fp8_activation_kernels::NormalizedShared<threads>;
    template <class Cta, class... A>
    __device__ static void run(const Cta& cta, Shared& shared, const A&... a) {
        linear::fp8_activation_kernels::quantize_normalized_body<threads>(cta, shared, a...);
    }
};

struct QuantizeGated : WorkGrid<6> {
    static constexpr std::uint32_t kVectors = 768U;
    static constexpr auto kernel =
        &linear::fp8_activation_kernels::fp8_quantize_gated_kernel<kVectors>;
    static constexpr std::uint32_t launch_threads = kVectors;
    static constexpr std::uint32_t threads = kBlockThreads;
    using Shared = linear::fp8_encode::EncodeShared<threads>;
    template <class Cta, class... A>
    __device__ static void run(const Cta& cta, Shared& shared, const A&... a) {
        linear::fp8_activation_kernels::quantize_gated_body<kVectors, threads>(cta, shared, a...);
    }
};

struct QuantizeGatedRmsNorm : WorkGrid<8> {
    static constexpr auto kernel =
        &linear::fp8_activation_kernels::fp8_quantize_gated_rmsnorm_kernel;
    static constexpr std::uint32_t launch_threads =
        linear::fp8_activation_kernels::kGatedNormThreads;
    static constexpr std::uint32_t threads = kBlockThreads;
    using Shared = linear::fp8_encode::EncodeShared<threads>;
    template <class Cta, class... A>
    __device__ static void run(const Cta& cta, Shared& shared, const A&... a) {
        linear::fp8_activation_kernels::quantize_gated_rmsnorm_body<threads>(cta, shared, a...);
    }
};

// The fold CTAs past the head CTAs are sized for the body width; the warm CTAs are dropped.
// Eight-token passes (one pass at a single-sequence verification width) fit the register budget.
// Whole-head fold items (8 tiles) fold a row in 16 CTAs, so the 48 head CTAs and a single
// sequence's fold fit one round of the grid.
struct GdnNormalizedFront {
    static constexpr int kFoldTiles = 8;
    static constexpr bool lds_reuse_safe = false;
    static constexpr auto kernel = &gdn_front::gdn_normalized_front_kernel<true>;
    static constexpr std::uint32_t launch_threads = gdn_front::kControlT1Threads;
    static constexpr std::uint32_t threads = kBlockThreads;
    using Shared = gdn_front::FrontShared<threads>;
    template <class Params>
    static uint3 grid(uint3, const Params& params) {
        const gdn_front::FrontFold& fold = slot_value<19>(params);
        return uint3{gdn_front::kControlHeads + gdn_front::front_fold_ctas<threads, kFoldTiles>(fold),
                     1U, 1U};
    }
    template <class Cta, class... A>
    __device__ static void run(const Cta& cta, Shared& shared, const A&... a) {
        gdn_front::normalized_front_body<true, threads, 8U, kFoldTiles>(cta, shared, a...);
    }
};

template <unsigned T>
struct GdnPairConvRecord {
    // As Fp8Lut4SmallT; the previous CTA's convolution reads only its staged rows.
    static constexpr bool lds_reuse_safe = true;
    static constexpr auto kernel = &gdn_front::gdn_fp8lut4_pair_conv_record_kernel<T>;
    static constexpr std::uint32_t launch_threads = 32U * gdn_front::kFp8Lut4PairSplit;
    static constexpr std::uint32_t threads = launch_threads;
    using Shared = gdn_front::PairConvShared<T>;
    template <class Params>
    static uint3 grid(uint3 captured, const Params&) {
        return captured;
    }
    template <class Params>
    static PhaseStreams streams(const Params& params) {
        return fp8lut4_streams(slot_value<3>(params), slot_value<4>(params));
    }
    template <class Cta, class... A>
    __device__ static void run(const Cta& cta, Shared& shared, const A&... a) {
        gdn_front::fp8lut4_pair_conv_record_body<T>(cta, shared, a...);
    }
};

// One whole-block virtual CTA per (sequence, head) staging each of at most eight tokens once,
// three value rows per row slot (the launch's four 256-thread row tiles per head each stage
// every token and fit one per block). Hosted up to a verification width of 8.
struct GdnRecord {
    static constexpr int kMaxWidth = 8;
    static constexpr bool lds_reuse_safe = false;
    static constexpr auto kernel = &recurrence::record_kernel<false>;
    static constexpr std::uint32_t launch_threads = recurrence::kOrdinaryBlock;
    static constexpr std::uint32_t threads = kBlockThreads;
    using Shared = recurrence::RecordHeadShared<threads, kMaxWidth>;
    template <class Params>
    static uint3 grid(uint3, const Params& params) {
        return uint3{slot_value<18>(params) / recurrence::kOrdinaryRowTiles, 1U, 1U};
    }
    template <class Params>
    static bool admits(const Params& params) {
        return slot_value<14>(params) <= kMaxWidth;  // width
    }
    template <class Cta, class... A>
    __device__ static void run(const Cta& cta, Shared& shared, const A&... a) {
        recurrence::record_head_body<threads, kMaxWidth>(cta, shared, a...);
    }
};

template <rope_kernels::Profile P>
struct QkNormRope : CapturedGrid {
    static constexpr bool lds_reuse_safe = true;  // no LDS
    static constexpr auto kernel = &rope_kernels::qk_norm_rope_kernel<P>;
    static constexpr std::uint32_t launch_threads = rope_kernels::kQkNormRopeWaves * kWaveSize;
    static constexpr std::uint32_t threads = launch_threads;
    using Shared = NoShared;
    template <class Cta, class... A>
    __device__ static void run(const Cta& cta, Shared&, const A&... a) {
        rope_kernels::qk_norm_rope_body<P>(cta, a...);
    }
};

using linear::fp8_small_t_kernels::Epilogue;
// The kernels of a single-sequence decode graph (verification width at most 8).
using Entries = std::tuple<
    Fp8Lut4SmallT<false, false>, Fp8Lut4SmallT<true, false>, Fp8Lut4SmallT<false, true>,
    Fp8SmallT<Epilogue::Store>, Fp8SmallT<Epilogue::PairSplit>, Fp8SmallT<Epilogue::Residual>,
    QuantizeActivation, QuantizeNormalized, QuantizeGated, QuantizeGatedRmsNorm,
    GdnNormalizedFront, GdnPairConvRecord<4U>, GdnPairConvRecord<5U>, GdnPairConvRecord<6U>,
    GdnPairConvRecord<7U>, GdnPairConvRecord<8U>, GdnRecord,
    QkNormRope<rope_kernels::Profile::Text1D>, QkNormRope<rope_kernels::Profile::TextMrope>>;
constexpr std::size_t kEntryCount = std::tuple_size_v<Entries>;

template <class E, std::size_t I = 0>
constexpr std::size_t entry_index() {
    if constexpr (std::is_same_v<E, std::tuple_element_t<I, Entries>>) return I;
    else return entry_index<E, I + 1U>();
}

template <class E>
using EntryParams = typename KernelParams<std::remove_const_t<decltype(E::kernel)>>::type;

constexpr std::size_t align16(std::size_t bytes) { return (bytes + 15U) / 16U * 16U; }

// Each block gets kBlockLds of the WGP's 64 KiB (less the group barriers and phase slots).
constexpr std::uint32_t kMaximumGroups = kBlockThreads / 128U;
constexpr std::size_t kLdsBudget = kBlockLds - 64U - 2U * 512U;  // group barriers, phase slots

// Groups per block: as many body-wide groups as the block's waves and LDS budget hold.
template <class E>
constexpr std::uint32_t entry_groups() {
    static_assert(E::threads % kWaveSize == 0U && E::threads <= kBlockThreads);
    constexpr std::uint32_t by_threads = kBlockThreads / E::threads;
    if constexpr (std::is_same_v<typename E::Shared, NoShared>) {
        return by_threads;
    } else {
        constexpr std::size_t by_lds = kLdsBudget / align16(sizeof(typename E::Shared));
        static_assert(by_lds >= 1U, "an entry's LDS exceeds the persistent block budget");
        return static_cast<std::uint32_t>(std::min<std::size_t>(by_threads, by_lds));
    }
}

template <class E>
constexpr std::size_t entry_lds() {
    return std::is_same_v<typename E::Shared, NoShared>
               ? 0U
               : entry_groups<E>() * align16(sizeof(typename E::Shared));
}

template <std::size_t... I>
constexpr std::size_t max_lds(std::index_sequence<I...>) {
    return std::max({std::size_t{16}, entry_lds<std::tuple_element_t<I, Entries>>()...});
}
constexpr std::size_t kLdsBytes = max_lds(std::make_index_sequence<kEntryCount>{});
static_assert(kLdsBytes <= kLdsBudget);

// ---- Device program.
struct Phase {
    std::uint32_t entry;
    std::uint32_t grid_x;
    std::uint32_t grid_y;
    std::uint32_t grid_z;
};

// Lines of this block's first-round virtual CTAs of a phase (c = g * gridDim.x + blockIdx.x), in
// order: line `line` addresses one kCacheWarmStride line (0: past them).
struct BlockStreams {
    const PhaseStreams* streams;
    std::uint64_t lines[kMaximumGroups];  // per first-round virtual CTA
    std::uint32_t groups;

    __device__ __forceinline__ static std::uint64_t range_lines(const StreamRange& range) {
        return range.base == 0U ? 0U : range.bytes / kCacheWarmStride;
    }
    __device__ __forceinline__ explicit BlockStreams(const PhaseStreams& next) : streams(&next) {
        groups = next.groups;
        for (std::uint32_t g = 0; g < kMaximumGroups; ++g) {
            const std::uint32_t c = g * gridDim.x + blockIdx.x;
            const StreamRange* part = next.parts[c < next.split ? 0U : 1U];
            lines[g] = g < groups && c < next.total
                ? range_lines(part[0]) + range_lines(part[1])
                : 0U;
        }
    }
    __device__ __forceinline__ std::uint64_t address(std::uint64_t line) const {
        for (std::uint32_t g = 0; g < kMaximumGroups; ++g) {
            if (line < lines[g]) {
                const std::uint32_t c = g * gridDim.x + blockIdx.x;
                const bool second = c >= streams->split;
                const StreamRange* ranges = streams->parts[second ? 1U : 0U];
                const std::uint64_t cta = second ? c - streams->split : c;
                const std::uint64_t head = range_lines(ranges[0]);
                const StreamRange& range = line < head ? ranges[0] : ranges[1];
                return range.base + cta * range.bytes +
                       (line < head ? line : line - head) * kCacheWarmStride;
            }
            line -= lines[g];
        }
        return 0U;
    }
};

// Two waves per block stream next-phase lines while the block waits: a bounded number of lines
// in flight, so the stream does not queue the latency-bound loads of the blocks still working
// (with weightless phases streaming the coming projection, a C1 DFlash round measured 0.25 ms
// faster than one wave and 0.15 / 0.4 ms faster than three / four; 128 or 384 threads were
// 0.4-1.1 ms slower than one wave before).
constexpr std::uint32_t kPrefetchThreads = 64U;

// Code of each entry's body, [first, end) as recorded by its first execution (benign races:
// every writer stores the same value), so a block waiting at the barrier before a phase can
// prefetch the phase's instructions. Without it a phase starts on a cold instruction cache:
// every phase runs a different body of a ~0.3 MB kernel while the weight stream evicts the
// instructions from L2 (C1 DFlash rounds took ~1 ms longer).
__device__ std::uint64_t entry_code[kEntryCount][2];

// Instruction bytes of a body prefetched before its phase (covers every hosted body).
constexpr std::uint64_t kCodePrefetch = 128U * 1024U;

// The recorded code address of entry `entry` (0: not yet run), in scalar registers.
__device__ __forceinline__ std::uint64_t next_code(std::uint32_t entry, std::uint32_t bound) {
    const std::uint64_t pc = entry_code[entry][bound];
    const auto word = [](std::uint64_t bits) {
        return static_cast<std::uint64_t>(static_cast<std::uint32_t>(
            __builtin_amdgcn_readfirstlane(static_cast<std::uint32_t>(bits))));
    };
    return word(pc) | word(pc >> 32U) << 32U;
}

// The first `bytes` of entry `entry`'s body into the instruction cache (one wave; 4 KiB per
// prefetch, never past the body's recorded end, nothing before its first execution).
__device__ __forceinline__ void prefetch_code(std::uint32_t entry, std::uint64_t bytes) {
    const std::uint64_t first = next_code(entry, 0U), end = next_code(entry, 1U);
    for (std::uint64_t pc = first; pc != 0U && pc < end && pc < first + bytes; pc += 4096U)
        asm volatile("s_prefetch_inst %0, 0x0, null, 31" ::"s"(pc));
}

// Grid-wide barrier: every block's prior global writes are visible to every block afterwards.
// Thread 0 of each block arrives on a monotonic 64-bit arrival count, which every launch leaves a
// multiple of the grid size; the barrier opens when the count reaches the end of the arrival's
// round, so the last arrival's own atomic opens it (one atomic on the critical path).
// While a block waits it prefetches the next phase's instructions and streams its first-round
// weight lines of the next phase (`next`) into the caches, one line per prefetch thread per batch
// (a touched dword fills its line, core/cache_warm.h), until the barrier opens: the wait of a
// block a phase leaves idle, or that finishes early, overlaps the next phase's memory stream. A
// pure cache hint: the loaded values are discarded.
__device__ __forceinline__ void grid_barrier(std::uint64_t* arrivals, const PhaseStreams& next,
                                             std::uint32_t next_entry) {
    __shared__ std::uint32_t open;
    __shared__ std::uint64_t target;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0U) {
        const std::uint64_t ticket =
            __hip_atomic_fetch_add(arrivals, 1ULL, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        const std::uint64_t end = (ticket / gridDim.x + 1U) * gridDim.x;
        target = end;
        open = ticket + 1U == end ? 1U : 0U;
    }
    // The next phase's instructions.
    if (threadIdx.x < kWaveSize) prefetch_code(next_entry, kCodePrefetch);
    if (next.groups != 0U) {
        __syncthreads();
        const BlockStreams streams(next);
        for (std::uint64_t line = threadIdx.x; open == 0U; line += kPrefetchThreads) {
            const std::uint64_t address =
                threadIdx.x < kPrefetchThreads ? streams.address(line) : 0U;
            const std::uint32_t loaded =
                address == 0U
                    ? 0U
                    : *persistent::global(reinterpret_cast<const std::uint32_t*>(address));
            asm volatile("" ::"v"(loaded));
            // Past the block's lines: wait without loading.
            if (__syncthreads_and(address == 0U)) break;
            if (threadIdx.x == 0U &&
                __hip_atomic_load(arrivals, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= target)
                open = 1U;
            __syncthreads();
        }
    }
    if (threadIdx.x == 0U) {
        while (__hip_atomic_load(arrivals, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target)
            __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

template <std::size_t... I>
constexpr std::size_t max_args(std::index_sequence<I...>) {
    return std::max({align16(sizeof(EntryParams<std::tuple_element_t<I, Entries>>))...});
}
constexpr std::size_t kArgsBytes = max_args(std::make_index_sequence<kEntryCount>{});

// A phase's descriptor and captured arguments: the program's record of the phase, staged in LDS
// before the grid barrier that precedes the phase (double-buffered), so the phase starts with
// them as a launch starts with its kernarg segment.
struct PhaseSlot {
    Phase phase;
    PhaseStreams streams;
    __attribute__((aligned(16))) std::uint8_t args[kArgsBytes];
};
constexpr std::uint32_t kSlotWords = sizeof(PhaseSlot) / 4U;
static_assert(sizeof(PhaseSlot) % 4U == 0U && kSlotWords <= kBlockThreads);
static_assert(kLdsBytes + 2U * sizeof(PhaseSlot) + 64U <= kBlockLds,
              "kBlocksPerWgp persistent blocks must fit the WGP's 64 KiB of LDS");

struct Program {
    const PhaseSlot* phases = nullptr;
    std::uint64_t* arrivals = nullptr;  // grid barrier arrival count
    std::uint32_t phase_count = 0;
    std::uint32_t first_entry = 0;  // entry of phase 0
};

// Namespace-scope LDS: the inlined bodies address it as LDS, not through generic pointers.
__shared__ __attribute__((aligned(16))) std::uint8_t phase_lds[kLdsBytes];
__shared__ persistent::GroupBarrier group_barriers[kMaximumGroups];
__shared__ PhaseSlot phase_slots[2];

// Thread t's word of phase `index`'s record (threads past the record read nothing).
__device__ __forceinline__ std::uint32_t fetch_phase_word(const Program& program,
                                                          std::uint32_t index) {
    return threadIdx.x < kSlotWords
        ? persistent::global(reinterpret_cast<const std::uint32_t*>(program.phases + index))
              [threadIdx.x]
        : 0U;
}

__device__ __forceinline__ void store_phase_word(std::uint32_t slot, std::uint32_t word) {
    if (threadIdx.x < kSlotWords)
        reinterpret_cast<std::uint32_t*>(&phase_slots[slot])[threadIdx.x] = word;
}

__device__ __forceinline__ std::uint64_t program_counter() {
    std::uint64_t pc;
    asm volatile("s_getpc_b64 %0" : "=s"(pc));
    return pc;
}

// Every body is inlined into the kernel (an out-of-line body would save and restore the
// callee-saved VGPRs it uses on every call); bodies only meet at the dispatch, so the kernel's
// register count is the largest body's.
template <class E>
__device__ __forceinline__ void run_phase(std::uint32_t slot) {
    constexpr std::uint32_t kGroups = entry_groups<E>();
    constexpr bool kWholeBlock = kGroups == 1U && E::threads == kBlockThreads;
    const PhaseSlot& staged = phase_slots[slot];
    const auto& params = *reinterpret_cast<const EntryParams<E>*>(staged.args);
    const uint3 grid{staged.phase.grid_x, staged.phase.grid_y, staged.phase.grid_z};
    const std::uint32_t plane = grid.x * grid.y;
    const std::uint32_t total = plane * grid.z;
    if (threadIdx.x == 0U) {
        const std::uint64_t pc = program_counter();
        if (entry_code[entry_index<E>()][0] != pc) entry_code[entry_index<E>()][0] = pc;
    }
    const std::uint32_t group = threadIdx.x / E::threads;
    if (group >= kGroups) return;
    using Shared = typename E::Shared;
    Shared& shared = *reinterpret_cast<Shared*>(phase_lds + group * align16(sizeof(Shared)));
    // Opaque per phase: thread-derived address arithmetic of the inlined bodies must not be
    // hoisted out of the phase loop, where it would stay live across every phase.
    std::uint32_t thread = threadIdx.x - group * E::threads;
    asm volatile("" : "+v"(thread));
    const persistent::PhaseCta<kWholeBlock> first{thread, E::threads,
                                                  uint3{}, grid, &group_barriers[group],
                                                  E::threads / kWaveSize};
    for (std::uint32_t index = group * gridDim.x + blockIdx.x; index < total;
         index += gridDim.x * kGroups) {
        persistent::PhaseCta<kWholeBlock> cta = first;
        cta.block_ = uint3{index % grid.x, index % plane / grid.x, index / plane};
        [&]<std::size_t... J>(std::index_sequence<J...>) {
            E::run(cta, shared, body_arg(slot_value<J>(params))...);
        }(std::make_index_sequence<EntryParams<E>::size>{});
        // The group's next virtual CTA reuses its LDS.
        if constexpr (!E::lds_reuse_safe) {
            if (index + gridDim.x * kGroups < total) cta.sync();
        }
    }
    if (threadIdx.x == 0U) {
        const std::uint64_t pc = program_counter();
        if (entry_code[entry_index<E>()][1] != pc) entry_code[entry_index<E>()][1] = pc;
    }
}

template <std::size_t... I>
__device__ __forceinline__ void dispatch(std::uint32_t entry, std::uint32_t slot,
                                         std::index_sequence<I...>) {
    (void)((entry == I && (run_phase<std::tuple_element_t<I, Entries>>(slot), true)) || ...);
}

// Six waves per SIMD (VGPRs in granules of 24, at most 240) keep two blocks resident per WGP.
__global__ __launch_bounds__(kBlockThreads) __attribute__((amdgpu_waves_per_eu(kWavesPerSimd)))
void persistent_decode_kernel(Program program) {
    if (threadIdx.x < kMaximumGroups)
        group_barriers[threadIdx.x] = persistent::GroupBarrier{0U, 0U};
    // The first phase's instructions, while its record loads.
    if (threadIdx.x < kWaveSize) prefetch_code(program.first_entry, kCodePrefetch);
    store_phase_word(0U, fetch_phase_word(program, 0U));
    __syncthreads();
    for (std::uint32_t index = 0; index < program.phase_count; ++index) {
        const std::uint32_t slot = index & 1U;
        const bool next = index + 1U < program.phase_count;
        // The next record is fetched while this phase runs and staged before the barrier.
        const std::uint32_t next_word = next ? fetch_phase_word(program, index + 1U) : 0U;
        dispatch(phase_slots[slot].phase.entry, slot, std::make_index_sequence<kEntryCount>{});
        if (next) {
            store_phase_word(slot ^ 1U, next_word);
            grid_barrier(program.arrivals, phase_slots[slot ^ 1U].streams,
                         phase_slots[slot ^ 1U].phase.entry);
        }
    }
}

// ---- Host lowering.
struct HostEntry {
    const void* kernel = nullptr;
    std::uint32_t launch_threads = 0;
    std::size_t args_bytes = 0;
    void (*pack)(void* const* params, void* out) = nullptr;
    uint3 (*grid)(uint3 captured, const void* args) = nullptr;
    PhaseStreams (*streams)(const void* args) = nullptr;
    bool (*admits)(const void* args) = nullptr;
};

template <class P, std::size_t... I>
void pack_params(void* const* params, P& out, std::index_sequence<I...>) {
    (std::memcpy(&slot_value<I>(out), params[I], sizeof(typename P::template Type<I>)), ...);
}

template <class E>
HostEntry host_entry() {
    using Params = EntryParams<E>;
    static_assert(std::is_trivially_copyable_v<Params>);
    return HostEntry{
        reinterpret_cast<const void*>(E::kernel), E::launch_threads, sizeof(Params),
        [](void* const* params, void* out) {
            auto* tuple = new (out) Params{};
            pack_params(params, *tuple, std::make_index_sequence<Params::size>{});
        },
        [](uint3 captured, const void* args) {
            return E::grid(captured, *static_cast<const Params*>(args));
        },
        [](const void* args) {
            PhaseStreams streams{};
            if constexpr (requires(const Params& params) { E::streams(params); }) {
                streams = E::streams(*static_cast<const Params*>(args));
                streams.groups = entry_groups<E>();
            }
            return streams;
        },
        [](const void* args) {
            if constexpr (requires(const Params& params) { E::admits(params); })
                return E::admits(*static_cast<const Params*>(args));
            else
                return true;
        }};
}

template <std::size_t... I>
std::array<HostEntry, kEntryCount> host_entries(std::index_sequence<I...>) {
    return {host_entry<std::tuple_element_t<I, Entries>>()...};
}

const std::array<HostEntry, kEntryCount>& entries() {
    static const auto table = host_entries(std::make_index_sequence<kEntryCount>{});
    return table;
}

std::vector<hipGraphNode_t> chain_nodes(hipGraph_t graph) {
    std::size_t roots = 0;
    HIP_CHECK(hipGraphGetRootNodes(graph, nullptr, &roots));
    if (roots == 0U) return {};
    if (roots != 1U)
        throw std::logic_error("persistent_decode_lower: the Device Graph is not a single chain");
    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipGraphGetRootNodes(graph, &node, &roots));
    std::vector<hipGraphNode_t> chain;
    while (node != nullptr) {
        chain.push_back(node);
        std::size_t dependents = 0;
        HIP_CHECK(hipGraphNodeGetDependentNodes(node, nullptr, &dependents));
        if (dependents > 1U)
            throw std::logic_error("persistent_decode_lower: the Device Graph is not a single chain");
        hipGraphNode_t next = nullptr;
        if (dependents == 1U) HIP_CHECK(hipGraphNodeGetDependentNodes(node, &next, &dependents));
        node = next;
    }
    std::size_t count = 0;
    HIP_CHECK(hipGraphGetNodes(graph, nullptr, &count));
    if (count != chain.size())
        throw std::logic_error("persistent_decode_lower: the Device Graph is not a single chain");
    return chain;
}

struct Hosted {
    std::size_t entry = 0;
    hipKernelNodeParams params{};
};

// The entry hosting `node`, if any.
bool hosted(hipGraphNode_t node, Hosted& out) {
    hipGraphNodeType type{};
    HIP_CHECK(hipGraphNodeGetType(node, &type));
    if (type != hipGraphNodeTypeKernel) return false;
    hipKernelNodeParams params{};
    HIP_CHECK(hipGraphKernelNodeGetParams(node, &params));
    const auto& table = entries();
    for (std::size_t index = 0; index < table.size(); ++index) {
        if (table[index].kernel != params.func) continue;
        if (params.kernelParams == nullptr || params.extra != nullptr ||
            params.sharedMemBytes != 0U || params.blockDim.x != table[index].launch_threads ||
            params.blockDim.y != 1U || params.blockDim.z != 1U)
            throw std::logic_error("persistent_decode_lower: unexpected launch of a hosted kernel");
        alignas(16) std::uint8_t args[kArgsBytes];
        table[index].pack(params.kernelParams, args);
        if (!table[index].admits(args)) return false;
        out = Hosted{index, params};
        return true;
    }
    return false;
}

} // namespace

DeviceBuffer persistent_decode_lower(hipGraph_t graph) {
    if (graph == nullptr) throw std::invalid_argument("persistent_decode_lower: null graph");
    const std::vector<hipGraphNode_t> chain = chain_nodes(graph);

    // Runs of at least two consecutive hosted launches.
    struct Run {
        std::size_t first = 0;
        std::size_t count = 0;
    };
    std::vector<Hosted> launches(chain.size());
    std::vector<bool> is_hosted(chain.size(), false);
    for (std::size_t index = 0; index < chain.size(); ++index)
        is_hosted[index] = hosted(chain[index], launches[index]);
    std::vector<Run> runs;
    for (std::size_t index = 0; index < chain.size();) {
        if (!is_hosted[index]) {
            ++index;
            continue;
        }
        std::size_t end = index;
        while (end < chain.size() && is_hosted[end]) ++end;
        if (end - index >= 2U) runs.push_back(Run{index, end - index});
        index = end;
    }
    if (runs.empty()) return DeviceBuffer{};

    // Device image: the barrier words, then one PhaseSlot record per phase.
    const auto& table = entries();
    std::size_t phase_count = 0;
    for (const Run& run : runs) phase_count += run.count;
    constexpr std::size_t kPhaseOffset = 256U;
    static_assert(alignof(PhaseSlot) <= kPhaseOffset);
    std::vector<std::uint8_t> image(kPhaseOffset + phase_count * sizeof(PhaseSlot), 0U);
    auto* records = reinterpret_cast<PhaseSlot*>(image.data() + kPhaseOffset);
    std::size_t record = 0;
    for (const Run& run : runs) {
        for (std::size_t index = run.first; index < run.first + run.count; ++index, ++record) {
            const Hosted& launch = launches[index];
            const HostEntry& entry = table[launch.entry];
            entry.pack(launch.params.kernelParams, records[record].args);
            const dim3 captured = launch.params.gridDim;
            const uint3 grid = entry.grid(uint3{captured.x, captured.y, captured.z},
                                          records[record].args);
            records[record].phase = Phase{static_cast<std::uint32_t>(launch.entry), grid.x,
                                          grid.y, grid.z};
            records[record].streams = entry.streams(records[record].args);
            records[record].streams.total = grid.x * grid.y * grid.z;
        }
        // A phase that streams no weights takes the streams of the next phase of its run that
        // does (if any): the barrier before it (a projection's tail) and its own barrier then
        // both stream the coming projection's lines, instead of the first waiting idle.
        for (std::size_t index = record - 1U; index > record - run.count; --index) {
            if (records[index - 1U].streams.groups == 0U)
                records[index - 1U].streams = records[index].streams;
        }
    }
    DeviceBuffer storage(image.size());
    storage.copy_from_host(image.data(), image.size());

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    int wgps = 0;
    HIP_CHECK(hipDeviceGetAttribute(&wgps, hipDeviceAttributeMultiprocessorCount, device));
    int resident = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident, persistent_decode_kernel,
                                                           kBlockThreads, 0));
    if (resident < static_cast<int>(kBlocksPerWgp))
        throw std::logic_error("persistent_decode_lower: " + std::to_string(resident) +
                               " persistent blocks per WGP are resident, expected " +
                               std::to_string(kBlocksPerWgp));

    auto* base = static_cast<std::uint8_t*>(storage.data());
    std::size_t first_phase = 0;
    for (const Run& run : runs) {
        Program program{reinterpret_cast<const PhaseSlot*>(base + kPhaseOffset) + first_phase,
                        reinterpret_cast<std::uint64_t*>(base),
                        static_cast<std::uint32_t>(run.count),
                        static_cast<std::uint32_t>(launches[run.first].entry)};
        first_phase += run.count;
        void* kernel_args[] = {&program};
        hipKernelNodeParams params{};
        params.func = reinterpret_cast<void*>(&persistent_decode_kernel);
        params.gridDim = dim3(static_cast<std::uint32_t>(wgps) * kBlocksPerWgp);
        params.blockDim = dim3(kBlockThreads);
        params.sharedMemBytes = 0U;
        params.kernelParams = kernel_args;
        params.extra = nullptr;

        std::vector<hipGraphNode_t> predecessors;
        std::size_t dependencies = 0;
        HIP_CHECK(hipGraphNodeGetDependencies(chain[run.first], nullptr, &dependencies));
        predecessors.resize(dependencies);
        if (dependencies != 0U)
            HIP_CHECK(hipGraphNodeGetDependencies(chain[run.first], predecessors.data(),
                                                  &dependencies));
        const std::size_t last = run.first + run.count - 1U;
        hipGraphNode_t successor = last + 1U < chain.size() ? chain[last + 1U] : nullptr;
        hipGraphNode_t node = nullptr;
        HIP_CHECK(hipGraphAddKernelNode(&node, graph, predecessors.data(), predecessors.size(),
                                        &params));
        for (std::size_t index = run.first; index <= last; ++index)
            HIP_CHECK(hipGraphDestroyNode(chain[index]));
        if (successor != nullptr) HIP_CHECK(hipGraphAddDependencies(graph, &node, &successor, 1U));
    }
    return storage;
}

} // namespace ninfer::ops
