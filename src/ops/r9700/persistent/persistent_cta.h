#pragma once

// Execution context of an Op kernel body. An ordinary launch runs the body on its hardware CTA
// (LaunchCta); the persistent decode kernel (persistent_decode.h) runs the same body on a virtual
// CTA of one of its phases (PhaseCta): a group of whole waves of a resident block, with the
// virtual block and grid indices of the captured launch. Bodies read thread, block and grid
// indices and synchronize only through the context and keep CTA-shared data in caller-provided
// LDS, so both forms execute the same instructions per CTA.
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ninfer::ops::r9700::persistent {

#if defined(__HIPCC__)
// Global-address-space views of a pointer the caller knows addresses global memory. A kernel's
// pointer arguments are known global; a body the persistent kernel runs reads its arguments from
// LDS as generic pointers, whose accesses compile to FLAT instructions (which also wait on the
// LDS counter and so serialize a body's LDS reads behind its in-flight global loads).
template <class T>
using Global = __attribute__((address_space(1))) T;

template <class T>
__device__ __forceinline__ Global<T>* global(T* pointer) {
    return (Global<T>*)(pointer);  // generic to global address space
}

// 16-byte load through a global pointer.
template <class T>
__device__ __forceinline__ uint4 load16(const Global<T>* pointer) {
    using Vector = unsigned __attribute__((ext_vector_type(4)));
    const Vector value = *reinterpret_cast<const Global<const Vector>*>(pointer);
    return uint4{value.x, value.y, value.z, value.w};
}

struct LaunchCta {
    __device__ __forceinline__ std::uint32_t thread() const { return threadIdx.x; }
    __device__ __forceinline__ std::uint32_t threads() const { return blockDim.x; }
    __device__ __forceinline__ uint3 block() const {
        return uint3{blockIdx.x, blockIdx.y, blockIdx.z};
    }
    __device__ __forceinline__ uint3 grid() const {
        return uint3{gridDim.x, gridDim.y, gridDim.z};
    }
    __device__ __forceinline__ void sync() const { __syncthreads(); }
};

// Arrival counter and generation of one wave group's barrier, in LDS.
struct GroupBarrier {
    std::uint32_t arrived;
    std::uint32_t generation;
};

// Barrier among the `waves` waves of one group: every wave's prior LDS and global accesses are
// visible to the group afterwards (workgroup-scope release/acquire, as __syncthreads).
__device__ __forceinline__ void group_sync(GroupBarrier* barrier, std::uint32_t waves) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    if ((threadIdx.x & 31U) == 0U) {
        const std::uint32_t generation =
            __hip_atomic_load(&barrier->generation, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        if (__hip_atomic_fetch_add(&barrier->arrived, 1U, __ATOMIC_ACQ_REL,
                                   __HIP_MEMORY_SCOPE_WORKGROUP) == waves - 1U) {
            __hip_atomic_store(&barrier->arrived, 0U, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_WORKGROUP);
            __hip_atomic_store(&barrier->generation, generation + 1U, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_WORKGROUP);
        } else {
            while (__hip_atomic_load(&barrier->generation, __ATOMIC_ACQUIRE,
                                     __HIP_MEMORY_SCOPE_WORKGROUP) == generation)
                __builtin_amdgcn_s_sleep(0);
        }
    }
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

// kWholeBlock: the group is the resident block and synchronizes with the hardware barrier.
template <bool kWholeBlock>
struct PhaseCta {
    std::uint32_t thread_;
    std::uint32_t threads_;
    uint3 block_;
    uint3 grid_;
    GroupBarrier* barrier_;
    std::uint32_t waves_;

    __device__ __forceinline__ std::uint32_t thread() const { return thread_; }
    __device__ __forceinline__ std::uint32_t threads() const { return threads_; }
    __device__ __forceinline__ uint3 block() const { return block_; }
    __device__ __forceinline__ uint3 grid() const { return grid_; }
    __device__ __forceinline__ void sync() const {
        if constexpr (kWholeBlock) {
            __syncthreads();
        } else {
            group_sync(barrier_, waves_);
        }
    }
};
#endif

} // namespace ninfer::ops::r9700::persistent
