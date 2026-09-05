#pragma once

#include <cstdint>

namespace ninfer::ops::r9700::kv {

// Compile-isolated XAttention qualification profile. The production Engine has no attention
// selector: tau=1 is the exact dense identity, while a sparse candidate is built in a separate
// tree and admitted only after its quality/retrieval/whole-Engine gates pass.
#ifndef NINFER_R9700_XATTENTION_STRIDE
#define NINFER_R9700_XATTENTION_STRIDE 16
#endif
#ifndef NINFER_R9700_XATTENTION_TAU_PERMILLE
#define NINFER_R9700_XATTENTION_TAU_PERMILLE 1000
#endif

static_assert(NINFER_R9700_XATTENTION_STRIDE == 8 ||
                  NINFER_R9700_XATTENTION_STRIDE == 16,
              "R9700 XAttention stride must be 8 or 16");
static_assert(NINFER_R9700_XATTENTION_TAU_PERMILLE > 0 &&
                  NINFER_R9700_XATTENTION_TAU_PERMILLE <= 1000,
              "R9700 XAttention tau must be in (0,1]");

inline constexpr std::uint32_t kXAttentionStride = NINFER_R9700_XATTENTION_STRIDE;
inline constexpr std::uint32_t kXAttentionTauPermille =
    NINFER_R9700_XATTENTION_TAU_PERMILLE;
inline constexpr float kXAttentionTau =
    static_cast<float>(kXAttentionTauPermille) / 1000.0F;
inline constexpr std::uint32_t kXAttentionPageSize = 64U;
inline constexpr std::uint32_t kXAttentionFindBlockSize = 128U;
inline constexpr std::uint32_t kXAttentionQueryBlockRows = 128U;

} // namespace ninfer::ops::r9700::kv
