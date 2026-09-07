#pragma once

namespace ninfer::ops::r9700::gdn_recurrence {

#ifndef NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE
#define NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE 0
#endif

static_assert(NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE == 0 ||
                  NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE == 1,
              "NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE must be 0 or 1");

inline constexpr bool kVerifyWaveQkCandidate =
    NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE == 1;

} // namespace ninfer::ops::r9700::gdn_recurrence
