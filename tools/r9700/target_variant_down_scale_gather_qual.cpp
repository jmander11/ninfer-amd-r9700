#include "targets/qwen3_8_27b/impl/variant.h"
#include "ninfer/ops/linear.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include <cstdio>
#include <stdexcept>

using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::targets::qwen3::TextPhase;
using State = ninfer::targets::qwen3_8_27b::detail::Variant::ExecutionState;

constexpr bool selected(bool enabled = true, unsigned bits = 8,
                        TextPhase phase = TextPhase::Verify, bool target_verify = true,
                        int route_tokens = 0, int layer = 0, unsigned tokens = 5,
                        unsigned rows = 5120, unsigned columns = 17408,
                        QType weight = QType::Q4G64_F16S,
                        QuantLayout layout = QuantLayout::Q4N16K16) {
    return State::dflash_down_scale_gather_selected(enabled, bits, phase, target_verify,
        route_tokens, layer, tokens, rows, columns, weight, layout);
}
static_assert(selected());
static_assert(selected(true, 8, TextPhase::Verify, true, 0, 63, 6));
static_assert(!selected(false));
static_assert(!selected(true, 4));
static_assert(!selected(true, 8, TextPhase::Prefill));
static_assert(!selected(true, 8, TextPhase::Verify, false));
// Compact C2/W3 and C3/W2 have T6 but are not either qualified C1 route.
static_assert(!selected(true, 8, TextPhase::Verify, true, 3, 0, 6));
static_assert(!selected(true, 8, TextPhase::Verify, true, 2, 0, 6));
static_assert(!selected(true, 8, TextPhase::Verify, true, 0, -1));
static_assert(!selected(true, 8, TextPhase::Verify, true, 0, 64));
static_assert(!selected(true, 8, TextPhase::Verify, true, 0, 0, 4));
static_assert(!selected(true, 8, TextPhase::Verify, true, 0, 0, 7));
static_assert(!selected(true, 8, TextPhase::Verify, true, 0, 0, 5, 34816));
static_assert(!selected(true, 8, TextPhase::Verify, true, 0, 0, 5, 5120, 5120));
static_assert(!selected(true, 8, TextPhase::Verify, true, 0, 0, 5, 5120, 17408,
                        QType::W8G32_F16S));
static_assert(!selected(true, 8, TextPhase::Verify, true, 0, 0, 5, 5120, 17408,
                        QType::Q4G64_F16S, QuantLayout::RowSplit));

int main() {
    for (int t : {5, 6}) {
        const auto ordinary = ninfer::ops::linear_workspace_capacity_bytes(
            QType::Q4G64_F16S, t, 17408);
        const auto verify = ninfer::ops::dflash_verify_down_linear_workspace_capacity_bytes(
            QType::Q4G64_F16S, t, 17408, 5120);
        if (ordinary != verify || ordinary != (t == 5 ? 89860U : 107780U))
            throw std::runtime_error("scale-gather must use the exact ordinary A8 workspace");
    }
    std::printf("down_scale_gather_route: PASS selector=%d; C1 W5/W6 only; no workspace growth\n",
        int(ninfer::ops::r9700::linear::kDFlashDownScaleGatherCandidateEnabled));
}
