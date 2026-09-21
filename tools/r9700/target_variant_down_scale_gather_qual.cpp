#include "targets/qwen3_8_27b/impl/variant.h"
#include "ninfer/ops/linear.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include <cstdio>
#include <stdexcept>

using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::targets::qwen3::TextPhase;
using State = ninfer::targets::qwen3_8_27b::detail::Variant::ExecutionState;

constexpr bool selected(unsigned bits = 8,
                        TextPhase phase = TextPhase::Verify, bool target_verify = true,
                        int route_tokens = 0, int layer = 0, unsigned tokens = 5,
                        unsigned rows = 5120, unsigned columns = 17408,
                        QType weight = QType::Q4G64_F16S,
                        QuantLayout layout = QuantLayout::Q4N16K16) {
    return State::dflash_down_scale_gather_selected(bits, phase, target_verify,
        route_tokens, layer, tokens, rows, columns, weight, layout);
}
static_assert(selected());
static_assert(selected(8, TextPhase::Verify, true, 0, 63, 6));
static_assert(!selected(4));
static_assert(!selected(8, TextPhase::Prefill));
static_assert(!selected(8, TextPhase::Verify, false));
// Compact C2/W3 and C3/W2 have T6 but are not either qualified C1 route.
static_assert(!selected(8, TextPhase::Verify, true, 3, 0, 6));
static_assert(!selected(8, TextPhase::Verify, true, 2, 0, 6));
static_assert(!selected(8, TextPhase::Verify, true, 0, -1));
static_assert(!selected(8, TextPhase::Verify, true, 0, 64));
static_assert(!selected(8, TextPhase::Verify, true, 0, 0, 4));
static_assert(!selected(8, TextPhase::Verify, true, 0, 0, 7));
static_assert(!selected(8, TextPhase::Verify, true, 0, 0, 5, 34816));
static_assert(!selected(8, TextPhase::Verify, true, 0, 0, 5, 5120, 5120));
static_assert(!selected(8, TextPhase::Verify, true, 0, 0, 5, 5120, 17408,
                        QType::W8G32_F16S));
static_assert(!selected(8, TextPhase::Verify, true, 0, 0, 5, 5120, 17408,
                        QType::Q4G64_F16S, QuantLayout::RowSplit));

int main() {
    for (int t : {5, 6}) {
        const auto ordinary = ninfer::ops::linear_workspace_capacity_bytes(
            QType::Q4G64_F16S, t, 17408);
        const auto verify = ninfer::ops::dflash_verify_down_linear_workspace_capacity_bytes(
            QType::Q4G64_F16S, t, 17408, 5120);
        if (ordinary != verify || ordinary != (t == 5 ? 89860U : 107780U))
            throw std::runtime_error("scale-gather must use the exact ordinary A8 workspace");
        ninfer::Tensor input(reinterpret_cast<void*>(0x10000000U), ninfer::DType::BF16,
                              {17408, t});
        ninfer::Tensor output(reinterpret_cast<void*>(0x40000000U), ninfer::DType::BF16,
                               {5120, t});
        ninfer::Weight weight{};
        weight.qtype=QType::Q4G64_F16S;weight.layout=QuantLayout::Q4N16K16;
        weight.ndim=2;weight.n=weight.shape[0]=weight.padded_shape[0]=5120;
        weight.k=weight.shape[1]=weight.padded_shape[1]=17408;
        weight.group=weight.group_size=64;weight.scale_dtype=ninfer::DType::FP16;
        weight.qdata=reinterpret_cast<void*>(0x20000000U);
        weight.scales=reinterpret_cast<void*>(0x30000000U);
        weight.qdata_bytes=5120U*17408U/2;weight.scale_bytes=5120U*272U*2;
        ninfer::DeviceSpan workspace{reinterpret_cast<void*>(0x50000000U),ordinary};
        for (int test=0;test<4;++test) {
            auto x=input;auto w=weight;auto span=workspace;
            if(test==0)--w.qdata_bytes;
            if(test==1)w.layout=QuantLayout::RowSplit;
            if(test==2)--span.bytes;
            if(test==3)x.dtype=ninfer::DType::FP16;
            bool rejected=false;
            try {ninfer::ops::dflash_verify_down_linear(x,w,output,span,nullptr);}
            catch(const std::invalid_argument&){rejected=true;}
            if(!rejected)throw std::runtime_error("malformed public verify-down Op accepted");
        }
    }
    std::printf("down_scale_gather_route: PASS production; C1 W5/W6 only; no workspace growth\n");
}
