#include "targets/qwen3_8_27b/impl/variant.h"

using ninfer::QType;
using Variant = ninfer::targets::qwen3_8_27b::detail::Variant;

#if NINFER_R9700_ATTENTION_Q4_PAIR_T1_CANDIDATE
static_assert(Variant::ExecutionState::attention_q4_pair_t1_selected(
    1U, QType::Q4G64_F16S, QType::Q4G64_F16S));
#else
static_assert(!Variant::ExecutionState::attention_q4_pair_t1_selected(
    1U, QType::Q4G64_F16S, QType::Q4G64_F16S));
#endif
static_assert(!Variant::ExecutionState::attention_q4_pair_t1_selected(
    2U, QType::Q4G64_F16S, QType::Q4G64_F16S));
static_assert(!Variant::ExecutionState::attention_q4_pair_t1_selected(
    3U, QType::Q4G64_F16S, QType::Q4G64_F16S));
static_assert(!Variant::ExecutionState::attention_q4_pair_t1_selected(
    4U, QType::Q4G64_F16S, QType::Q4G64_F16S));
static_assert(!Variant::ExecutionState::attention_q4_pair_t1_selected(
    1U, QType::W8G32_F16S, QType::Q4G64_F16S));
static_assert(!Variant::ExecutionState::attention_q4_pair_t1_selected(
    1U, QType::Q4G64_F16S, QType::W8G32_F16S));

int main() { return 0; }
