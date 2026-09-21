#include "targets/qwen3_8_27b/impl/variant.h"

using ninfer::QType;
using Variant = ninfer::targets::qwen3_8_27b::detail::Variant;

static_assert(Variant::ExecutionState::q4_pair_c2c4_selected(
    2U, QType::Q4G64_F16S, QType::Q4G64_F16S));
static_assert(Variant::ExecutionState::q4_pair_c2c4_selected(
    3U, QType::Q4G64_F16S, QType::Q4G64_F16S));
static_assert(Variant::ExecutionState::q4_pair_c2c4_selected(
    4U, QType::Q4G64_F16S, QType::Q4G64_F16S));
static_assert(!Variant::ExecutionState::q4_pair_c2c4_selected(
    0U, QType::Q4G64_F16S, QType::Q4G64_F16S));
static_assert(!Variant::ExecutionState::q4_pair_c2c4_selected(
    1U, QType::Q4G64_F16S, QType::Q4G64_F16S));
static_assert(!Variant::ExecutionState::q4_pair_c2c4_selected(
    5U, QType::Q4G64_F16S, QType::Q4G64_F16S));
static_assert(!Variant::ExecutionState::q4_pair_c2c4_selected(
    2U, QType::W8G32_F16S, QType::Q4G64_F16S));
static_assert(!Variant::ExecutionState::q4_pair_c2c4_selected(
    2U, QType::Q4G64_F16S, QType::W8G32_F16S));

int main() { return 0; }
