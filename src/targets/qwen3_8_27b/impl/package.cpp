#include <ninfer/targets/qwen3_8_27b/package.h>
#include <ninfer/targets/qwen3/frontend_resources.h>
#include <ninfer/targets/qwen3/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_8_27b/impl/load/bindings.h"
#include "targets/qwen3_8_27b/impl/variant.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_27b::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::qwen3_8_27b::detail

namespace ninfer::targets::qwen3_8_27b {
Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    const auto features = qwen3::startup_features(options);
    auto plan = detail::bind_artifact(binder, weights_profile, features);
    plan.bindings.linear_prepared_widths = detail::Variant::ExecutionState::eager_widths(
        std::min(options.prefill_chunk, options.max_context), options.max_concurrency,
        features.mtp() ? options.speculative.draft_tokens + 1U : 0U,
        features.dflash() ? qwen3::dflash_verify_width<detail::DFlashConfig>(
            options.speculative.draft_tokens, options.speculative.dflash_verify_width) : 0U);
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile, std::move(plan)));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3::make_frontend(model.impl_->data.frontend,
                                  model.impl_->data.runtime.features.vision);
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return qwen3::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device,
                        std::unique_ptr<HostPinnedArena> kv_ram_arena) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device,
        std::move(kv_ram_arena));
}

} // namespace ninfer::targets::qwen3_8_27b
