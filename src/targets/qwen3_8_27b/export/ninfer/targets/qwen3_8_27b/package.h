#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3/frontend.h>
#include <ninfer/targets/qwen3/runtime.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::qwen3_8_27b {

struct Package;

namespace detail {

struct Variant;

enum class WeightsProfile : std::uint8_t {
    R9700W8G32Candidate,
    R9700W8Bf16EmbeddingEvaluation,
    R9700W8Bf16AttentionQueryKeyEvaluation,
    R9700W8Bf16AttentionValueOutputEvaluation,
    R9700W8Bf16GdnQueryKeyEvaluation,
    R9700Q4G64Evaluation,
    R9700Q4G64Fp8FourRoleN16K16Evaluation,
    R9700Q4W8Evaluation,
    R9700Q4G64DFlash2Q4Evaluation,
    R9700Q4W8MseDFlash2Q4Evaluation,
    R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation,
};

using Frontend       = qwen3::Frontend;
using PreparedPrompt = qwen3::PreparedPrompt;
using OutputSession  = qwen3::OutputSession;

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_8_27b::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_8_27b::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "qwen3.8-27b";
    static constexpr std::string_view target_key = "qwen3_8_27b_r9700";

    using WeightsProfile  = detail::WeightsProfile;
    using LoadPlan        = detail::LoadPlan;
    using LoadedModel     = detail::LoadedModel;
    using Frontend        = detail::Frontend;
    using PreparedPrompt  = detail::PreparedPrompt;
    using OutputSession   = detail::OutputSession;
    using SequencePlanner = qwen3::SequencePlanner<detail::Variant>;
    using SequencePlan    = qwen3::SequencePlan<detail::Variant>;
    using RequestBasePlan = qwen3::RequestBasePlan<detail::Variant>;
    using RequestPlan     = qwen3::RequestPlan<detail::Variant>;
    using Program         = qwen3::Program<detail::Variant>;

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::qwen3_8_27b
} // namespace ninfer
