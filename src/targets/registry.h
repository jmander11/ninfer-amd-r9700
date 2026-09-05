#pragma once

#include "ninfer/types.h"
#include "runtime/engine/request_memory.h"
#include <ninfer/targets/qwen3_8_27b/package.h>

#include <memory>
#include <optional>
#include <string_view>

namespace ninfer {

struct DeviceContext;

namespace targets {

[[nodiscard]] constexpr std::optional<std::string_view>
registered_target_key(std::string_view model_id) noexcept {
    if (model_id == qwen3_8_27b::Package::model_id) {
        return qwen3_8_27b::Package::target_key;
    }
    return std::nullopt;
}

struct LoadedQwen3_8_27B {
    std::unique_ptr<qwen3_8_27b::Package::LoadedModel> model;
    qwen3_8_27b::Package::Frontend frontend;

    explicit LoadedQwen3_8_27B(
        std::unique_ptr<qwen3_8_27b::Package::LoadedModel> stable_model);
    ~LoadedQwen3_8_27B();

    LoadedQwen3_8_27B(const LoadedQwen3_8_27B&)            = delete;
    LoadedQwen3_8_27B& operator=(const LoadedQwen3_8_27B&) = delete;
};

struct Qwen3_8_27BInstance {
    using Package = qwen3_8_27b::Package;

    std::unique_ptr<LoadedQwen3_8_27B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Package::Program> program;

    Qwen3_8_27BInstance(std::unique_ptr<LoadedQwen3_8_27B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Package::SequencePlan sequence_plan, DeviceContext& device);
    ~Qwen3_8_27BInstance();

    Qwen3_8_27BInstance(const Qwen3_8_27BInstance&)            = delete;
    Qwen3_8_27BInstance& operator=(const Qwen3_8_27BInstance&) = delete;
};

using ActiveTarget = std::unique_ptr<Qwen3_8_27BInstance>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
};

[[nodiscard]] ConstructedTarget construct_target(const EngineOptions& options,
                                                 DeviceContext& device);

} // namespace targets
} // namespace ninfer
