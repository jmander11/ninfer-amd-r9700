#pragma once

#include <ninfer/targets/qwen3/frontend.h>
#include <ninfer/targets/qwen3/frontend_resources.h>
#include <ninfer/targets/qwen3/prepared_prompt.h>

namespace ninfer::targets::qwen3 {

class FrontendTestAccess {
public:
    [[nodiscard]] static Frontend create_component(const FrontendResources& resources,
                                                   bool vision_enabled = true);
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
};

} // namespace ninfer::targets::qwen3
