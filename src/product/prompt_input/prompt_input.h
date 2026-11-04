#pragma once

#include "ninfer/types.h"

#include <filesystem>
#include <string>

namespace ninfer::product {

[[nodiscard]] PromptInput prompt_from_text(std::string text, bool enable_thinking);
// Local media paths are relative to the containing messages file.
[[nodiscard]] PromptInput prompt_from_messages(const std::filesystem::path& path,
                                               bool enable_thinking, bool vision_enabled);

} // namespace ninfer::product
