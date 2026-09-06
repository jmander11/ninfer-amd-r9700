#pragma once

#include <filesystem>
#include <stdexcept>
#include <system_error>

namespace ninfer::targets::qwen3::detail::prefill_tail_trace_path {

inline void require_new(const char* path) {
    std::error_code error;
    const std::filesystem::file_status status = std::filesystem::symlink_status(path, error);
    if (error && error != std::make_error_code(std::errc::no_such_file_or_directory)) {
        throw std::runtime_error("P129 prefill trace could not inspect output path");
    }
    if (std::filesystem::is_symlink(status) || std::filesystem::exists(status)) {
        throw std::runtime_error("P129 prefill trace refuses to overwrite its output");
    }
}

} // namespace ninfer::targets::qwen3::detail::prefill_tail_trace_path
