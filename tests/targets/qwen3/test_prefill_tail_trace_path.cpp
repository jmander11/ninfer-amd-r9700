#include "targets/qwen3/impl/runtime/prefill_tail_trace_path.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

template <class Function>
void require_rejected(Function&& function, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, message);
}

} // namespace

int main() try {
    namespace fs = std::filesystem;
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() /
                          ("ninfer-prefill-tail-path-" + std::to_string(nonce));
    fs::create_directory(root);
    const fs::path fresh = root / "fresh.json";
    ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
        fresh.c_str());

    {
        std::ofstream output(fresh);
        output << "retained\n";
    }
    require_rejected(
        [&] {
            ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
                fresh.c_str());
        },
        "regular output was not rejected");

    const fs::path dangling = root / "dangling.json";
    fs::create_symlink(root / "absent-target.json", dangling);
    require_rejected(
        [&] {
            ninfer::targets::qwen3::detail::prefill_tail_trace_path::require_new(
                dangling.c_str());
        },
        "dangling output symlink was not rejected");
    fs::remove(dangling);
    fs::remove(fresh);
    fs::remove(root);
    std::cout << "prefill tail trace output path PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "test_prefill_tail_trace_path: " << error.what() << '\n';
    return 1;
}
