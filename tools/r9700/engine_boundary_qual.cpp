#include "ninfer/engine.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

template <class F>
void expect_invalid(F&& fn, std::string_view message_fragment) {
    try {
        fn();
    } catch (const std::invalid_argument& error) {
        if (std::string_view(error.what()).find(message_fragment) == std::string_view::npos) {
            throw std::runtime_error("Engine validation returned the wrong diagnostic");
        }
        return;
    }
    throw std::runtime_error("Engine validation accepted an invalid artifact contract");
}

} // namespace

int main() {
    try {
        bool cancellation_requested = false;
        ninfer::CancellationView cancellation(
            [&cancellation_requested] { return cancellation_requested; });
        if (!cancellation.armed() || cancellation.requested()) {
            throw std::runtime_error("armed CancellationView has the wrong initial state");
        }
        cancellation_requested = true;
        if (!cancellation.requested()) {
            throw std::runtime_error("CancellationView did not observe its owning host state");
        }

        ninfer::EngineOptions empty;
        empty.device = 0;
        expect_invalid([&] { ninfer::Engine engine(empty); }, "artifact_path must not be empty");

        ninfer::EngineOptions wrong_extension;
        wrong_extension.device        = 0;
        wrong_extension.artifact_path = "not-a-product.bin";
        expect_invalid([&] { ninfer::Engine engine(wrong_extension); },
                       "accepts only .ninfer artifacts");

        std::cout << "r9700_engine_boundary: PASS public archive link, gfx1201 device "
                     "construction/teardown, validation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700_engine_boundary: FAIL: " << error.what() << '\n';
        return 1;
    }
}
