#include "ops/r9700/linear/fp8_activation.h"
#include "ops/r9700/linear/linear_execution.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace linear = ninfer::ops::r9700::linear;
namespace ops = ninfer::ops;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        static_assert(linear::Fp8ActivationOk == 0U);
        static_assert(linear::Fp8ActivationNonfinite == 1U);
        require(linear::fp8_activation_workspace_capacity_bytes(1, 1) == 516,
                "scalar FP8 activation workspace geometry is wrong");
        constexpr std::size_t kBytes = 1028;
        require(linear::fp8_activation_workspace_capacity_bytes(3, 129) == kBytes,
                "K128-padded FP8 activation workspace geometry is wrong");
        require(ops::LinearExecution::activation_workspace_capacity_bytes(3, 129) == kBytes,
                "prepared Linear activation capacity differs from the activation contract");
        require(linear::fp8_activation_workspace_capacity_bytes(0, 129) == 0 &&
                    linear::fp8_activation_workspace_capacity_bytes(3, 0) == 0 &&
                    linear::fp8_activation_workspace_capacity_bytes(
                        1, std::numeric_limits<std::uint32_t>::max()) == 0,
                "invalid FP8 activation extents were accepted");

        alignas(256) std::array<std::byte, kBytes> storage{};
        linear::Fp8ActivationWorkspace workspace{};
        require(linear::fp8_bind_activation_workspace(storage.data(), storage.size(), 3, 129,
                                                       &workspace) == hipSuccess,
                "valid FP8 activation workspace did not bind");
        auto* base = reinterpret_cast<std::uint8_t*>(storage.data());
        require(workspace.codes == base && workspace.code_bytes == 768 &&
                    workspace.scales == reinterpret_cast<float*>(base + 768) &&
                    workspace.scale_bytes == 12 &&
                    workspace.status == reinterpret_cast<std::uint32_t*>(base + 1024) &&
                    workspace.tokens == 3 && workspace.columns == 129 &&
                    workspace.padded_columns == 256,
                "bound FP8 activation planes differ from the contract");
        require(linear::fp8_bind_activation_workspace(storage.data(), storage.size() - 1, 3, 129,
                                                       &workspace) == hipErrorInvalidValue &&
                    linear::fp8_bind_activation_workspace(storage.data() + 1, storage.size() - 1,
                                                          3, 129,
                                                          &workspace) == hipErrorInvalidValue,
                "malformed FP8 activation storage was accepted");
        std::cout << "R9700 FP8 activation workspace contract passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
