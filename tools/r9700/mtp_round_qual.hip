#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/mtp_round.h"
#include "ninfer/types.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::int32_t kMaxContext = 97;
constexpr std::int32_t kSentinel = static_cast<std::int32_t>(0x5a17c3e9U);

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void require(bool condition, const std::string& message) {
    if (!condition) { fail(message); }
}

template <class Function>
void require_invalid(Function&& function, const char* label) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    fail(std::string(label) + " did not reject its invalid contract");
}

struct I32Device {
    explicit I32Device(std::size_t count) : storage(count * sizeof(std::int32_t)), count(count) {}

    void upload(const std::vector<std::int32_t>& values, hipStream_t stream) {
        require(values.size() == count, "upload extent mismatch");
        storage.copy_from_host_async(values.data(), values.size() * sizeof(std::int32_t), stream);
    }

    [[nodiscard]] std::vector<std::int32_t> download(hipStream_t stream) const {
        std::vector<std::int32_t> values(count);
        storage.copy_to_host_async(values.data(), values.size() * sizeof(std::int32_t), stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        return values;
    }

    [[nodiscard]] std::int32_t* data() const {
        return static_cast<std::int32_t*>(storage.data());
    }

    ninfer::DeviceBuffer storage;
    std::size_t count;
};

ninfer::Tensor vector_tensor(I32Device& buffer, std::int32_t count) {
    return {buffer.data(), ninfer::DType::I32, {count}};
}

ninfer::Tensor matrix_tensor(I32Device& buffer, std::int32_t rows, std::int32_t columns) {
    return {buffer.data(), ninfer::DType::I32, {rows, columns}};
}

ninfer::Tensor pitched_tensor(I32Device& buffer, std::int32_t rows, std::int32_t columns,
                              std::int32_t stride) {
    ninfer::Tensor tensor(buffer.data(), ninfer::DType::I32, {rows, columns});
    tensor.nb[1] = static_cast<std::int64_t>(stride) * sizeof(std::int32_t);
    tensor.nb[2] = tensor.nb[1] * columns;
    tensor.nb[3] = tensor.nb[2];
    return tensor;
}

void require_exact(const std::vector<std::int32_t>& actual,
                   const std::vector<std::int32_t>& expected, const std::string& label) {
    require(actual.size() == expected.size(), label + " extent mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            fail(label + " mismatch at " + std::to_string(index) + ": got " +
                 std::to_string(actual[index]) + ", expected " +
                 std::to_string(expected[index]));
        }
    }
}

void qualify_case(std::int32_t k, std::int32_t batch, hipStream_t stream) {
    const std::int32_t width = k + 1;
    const std::int32_t steps = std::max(k - 1, 1);
    const std::int32_t pitch = batch + 3;

    std::vector<std::int32_t> verify(static_cast<std::size_t>(width * batch));
    std::vector<std::int32_t> anchors(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> accepted(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> frontiers(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> budgets(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> licensed(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> rope_deltas(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> expected_alignment(verify.size());
    std::vector<std::int32_t> expected_extents(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> expected_positions(static_cast<std::size_t>(pitch * steps),
                                                  kSentinel);
    std::vector<std::int32_t> expected_rope(static_cast<std::size_t>(pitch * steps), kSentinel);
    std::vector<std::int32_t> expected_valid(static_cast<std::size_t>(pitch * steps), kSentinel);

    for (std::int32_t row = 0; row < batch; ++row) {
        accepted[static_cast<std::size_t>(row)] =
            row == 0 ? 0 : (row == batch - 1 ? k : (row * 3 + k) % (k + 1));
        licensed[static_cast<std::size_t>(row)] = accepted[static_cast<std::size_t>(row)] + 1;
        anchors[static_cast<std::size_t>(row)] = 90000 + k * 101 + row * 17;
        frontiers[static_cast<std::size_t>(row)] =
            row == batch - 1 ? kMaxContext - 1 : 5 + ((row * 11 + k * 7) % 61);
        const std::int32_t extra =
            row % 4 == 0 ? 0 : (row % 4 == 1 ? 1 : (row % 4 == 2 ? 3 : k + 5));
        budgets[static_cast<std::size_t>(row)] = licensed[static_cast<std::size_t>(row)] + extra;
        rope_deltas[static_cast<std::size_t>(row)] = row * 5 - 13;
        for (std::int32_t column = 0; column < width; ++column) {
            const std::size_t offset = static_cast<std::size_t>(row * width + column);
            verify[offset] = 1000 + k * 100 + row * 19 + column * 3;
        }
        for (std::int32_t column = 0; column < width; ++column) {
            const std::size_t offset = static_cast<std::size_t>(row * width + column);
            expected_alignment[offset] =
                column < accepted[static_cast<std::size_t>(row)]
                    ? verify[static_cast<std::size_t>(row * width + column + 1)]
                    : anchors[static_cast<std::size_t>(row)];
        }
        const std::int32_t remaining = budgets[static_cast<std::size_t>(row)] -
                                       licensed[static_cast<std::size_t>(row)];
        const std::int32_t budget_extent = remaining > 1 ? remaining - 1 : 0;
        const std::int32_t context_extent =
            std::max(kMaxContext - frontiers[static_cast<std::size_t>(row)] - 1, 0);
        expected_extents[static_cast<std::size_t>(row)] =
            std::min({k, budget_extent, context_extent});
        for (std::int32_t step = 0; step < steps; ++step) {
            const std::size_t offset = static_cast<std::size_t>(step * pitch + row);
            expected_positions[offset] = frontiers[static_cast<std::size_t>(row)] + step;
            expected_rope[offset] = expected_positions[offset] + rope_deltas[static_cast<std::size_t>(row)];
            expected_valid[offset] =
                step + 1 < expected_extents[static_cast<std::size_t>(row)] ? 1 : 0;
        }
    }

    I32Device d_verify(verify.size());
    I32Device d_anchors(anchors.size());
    I32Device d_accepted(accepted.size());
    I32Device d_frontiers(frontiers.size());
    I32Device d_budgets(budgets.size());
    I32Device d_licensed(licensed.size());
    I32Device d_rope_deltas(rope_deltas.size());
    I32Device d_alignment(expected_alignment.size());
    I32Device d_extents(expected_extents.size());
    I32Device d_positions(expected_positions.size());
    I32Device d_rope(expected_rope.size());
    I32Device d_valid(expected_valid.size());
    d_verify.upload(verify, stream);
    d_anchors.upload(anchors, stream);
    d_accepted.upload(accepted, stream);
    d_frontiers.upload(frontiers, stream);
    d_budgets.upload(budgets, stream);
    d_licensed.upload(licensed, stream);
    d_rope_deltas.upload(rope_deltas, stream);
    d_alignment.upload(std::vector<std::int32_t>(expected_alignment.size(), kSentinel), stream);
    d_extents.upload(std::vector<std::int32_t>(expected_extents.size(), kSentinel), stream);
    d_positions.upload(std::vector<std::int32_t>(expected_positions.size(), kSentinel), stream);
    d_rope.upload(std::vector<std::int32_t>(expected_rope.size(), kSentinel), stream);
    d_valid.upload(std::vector<std::int32_t>(expected_valid.size(), kSentinel), stream);

    ninfer::Tensor t_verify = matrix_tensor(d_verify, width, batch);
    ninfer::Tensor t_anchors = vector_tensor(d_anchors, batch);
    ninfer::Tensor t_accepted = vector_tensor(d_accepted, batch);
    ninfer::Tensor t_frontiers = vector_tensor(d_frontiers, batch);
    ninfer::Tensor t_budgets = vector_tensor(d_budgets, batch);
    ninfer::Tensor t_licensed = vector_tensor(d_licensed, batch);
    ninfer::Tensor t_rope_deltas = vector_tensor(d_rope_deltas, batch);
    ninfer::Tensor t_alignment = matrix_tensor(d_alignment, width, batch);
    ninfer::Tensor t_extents = vector_tensor(d_extents, batch);
    ninfer::Tensor t_positions = pitched_tensor(d_positions, batch, steps, pitch);
    ninfer::Tensor t_rope = pitched_tensor(d_rope, batch, steps, pitch);
    ninfer::Tensor t_valid = pitched_tensor(d_valid, batch, steps, pitch);

    ninfer::ops::mtp_prepare_next_round(
        t_verify, t_anchors, t_accepted, t_frontiers, t_budgets, t_licensed, t_rope_deltas,
        t_alignment, t_extents, t_positions, t_rope, t_valid, kMaxContext, stream);

    const std::string label = "K=" + std::to_string(k) + "/B=" + std::to_string(batch);
    require_exact(d_alignment.download(stream), expected_alignment, label + " alignment");
    require_exact(d_extents.download(stream), expected_extents, label + " extent");
    require_exact(d_positions.download(stream), expected_positions, label + " AR position/padding");
    require_exact(d_rope.download(stream), expected_rope, label + " AR rope/padding");
    require_exact(d_valid.download(stream), expected_valid, label + " AR valid/padding");
    require_exact(d_verify.download(stream), verify, label + " immutable verify");
    require_exact(d_anchors.download(stream), anchors, label + " immutable anchor");
    require_exact(d_accepted.download(stream), accepted, label + " immutable accepted");
    require_exact(d_frontiers.download(stream), frontiers, label + " immutable frontier");
    require_exact(d_budgets.download(stream), budgets, label + " immutable budget");
    require_exact(d_licensed.download(stream), licensed, label + " immutable licensed");
    require_exact(d_rope_deltas.download(stream), rope_deltas, label + " immutable rope delta");

    if (k == 1 && batch == 1) {
        ninfer::Tensor bad_width = t_verify;
        bad_width.ne[0] = 1;
        require_invalid(
            [&] {
                ninfer::ops::mtp_prepare_next_round(
                    bad_width, t_anchors, t_accepted, t_frontiers, t_budgets, t_licensed,
                    t_rope_deltas, t_alignment, t_extents, t_positions, t_rope, t_valid,
                    kMaxContext, stream);
            },
            "T=1");
        ninfer::Tensor bad_stride = t_positions;
        bad_stride.nb[1] = 0;
        require_invalid(
            [&] {
                ninfer::ops::mtp_prepare_next_round(
                    t_verify, t_anchors, t_accepted, t_frontiers, t_budgets, t_licensed,
                    t_rope_deltas, t_alignment, t_extents, bad_stride, t_rope, t_valid,
                    kMaxContext, stream);
            },
            "short AR stride");
        require_invalid(
            [&] {
                ninfer::ops::mtp_prepare_next_round(
                    t_verify, t_anchors, t_accepted, t_frontiers, t_budgets, t_licensed,
                    t_rope_deltas, t_alignment, t_extents, t_positions, t_rope, t_valid, 0,
                    stream);
            },
            "zero max context");
    }
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(0);
        require(device.is_gfx1201() && device.props.warpSize == 32,
                "MTP qualifier requires wave32 gfx1201");
        for (std::int32_t k = 1; k <= 5; ++k) {
            for (std::int32_t batch = 1;
                 batch <= static_cast<std::int32_t>(ninfer::kMaximumConcurrency); ++batch) {
                qualify_case(k, batch, device.stream);
            }
        }
        std::cout << "mtp_round: PASS exact K=1..5 B=1..4 pitched-padding and immutable-input oracle\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mtp_round: FAIL: " << error.what() << '\n';
        return 1;
    }
}
