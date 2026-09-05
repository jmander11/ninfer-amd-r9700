#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "artifact_fixture.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::array<std::byte, 3> kResource = {
    std::byte{1},
    std::byte{1},
    std::byte{1},
};
constexpr std::array<std::byte, 4> kTensor = {
    std::byte{2},
    std::byte{2},
    std::byte{2},
    std::byte{2},
};
constexpr std::array<std::byte, 8> kSecondTensor = {
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
};
constexpr std::size_t kFp8PayloadBytes = 260;

ninfer::test::artifact_fixture::TemporaryArtifact write_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", Json::array({
                            {{"name", "frontend/test.json"},
                             {"kind", "resource"},
                             {"encoding", "raw-bytes-v1"},
                             {"offset", 0},
                             {"bytes", 3}},
                            {{"name", "weights/test"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 256},
                             {"bytes", 4}},
                            {{"name", "weights/second"},
                             {"kind", "tensor"},
                             {"shape", {4}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 8192},
                             {"bytes", 8}},
                            {{"name", "weights/fp8-row"},
                             {"kind", "tensor"},
                             {"shape", {1, 1}},
                             {"format", "F8E4M3_ROW_F32S"},
                             {"layout", "row-scaled-k128-v1"},
                             {"offset", 8448},
                             {"bytes", kFp8PayloadBytes}},
                        })},
        },
        "materialization");
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    try {
        auto fixture = write_fixture();
        ninfer::artifact::Reader reader(fixture.path);
        ninfer::artifact::Binder validation_binder(reader);
        const auto validated_resource = validation_binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        validation_binder.retain_on_host(validated_resource);
        constexpr std::array<std::uint64_t, 1> validated_shape = {2};
        const auto validated_only                              = validation_binder.require_tensor(
            "weights/test", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, validated_shape);
        validation_binder.validate_only(validated_only);
        constexpr std::array<std::uint64_t, 1> retained_shape = {4};
        const auto retained_tensor                            = validation_binder.require_tensor(
            "weights/second", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, retained_shape);
        validation_binder.materialize_on_device(retained_tensor);
        constexpr std::array<std::uint64_t, 2> validated_fp8_shape = {1, 1};
        const auto validated_fp8 = validation_binder.require_tensor(
            "weights/fp8-row", ninfer::artifact::NumericFormat::F8E4M3_ROW_F32S,
            ninfer::artifact::StorageLayout::RowScaledK128V1, validated_fp8_shape);
        validation_binder.validate_only(validated_fp8);
        const auto validation_plan = validation_binder.finish();
        require(validation_plan.object_count == 4 && validation_plan.host_objects.size() == 1 &&
                    validation_plan.device_objects.size() == 1 &&
                    validation_plan.device_capacity_bytes == kSecondTensor.size(),
                "validate-only tensor was included in the materialization plan");

        ninfer::artifact::Binder binder(reader);

        const auto resource = binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        binder.retain_on_host(resource);
        constexpr std::array<std::uint64_t, 1> second_shape = {4};
        const auto second =
            binder.require_tensor("weights/second", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
        binder.materialize_on_device(second);

        // Bind in the opposite order from the artifact. Device placement order and file read order
        // are intentionally independent, exercising the direct-I/O scatter path.
        constexpr std::array<std::uint64_t, 1> tensor_shape = {2};
        const auto tensor =
            binder.require_tensor("weights/test", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
        binder.materialize_on_device(tensor);
        constexpr std::array<std::uint64_t, 2> fp8_shape = {1, 1};
        const auto fp8 = binder.require_tensor(
            "weights/fp8-row", ninfer::artifact::NumericFormat::F8E4M3_ROW_F32S,
            ninfer::artifact::StorageLayout::RowScaledK128V1, fp8_shape);
        binder.materialize_on_device(fp8);

        const ninfer::artifact::MaterializationPlan plan = binder.finish();
        require(plan.object_count == 4 && plan.host_objects.size() == 1 &&
                    plan.device_objects.size() == 3 && plan.device_capacity_bytes == 772,
                "binder produced the wrong materialization plan");

        ninfer::DeviceContext device(0);
        auto materialized = ninfer::artifact::materialize(reader, plan, device);

        std::array<std::byte, kTensor.size()> copied{};
        HIP_CHECK(hipMemcpy(copied.data(), materialized.device_data(tensor), copied.size(),
                            hipMemcpyDeviceToHost));
        require(copied == kTensor, "device tensor payload differs from the artifact");
        std::array<std::byte, kSecondTensor.size()> second_copied{};
        HIP_CHECK(hipMemcpy(second_copied.data(), materialized.device_data(second),
                            second_copied.size(), hipMemcpyDeviceToHost));
        require(second_copied == kSecondTensor,
                "second device tensor payload differs from the artifact");

        const ninfer::Weight fp8_weight = ninfer::artifact::materialized_weight(
            materialized, fp8, ninfer::artifact::NumericFormat::F8E4M3_ROW_F32S, 1, 1);
        const auto* fp8_payload = static_cast<const std::byte*>(fp8_weight.payload);
        require(fp8_weight.qtype == ninfer::QType::F8E4M3_ROW_F32S &&
                    fp8_weight.layout == ninfer::QuantLayout::RowScaled &&
                    fp8_weight.payload_bytes == kFp8PayloadBytes &&
                    fp8_weight.qdata == fp8_payload && fp8_weight.qdata_bytes == 128 &&
                    fp8_weight.qhigh == nullptr && fp8_weight.high_plane_bytes == 0 &&
                    fp8_weight.scales == fp8_payload + 256 && fp8_weight.scale_bytes == 4 &&
                    fp8_weight.scale_dtype == ninfer::DType::FP32 &&
                    fp8_weight.group == 0 && fp8_weight.group_size == 0 &&
                    fp8_weight.n == 1 && fp8_weight.k == 1 && fp8_weight.ndim == 2 &&
                    fp8_weight.shape[0] == 1 && fp8_weight.shape[1] == 1 &&
                    fp8_weight.padded_shape[0] == 1 && fp8_weight.padded_shape[1] == 128 &&
                    fp8_weight.scale_ne[0] == 1 && fp8_weight.scale_nb[0] == 4,
                "materialized FP8 row-scaled Weight view is malformed");

        const auto retained = materialized.resource_bytes(resource);
        require(std::equal(retained.begin(), retained.end(), kResource.begin(), kResource.end()),
                "retained resource payload differs from the artifact");

        const auto& stats = materialized.stats();
        require(stats.tensor_count == 3 && stats.resource_count == 1 &&
                    stats.h2d_bytes == kTensor.size() + kSecondTensor.size() + kFp8PayloadBytes &&
                    stats.retained_resource_bytes == kResource.size() &&
                    stats.file_bytes == kResource.size() +
                                            ninfer::artifact::Reader::direct_io_alignment + 516,
                "materialization statistics are incomplete");
        require(materialized.device_arena().capacity() == plan.device_capacity_bytes &&
                    materialized.device_arena().used() == plan.device_capacity_bytes,
                "materialized tensor does not own the planned device backing");
        std::cout << "artifact_materialization: PASS h2d_bytes=" << stats.h2d_bytes
                  << " staging_bytes=" << stats.peak_staging_bytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
