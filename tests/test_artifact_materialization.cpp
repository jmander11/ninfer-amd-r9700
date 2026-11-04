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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>

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

constexpr std::size_t kLargeHostBytes = 9ULL * 64ULL * 1024ULL * 1024ULL + 512;
constexpr std::size_t kBoundaryBytes  = 256;
constexpr std::size_t kMarkerPage     = 4096;

std::byte host_marker(std::size_t offset) {
    return std::byte{static_cast<unsigned char>((offset / kMarkerPage) % 251)};
}

ninfer::test::artifact_fixture::TemporaryArtifact write_ring_fixture() {
    using namespace ninfer::test::artifact_fixture;
    const auto tensor = [](const char* name, std::size_t offset, std::size_t bytes) {
        return Json{{"name", name},
                    {"kind", "tensor"},
                    {"shape", {bytes / 2}},
                    {"format", "BF16"},
                    {"layout", "contiguous-le-v1"},
                    {"offset", offset},
                    {"bytes", bytes}};
    };
    const Json directory = {
        {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
        {"objects",
         Json::array({tensor("weights/before", 0, kBoundaryBytes),
                      tensor("weights/mapped", kBoundaryBytes, kLargeHostBytes),
                      tensor("weights/after", kBoundaryBytes + kLargeHostBytes, kBoundaryBytes)})}};
    const std::string metadata = directory.dump();
    std::vector<std::byte> header(align_up(16 + metadata.size(), 4096), std::byte{0});
    std::copy(kMagic.begin(), kMagic.end(), reinterpret_cast<std::uint8_t*>(header.data()));
    write_u64_le(header.data() + 8, metadata.size());
    std::memcpy(header.data() + 16, metadata.data(), metadata.size());
    TemporaryArtifact fixture{std::filesystem::temp_directory_path() /
                              "ninfer_artifact_materialization_ring.ninfer"};
    std::ofstream output(fixture.path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(header.data()),
                 static_cast<std::streamsize>(header.size()));
    std::array<std::byte, kBoundaryBytes> boundary;
    boundary.fill(std::byte{0xd3});
    output.write(reinterpret_cast<const char*>(boundary.data()), boundary.size());
    // Keep fixture construction bounded: no second complete embedding-sized host vector.
    std::vector<std::byte> block(1024 * 1024);
    for (std::size_t offset = 0; offset < kLargeHostBytes; offset += block.size()) {
        const std::size_t bytes = std::min(block.size(), kLargeHostBytes - offset);
        for (std::size_t page = 0; page < bytes; page += kMarkerPage) {
            std::fill_n(block.data() + page, std::min(kMarkerPage, bytes - page),
                        host_marker(offset + page));
        }
        output.write(reinterpret_cast<const char*>(block.data()),
                     static_cast<std::streamsize>(bytes));
    }
    boundary.fill(std::byte{0x7e});
    output.write(reinterpret_cast<const char*>(boundary.data()), boundary.size());
    if (!output) { throw std::runtime_error("failed to write materialization ring fixture"); }
    return fixture;
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
        const auto fp8                                   = binder.require_tensor(
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
                    fp8_weight.scale_dtype == ninfer::DType::FP32 && fp8_weight.group == 0 &&
                    fp8_weight.group_size == 0 && fp8_weight.n == 1 && fp8_weight.k == 1 &&
                    fp8_weight.ndim == 2 && fp8_weight.shape[0] == 1 && fp8_weight.shape[1] == 1 &&
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
                    stats.file_bytes ==
                        kResource.size() + ninfer::artifact::Reader::direct_io_alignment + 516,
                "materialization statistics are incomplete");
        require(materialized.device_arena().capacity() == plan.device_capacity_bytes &&
                    materialized.device_arena().used() == plan.device_capacity_bytes,
                "materialized tensor does not own the planned device backing");

        // Mapped host placement: outside the device backing, pinned, read by kernels in place.
        ninfer::artifact::Binder mapped_binder(reader);
        mapped_binder.retain_on_host(mapped_binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1));
        const auto mapped = mapped_binder.require_tensor(
            "weights/test", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
        mapped_binder.materialize_on_mapped_host(mapped);
        const auto mapped_device = mapped_binder.require_tensor(
            "weights/second", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
        mapped_binder.materialize_on_device(mapped_device);
        mapped_binder.validate_only(mapped_binder.require_tensor(
            "weights/fp8-row", ninfer::artifact::NumericFormat::F8E4M3_ROW_F32S,
            ninfer::artifact::StorageLayout::RowScaledK128V1, fp8_shape));
        const auto mapped_plan = mapped_binder.finish();
        require(mapped_plan.device_capacity_bytes == kSecondTensor.size() &&
                    mapped_plan.device_objects.size() == 1 &&
                    mapped_plan.mapped_host_capacity_bytes == kTensor.size() &&
                    mapped_plan.mapped_host_objects.size() == 1,
                "mapped host tensor was charged to the device backing");
        auto mapped_materialized = ninfer::artifact::materialize(reader, mapped_plan, device);
        void* mapped_data        = mapped_materialized.device_data(mapped);
        hipPointerAttribute_t attributes{};
        HIP_CHECK(hipPointerGetAttributes(&attributes, mapped_data));
        require(attributes.type == hipMemoryTypeHost && attributes.devicePointer == mapped_data &&
                    attributes.hostPointer == mapped_data,
                "mapped host tensor is not pinned at its unified device address");
        require(
            std::equal(kTensor.begin(), kTensor.end(), static_cast<const std::byte*>(mapped_data)),
            "mapped host tensor payload differs from the artifact");
        std::array<std::byte, kSecondTensor.size()> mapped_second{};
        HIP_CHECK(hipMemcpy(mapped_second.data(), mapped_materialized.device_data(mapped_device),
                            mapped_second.size(), hipMemcpyDeviceToHost));
        require(mapped_second == kSecondTensor, "device tensor beside a mapped one differs");
        const auto& mapped_stats = mapped_materialized.stats();
        require(mapped_stats.tensor_count == 2 &&
                    mapped_stats.mapped_host_bytes == kTensor.size() &&
                    mapped_stats.h2d_bytes == kSecondTensor.size(),
                "mapped host materialization statistics are incomplete");
        // More than eight 64-MiB reads: the first ring slot changes from mixed to
        // host-only, while a slot initially used for host-only data later uploads a
        // device boundary. Both copies must survive wraparound and asynchronous H2D.
        auto ring_fixture = write_ring_fixture();
        ninfer::artifact::Reader ring_reader(ring_fixture.path);
        ninfer::artifact::Binder ring_binder(ring_reader);
        const auto bind_ring = [&](const char* name, std::uint64_t bytes) {
            const std::array<std::uint64_t, 1> shape = {bytes / 2};
            return ring_binder.require_tensor(name, ninfer::artifact::NumericFormat::BF16,
                                              ninfer::artifact::StorageLayout::ContiguousLeV1,
                                              shape);
        };
        const auto before     = bind_ring("weights/before", kBoundaryBytes);
        const auto large_host = bind_ring("weights/mapped", kLargeHostBytes);
        const auto after      = bind_ring("weights/after", kBoundaryBytes);
        ring_binder.materialize_on_device(after);
        ring_binder.materialize_on_mapped_host(large_host);
        ring_binder.materialize_on_device(before);
        auto ring_materialized =
            ninfer::artifact::materialize(ring_reader, ring_binder.finish(), device);
        const auto* host = static_cast<const std::byte*>(ring_materialized.device_data(large_host));
        for (std::size_t offset = 0; offset < kLargeHostBytes; offset += kMarkerPage) {
            const std::size_t bytes = std::min(kMarkerPage, kLargeHostBytes - offset);
            const std::byte marker  = host_marker(offset);
            require(std::all_of(host + offset, host + offset + bytes,
                                [marker](std::byte value) { return value == marker; }),
                    "mapped host payload changed across a materialization ring wrap");
        }
        std::array<std::byte, kBoundaryBytes> boundary;
        HIP_CHECK(hipMemcpy(boundary.data(), ring_materialized.device_data(before), boundary.size(),
                            hipMemcpyDeviceToHost));
        require(std::all_of(boundary.begin(), boundary.end(),
                            [](std::byte value) { return value == std::byte{0xd3}; }),
                "device payload before mapped host tensor differs");
        HIP_CHECK(hipMemcpy(boundary.data(), ring_materialized.device_data(after), boundary.size(),
                            hipMemcpyDeviceToHost));
        require(std::all_of(boundary.begin(), boundary.end(),
                            [](std::byte value) { return value == std::byte{0x7e}; }),
                "device payload after mapped host tensor differs");
        require(ring_materialized.stats().mapped_host_bytes == kLargeHostBytes &&
                    ring_materialized.stats().h2d_bytes == 2 * kBoundaryBytes,
                "ring materialization byte accounting differs");
        std::cout << "artifact_materialization: PASS h2d_bytes=" << stats.h2d_bytes
                  << " staging_bytes=" << stats.peak_staging_bytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
