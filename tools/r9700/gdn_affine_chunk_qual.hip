#define main ninfer_retained_gdn_recurrence_qual_main
#include "gdn_recurrence_qual.hip"
#undef main

#include "ops/r9700/gdn_recurrence/gdn_affine_chunk_qualification.h"

#include <array>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <unistd.h>

namespace {

void qualify_affine(int width, std::uint32_t chunk, bool in_place, hipStream_t stream) {
    const Inputs in = inputs(width, 1, 701 + width + static_cast<int>(chunk));
    DeviceInputs device(in);
    const std::size_t state_count = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    const std::vector<float> initial = fp32_fixture(state_count, 709, 8192.0F);
    ninfer::DeviceBuffer state_in = uploaded(initial);
    ninfer::DeviceBuffer state_out(in_place ? 1 : state_in.size());
    ninfer::DeviceBuffer output(in.v.size() * sizeof(hip_bfloat16));
    const std::size_t workspace_bytes =
        ninfer::ops::r9700::gdn_recurrence::affine_chunk_workspace_capacity_bytes(width, chunk);
    ninfer::DeviceBuffer workspace(workspace_bytes);
    HIP_CHECK(ninfer::ops::r9700::gdn_recurrence::affine_chunk_qualification(
        static_cast<const hip_bfloat16*>(device.q.data()),
        static_cast<const hip_bfloat16*>(device.k.data()),
        static_cast<const hip_bfloat16*>(device.v.data()),
        static_cast<const float*>(device.g.data()), static_cast<const float*>(device.beta.data()),
        static_cast<const float*>(state_in.data()),
        static_cast<float*>(in_place ? state_in.data() : state_out.data()),
        static_cast<hip_bfloat16*>(output.data()), workspace.data(), workspace.size(), width, chunk,
        kScale, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> state(initial.begin(), initial.end());
    std::vector<double> expected_output;
    expected_output.reserve(in.v.size());
    for (int token = 0; token < width; ++token) {
        OracleStep next = step(in, token, state, true);
        state = std::move(next.state);
        expected_output.insert(expected_output.end(), next.output.begin(), next.output.end());
    }
    require_output(copied<hip_bfloat16>(output, in.v.size()), expected_output,
                   "affine chunk output C" + std::to_string(chunk));
    require_state(copied<float>(in_place ? state_in : state_out, state_count), state,
                  "affine chunk state C" + std::to_string(chunk));
    if (!in_place && copied<float>(state_in, state_count) != initial) {
        fail("affine chunk mutated distinct input state");
    }
}

struct TimingSamples {
    std::array<float, 9> raw{};
    float median_ms = 0.0F;
};

template <typename Launch>
TimingSamples timing_samples(hipStream_t stream, Launch&& launch) {
    for (int warmup = 0; warmup < 3; ++warmup) launch();
    HIP_CHECK(hipStreamSynchronize(stream));
    TimingSamples result;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    for (float& sample : result.raw) {
        HIP_CHECK(hipEventRecord(begin, stream));
        launch();
        HIP_CHECK(hipEventRecord(end, stream));
        HIP_CHECK(hipEventSynchronize(end));
        HIP_CHECK(hipEventElapsedTime(&sample, begin, end));
    }
    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipEventDestroy(end));
    auto ordered = result.raw;
    std::sort(ordered.begin(), ordered.end());
    result.median_ms = ordered[ordered.size() / 2U];
    return result;
}

struct TimingRow {
    int width;
    std::uint32_t chunk;
    TimingSamples incumbent;
    TimingSamples candidate;
};

TimingRow timing_row(int width, std::uint32_t chunk, hipStream_t stream) {
    const Inputs in = inputs(width, 1, 907 + width + static_cast<int>(chunk));
    DeviceInputs device(in);
    const std::size_t state_count = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    ninfer::DeviceBuffer state_in = uploaded(fp32_fixture(state_count, 911, 8192.0F));
    ninfer::DeviceBuffer state_out(state_in.size());
    ninfer::DeviceBuffer output(in.v.size() * sizeof(hip_bfloat16));
    ninfer::Tensor tin(state_in.data(), ninfer::DType::FP32, {kDim, kDim, kValueHeads});
    ninfer::Tensor tout(state_out.data(), ninfer::DType::FP32, {kDim, kDim, kValueHeads});
    ninfer::Tensor out(output.data(), ninfer::DType::BF16, {kDim, kValueHeads, width});
    ninfer::WorkspaceArena arena(1);
    const std::size_t workspace_bytes =
        ninfer::ops::r9700::gdn_recurrence::affine_chunk_workspace_capacity_bytes(width, chunk);
    ninfer::DeviceBuffer workspace(workspace_bytes);
    const TimingSamples incumbent = timing_samples(stream, [&] {
        ninfer::ops::gated_delta_net(device.tq, device.tk, device.tv, device.tg, device.tbeta,
                                     kScale, true, arena, tin, tout, out, stream);
    });
    const TimingSamples candidate = timing_samples(stream, [&] {
        HIP_CHECK(ninfer::ops::r9700::gdn_recurrence::affine_chunk_qualification(
            static_cast<const hip_bfloat16*>(device.q.data()),
            static_cast<const hip_bfloat16*>(device.k.data()),
            static_cast<const hip_bfloat16*>(device.v.data()),
            static_cast<const float*>(device.g.data()),
            static_cast<const float*>(device.beta.data()), static_cast<const float*>(state_in.data()),
            static_cast<float*>(state_out.data()), static_cast<hip_bfloat16*>(output.data()),
            workspace.data(), workspace.size(), width, chunk, kScale, stream));
    });
    return {width, chunk, incumbent, candidate};
}

void append_samples(std::ostringstream& out, const std::array<float, 9>& samples) {
    out << '[';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0U) out << ',';
        out << samples[index];
    }
    out << ']';
}

void write_report(const std::filesystem::path& path, const std::vector<TimingRow>& rows,
                  bool admitted) {
    std::ostringstream out;
    out << std::setprecision(9)
        << "{\n  \"schema\": \"ninfer.r9700.gdn_affine_chunk_qualification.v2\",\n"
        << "  \"architecture\": \"gfx1201\",\n"
        << "  \"disposition\": \""
        << (admitted ? "admitted_qualification_only_unpromoted"
                     : "rejected_performance_gate")
        << "\",\n"
        << "  \"production_dispatch\": false,\n"
        << "  \"numerical\": {\"complete_fp64_oracle\": true},\n"
        << "  \"admission_gate\": {\"tokens\":2048,\"chunk\":64,"
           "\"maximum_candidate_ms\":1.247,\"minimum_speedup\":2.25,\"passed\":"
        << (admitted ? "true" : "false") << "},\n  \"rows\": [\n";
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto& row = rows[index];
        out << "    {\"tokens\":" << row.width << ",\"chunk\":" << row.chunk
            << ",\"incumbent_raw_ms\":";
        append_samples(out, row.incumbent.raw);
        out << ",\"incumbent_median_ms\":" << row.incumbent.median_ms
            << ",\"candidate_raw_ms\":";
        append_samples(out, row.candidate.raw);
        out << ",\"candidate_median_ms\":" << row.candidate.median_ms
            << ",\"median_speedup\":"
            << row.incumbent.median_ms / row.candidate.median_ms << "}"
            << (index + 1U == rows.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    const std::string payload = out.str();
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) fail("refusing to overwrite or could not create affine timing report");
    std::size_t offset = 0U;
    while (offset < payload.size()) {
        const ssize_t written =
            ::write(descriptor, payload.data() + offset, payload.size() - offset);
        if (written <= 0) {
            (void)::close(descriptor);
            (void)::unlink(path.c_str());
            fail("could not finish affine timing report");
        }
        offset += static_cast<std::size_t>(written);
    }
    if (::close(descriptor) != 0) {
        (void)::unlink(path.c_str());
        fail("could not close affine timing report");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool benchmark = false;
        std::filesystem::path output;
        if (argc == 4 && std::string_view(argv[1]) == "--benchmark" &&
            std::string_view(argv[2]) == "--out-json") {
            benchmark = true;
            output = argv[3];
        } else if (argc != 1) {
            fail("usage: gdn_affine_chunk_qual [--benchmark --out-json PATH]");
        }
        HIP_CHECK(hipSetDevice(select_r9700()));
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        for (const std::uint32_t chunk : {32U, 64U, 128U}) {
            qualify_affine(64, chunk, false, stream);
            qualify_affine(128, chunk, chunk == 64U, stream);
        }
        qualify_affine(512, 64U, false, stream);
        qualify_affine(2048, 64U, false, stream);
        if (benchmark) {
            std::vector<TimingRow> rows;
            for (const int width : {512, 2048}) {
                for (const std::uint32_t chunk : {32U, 64U, 128U}) {
                    rows.push_back(timing_row(width, chunk, stream));
                }
            }
            const auto selected = std::find_if(rows.begin(), rows.end(), [](const TimingRow& row) {
                return row.width == 2048 && row.chunk == 64U;
            });
            const bool admitted =
                selected != rows.end() && selected->candidate.median_ms <= 1.247F &&
                selected->incumbent.median_ms / selected->candidate.median_ms >= 2.25F;
            write_report(output, rows, admitted);
            if (!admitted) {
                fail("C64 misses the >=75 ms aggregate P2048 admission gate");
            }
        }
        HIP_CHECK(hipStreamDestroy(stream));
        std::cout << "GDN affine chunk qualification: PASS max_output_abs=" << g_max_output_abs
                  << " max_output_rel_l2=" << g_max_output_rel_l2
                  << " max_state_abs=" << g_max_state_abs << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GDN affine chunk qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
