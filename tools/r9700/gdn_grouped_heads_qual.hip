#define main ninfer_retained_gdn_recurrence_qual_main
#include "gdn_recurrence_qual.hip"
#undef main

#include "ops/r9700/gdn_recurrence/gdn_grouped_heads_qualification.h"

#include <array>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace {

void launch_grouped(const DeviceInputs& device, const ninfer::DeviceBuffer& state_in,
                    ninfer::DeviceBuffer& state_out, ninfer::DeviceBuffer& output,
                    int width, hipStream_t stream) {
    HIP_CHECK(ninfer::ops::r9700::gdn_recurrence::grouped_heads_qualification(
        device.q.data(), device.k.data(), device.v.data(),
        static_cast<const float*>(device.g.data()),
        static_cast<const float*>(device.beta.data()),
        static_cast<const float*>(state_in.data()), static_cast<float*>(state_out.data()),
        output.data(), width, kScale, stream));
}

void qualify(int width, bool in_place, hipStream_t stream) {
    std::cerr << "grouped-head oracle begin T=" << width
              << " state=" << (in_place ? "in-place" : "distinct") << std::endl;
    const Inputs in = inputs(width, 1, 1301 + width);
    DeviceInputs device(in);
    const std::size_t state_count = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    const std::vector<float> initial = fp32_fixture(state_count, 1319 + width, 8192.0F);
    ninfer::DeviceBuffer state_in = uploaded(initial);
    ninfer::DeviceBuffer state_out(in_place ? 1 : state_in.size());
    ninfer::DeviceBuffer output(in.v.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer incumbent_state_in = uploaded(initial);
    ninfer::DeviceBuffer incumbent_state_out(in_place ? 1 : incumbent_state_in.size());
    ninfer::DeviceBuffer incumbent_output(in.v.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer& incumbent_destination =
        in_place ? incumbent_state_in : incumbent_state_out;
    ninfer::Tensor incumbent_in(incumbent_state_in.data(), ninfer::DType::FP32,
                                {kDim, kDim, kValueHeads});
    ninfer::Tensor incumbent_state(incumbent_destination.data(), ninfer::DType::FP32,
                                   {kDim, kDim, kValueHeads});
    ninfer::Tensor incumbent_out(incumbent_output.data(), ninfer::DType::BF16,
                                 {kDim, kValueHeads, width});
    ninfer::WorkspaceArena arena(1);
    ninfer::ops::gated_delta_net(device.tq, device.tk, device.tv, device.tg,
                                 device.tbeta, kScale, true, arena, incumbent_in,
                                 incumbent_state, incumbent_out, stream);
    ninfer::DeviceBuffer& destination = in_place ? state_in : state_out;
    launch_grouped(device, state_in, destination, output, width, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (copied<hip_bfloat16>(output, in.v.size()) !=
            copied<hip_bfloat16>(incumbent_output, in.v.size()) ||
        copied<float>(destination, state_count) !=
            copied<float>(incumbent_destination, state_count)) {
        fail("qualification challenger is not bit-exact with ordinary<true>");
    }
    std::vector<double> state(initial.begin(), initial.end());
    std::vector<double> expected;
    expected.reserve(in.v.size());
    for (int token = 0; token < width; ++token) {
        OracleStep next = step(in, token, state, true);
        state = std::move(next.state);
        expected.insert(expected.end(), next.output.begin(), next.output.end());
    }
    require_output(copied<hip_bfloat16>(output, in.v.size()), expected,
                   "grouped-head output T" + std::to_string(width));
    require_state(copied<float>(destination, state_count), state,
                  "grouped-head state T" + std::to_string(width));
    if (!in_place && copied<float>(state_in, state_count) != initial) {
        fail("grouped-head candidate mutated distinct input state");
    }
    std::cerr << "grouped-head oracle pass T=" << width
              << " state=" << (in_place ? "in-place" : "distinct") << std::endl;
}

struct Samples { std::array<float, 9> raw{}; float median = 0.0F; };

template <typename Launch>
Samples sample(hipStream_t stream, Launch&& launch) {
    for (int i = 0; i < 3; ++i) launch();
    HIP_CHECK(hipStreamSynchronize(stream));
    hipEvent_t begin = nullptr, end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    Samples result;
    for (float& value : result.raw) {
        HIP_CHECK(hipEventRecord(begin, stream));
        launch();
        HIP_CHECK(hipEventRecord(end, stream));
        HIP_CHECK(hipEventSynchronize(end));
        HIP_CHECK(hipEventElapsedTime(&value, begin, end));
    }
    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipEventDestroy(end));
    auto ordered = result.raw;
    std::sort(ordered.begin(), ordered.end());
    result.median = ordered[ordered.size() / 2U];
    return result;
}

struct Row { int width; Samples incumbent; Samples candidate; };

Row time_row(int width, hipStream_t stream) {
    std::cerr << "grouped-head timing begin T=" << width << std::endl;
    const Inputs in = inputs(width, 1, 1427 + width);
    DeviceInputs device(in);
    const std::size_t state_count = static_cast<std::size_t>(kValueHeads) * kDim * kDim;
    ninfer::DeviceBuffer state_in = uploaded(fp32_fixture(state_count, 1433, 8192.0F));
    ninfer::DeviceBuffer state_out(state_in.size());
    ninfer::DeviceBuffer output(in.v.size() * sizeof(hip_bfloat16));
    ninfer::Tensor tin(state_in.data(), ninfer::DType::FP32, {kDim, kDim, kValueHeads});
    ninfer::Tensor tout(state_out.data(), ninfer::DType::FP32, {kDim, kDim, kValueHeads});
    ninfer::Tensor out(output.data(), ninfer::DType::BF16, {kDim, kValueHeads, width});
    ninfer::WorkspaceArena arena(1);
    const Samples incumbent = sample(stream, [&] {
        ninfer::ops::gated_delta_net(device.tq, device.tk, device.tv, device.tg,
                                     device.tbeta, kScale, true, arena, tin, tout,
                                     out, stream);
    });
    const Samples candidate = sample(stream, [&] {
        launch_grouped(device, state_in, state_out, output, width, stream);
    });
    std::cerr << "grouped-head timing pass T=" << width
              << " incumbent_ms=" << incumbent.median
              << " candidate_ms=" << candidate.median << std::endl;
    return {width, incumbent, candidate};
}

void emit_samples(std::ostringstream& out, const std::array<float, 9>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

void report(const std::filesystem::path& path, const std::vector<Row>& rows,
            bool admitted) {
    std::ostringstream out;
    out << std::setprecision(9)
        << "{\n  \"schema\":\"ninfer.r9700.gdn_grouped_heads.v1\","
           "\n  \"production_dispatch\":false,\n  \"complete_fp64_oracle\":true,"
           "\n  \"oracle_token_extents\":[3,64,128],"
           "\n  \"admission\":{\"minimum_p2048_speedup\":1.5,"
           "\"no_row_regression\":true,\"passed\":"
        << (admitted ? "true" : "false") << "},\n  \"rows\":[\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const Row& row = rows[i];
        out << "    {\"tokens\":" << row.width << ",\"incumbent_raw_ms\":";
        emit_samples(out, row.incumbent.raw);
        out << ",\"incumbent_median_ms\":" << row.incumbent.median
            << ",\"candidate_raw_ms\":";
        emit_samples(out, row.candidate.raw);
        out << ",\"candidate_median_ms\":" << row.candidate.median
            << ",\"speedup\":" << row.incumbent.median / row.candidate.median << '}'
            << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    const std::string bytes = out.str();
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) fail("refusing to overwrite grouped-head report");
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t written = ::write(fd, bytes.data() + offset, bytes.size() - offset);
        if (written <= 0) { (void)::close(fd); (void)::unlink(path.c_str()); fail("report write failed"); }
        offset += static_cast<std::size_t>(written);
    }
    if (::close(fd) != 0) { (void)::unlink(path.c_str()); fail("report close failed"); }
}

void require_fresh(const std::filesystem::path& path) {
    struct stat status {};
    errno = 0;
    if (::lstat(path.c_str(), &status) == 0 || errno != ENOENT) {
        fail("report path must be fresh and not a dangling alias");
    }
    if (path.parent_path().empty() ||
        !std::filesystem::is_directory(path.parent_path())) {
        fail("report parent directory must already exist");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const bool check_path = argc == 3 &&
            std::string_view(argv[1]) == "--check-output-path";
        if (argc != 1 && !(argc == 3 && std::string_view(argv[1]) == "--out-json") &&
            !check_path) {
            std::cerr << "usage: gdn_grouped_heads_qual [--out-json FRESH.json | "
                         "--check-output-path PATH]\n";
            return 2;
        }
        if (check_path) {
            require_fresh(argv[2]);
            return 0;
        }
        const bool benchmark = argc == 3;
        const std::filesystem::path path = benchmark ? argv[2] : std::filesystem::path{};
        if (benchmark) require_fresh(path);
        ninfer::DeviceContext device(select_r9700());
        std::cerr << "grouped-head selected gfx1201 device" << std::endl;
        qualify(3, true, device.stream);
        qualify(3, false, device.stream);
        qualify(64, false, device.stream);
        qualify(128, false, device.stream);
        if (!benchmark) {
            std::cout << "GDN grouped-head FP64 qualification: PASS\n";
            return 0;
        }
        std::vector<Row> rows;
        for (int width : {128, 512, 2048, 4096}) {
            rows.push_back(time_row(width, device.stream));
        }
        const bool no_regression = std::all_of(rows.begin(), rows.end(), [](const Row& row) {
            return row.candidate.median <= row.incumbent.median;
        });
        const Row& p2048 = *std::find_if(rows.begin(), rows.end(), [](const Row& row) {
            return row.width == 2048;
        });
        const bool admitted = no_regression && p2048.incumbent.median >= 1.5F * p2048.candidate.median;
        report(path, rows, admitted);
        std::cerr << "grouped-head report published path=" << path
                  << " admitted=" << (admitted ? "true" : "false") << std::endl;
        return admitted ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "grouped-head qualification failed: " << error.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "grouped-head qualification failed: unknown exception" << std::endl;
        return 1;
    }
}
