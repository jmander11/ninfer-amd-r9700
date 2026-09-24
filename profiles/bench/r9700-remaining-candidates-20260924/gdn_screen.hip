// Task-local public-Op screen: canonical FP64 oracle/criteria, unchanged snapshot
// arithmetic for supplementary exact parity, and graph event timing of record.
#define main retained_gdn_recurrence_qual_main
#include "gdn_recurrence_qual.hip"
#undef main
#include "core/decode_graph.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>

namespace {
using Json = nlohmann::json;
void require_auto() {
    const std::filesystem::path pci("/sys/bus/pci/devices/0000:13:00.0");
    const auto read = [&](const char* name) {
        std::ifstream f(pci / name); std::string value; f >> value; return value;
    };
    if (read("vendor") != "0x1002" || read("device") != "0x7551" ||
        read("power_dpm_force_performance_level") != "auto") fail("R9700 auto required");
}
struct Event {
    hipEvent_t value{};
    Event() { HIP_CHECK(hipEventCreate(&value)); }
    ~Event() { (void)hipEventDestroy(value); }
};
Json cell(int width, int batch, ninfer::DeviceContext& context,
          ninfer::DeviceBuffer& scrub) {
    const int salt = 901 + width * 13 + batch;
    // Canonical checks include ragged extents, untouched records/state, and
    // independent FP64 output/state; neither oracle stages normalized BF16.
    qualify_record(false, width, batch, salt, context.stream);
    qualify_snapshot(width, batch, salt, context.stream);
    auto test = pool_case(width, batch, salt);
    std::fill(test.valid.begin(), test.valid.end(), width);
    DeviceInputs input(test.in);
    auto pool = uploaded(test.pool);
    auto snapshot_pool = uploaded(test.pool);
    auto valid = uploaded(test.valid);
    auto initial = uploaded(test.initial);
    auto bases = uploaded(test.bases);
    ninfer::DeviceBuffer keys(test.in.k.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer values(test.in.v.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer gates(test.in.g.size() * 2U * sizeof(float));
    ninfer::DeviceBuffer output(test.in.v.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer snapshot_output(output.size());
    ninfer::Tensor state(pool.data(), ninfer::DType::FP32,
                        {kDim, kDim, kValueHeads, test.slots});
    ninfer::Tensor snapshot_state(snapshot_pool.data(), ninfer::DType::FP32,
                                 {kDim, kDim, kValueHeads, test.slots});
    ninfer::Tensor tv(valid.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor ti(initial.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor tb(bases.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor tk(keys.data(), ninfer::DType::BF16, {kDim, kQHeads, width, batch});
    ninfer::Tensor tvalue(values.data(), ninfer::DType::BF16, {kDim, kValueHeads, width, batch});
    ninfer::Tensor tg(gates.data(), ninfer::DType::FP32, {2, kValueHeads, width, batch});
    ninfer::Tensor out(output.data(), ninfer::DType::BF16, {kDim, kValueHeads, width, batch});
    ninfer::Tensor so(snapshot_output.data(), ninfer::DType::BF16,
                      {kDim, kValueHeads, width, batch});
    const auto launch = [&] {
        ninfer::ops::gated_delta_net_replay_record(input.tq, input.tk, input.tv, input.tg,
            input.tbeta, kScale, state, tv, ti, tk, tvalue, tg, out, context.stream);
    };
    launch();
    ninfer::ops::gated_delta_net_snapshot(input.tq, input.tk, input.tv, input.tg,
        input.tbeta, kScale, true, snapshot_state, tv, ti, tb, so, context.stream);
    context.synchronize();
    const auto exact_output = copied<std::uint16_t>(snapshot_output, test.in.v.size());
    const auto check = [&] {
        if (copied<std::uint16_t>(output, test.in.v.size()) != exact_output)
            fail("record differs from unchanged snapshot tree");
        const auto actual_k = copied<hip_bfloat16>(keys, test.in.k.size());
        const auto actual_v = copied<hip_bfloat16>(values, test.in.v.size());
        for (std::size_t i = 0; i < actual_k.size(); ++i)
            if (actual_k[i].data != test.in.k[i].data) fail("key record changed");
        for (std::size_t i = 0; i < actual_v.size(); ++i)
            if (actual_v[i].data != test.in.v[i].data) fail("value record changed");
        const auto actual_g = copied<std::uint32_t>(gates, test.in.g.size() * 2U);
        for (std::size_t i = 0; i < test.in.g.size(); ++i) {
            if (actual_g[2*i] != std::bit_cast<std::uint32_t>(test.in.g[i]) ||
                actual_g[2*i+1] != std::bit_cast<std::uint32_t>(test.in.beta[i]))
                fail("gate record changed");
        }
        if (copied<float>(pool, test.pool.size()) != test.pool) fail("input state changed");
    };
    check();
    // Dense timing inputs also receive their own full public oracle check.
    const auto actual_pool = copied<float>(snapshot_pool, test.pool.size());
    const auto actual_output = copied<hip_bfloat16>(output, test.in.v.size());
    constexpr std::size_t state_count = kValueHeads * kDim * kDim;
    for (int row = 0; row < batch; ++row) {
        const auto offset = static_cast<std::size_t>(test.initial[row]) * state_count;
        std::vector<double> oracle_state(test.pool.begin() + offset,
                                         test.pool.begin() + offset + state_count);
        for (int token = 0; token < width; ++token) {
            auto next = step(test.in, row * width + token, oracle_state);
            oracle_state = std::move(next.state);
            const auto o = static_cast<std::size_t>(row * width + token) * kValueHeads * kDim;
            require_output({actual_output.begin() + o, actual_output.begin() + o + kValueHeads*kDim},
                           next.output, "dense record public FP64");
            const auto s = static_cast<std::size_t>(test.bases[row] + token) * state_count;
            require_state({actual_pool.begin() + s, actual_pool.begin() + s + state_count},
                          oracle_state, "dense snapshot publication public FP64");
        }
    }
    ninfer::DecodeGraphDefinition definition;
    definition.capture(context.stream, launch);
    ninfer::DecodeGraphExecutable graph;
    graph.instantiate(definition); graph.upload(context.stream);
    output.fill_async(0xff, context.stream);
    keys.fill_async(0xff, context.stream); values.fill_async(0xff, context.stream);
    gates.fill_async(0xff, context.stream);
    graph.launch(context.stream); context.synchronize(); check();
    require_auto();
    Event begin, end;
    const auto sample = [&](bool cold) {
        if (cold) scrub.fill_async(0x5a, context.stream);
        else graph.launch(context.stream);
        HIP_CHECK(hipEventRecord(begin.value, context.stream));
        graph.launch(context.stream);
        HIP_CHECK(hipEventRecord(end.value, context.stream));
        HIP_CHECK(hipEventSynchronize(end.value));
        float ms{}; HIP_CHECK(hipEventElapsedTime(&ms, begin.value, end.value)); return ms;
    };
    (void)sample(false); (void)sample(true);
    std::vector<float> warm, cold;
    for (int trial = 0; trial < 16; ++trial) {
        if (trial & 1) { cold.push_back(sample(true)); warm.push_back(sample(false)); }
        else { warm.push_back(sample(false)); cold.push_back(sample(true)); }
    }
    check();
    return {{"width", width}, {"concurrency", batch}, {"public_fp64", true},
        {"ragged_records_and_state_preserved", true}, {"snapshot_exact", true},
        {"graph_exact", true}, {"warm_ms", warm}, {"scrub_cold_ms", cold}};
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) fail("usage: gdn_screen FRESH.json");
        if (std::filesystem::exists(argv[1])) fail("output exists");
        require_auto();
        ninfer::DeviceContext device(select_r9700());
        char pci[32]{}; int ordinal{};
        HIP_CHECK(hipGetDevice(&ordinal));
        HIP_CHECK(hipDeviceGetPCIBusId(pci, sizeof(pci), ordinal));
        if (std::string(pci) != "0000:13:00.0" ||
            std::string(device.props.name) != "AMD Radeon AI PRO R9700") fail("wrong GPU");
        ninfer::DeviceBuffer scrub(80U << 20U);
        Json cells = Json::array();
        for (int batch : {1, 2, 3, 4}) for (int width : {4, 5, 6}) {
            cells.push_back(cell(width, batch, device, scrub));
            std::cout << "C" << batch << " W" << width << " qualified/timed\n" << std::flush;
        }
        // Source applies to the complete sequential-record width domain.
        qualify_record(false, 2, 1, 1401, device.stream);
        qualify_record(false, 16, 1, 1411, device.stream);
        require_auto();
        Json report{{"schema", "ninfer.r9700.gdn-exact-tree-screen.v1"},
            {"pci", pci}, {"power", "auto"}, {"scrub_bytes", scrub.size()},
            {"timing_scope", "public sequential record Op; unprofiled graph GPU events"},
            {"replay_fold", "separate canonical gdn_replay_fold_qual required"},
            {"public_width_boundaries_2_16", true}, {"cells", cells}};
        std::ofstream file(argv[1]); file.exceptions(std::ios::badbit | std::ios::failbit);
        file << report.dump(2) << '\n'; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
