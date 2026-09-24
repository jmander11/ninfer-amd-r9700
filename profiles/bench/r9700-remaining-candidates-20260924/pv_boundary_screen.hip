// Task-local complete-Op screen. Reuse the canonical independent oracle and
// fixtures, but never run its unrelated Text P129 qualification/timing main.
#define main retained_attention_discriminator_main
#include "dflash_attention_route_discriminator.hip"
#undef main
#include <filesystem>
#include <fstream>

namespace {
void screen_preflight() {
    const std::filesystem::path pci("/sys/bus/pci/devices/0000:13:00.0");
    const auto read = [&](const char* name) {
        std::ifstream file(pci / name); std::string value; file >> value; return value;
    };
    if (read("vendor") != "0x1002" || read("device") != "0x7551" ||
        read("power_dpm_force_performance_level") != "auto") fail("R9700 auto required");
}
struct ScreenEvent {
    hipEvent_t value{};
    ScreenEvent() { hip_check(hipEventCreate(&value), "create event"); }
    ~ScreenEvent() { (void)hipEventDestroy(value); }
};
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) fail("usage: attention_screen FRESH.json");
        const std::filesystem::path path(argv[1]);
        if (std::filesystem::exists(path)) fail("output already exists");
        screen_preflight();
        ninfer::DeviceContext device(0);
        char pci[32]{};
        hip_check(hipDeviceGetPCIBusId(pci, sizeof(pci), 0), "PCI identity");
        if (!device.is_gfx1201() || device.props.warpSize != 32 ||
            std::string(device.props.name) != "AMD Radeon AI PRO R9700" ||
            std::string(pci) != "0000:13:00.0") fail("wrong device");
        kW6Context = 8192U; kPhysicalPages = 131U;
        kPhysicalTokens = kPhysicalPages * kPageSize;
        const auto fixture = make_fixture();
        std::vector<hip_bfloat16> queries;
        for (const auto value : fixture.queries) queries.emplace_back(value);
        std::vector<float> profile_queries = fixture.queries;
        for (auto& value : profile_queries)
            value = static_cast<float>(oracle::decode_e4m3fn(oracle::encode_e4m3fn(value)));
        DeviceArray<hip_bfloat16> query(queries.size()); upload(query, queries);
        DeviceArray<std::uint8_t> keys(fixture.physical_keys.size()); upload(keys, fixture.physical_keys);
        DeviceArray<std::uint8_t> values(fixture.physical_values.size()); upload(values, fixture.physical_values);
        DeviceArray<std::uint16_t> scales(fixture.physical_scales.size()); upload(scales, fixture.physical_scales);
        DeviceArray<std::uint32_t> pages(fixture.pages.size()); upload(pages, fixture.pages);
        constexpr std::size_t scrub_bytes = 80U << 20U;
        DeviceArray<std::uint8_t> scrub(scrub_bytes);
        Json cells = Json::array();
        for (const std::uint32_t rows : {6U}) {
            for (const std::size_t context : {std::size_t{2048}, std::size_t{4095}, std::size_t{4096},
                                             std::size_t{4097}, std::size_t{8191}, std::size_t{8192}}) {
                const std::size_t count = rows * kQueryHeads * kHeadDim;
                const std::size_t score_count = rows * kQueryHeads * context;
                std::vector<std::int32_t> positions(rows);
                for (std::uint32_t row = 0; row < rows; ++row)
                    positions[row] = static_cast<std::int32_t>(context - rows + row);
                DeviceArray<std::int32_t> device_positions(rows); upload(device_positions, positions);
                DeviceArray<float> output(count + 2U), serial_output(count + 2U);
                DeviceArray<float> scores(score_count + 2U), serial_scores(kQueryHeads * context + 2U);
                fill_guarded(output, count); fill_guarded(serial_output, count);
                fill_guarded(scores, score_count); fill_guarded(serial_scores, kQueryHeads * context);
                kv::Fp8Int4KvAttentionArgs args{
                    .query = query.get(), .key_codes = keys.get(), .value_codes = values.get(),
                    .value_scale_bits = scales.get(), .page_table = pages.get(),
                    .page_table_pages = static_cast<std::uint32_t>(fixture.pages.size()),
                    .context = context, .physical_tokens = kPhysicalTokens,
                    .query_heads = kQueryHeads, .kv_heads = kKvHeads,
                    .q_to_kv = kQueriesPerKv, .value_group = kValueGroup,
                    .query_rows = rows, .row_positions = device_positions.get(),
                    .attention_scale = kAttentionScale,
                    .key_layout = kv::Fp8Int4KvPlaneLayout::TokenFastest,
                    .value_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
                    .value_scale_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
                    .score_workspace = scores.get() + 1U, .output = output.get() + 1U};
                auto serial = args;
                serial.score_workspace = serial_scores.get() + 1U;
                serial.output = serial_output.get() + 1U;
                hip_check(kv::fp8_int4_kv_attention_wmma(serial, device.stream), "serial attention");
                hip_check(kv::fp8_int4_kv_attention_dflash_verify_batched_wmma(args, device.stream),
                          "selected attention");
                device.synchronize();
                const auto actual = read_guarded(output, count, "selected output");
                const auto reference = read_guarded(serial_output, count, "serial output");
                const auto public_oracle = attention_oracle(fixture, fixture.queries, rows, positions);
                const auto profile_oracle = attention_oracle(fixture, profile_queries, rows, positions);
                const auto error = require_profile_bound(actual, public_oracle, profile_oracle,
                    2.0e-4, "public BF16 attention");
                (void)require_oracle(actual, profile_oracle, 2.0e-4, "FP8-Q arithmetic profile");
                const auto exact = [&](const auto& result) {
                    return std::equal(reference.begin(), reference.end(), result.begin(), [](float a, float b) {
                        return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
                    });
                };
                if (!exact(actual)) fail("selected attention differs from serial");
                ninfer::DecodeGraphDefinition definition;
                definition.capture(device.stream, [&] {
                    hip_check(kv::fp8_int4_kv_attention_dflash_verify_batched_wmma(args, device.stream),
                              "capture complete attention");
                });
                ninfer::DecodeGraphExecutable executable;
                executable.instantiate(definition); executable.upload(device.stream);
                fill_guarded(output, count); fill_guarded(scores, score_count);
                executable.launch(device.stream); device.synchronize();
                if (!exact(read_guarded(output, count, "graph output"))) fail("graph differs from serial");
                (void)read_guarded(scores, score_count, "graph scores");
                (void)read_guarded(serial_scores, kQueryHeads * context, "serial scores");
                screen_preflight();
                ScreenEvent begin, end;
                const auto measure = [&](bool cold) {
                    if (cold) hip_check(hipMemsetAsync(scrub.get(), 0x5a, scrub_bytes, device.stream), "scrub");
                    else executable.launch(device.stream);
                    hip_check(hipEventRecord(begin.value, device.stream), "begin event");
                    executable.launch(device.stream);
                    hip_check(hipEventRecord(end.value, device.stream), "end event");
                    hip_check(hipEventSynchronize(end.value), "synchronize event");
                    float ms = 0.0F;
                    hip_check(hipEventElapsedTime(&ms, begin.value, end.value), "elapsed event");
                    return ms;
                };
                (void)measure(false); (void)measure(true);
                std::vector<float> warm, cold;
                for (unsigned trial = 0; trial < 16U; ++trial) {
                    if (trial & 1U) { cold.push_back(measure(true)); warm.push_back(measure(false)); }
                    else { warm.push_back(measure(false)); cold.push_back(measure(true)); }
                }
                cells.push_back({{"rows", rows}, {"context", context},
                    {"public_oracle_max_abs", error.maximum_absolute},
                    {"serial_exact", true}, {"graph_exact", true}, {"guards", true},
                    {"warm_ms", warm}, {"scrub_cold_ms", cold}});
                std::cout << "W" << rows << " Ctx" << context << " qualified/timed\n" << std::flush;
            }
        }
        screen_preflight();
        const Json result{{"schema", "ninfer.r9700.complete-attention-screen.v1"},
            {"power", "auto"}, {"pci", pci}, {"scrub_bytes", scrub_bytes},
            {"timing_scope", "unprofiled complete Op graph GPU events; not whole inference"},
            {"cells", cells}};
        std::ofstream file(path); file.exceptions(std::ios::failbit | std::ios::badbit);
        file << result.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
