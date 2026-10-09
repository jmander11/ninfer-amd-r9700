// Complete public-input projection oracle shared with the native Linear qualifiers.
#define NINFER_A8Q4_VERIFY_QUAL_NO_MAIN
#include "a8q4_verify_qual_impl.h"
#include "ninfer/ops/normalized_rope_kv_append.h"
#include "ninfer/ops/rope.h"
#include "qualification_device_guard.h"

namespace {

void require(bool condition, const char* message) {
    if (!condition) fail(message);
}

void projection(unsigned tokens, hipStream_t stream) {
    K                   = 5120;
    G                   = K / 64;
    N                   = 6144;
    const auto original = make_decode_dot8_weights(N, K);
    Weights weights(original, stream);
    const auto x = activation(tokens);
    Guarded<hip_bfloat16> input(x.size(), stream);
    input.put(x, stream);
    Output full(tokens, stream);
    N = 2048;
    Output selected(tokens, stream);
    auto args = arguments(tokens, input, weights, selected);
    args.weight_codes += 4096U * K / 2;
    args.weight_code_bytes = N * K / 2;
    args.weight_scales += 4096U * G;
    args.weight_scale_bytes = N * G * sizeof(std::uint16_t);
    auto host               = original;
    host.codes.erase(host.codes.begin(), host.codes.begin() + 4096U * K / 2);
    host.scales.erase(host.scales.begin(), host.scales.begin() + 4096U * G);
    const auto represented = quantize_host(x, tokens, K);
    const auto budget      = a8_error_budget(tokens, represented, host);
    const auto reference   = public_oracle(tokens, x, host);
    launch(args, stream);
    (void)compare_public(tokens, selected.value.read(stream), reference, budget);
    Graph graph(stream, [&] { launch(args, stream); });
    graph.run(stream);
    (void)compare_public(tokens, selected.value.read(stream), reference, budget);
    selected.value.guards(stream);
    selected.scratch.guards(stream);
    exact(input.read(stream), x, "context projection mutated represented input");
    const auto old = [&] {
        N = 6144;
        launch(arguments(tokens, input, weights, full), stream);
    };
    const auto fresh = [&] {
        N = 2048;
        launch(args, stream);
    };
    old();
    const auto full_values = full.value.read(stream);
    const auto values      = selected.value.read(stream);
    N                      = 2048;
    // The shared producer image must reach the same public Linear oracle at the selected rows.
    ninfer::Weight view{};
    view.payload       = args.weight_codes;
    view.payload_bytes = args.weight_code_bytes + args.weight_scale_bytes;
    view.qtype         = ninfer::QType::Q4G64_F16S;
    view.group_size = view.group = 64;
    view.ndim                    = 2;
    view.n = view.shape[0] = view.padded_shape[0] = 2048;
    view.k = view.shape[1] = view.padded_shape[1] = K;
    view.qdata                                    = args.weight_codes;
    view.qdata_bytes                              = args.weight_code_bytes;
    view.scales                                   = args.weight_scales;
    view.scale_bytes                              = args.weight_scale_bytes;
    view.layout                                   = ninfer::QuantLayout::Q4N16K16;
    view.scale_dtype                              = ninfer::DType::FP16;
    ninfer::DeviceBuffer image(ninfer::ops::q4_activation_image_bytes(tokens, K));
    const ninfer::DeviceSpan span{image.data(), image.size()};
    ninfer::Tensor input_tensor(input.data(), ninfer::DType::BF16,
                                {static_cast<std::int32_t>(K), static_cast<std::int32_t>(tokens)});
    ninfer::Tensor output_tensor(selected.value.data(), ninfer::DType::BF16,
                                 {2048, static_cast<std::int32_t>(tokens)});
    const auto image_projection = [&] {
        ninfer::ops::quantize_q4_activation_image(input_tensor, span, stream);
        ninfer::ops::linear_q4_activation_image(view, output_tensor, span, stream);
    };
    image_projection();
    (void)compare_public(tokens, selected.value.read(stream), reference, budget);
    exact(selected.value.read(stream), values, "producer image changed selected K/V projection");
    Graph image_graph(stream, image_projection);
    image_graph.run(stream);
    (void)compare_public(tokens, selected.value.read(stream), reference, budget);
    exact(selected.value.read(stream), values, "captured producer image changed K/V projection");
    selected.value.guards(stream);
    // Pairwise differences are supplementary; the complete FP64 public formula is the oracle.
    double difference = 0;
    for (unsigned t = 0; t < tokens; ++t)
        for (unsigned row = 0; row < 2048; ++row)
            difference =
                std::max(difference,
                         std::abs(static_cast<double>(static_cast<float>(values[t * 2048 + row])) -
                                  static_cast<float>(full_values[t * 6144 + 4096 + row])));
    constexpr int repeats = 64;
    Graph old_graph(stream, [&] {
        for (int i = 0; i < repeats; ++i) old();
    });
    Graph new_graph(stream, [&] {
        for (int i = 0; i < repeats; ++i) fresh();
    });
    hipEvent_t start{}, stop{};
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> timing[2];
    ninfer::r9700_qualification::device_guard().check("projection-after-oracle");
    for (int trial = 0; trial < 14; ++trial) {
        for (int item = 0; item < 2; ++item) {
            const int route = (trial + item) % 2;
            HIP_CHECK(hipEventRecord(start, stream));
            (route == 0 ? old_graph : new_graph).run(stream);
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            if (trial >= 4) timing[route].push_back(ms * 1000 / repeats);
        }
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    for (auto& times : timing) std::sort(times.begin(), times.end());
    std::cout << "context_projection T=" << tokens << " full_us=" << timing[0][5]
              << " kv_us=" << timing[1][5] << " max_pairwise_abs=" << difference << '\n';
    N = 2048;
}

void append(int width, int batch, hipStream_t stream) {
    constexpr int D = 128, H = 8, capacity = 2048, lane_capacity = 8;
    const int columns = width * batch;
    std::vector<hip_bfloat16> compact(static_cast<std::size_t>(2048) * columns);
    std::vector<hip_bfloat16> fused(static_cast<std::size_t>(6144) * columns);
    std::vector<hip_bfloat16> gains(D);
    for (std::size_t i = 0; i < compact.size(); ++i)
        compact[i] =
            hip_bfloat16(static_cast<float>(static_cast<int>((i * 17 + 11) % 101) - 50) / 32);
    for (int d = 0; d < D; ++d) gains[d] = hip_bfloat16(1.0F + (d % 9) / 32.0F);
    for (int c = 0; c < columns; ++c)
        std::copy_n(compact.begin() + c * 2048, 2048, fused.begin() + c * 6144 + 4096);
    std::vector<std::int32_t> positions(columns), counts(batch), lanes(batch);
    for (int b = 0; b < batch; ++b) {
        counts[b] = batch == 1 ? width : b == 0 ? 0 : b == batch - 1 ? width : width / 2;
        lanes[b]  = lane_capacity - 1 - b;
        for (int t = 0; t < width; ++t) positions[b * width + t] = 262140 + t;
    }
    Guarded<hip_bfloat16> kv(compact.size(), stream), old_fused(fused.size(), stream),
        norm(D, stream), key(columns * D * H, stream), value(columns * D * H, stream);
    Guarded<std::int32_t> pos(columns, stream), count(batch, stream), lane(batch, stream);
    kv.put(compact, stream);
    old_fused.put(fused, stream);
    norm.put(gains, stream);
    pos.put(positions, stream);
    count.put(counts, stream);
    lane.put(lanes, stream);
    const std::size_t elements = static_cast<std::size_t>(D) * H * capacity * lane_capacity;
    std::vector<hip_bfloat16> sentinel(elements, hip_bfloat16(0.25F));
    Guarded<hip_bfloat16> cache_k(elements, stream), cache_v(elements, stream);
    cache_k.put(sentinel, stream);
    cache_v.put(sentinel, stream);
    ninfer::Tensor kv_tensor(kv.data(), ninfer::DType::BF16, {2048, columns});
    ninfer::Tensor norm_tensor(norm.data(), ninfer::DType::BF16, {128});
    ninfer::Tensor pos_tensor(pos.data(), ninfer::DType::I32, {width, batch});
    ninfer::Tensor count_tensor(count.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor lane_tensor(lane.data(), ninfer::DType::I32, {batch});
    ninfer::CyclicKVCacheLayerView cache{
        .k = ninfer::Tensor(cache_k.data(), ninfer::DType::BF16, {D, capacity, H, lane_capacity}),
        .v = ninfer::Tensor(cache_v.data(), ninfer::DType::BF16, {D, capacity, H, lane_capacity}),
        .capacity        = capacity,
        .padded_capacity = capacity,
        .num_kv_heads    = H,
        .head_dim        = D,
        .lane_capacity   = lane_capacity};
    const auto selected = [&] {
        ninfer::ops::normalized_rope_kv_append(kv_tensor, norm_tensor, pos_tensor, count_tensor,
                                               lane_tensor, 1.0e-6F,
                                               {0, static_cast<unsigned>(width)}, cache, stream);
    };
    const auto validate = [&] {
        const auto got_k = cache_k.read(stream), got_v = cache_v.read(stream);
        std::vector<bool> written(elements, false);
        double error2 = 0, reference2 = 0;
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < counts[b]; ++t) {
                const int column = b * width + t;
                for (int head = 0; head < H; ++head) {
                    double squares     = 0;
                    const int raw_base = column * 2048 + head * D;
                    for (int d = 0; d < D; ++d) {
                        const double x = static_cast<float>(compact[raw_base + d]);
                        squares += x * x;
                    }
                    const double inverse = 1.0 / std::sqrt(squares / D + 1.0e-6F);
                    double normalized[D];
                    for (int d = 0; d < D; ++d)
                        normalized[d] =
                            static_cast<double>(static_cast<float>(compact[raw_base + d])) *
                            inverse * static_cast<double>(static_cast<float>(gains[d]));
                    for (int d = 0; d < D; ++d) {
                        const int pair      = d % (D / 2);
                        const double angle  = positions[column] * std::pow(1.0e7, -2.0 * pair / D);
                        const double cosine = std::cos(angle), sine = std::sin(angle);
                        const double expected =
                            d < D / 2 ? normalized[d] * cosine - normalized[d + D / 2] * sine
                                      : normalized[d] * cosine + normalized[d - D / 2] * sine;
                        const auto dst =
                            ((static_cast<std::size_t>(lanes[b]) * H + head) * capacity +
                             positions[column] % capacity) *
                                D +
                            d;
                        written[dst]       = true;
                        const double error = static_cast<float>(got_k[dst]) - expected;
                        require(std::isfinite(error) && std::abs(error) <= 0.02,
                                "normalized cache K exceeds FP64 absolute criterion");
                        error2 += error * error;
                        reference2 += expected * expected;
                        require(got_v[dst].data == compact[raw_base + 1024 + d].data,
                                "normalized append changed represented V bits");
                    }
                }
            }
        require(std::sqrt(error2 / std::max(reference2, 1.0e-30)) < 0.004,
                "normalized cache K exceeds FP64 relative RMS criterion");
        for (std::size_t i = 0; i < elements; ++i)
            if (!written[i])
                require(got_k[i].data == sentinel[i].data && got_v[i].data == sentinel[i].data,
                        "normalized append wrote an uncommitted cache slot");
        cache_k.guards(stream);
        cache_v.guards(stream);
        exact(kv.read(stream), compact, "normalized append mutated K/V input");
    };
    selected();
    validate();
    Graph replay(stream, selected);
    replay.run(stream);
    validate();
    // Same captured graph, changed device counts: the old committed cache must be unchanged.
    const auto before_k = cache_k.read(stream), before_v = cache_v.read(stream);
    count.put(std::vector<std::int32_t>(batch, 0), stream);
    replay.run(stream);
    exact(cache_k.read(stream), before_k, "zero replay wrote K");
    exact(cache_v.read(stream), before_v, "zero replay wrote V");
    count.put(counts, stream);
    ninfer::Tensor fused_tensor(old_fused.data(), ninfer::DType::BF16, {6144, columns});
    ninfer::Tensor key_tensor(key.data(), ninfer::DType::BF16, {128, 8, columns});
    ninfer::Tensor value_tensor(value.data(), ninfer::DType::BF16, {128, 8, columns});
    const auto baseline = [&] {
        ninfer::ops::dflash_qkv_norm_rope(pos_tensor.view({columns}), fused_tensor, norm_tensor,
                                          norm_tensor, 1.0e-6F, nullptr, key_tensor, value_tensor,
                                          stream);
        ninfer::ops::kv_cache_append_prefix(key_tensor.view({128, 8, width, batch}),
                                            value_tensor.view({128, 8, width, batch}), pos_tensor,
                                            count_tensor, lane_tensor,
                                            {0, static_cast<unsigned>(width)}, cache, stream);
    };
    Graph controls(stream, [&] {
        for (int i = 0; i < 64; ++i) baseline();
    });
    Graph candidates(stream, [&] {
        for (int i = 0; i < 64; ++i) selected();
    });
    hipEvent_t start{}, stop{};
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> timing[2];
    for (int trial = 0; trial < 14; ++trial)
        for (int item = 0; item < 2; ++item) {
            const int route = (trial + item) % 2;
            HIP_CHECK(hipEventRecord(start, stream));
            (route == 0 ? controls : candidates).run(stream);
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            if (trial >= 4) timing[route].push_back(ms * 1000 / 64);
        }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    for (auto& times : timing) std::sort(times.begin(), times.end());
    std::cout << "context_append W=" << width << " C=" << batch << " baseline_us=" << timing[0][5]
              << " fused_us=" << timing[1][5] << '\n';
}

} // namespace

int main() {
    try {
        auto& guard = ninfer::r9700_qualification::device_guard();
        HIP_CHECK(hipSetDevice(guard.select()));
        hipStream_t stream{};
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        for (unsigned tokens : {1U, 4U, 8U, 9U, 16U, 17U, 32U, 64U, 65U, 128U, 129U})
            projection(tokens, stream);
        for (int batch : {1, 4, 8})
            for (int width : {1, 8, 64}) append(width, batch, stream);
        append(2048, 1, stream);
        guard.check("context-after-timing");
        HIP_CHECK(hipStreamDestroy(stream));
        std::cout << "dflash_context_kv: PASS projection_FP64 normalized_append_FP64 "
                     "exact_V_and_unwritten_bytes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dflash_context_kv: FAIL " << error.what() << '\n';
        return 1;
    }
}
