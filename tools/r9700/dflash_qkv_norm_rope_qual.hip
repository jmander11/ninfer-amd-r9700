// Qualification of the DFlash drafter attention front (dflash_qkv_norm_rope) against the chain it
// replaces: the fused-row copies, the eager Q/K RMSNorm (no unit offset) and the DFlash
// full-dimension RoPE, with and without the query heads (the context-append form), at
// verification and prefill widths.
#include "ninfer/ops/rope.h"
#include "ops/r9700/eager/eager_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::Tensor;
namespace eager = ninfer::ops::r9700::eager;

constexpr int kHead = 128, kQ = 32, kK = 8, kRows = (kQ + 2 * kK) * kHead;

void check(hipError_t error, const char* what) {
    if (error != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(error));
}
void require(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error(what);
}

template <class T>
struct Device {
    T* data = nullptr;
    std::size_t count = 0;
    explicit Device(std::size_t n) : count(n) { check(hipMalloc(&data, n * sizeof(T)), "malloc"); }
    ~Device() { (void)hipFree(data); }
    Device(const Device&) = delete;
    void upload(const std::vector<T>& host) {
        check(hipMemcpy(data, host.data(), count * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    std::vector<T> download() const {
        std::vector<T> host(count);
        check(hipMemcpy(host.data(), data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return host;
    }
};

struct Rng {
    std::uint64_t state;
    std::uint32_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(state >> 32U);
    }
    double uniform() { return (next() + 0.5) / 4294967296.0; }
    double normal() {
        return std::sqrt(-2.0 * std::log(uniform())) * std::cos(6.283185307 * uniform());
    }
};

std::uint16_t bf16_bits(float value) { return hip_bfloat16(value).data; }

void run(int tokens, bool with_query, Rng& rng, hipStream_t stream) {
    const std::string label =
        "T" + std::to_string(tokens) + (with_query ? " query+key+value" : " key+value");
    const std::size_t fused_elements = static_cast<std::size_t>(tokens) * kRows;
    std::vector<std::uint16_t> fused(fused_elements), q_norm(kHead), k_norm(kHead);
    for (auto& v : fused) v = bf16_bits(static_cast<float>(rng.normal() * 2.0));
    for (auto& v : q_norm) v = bf16_bits(static_cast<float>(1.0 + rng.normal() * 0.2));
    for (auto& v : k_norm) v = bf16_bits(static_cast<float>(1.0 + rng.normal() * 0.2));
    std::vector<std::int32_t> positions(tokens);
    for (auto& p : positions) p = static_cast<std::int32_t>(rng.next() % 262144U);
    Device<std::uint16_t> device_fused(fused_elements), device_q_norm(kHead),
        device_k_norm(kHead);
    Device<std::int32_t> device_positions(tokens);
    device_fused.upload(fused);
    device_q_norm.upload(q_norm);
    device_k_norm.upload(k_norm);
    device_positions.upload(positions);
    const std::size_t q_elements = static_cast<std::size_t>(tokens) * kQ * kHead;
    const std::size_t kv_elements = static_cast<std::size_t>(tokens) * kK * kHead;
    Device<std::uint16_t> raw_q(q_elements), raw_k(kv_elements), ref_q(q_elements),
        ref_k(kv_elements), ref_v(kv_elements), got_q(q_elements), got_k(kv_elements),
        got_v(kv_elements);
    // Reference chain: row copies, eager RMSNorm, RoPE.
    const std::size_t pitch = kRows * sizeof(std::uint16_t);
    const auto copy_rows = [&](int first, int rows, std::uint16_t* out) {
        check(hipMemcpy2DAsync(out, rows * sizeof(std::uint16_t), device_fused.data + first, pitch,
                               rows * sizeof(std::uint16_t), tokens, hipMemcpyDeviceToDevice,
                               stream), "copy rows");
    };
    copy_rows(0, kQ * kHead, raw_q.data);
    copy_rows(kQ * kHead, kK * kHead, raw_k.data);
    copy_rows((kQ + kK) * kHead, kK * kHead, ref_v.data);
    const float eps = 1.0e-6F;
    check(eager::rmsnorm_bf16(reinterpret_cast<const hip_bfloat16*>(raw_q.data),
                              reinterpret_cast<const hip_bfloat16*>(device_q_norm.data),
                              reinterpret_cast<hip_bfloat16*>(ref_q.data), kHead, kQ * tokens,
                              eps, false, stream), "q rmsnorm");
    check(eager::rmsnorm_bf16(reinterpret_cast<const hip_bfloat16*>(raw_k.data),
                              reinterpret_cast<const hip_bfloat16*>(device_k_norm.data),
                              reinterpret_cast<hip_bfloat16*>(ref_k.data), kHead, kK * tokens,
                              eps, false, stream), "k rmsnorm");
    Tensor position_tensor(device_positions.data, DType::I32, {tokens});
    Tensor q_tensor(ref_q.data, DType::BF16, {kHead, kQ, tokens});
    Tensor k_tensor(ref_k.data, DType::BF16, {kHead, kK, tokens});
    ninfer::ops::rope(position_tensor, kHead, 1.0e7F, q_tensor, k_tensor, stream);
    // Fused Op.
    Tensor fused_tensor(device_fused.data, DType::BF16, {kRows, tokens});
    Tensor q_norm_tensor(device_q_norm.data, DType::BF16, {kHead});
    Tensor k_norm_tensor(device_k_norm.data, DType::BF16, {kHead});
    Tensor got_q_tensor(got_q.data, DType::BF16, {kHead, kQ, tokens});
    Tensor got_k_tensor(got_k.data, DType::BF16, {kHead, kK, tokens});
    Tensor got_v_tensor(got_v.data, DType::BF16, {kHead, kK, tokens});
    ninfer::ops::dflash_qkv_norm_rope(position_tensor, fused_tensor, q_norm_tensor, k_norm_tensor,
                                      eps, with_query ? &got_q_tensor : nullptr, got_k_tensor,
                                      got_v_tensor, stream);
    check(hipStreamSynchronize(stream), "sync");
    // T <= 8 (every verification width) is bit-identical. Wider rows meet the eager RoPE's
    // combined kernel, whose contraction of x cos - y sin rounds differently; there each element
    // must agree within one rotation rounding of its head's magnitude.
    const auto compare = [&](const Device<std::uint16_t>& ref, const Device<std::uint16_t>& got,
                             const char* what) {
        const auto a = ref.download(), b = got.download();
        const auto value = [](std::uint16_t bits) {
            const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
            float result;
            std::memcpy(&result, &word, sizeof(result));
            return result;
        };
        for (std::size_t head = 0; head < a.size() / kHead; ++head) {
            float scale = 0.0F;
            for (int f = 0; f < kHead; ++f)
                scale = std::max(scale, std::fabs(value(a[head * kHead + f])));
            for (int f = 0; f < kHead; ++f) {
                const std::size_t i = head * kHead + f;
                if (a[i] == b[i]) continue;
                require(tokens > 8 && std::fabs(value(a[i]) - value(b[i])) <= scale / 128.0F,
                        label + " " + what + " differs at " + std::to_string(i));
            }
        }
    };
    compare(ref_k, got_k, "key");
    require(ref_v.download() == got_v.download(), label + " value differs");
    if (with_query) compare(ref_q, got_q, "query");
}

} // namespace

int main() {
    try {
        hipStream_t stream = nullptr;
        check(hipStreamCreate(&stream), "stream");
        Rng rng{0xdf1a5bULL};
        for (const int tokens : {1, 5, 6, 24, 2048}) {
            run(tokens, true, rng, stream);
            run(tokens, false, rng, stream);
        }
        std::cout << "dflash qkv norm rope qualification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
