#include "ops/r9700/linear/linear_execution.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr std::uint16_t kCanonicalBf16QuietNan = 0x7FC0U;
constexpr std::uint32_t kPoisonBlockSize       = 256U;
constexpr std::uint32_t kColumnAlignment       = 128U;
constexpr std::size_t kPlaneAlignment          = 256U;
constexpr std::array<std::uint8_t, 16> kQualifiedPrefixCommonAlgorithm{
    0xe2U, 0xe0U, 0x01U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
    0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U};

#ifndef NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE
#define NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE 0
#endif
static_assert(NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE == 0 ||
              NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE == 1);

[[noreturn]] void fail_hipblaslt(const char* operation, hipblasStatus_t status) {
    std::ostringstream message;
    message << operation << " failed with hipBLASLt status " << static_cast<int>(status);
    throw std::runtime_error(message.str());
}

void check_hipblaslt(const char* operation, hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) fail_hipblaslt(operation, status);
}

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* out) noexcept {
    if (out == nullptr || (rhs != 0U && lhs > std::numeric_limits<std::size_t>::max() / rhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t* out) noexcept {
    if (out == nullptr || value > std::numeric_limits<std::size_t>::max() - (alignment - 1U)) {
        return false;
    }
    *out = (value + alignment - 1U) / alignment * alignment;
    return true;
}

bool weight_geometry(const Weight& weight, std::uint32_t* rows, std::uint32_t* columns,
                     std::size_t* code_bytes, std::size_t* scale_offset,
                     std::size_t* scale_bytes, std::size_t* payload_bytes) noexcept {
    if (weight.n <= 0 || weight.k <= 0 || rows == nullptr || columns == nullptr ||
        code_bytes == nullptr || scale_offset == nullptr || scale_bytes == nullptr ||
        payload_bytes == nullptr) {
        return false;
    }
    *rows    = static_cast<std::uint32_t>(weight.n);
    *columns = static_cast<std::uint32_t>(weight.k);
    if (*columns > std::numeric_limits<std::uint32_t>::max() - (kColumnAlignment - 1U)) {
        return false;
    }
    const std::uint32_t padded =
        (*columns + kColumnAlignment - 1U) / kColumnAlignment * kColumnAlignment;
    if (!checked_mul(*rows, padded, code_bytes) ||
        !align_up(*code_bytes, kPlaneAlignment, scale_offset) ||
        !checked_mul(*rows, sizeof(float), scale_bytes) ||
        *scale_bytes > std::numeric_limits<std::size_t>::max() - *scale_offset) {
        return false;
    }
    *payload_bytes = *scale_offset + *scale_bytes;
    return true;
}

bool valid_weight(const Weight& weight, std::uint32_t* rows,
                  std::uint32_t* columns, std::uint32_t* padded_columns) noexcept {
    std::size_t code_bytes = 0;
    std::size_t scale_offset = 0;
    std::size_t scale_bytes = 0;
    std::size_t payload_bytes = 0;
    if (padded_columns == nullptr ||
        !weight_geometry(weight, rows, columns, &code_bytes, &scale_offset,
                         &scale_bytes, &payload_bytes)) {
        return false;
    }
    const auto* payload = static_cast<const std::byte*>(weight.payload);
    const std::uint32_t padded = (*columns + kColumnAlignment - 1U) /
                                 kColumnAlignment * kColumnAlignment;
    *padded_columns = padded;
    return payload != nullptr && weight.qdata == payload &&
           weight.scales == payload + scale_offset && weight.qhigh == nullptr &&
           weight.payload_bytes == payload_bytes && weight.qdata_bytes == code_bytes &&
           weight.scale_bytes == scale_bytes && weight.high_plane_bytes == 0U &&
           weight.qtype == QType::F8E4M3_ROW_F32S &&
           weight.layout == QuantLayout::RowScaled && weight.scale_dtype == DType::FP32 &&
           weight.group == 0 && weight.group_size == 0U && weight.ndim == 2U &&
           weight.shape[0] == weight.n && weight.shape[1] == weight.k &&
           weight.padded_shape[0] == weight.n &&
           weight.padded_shape[1] == static_cast<std::int32_t>(padded) &&
           weight.scale_ne[0] == weight.n &&
           weight.scale_nb[0] == static_cast<std::int64_t>(sizeof(float));
}

bool range(const void* pointer, std::size_t bytes, std::uintptr_t* begin,
           std::uintptr_t* end) noexcept {
    if (pointer == nullptr || bytes == 0U || begin == nullptr || end == nullptr) return false;
    *begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (*begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
    *end = *begin + bytes;
    return true;
}

bool ranges_overlap(const void* lhs, std::size_t lhs_bytes, const void* rhs,
                    std::size_t rhs_bytes) noexcept {
    std::uintptr_t lhs_begin = 0, lhs_end = 0, rhs_begin = 0, rhs_end = 0;
    if (!range(lhs, lhs_bytes, &lhs_begin, &lhs_end) ||
        !range(rhs, rhs_bytes, &rhs_begin, &rhs_end)) {
        return true;
    }
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

__global__ void poison_nonfinite_output(const std::uint32_t* status,
                                        hip_bfloat16* output,
                                        std::size_t output_elements) {
    __shared__ bool poison;
    if (threadIdx.x == 0U) poison = *status != r9700::linear::Fp8ActivationOk;
    __syncthreads();
    if (!poison) return;
    for (std::size_t index = threadIdx.x; index < output_elements; index += blockDim.x) {
        reinterpret_cast<std::uint16_t*>(output)[index] = kCanonicalBf16QuietNan;
    }
}

} // namespace

struct LinearExecution::Impl {
    struct Profile {
        struct ViableAlgorithm {
            hipblasLtMatmulAlgo_t algorithm{};
            std::size_t workspace_bytes = 0;
        };
        PreparedProfile public_profile{};
        r9700::linear::Fp8ActivationWorkspace activation{};
        hipblasLtMatmulDesc_t operation       = nullptr;
        hipblasLtMatrixLayout_t activation_layout = nullptr;
        hipblasLtMatrixLayout_t output_layout = nullptr;
        hipblasLtMatmulAlgo_t algorithm{};
        hipblasLtMatmulAlgo_t default_algorithm{};
        std::size_t default_workspace_bytes = 0;
        std::vector<ViableAlgorithm> viable_algorithms;

        ~Profile() {
            if (output_layout != nullptr) (void)hipblasLtMatrixLayoutDestroy(output_layout);
            if (activation_layout != nullptr) {
                (void)hipblasLtMatrixLayoutDestroy(activation_layout);
            }
            if (operation != nullptr) (void)hipblasLtMatmulDescDestroy(operation);
        }
    };

    hipblasLtHandle_t handle              = nullptr;
    hipblasLtMatrixLayout_t weight_layout = nullptr;
    hipblasLtMatmulPreference_t preference = nullptr;
    int hipblaslt_version = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t padded_columns = 0;
    const void* weight_payload = nullptr;
    std::size_t weight_payload_bytes = 0;
    const void* weight_codes = nullptr;
    const void* weight_scales = nullptr;
    void* activation_storage = nullptr;
    std::size_t activation_storage_capacity_bytes = 0;
    void* matmul_workspace = nullptr;
    std::size_t matmul_workspace_bytes = 0;
    std::vector<std::unique_ptr<Profile>> profiles;

    ~Impl() {
        profiles.clear();
        if (preference != nullptr) (void)hipblasLtMatmulPreferenceDestroy(preference);
        if (weight_layout != nullptr) (void)hipblasLtMatrixLayoutDestroy(weight_layout);
        if (handle != nullptr) (void)hipblasLtDestroy(handle);
    }

    [[nodiscard]] Profile* find(std::uint32_t tokens) noexcept {
        for (const auto& profile : profiles) {
            if (profile->public_profile.tokens == tokens) return profile.get();
        }
        return nullptr;
    }

    [[nodiscard]] const Profile* find(std::uint32_t tokens) const noexcept {
        for (const auto& profile : profiles) {
            if (profile->public_profile.tokens == tokens) return profile.get();
        }
        return nullptr;
    }
};

LinearExecution::LinearExecution(const Weight& weight, void* activation_storage,
                                 std::size_t activation_storage_capacity_bytes,
                                 void* matmul_workspace,
                                 std::size_t matmul_workspace_bytes)
    : impl_(std::make_unique<Impl>()) {
    if (!valid_weight(weight, &impl_->rows, &impl_->columns,
                      &impl_->padded_columns) ||
        activation_storage == nullptr || activation_storage_capacity_bytes == 0U ||
        reinterpret_cast<std::uintptr_t>(activation_storage) % kPlaneAlignment != 0U ||
        matmul_workspace_bytes > kMaximumMatmulWorkspaceBytes ||
        (matmul_workspace_bytes != 0U &&
         (matmul_workspace == nullptr ||
          reinterpret_cast<std::uintptr_t>(matmul_workspace) % 16U != 0U))) {
        throw std::invalid_argument("invalid row-scaled E4M3 Linear execution binding");
    }
    const bool matmul_overlaps =
        matmul_workspace_bytes != 0U &&
        (ranges_overlap(weight.payload, weight.payload_bytes, matmul_workspace,
                        matmul_workspace_bytes) ||
         ranges_overlap(activation_storage, activation_storage_capacity_bytes,
                        matmul_workspace, matmul_workspace_bytes));
    if (ranges_overlap(weight.payload, weight.payload_bytes, activation_storage,
                       activation_storage_capacity_bytes) ||
        matmul_overlaps) {
        throw std::invalid_argument("row-scaled E4M3 Linear retained regions overlap");
    }
    impl_->weight_payload = weight.payload;
    impl_->weight_payload_bytes = static_cast<std::size_t>(weight.payload_bytes);
    impl_->weight_codes = weight.qdata;
    impl_->weight_scales = weight.scales;
    impl_->activation_storage = activation_storage;
    impl_->activation_storage_capacity_bytes = activation_storage_capacity_bytes;
    impl_->matmul_workspace = matmul_workspace;
    impl_->matmul_workspace_bytes = matmul_workspace_bytes;

    check_hipblaslt("hipblasLtCreate", hipblasLtCreate(&impl_->handle));
    check_hipblaslt("hipblasLtGetVersion",
                    hipblasLtGetVersion(impl_->handle, &impl_->hipblaslt_version));
    check_hipblaslt("hipblasLtMatrixLayoutCreate(weight)",
                    hipblasLtMatrixLayoutCreate(&impl_->weight_layout, HIP_R_8F_E4M3,
                                                impl_->columns, impl_->rows,
                                                impl_->padded_columns));
    check_hipblaslt("hipblasLtMatmulPreferenceCreate",
                    hipblasLtMatmulPreferenceCreate(&impl_->preference));
    const std::uint64_t workspace_limit = matmul_workspace_bytes;
    check_hipblaslt(
        "hipblasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE)",
        hipblasLtMatmulPreferenceSetAttribute(impl_->preference,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &workspace_limit, sizeof(workspace_limit)));
}

LinearExecution::~LinearExecution() = default;
LinearExecution::LinearExecution(LinearExecution&&) noexcept = default;
LinearExecution& LinearExecution::operator=(LinearExecution&&) noexcept = default;

std::size_t LinearExecution::activation_workspace_capacity_bytes(
    std::uint32_t tokens, std::uint32_t columns) noexcept {
    return r9700::linear::fp8_activation_workspace_capacity_bytes(tokens, columns);
}

const LinearExecution::PreparedProfile& LinearExecution::prepare(std::uint32_t tokens) {
    if (impl_ == nullptr || tokens == 0U) {
        throw std::invalid_argument("invalid row-scaled E4M3 Linear preparation width");
    }
    if (const Impl::Profile* existing = impl_->find(tokens)) {
        return existing->public_profile;
    }
    const std::size_t activation_bytes =
        activation_workspace_capacity_bytes(tokens, impl_->columns);
    if (activation_bytes == 0U ||
        activation_bytes > impl_->activation_storage_capacity_bytes) {
        throw std::invalid_argument("row-scaled E4M3 activation region is too small");
    }

    auto profile = std::make_unique<Impl::Profile>();
    profile->public_profile.tokens = tokens;
    profile->public_profile.activation_workspace_bytes = activation_bytes;
    const hipError_t bind_status = r9700::linear::fp8_bind_activation_workspace(
        impl_->activation_storage, activation_bytes, tokens, impl_->columns,
        &profile->activation);
    if (bind_status != hipSuccess) {
        throw std::invalid_argument("invalid row-scaled E4M3 activation workspace");
    }

    check_hipblaslt("hipblasLtMatmulDescCreate",
                    hipblasLtMatmulDescCreate(&profile->operation,
                                              HIPBLAS_COMPUTE_32F, HIP_R_32F));
    const hipblasOperation_t transpose_weight = HIPBLAS_OP_T;
    check_hipblaslt(
        "hipblasLtMatmulDescSetAttribute(TRANSA)",
        hipblasLtMatmulDescSetAttribute(profile->operation,
                                        HIPBLASLT_MATMUL_DESC_TRANSA,
                                        &transpose_weight, sizeof(transpose_weight)));
    const hipblasLtMatmulMatrixScale_t outer_scale =
        HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    check_hipblaslt(
        "hipblasLtMatmulDescSetAttribute(A_SCALE_MODE)",
        hipblasLtMatmulDescSetAttribute(profile->operation,
                                        HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                        &outer_scale, sizeof(outer_scale)));
    check_hipblaslt(
        "hipblasLtMatmulDescSetAttribute(B_SCALE_MODE)",
        hipblasLtMatmulDescSetAttribute(profile->operation,
                                        HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                        &outer_scale, sizeof(outer_scale)));
    const void* weight_scale = impl_->weight_scales;
    const void* token_scale = profile->activation.scales;
    check_hipblaslt(
        "hipblasLtMatmulDescSetAttribute(A_SCALE_POINTER)",
        hipblasLtMatmulDescSetAttribute(profile->operation,
                                        HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                        &weight_scale, sizeof(weight_scale)));
    check_hipblaslt(
        "hipblasLtMatmulDescSetAttribute(B_SCALE_POINTER)",
        hipblasLtMatmulDescSetAttribute(profile->operation,
                                        HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                        &token_scale, sizeof(token_scale)));
    check_hipblaslt("hipblasLtMatrixLayoutCreate(activation)",
                    hipblasLtMatrixLayoutCreate(&profile->activation_layout,
                                                HIP_R_8F_E4M3, impl_->columns,
                                                tokens, impl_->padded_columns));
    check_hipblaslt("hipblasLtMatrixLayoutCreate(output)",
                    hipblasLtMatrixLayoutCreate(&profile->output_layout, HIP_R_16BF,
                                                impl_->rows, tokens, impl_->rows));

    std::array<hipblasLtMatmulHeuristicResult_t, 8> results{};
    int result_count = 0;
    check_hipblaslt(
        "hipblasLtMatmulAlgoGetHeuristic",
        hipblasLtMatmulAlgoGetHeuristic(
            impl_->handle, profile->operation, impl_->weight_layout,
            profile->activation_layout, profile->output_layout,
            profile->output_layout, impl_->preference,
            static_cast<int>(results.size()), results.data(), &result_count));
    const hipblasLtMatmulHeuristicResult_t* selected = nullptr;
    for (int index = 0; index < result_count; ++index) {
        if (results[index].state == HIPBLAS_STATUS_SUCCESS &&
            results[index].workspaceSize <= impl_->matmul_workspace_bytes) {
            LinearExecution::AlgorithmProfile candidate;
            candidate.matmul_workspace_bytes = results[index].workspaceSize;
            candidate.algorithm_max_workspace_bytes = results[index].algo.max_workspace_bytes;
            candidate.waves_count = results[index].wavesCount;
            candidate.heuristic_rank = index;
            std::copy(std::begin(results[index].algo.data), std::end(results[index].algo.data),
                      candidate.fingerprint.begin());
            profile->public_profile.viable_algorithms.push_back(candidate);
            profile->viable_algorithms.push_back(
                Impl::Profile::ViableAlgorithm{results[index].algo,
                                               results[index].workspaceSize});
            if (selected == nullptr) selected = &results[index];
        }
    }
    if (selected == nullptr) {
        throw std::runtime_error("no supported row-scaled E4M3 hipBLASLt algorithm");
    }
    profile->default_algorithm = selected->algo;
    profile->default_workspace_bytes = selected->workspaceSize;
    if constexpr (NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE == 1) {
        if (impl_->rows == 34816U && impl_->columns == 5120U && tokens == 128U) {
            std::array<std::uint8_t, 16> fingerprint{};
            std::copy(std::begin(selected->algo.data), std::end(selected->algo.data),
                      fingerprint.begin());
            if (fingerprint != kQualifiedPrefixCommonAlgorithm) {
                throw std::runtime_error(
                    "T128 default FP8 algorithm differs from qualified common fingerprint");
            }
        } else if (impl_->rows == 34816U && impl_->columns == 5120U && tokens == 129U) {
            if (impl_->find(128U) == nullptr) (void)prepare(128U);
            const Impl::Profile* reference = impl_->find(128U);
            if (reference == nullptr ||
                reference->public_profile.selected_algorithm_fingerprint !=
                    kQualifiedPrefixCommonAlgorithm) {
                throw std::logic_error("internal qualified T128 preparation failed");
            }
            selected = nullptr;
            for (int index = 0; index < result_count; ++index) {
                std::array<std::uint8_t, 16> fingerprint{};
                std::copy(std::begin(results[index].algo.data), std::end(results[index].algo.data),
                          fingerprint.begin());
                if (results[index].state == HIPBLAS_STATUS_SUCCESS &&
                    results[index].workspaceSize <= impl_->matmul_workspace_bytes &&
                    fingerprint == kQualifiedPrefixCommonAlgorithm) {
                    selected = &results[index];
                    break;
                }
            }
            if (selected == nullptr) {
                throw std::runtime_error(
                    "T129 heuristic set lacks qualified common FP8 algorithm fingerprint");
            }
        }
    }
    profile->algorithm = selected->algo;
    profile->public_profile.selected_matmul_workspace_bytes = selected->workspaceSize;
    profile->public_profile.selected_algorithm_max_workspace_bytes =
        selected->algo.max_workspace_bytes;
    profile->public_profile.selected_waves_count = selected->wavesCount;
    profile->public_profile.selected_heuristic_rank =
        static_cast<int>(selected - results.data());
    profile->public_profile.heuristic_result_count = result_count;
    std::copy(std::begin(selected->algo.data), std::end(selected->algo.data),
              profile->public_profile.selected_algorithm_fingerprint.begin());

    impl_->profiles.push_back(std::move(profile));
    return impl_->profiles.back()->public_profile;
}

const LinearExecution::PreparedProfile* LinearExecution::prepared_profile(
    std::uint32_t tokens) const noexcept {
    if (impl_ == nullptr) return nullptr;
    const Impl::Profile* profile = impl_->find(tokens);
    return profile == nullptr ? nullptr : &profile->public_profile;
}

const r9700::linear::Fp8ActivationWorkspace* LinearExecution::activation_workspace(
    std::uint32_t tokens) const noexcept {
    if (impl_ == nullptr) return nullptr;
    const Impl::Profile* profile = impl_->find(tokens);
    return profile == nullptr ? nullptr : &profile->activation;
}

std::uint32_t LinearExecution::rows() const noexcept {
    return impl_ == nullptr ? 0U : impl_->rows;
}

std::uint32_t LinearExecution::columns() const noexcept {
    return impl_ == nullptr ? 0U : impl_->columns;
}

int LinearExecution::hipblaslt_version() const noexcept {
    return impl_ == nullptr ? 0 : impl_->hipblaslt_version;
}

LinearExecution::LaunchStatus LinearExecution::run(std::uint32_t tokens,
                                                    const hip_bfloat16* input,
                                                    hip_bfloat16* output,
                                                    hipStream_t stream) noexcept {
    Impl::Profile* profile = impl_ == nullptr ? nullptr : impl_->find(tokens);
    std::size_t input_elements = 0;
    std::size_t output_elements = 0;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    if (profile == nullptr || input == nullptr || output == nullptr ||
        !checked_mul(tokens, impl_->columns, &input_elements) ||
        !checked_mul(input_elements, sizeof(hip_bfloat16), &input_bytes) ||
        !checked_mul(tokens, impl_->rows, &output_elements) ||
        !checked_mul(output_elements, sizeof(hip_bfloat16), &output_bytes) ||
        ranges_overlap(input, input_bytes, output, output_bytes) ||
        ranges_overlap(input, input_bytes, impl_->weight_payload,
                       impl_->weight_payload_bytes) ||
        ranges_overlap(input, input_bytes, impl_->activation_storage,
                       impl_->activation_storage_capacity_bytes) ||
        (impl_->matmul_workspace_bytes != 0U &&
         ranges_overlap(input, input_bytes, impl_->matmul_workspace,
                        impl_->matmul_workspace_bytes)) ||
        ranges_overlap(output, output_bytes, impl_->weight_payload,
                       impl_->weight_payload_bytes) ||
        ranges_overlap(output, output_bytes, impl_->activation_storage,
                       impl_->activation_storage_capacity_bytes) ||
        (impl_->matmul_workspace_bytes != 0U &&
         ranges_overlap(output, output_bytes, impl_->matmul_workspace,
                        impl_->matmul_workspace_bytes))) {
        return {hipErrorInvalidValue, HIPBLAS_STATUS_SUCCESS};
    }

    const hipError_t quantize_status = r9700::linear::fp8_quantize_activation(
        r9700::linear::Fp8ActivationQuantizeArgs{input, profile->activation}, stream);
    if (quantize_status != hipSuccess) {
        return {quantize_status, HIPBLAS_STATUS_SUCCESS};
    }
    constexpr float alpha = 1.0F;
    constexpr float beta  = 0.0F;
    const hipblasStatus_t matmul_status = hipblasLtMatmul(
        impl_->handle, profile->operation, &alpha, impl_->weight_codes,
        impl_->weight_layout, profile->activation.codes,
        profile->activation_layout, &beta, output, profile->output_layout, output,
        profile->output_layout, &profile->algorithm, impl_->matmul_workspace,
        profile->public_profile.selected_matmul_workspace_bytes, stream);
    if (matmul_status != HIPBLAS_STATUS_SUCCESS) {
        return {hipSuccess, matmul_status};
    }
    hipLaunchKernelGGL(poison_nonfinite_output, dim3(1), dim3(kPoisonBlockSize), 0,
                       stream, profile->activation.status, output, output_elements);
    return {hipGetLastError(), HIPBLAS_STATUS_SUCCESS};
}

LinearExecution::LaunchStatus LinearExecution::run_default_heuristic(
    std::uint32_t tokens, const hip_bfloat16* input, hip_bfloat16* output,
    hipStream_t stream) noexcept {
    Impl::Profile* profile = impl_ == nullptr ? nullptr : impl_->find(tokens);
    if (profile == nullptr) return {hipErrorInvalidValue, HIPBLAS_STATUS_SUCCESS};
    const hipblasLtMatmulAlgo_t selected = profile->algorithm;
    const std::size_t selected_workspace =
        profile->public_profile.selected_matmul_workspace_bytes;
    profile->algorithm = profile->default_algorithm;
    profile->public_profile.selected_matmul_workspace_bytes = profile->default_workspace_bytes;
    const LaunchStatus status = run(tokens, input, output, stream);
    profile->algorithm = selected;
    profile->public_profile.selected_matmul_workspace_bytes = selected_workspace;
    return status;
}

LinearExecution::LaunchStatus LinearExecution::run_qualified_algorithm(
    std::uint32_t tokens, const std::array<std::uint8_t, 16>& fingerprint,
    const hip_bfloat16* input, hip_bfloat16* output, hipStream_t stream) noexcept {
    Impl::Profile* profile = impl_ == nullptr ? nullptr : impl_->find(tokens);
    if (profile == nullptr ||
        profile->viable_algorithms.size() != profile->public_profile.viable_algorithms.size()) {
        return {hipErrorInvalidValue, HIPBLAS_STATUS_SUCCESS};
    }
    std::size_t found = profile->viable_algorithms.size();
    for (std::size_t index = 0; index < profile->public_profile.viable_algorithms.size(); ++index) {
        if (profile->public_profile.viable_algorithms[index].fingerprint == fingerprint) {
            if (found != profile->viable_algorithms.size()) {
                return {hipErrorInvalidValue, HIPBLAS_STATUS_SUCCESS};
            }
            found = index;
        }
    }
    if (found == profile->viable_algorithms.size()) {
        return {hipErrorInvalidValue, HIPBLAS_STATUS_SUCCESS};
    }
    const hipblasLtMatmulAlgo_t selected = profile->algorithm;
    const std::size_t selected_workspace =
        profile->public_profile.selected_matmul_workspace_bytes;
    profile->algorithm = profile->viable_algorithms[found].algorithm;
    profile->public_profile.selected_matmul_workspace_bytes =
        profile->viable_algorithms[found].workspace_bytes;
    const LaunchStatus status = run(tokens, input, output, stream);
    profile->algorithm = selected;
    profile->public_profile.selected_matmul_workspace_bytes = selected_workspace;
    return status;
}

} // namespace ninfer::ops
