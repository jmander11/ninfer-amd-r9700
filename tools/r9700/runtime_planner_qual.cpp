#include "core/device.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/dflash2_path_select.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/grouped_dynamic_conv.h"
#include "ninfer/ops/swa.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"
#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#include "targets/qwen3_8_27b/impl/r9700_full_attention.h"
#include "targets/qwen3_8_27b/impl/variant.h"

#define NINFER_QWEN3_VARIANT ::ninfer::targets::qwen3_8_27b::detail::Variant
#define NINFER_QWEN3_RUNTIME_NS qwen3_8_27b_r9700
#include "targets/qwen3/impl/runtime/layouts.h"
#include "targets/qwen3/impl/runtime/workspace_recipe.h"
#undef NINFER_QWEN3_RUNTIME_NS
#undef NINFER_QWEN3_VARIANT

#include <ninfer/targets/qwen3/runtime.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Variant = ninfer::targets::qwen3_8_27b::detail::Variant;

void require(bool condition, const char* message);

template <class Function>
void require_invalid_argument(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void qualify_host_request_lane_cap() {
    constexpr auto max_batch = static_cast<std::int32_t>(ninfer::kMaximumConcurrency);
    constexpr auto too_many = max_batch + 1;
    constexpr auto q4 = ninfer::QType::Q4G64_F16S;
    constexpr auto profile = Variant::WeightsProfile::R9700Q4G64DFlash2Q4Evaluation;

    (void)ninfer::ops::grouped_dynamic_conv_prepare_workspace_capacity_bytes(q4, 1, 16,
                                                                             max_batch);
    (void)ninfer::ops::dflash2_path_select_workspace_capacity_bytes(q4, 1, 16, max_batch);
    (void)ninfer::ops::swa_workspace_capacity_bytes({0, 4096}, 1, 16, max_batch);
    (void)ninfer::ops::bidirectional_gqa_attention_workspace_capacity_bytes(
        {0, 4096}, 1, 16, max_batch);
    (void)ninfer::ops::gated_delta_net_replay_record_workspace_capacity_bytes(
        48, max_batch, 16);
    (void)Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
        profile, ninfer::targets::qwen3::TextPhase::Verify, max_batch, 1, 16);
    (void)Variant::gdn_input_projection_record_workspace_capacity_bytes(
        profile, ninfer::targets::qwen3::TextPhase::Verify, max_batch, 2, 16);

    require_invalid_argument(
        [&] {
            (void)ninfer::ops::grouped_dynamic_conv_prepare_workspace_capacity_bytes(
                q4, 1, 16, too_many);
        },
        "grouped dynamic convolution accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)ninfer::ops::dflash2_path_select_workspace_capacity_bytes(q4, 1, 16,
                                                                            too_many);
        },
        "DFlash path selection accepted an oversized request batch");
    require_invalid_argument(
        [&] { (void)ninfer::ops::swa_workspace_capacity_bytes({0, 4096}, 1, 16, too_many); },
        "SWA accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)ninfer::ops::bidirectional_gqa_attention_workspace_capacity_bytes(
                {0, 4096}, 1, 16, too_many);
        },
        "bidirectional attention accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)ninfer::ops::gated_delta_net_replay_record_workspace_capacity_bytes(
                48, too_many, 16);
        },
        "GDN replay accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
                profile, ninfer::targets::qwen3::TextPhase::Verify, too_many, 1, 16);
        },
        "target GDN snapshot accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)Variant::gdn_input_projection_record_workspace_capacity_bytes(
                profile, ninfer::targets::qwen3::TextPhase::Verify, too_many, 2, 16);
        },
        "target GDN record accepted an oversized request batch");
}

void qualify_host_split512_routing() {
    namespace kv = ninfer::ops::r9700::kv;
    namespace q27 = ninfer::targets::qwen3_8_27b::detail;
    require(!kv::use_split512_attention(1U, 8191U) &&
                kv::use_split512_attention(1U, 8192U) &&
                !kv::use_split512_attention(1U, 8192U, true) &&
                !kv::use_split512_attention(4U, 8191U) &&
                kv::use_split512_attention(4U, 8192U) &&
                kv::use_split512_attention(4U, 8192U, true) &&
                !kv::use_split512_attention(2U, 32768U) &&
                !kv::use_split512_attention(3U, 32768U) &&
                !kv::use_split512_attention(5U, 32768U),
            "split-512 production selector differs from its admitted boundary");
    require(!kv::use_dense_prefill_attention(127U, 127U) &&
                kv::use_dense_prefill_attention(128U, 128U) &&
                kv::use_dense_prefill_attention(4096U, 4096U) &&
                !kv::use_dense_prefill_attention(4097U, 4097U) &&
                !kv::use_dense_prefill_attention(4096U, 8192U),
            "dense-prefill production selector differs from its admitted boundary");
    require(q27::r9700_full_attention_workspace_capacity_bytes(1U, 8192U, false) ==
                kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(1U, 8192U) &&
                q27::r9700_full_attention_workspace_capacity_bytes(1U, 8192U, true) == 0U &&
                q27::r9700_full_attention_workspace_capacity_bytes(4U, 8192U, false) ==
                    kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(4U, 8192U) &&
                q27::r9700_full_attention_workspace_capacity_bytes(4U, 8192U, true) ==
                    kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(4U, 8192U) &&
                q27::r9700_full_attention_workspace_capacity_bytes(4U, 8191U, true) == 0U,
            "split-512 caller-owned workspace selection differs");
    require(q27::r9700_full_attention_workspace_capacity_bytes(2048U, 2048U, false) ==
                kv::fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(2048U) &&
                q27::r9700_full_attention_workspace_capacity_bytes(4096U, 262144U, false) ==
                kv::fp8_int4_kv_attention_dense_prefill_full_score_workspace_bytes(4096U),
            "dense full-score caller-owned planner peak differs");
    const auto has_classes = [](const std::vector<Variant::GraphExecutionProfile>& profiles,
                                std::initializer_list<std::uint32_t> wanted) {
        return std::all_of(wanted.begin(), wanted.end(), [&](std::uint32_t value) {
            return std::any_of(profiles.begin(), profiles.end(), [&](const auto& profile) {
                return profile.topology_class == value;
            });
        });
    };
    const auto class_at = [](const std::vector<Variant::GraphExecutionProfile>& profiles,
                             std::uint32_t frontier) {
        const auto found = std::find_if(profiles.begin(), profiles.end(), [&](const auto& profile) {
            return profile.min <= frontier && frontier <= profile.max;
        });
        if (found == profiles.end()) {
            throw std::runtime_error("split-512 graph profile coverage is incomplete");
        }
        return found->topology_class;
    };
    const auto ordinary = Variant::ordinary_graph_profiles(32768U);
    const auto mtp = Variant::mtp_graph_profiles(32768U, 3U);
    const auto dflash = Variant::dflash_graph_profiles(32768U, 3U, 1U, 4U);
    require(has_classes(ordinary, {0U, 1U}) && has_classes(mtp, {0U, 1U, 3U}) &&
                has_classes(dflash, {0U, 2U}) &&
                class_at(ordinary, 8190U) == 0U && class_at(ordinary, 8191U) == 1U &&
                class_at(mtp, 8185U) == 0U && class_at(mtp, 8186U) == 1U &&
                class_at(mtp, 8187U) == 1U && class_at(mtp, 8188U) == 3U &&
                class_at(dflash, 8187U) == 0U && class_at(dflash, 8188U) == 2U,
            "split-512 Device Graph topology classes are incomplete");
    std::printf("r9700_runtime_planner: PASS host split-512 production routing\n");
}
using WeightsProfile = ninfer::targets::qwen3_8_27b::detail::WeightsProfile;

constexpr std::uint32_t kCapacityEnvelopeContext = 262144U;
constexpr std::uint32_t kCapacityEnvelopeChunk   = 4096U;
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION) && \
    NINFER_R9700_XATTENTION_STRIDE == 16 && NINFER_R9700_XATTENTION_TAU_PERMILLE == 900
constexpr std::size_t kCapacityEnvelopeSparseLeafBytes = 601361408ULL;
constexpr std::size_t kCapacityEnvelopeWorkspaceBytes  = 1085967619ULL;
#endif

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::size_t qualify_plan(ninfer::DeviceContext& device, std::uint32_t concurrency,
                         ninfer::SpeculativeBackend backend, std::uint32_t drafts,
                         bool vision, std::uint32_t dflash_verify_width = 0,
                         WeightsProfile weights_profile =
                             WeightsProfile::R9700W8G32Candidate,
                         bool use_device_graph = false, std::uint32_t max_context = 2048,
                         std::uint32_t prefill_chunk = 128) {
    ninfer::EngineOptions options;
    options.device            = device.device;
    options.max_context       = max_context;
    options.kv_capacity       = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
    options.max_concurrency   = concurrency;
    options.prefill_chunk     = prefill_chunk;
    options.enable_vision     = vision;
    options.use_device_graph  = use_device_graph;
    options.speculative.backend      = backend;
    options.speculative.draft_tokens = drafts;
    options.speculative.dflash_verify_width = dflash_verify_width;

    auto planner = ninfer::targets::qwen3::make_sequence_planner<Variant>(
        device, options, weights_profile);
    const auto curve = planner.capacity_curve();
    require(curve.main_page_tokens == 64U, "planner page size is not the typed-cache page size");
    const std::uint32_t logical_pages = 1U + (max_context - 1U) / curve.main_page_tokens;
    require(curve.minimum_main_page_groups == std::max(logical_pages, concurrency),
            "planner minimum physical page count is wrong");
    require(curve.maximum_main_page_groups == concurrency * logical_pages,
            "planner maximum physical page count is wrong");
    require(curve.minimum_device_reservation_bytes != 0U,
            "planner produced an empty device reservation");

    auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);
    require(plan.capacity() == options.max_context, "plan logical capacity changed");
    require(plan.kv_capacity() >= options.max_context &&
                plan.kv_capacity() - options.max_context < curve.main_page_tokens,
            "plan physical KV capacity does not match page-rounded logical capacity");
    require(plan.max_concurrency() == concurrency, "plan concurrency changed");
    require(plan.device_reservation_bytes() == curve.minimum_device_reservation_bytes,
            "plan reservation disagrees with its affine curve");
    require(plan.workspace_capacity_bytes() != 0U, "plan workspace is empty");
    require((plan.request_transient_capacity_bytes() != 0U) == vision,
            "plan Vision transient ownership is wrong");
    // Rebuild the runtime allocation prefix through ordinary Text prefill attention. Comparing
    // only the leaf scratch size with the global arena would miss live query/gate/K/V/result
    // tensors that precede Variant::full_attention in the same scope.
    {
        using TextConfig = Variant::TextConfig;
        const std::int32_t chunk = static_cast<std::int32_t>(
            std::min<std::uint32_t>(options.prefill_chunk, options.max_context));
        ninfer::WorkspaceLayoutBuilder runtime_prefix;
        (void)ninfer::targets::qwen3::detail::qwen3_8_27b_r9700::workspace_recipe::
            text_prefill_roots<TextConfig>(runtime_prefix, chunk, vision ? 3 : 0,
                                           vision ? chunk : 0);
        {
            auto stage = runtime_prefix.scope();
            (void)ninfer::targets::qwen3::detail::qwen3_8_27b_r9700::workspace_recipe::
                text_attention_projection<TextConfig>(runtime_prefix, chunk);
            {
                auto scratch = runtime_prefix.scope();
                const std::size_t bytes = Variant::attention_projection_workspace_capacity_bytes(
                    weights_profile, ninfer::targets::qwen3::TextPhase::Prefill, 1, chunk);
                if (bytes != 0U) (void)runtime_prefix.alloc_bytes(bytes);
            }
            (void)ninfer::targets::qwen3::detail::qwen3_8_27b_r9700::workspace_recipe::
                text_attention_results<TextConfig>(runtime_prefix, chunk);
            {
                auto scratch = runtime_prefix.scope();
                const std::size_t bytes =
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
                    Variant::text_prefill_attention_workspace_capacity_bytes(
                        chunk, options.max_context);
#else
                    Variant::full_attention_workspace_capacity_bytes(
                        chunk, options.max_context, false);
#endif
                if (bytes != 0U) (void)runtime_prefix.alloc_bytes(bytes);
            }
        }
        require(plan.impl_->workspace.text_prefill >= runtime_prefix.peak_bytes(1),
                "Text-prefill plan does not cover its runtime attention allocation prefix");
    }
    const std::uint32_t expected_dflash_width =
        backend == ninfer::SpeculativeBackend::DFlash
            ? ninfer::targets::qwen3::detail::qwen3_8_27b_r9700::dflash_verify_width(
                  drafts, dflash_verify_width)
            : 0U;
    require(plan.impl_->dflash_verify_width == expected_dflash_width,
            "plan did not preserve its fixed DFlash verify width");
    if (use_device_graph && backend == ninfer::SpeculativeBackend::Mtp) {
        constexpr std::size_t kMiB                     = 1024ULL * 1024ULL;
        constexpr std::size_t kMtpGraphFamilyBytes     = 22ULL * kMiB;
        constexpr std::size_t kMtpGraphExecutableBytes = 26ULL * kMiB;
        const auto profiles = Variant::mtp_graph_profiles(max_context, drafts);
        const std::size_t definitions =
            profiles.size() * static_cast<std::size_t>(concurrency);
        std::vector<std::uint32_t> topology_classes;
        for (std::uint32_t batch_size = 1; batch_size <= concurrency; ++batch_size) {
            for (const auto profile : profiles) {
                const std::uint32_t folded =
                    profile.topology_class * concurrency + (batch_size - 1U);
                if (std::find(topology_classes.begin(), topology_classes.end(), folded) ==
                    topology_classes.end()) {
                    topology_classes.push_back(folded);
                }
            }
        }
        require(plan.impl_->graph_definition_count == definitions,
                "MTP graph plan omitted a reachable exact-B/profile definition");
        require(plan.impl_->graph_executable_count == topology_classes.size(),
                "MTP graph plan executable inventory does not match reachable topologies");
        require(plan.impl_->graph_allowance_bytes ==
                    kMtpGraphFamilyBytes +
                        kMtpGraphExecutableBytes * topology_classes.size(),
                "MTP graph allowance does not match its ROCm family/executable inventory");
    }
    if (use_device_graph && backend == ninfer::SpeculativeBackend::None) {
        constexpr std::size_t kMiB                          = 1024ULL * 1024ULL;
        constexpr std::size_t kOrdinaryGraphFamilyBytes     = 20ULL * kMiB;
        constexpr std::size_t kOrdinaryGraphExecutableBytes = 24ULL * kMiB;
        const auto profiles = Variant::ordinary_graph_profiles(max_context);
        const std::size_t definitions =
            profiles.size() * static_cast<std::size_t>(concurrency);
        std::vector<std::uint32_t> topology_classes;
        for (std::uint32_t batch_size = 1; batch_size <= concurrency; ++batch_size) {
            for (const auto profile : profiles) {
                const std::uint32_t folded =
                    profile.topology_class * concurrency + (batch_size - 1U);
                if (std::find(topology_classes.begin(), topology_classes.end(), folded) ==
                    topology_classes.end()) {
                    topology_classes.push_back(folded);
                }
            }
        }
        require(plan.impl_->graph_definition_count == definitions,
                "ordinary graph plan omitted a reachable exact-B/profile definition");
        require(plan.impl_->graph_executable_count == topology_classes.size(),
                "ordinary graph plan executable inventory does not match reachable topologies");
        require(plan.impl_->graph_allowance_bytes ==
                    kOrdinaryGraphFamilyBytes +
                        kOrdinaryGraphExecutableBytes * topology_classes.size(),
                "ordinary graph allowance does not match its ROCm family/executable inventory");
    }
    if (use_device_graph && backend == ninfer::SpeculativeBackend::DFlash) {
        constexpr std::size_t kMiB                   = 1024ULL * 1024ULL;
        constexpr std::size_t kDFlashFamilyBytes     = 40ULL * kMiB;
        constexpr std::size_t kDFlashExecutableBytes = 26ULL * kMiB;
        constexpr std::size_t kDFlashK1FusedBytes    = 10ULL * kMiB;
        constexpr std::size_t kDFlashK1WmmaBytes     = 18ULL * kMiB;
        std::size_t definitions = 0;
        std::vector<std::uint32_t> topology_classes;
        std::size_t expected_allowance = kDFlashFamilyBytes;
        for (std::uint32_t batch_size = 1; batch_size <= concurrency; ++batch_size) {
            const auto profiles = Variant::dflash_graph_profiles(
                max_context, drafts, batch_size, expected_dflash_width);
            definitions += profiles.size();
            for (const auto profile : profiles) {
                const std::uint32_t folded =
                    profile.topology_class * concurrency + (batch_size - 1U);
                if (std::find(topology_classes.begin(), topology_classes.end(), folded) !=
                    topology_classes.end()) {
                    continue;
                }
                topology_classes.push_back(folded);
                if (drafts != 1U) {
                    expected_allowance += kDFlashExecutableBytes;
                } else if (profile.topology_class == 0U) {
                    expected_allowance += kDFlashK1FusedBytes;
                } else if (profile.topology_class == 1U) {
                    expected_allowance += kDFlashK1WmmaBytes;
                } else {
                    throw std::runtime_error("unexpected DFlash K=1 topology class");
                }
            }
        }
        require(plan.impl_->graph_definition_count == definitions,
                "DFlash graph plan omitted a reachable exact-B/profile definition");
        require(plan.impl_->graph_executable_count == topology_classes.size(),
                "DFlash graph plan executable inventory does not match reachable topologies");
        require(plan.impl_->graph_allowance_bytes == expected_allowance,
                "DFlash graph allowance does not match its ROCm topology inventory");
    }
    return plan.workspace_capacity_bytes();
}

void qualify_host_xattention_capacity_envelope() {
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION) && \
    NINFER_R9700_XATTENTION_STRIDE == 16 && NINFER_R9700_XATTENTION_TAU_PERMILLE == 900
    const std::size_t sparse_leaf = Variant::text_prefill_attention_workspace_capacity_bytes(
        static_cast<std::int32_t>(kCapacityEnvelopeChunk),
        static_cast<std::int32_t>(kCapacityEnvelopeContext));
    require(sparse_leaf == kCapacityEnvelopeSparseLeafBytes,
            "S16/tau900 Text-prefill sparse leaf workspace envelope changed");
    std::printf(
        "r9700_runtime_planner: PASS host XAttention S16/tau900 T=%u context=%u "
        "sparse_leaf_bytes=%zu expected_planner_workspace_bytes=%zu\n",
        kCapacityEnvelopeChunk, kCapacityEnvelopeContext, sparse_leaf,
        kCapacityEnvelopeWorkspaceBytes);
#else
    throw std::runtime_error(
        "host XAttention capacity envelope requires the S16/tau900 qualification build");
#endif
}

void print_host_hybrid_capacity_authority() {
    namespace runtime = ninfer::targets::qwen3::detail::qwen3_8_27b_r9700;
    constexpr std::uint32_t capacity = 262144U;
    constexpr std::uint32_t page_tokens = 64U;
    constexpr std::uint32_t minimum_groups = capacity / page_tokens;
    std::printf("target,weights_profile,capacity_tokens,page_tokens,prefill_chunk,kv_value_group,"
                "speculative_backend,draft_tokens,proposal_head,device_graph,"
                "concurrency,minimum_groups,maximum_groups,minimum_sequence_bytes,workspace_bytes,"
                "graph_allowance_bytes,request_transient_bytes,minimum_reservation_bytes,"
                "kv_payload_bytes,kv_increment_bytes\n");
    for (std::uint32_t concurrency = 1; concurrency <= ninfer::kMaximumConcurrency;
         ++concurrency) {
        const runtime::SequencePlanningInputs inputs{
            .weights_profile = WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation,
            .capacity = capacity,
            .max_concurrency = concurrency,
            .prefill_chunk = 8192U,
            .draft_window = 3U,
            .dflash_verify_width = 0U,
            .speculative_backend = ninfer::SpeculativeBackend::Mtp,
            .proposal_head = ninfer::ProposalHead::Optimized,
            .features = {.vision = false,
                         .speculative = ninfer::SpeculativeBackend::Mtp,
                         .proposal_head = ninfer::ProposalHead::Optimized},
            .use_device_graph = true,
            .device = 0,
            .kv_ram_capacity_bytes = 0U,
            .context_checkpoint_marks = {},
        };
        const auto minimum = runtime::build_sequence_candidate_for_qualification(
            inputs, minimum_groups);
        const auto adjacent = runtime::build_sequence_candidate_for_qualification(
            inputs, minimum_groups + 1U);
        const std::size_t increment =
            adjacent->device_reservation_bytes - minimum->device_reservation_bytes;
        std::printf("qwen3_8_27b_r9700,R9700Q4G64Fp8FourRoleN16K16Evaluation,"
                    "262144,64,8192,16,mtp,3,optimized,1,"
                    "%u,%u,%u,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n", concurrency,
                    minimum_groups, concurrency * minimum_groups, minimum->persistent.bytes,
                    minimum->workspace.capacity, minimum->graph_allowance_bytes,
                    minimum->request_transient_capacity_bytes,
                    minimum->device_reservation_bytes, minimum->persistent.kv_payload_bytes,
                    increment);
    }
}

std::uint32_t parse_host_width_argument(const char* text, const char* label) {
    std::uint32_t value = 0;
    const std::string_view input{text};
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
    if (error != std::errc{} || end != input.data() + input.size()) {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    return value;
}

void print_host_hybrid_width_authority(int argc, char** argv) {
    if (argc != 6) {
        throw std::invalid_argument(
            "--host-hybrid-widths-csv requires PREFILL MAX_CONCURRENCY MTP_WIDTH DFLASH_WIDTH");
    }
    const std::uint32_t prefill = parse_host_width_argument(argv[2], "prefill width");
    const std::uint32_t concurrency =
        parse_host_width_argument(argv[3], "maximum concurrency");
    const std::uint32_t mtp_width = parse_host_width_argument(argv[4], "MTP width");
    const std::uint32_t dflash_width = parse_host_width_argument(argv[5], "DFlash width");
    if (concurrency > ninfer::kMaximumConcurrency) {
        throw std::invalid_argument("hybrid width inventory exceeds product concurrency");
    }
    const auto widths = Variant::ExecutionState::eager_widths(
        prefill, concurrency, mtp_width, dflash_width);
    std::printf("prefill,max_concurrency,mtp_width,dflash_width,prepared_width\n");
    for (const std::uint32_t width : widths) {
        std::printf("%u,%u,%u,%u,%u\n", prefill, concurrency, mtp_width, dflash_width,
                    width);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--host-xattention-capacity-envelope") {
            qualify_host_xattention_capacity_envelope();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-request-lane-cap") {
            qualify_host_request_lane_cap();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-split512-routing") {
            qualify_host_split512_routing();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-hybrid-capacity-csv") {
            print_host_hybrid_capacity_authority();
            return 0;
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--host-hybrid-widths-csv") {
            print_host_hybrid_width_authority(argc, argv);
            return 0;
        }
        if (argc != 1) {
            throw std::invalid_argument(
                "usage: ninfer_r9700_runtime_planner_qual "
                "[--host-xattention-capacity-envelope|--host-request-lane-cap|"
                "--host-split512-routing|--host-hybrid-capacity-csv|"
                "--host-hybrid-widths-csv PREFILL MAX_CONCURRENCY MTP_WIDTH DFLASH_WIDTH]");
        }
        ninfer::DeviceContext device(0);
        for (std::uint32_t concurrency = 1; concurrency <= ninfer::kMaximumConcurrency;
             ++concurrency) {
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::None, 0, false);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::None, 0, true);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::Mtp, 3, false);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::Mtp, 3, true);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::DFlash, 4, false);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::DFlash, 4, false, 6);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::None, 0, false, 0,
                         WeightsProfile::R9700Q4G64Evaluation, true, 32768);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::Mtp, 3, false, 0,
                         WeightsProfile::R9700Q4G64Evaluation, true, 32774);
            if (concurrency == 1U) {
                qualify_plan(device, concurrency, ninfer::SpeculativeBackend::DFlash, 1, false, 2,
                             WeightsProfile::R9700Q4G64DFlash2Q4Evaluation, true, 128);
            }
            for (std::uint32_t k = 1; k <= 11; ++k) {
                qualify_plan(device, concurrency, ninfer::SpeculativeBackend::DFlash, k, false, 0,
                             WeightsProfile::R9700Q4G64DFlash2Q4Evaluation, true,
                             32768U + k + 1U);
            }
        }
        const std::size_t w8_workspace = qualify_plan(
            device, ninfer::kMaximumConcurrency, ninfer::SpeculativeBackend::None, 0, true);
        const std::size_t mixed_workspace = qualify_plan(
            device, ninfer::kMaximumConcurrency, ninfer::SpeculativeBackend::None, 0, true, 0,
            WeightsProfile::R9700Q4W8Evaluation);
        const std::size_t q4_workspace = qualify_plan(
            device, ninfer::kMaximumConcurrency, ninfer::SpeculativeBackend::None, 0, true, 0,
            WeightsProfile::R9700Q4G64Evaluation);
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION) && \
    NINFER_R9700_XATTENTION_STRIDE == 16 && NINFER_R9700_XATTENTION_TAU_PERMILLE == 900
        const std::size_t capacity_envelope_workspace = qualify_plan(
            device, 1, ninfer::SpeculativeBackend::Mtp, 3, false, 0,
            WeightsProfile::R9700Q4G64Evaluation, true, kCapacityEnvelopeContext,
            kCapacityEnvelopeChunk);
        require(capacity_envelope_workspace == kCapacityEnvelopeWorkspaceBytes,
                "S16/tau900 4K/262K global planner workspace envelope changed");
#endif
        (void)qualify_plan(device, ninfer::kMaximumConcurrency,
                           ninfer::SpeculativeBackend::DFlash, 4, false, 6,
                           WeightsProfile::R9700Q4G64DFlash2Q4Evaluation);
        (void)qualify_plan(device, ninfer::kMaximumConcurrency,
                           ninfer::SpeculativeBackend::DFlash, 4, false, 6,
                           WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation);
        const std::size_t dflash_q4_linear = Variant::linear_workspace_capacity_bytes(
            WeightsProfile::R9700Q4G64DFlash2Q4Evaluation, 128);
        const std::size_t dflash_mixed_linear = Variant::linear_workspace_capacity_bytes(
            WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation, 128);
        const std::size_t base_q4_linear = Variant::linear_workspace_capacity_bytes(
            WeightsProfile::R9700Q4G64Evaluation, 128);
        require(Variant::dflash_matrix_qtype(
                    WeightsProfile::R9700Q4G64DFlash2Q4Evaluation) ==
                    ninfer::QType::Q4G64_F16S &&
                    Variant::dflash_matrix_qtype(
                        WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation) ==
                        ninfer::QType::Q4G64_F16S,
                "DFlash evaluation profile did not select persistent Q4 child matrices");
        require(dflash_q4_linear == dflash_mixed_linear,
                "DFlash Q4 companions disagree on their activation workspace");
        require(dflash_q4_linear > base_q4_linear,
                "DFlash feature projection did not enlarge Q4 activation workspace to K=25600");
        if constexpr (ninfer::ops::r9700::linear::kW8ActivationBits == 8) {
            require(w8_workspace == mixed_workspace,
                    "A8 W8 and mixed plans disagree on their largest activation image");
            require(w8_workspace > q4_workspace,
                    "A8G32 W8 plan did not reserve its larger scale image");
        } else {
            require(mixed_workspace > w8_workspace,
                    "mixed Q4/W8 plan did not reserve caller-owned Q4 activation workspace");
            require(q4_workspace > mixed_workspace,
                    "all-Q4 plan did not reserve its larger MLP/Vision activation workspace");
        }
        std::printf(
            "r9700_runtime_planner: PASS q4_activation_bits=%d q4_prefill_cta_profile=%.*s "
            "w8_activation_bits=%d C=1..4 "
            "ordinary/vision/MTP/DFlash fixed widths; 32K MTP3 graph allowance; "
            "exact 32K DFlash K=1..11 C=1..4 graph inventories; profile-consistent integer "
            "workspace including DFlash Q4 companions\n",
            ninfer::ops::r9700::linear::kQ4ActivationBits,
            static_cast<int>(ninfer::ops::r9700::linear::kQ4PrefillCtaProfile.size()),
            ninfer::ops::r9700::linear::kQ4PrefillCtaProfile.data(),
            ninfer::ops::r9700::linear::kW8ActivationBits);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "r9700_runtime_planner: FAIL: %s\n", error.what());
        return 1;
    }
}
