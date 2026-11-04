// Real production Program: compact graph batches, lane turnover, and failure before publication.
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_27b/impl/load/bindings.h"
#include "core/device.h"
#include "targets/qwen3_8_27b/impl/variant.h"

#include <ninfer/targets/qwen3/frontend.h>
#include <ninfer/targets/qwen3/prepared_prompt.h>
#include <ninfer/targets/qwen3/startup_features.h>
#include <ninfer/targets/qwen3_8_27b/package.h>

#define NINFER_QWEN3_VARIANT    ::ninfer::targets::qwen3_8_27b::detail::Variant
#define NINFER_QWEN3_RUNTIME_NS qwen3_8_27b_r9700
#include "targets/qwen3/impl/runtime/program.h"
#undef NINFER_QWEN3_RUNTIME_NS
#undef NINFER_QWEN3_VARIANT

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace target                 = ninfer::targets::qwen3_8_27b::detail;
namespace family                 = ninfer::targets::qwen3;
namespace execution              = family::detail::qwen3_8_27b_r9700;
using Package                    = ninfer::targets::qwen3_8_27b::Package;
constexpr std::uint32_t kLanes   = 6;
constexpr std::uint32_t kBadLane = 2;
constexpr std::array<std::uint32_t, kLanes> kAll{0, 1, 2, 3, 4, 5};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// The shared head row is eligible in exactly one request. This tests logical lane isolation
// without adding a production injection hook or modifying the artifact on disk.
class HeadFault {
public:
    explicit HeadFault(const ninfer::Weight& head) {
        require(head.qtype == ninfer::QType::LQ4 && head.scale_dtype == ninfer::DType::FP32 &&
                    head.scales != nullptr && head.scale_bytes >= 8U * sizeof(std::uint32_t),
                "fault requires the production LQ4 output head");
        address_ = const_cast<std::uint32_t*>(static_cast<const std::uint32_t*>(head.scales)) + 7;
        HIP_CHECK(hipMemcpy(&original_, address_, sizeof(original_), hipMemcpyDeviceToHost));
    }

    HeadFault(const HeadFault&)            = delete;
    HeadFault& operator=(const HeadFault&) = delete;

    ~HeadFault() {
        if (dirty_) (void)hipMemcpy(address_, &original_, sizeof(original_), hipMemcpyHostToDevice);
    }

    void inject() {
        constexpr std::uint32_t poison = 0x7fc00000U;
        dirty_                         = true;
        HIP_CHECK(hipMemcpy(address_, &poison, sizeof(poison), hipMemcpyHostToDevice));
    }

    void restore() {
        HIP_CHECK(hipMemcpy(address_, &original_, sizeof(original_), hipMemcpyHostToDevice));
        dirty_ = false;
    }
private:
    std::uint32_t* address_ = nullptr;
    std::uint32_t original_ = 0;
    bool dirty_             = false;
};

ninfer::runtime::ResolvedExecutionOptions request(std::uint32_t lane) {
    ninfer::runtime::ResolvedExecutionOptions result;
    result.requested_output_tokens = 128;
    result.allow_prefix_reuse      = false;
    result.cache_write             = false;
    result.sampling.temperature    = 1.5F;
    result.sampling.p_less         = true;
    result.sampling.top_p          = 1.0F;
    result.sampling.seed           = 0x42cafeULL + lane;
    if (lane != kBadLane) {
        result.suppressed_token_count  = 1;
        result.suppressed_token_ids[0] = 7;
    }
    return result;
}

auto admission_prompt(family::Frontend& frontend, std::uint32_t lane) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    ninfer::ChatMessage message;
    message.parts.push_back(ninfer::MessagePart{
        .text = "Continue this sentence with a short account of experiment number " +
                std::to_string(lane) + ": The researcher opened the notebook and"});
    input.messages.push_back(std::move(message));
    return family::PreparedPromptAccess::take(frontend.prepare(input));
}

ninfer::TokenId admit(execution::ProgramImplCore& program, family::Frontend& frontend,
                      std::uint32_t lane) {
    auto prompt = admission_prompt(frontend, lane);
    auto base   = program.plan_request_base(prompt, request(lane));
    auto plan   = program.plan_request_for_lane(lane, prompt, base);
    auto step   = program.start_prefill_lane(lane, std::move(prompt), std::move(plan), {});
    while (!step.complete) step = program.advance_prefill_lane(lane);
    require(step.round.tokens.size() == 1U, "prefill did not return one token");
    const auto first = step.round.tokens[0];
    program.resolve_prefill_lane(lane, false);
    require(program.requests[lane].sampling_host.suppressed_token_count ==
                (lane == kBadLane ? 0 : 1),
            "lane suppression was not installed");
    return first;
}

// Exercise the exact cancellation window before the first mixed target forward. A
// queued host gate makes pending compute deterministic without an injection hook.
void deferred_prefill_cancel(execution::ProgramImplCore& program, family::Frontend& frontend,
                             ninfer::DeviceContext& device, std::uint32_t lane,
                             ninfer::TokenId cold_first) {
    require(program.mixed_forward != 0, "deferred cancellation requires mixed forwarding");
    auto prompt = admission_prompt(frontend, lane);
    auto base   = program.plan_request_base(prompt, request(lane));
    auto plan   = program.plan_request_for_lane(lane, prompt, base);

    struct Gate {
        hipStream_t stream;
        std::atomic<bool> entered{false};
        std::atomic<bool> released{false};
        std::mutex mutex;
        std::condition_variable cancelled;
        bool cancellation_done = false;

        ~Gate() {
            released.store(true, std::memory_order_release);
            released.notify_all();
            // The callback holds this object's address, including during exception teardown.
            (void)hipStreamSynchronize(stream);
        }
    } gate{device.stream};

    HIP_CHECK(hipLaunchHostFunc(
        device.stream,
        [](void* opaque) {
            auto& pending = *static_cast<Gate*>(opaque);
            pending.entered.store(true, std::memory_order_release);
            pending.entered.notify_all();
            pending.released.wait(false, std::memory_order_acquire);
        },
        &gate));
    std::jthread release([&] {
        gate.entered.wait(false, std::memory_order_acquire);
        std::unique_lock lock(gate.mutex);
        // Bound an inconclusive synchronous runtime without hanging exception teardown.
        (void)gate.cancelled.wait_for(lock, std::chrono::seconds(2),
                                      [&] { return gate.cancellation_done; });
        gate.released.store(true, std::memory_order_release);
        gate.released.notify_all();
    });
    const auto first =
        program.start_prefill_lane(lane, std::move(prompt), std::move(plan), {}, nullptr, true);
    const auto& prefill = program.requests[lane].prefill;
    require(prefill.has_value(), "deferred prefill state missing");
    require(!first.complete && first.round.tokens.empty() &&
                prefill.value().cursor == prefill.value().base,
            "cancellation did not precede the first deferred mixed forward");
    require(!gate.released.load(std::memory_order_acquire),
            "deferred cancellation phase was not reached before pending compute settled");
    require(!program.revert_cancelled_prefill_lane(lane),
            "cold cancelled prefill unexpectedly retained a prefix");
    program.abort_lane(lane);
    require(!gate.released.load(std::memory_order_acquire),
            "cancellation waited for the fallback release instead of completing with pending work");
    require(!program.sequences[lane].kv &&
                program.requests[lane].lifecycle == execution::Lifecycle::Empty,
            "deferred cancellation retained its lane or KV pages");
    {
        std::lock_guard lock(gate.mutex);
        gate.cancellation_done = true;
    }
    gate.cancelled.notify_all();
    require(admit(program, frontend, lane) == cold_first,
            "deferred cancellation/reuse changed the seeded cold first token");
    program.abort_lane(lane);
}

void require_empty(const execution::ProgramImplCore& program) {
    for (const auto lane : kAll) {
        require(program.requests[lane].lifecycle == execution::Lifecycle::Empty &&
                    !program.sequences[lane].kv && program.sequences[lane].ledger.empty() &&
                    program.sequences[lane].execution_frontier == 0,
                "failed batch retained request or model state on lane " + std::to_string(lane));
    }
}

using TokensByLane = std::array<std::vector<ninfer::TokenId>, kLanes>;

TokensByLane decode(execution::ProgramImplCore& program, std::span<const std::uint32_t> lanes,
                    bool heterogeneous) {
    std::vector<ninfer::runtime::RoundBudget> budgets(lanes.size());
    for (std::size_t row = 0; row < lanes.size(); ++row)
        budgets[row].generated_tokens_remaining =
            heterogeneous ? 2U + static_cast<unsigned>(row) : 16U;
    std::uint32_t frontier = 0;
    for (const auto lane : lanes)
        frontier = std::max(frontier, program.sequences[lane].execution_frontier);
    const auto round = program.decode_batch(lanes, budgets);
    const auto k     = program.adaptive_batch_k_by_c[lanes.size() - 1U].live_k;
    require(k >= 3U && k <= 7U, "adaptive round selected an unexpected K");
    require(std::any_of(
                program.dflash_graphs.profiles.begin(), program.dflash_graphs.profiles.end(),
                [&](const execution::DecodeGraphProfile& profile) {
                    return profile.batch_size == lanes.size() && profile.draft_tokens == k &&
                           profile.min_execution_frontier <= frontier &&
                           frontier <= profile.max_execution_frontier && profile.executable.ready();
                }),
            "live batch has no matching ready Device Graph");
    TokensByLane result;
    std::vector<std::uint32_t> accepted(lanes.size());
    std::vector<std::uint8_t> flags(lanes.size(), 0U);
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        require(program.dflash_host_ingress->lanes[row] == static_cast<int>(lanes[row]),
                "compact ingress lost the physical lane mapping");
        const auto count = round.row_counts[row];
        require(count > 0 &&
                    static_cast<std::uint32_t>(count) <= budgets[row].generated_tokens_remaining,
                "decode violated its heterogeneous row budget");
        const auto begin = round.tokens.begin() + row * round.row_stride;
        result[lanes[row]].assign(begin, begin + count);
        accepted[row] = static_cast<std::uint32_t>(count);
    }
    program.resolve_pending_batch(lanes, accepted, flags, flags);
    std::cout << "graph round B=" << lanes.size() << " actual_K=" << k << " lanes=";
    for (const auto lane : lanes) std::cout << lane << ',';
    std::cout << '\n';
    return result;
}

void exercise(const std::filesystem::path& artifact) {
    ninfer::DeviceContext device(0);
    ninfer::EngineOptions options;
    options.artifact_path              = artifact;
    options.max_context                = 262144;
    options.max_concurrency            = kLanes;
    options.kv_capacity                = ninfer::KvCapacityPolicy::automatic();
    options.prefill_chunk              = 2048;
    options.kv_ram_capacity_bytes      = 0;
    options.enable_vision              = false;
    options.use_device_graph           = true;
    options.speculative.backend        = ninfer::SpeculativeBackend::DFlash;
    options.speculative.proposal_head  = ninfer::ProposalHead::Optimized;
    options.speculative.draft_tokens   = 7;
    options.speculative.adaptive_draft = true;
    ninfer::artifact::Reader reader(artifact);
    ninfer::artifact::Binder binder(reader);
    options.model_id                 = reader.identity().model_id;
    options.weights_id               = reader.identity().weights_id;
    options.artifact_tensor_identity = reader.tensor_content_identity(
        ninfer::targets::qwen3_8_27b::detail::cache_tensor_selections(
            reader, ninfer::targets::qwen3::startup_features(options)));
    const auto profile = Package::resolve_weights(reader.identity());
    auto load          = target::bind_artifact(binder, profile, family::startup_features(options));
    load.bindings.linear_widths = target::Variant::ExecutionState::eager_widths(
        options.prefill_chunk, kLanes,
        family::startup_verify_widths<target::DFlashConfig>(options));
    auto materialized = ninfer::artifact::materialize(reader, load.materialization, device);
    target::LoadedModelData model(std::move(load.bindings), std::move(materialized));
    auto frontend    = family::make_frontend(model.frontend, false);
    auto planner     = Package::make_sequence_planner(device, options, profile);
    const auto pages = planner.capacity_curve().minimum_main_page_groups;
    auto plan        = std::move(planner).finalize(pages);
    execution::ProgramImplCore program(model.runtime, *plan.impl_, device, nullptr);
    require(program.use_device_graph && program.adaptive_draft && program.draft_window == 7U &&
                program.max_concurrency == kLanes &&
                program.proposal_head == ninfer::ProposalHead::Optimized &&
                program.prefill_chunk == 2048U && program.capacity == 262144U,
            "qualification changed the requested production geometry");
    for (const auto batch : {3U, 6U})
        require(std::any_of(program.dflash_graphs.profiles.begin(),
                            program.dflash_graphs.profiles.end(),
                            [&](const execution::DecodeGraphProfile& p) {
                                return p.batch_size == batch && p.draft_tokens == 7U &&
                                       p.executable.ready();
                            }),
                "startup did not prepare B3/B6 K7 graphs");
    std::cout << "context=262144 C6 adaptive_max_K7 prefill2048 graph=true physical_page_groups="
              << pages << " (minimum resident pool; full auto-capacity sizing is not under test)\n";

    // Save only the learned host policy inputs. Reusing them below compares the same sampler
    // route/profile and seed; a different adaptive K need not map an RNG draw identically.
    const auto initial_times       = program.adaptive_t_by_batch;
    const auto initial_batch_k     = program.adaptive_batch_k_by_c;
    const auto initial_calibration = program.p_less_calibration;
    std::array<ninfer::TokenId, kLanes> first{};
    for (const auto lane : kAll) first[lane] = admit(program, frontend, lane);
    const auto control = decode(program, kAll, false);
    (void)decode(program, kAll, true);
    constexpr std::array<std::uint32_t, 3> retired{0, 2, 4}, compact{5, 1, 3};
    for (const auto lane : retired) program.abort_lane(lane);
    (void)decode(program, compact, true);
    for (const auto lane : retired) (void)admit(program, frontend, lane);
    constexpr std::array<std::uint32_t, kLanes> repacked{4, 1, 0, 5, 2, 3};
    (void)decode(program, repacked, true);

    // Exercise the captured K7 route without pretending the natural adaptive policy must pick
    // it. Synthetic qualification cost observations make K7 the selected arm for this one round.
    auto& cost = program.adaptive_t_by_batch[kLanes - 1U];
    cost       = {};
    for (const auto k : program.captured_ks)
        family::adaptive_observe_round_time(cost, k, k == 7U ? 0.0001F : 1.0F, 0);
    (void)decode(program, repacked, false);
    require(program.adaptive_batch_k_by_c[kLanes - 1U].live_k == 7U,
            "synthetic qualification costs did not execute the K7 graph");
    std::cout
        << "K7 route executed using synthetic qualification costs (not a performance result)\n";

    HeadFault fault(model.runtime.output_head);
    fault.inject();
    bool rejected = false;
    try {
        const std::array<ninfer::runtime::RoundBudget, kLanes> budgets{
            {{16}, {16}, {16}, {16}, {16}, {16}}};
        (void)program.decode_batch(repacked, budgets);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("nonfinite eligible logits") != std::string::npos;
        if (!rejected) throw;
    }
    fault.restore();
    require(rejected, "nonfinite graph batch returned a generated round");
    require_empty(program);
    const auto* host =
        static_cast<const execution::ProgramImplCore::PrefillRoundHost*>(program.round_host.data());
    for (const auto lane : kAll) {
        const auto begin = host->nonfinite_logits.begin() +
                           static_cast<std::size_t>(lane) * family::kDFlashDecodeMaximumWidth;
        const bool bad   = std::any_of(begin, begin + family::kDFlashDecodeMaximumWidth,
                                       [](std::uint32_t value) { return value != 0U; });
        require(bad == (lane == kBadLane), "nonfinite status crossed a logical lane boundary");
    }

    program.adaptive_t_by_batch   = initial_times;
    program.adaptive_batch_k_by_c = initial_batch_k;
    program.p_less_calibration    = initial_calibration;
    for (const auto lane : kAll)
        require(admit(program, frontend, lane) == first[lane],
                "clean readmission changed prefill token");
    require(decode(program, kAll, false) == control,
            "clean readmission changed the same-profile seeded cold control");
    for (const auto lane : kAll) program.abort_lane(lane);
    require_empty(program);
    const std::array<std::uint32_t, 1> peer{1U};
    const auto peer_times       = program.adaptive_t_by_batch;
    const auto peer_batch_k     = program.adaptive_batch_k_by_c;
    const auto peer_calibration = program.p_less_calibration;
    (void)admit(program, frontend, 1U);
    const auto peer_decode_times       = program.adaptive_t_by_batch;
    const auto peer_decode_k           = program.adaptive_batch_k_by_c;
    const auto peer_decode_calibration = program.p_less_calibration;
    const auto peer_control            = decode(program, peer, false);
    program.abort_lane(1U);
    program.adaptive_t_by_batch   = peer_times;
    program.adaptive_batch_k_by_c = peer_batch_k;
    program.p_less_calibration    = peer_calibration;
    (void)admit(program, frontend, 1U);
    deferred_prefill_cancel(program, frontend, device, 0U, first[0]);
    deferred_prefill_cancel(program, frontend, device, kBadLane, first[kBadLane]);
    program.adaptive_t_by_batch   = peer_decode_times;
    program.adaptive_batch_k_by_c = peer_decode_k;
    program.p_less_calibration    = peer_decode_calibration;
    require(decode(program, peer, false) == peer_control,
            "deferred cancellation or reuse changed the resident peer's same-profile continuation");
    program.abort_lane(1U);
    require_empty(program);
    std::cout << "PASS: deferred mixed-prefill cancellation before its first forward, "
                 "ordered reuse, exact cold readmission with a live peer\n";
    std::cout << "PASS: six-lane graph repack, isolated nonfinite detection, whole-batch "
                 "rejection, clean readmission\n";
}
} // namespace

int main(int argc, char** argv) {
    const char* artifact = argc == 2 ? argv[1] : std::getenv("NINFER_R9700_WEIGHTS");
    if (argc > 2) {
        std::cerr << "usage: nonfinite_batch_qual [explicit-production-artifact.ninfer]\n";
        return 2;
    }
    if (!artifact || !*artifact) {
        std::cout << "skip: set NINFER_R9700_WEIGHTS or pass an explicit production artifact\n";
        return 77;
    }
    try {
        exercise(artifact);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
