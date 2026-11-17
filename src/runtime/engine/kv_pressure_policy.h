#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

// Logical identity survives physical-lane release. suspend_ready means that the target has
// secured an allocatable image credit and the settled lane can be captured without rollback.
struct KvPressureRequest {
    std::uint64_t id               = 0;
    std::uint64_t queue_order      = 0;
    bool resident                  = false;
    bool growth_blocked            = false;
    bool completion_member         = false;
    bool suspend_ready             = false;
    bool useful_chunk_since_resume = true;
};

// Worker-local RAM-on policy. Resource geometry, image reservations, transfer ownership and
// completion-cohort claims belong to the executor/target transaction.
class KvPressurePolicy {
public:
    KvPressurePolicy(std::uint32_t concurrency, std::size_t max_outstanding)
        : concurrency_(concurrency), max_outstanding_(max_outstanding) {
        if (concurrency == 0 || concurrency > kMaximumConcurrency ||
            max_outstanding < concurrency) {
            throw std::invalid_argument("invalid KV pressure request bounds");
        }
    }

    // Call only on settled boundary snapshots, including suspended and transfer-held admitted
    // identities. executable_work includes a legal prefill unit.
    void observe(std::span<const KvPressureRequest> admitted, bool executable_work) {
        if (admitted.size() > max_outstanding_) {
            throw std::logic_error("KV pressure logical admission bound exceeded");
        }
        const auto residents = std::count_if(admitted.begin(), admitted.end(),
                                             [](const auto& r) { return r.resident; });
        const bool blocked   = std::any_of(admitted.begin(), admitted.end(), [](const auto& r) {
            return r.resident && r.growth_blocked;
        });
        no_executable_work_  = !executable_work;
        // A lone resident grows into the whole pool, so suspending it frees nothing for anyone.
        pressure_needed_ = (blocked && residents >= 2) || (!executable_work && !admitted.empty());
    }

    // Suspended requests hold older queue order, so they resume before any fresh admission.
    // A fresh request admitted beside a growth-blocked resident would take the pages that
    // resident needs and become its first victim, so ingress also waits for that growth.
    [[nodiscard]] bool allows_ingress(std::span<const KvPressureRequest> admitted) const noexcept {
        return admitted.size() < max_outstanding_ &&
               std::none_of(
                   admitted.begin(), admitted.end(),
                   [](const auto& r) { return !r.resident || r.growth_blocked; });
    }

    [[nodiscard]] bool pressure_needed() const noexcept { return pressure_needed_; }

    [[nodiscard]] bool no_executable_work() const noexcept { return no_executable_work_; }

    [[nodiscard]] static std::optional<std::uint64_t>
    oldest_suspended(std::span<const KvPressureRequest> admitted) noexcept {
        const KvPressureRequest* oldest = nullptr;
        for (const auto& request : admitted) {
            if (!request.resident && (!oldest || older(request, *oldest))) { oldest = &request; }
        }
        return oldest ? std::optional(oldest->id) : std::nullopt;
    }

    // Indices into resident of the victims that may make room for resident[recipient], latest
    // admitted first. An ordinary recipient never displaces an older resident: with no newer
    // eligible victim it waits for peers to complete or block. A completion-cohort recipient
    // may displace any eligible resident. The caller suspends the shortest latest-first prefix
    // whose release proves the exact post-transition fit, one capture at a time.
    [[nodiscard]] std::vector<std::size_t>
    ranked_victims(std::span<const KvPressureRequest> resident, std::size_t recipient) const {
        if (resident.size() > concurrency_ || recipient >= resident.size()) {
            throw std::logic_error("KV pressure resident bound exceeded");
        }
        std::vector<std::size_t> result;
        if (!pressure_needed_) { return result; }
        const auto& owner = resident[recipient];
        for (std::size_t i = 0; i < resident.size(); ++i) {
            const auto& r = resident[i];
            if (i != recipient && r.resident && !r.completion_member && r.suspend_ready &&
                (r.useful_chunk_since_resume || no_executable_work_) &&
                (owner.completion_member || older(owner, r))) {
                result.push_back(i);
            }
        }
        std::sort(result.begin(), result.end(),
                  [&](std::size_t a, std::size_t b) { return older(resident[b], resident[a]); });
        return result;
    }

private:
    [[nodiscard]] static bool older(const KvPressureRequest& a,
                                    const KvPressureRequest& b) noexcept {
        return a.queue_order < b.queue_order || (a.queue_order == b.queue_order && a.id < b.id);
    }

    std::uint32_t concurrency_;
    std::size_t max_outstanding_;
    bool pressure_needed_    = false;
    bool no_executable_work_ = false;
};

} // namespace ninfer::runtime
