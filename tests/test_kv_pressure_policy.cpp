#include "runtime/engine/kv_pressure_policy.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::runtime::KvPressureRequest request(std::uint64_t id, bool blocked = false) {
    return {.id             = id,
            .queue_order    = id,
            .resident       = true,
            .growth_blocked = blocked,
            .suspend_ready  = true};
}

} // namespace

int main() {
    using ninfer::runtime::KvPressurePolicy;
    using ninfer::runtime::KvPressureReason;
    using ninfer::runtime::KvPressureRequest;
    int failures = 0;

    // One blocked lane beside other residents opens pressure; victims are single residents,
    // latest admitted first, and never a completion member.
    KvPressurePolicy policy(4, 12);
    std::vector<KvPressureRequest> active{request(1, true), request(2), request(3), request(4)};
    active[2].completion_member = true;
    policy.observe(active, true);
    failures +=
        check(policy.pressure_needed(), "a blocked lane beside residents opened no pressure");
    auto victims = policy.ranked_victims(active, KvPressureReason::Growth);
    failures += check(victims == std::vector<std::size_t>{3, 1, 0},
                      "growth victims were not the eligible residents, latest first");

    // A queued replacement may only take a blocked lane.
    victims = policy.ranked_victims(active, KvPressureReason::WaitingReplacement);
    failures += check(victims == std::vector<std::size_t>{0},
                      "queued replacement evicted an unblocked or completion member");

    // No blocked lane: growth needs no victim, and fresh ingress stays open.
    active[0].growth_blocked = false;
    policy.observe(active, true);
    failures += check(!policy.pressure_needed() &&
                          policy.ranked_victims(active, KvPressureReason::Growth).empty() &&
                          policy.allows_ingress(active.size(), false),
                      "an unblocked batch suspended work or stopped ingress");

    // A lone resident grows into the whole pool instead of being suspended.
    KvPressurePolicy alone(1, 2);
    std::array single{request(1, true)};
    alone.observe(single, true);
    failures += check(!alone.pressure_needed() &&
                          alone.ranked_victims(single, KvPressureReason::Growth).empty(),
                      "a lone resident was suspended for its own growth");

    // A recent resume is protected until it advances a useful chunk, unless nothing can run.
    std::array rows{request(1, true), request(2), request(3)};
    rows[2].useful_chunk_since_resume = false;
    KvPressurePolicy pressure(4, 12);
    pressure.observe(rows, true);
    failures += check(pressure.ranked_victims(rows, KvPressureReason::Growth) ==
                          std::vector<std::size_t>{1, 0},
                      "recent resume was suspended before useful progress");
    pressure.observe(rows, false);
    failures += check(pressure.ranked_victims(rows, KvPressureReason::Growth).front() == 2,
                      "no-runnable emergency could not evacuate a recent resume");
    rows[1].suspend_ready = false;
    victims               = pressure.ranked_victims(rows, KvPressureReason::Growth);
    failures += check(std::find(victims.begin(), victims.end(), 1) == victims.end(),
                      "unfunded or unsettled request became a suspension victim");

    // Suspended identities keep their arrival order, resume before fresh ingress, and count
    // against the logical admission bound although they hold no lane.
    std::array waiting{request(10), request(2), request(6)};
    waiting[0].resident = false;
    waiting[1].resident = false;
    failures += check(KvPressurePolicy::oldest_suspended(waiting) == 2,
                      "resume order lost original request age");
    KvPressurePolicy bounded(2, 5);
    bounded.observe(waiting, true);
    failures += check(!bounded.allows_ingress(3, true) && !bounded.allows_ingress(5, false) &&
                          bounded.allows_ingress(3, false),
                      "ingress ignored suspended work or the logical admission bound");

    // Every resident of a full batch can be a victim, the latest first.
    KvPressurePolicy full(ninfer::kMaximumConcurrency, ninfer::kMaximumConcurrency);
    std::vector<KvPressureRequest> residents;
    for (std::uint64_t id = 1; id <= ninfer::kMaximumConcurrency; ++id) {
        residents.push_back(request(id, true));
    }
    full.observe(residents, true);
    victims = full.ranked_victims(residents, KvPressureReason::Growth);
    failures +=
        check(victims.size() == ninfer::kMaximumConcurrency &&
                  victims.front() == ninfer::kMaximumConcurrency - 1U && victims.back() == 0,
              "full-batch ranking omitted a resident or was not latest first");

    return failures == 0 ? 0 : 1;
}
