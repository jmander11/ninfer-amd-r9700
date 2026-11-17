#include "runtime/engine/kv_pressure_policy.h"

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
    using ninfer::runtime::KvPressureRequest;
    int failures = 0;

    // One blocked lane beside other residents opens pressure. An ordinary recipient may only
    // displace newer residents, latest first, never a completion member.
    KvPressurePolicy policy(4, 12);
    std::vector<KvPressureRequest> active{request(1), request(2, true), request(3), request(4)};
    active[2].completion_member = true;
    policy.observe(active, true);
    failures +=
        check(policy.pressure_needed(), "a blocked lane beside residents opened no pressure");
    failures += check(policy.ranked_victims(active, 1) == std::vector<std::size_t>{3},
                      "victims were not the newer non-member residents, latest first");

    // The newest blocked resident has no newer victim and waits for its older peers.
    active[1].growth_blocked = false;
    active[3].growth_blocked = true;
    policy.observe(active, true);
    failures += check(policy.pressure_needed() && policy.ranked_victims(active, 3).empty(),
                      "the newest blocked resident displaced an older running peer");

    // A completion member may displace any eligible resident, latest first.
    active[3].growth_blocked = false;
    active[2].growth_blocked = true;
    policy.observe(active, true);
    failures += check(policy.ranked_victims(active, 2) == std::vector<std::size_t>{3, 1, 0},
                      "a completion member could not evacuate its non-member peers");

    // No blocked lane: growth needs no victim, and fresh ingress stays open.
    active[2].growth_blocked = false;
    policy.observe(active, true);
    failures += check(!policy.pressure_needed() && policy.ranked_victims(active, 0).empty() &&
                          policy.allows_ingress(active),
                      "an unblocked batch suspended work or stopped ingress");

    // A lone resident grows into the whole pool instead of being suspended.
    KvPressurePolicy alone(1, 2);
    std::array single{request(1, true)};
    alone.observe(single, true);
    failures += check(!alone.pressure_needed() && alone.ranked_victims(single, 0).empty(),
                      "a lone resident was suspended for its own growth");

    // A recent resume is protected until it advances a useful chunk, unless nothing can run.
    std::array rows{request(1, true), request(2), request(3)};
    rows[2].useful_chunk_since_resume = false;
    KvPressurePolicy pressure(4, 12);
    pressure.observe(rows, true);
    failures += check(pressure.ranked_victims(rows, 0) == std::vector<std::size_t>{1},
                      "recent resume was suspended before useful progress");
    pressure.observe(rows, false);
    failures += check(pressure.no_executable_work() &&
                          pressure.ranked_victims(rows, 0) == std::vector<std::size_t>{2, 1},
                      "no-runnable emergency could not evacuate a recent resume");
    rows[1].suspend_ready = false;
    failures += check(pressure.ranked_victims(rows, 0) == std::vector<std::size_t>{2},
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
    auto settled = waiting;
    for (auto& r : settled) r.resident = true;
    const std::array fuller{request(1), request(2), request(3), request(4), request(5)};
    auto blocked              = settled;
    blocked[2].growth_blocked = true;
    failures += check(!bounded.allows_ingress(waiting) && !bounded.allows_ingress(fuller) &&
                          !bounded.allows_ingress(blocked) && bounded.allows_ingress(settled),
                      "ingress ignored suspended or blocked work or the logical admission bound");

    // Every resident newer than the oldest of a full batch can be a victim, the latest first.
    KvPressurePolicy full(ninfer::kMaximumConcurrency, ninfer::kMaximumConcurrency);
    std::vector<KvPressureRequest> residents;
    for (std::uint64_t id = 1; id <= ninfer::kMaximumConcurrency; ++id) {
        residents.push_back(request(id, true));
    }
    full.observe(residents, true);
    const auto victims = full.ranked_victims(residents, 0);
    failures +=
        check(victims.size() == ninfer::kMaximumConcurrency - 1U &&
                  victims.front() == ninfer::kMaximumConcurrency - 1U && victims.back() == 1,
              "full-batch ranking omitted a resident or was not latest first");

    return failures == 0 ? 0 : 1;
}
