#pragma once

#include <ninfer/types.h>
#include <nlohmann/json.hpp>

namespace ninfer::serve {

inline const char* constraint_branch_name(ninfer::ConstraintBranch branch) {
    switch (branch) {
    case ninfer::ConstraintBranch::None:
        return "none";
    case ninfer::ConstraintBranch::Reasoning:
        return "reasoning";
    case ninfer::ConstraintBranch::Answer:
        return "answer";
    case ninfer::ConstraintBranch::ToolCall:
        return "tool_call";
    }
    return "none";
}

inline const char* constraint_compile_cache_name(ninfer::ConstraintCompileCache cache) {
    switch (cache) {
    case ninfer::ConstraintCompileCache::None:
        return "none";
    case ninfer::ConstraintCompileCache::Hit:
        return "hit";
    case ninfer::ConstraintCompileCache::Built:
        return "built";
    case ninfer::ConstraintCompileCache::Waited:
        return "waited";
    }
    return "none";
}

inline nlohmann::json constraint_stats_json(const ninfer::ConstraintStats& stats) {
    return {{"complete", stats.complete},
            {"terminated", stats.terminated},
            {"branch", constraint_branch_name(stats.branch)},
            {"compile_cache", constraint_compile_cache_name(stats.compile_cache)},
            {"compile_seconds", stats.compile_seconds},
            {"compile_wait_seconds", stats.compile_wait_seconds},
            {"mask_calls", stats.mask_calls},
            {"mask_nodes", stats.mask_nodes},
            {"mask_upload_bytes", stats.mask_upload_bytes},
            {"mask_seconds", stats.mask_seconds}};
}

} // namespace ninfer::serve
