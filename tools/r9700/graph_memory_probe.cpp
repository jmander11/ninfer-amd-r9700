// Startup-only gfx1201 graph allowance diagnosis. LD_PRELOAD into the real product
// command; never changes its capacity plan or suppresses its allocation guard.
// Upload is synchronized to attribute its physical residency, so this is not a
// performance measurement tool. All records go to stderr alongside product logs.
// Build: c++ -std=c++20 -shared -fPIC -D__HIP_PLATFORM_AMD__ \
//   -I/opt/rocm/include tools/r9700/graph_memory_probe.cpp -ldl -o /tmp/graph_memory_probe.so
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

namespace {
template <class F> F next(const char* name) {
    auto value = reinterpret_cast<F>(dlsym(RTLD_NEXT, name));
    if (!value) { std::fprintf(stderr, "graph-memory: missing %s\n", name); std::abort(); }
    return value;
}
std::size_t free_bytes() {
    std::size_t available = 0, total = 0;
    if (next<decltype(&hipMemGetInfo)>("hipMemGetInfo")(&available, &total) != hipSuccess) {
        std::fprintf(stderr, "graph-memory: memory query failed\n"); std::abort();
    }
    return available;
}
std::size_t node_count(hipGraph_t graph) {
    std::size_t count = 0;
    if (next<decltype(&hipGraphGetNodes)>("hipGraphGetNodes")(graph, nullptr, &count) != hipSuccess) {
        std::fprintf(stderr, "graph-memory: node query failed\n"); std::abort();
    }
    return count;
}
void record(const char* event, const void* handle, std::size_t nodes,
            std::size_t before, std::size_t after, hipError_t status) {
    std::fprintf(stderr,
        "graph-memory event=%s handle=%p nodes=%zu free_before=%zu free_after=%zu delta=%lld status=%d\n",
        event, handle, nodes, before, after,
        static_cast<long long>(before) - static_cast<long long>(after), static_cast<int>(status));
}
}

extern "C" hipError_t hipMemGetInfo(std::size_t* available, std::size_t* total) {
    const auto status = next<decltype(&hipMemGetInfo)>("hipMemGetInfo")(available, total);
    if (status == hipSuccess) record("product_query", nullptr, 0, *available, *available, status);
    return status;
}

extern "C" hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* graph) {
    // Query only after capture ends: a memory query inside capture is not legal.
    const auto status = next<decltype(&hipStreamEndCapture)>("hipStreamEndCapture")(stream, graph);
    if (status == hipSuccess) {
        const auto available = free_bytes();
        record("captured", *graph, node_count(*graph), available, available, status);
    }
    return status;
}

extern "C" hipError_t hipGraphInstantiate(hipGraphExec_t* executable, hipGraph_t graph,
                                          hipGraphNode_t* error_node, char* log, std::size_t size) {
    const auto before = free_bytes();
    const auto status = next<decltype(&hipGraphInstantiate)>("hipGraphInstantiate")(
        executable, graph, error_node, log, size);
    record("instantiate", status == hipSuccess ? *executable : nullptr,
           node_count(graph), before, free_bytes(), status);
    return status;
}

extern "C" hipError_t hipGraphExecUpdate(hipGraphExec_t executable, hipGraph_t graph,
                                         hipGraphNode_t* error_node,
                                         hipGraphExecUpdateResult* result) {
    const auto before = free_bytes();
    const auto status = next<decltype(&hipGraphExecUpdate)>("hipGraphExecUpdate")(
        executable, graph, error_node, result);
    record("update", executable, node_count(graph), before, free_bytes(), status);
    return status;
}

extern "C" hipError_t hipGraphUpload(hipGraphExec_t executable, hipStream_t stream) {
    const auto before = free_bytes();
    const auto status = next<decltype(&hipGraphUpload)>("hipGraphUpload")(executable, stream);
    if (status == hipSuccess) {
        const auto sync = next<decltype(&hipStreamSynchronize)>("hipStreamSynchronize")(stream);
        if (sync != hipSuccess) { record("upload_sync_failed", executable, 0, before, free_bytes(), sync); return sync; }
    }
    record("upload", executable, 0, before, free_bytes(), status);
    return status;
}
