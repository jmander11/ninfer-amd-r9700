#include "decode_graph.h"

#include "device.h"

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void log_hip_error(const char* op, hipError_t error) noexcept {
    if (error != hipSuccess) {
        std::fprintf(stderr, "HIP cleanup failed during %s: %s\n", op,
                     hipGetErrorString(error));
    }
}

void destroy_graph_exec(hipGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_hip_error("hipGraphExecDestroy", hipGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(hipGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_hip_error("hipGraphDestroy", hipGraphDestroy(graph));
        graph = nullptr;
    }
}

void discard_capture(hipStream_t stream) noexcept {
    hipGraph_t discard = nullptr;
    log_hip_error("hipStreamEndCapture(discard)", hipStreamEndCapture(stream, &discard));
    destroy_graph(discard);
}

std::string describe_graph_node(hipGraphNode_t node) {
    if (node == nullptr) { return "none"; }
    hipGraphNodeType type{};
    const hipError_t error = hipGraphNodeGetType(node, &type);
    if (error != hipSuccess) { return std::string("unavailable:") + hipGetErrorName(error); }
    std::ostringstream out;
    out << "type=" << static_cast<int>(type);
    if (type == hipGraphNodeTypeKernel) {
        hipKernelNodeParams params{};
        if (hipGraphKernelNodeGetParams(node, &params) == hipSuccess) {
            const char* name = hipKernelNameRefByPtr(params.func, nullptr);
            out << " kernel=" << (name != nullptr ? name : "unknown")
                << " grid=" << params.gridDim.x << ',' << params.gridDim.y << ',' << params.gridDim.z
                << " block=" << params.blockDim.x << ',' << params.blockDim.y << ','
                << params.blockDim.z << " shared_bytes=" << params.sharedMemBytes;
        }
    } else if (type == hipGraphNodeTypeMemcpy) {
        hipMemcpy3DParms params{};
        if (hipGraphMemcpyNodeGetParams(node, &params) == hipSuccess) {
            out << " memcpy_kind=" << static_cast<int>(params.kind) << " extent="
                << params.extent.width << ',' << params.extent.height << ',' << params.extent.depth
                << " src_pitch=" << params.srcPtr.pitch << " dst_pitch=" << params.dstPtr.pitch;
        }
    } else if (type == hipGraphNodeTypeMemset) {
        hipMemsetParams params{};
        if (hipGraphMemsetNodeGetParams(node, &params) == hipSuccess) {
            out << " memset_width=" << params.width << " height=" << params.height
                << " pitch=" << params.pitch << " element_bytes=" << params.elementSize;
        }
    }
    return out.str();
}

} // namespace

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_) {
    other.graph_ = nullptr;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this != &other) {
        reset();
        graph_       = other.graph_;
        other.graph_ = nullptr;
    }
    return *this;
}

void DecodeGraphDefinition::capture(hipStream_t stream, const std::function<void()>& body) {
    reset();
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));

    try {
        body();
    } catch (...) {
        discard_capture(stream);
        throw;
    }

    hipGraph_t graph       = nullptr;
    const hipError_t error = hipStreamEndCapture(stream, &graph);
    if (error != hipSuccess) {
        destroy_graph(graph);
        HIP_CHECK(error);
    }
    graph_ = graph;
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

void DecodeGraphDefinition::reset() noexcept { destroy_graph(graph_); }

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_) {
    other.exec_ = nullptr;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this != &other) {
        reset();
        exec_       = other.exec_;
        other.exec_ = nullptr;
    }
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty HIP Graph definition");
    }
    reset();

    hipGraphNode_t error_node = nullptr;
    char log[1024]{};
    const hipError_t error =
        hipGraphInstantiate(&exec_, definition.graph_, &error_node, log, sizeof(log));
    if (error != hipSuccess) {
        destroy_graph_exec(exec_);
        throw std::runtime_error(std::string("HIP Graph instantiate failed: ") +
                                 hipGetErrorString(error) + ": " + log);
    }
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    if (!ready() || !definition.ready()) {
        throw std::logic_error("HIP Graph update requires a definition and executable");
    }

    hipGraphNode_t error_node = nullptr;
    hipGraphExecUpdateResult result{};
    const hipError_t error =
        hipGraphExecUpdate(exec_, definition.graph_, &error_node, &result);
    if (error != hipSuccess || result != hipGraphExecUpdateSuccess) {
        throw std::runtime_error(
            "HIP Graph executable update failed: " + std::string(hipGetErrorString(error)) +
            " (update result " + std::to_string(static_cast<int>(result)) + "); error_node={" +
            describe_graph_node(error_node) + "}");
    }
}

void DecodeGraphExecutable::upload(hipStream_t stream) {
    if (!ready()) { throw std::logic_error("cannot upload an empty HIP Graph executable"); }
    HIP_CHECK(hipGraphUpload(exec_, stream));
}

void DecodeGraphExecutable::launch(hipStream_t stream) {
    if (!ready()) { throw std::logic_error("cannot launch an empty HIP Graph executable"); }
    HIP_CHECK(hipGraphLaunch(exec_, stream));
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraphExecutable::reset() noexcept { destroy_graph_exec(exec_); }

} // namespace ninfer
