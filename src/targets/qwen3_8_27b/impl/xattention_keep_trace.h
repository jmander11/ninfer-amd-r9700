#pragma once

// Diagnostic-only, opt-in capture of real model sparse keep counts. Compiled only
// in a separate diagnostic build; synchronous copies make its runs timing-ineligible.
#include "ops/r9700/kv/fp8_int4_kv_xattention.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ninfer::targets::qwen3_8_27b::detail {
inline hipError_t trace_xattention_keep(
    const ops::r9700::kv::Fp8Int4KvXAttentionArgs& args, hipStream_t stream) noexcept {
    const char* path = std::getenv("NINFER_XATTENTION_KEEP_TRACE");
    if (path == nullptr || *path == '\0') return hipSuccess;
    try {
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        auto status = hipStreamIsCapturing(stream, &capture);
        if (status != hipSuccess) return status;
        if (capture != hipStreamCaptureStatusNone) return hipErrorStreamCaptureUnsupported;
        const auto& a = args.attention;
        const auto work = ops::r9700::kv::bind_fp8_int4_kv_xattention_workspace(
            args.workspace, args.workspace_bytes, a.query_heads, a.kv_heads,
            a.query_rows, a.context);
        if (work.keep_count == nullptr) return hipErrorInvalidValue;
        std::vector<std::int32_t> counts(std::size_t(a.query_heads) * work.query_blocks);
        status = hipMemcpyAsync(counts.data(), work.keep_count,
                                counts.size() * sizeof(counts[0]), hipMemcpyDeviceToHost, stream);
        if (status != hipSuccess) return status;
        status = hipStreamSynchronize(stream);
        if (status != hipSuccess) return status;
        // One trusted, serialized C1 diagnostic process; no global product telemetry.
        struct Output {
            std::FILE* file;
            std::size_t dispatch = 0;
            explicit Output(const char* name) : file(std::fopen(name, "wx")) {}
            ~Output() { if (file) std::fclose(file); }
        };
        static Output output(path);
        if (output.file == nullptr) return hipErrorInvalidValue;
        if (std::fprintf(output.file,
                "{\"schema_version\":1,\"diagnostic_only\":true,\"timing_eligible\":false,"
                "\"dispatch\":%zu,\"context\":%zu,\"query_rows\":%u,\"query_heads\":%u,"
                "\"kv_value_group\":%u,\"query_block_rows\":128,\"page_tokens\":64,"
                "\"keep_counts\":[", output.dispatch++, a.context, a.query_rows,
                a.query_heads, a.value_group) < 0) return hipErrorUnknown;
        for (std::size_t i = 0; i < counts.size(); ++i) {
            if (std::fprintf(output.file, "%s%d", i == 0 ? "" : ",", counts[i]) < 0)
                return hipErrorUnknown;
        }
        if (std::fputs("]}\n", output.file) < 0 || std::fflush(output.file) != 0)
            return hipErrorUnknown;
        return hipSuccess;
    } catch (...) {
        return hipErrorUnknown;
    }
}
} // namespace ninfer::targets::qwen3_8_27b::detail
