#include "serve/request_log.h"
#include "serve/serve_options.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        const ninfer::serve::ServerLogEnvironment environment =
            ninfer::serve::query_server_log_environment(0);
        if (environment.gpu_name.find("R9700") == std::string::npos ||
            environment.architecture_name.rfind("gfx1201", 0) != 0 ||
            environment.total_device_memory_bytes == 0 || environment.hip_compile_version.empty() ||
            environment.hip_runtime_version.empty() || environment.hip_driver_version.empty()) {
            throw std::runtime_error("HIP server environment is incomplete or is not gfx1201");
        }
        ninfer::serve::ServeOptions options;
        options.artifact_path = "model.ninfer";
        ninfer::LoadSummary load;
        load.target     = "qwen3_8_27b_r9700";
        load.weights_id = "r9700-integer";
        ninfer::MemorySummary memory;
        const std::string server_record = ninfer::serve::format_server_start_json(
            "qual", 1, options, {}, "qwen3.8-27b-r9700", load, memory, environment,
            std::nullopt);
        const std::string schema_field =
            "\"schema_version\":" +
            std::to_string(ninfer::serve::kRequestLogSchemaVersion);
        if (server_record.find(schema_field) == std::string::npos ||
            server_record.find("\"kv_cache_format\":\"fp8-k-int4-v\"") ==
                std::string::npos) {
            throw std::runtime_error("server-start JSON does not expose the fixed HIP contract");
        }
        std::cout << "serve_environment: PASS gpu=" << environment.gpu_name
                  << " arch=" << environment.architecture_name
                  << " hip_compile=" << environment.hip_compile_version
                  << " hip_runtime=" << environment.hip_runtime_version
                  << " hip_driver=" << environment.hip_driver_version
                  << " fixed_kv=fp8-k-int4-v\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "serve_environment: FAIL: " << error.what() << '\n';
        return 1;
    }
}
