// Real public-Engine budget/context transition check; never a timing authority.
#include <ninfer/engine.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) throw std::runtime_error("tail_check ARTIFACT CORPUS ordinary|graph|eager FRESH.json [PROMPT_STRIDE]");
        const unsigned stride = argc == 6 ? std::stoul(argv[5]) : 17U;
        const std::string mode = argv[3];
        if (mode != "ordinary" && mode != "graph" && mode != "eager" &&
            mode != "fixed3" && mode != "fixed4" && mode != "fixed5")
            throw std::runtime_error("invalid mode");
        if (std::filesystem::exists(argv[4])) throw std::runtime_error("output exists");
        std::ifstream corpus_file(argv[2]);
        std::vector<ninfer::TokenId> corpus;
        ninfer::TokenId token;
        while (corpus_file >> token) corpus.push_back(token);
        if (corpus.size() < 4096) throw std::runtime_error("short corpus");
        std::ofstream report(argv[4]);
        report << "{\"mode\":\"" << mode << "\",\"timing_eligible\":false,\"cases\":[";
        bool first = true;
        for (unsigned context : {77U, 4240U}) {
            ninfer::EngineOptions options;
            options.artifact_path = argv[1];
            options.max_context = context;
            options.max_concurrency = 4;
            options.max_pending_requests = 4;
            options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(((context+63)/64)*64*4);
            options.prefill_chunk = 2048;
            options.use_device_graph = mode != "eager";
            if (mode != "ordinary") {
                options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
                options.speculative.draft_tokens = mode.starts_with("fixed") ? mode.back()-'0' : 5;
                options.speculative.adaptive_draft = !mode.starts_with("fixed");
            }
            ninfer::Engine engine(options);
            const unsigned prompt_length = context == 77 ? 64 : 4096;
            std::vector<std::vector<unsigned>> waves = context == 77
                ? std::vector<std::vector<unsigned>>{{2},{3},{4},{13,11,7,3}}
                : std::vector<std::vector<unsigned>>{{129,125,119,113}};
            for (const auto& limits : waves) {
                std::vector<ninfer::GenerationHandle> handles;
                for (unsigned lane = 0; lane < limits.size(); ++lane) {
                    std::vector<ninfer::TokenId> ids(prompt_length);
                    for (unsigned i=0;i<prompt_length;++i) ids[i]=corpus[(lane*stride+i)%corpus.size()];
                    auto prompt = engine.prepare_tokens(std::move(ids), false);
                    ninfer::RequestOptions request;
                    request.execution.requested_output_tokens = limits[lane];
                    request.execution.allow_prefix_reuse = false;
                    request.execution.sampling.temperature = 0.0F;
                    request.stop.include_model_defaults = false;
                    request.output.raw = true;
                    request.output.preserve_special_tokens = true;
                    handles.push_back(engine.submit(std::move(prompt), request,
                        ninfer::OutputDelivery::TerminalOnly,
                        std::chrono::steady_clock::time_point::max()));
                }
                for (unsigned lane = 0; lane < limits.size(); ++lane) {
                    auto result = handles[lane].wait();
                    if (result.generated_token_ids.size() != limits[lane] ||
                        result.finish_reason != ninfer::FinishReason::OutputLimit)
                        throw std::runtime_error("wrong output allowance or terminal reason");
                    if (!first) report << ',';
                    first = false;
                    report << "{\"context\":" << context << ",\"prompt\":" << prompt_length
                           << ",\"concurrency\":" << limits.size() << ",\"lane\":" << lane
                           << ",\"output_limit\":" << limits[lane] << ",\"tokens\":[";
                    for (unsigned i=0;i<result.generated_token_ids.size();++i) {
                        if (i) report << ',';
                        report << result.generated_token_ids[i];
                    }
                    report << "]}";
                    report.flush();
                }
            }
        }
        report << "]}\n";
        std::cout << "PASS public Engine output/context limits\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
