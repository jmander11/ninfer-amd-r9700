#include "cli/options.h"

#include <array>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;
    const auto constrained =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--json-schema", "schema.json"});
    failures +=
        check(constrained.json_schema_path == "schema.json", "JSON schema file was not parsed");
    const auto grammar =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--grammar", "answer.ebnf"});
    failures += check(grammar.grammar_path == "answer.ebnf", "grammar file was not parsed");
    bool conflict_rejected = false;
    try {
        (void)parse({"ninfer", "model.ninfer", "--prompt", "hi", "--json-object", "--grammar",
                     "answer.ebnf"});
    } catch (const std::invalid_argument&) { conflict_rejected = true; }
    failures += check(conflict_rejected, "CLI accepted competing output constraints");


    const ninfer::cli::Options defaults = parse({"ninfer", "model.ninfer", "--prompt", "hi"});
    failures += check(defaults.prefill_chunk == ninfer::kDefaultPrefillChunk,
                      "CLI prefill chunk diverges from the product default");

    const ninfer::cli::Options pin =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--capture-context-checkpoint"});
    failures += check(pin.capture_context_checkpoint,
                      "--capture-context-checkpoint did not set the request pin");
    failures += check(!pin.context_checkpoint_marks.has_value(),
                      "omitted --context-checkpoints is not the product default table");

    const ninfer::cli::Options off =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--context-checkpoints", "off"});
    failures +=
        check(off.context_checkpoint_marks.has_value() && off.context_checkpoint_marks->empty(),
              "--context-checkpoints off did not disable the ladder");

    const ninfer::cli::Options custom =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--context-checkpoints", "8192,16384"});
    failures +=
        check(custom.context_checkpoint_marks == std::optional<std::vector<std::uint32_t>>(
                                                     std::vector<std::uint32_t>{8192u, 16384u}),
              "--context-checkpoints custom list was not parsed");

    failures += check(ninfer::cli::usage_text("ninfer").find("--capture-context-checkpoint") !=
                          std::string::npos,
                      "CLI help omits --capture-context-checkpoint");
    failures +=
        check(ninfer::cli::usage_text("ninfer").find("--no-p-less-sampling") != std::string::npos,
              "CLI help omits --no-p-less-sampling");

    failures += check(defaults.sampling.p_less, "CLI did not enable p-less by default");
    failures += check(!defaults.speculative.dflash_p_less_draft_temperature.has_value(),
                      "CLI p-less draft temperature is pinned by default");
    const ninfer::cli::Options greedy_drafts =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--spec", "dflash", "--draft-tokens",
               "5", "--dflash-p-less-draft-temperature", "0"});
    failures += check(greedy_drafts.speculative.dflash_p_less_draft_temperature == 0.0F,
                      "--dflash-p-less-draft-temperature did not set SpeculativeOptions");
    const ninfer::cli::Options production = parse(
        {"ninfer", "model.ninfer", "--prompt", "hi", "--no-p-less-sampling", "--top-p", "0.5"});
    failures += check(!production.sampling.p_less && production.sampling.top_p == 0.5F,
                      "--no-p-less-sampling did not opt into the production sampler");

    const ninfer::cli::Options dflash_vision =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--spec", "dflash", "--draft-tokens",
               "3", "--vision"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "CLI did not accept DFlash and Vision together");

    const ninfer::cli::Options eager =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--no-device-graph"});
    failures +=
        check(!eager.use_device_graph, "--no-device-graph did not disable Device Graph decode");
    failures +=
        check(ninfer::cli::usage_text("ninfer").find("--no-device-graph") != std::string::npos,
              "CLI help omits --no-device-graph");
    const std::string help = ninfer::cli::usage_text("ninfer");
    failures += check(help.find("--kv-capacity") != std::string::npos &&
                          help.find("--kv-ram-capacity") != std::string::npos,
                      "CLI help omits the fixed-cache capacity controls");

    const ninfer::cli::Options headroom =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--kv-capacity", "auto",
               "--kv-capacity-headroom", "512"});
    failures +=
        check(headroom.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                  headroom.kv_capacity.automatic_headroom_bytes == 512ULL * 1024ULL * 1024ULL,
              "--kv-capacity-headroom did not set automatic headroom in MiB");
    bool headroom_rejected = false;
    try {
        (void)parse({"ninfer", "model.ninfer", "--prompt", "hi", "--kv-capacity-headroom", "512"});
    } catch (const std::invalid_argument&) { headroom_rejected = true; }
    failures += check(headroom_rejected, "--kv-capacity-headroom without auto was accepted");

    const ninfer::cli::Options mtp_vision =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--spec", "mtp", "--draft-tokens", "3",
               "--vision"});
    failures += check(mtp_vision.enable_vision &&
                          mtp_vision.speculative.backend == ninfer::SpeculativeBackend::Mtp,
                      "MTP and Vision were not accepted together");

    bool dflash_vision_rejected = false;
    try {
        (void)parse({"ninfer", "model.ninfer", "--prompt", "hi", "--spec", "dflash",
                     "--draft-tokens", "11", "--vision"});
    } catch (const std::invalid_argument&) { dflash_vision_rejected = true; }
    failures += check(dflash_vision_rejected, "DFlash accepted draft length above seven");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
