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

    const ninfer::cli::Options defaults = parse(
        {"ninfer", "model.ninfer", "--prompt", "hi"});
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
    failures += check(off.context_checkpoint_marks.has_value() &&
                          off.context_checkpoint_marks->empty(),
                      "--context-checkpoints off did not disable the ladder");

    const ninfer::cli::Options custom = parse(
        {"ninfer", "model.ninfer", "--prompt", "hi", "--context-checkpoints", "8192,16384"});
    failures += check(custom.context_checkpoint_marks ==
                          std::optional<std::vector<std::uint32_t>>(
                              std::vector<std::uint32_t>{8192u, 16384u}),
                      "--context-checkpoints custom list was not parsed");

    failures += check(ninfer::cli::usage_text("ninfer").find("--capture-context-checkpoint") !=
                          std::string::npos,
                      "CLI help omits --capture-context-checkpoint");
    failures += check(ninfer::cli::usage_text("ninfer").find("--context-checkpoints") !=
                          std::string::npos,
                      "CLI help omits --context-checkpoints");

    const ninfer::cli::Options eager =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--no-device-graph"});
    failures += check(!eager.use_device_graph,
                      "--no-device-graph did not disable Device Graph decode");
    failures += check(ninfer::cli::usage_text("ninfer").find("--no-device-graph") !=
                          std::string::npos,
                      "CLI help omits --no-device-graph");
    const std::string help = ninfer::cli::usage_text("ninfer");
    failures += check(help.find("--kv-capacity") != std::string::npos &&
                          help.find("--kv-ram-capacity") != std::string::npos,
                      "CLI help omits the fixed-cache capacity controls");

    const ninfer::cli::Options mtp_vision =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--spec", "mtp",
               "--draft-tokens", "3", "--vision"});
    failures += check(mtp_vision.enable_vision &&
                          mtp_vision.speculative.backend == ninfer::SpeculativeBackend::Mtp,
                      "MTP and Vision were not accepted together");

    bool dflash_vision_rejected = false;
    try {
        (void)parse({"ninfer", "model.ninfer", "--prompt", "hi", "--spec", "dflash",
                     "--draft-tokens", "11", "--vision"});
    } catch (const std::invalid_argument&) { dflash_vision_rejected = true; }
    failures += check(dflash_vision_rejected, "DFlash and Vision were accepted together");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
