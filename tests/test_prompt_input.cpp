#include "product/prompt_input/prompt_input.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
struct Scratch {
    std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("ninfer-message-input-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

    Scratch() { std::filesystem::create_directories(root); }

    ~Scratch() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};
} // namespace

int main() {
    try {
        Scratch scratch;
        const auto media    = scratch.root / "media.bin";
        const auto messages = scratch.root / "messages.json";
        { std::ofstream(media, std::ios::binary) << "owned media"; }
        for (const auto& source :
             {std::string("media.bin"), media.string(), std::string("file://media.bin"),
              std::string("data:application/octet-stream;base64,b3duZWQgbWVkaWE=")}) {
            const nlohmann::json input = nlohmann::json::array(
                {{{"role", "user"},
                  {"content", nlohmann::json::array({{{"type", "image"}, {"image", source}}})}}});
            { std::ofstream(messages) << input.dump(); }
            const auto prompt = ninfer::product::prompt_from_messages(messages, false, true);
            const auto& bytes = prompt.messages.at(0).parts.at(0).media.bytes;
            if (std::string(bytes.begin(), bytes.end()) != "owned media") {
                throw std::runtime_error("media acquisition did not preserve the file-owned bytes");
            }
        }
        // A/../media is evaluated after following A if A is a symlink.
        std::filesystem::create_directories(scratch.root / "actual" / "child");
        std::filesystem::create_directory_symlink(scratch.root / "actual" / "child",
                                                  scratch.root / "alias");
        { std::ofstream(scratch.root / "actual" / "media.bin") << "symlink-owned media"; }
        {
            const nlohmann::json input = nlohmann::json::array(
                {{{"role", "user"},
                  {"content", nlohmann::json::array(
                                  {{{"type", "image"}, {"image", "alias/../media.bin"}}})}}});
            { std::ofstream(messages) << input.dump(); }
            const auto prompt = ninfer::product::prompt_from_messages(messages, false, true);
            const auto& bytes = prompt.messages.at(0).parts.at(0).media.bytes;
            if (std::string(bytes.begin(), bytes.end()) != "symlink-owned media")
                throw std::runtime_error("relative media changed symlink traversal");
        }
        {
            std::ofstream(messages)
                << R"({"messages":[{"role":"user","content":"number"}],"tools":[{"function":{"name":"n","parameters":{"const":9007199254740993.0}}}]})";
        }
        bool rejected = false;
        try {
            (void)ninfer::product::prompt_from_messages(messages, false, false);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected)
            throw std::runtime_error("CLI tool schema silently rounded its source number");
        const std::string tool =
            R"({"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"title":{"type":"string"},"people":{"type":"array"}},"required":["title","people"]}}})";
        {
            std::ofstream(messages)
                << R"({"messages":[{"role":"user","content":"hello"}],"tools":[)" << tool << "]}";
        }
        const auto prompt = ninfer::product::prompt_from_messages(messages, false, false);
        if (prompt.options.tool_jsons != std::vector<std::string>{tool}) {
            throw std::runtime_error("messages-file tool schema lost property order");
        }
        std::cout << "PASS: messages-relative acquisition, schema numeric source and order\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
