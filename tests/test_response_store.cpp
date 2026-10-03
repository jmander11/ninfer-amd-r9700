#include "serve/response_store.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

ChatTurn text_turn(ninfer::ChatRole role, std::string text) {
    ChatTurn turn;
    turn.role = role;
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "input_text";
    part.text     = std::move(text);
    turn.content.push_back(std::move(part));
    return turn;
}

StoredResponse record(std::string id, ResponseContext context) {
    StoredResponse value;
    value.id = std::move(id);
    value.response =
        nlohmann::json{{"id", value.id}, {"object", "response"}, {"status", "completed"}};
    value.input_items.push_back(nlohmann::json{{"id", "msg_" + value.id}, {"type", "message"}});
    value.context = std::move(context);
    return value;
}

int test_context_dag() {
    const ResponseContext first =
        append_response_context({}, {text_turn(ninfer::ChatRole::User, "one"),
                                     text_turn(ninfer::ChatRole::Assistant, "a")});
    const ResponseContext second =
        append_response_context(first, {text_turn(ninfer::ChatRole::User, "two"),
                                        text_turn(ninfer::ChatRole::Assistant, "b")});
    const std::vector<ChatTurn> flattened = flatten_response_context(second);
    int failures                          = 0;
    failures += check(flattened.size() == 4, "context chain flattened all turns");
    failures += check(flattened[0].content[0].text == "one" && flattened[3].content[0].text == "b",
                      "context chain preserves chronological order");
    failures += check(second->parent.get() == first.get(), "context nodes share their parent");
    return failures;
}

int test_lru_and_delete() {
    ResponseStore store(2, 1ULL << 20);
    const ResponseContext root =
        append_response_context({}, {text_turn(ninfer::ChatRole::User, "root")});
    store.put(record("resp_1", root));
    const ResponseContext child =
        append_response_context(root, {text_turn(ninfer::ChatRole::Assistant, "child")});
    store.put(record("resp_2", child));
    (void)store.get("resp_1"); // resp_2 becomes the least-recently used entry.
    store.put(record("resp_3",
                     append_response_context(root, {text_turn(ninfer::ChatRole::User, "fork")})));

    int failures = 0;
    failures += check(store.get("resp_1") != nullptr, "get refreshes LRU recency");
    failures += check(store.get("resp_2") == nullptr, "least-recent response evicted");
    failures += check(store.get("resp_3") != nullptr, "new response retained");
    failures += check(store.size() == 2 && store.bytes() != 0, "store reports bounded usage");
    failures += check(store.erase("resp_1"), "stored response deleted");
    failures += check(!store.erase("resp_1") && store.get("resp_1") == nullptr,
                      "deleted response is no longer addressable");
    // The child/fork context owns a shared parent even when the parent's public
    // response entry is deleted.
    const std::shared_ptr<const StoredResponse> fork = store.get("resp_3");
    failures += check(fork && flatten_response_context(fork->context).size() == 2,
                      "descendant context survives parent response deletion");
    return failures;
}

int test_oversized_record() {
    ResponseStore store(4, 256);
    StoredResponse large = record(
        "resp_large",
        append_response_context({}, {text_turn(ninfer::ChatRole::User, std::string(1024, 'x'))}));
    std::string code;
    try {
        store.put(std::move(large));
    } catch (const ApiException& exception) { code = exception.error().code; }
    int failures = 0;
    failures += check(code == "response_store_capacity_exceeded",
                      "oversized response fails deterministically");
    failures += check(store.size() == 0, "oversized insertion does not mutate store");
    return failures;
}

struct TemporaryStore {
    std::filesystem::path path;

    TemporaryStore() {
        std::string pattern =
            (std::filesystem::temp_directory_path() / "ninfer-responses-XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        if (!created) { throw std::runtime_error("cannot create test directory"); }
        path = created;
    }

    ~TemporaryStore() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

int test_persistent_dag() {
    TemporaryStore directory;
    int failures    = 0;
    const auto root = append_response_context({}, {text_turn(ninfer::ChatRole::User, "root")});
    ChatTurn output = text_turn(ninfer::ChatRole::Assistant, "answer");
    output.reasoning_content = "private reasoning";
    output.tool_calls.push_back(ToolCall{"call_1", "read", R"({"path":"example"})"});
    ChatTurn result     = text_turn(ninfer::ChatRole::Tool, "tool result");
    result.tool_call_id = "call_1";
    ContentPart image;
    image.kind              = ContentKind::Image;
    image.type_raw          = "input_image";
    image.source.kind       = ninfer::product::media_acquire::SourceKind::Bytes;
    image.source.bytes      = {0, 255, 13, 10};
    image.source.media_type = "image/png";
    result.content.push_back(image);
    const auto left = append_response_context(root, {output, result});
    const auto right =
        append_response_context(root, {text_turn(ninfer::ChatRole::User, "sibling")});
    std::size_t expected_bytes = 0;
    nlohmann::json expected_body;
    std::vector<nlohmann::json> expected_items;
    {
        ResponseStore store(4, 1ULL << 20, directory.path.string());
        store.put(record("parent", root));
        auto value                 = record("left", left);
        value.preserve_thinking    = true;
        value.response["metadata"] = {{"nested", {1, true, "value"}}};
        value.input_items.push_back(
            {{"type", "function_call_output"}, {"call_id", "call_1"}, {"output", "tool result"}});
        expected_body  = value.response;
        expected_items = value.input_items;
        store.put(std::move(value));
        store.put(record("right", right));
        failures += check(store.erase("parent"), "persistent parent delete failed");
        expected_bytes = store.bytes();
        bool locked    = false;
        try {
            ResponseStore second(4, 1ULL << 20, directory.path.string());
        } catch (const std::exception&) { locked = true; }
        failures += check(locked, "simultaneous writer admitted");
    }
    // An interrupted pre-publication payload must never create a public ID.
    {
        std::ofstream orphan(directory.path / "response-999999.json");
        orphan << "{}";
    }
    {
        ResponseStore store(4, 1ULL << 20, directory.path.string());
        const auto a = store.get("left");
        const auto b = store.get("right");
        failures += check(a && b && !store.get("parent") && store.size() == 2,
                          "restart lost children or resurrected deleted parent");
        failures +=
            check(store.bytes() == expected_bytes, "restart duplicated shared DAG accounting");
        failures += check(a->context->parent == b->context->parent,
                          "restart failed to share sibling ancestry");
        failures += check(a->response == expected_body && a->input_items == expected_items &&
                              a->preserve_thinking,
                          "restart lost typed Items, envelope or thinking policy");
        const auto turns = flatten_response_context(a->context);
        failures += check(turns.size() == 3 && turns[0].content[0].text == "root" &&
                              turns[1].reasoning_content == output.reasoning_content &&
                              turns[1].tool_calls[0].arguments_json ==
                                  output.tool_calls[0].arguments_json &&
                              turns[2].tool_call_id == "call_1" &&
                              turns[2].content[1].source.bytes == image.source.bytes &&
                              turns[2].content[1].source.media_type == "image/png" &&
                              turns[2].content[1].kind == ContentKind::Image,
                          "restart changed prompt turns or owned media");
        failures += check(!std::filesystem::exists(directory.path / "response-999999.json"),
                          "unpublished orphan was not reclaimed");
        // This get must survive restart as the most recent access.
        (void)store.get("left");
    }
    {
        ResponseStore store(2, 1ULL << 20, directory.path.string());
        store.put(record("new", {}));
        failures += check(!store.get("right") && store.get("left") && store.get("new"),
                          "restart did not preserve LRU eviction order");
    }
    {
        ResponseStore store(1, 1ULL << 20, directory.path.string());
        failures += check(store.size() == 1 && store.get("new") && !store.get("left"),
                          "reduced startup limits did not evict oldest IDs");
    }
    return failures;
}

int test_failed_publication() {
    TemporaryStore directory;
    int failures = 0;
    {
        ResponseStore store(4, 1ULL << 20, directory.path.string());
        store.put(record("committed", {}));
        // Force the manifest temporary open to fail after new immutable payloads
        // have been written. The old manifest must remain the restart authority.
        std::filesystem::create_directory(directory.path / "manifest.json.tmp");
        bool failed = false;
        try {
            store.put(record("unpublished", {}));
        } catch (const std::exception&) { failed = true; }
        failures += check(failed && store.size() == 1, "failed persistence changed live index");
        bool unavailable = false;
        try {
            (void)store.get("committed");
        } catch (const std::exception&) { unavailable = true; }
        failures += check(unavailable, "I/O failure did not fail closed until restart");
        std::filesystem::remove(directory.path / "manifest.json.tmp");
    }
    {
        ResponseStore restored(4, 1ULL << 20, directory.path.string());
        failures += check(restored.get("committed") && !restored.get("unpublished"),
                          "failed publication became visible after restart");
        restored.erase("committed");
    }
    {
        ResponseStore restored(4, 1ULL << 20, directory.path.string());
        failures += check(restored.size() == 0, "last deletion was not durable");
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_context_dag();
    failures += test_lru_and_delete();
    failures += test_oversized_record();
    failures += test_persistent_dag();
    failures += test_failed_publication();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
