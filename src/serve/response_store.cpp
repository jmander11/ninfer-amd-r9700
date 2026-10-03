#include "serve/response_store.h"

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace ninfer::serve {
namespace {

std::size_t estimate_turn_bytes(const ChatTurn& turn) {
    std::size_t bytes = sizeof(ChatTurn) + turn.tool_call_id.size() + turn.reasoning_content.size();
    for (const ContentPart& part : turn.content) {
        bytes += sizeof(ContentPart) + part.text.size() + part.type_raw.size() +
                 part.source.value.size() + part.source.media_type.size() +
                 part.source.bytes.size();
    }
    for (const ToolCall& call : turn.tool_calls) {
        bytes += sizeof(ToolCall) + call.id.size() + call.name.size() + call.arguments_json.size();
    }
    return bytes;
}

std::size_t record_envelope_bytes(const StoredResponse& record) {
    std::size_t bytes = sizeof(StoredResponse) + record.id.size() + record.response.dump().size();
    for (const nlohmann::json& item : record.input_items) {
        bytes += sizeof(nlohmann::json) + item.dump().size();
    }
    return bytes;
}

std::size_t standalone_bytes(const StoredResponse& record) {
    std::size_t bytes = record_envelope_bytes(record);
    for (ResponseContext node = record.context; node != nullptr; node = node->parent) {
        bytes += node->owned_bytes;
    }
    return bytes;
}

[[noreturn]] void throw_store_capacity() {
    ApiError error;
    error.status  = 500;
    error.type    = "server_error";
    error.code    = "response_store_capacity_exceeded";
    error.message = "response exceeds the configured local response store capacity";
    throw ApiException(std::move(error));
}

} // namespace

ResponseContext append_response_context(ResponseContext parent, std::vector<ChatTurn> turns) {
    auto node         = std::make_shared<ResponseContextNode>();
    node->parent      = std::move(parent);
    node->turns       = std::move(turns);
    node->owned_bytes = sizeof(ResponseContextNode);
    for (const ChatTurn& turn : node->turns) { node->owned_bytes += estimate_turn_bytes(turn); }
    return node;
}

std::vector<ChatTurn> flatten_response_context(const ResponseContext& context) {
    std::vector<const ResponseContextNode*> nodes;
    std::size_t turn_count = 0;
    for (ResponseContext node = context; node != nullptr; node = node->parent) {
        nodes.push_back(node.get());
        turn_count += node->turns.size();
    }
    std::vector<ChatTurn> turns;
    turns.reserve(turn_count);
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        turns.insert(turns.end(), (*it)->turns.begin(), (*it)->turns.end());
    }
    return turns;
}

namespace {
using Json = nlohmann::json;

Json encode_turn(const ChatTurn& turn) {
    Json parts = Json::array();
    for (const auto& part : turn.content) {
        parts.push_back(Json{{"kind", part.kind},
                             {"text", part.text},
                             {"type", part.type_raw},
                             {"source",
                              {{"kind", part.source.kind},
                               {"value", part.source.value},
                               {"media_type", part.source.media_type},
                               {"bytes", part.source.bytes}}}});
    }
    Json calls = Json::array();
    for (const auto& call : turn.tool_calls) {
        calls.push_back(
            Json{{"id", call.id}, {"name", call.name}, {"arguments", call.arguments_json}});
    }
    return Json{{"role", turn.role},
                {"parts", parts},
                {"calls", calls},
                {"tool_call_id", turn.tool_call_id},
                {"reasoning", turn.reasoning_content}};
}

ChatTurn decode_turn(const Json& value) {
    ChatTurn turn;
    turn.role              = value.at("role").get<ChatRole>();
    turn.tool_call_id      = value.at("tool_call_id").get<std::string>();
    turn.reasoning_content = value.at("reasoning").get<std::string>();
    for (const auto& encoded : value.at("parts")) {
        ContentPart part;
        part.kind          = encoded.at("kind").get<ContentKind>();
        part.text          = encoded.at("text").get<std::string>();
        part.type_raw      = encoded.at("type").get<std::string>();
        const auto& source = encoded.at("source");
        part.source.kind   = source.at("kind").get<ninfer::product::media_acquire::SourceKind>();
        part.source.value  = source.at("value").get<std::string>();
        part.source.media_type = source.at("media_type").get<std::string>();
        part.source.bytes      = source.at("bytes").get<std::vector<std::uint8_t>>();
        turn.content.push_back(std::move(part));
    }
    for (const auto& call : value.at("calls")) {
        turn.tool_calls.push_back(ToolCall{call.at("id").get<std::string>(),
                                           call.at("name").get<std::string>(),
                                           call.at("arguments").get<std::string>()});
    }
    return turn;
}

std::size_t ordered_bytes(const std::vector<std::shared_ptr<const StoredResponse>>& records) {
    std::size_t bytes = 0;
    std::unordered_set<const ResponseContextNode*> seen;
    for (const auto& record : records) {
        bytes += record_envelope_bytes(*record);
        for (auto node = record->context; node; node = node->parent) {
            if (seen.insert(node.get()).second) { bytes += node->owned_bytes; }
        }
    }
    return bytes;
}

[[noreturn]] void disk_error(const std::string& detail) {
    throw std::runtime_error("response store persistence: " + detail);
}
} // namespace

// Only immutable protocol payloads are stored here. The small manifest publishes
// the live set and exact LRU order; it is the sole restart commit point. Context
// nodes are shared on disk and in memory even after their public parent is deleted.
struct ResponseStore::DiskStore {
    std::filesystem::path directory;
    int lock_fd             = -1;
    bool failed             = false;
    std::uint64_t next_file = 1;
    std::unordered_map<const ResponseContextNode*, std::string> nodes;
    std::unordered_map<const StoredResponse*, std::string> records;

    explicit DiskStore(std::string location) : directory(std::move(location)) {
        std::filesystem::create_directories(directory);
        lock_fd = ::open((directory / "store.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (lock_fd < 0) { disk_error("cannot open directory lock"); }
        if (::flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
            ::close(lock_fd);
            lock_fd = -1;
            disk_error("directory is already in use");
        }
    }

    ~DiskStore() {
        if (lock_fd >= 0) { ::close(lock_fd); }
    }

    void sync_directory() {
        const int fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd < 0) { disk_error("cannot open directory for sync"); }
        const int result = ::fsync(fd);
        ::close(fd);
        if (result != 0) { disk_error("cannot sync directory"); }
    }

    void write(const std::string& name, const Json& data) {
        const auto temporary    = directory / (name + ".tmp");
        const std::string bytes = data.dump();
        const int fd = ::open(temporary.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0600);
        if (fd < 0) { disk_error("cannot write " + name); }
        std::size_t offset = 0;
        while (offset != bytes.size()) {
            const auto n = ::write(fd, bytes.data() + offset, bytes.size() - offset);
            if (n < 0 && errno == EINTR) { continue; }
            if (n <= 0) {
                ::close(fd);
                disk_error("short write of " + name);
            }
            offset += static_cast<std::size_t>(n);
        }
        const int synced = ::fsync(fd);
        const int closed = ::close(fd);
        if (synced != 0 || closed != 0) { disk_error("cannot sync " + name); }
        std::filesystem::rename(temporary, directory / name);
    }

    Json read(const std::string& name) const {
        // File references come only from this store's manifest/payloads, never
        // response IDs. Reject corrupted path references instead of following them.
        if (name.empty() || std::filesystem::path(name).filename() != name) {
            disk_error("invalid payload filename");
        }
        std::ifstream stream(directory / name);
        if (!stream) { disk_error("missing payload " + name); }
        return Json::parse(stream);
    }

    std::vector<std::shared_ptr<const StoredResponse>> load() {
        std::vector<std::shared_ptr<const StoredResponse>> ordered;
        if (!std::filesystem::exists(directory / "manifest.json")) { return ordered; }
        const auto manifest = read("manifest.json");
        if (manifest.at("version") != 1) { disk_error("unsupported manifest version"); }
        next_file = manifest.at("next_file").get<std::uint64_t>();
        std::unordered_map<std::string, ResponseContext> loaded;
        std::unordered_set<std::string> ids;
        for (const auto& file : manifest.at("records")) {
            const auto filename = file.get<std::string>();
            const auto encoded  = read(filename);
            std::string current = encoded.at("context").get<std::string>();
            std::vector<std::pair<std::string, Json>> chain;
            std::unordered_set<std::string> visiting;
            while (!current.empty() && !loaded.contains(current)) {
                if (!visiting.insert(current).second) { disk_error("cyclic context chain"); }
                auto value = read(current);
                chain.emplace_back(current, std::move(value));
                current = chain.back().second.at("parent").get<std::string>();
            }
            ResponseContext context = current.empty() ? nullptr : loaded.at(current);
            for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                std::vector<ChatTurn> turns;
                for (const auto& turn : it->second.at("turns")) {
                    turns.push_back(decode_turn(turn));
                }
                context = append_response_context(context, std::move(turns));
                loaded.emplace(it->first, context);
                nodes.emplace(context.get(), it->first);
            }
            auto record               = std::make_shared<StoredResponse>();
            record->id                = encoded.at("id").get<std::string>();
            record->response          = encoded.at("response");
            record->input_items       = encoded.at("input_items").get<std::vector<Json>>();
            record->preserve_thinking = encoded.at("preserve_thinking").get<bool>();
            record->context           = std::move(context);
            if (record->id.empty() || !record->response.is_object() ||
                !ids.insert(record->id).second) {
                disk_error("invalid or duplicate response record");
            }
            records.emplace(record.get(), filename);
            ordered.push_back(std::move(record));
        }
        return ordered;
    }

    void reorder(const std::vector<std::shared_ptr<const StoredResponse>>& ordered) {
        Json files = Json::array();
        for (const auto& record : ordered) { files.push_back(records.at(record.get())); }
        write("manifest.json", Json{{"version", 1}, {"next_file", next_file}, {"records", files}});
        sync_directory();
    }

    void publish(const std::vector<std::shared_ptr<const StoredResponse>>& ordered) {
        decltype(nodes) live_nodes;
        decltype(records) live_records;
        Json files = Json::array();
        for (const auto& record : ordered) {
            std::vector<ResponseContext> chain;
            for (auto node = record->context; node && !live_nodes.contains(node.get());
                 node      = node->parent) {
                chain.push_back(node);
            }
            for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                const auto& node = *it;
                std::string filename;
                if (const auto known = nodes.find(node.get()); known != nodes.end()) {
                    filename = known->second;
                } else {
                    filename   = "context-" + std::to_string(next_file++) + ".json";
                    Json turns = Json::array();
                    for (const auto& turn : node->turns) { turns.push_back(encode_turn(turn)); }
                    write(filename,
                          Json{{"parent", node->parent ? live_nodes.at(node->parent.get()) : ""},
                               {"turns", turns}});
                }
                live_nodes.emplace(node.get(), std::move(filename));
            }
            std::string filename;
            if (const auto known = records.find(record.get()); known != records.end()) {
                filename = known->second;
            } else {
                filename = "response-" + std::to_string(next_file++) + ".json";
                write(
                    filename,
                    Json{{"id", record->id},
                         {"response", record->response},
                         {"input_items", record->input_items},
                         {"preserve_thinking", record->preserve_thinking},
                         {"context", record->context ? live_nodes.at(record->context.get()) : ""}});
            }
            live_records.emplace(record.get(), filename);
            files.push_back(std::move(filename));
        }
        // Payload data and directory entries must precede the manifest commit.
        sync_directory();
        write("manifest.json", Json{{"version", 1}, {"next_file", next_file}, {"records", files}});
        sync_directory();
        nodes   = std::move(live_nodes);
        records = std::move(live_records);
        // Garbage collection follows publication. Failure leaves harmless orphan
        // payloads, never an incomplete committed manifest, and is retried later.
        std::unordered_set<std::string> retained{"manifest.json", "store.lock"};
        for (const auto& [node, name] : nodes) {
            (void)node;
            retained.insert(name);
        }
        for (const auto& [record, name] : records) {
            (void)record;
            retained.insert(name);
        }
        std::error_code error;
        for (std::filesystem::directory_iterator it(directory, error), end; !error && it != end;
             it.increment(error)) {
            const auto name = it->path().filename().string();
            if (!retained.contains(name) &&
                (name.starts_with("context-") || name.starts_with("response-") ||
                 name == "manifest.json.tmp")) {
                std::error_code ignored;
                std::filesystem::remove(it->path(), ignored);
            }
        }
    }
};

ResponseStore::ResponseStore(std::size_t max_records, std::size_t max_bytes, std::string location)
    : max_records_(max_records), max_bytes_(max_bytes) {
    if (max_records_ == 0 || max_bytes_ == 0) {
        throw std::invalid_argument("response store limits must be positive");
    }
    if (!location.empty()) {
        disk_        = std::make_unique<DiskStore>(std::move(location));
        auto ordered = disk_->load();
        // Restart with smaller limits evicts oldest public IDs, preserving any
        // ancestry still owned by retained descendants.
        while (!ordered.empty() &&
               (ordered.size() > max_records_ || ordered_bytes(ordered) > max_bytes_)) {
            ordered.pop_back();
        }
        publish_locked(ordered);
    }
}

ResponseStore::~ResponseStore() = default;

void ResponseStore::publish_locked(
    const std::vector<std::shared_ptr<const StoredResponse>>& ordered) {
    // Allocate the replacement index before the disk commit. A failed payload or
    // manifest write leaves the live in-memory store unchanged.
    std::list<std::string> lru;
    std::unordered_map<std::string, Entry> records;
    for (const auto& record : ordered) {
        lru.push_back(record->id);
        records.emplace(record->id, Entry{record, std::prev(lru.end())});
    }
    const auto bytes = ordered_bytes(ordered);
    if (disk_) {
        if (disk_->failed) {
            disk_error("store unavailable after an I/O failure; restart required");
        }
        try {
            disk_->publish(ordered);
        } catch (...) {
            // A rename may have succeeded even when its final directory sync
            // failed. Do not serve divergent memory/disk histories afterward.
            disk_->failed = true;
            throw;
        }
    }
    records_.swap(records);
    lru_.swap(lru);
    current_bytes_ = bytes;
}

std::shared_ptr<const StoredResponse> ResponseStore::get(const std::string& id) {
    std::lock_guard lock(mutex_);
    if (disk_ && disk_->failed) {
        disk_error("store unavailable after an I/O failure; restart required");
    }
    const auto found = records_.find(id);
    if (found == records_.end()) { return {}; }
    const auto response = found->second.response;
    if (disk_ && found->second.lru != lru_.begin()) {
        std::vector<std::shared_ptr<const StoredResponse>> ordered{response};
        for (const auto& existing : lru_) {
            if (existing != id) { ordered.push_back(records_.at(existing).response); }
        }
        try {
            disk_->reorder(ordered);
        } catch (...) {
            disk_->failed = true;
            throw;
        }
    }
    lru_.splice(lru_.begin(), lru_, found->second.lru);
    return response;
}

void ResponseStore::put(StoredResponse response) {
    if (response.id.empty() || !response.response.is_object()) {
        throw std::invalid_argument("stored response must have an id and object body");
    }
    if (standalone_bytes(response) > max_bytes_) { throw_store_capacity(); }
    auto owned = std::make_shared<const StoredResponse>(std::move(response));
    std::lock_guard lock(mutex_);
    if (records_.contains(owned->id)) {
        throw std::logic_error("duplicate response id in response store");
    }
    std::vector<std::shared_ptr<const StoredResponse>> ordered{owned};
    for (const auto& id : lru_) { ordered.push_back(records_.at(id).response); }
    while (ordered.size() > max_records_ || ordered_bytes(ordered) > max_bytes_) {
        ordered.pop_back();
    }
    publish_locked(ordered);
}

bool ResponseStore::erase(const std::string& id) {
    std::lock_guard lock(mutex_);
    if (disk_ && disk_->failed) {
        disk_error("store unavailable after an I/O failure; restart required");
    }
    if (!records_.contains(id)) { return false; }
    std::vector<std::shared_ptr<const StoredResponse>> ordered;
    for (const auto& existing : lru_) {
        if (existing != id) { ordered.push_back(records_.at(existing).response); }
    }
    publish_locked(ordered);
    return true;
}

std::size_t ResponseStore::size() const {
    std::lock_guard lock(mutex_);
    return records_.size();
}

std::size_t ResponseStore::bytes() const {
    std::lock_guard lock(mutex_);
    return current_bytes_;
}

} // namespace ninfer::serve
