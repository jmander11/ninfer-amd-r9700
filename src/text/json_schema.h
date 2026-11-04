#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::text {

enum class SchemaErrorKind { InvalidJsonSchema, UnsupportedJsonSchema, UnsatisfiableJsonSchema };

class SchemaError final : public std::invalid_argument {
public:
    SchemaError(SchemaErrorKind kind, const std::string& message, std::string pointer = {})
        : std::invalid_argument(message), kind_(kind), pointer_(std::move(pointer)) {}

    [[nodiscard]] SchemaErrorKind kind() const noexcept { return kind_; }

    [[nodiscard]] const std::string& pointer() const noexcept { return pointer_; }

private:
    SchemaErrorKind kind_;
    std::string pointer_;
};

struct PreparedJsonSchema {
    std::string json;
    // Generated schema locations mapped to the original source. Longest matching prefix wins.
    std::vector<std::pair<std::string, std::string>> origins;
    [[nodiscard]] std::string source_pointer(std::string_view generated) const;
};

// Validate and reduce the admitted JSON Schema dialect, preserving source locations.
// Model framing and tool-schema relaxation belong to the caller.
PreparedJsonSchema prepare_json_schema(std::string_view source, bool json_object_keys = true);

} // namespace ninfer::text
