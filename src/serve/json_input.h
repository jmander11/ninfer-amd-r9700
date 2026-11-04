#pragma once

#include <nlohmann/json.hpp>

#include <string_view>

namespace ninfer::serve {

// Preserve member order and validate schema numbers before forwarding loses their spelling.
nlohmann::ordered_json parse_request_body(std::string_view source);

} // namespace ninfer::serve
