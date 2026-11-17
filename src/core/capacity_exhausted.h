#pragma once

#include <new>
#include <stdexcept>
#include <string>

namespace ninfer {

// A bounded pool or arena refused a reservation. It is a std::bad_alloc, so existing handlers
// keep their meaning, and what() names the request and the capacity that refused it.
class CapacityExhausted final : public std::bad_alloc {
public:
    explicit CapacityExhausted(const std::string& message) : message_(message) {}

    [[nodiscard]] const char* what() const noexcept override { return message_.what(); }

private:
    std::runtime_error message_; // nothrow-copyable message storage
};

} // namespace ninfer
