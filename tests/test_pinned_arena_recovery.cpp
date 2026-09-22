#include "core/arena.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>

namespace {
thread_local int allocation_budget = -1;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

void* operator new(std::size_t bytes) {
    if (allocation_budget == 0) throw std::bad_alloc();
    if (allocation_budget > 0) --allocation_budget;
    if (void* p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

int main() {
    try {
        unsigned failures_exercised = 0;
        for (int fail_after = 0; fail_after < 5; ++fail_after) {
            ninfer::HostPinnedArena arena(16384);
            allocation_budget = fail_after;
            void* block = nullptr;
            try { block = arena.try_alloc(256); }
            catch (const std::bad_alloc&) { ++failures_exercised; }
            allocation_budget = -1;
            if (block) arena.free(block);
            require(arena.used() == 0, "failed allocation changed arena accounting");
            void* whole = arena.try_alloc(arena.capacity());
            require(whole == arena.base(), "failed allocation lost a free span");
            arena.free(whole);
        }
        require(failures_exercised >= 2, "metadata allocation failures were not exercised");

        ninfer::HostPinnedArena arena(64 * 256);
        std::array<void*, 64> blocks{};
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            blocks[i] = arena.try_alloc(256);
            require(blocks[i] != nullptr, "dense allocation failed");
            std::memset(blocks[i], static_cast<int>(i), 256);
        }
        allocation_budget = 0;
        for (std::size_t i = 1; i < blocks.size(); i += 2) arena.free(blocks[i]);
        allocation_budget = -1;
        for (std::size_t i = 0; i < blocks.size(); i += 2) {
            const auto* bytes = static_cast<const unsigned char*>(blocks[i]);
            for (unsigned j = 0; j < 256; ++j)
                require(bytes[j] == i, "fragmented free corrupted live allocation");
        }
        allocation_budget = 0;
        for (std::size_t i = 0; i < blocks.size(); i += 2) arena.free(blocks[i]);
        allocation_budget = -1;
        void* whole = arena.try_alloc(arena.capacity());
        require(whole == arena.base(), "fragmented arena did not fully recover");
        arena.free(whole);
        std::cout << "pinned arena allocation recovery and allocation-free release: PASS\n";
    } catch (const std::exception& error) {
        allocation_budget = -1;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
