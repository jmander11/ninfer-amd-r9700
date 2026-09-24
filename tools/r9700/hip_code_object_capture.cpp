#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>

#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <algorithm>

namespace {

std::size_t elf_size(const void* image) noexcept {
    if (image == nullptr) return 0;
    const auto* bytes = static_cast<const std::uint8_t*>(image);
    if (std::memcmp(bytes, ELFMAG, SELFMAG) != 0 || bytes[EI_CLASS] != ELFCLASS64 ||
        bytes[EI_DATA] != ELFDATA2LSB) return 0;
    const auto* header = reinterpret_cast<const Elf64_Ehdr*>(bytes);
    if (header->e_ehsize != sizeof(Elf64_Ehdr) || header->e_shentsize != sizeof(Elf64_Shdr) ||
        header->e_shnum == 0 || header->e_shoff > (1ULL << 32)) return 0;
    std::size_t end = header->e_shoff + header->e_shnum * sizeof(Elf64_Shdr);
    const auto* sections = reinterpret_cast<const Elf64_Shdr*>(bytes + header->e_shoff);
    for (std::size_t index = 0; index < header->e_shnum; ++index) {
        if (sections[index].sh_type == SHT_NOBITS) continue;
        if (sections[index].sh_offset > std::numeric_limits<std::size_t>::max() -
                                             sections[index].sh_size) return 0;
        end = std::max(end, static_cast<std::size_t>(sections[index].sh_offset +
                                                     sections[index].sh_size));
    }
    return end;
}

void capture(const void* image, std::size_t supplied_bytes = 0) noexcept {
    const char* directory = std::getenv("NINFER_CODE_OBJECT_CAPTURE_DIR");
    const std::size_t parsed_bytes = elf_size(image);
    const std::size_t bytes = supplied_bytes != 0 ? supplied_bytes : parsed_bytes;
    if (parsed_bytes == 0 || (supplied_bytes != 0 && parsed_bytes > supplied_bytes)) return;
    if (directory == nullptr || directory[0] != '/' || bytes == 0) return;
    (void)::mkdir(directory, 0755);
    char name[4096];
    const int length = std::snprintf(name, sizeof(name), "%s/%ld-0x%lx.co", directory,
                                     static_cast<long>(::getpid()),
                                     static_cast<unsigned long>(reinterpret_cast<std::uintptr_t>(image)));
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(name)) return;
    const int descriptor = ::open(name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor < 0) return;
    std::size_t offset = 0;
    while (offset < bytes) {
        const ssize_t written = ::write(descriptor,
            static_cast<const std::uint8_t*>(image) + offset, bytes - offset);
        if (written <= 0) break;
        offset += static_cast<std::size_t>(written);
    }
    (void)::fsync(descriptor);
    (void)::close(descriptor);
    if (offset != bytes) (void)::unlink(name);
}

} // namespace

extern "C" hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
    using Function = hipError_t (*)(hipModule_t*, const void*);
    static const auto next = reinterpret_cast<Function>(::dlsym(RTLD_NEXT, "hipModuleLoadData"));
    if (next == nullptr) return hipErrorSharedObjectSymbolNotFound;
    capture(image);
    return next(module, image);
}

extern "C" hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                            unsigned int count, hipJitOption* options,
                                            void** option_values) {
    using Function = hipError_t (*)(hipModule_t*, const void*, unsigned int, hipJitOption*, void**);
    static const auto next = reinterpret_cast<Function>(::dlsym(RTLD_NEXT, "hipModuleLoadDataEx"));
    if (next == nullptr) return hipErrorSharedObjectSymbolNotFound;
    capture(image);
    return next(module, image, count, options, option_values);
}

extern "C" hsa_status_t hsa_code_object_reader_create_from_memory(
    const void* code_object, size_t size, hsa_code_object_reader_t* reader) {
    using Function = hsa_status_t (*)(const void*, size_t, hsa_code_object_reader_t*);
    static const auto next = reinterpret_cast<Function>(
        ::dlsym(RTLD_NEXT, "hsa_code_object_reader_create_from_memory"));
    if (next == nullptr) return HSA_STATUS_ERROR;
    capture(code_object, size);
    return next(code_object, size, reader);
}
