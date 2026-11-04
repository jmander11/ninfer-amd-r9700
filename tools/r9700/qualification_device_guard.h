#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::r9700_qualification {

inline std::string configured_pci_bus_id() {
    const char* value = std::getenv("NINFER_R9700_PCI_BUS_ID");
    if (value == nullptr || value[0] == '\0')
        throw std::runtime_error("set NINFER_R9700_PCI_BUS_ID to the R9700 PCI address");
    const std::string pci(value);
    if (pci.size() != 12U || pci[4] != ':' || pci[7] != ':' || pci[10] != '.')
        throw std::runtime_error("NINFER_R9700_PCI_BUS_ID must be domain:bus:device.function");
    for (std::size_t i = 0; i < pci.size(); ++i) {
        if (i == 4U || i == 7U || i == 10U) continue;
        if (!std::isxdigit(static_cast<unsigned char>(pci[i])))
            throw std::runtime_error("invalid R9700 PCI address");
    }
    return pci;
}

inline std::filesystem::path configured_pci_path() {
    return std::filesystem::path("/sys/bus/pci/devices") / configured_pci_bus_id();
}

// Bind measurements to the configured PCI device. Construct before the first HIP call,
// then select that device.
class DeviceGuard {
public:
    DeviceGuard() {
        pci_ = configured_pci_bus_id();
        std::transform(pci_.begin(), pci_.end(), pci_.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        directory_ = "/sys/bus/pci/devices/" + pci_ + '/';
        check_sysfs("before-HIP");
    }

    int select() {
        check_sysfs("before-HIP-select");
        hip_check(hipDeviceGetByPCIBusId(&ordinal_, pci_.c_str()), "resolve PCI device");
        hip_check(hipSetDevice(ordinal_), "select PCI device");
        check("selected-HIP");
        return ordinal_;
    }

    void check(std::string_view phase) const {
        check_sysfs(phase);
        int selected = -1;
        hip_check(hipGetDevice(&selected), "read selected HIP device");
        if (selected != ordinal_) throw std::runtime_error("qualification HIP device changed");
        char pci[32]{};
        hipDeviceProp_t props{};
        hip_check(hipDeviceGetPCIBusId(pci, sizeof(pci), selected), "read HIP PCI address");
        hip_check(hipGetDeviceProperties(&props, selected), "read HIP properties");
        std::string reported = pci;
        std::transform(reported.begin(), reported.end(), reported.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const std::string_view architecture = props.gcnArchName;
        if (reported != pci_ || props.warpSize != 32 ||
            !(architecture == "gfx1201" || architecture.starts_with("gfx1201:")))
            throw std::runtime_error("qualification requires bound PCI gfx1201 wave32 device");
        std::cerr << "r9700-qualification phase=" << phase << " ordinal=" << selected
                  << " pci=" << reported << " arch=" << architecture << " wave=" << props.warpSize
                  << " name=" << props.name << '\n';
    }

private:
    static void hip_check(hipError_t error, const char* operation) {
        if (error != hipSuccess)
            throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
    }

    std::string read(const char* name) const {
        std::ifstream file(directory_ + name);
        std::string value;
        if (!(file >> value))
            throw std::runtime_error("cannot read R9700 qualification attribute " + directory_ +
                                     name);
        return value;
    }

    void check_sysfs(std::string_view phase) const {
        const auto vendor = read("vendor"), device = read("device");
        const auto power = read("power_dpm_force_performance_level");
        if (vendor != "0x1002" || device != "0x7551")
            throw std::runtime_error("bound PCI device is not AMD 1002:7551");
        if (power != "auto")
            throw std::runtime_error(
                "R9700 qualification requires auto power before and after timing; got " + power);
        std::cerr << "r9700-qualification phase=" << phase << " pci=" << pci_
                  << " vendor=" << vendor << " device=" << device << " power=" << power << '\n';
    }

    std::string pci_, directory_;
    int ordinal_ = -1;
};

inline DeviceGuard& device_guard() {
    static DeviceGuard guard;
    return guard;
}

} // namespace ninfer::r9700_qualification
