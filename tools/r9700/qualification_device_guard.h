#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::r9700_qualification {

// This is a physical-target receipt for local baseline measurements, not a device-discovery
// policy. Construct before the first HIP call, then select exactly the bound PCI function.
class DeviceGuard {
public:
    DeviceGuard() {
        const char* requested = std::getenv("NINFER_R9700_PCI_BUS_ID");
        pci_                  = requested == nullptr ? "0000:13:00.0" : requested;
        if (pci_.size() != 12U || pci_[4] != ':' || pci_[7] != ':' || pci_[10] != '.')
            throw std::runtime_error("NINFER_R9700_PCI_BUS_ID must be domain:bus:device.function");
        for (std::size_t i = 0; i < pci_.size(); ++i) {
            if (i == 4U || i == 7U || i == 10U) continue;
            const auto ch = static_cast<unsigned char>(pci_[i]);
            if (!std::isxdigit(ch)) throw std::runtime_error("invalid R9700 PCI address");
            pci_[i] = static_cast<char>(std::tolower(ch));
        }
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
