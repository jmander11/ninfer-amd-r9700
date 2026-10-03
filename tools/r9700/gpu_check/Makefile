# Builds the R9700 device-check tools: the LD_PRELOAD shim, the LDS race instrumenter, and the
# race-check device runtime bitcode. scripts/gpu-check.sh runs this before configuring a build.

ROCM ?= /opt/rocm
LLVM := $(ROCM)/lib/llvm
HIPCC := $(LLVM)/bin/clang++
CXX ?= c++
BUILD := build

all: $(BUILD)/libninfer_gpu_check.so $(BUILD)/ninfer_lds_race_instrument $(BUILD)/lds_race_rt.bc

$(BUILD):
	mkdir -p $@

$(BUILD)/libninfer_gpu_check.so: gpu_check_shim.cpp lds_race_layout.h | $(BUILD)
	$(CXX) -std=c++20 -O2 -Wall -Wextra -fPIC -shared -D__HIP_PLATFORM_AMD__ -I$(ROCM)/include $< \
	  -o $@ -L$(ROCM)/lib -lamdhip64 -ldl -Wl,-rpath,$(ROCM)/lib

$(BUILD)/ninfer_lds_race_instrument: lds_race_instrument.cpp lds_race_layout.h | $(BUILD)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -I$(LLVM)/include $< -o $@ -L$(LLVM)/lib -lLLVM-23git \
	  -Wl,-rpath,$(LLVM)/lib

$(BUILD)/lds_race_rt.bc: lds_race_rt.hip lds_race_layout.h | $(BUILD)
	$(HIPCC) -std=c++20 -O3 -x hip --offload-arch=gfx1201 --cuda-device-only -fgpu-rdc -emit-llvm -c \
	  $< -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
