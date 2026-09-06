ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
HIPCC := /opt/rocm/bin/hipcc
BUILD := $(ROOT)/tools/r9700/build
QUALIFIER := $(BUILD)/fp8_gate_up_small_t_qual
ASSEMBLY := $(BUILD)/fp8_gate_up_small_t.s
COMMON := -O3 -std=c++20 --offload-arch=gfx1201 -I$(ROOT)/src -I$(ROOT)/include -I$(ROOT) -Wall -Wextra -Werror
DEVICE_ONLY := $(COMMON) -Wno-unused-command-line-argument

.PHONY: fp8_gate_up_small_t_build fp8_gate_up_small_t_static fp8_gate_up_small_t_regression

fp8_gate_up_small_t_build: $(QUALIFIER)

$(QUALIFIER): $(ROOT)/tools/r9700/fp8_gate_up_small_t_harness.hip \
              $(ROOT)/tools/r9700/fp8_gate_up_small_t_qual.hip \
              $(ROOT)/tools/r9700/fp8_gate_up_small_t_qual.h \
              $(ROOT)/src/ops/r9700/linear/fp8_activation.hip \
              $(ROOT)/src/ops/r9700/linear/linear_execution.hip
	mkdir -p $(BUILD)
	$(HIPCC) $(COMMON) \
	  $(ROOT)/tools/r9700/fp8_gate_up_small_t_harness.hip \
	  $(ROOT)/tools/r9700/fp8_gate_up_small_t_qual.hip \
	  $(ROOT)/src/ops/r9700/linear/fp8_activation.hip \
	  $(ROOT)/src/ops/r9700/linear/linear_execution.hip \
	  -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lhipblaslt -o $@

$(ASSEMBLY): $(ROOT)/tools/r9700/fp8_gate_up_small_t_qual.hip \
             $(ROOT)/tools/r9700/fp8_gate_up_small_t_qual.h
	mkdir -p $(BUILD)
	$(HIPCC) $(DEVICE_ONLY) -S --cuda-device-only $< -o $@

fp8_gate_up_small_t_static: $(ASSEMBLY)
	python3 $(ROOT)/tools/r9700/check_fp8_gate_up_small_t_static.py $(ASSEMBLY)

# Physical R9700 execution. This target is intentionally never a production route.
fp8_gate_up_small_t_regression: $(QUALIFIER) fp8_gate_up_small_t_static
	$(QUALIFIER) --regression
