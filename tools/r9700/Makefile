HIPCC ?= /opt/rocm/bin/hipcc
HIPCC_PATH := $(shell command -v $(HIPCC))
ifeq ($(HIPCC_PATH),)
$(error HIPCC '$(HIPCC)' was not found)
endif
ROCM_ROOT := $(realpath $(dir $(HIPCC_PATH))/..)
ROCPROFV3 := $(ROCM_ROOT)/bin/rocprofv3
PYTHON ?= python3
ARCH := gfx1201
BUILD_DIR := build
PROFILE_DIR ?= ../../profiles/r9700
ROCM_AUDIT_JSON ?= ../../profiles/r9700/rocm10-tool-capability-audit.json
ROCM_AUDIT := audit_rocm_tools.py
ROCM_AUDIT_TEST := test_audit_rocm_tools.py
TRACE_EVIDENCE_DB ?= ../../profiles/r9700/rocm10-final-trace-page66/r2d2/510178_results.db
PMC_EVIDENCE_DB ?= ../../profiles/r9700/rocm10-final-pmc-page66/r2d2/509405_results.db
TRACE_EVIDENCE_COMMAND ?= ./build/kv_op_qual --context 66 --iterations 1
PMC_EVIDENCE_COMMAND ?= ./build/kv_op_qual --context 66 --iterations 1
VALU_EVIDENCE_DB ?= ../../profiles/r9700/rocm10-gfx12-pmc-probe/wave32-valu/r2d2/3329702_results.db
LDS_EVIDENCE_DB ?= ../../profiles/r9700/rocm10-gfx12-pmc-probe/wave32-lds/r2d2/3329827_results.db
TCP_EVIDENCE_DB ?= ../../profiles/r9700/rocm10-gfx12-pmc-probe/tcp/r2d2/3329949_results.db
GL2C_EVIDENCE_DB ?= ../../profiles/r9700/rocm10-gfx12-pmc-probe/gl2c/r2d2/3329684_results.db
GENERIC_VALU_EVIDENCE_DB ?= ../../profiles/r9700/rocm10-pmc-isolated-valu/r2d2/524501_results.db
Q4G64_LINEAR_JSON ?= ../../profiles/bench/r9700-a8q4g64-linear-qualification.json
Q4_HARDWARE_PEAK_JSON ?= ../../profiles/bench/r9700-q4-hardware-peak-probe.json
A8Q4_SHAPE_SWEEP_JSON ?= ../../profiles/bench/r9700-a8q4g64-artifact-shape-sweep.json
Q4_TENSOR_A4_JSON ?= ../../profiles/bench/r9700-q4-tensor-dispatch-a4.json
Q4_TENSOR_A8_JSON ?= ../../profiles/bench/r9700-q4-tensor-dispatch-a8.json
W8A8_WMMA_JSON ?= ../../profiles/bench/r9700-w8a8g32-linear-qualification.json
W8A8_TENSOR_JSON ?= ../../profiles/bench/r9700-w8a8g32-linear-tensor-dispatch.json
XATTENTION_STRIDE ?= 16
XATTENTION_TAU_PERMILLE ?= 900
XATTENTION_EVIDENCE_DIR ?= ../../profiles/bench/r9700-xattention-requal-s$(XATTENTION_STRIDE)-tau$(XATTENTION_TAU_PERMILLE)
PROBE_TARGET := $(BUILD_DIR)/r9700_qual
KV_TARGET := $(BUILD_DIR)/kv_op_qual
SPLIT512_TARGET := $(BUILD_DIR)/split512_attention_qual
DENSE_PREFILL_ATTENTION_TARGET := $(BUILD_DIR)/dense_prefill_attention_qual
HIP_DEVICE_TARGET := $(BUILD_DIR)/hip_device_qual
HIP_DEVICE_SANITIZE_TARGET := $(BUILD_DIR)/hip_device_qual_sanitize
LINEAR_TARGET := $(BUILD_DIR)/linear_op_qual
EAGER_TARGET := $(BUILD_DIR)/eager_op_qual
RMSNORM_PREFILL_TARGET := $(BUILD_DIR)/rmsnorm_prefill_qual
RMSNORM_K256_PREFILL_TARGET := $(BUILD_DIR)/rmsnorm_k256_prefill_qual
RMSNORM_DECODE_PRODUCTION_TARGET := $(BUILD_DIR)/rmsnorm_decode_production
RMSNORM_DECODE_ROUTING_TARGET := $(BUILD_DIR)/test_rmsnorm_decode_routing
GATED_RMSNORM_PREFILL_TARGET := $(BUILD_DIR)/gated_rmsnorm_prefill_qual
SILU_MUL_SPLIT_PREFILL_TARGET := $(BUILD_DIR)/silu_mul_split_prefill_qual
GDN_TARGET := $(BUILD_DIR)/gdn_op_qual
SAMPLING_TARGET := $(BUILD_DIR)/sampling_op_qual
ROPE_TARGET := $(BUILD_DIR)/rope_op_qual
GDN_RECURRENCE_TARGET := $(BUILD_DIR)/gdn_recurrence_qual
GDN_AFFINE_CHUNK_TARGET := $(BUILD_DIR)/gdn_affine_chunk_qual
GDN_GROUPED_HEADS_TARGET := $(BUILD_DIR)/gdn_grouped_heads_qual
SERVE_ENV_TARGET := $(BUILD_DIR)/serve_environment_qual
HBM_TARGET := $(BUILD_DIR)/hbm_bandwidth_probe
Q4_HARDWARE_PEAK_TARGET := $(BUILD_DIR)/q4_hardware_peak_probe
FP8_HIPBLASLT_TARGET := $(BUILD_DIR)/fp8_hipblaslt_probe
INT4_WMMA_TARGET := $(BUILD_DIR)/int4_wmma_qual
W8A8_WMMA_TARGET := $(BUILD_DIR)/w8a8_wmma_linear_qual
Q4G64_LINEAR_TARGET := $(BUILD_DIR)/q4g64_linear_qual
A8Q4_SHAPE_SWEEP_TARGET := $(BUILD_DIR)/a8q4_shape_sweep_qual
A8Q4_DFLASH_SMALL_T_TARGET := $(BUILD_DIR)/a8q4_dflash_small_t_qual
A8Q4_DFLASH_MLP_DOWN_SMALL_T_TARGET := $(BUILD_DIR)/a8q4_dflash_mlp_down_small_t_qual
DFLASH_SMALL_T_OUTPUT_DIR ?=
DFLASH_SMALL_T_C1_SUMMARY ?= $(abspath ../..)/profiles/bench/r9700-dflash-small-t-c1-20260906/summary.json
A4Q4_PREFILL_TARGET := $(BUILD_DIR)/a4q4_prefill_qual
Q4_TENSOR_A4_TARGET := $(BUILD_DIR)/q4_tensor_dispatch_a4_qual
Q4_TENSOR_A8_TARGET := $(BUILD_DIR)/q4_tensor_dispatch_a8_qual
PROBE_SOURCE := r9700_qual.hip
KV_SOURCE := kv_op_qual.hip
SPLIT512_SOURCE := split512_attention_qual.hip
SPLIT512_STATIC_CHECK := check_split512_static.py
DENSE_PREFILL_ATTENTION_SOURCE := dense_prefill_attention_qual.hip
DENSE_PREFILL_ATTENTION_STATIC_CHECK := check_dense_prefill_attention_static.py
HIP_DEVICE_SOURCE := hip_device_qual.hip
LINEAR_SOURCE := linear_op_qual.hip
EAGER_SOURCE := eager_op_qual.hip
RMSNORM_PREFILL_SOURCE := rmsnorm_prefill_qual.hip
RMSNORM_K256_PREFILL_SOURCE := rmsnorm_k256_prefill_qual.hip
RMSNORM_DECODE_PRODUCTION_SOURCE := rmsnorm_decode_production.hip
GATED_RMSNORM_PREFILL_SOURCE := gated_rmsnorm_prefill_qual.hip
SILU_MUL_SPLIT_PREFILL_SOURCE := silu_mul_split_prefill_qual.hip
GDN_SOURCE := gdn_op_qual.hip
SAMPLING_SOURCE := sampling_op_qual.hip
ROPE_SOURCE := rope_op_qual.hip
GDN_RECURRENCE_SOURCE := gdn_recurrence_qual.hip
GDN_AFFINE_CHUNK_SOURCE := gdn_affine_chunk_qual.hip
SERVE_ENV_SOURCE := serve_environment_qual.cpp
HBM_SOURCE := hbm_bandwidth_probe.hip
Q4_HARDWARE_PEAK_SOURCE := q4_hardware_peak_probe.hip
FP8_HIPBLASLT_SOURCE := fp8_hipblaslt_probe.hip
Q4_HARDWARE_PEAK_STATIC_CHECK := check_q4_hardware_peak_static.py
INT4_WMMA_SOURCE := int4_wmma_qual.hip
W8A8_WMMA_SOURCE := w8a8_wmma_linear_qual.hip
Q4G64_LINEAR_SOURCE := q4g64_linear_qual.hip
A8Q4_SHAPE_SWEEP_SOURCE := a8q4_shape_sweep_qual.hip
A8Q4_DFLASH_SMALL_T_SOURCE := a8q4_dflash_small_t_qual.hip
A8Q4_DFLASH_MLP_DOWN_SMALL_T_SOURCE := a8q4_dflash_mlp_down_small_t_qual.hip
A8Q4_DFLASH_MLP_DOWN_SMALL_T_STATIC_CHECK := check_a8q4_dflash_mlp_down_small_t_static.py
Q4_TENSOR_SOURCE := q4_tensor_dispatch_qual.hip
SERVE_LOG_IMPL := ../../src/serve/request_log.cpp
SERVE_LOG_HEADER := ../../src/serve/request_log.h
SERVE_OPTIONS_IMPL := ../../src/serve/serve_options.cpp
SERVE_OPTIONS_HEADER := ../../src/serve/serve_options.h
LINEAR_IMPL := ../../src/ops/r9700/linear/r9700_linear.hip
LINEAR_HEADER := ../../src/ops/r9700/linear/r9700_linear.h
LINEAR_TENSOR_IMPL := ../../src/ops/r9700/linear/linear_tensor_op.cpp
LINEAR_PUBLIC_HEADER := ../../include/ninfer/ops/linear.h
W8_ACTIVATION_PROFILE := ../../src/ops/r9700/linear/r9700_w8_activation_profile.h
Q4_ACTIVATION_PROFILE := ../../src/ops/r9700/linear/r9700_q4_activation_profile.h
EAGER_IMPL := ../../src/ops/r9700/eager/eager_ops.hip
EAGER_HEADER := ../../src/ops/r9700/eager/eager_ops.h
EAGER_TENSOR_IMPL := ../../src/ops/r9700/eager/eager_tensor_ops.cpp
MTP_PACK_HEADER := ../../include/ninfer/ops/mtp_pack.h
GDN_IMPL := ../../src/ops/r9700/gdn/gdn_ops.hip
GDN_HEADER := ../../src/ops/r9700/gdn/gdn_ops.h
GDN_TENSOR_IMPL := ../../src/ops/r9700/gdn/causal_conv1d_silu.cpp
GDN_PUBLIC_HEADER := ../../include/ninfer/ops/causal_conv1d_silu.h
SAMPLING_IMPL := ../../src/ops/r9700/sampling/sampling.hip
SAMPLING_HEADER := ../../include/ninfer/ops/sampling.h
ROPE_IMPL := ../../src/ops/r9700/rope/rope.hip
ROPE_HEADER := ../../include/ninfer/ops/rope.h
GDN_RECURRENCE_IMPL := ../../src/ops/r9700/gdn_recurrence/gated_delta_net.hip
GDN_RECURRENCE_HEADER := ../../include/ninfer/ops/gated_delta_net.h
GDN_AFFINE_CHUNK_IMPL := ../../src/ops/r9700/gdn_recurrence/gdn_affine_chunk_qualification.hip
GDN_AFFINE_CHUNK_HEADER := ../../src/ops/r9700/gdn_recurrence/gdn_affine_chunk_qualification.h
GDN_GROUPED_HEADS_SOURCE := gdn_grouped_heads_qual.hip
GDN_GROUPED_HEADS_IMPL := ../../src/ops/r9700/gdn_recurrence/gdn_grouped_heads_qualification.hip
GDN_GROUPED_HEADS_HEADER := ../../src/ops/r9700/gdn_recurrence/gdn_grouped_heads_qualification.h
HIP_DEVICE_IMPL := ../../src/core/device.hip
HIP_ARENA_IMPL := ../../src/core/arena.hip
CORE_DTYPE_IMPL := ../../src/core/dtype.cpp
CORE_TENSOR_IMPL := ../../src/core/tensor.cpp
HIP_GRAPH_IMPL := ../../src/core/decode_graph.hip
KV_ORACLE := ../../tests/ops/fp8_int4_kv_oracle.h
KV_APPEND_SOURCE := ../../src/ops/r9700/kv/fp8_int4_kv_append.hip
KV_APPEND_HEADER := ../../src/ops/r9700/kv/fp8_int4_kv_append.h
KV_COMPACT_SOURCE := ../../src/ops/r9700/kv/fp8_int4_kv_compact.hip
KV_COMPACT_HEADER := ../../src/ops/r9700/kv/fp8_int4_kv_compact.h
KV_ATTENTION_SOURCE := ../../src/ops/r9700/kv/fp8_int4_kv_attention.hip
KV_ATTENTION_HEADER := ../../src/ops/r9700/kv/fp8_int4_kv_attention.h
XATTENTION_SOURCE := ../../src/ops/r9700/kv/fp8_int4_kv_xattention.hip
XATTENTION_HEADER := ../../src/ops/r9700/kv/fp8_int4_kv_xattention.h
XATTENTION_PROFILE := ../../src/ops/r9700/kv/r9700_xattention_profile.h
GXX_VERSION := $(shell g++ -dumpversion)
MULTIARCH := $(shell gcc -print-multiarch)
CLANG_RESOURCE := $(shell $(HIPCC) --print-resource-dir)

# This host still has stale distro HIP headers in /usr/include. Keep the selected ROCm release's
# headers authoritative while retaining the ordinary host C/C++ search roots.
SYSTEM_INCLUDES := -nostdinc \
	-isystem $(ROCM_ROOT)/include \
	-isystem $(CLANG_RESOURCE)/include \
	-isystem /usr/include/c++/$(GXX_VERSION) \
	-isystem /usr/include/$(MULTIARCH)/c++/$(GXX_VERSION) \
	-isystem /usr/include/c++/$(GXX_VERSION)/backward \
	-isystem /usr/local/include \
	-isystem /usr/include/$(MULTIARCH) \
	-isystem /usr/include
CXXFLAGS := -O3 -std=c++20 --offload-arch=$(ARCH) -I../../src $(SYSTEM_INCLUDES) -Wall -Wextra -Werror
LDFLAGS := -L$(ROCM_ROOT)/lib -Wl,-rpath,$(ROCM_ROOT)/lib


.PHONY: all clean hbm fp8-hipblaslt-probe int4-wmma int4-wmma-isa int4-wmma-resources w8a8-wmma w8a8-prefill-cta-regression w8a8-prefill-cta-lds-scope-regression w8a8-prefill-cta-lds-scope-benchmark w8a8-prefill-cta-lds-scope-report w8a8-wmma-isa w8a8-wmma-resources prefill-cta-static-q4 prefill-cta-static-w8 prefill-cta-static-incumbent-diagnostic q4g64-linear a8q4-shape-sweep a8q4-dflash-small-t-build a8q4-dflash-small-t-static a8q4-dflash-small-t-test a8q4-dflash-small-t-screen a8q4-dflash-mlp-down-small-t-build a8q4-dflash-mlp-down-small-t-static a8q4-dflash-mlp-down-small-t-static-test a8q4-prefill-cta-regression a8q4-prefill-cta-lds-scope-regression a8q4-prefill-cta-lds-scope-benchmark a8q4-prefill-cta-lds-scope-report a8q4-prefill-cta-m64n64-regression q4g64-linear-isa q4g64-linear-resources xattention-isa xattention-resources xattention-reports split512-attention split512-attention-isa split512-attention-resources dense-prefill-attention dense-prefill-attention-full-score-isa dense-prefill-attention-full-score-static dense-prefill-attention-static-test isa kv-resources linear-isa linear-resources eager-isa rmsnorm-prefill-static rmsnorm-prefill-static-test rmsnorm-prefill-benchmark rmsnorm-k256-prefill-static rmsnorm-k256-prefill-static-test rmsnorm-k256-prefill-benchmark rmsnorm-decode-production-static rmsnorm-decode-production-static-test rmsnorm-decode-production-routing-test rmsnorm-decode-production-regression gated-rmsnorm-prefill-static gated-rmsnorm-prefill-static-test gated-rmsnorm-prefill-path-test gated-rmsnorm-prefill-benchmark gated-rmsnorm-k128-production-static gated-rmsnorm-k128-production-static-test silu-mul-split-prefill-static silu-mul-split-prefill-static-test silu-mul-split-prefill-benchmark gdn-isa gdn-prefill-benchmark gdn-prefill-static gdn-prefill-static-test gdn-recurrence-resources sampling-isa rope-resources run sanitize rocm-tool-audit rocm-tool-audit-test profile-trace profile-attention-trace profile-pmc

all: $(PROBE_TARGET) $(KV_TARGET) $(HIP_DEVICE_TARGET) $(LINEAR_TARGET) $(EAGER_TARGET) $(GDN_TARGET) $(GDN_RECURRENCE_TARGET) $(SAMPLING_TARGET) $(ROPE_TARGET) $(SERVE_ENV_TARGET)

$(FP8_HIPBLASLT_TARGET): $(FP8_HIPBLASLT_SOURCE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $< $(LDFLAGS) -lhipblaslt -o $@

fp8-hipblaslt-probe: $(FP8_HIPBLASLT_TARGET)
	./$(FP8_HIPBLASLT_TARGET)

$(PROBE_TARGET): $(PROBE_SOURCE) $(KV_ORACLE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $< $(LDFLAGS) -o $@

$(KV_TARGET): $(KV_SOURCE) $(KV_APPEND_SOURCE) $(KV_APPEND_HEADER) $(KV_COMPACT_SOURCE) $(KV_COMPACT_HEADER) $(KV_ATTENTION_SOURCE) $(KV_ATTENTION_HEADER) $(KV_ORACLE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(KV_SOURCE) $(KV_APPEND_SOURCE) $(KV_COMPACT_SOURCE) $(KV_ATTENTION_SOURCE) $(LDFLAGS) -o $@

$(SPLIT512_TARGET): $(SPLIT512_SOURCE) $(KV_ATTENTION_SOURCE) $(KV_ATTENTION_HEADER) $(KV_ORACLE) \
		$(HIP_DEVICE_IMPL) $(HIP_GRAPH_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include $(SPLIT512_SOURCE) $(KV_ATTENTION_SOURCE) \
		$(HIP_DEVICE_IMPL) $(HIP_GRAPH_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) \
		$(CORE_TENSOR_IMPL) $(LDFLAGS) -o $@

split512-attention: $(SPLIT512_TARGET)
	./$(SPLIT512_TARGET)

$(DENSE_PREFILL_ATTENTION_TARGET): $(DENSE_PREFILL_ATTENTION_SOURCE) $(KV_ATTENTION_SOURCE) \
		$(KV_ATTENTION_HEADER) $(KV_ORACLE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -DNINFER_SOURCE_DIR='"$(abspath ../..)"' \
		$(DENSE_PREFILL_ATTENTION_SOURCE) $(KV_ATTENTION_SOURCE) \
		$(LDFLAGS) -o $@

dense-prefill-attention: $(DENSE_PREFILL_ATTENTION_TARGET)
	./$(DENSE_PREFILL_ATTENTION_TARGET)

$(BUILD_DIR)/dense_prefill_attention.s: $(KV_ATTENTION_SOURCE) $(KV_ATTENTION_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S \
		$(KV_ATTENTION_SOURCE) -o $@

dense-prefill-attention-full-score-isa: $(BUILD_DIR)/dense_prefill_attention.s
	awk '/-- Begin function/ { selected = ($$0 ~ /dense_full_score_(qk(_bk32)?|maximum|pv)_kernel/) } selected && $$0 ~ /-- Begin function|v_wmma_|v_exp_|v_fma|amdhsa_group_segment_fixed_size|amdhsa_private_segment_fixed_size|amdhsa_next_free_vgpr|ScratchSize|Occupancy/' $<

# Codegen selects the exact unrolled WMMA count and final resource envelope. Bind those explicit
# observations rather than guessing them before the first gfx1201 compilation.
dense-prefill-attention-full-score-static:
	test -n "$(DENSE_ATTN_ASSEMBLY)" && test -n "$(DENSE_ATTN_METADATA)" && \
		test -n "$(DENSE_ATTN_SYMBOL)" && test -n "$(DENSE_ATTN_VALUE_GROUP)" && \
		test -n "$(DENSE_ATTN_STAGE)"
	python3 $(DENSE_PREFILL_ATTENTION_STATIC_CHECK) --full-score-stage "$(DENSE_ATTN_STAGE)" \
		--assembly "$(DENSE_ATTN_ASSEMBLY)" --metadata "$(DENSE_ATTN_METADATA)" \
		--symbol "$(DENSE_ATTN_SYMBOL)" --value-group "$(DENSE_ATTN_VALUE_GROUP)"

dense-prefill-attention-static-test:
	cd ../.. && $(PYTHON) -m unittest tools.r9700.test_check_dense_prefill_attention_static

$(HIP_DEVICE_TARGET): $(HIP_DEVICE_SOURCE) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(HIP_GRAPH_IMPL) ../../src/core/device.h ../../src/core/arena.h ../../src/core/decode_graph.h Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(HIP_DEVICE_SOURCE) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(HIP_GRAPH_IMPL) $(LDFLAGS) -o $@

$(HIP_DEVICE_SANITIZE_TARGET): $(HIP_DEVICE_SOURCE) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(HIP_GRAPH_IMPL) ../../src/core/device.h ../../src/core/arena.h ../../src/core/decode_graph.h Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(filter-out -O3,$(CXXFLAGS)) -O1 -g -fsanitize=address -fno-gpu-sanitize \
		$(HIP_DEVICE_SOURCE) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) \
		$(HIP_GRAPH_IMPL) $(LDFLAGS) -fsanitize=address -o $@

$(LINEAR_TARGET): $(LINEAR_SOURCE) $(LINEAR_IMPL) $(LINEAR_HEADER) $(LINEAR_TENSOR_IMPL) \
	$(LINEAR_PUBLIC_HEADER) $(HIP_DEVICE_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include $(LINEAR_SOURCE) $(LINEAR_IMPL) $(LINEAR_TENSOR_IMPL) \
		$(HIP_DEVICE_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(LDFLAGS) -o $@

$(EAGER_TARGET): $(EAGER_SOURCE) $(EAGER_IMPL) $(EAGER_HEADER) $(EAGER_TENSOR_IMPL) $(MTP_PACK_HEADER) \
	$(HIP_DEVICE_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include $(EAGER_SOURCE) $(EAGER_IMPL) $(EAGER_TENSOR_IMPL) \
		$(HIP_DEVICE_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(LDFLAGS) -o $@

$(RMSNORM_PREFILL_TARGET): $(RMSNORM_PREFILL_SOURCE) $(EAGER_IMPL) $(EAGER_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(RMSNORM_PREFILL_SOURCE) $(EAGER_IMPL) $(LDFLAGS) -o $@

$(RMSNORM_K256_PREFILL_TARGET): $(RMSNORM_K256_PREFILL_SOURCE) $(EAGER_IMPL) $(EAGER_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(RMSNORM_K256_PREFILL_SOURCE) $(EAGER_IMPL) $(LDFLAGS) -o $@

$(RMSNORM_DECODE_PRODUCTION_TARGET): $(RMSNORM_DECODE_PRODUCTION_SOURCE) $(EAGER_IMPL) $(EAGER_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(RMSNORM_DECODE_PRODUCTION_SOURCE) $(EAGER_IMPL) $(LDFLAGS) -o $@

$(RMSNORM_DECODE_ROUTING_TARGET): test_rmsnorm_decode_routing.cpp $(EAGER_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) test_rmsnorm_decode_routing.cpp $(LDFLAGS) -o $@

$(GATED_RMSNORM_PREFILL_TARGET): $(GATED_RMSNORM_PREFILL_SOURCE) $(EAGER_IMPL) $(EAGER_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(GATED_RMSNORM_PREFILL_SOURCE) $(EAGER_IMPL) $(LDFLAGS) -o $@

$(SILU_MUL_SPLIT_PREFILL_TARGET): $(SILU_MUL_SPLIT_PREFILL_SOURCE) $(EAGER_IMPL) $(EAGER_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(SILU_MUL_SPLIT_PREFILL_SOURCE) $(EAGER_IMPL) $(LDFLAGS) -o $@

$(GDN_TARGET): $(GDN_SOURCE) $(GDN_IMPL) $(GDN_HEADER) $(GDN_TENSOR_IMPL) $(GDN_PUBLIC_HEADER) \
	$(HIP_DEVICE_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include $(GDN_SOURCE) $(GDN_IMPL) $(GDN_TENSOR_IMPL) \
		$(HIP_DEVICE_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(LDFLAGS) -o $@

$(SAMPLING_TARGET): $(SAMPLING_SOURCE) $(SAMPLING_IMPL) $(SAMPLING_HEADER) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include $(SAMPLING_SOURCE) $(SAMPLING_IMPL) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(LDFLAGS) -o $@

$(ROPE_TARGET): $(ROPE_SOURCE) $(ROPE_IMPL) $(ROPE_HEADER) $(HIP_DEVICE_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include $(ROPE_SOURCE) $(ROPE_IMPL) $(HIP_DEVICE_IMPL) \
		$(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(LDFLAGS) -o $@

$(GDN_RECURRENCE_TARGET): $(GDN_RECURRENCE_SOURCE) $(GDN_RECURRENCE_IMPL) $(GDN_RECURRENCE_HEADER) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include $(GDN_RECURRENCE_SOURCE) $(GDN_RECURRENCE_IMPL) \
		$(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(LDFLAGS) -o $@

$(GDN_AFFINE_CHUNK_TARGET): $(GDN_AFFINE_CHUNK_SOURCE) $(GDN_AFFINE_CHUNK_IMPL) $(GDN_AFFINE_CHUNK_HEADER) $(GDN_RECURRENCE_IMPL) $(GDN_RECURRENCE_HEADER) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include $(GDN_AFFINE_CHUNK_SOURCE) $(GDN_AFFINE_CHUNK_IMPL) \
		$(GDN_RECURRENCE_IMPL) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) \
		$(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(LDFLAGS) -o $@

$(SERVE_ENV_TARGET): $(SERVE_ENV_SOURCE) $(SERVE_LOG_IMPL) $(SERVE_LOG_HEADER) $(SERVE_OPTIONS_IMPL) $(SERVE_OPTIONS_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -ffunction-sections -fdata-sections -I../../include -I../../third_party \
		-DNINFER_R9700_KV_VALUE_GROUP=16 \
		$(SERVE_ENV_SOURCE) $(SERVE_LOG_IMPL) $(SERVE_OPTIONS_IMPL) $(LDFLAGS) -Wl,--gc-sections -o $@

$(HBM_TARGET): $(HBM_SOURCE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(HBM_SOURCE) $(LDFLAGS) -o $@

hbm: $(HBM_TARGET)

.PHONY: q4-hardware-peak q4-hardware-peak-static q4-hardware-peak-run

$(Q4_HARDWARE_PEAK_TARGET): $(Q4_HARDWARE_PEAK_SOURCE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(Q4_HARDWARE_PEAK_SOURCE) $(LDFLAGS) -o $@

$(BUILD_DIR)/q4_hardware_peak_probe.s: $(Q4_HARDWARE_PEAK_SOURCE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S $< -o $@

q4-hardware-peak: $(Q4_HARDWARE_PEAK_TARGET) q4-hardware-peak-static

q4-hardware-peak-static: $(BUILD_DIR)/q4_hardware_peak_probe.s $(Q4_HARDWARE_PEAK_STATIC_CHECK)
	$(PYTHON) $(Q4_HARDWARE_PEAK_STATIC_CHECK) $(BUILD_DIR)/q4_hardware_peak_probe.s

q4-hardware-peak-run: q4-hardware-peak
	test -n "$(Q4_HARDWARE_PEAK_JSON)"
	mkdir -p $(dir $(Q4_HARDWARE_PEAK_JSON))
	$(Q4_HARDWARE_PEAK_TARGET) --out-json "$(Q4_HARDWARE_PEAK_JSON)"

$(INT4_WMMA_TARGET): $(INT4_WMMA_SOURCE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(INT4_WMMA_SOURCE) $(LDFLAGS) -o $@

int4-wmma: $(INT4_WMMA_TARGET)
	./$(INT4_WMMA_TARGET)

$(BUILD_DIR)/int4_wmma_qual.s: $(INT4_WMMA_SOURCE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S $< -o $@

int4-wmma-isa: $(BUILD_DIR)/int4_wmma_qual.s
	grep -E "v_wmma_i32_16x16x(16|32)_iu4" $<

int4-wmma-resources: $(BUILD_DIR)/int4_wmma_qual.s
	awk '/-- Begin function/ { selected = ($$0 ~ /int4_wmma(16|32)_kernel|int4_vector_dot_kernel/) } selected && $$0 ~ /-- Begin function|v_wmma_i32_16x16x(16|32)_iu4|amdhsa_group_segment_fixed_size|amdhsa_private_segment_fixed_size|amdhsa_next_free_vgpr|NumVgprs|ScratchSize|LDSByteSize|Occupancy/' $<

$(W8A8_WMMA_TARGET): $(W8A8_WMMA_SOURCE) $(LINEAR_IMPL) $(LINEAR_HEADER) \
	$(LINEAR_TENSOR_IMPL) $(LINEAR_PUBLIC_HEADER) $(W8_ACTIVATION_PROFILE) \
	$(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) \
	$(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include -DNINFER_R9700_W8_ACTIVATION_BITS=8 \
		-DNINFER_SOURCE_DIR='"$(abspath ../..)"' \
		$(W8A8_WMMA_SOURCE) $(LINEAR_IMPL) $(LINEAR_TENSOR_IMPL) $(HIP_DEVICE_IMPL) \
		$(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(LDFLAGS) -o $@

w8a8-wmma: $(W8A8_WMMA_TARGET)
	mkdir -p $(dir $(W8A8_WMMA_JSON))
	$(W8A8_WMMA_TARGET) --out-json $(W8A8_WMMA_JSON)

w8a8-wmma-tensor: $(W8A8_WMMA_TARGET)
	mkdir -p $(dir $(W8A8_TENSOR_JSON))
	$(W8A8_WMMA_TARGET) --tensor-only --out-json $(W8A8_TENSOR_JSON)

w8a8-prefill-cta-regression: $(W8A8_WMMA_TARGET) w8a8-wmma-isa w8a8-wmma-resources
	$(W8A8_WMMA_TARGET) --prefill-cta-regression

w8a8-prefill-cta-lds-scope-regression: $(W8A8_WMMA_TARGET)
	$(W8A8_WMMA_TARGET) --prefill-cta-lds-scope-regression

w8a8-prefill-cta-lds-scope-benchmark: $(W8A8_WMMA_TARGET)
	test -n "$(PREFILL_CTA_LDS_SCOPE_OUT)"
	$(W8A8_WMMA_TARGET) --prefill-cta-lds-scope-benchmark \
		--out-json "$(PREFILL_CTA_LDS_SCOPE_OUT)"

w8a8-prefill-cta-lds-scope-report:
	test -n "$(PREFILL_CTA_LDS_SCOPE_OUT)"
	python3 check_prefill_cta_lds_scope_report.py w8 \
		--executable "$(abspath $(W8A8_WMMA_TARGET))" "$(PREFILL_CTA_LDS_SCOPE_OUT)"

$(BUILD_DIR)/w8a8_wmma_linear_qual.s: $(LINEAR_IMPL) $(LINEAR_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S $(LINEAR_IMPL) -o $@

w8a8-wmma-isa: $(BUILD_DIR)/w8a8_wmma_linear_qual.s
	grep -E "a8w8g32_linear_(wmma32|prefill_cta)_kernel|v_wmma_i32_16x16x16_iu8|v_cvt_pk_i16_f32|v_cvt_i32_f32" $<

w8a8-wmma-resources: $(BUILD_DIR)/w8a8_wmma_linear_qual.s
	awk '/-- Begin function/ { selected = ($$0 ~ /a8g32_quantize_activation_kernel|a8w8g32_linear_(wmma32|prefill_cta)_kernel/) } selected && $$0 ~ /-- Begin function|v_wmma_i32_16x16x16_iu8|amdhsa_group_segment_fixed_size|amdhsa_private_segment_fixed_size|amdhsa_next_free_vgpr|NumVgprs|ScratchSize|LDSByteSize|Occupancy/' $<

# Explicit inputs keep this admission gate independent of compilation and physical execution.
# A challenger must provide both variables, even when assembly and metadata are the same .s file.
prefill-cta-static-q4:
	test -n "$(PREFILL_CTA_ASSEMBLY)" && test -n "$(PREFILL_CTA_METADATA)"
	python3 check_prefill_cta_static.py --recipe q4 --mode lds-scope \
		--assembly "$(PREFILL_CTA_ASSEMBLY)" --metadata "$(PREFILL_CTA_METADATA)"

.PHONY: a8q4-decode-dot8-static a8q4-decode-dot8-regression
a8q4-decode-dot8-static: $(BUILD_DIR)/w8a8_wmma_linear_qual.s
	python3 check_prefill_cta_static.py --recipe q4-dot8 --mode decode-dot8 \
		--assembly "$<" --metadata "$<"

a8q4-decode-dot8-regression: $(A8Q4_SHAPE_SWEEP_TARGET) a8q4-decode-dot8-static
	$(A8Q4_SHAPE_SWEEP_TARGET) --decode-dot8-t1-regression

.PHONY: prefill-cta-static-q4-m128n128 a8q4-prefill-cta-m128n128-regression a8q4-prefill-cta-m128n128-benchmark a8q4-prefill-cta-m128n128-report a8q4-prefill-cta-pingpong-regression a8q4-prefill-cta-pingpong-benchmark a8q4-prefill-cta-pingpong-report prefill-cta-static-q4-a4-m64n128
prefill-cta-static-q4-m128n128:
	test -n "$(PREFILL_CTA_ASSEMBLY)" && test -n "$(PREFILL_CTA_METADATA)"
	python3 check_prefill_cta_static.py --recipe q4-m128n128 --mode m128n128 \
		--assembly "$(PREFILL_CTA_ASSEMBLY)" --metadata "$(PREFILL_CTA_METADATA)"

prefill-cta-static-q4-a4-m64n128:
	test -n "$(PREFILL_CTA_ASSEMBLY)" && test -n "$(PREFILL_CTA_METADATA)"
	python3 check_prefill_cta_static.py --recipe q4-a4-m64n128 --mode a4-m64n128 \
		--assembly "$(PREFILL_CTA_ASSEMBLY)" --metadata "$(PREFILL_CTA_METADATA)"

prefill-cta-static-w8:
	test -n "$(PREFILL_CTA_ASSEMBLY)" && test -n "$(PREFILL_CTA_METADATA)"
	python3 check_prefill_cta_static.py --recipe w8 --mode lds-scope \
		--assembly "$(PREFILL_CTA_ASSEMBLY)" --metadata "$(PREFILL_CTA_METADATA)"

prefill-cta-static-incumbent-diagnostic:
	python3 check_prefill_cta_static.py --recipe q4 --mode incumbent-diagnostic \
		--assembly "$(BUILD_DIR)/w8a8_wmma_linear_qual.s" \
		--metadata "$(BUILD_DIR)/w8a8_wmma_linear_qual.s"
	python3 check_prefill_cta_static.py --recipe w8 --mode incumbent-diagnostic \
		--assembly "$(BUILD_DIR)/w8a8_wmma_linear_qual.s" \
		--metadata "$(BUILD_DIR)/w8a8_wmma_linear_qual.s"

$(Q4G64_LINEAR_TARGET): $(Q4G64_LINEAR_SOURCE) $(LINEAR_IMPL) $(LINEAR_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(Q4G64_LINEAR_SOURCE) $(LINEAR_IMPL) $(LDFLAGS) -o $@

$(A8Q4_SHAPE_SWEEP_TARGET): $(A8Q4_SHAPE_SWEEP_SOURCE) $(LINEAR_IMPL) $(LINEAR_HEADER) $(Q4_ACTIVATION_PROFILE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -DNINFER_SOURCE_DIR='"$(abspath ../..)"' \
		$(A8Q4_SHAPE_SWEEP_SOURCE) $(LINEAR_IMPL) $(LDFLAGS) -o $@

$(A8Q4_DFLASH_SMALL_T_TARGET): $(A8Q4_DFLASH_SMALL_T_SOURCE) $(A8Q4_SHAPE_SWEEP_SOURCE) \
		$(LINEAR_IMPL) $(LINEAR_HEADER) $(Q4_ACTIVATION_PROFILE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-function -DNINFER_SOURCE_DIR='"$(abspath ../..)"' \
		$(A8Q4_DFLASH_SMALL_T_SOURCE) \
		$(LINEAR_IMPL) $(LDFLAGS) -o $@

a8q4-dflash-small-t-build: $(A8Q4_DFLASH_SMALL_T_TARGET)

$(A8Q4_DFLASH_MLP_DOWN_SMALL_T_TARGET): $(A8Q4_DFLASH_MLP_DOWN_SMALL_T_SOURCE) \
		$(A8Q4_DFLASH_SMALL_T_SOURCE) $(A8Q4_SHAPE_SWEEP_SOURCE) $(LINEAR_IMPL) \
		$(LINEAR_HEADER) $(Q4_ACTIVATION_PROFILE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-function -DNINFER_SOURCE_DIR='"$(abspath ../..)"' \
		$(A8Q4_DFLASH_MLP_DOWN_SMALL_T_SOURCE) \
		$(LINEAR_IMPL) $(LDFLAGS) -o $@

a8q4-dflash-mlp-down-small-t-build: $(A8Q4_DFLASH_MLP_DOWN_SMALL_T_TARGET)

$(BUILD_DIR)/a8q4_dflash_mlp_down_small_t.s: $(A8Q4_DFLASH_MLP_DOWN_SMALL_T_SOURCE) \
		$(A8Q4_DFLASH_SMALL_T_SOURCE) $(A8Q4_SHAPE_SWEEP_SOURCE) $(LINEAR_HEADER) \
		$(Q4_ACTIVATION_PROFILE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-function -Wno-unused-command-line-argument \
		-DNINFER_SOURCE_DIR='"$(abspath ../..)"' --offload-device-only -S \
		$(A8Q4_DFLASH_MLP_DOWN_SMALL_T_SOURCE) -o $@

a8q4-dflash-mlp-down-small-t-static: $(BUILD_DIR)/a8q4_dflash_mlp_down_small_t.s
	$(PYTHON) $(A8Q4_DFLASH_MLP_DOWN_SMALL_T_STATIC_CHECK) $<

a8q4-dflash-mlp-down-small-t-static-test: $(BUILD_DIR)/a8q4_dflash_mlp_down_small_t.s
	cd ../.. && $(PYTHON) -m unittest \
		tools.r9700.test_check_a8q4_dflash_mlp_down_small_t_static

a8q4-dflash-small-t-static: $(BUILD_DIR)/q4g64_linear.s
	$(PYTHON) check_a8q4_dflash_small_t_static.py $<

a8q4-dflash-small-t-test:
	cd ../.. && $(PYTHON) -m unittest \
		tools.r9700.test_check_a8q4_dflash_small_t_static \
		tools.r9700.test_run_a8q4_dflash_small_t_gate

a8q4-dflash-small-t-screen: a8q4-dflash-small-t-build a8q4-dflash-small-t-static \
		a8q4-dflash-small-t-test
	test -n "$(DFLASH_SMALL_T_OUTPUT_DIR)"
	$(PYTHON) run_a8q4_dflash_small_t_gate.py \
		--binary $(A8Q4_DFLASH_SMALL_T_TARGET) \
		--assembly $(BUILD_DIR)/q4g64_linear.s \
		--c1-summary "$(DFLASH_SMALL_T_C1_SUMMARY)" \
		--output-dir "$(DFLASH_SMALL_T_OUTPUT_DIR)"

$(A4Q4_PREFILL_TARGET): a4q4_prefill_qual.hip $(A8Q4_SHAPE_SWEEP_SOURCE) $(LINEAR_IMPL) $(LINEAR_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-function -DNINFER_SOURCE_DIR='"$(CURDIR)/../.."' \
		a4q4_prefill_qual.hip $(LINEAR_IMPL) $(LDFLAGS) -o $@

.PHONY: a4q4-prefill-regression a4q4-prefill-benchmark
a4q4-prefill-regression: $(A4Q4_PREFILL_TARGET)
	$(A4Q4_PREFILL_TARGET) --regression

a4q4-prefill-benchmark: $(A4Q4_PREFILL_TARGET)
	test -n "$(A4Q4_PREFILL_OUT)"
	$(A4Q4_PREFILL_TARGET) --benchmark --out-json "$(A4Q4_PREFILL_OUT)"

a8q4-shape-sweep: $(A8Q4_SHAPE_SWEEP_TARGET)
	mkdir -p $(dir $(A8Q4_SHAPE_SWEEP_JSON))
	$(A8Q4_SHAPE_SWEEP_TARGET) --out-json $(A8Q4_SHAPE_SWEEP_JSON)

a8q4-prefill-cta-regression: $(A8Q4_SHAPE_SWEEP_TARGET) q4g64-linear-isa q4g64-linear-resources
	$(A8Q4_SHAPE_SWEEP_TARGET) --prefill-cta-regression

a8q4-prefill-cta-lds-scope-regression: $(A8Q4_SHAPE_SWEEP_TARGET)
	$(A8Q4_SHAPE_SWEEP_TARGET) --prefill-cta-lds-scope-regression

a8q4-prefill-cta-lds-scope-benchmark: $(A8Q4_SHAPE_SWEEP_TARGET)
	test -n "$(PREFILL_CTA_LDS_SCOPE_OUT)"
	$(A8Q4_SHAPE_SWEEP_TARGET) --prefill-cta-lds-scope-benchmark \
		--out-json "$(PREFILL_CTA_LDS_SCOPE_OUT)"

a8q4-prefill-cta-lds-scope-report:
	test -n "$(PREFILL_CTA_LDS_SCOPE_OUT)"
	python3 check_prefill_cta_lds_scope_report.py q4 \
		--executable "$(abspath $(A8Q4_SHAPE_SWEEP_TARGET))" "$(PREFILL_CTA_LDS_SCOPE_OUT)"

a8q4-prefill-cta-m64n64-regression: $(A8Q4_SHAPE_SWEEP_TARGET)
	$(A8Q4_SHAPE_SWEEP_TARGET) --prefill-cta-m64n64-regression

a8q4-prefill-cta-m128n128-regression: $(A8Q4_SHAPE_SWEEP_TARGET)
	$(A8Q4_SHAPE_SWEEP_TARGET) --prefill-cta-m128n128-regression

a8q4-prefill-cta-m128n128-benchmark: $(A8Q4_SHAPE_SWEEP_TARGET)
	test -n "$(PREFILL_CTA_M128N128_OUT)"
	$(A8Q4_SHAPE_SWEEP_TARGET) --prefill-cta-m128n128-benchmark \
		--out-json "$(PREFILL_CTA_M128N128_OUT)"

a8q4-prefill-cta-m128n128-report:
	test -n "$(PREFILL_CTA_M128N128_OUT)"
	python3 check_prefill_cta_lds_scope_report.py q4-m128n128 \
		--executable "$(abspath $(A8Q4_SHAPE_SWEEP_TARGET))" \
		"$(PREFILL_CTA_M128N128_OUT)"

a8q4-prefill-cta-pingpong-regression: $(A8Q4_SHAPE_SWEEP_TARGET)
	$(A8Q4_SHAPE_SWEEP_TARGET) --prefill-cta-pingpong-regression

a8q4-prefill-cta-pingpong-benchmark: $(A8Q4_SHAPE_SWEEP_TARGET)
	test -n "$(PREFILL_CTA_PINGPONG_OUT)"
	$(A8Q4_SHAPE_SWEEP_TARGET) --prefill-cta-pingpong-benchmark \
		--out-json "$(PREFILL_CTA_PINGPONG_OUT)"

a8q4-prefill-cta-pingpong-report:
	test -n "$(PREFILL_CTA_PINGPONG_OUT)"
	python3 check_prefill_cta_lds_scope_report.py q4-pingpong \
		--executable "$(abspath $(A8Q4_SHAPE_SWEEP_TARGET))" \
		"$(PREFILL_CTA_PINGPONG_OUT)"

$(Q4_TENSOR_A4_TARGET): $(Q4_TENSOR_SOURCE) $(LINEAR_IMPL) $(LINEAR_HEADER) $(LINEAR_TENSOR_IMPL) $(LINEAR_PUBLIC_HEADER) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include -DNINFER_R9700_Q4_ACTIVATION_BITS=4 $(Q4_TENSOR_SOURCE) $(LINEAR_IMPL) $(LINEAR_TENSOR_IMPL) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(LDFLAGS) -o $@

$(Q4_TENSOR_A8_TARGET): $(Q4_TENSOR_SOURCE) $(LINEAR_IMPL) $(LINEAR_HEADER) $(LINEAR_TENSOR_IMPL) $(LINEAR_PUBLIC_HEADER) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include -DNINFER_R9700_Q4_ACTIVATION_BITS=8 $(Q4_TENSOR_SOURCE) $(LINEAR_IMPL) $(LINEAR_TENSOR_IMPL) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) $(CORE_TENSOR_IMPL) $(LDFLAGS) -o $@

q4g64-linear: $(Q4G64_LINEAR_TARGET) $(Q4_TENSOR_A4_TARGET) $(Q4_TENSOR_A8_TARGET)
	mkdir -p $(dir $(Q4G64_LINEAR_JSON))
	$(Q4G64_LINEAR_TARGET) --out-json $(Q4G64_LINEAR_JSON)
	$(Q4_TENSOR_A4_TARGET) --out-json $(Q4_TENSOR_A4_JSON)
	$(Q4_TENSOR_A8_TARGET) --out-json $(Q4_TENSOR_A8_JSON)

$(BUILD_DIR)/q4g64_linear.s: $(LINEAR_IMPL) $(LINEAR_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S $(LINEAR_IMPL) -o $@

q4g64-linear-isa: $(BUILD_DIR)/q4g64_linear.s
	grep -E "v_wmma_i32_16x16x32_iu4|v_cvt|a8q4g64_linear_(wmma32|prefill_cta)_kernel|q4g64_linear_wmma32_kernel" $<

q4g64-linear-resources: $(BUILD_DIR)/q4g64_linear.s
	awk '/-- Begin function/ { selected = ($$0 ~ /q4g64_quantize_activation_kernel|a8g64_quantize_activation_kernel|q4g64_linear_(wmma32|vector)_kernel|a8q4g64_linear_(wmma32|prefill_cta)_kernel/) } selected && $$0 ~ /-- Begin function|v_wmma_i32_16x16x32_iu4|amdhsa_group_segment_fixed_size|amdhsa_private_segment_fixed_size|amdhsa_next_free_vgpr|NumVgprs|ScratchSize|LDSByteSize|Occupancy/' $<

sanitize: $(HIP_DEVICE_SANITIZE_TARGET)
	# ROCm's process-global HSA runtime retains internal allocations through exit; ASan remains
	# useful here for host memory errors, but LeakSanitizer cannot distinguish those allocations.
	ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 ./$(HIP_DEVICE_SANITIZE_TARGET)

$(BUILD_DIR)/r9700_qual.s: $(PROBE_SOURCE) $(KV_ORACLE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S $< -o $@

isa: $(BUILD_DIR)/r9700_qual.s $(BUILD_DIR)/kv_op_qual.s
	grep -E "v_wmma.*fp8" $(BUILD_DIR)/r9700_qual.s $(BUILD_DIR)/kv_op_qual.s

$(BUILD_DIR)/kv_append.s: $(KV_APPEND_SOURCE) $(KV_APPEND_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S $(KV_APPEND_SOURCE) -o $@

kv-resources: $(BUILD_DIR)/kv_op_qual.s $(BUILD_DIR)/kv_append.s
	awk '/-- Begin function/ { selected = ($$0 ~ /softmax_wmma_scores_in_place_kernel|qk_wmma_kernel|pv_vector_kernelILj16ELb0ELb0|fused_attention_causal_kernelILj16ELb1ELb0ELb0ELb0/) } selected && $$0 ~ /-- Begin function|v_wmma_f32_16x16x16_fp8_fp8|amdhsa_group_segment_fixed_size|amdhsa_private_segment_fixed_size|amdhsa_next_free_vgpr|NumVgprs|ScratchSize|LDSByteSize|Occupancy/' $(BUILD_DIR)/kv_op_qual.s
	awk '/-- Begin function/ { selected = ($$0 ~ /append_kernelILj16ELb1ELb0ELb0/) } selected && $$0 ~ /-- Begin function|amdhsa_group_segment_fixed_size|amdhsa_private_segment_fixed_size|amdhsa_next_free_vgpr|NumVgprs|ScratchSize|LDSByteSize|Occupancy/' $(BUILD_DIR)/kv_append.s

$(BUILD_DIR)/xattention_s$(XATTENTION_STRIDE)_tau$(XATTENTION_TAU_PERMILLE).s: \
		$(XATTENTION_SOURCE) $(XATTENTION_HEADER) $(XATTENTION_PROFILE) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S \
		-DNINFER_R9700_XATTENTION_STRIDE=$(XATTENTION_STRIDE) \
		-DNINFER_R9700_XATTENTION_TAU_PERMILLE=$(XATTENTION_TAU_PERMILLE) \
		$(XATTENTION_SOURCE) -o $@

xattention-isa: $(BUILD_DIR)/xattention_s$(XATTENTION_STRIDE)_tau$(XATTENTION_TAU_PERMILLE).s
	awk '/-- Begin function/ { selected = ($$0 ~ /xattention_(pack_keys|rank|consumer|flash_consumer)_kernel/) } selected && $$0 ~ /-- Begin function|v_wmma|v_exp|v_fma|global_load|global_store/' $<

xattention-resources: $(BUILD_DIR)/xattention_s$(XATTENTION_STRIDE)_tau$(XATTENTION_TAU_PERMILLE).s
	awk '/-- Begin function/ { selected = ($$0 ~ /xattention_(pack_keys|rank|consumer|flash_consumer)_kernel/) } selected && $$0 ~ /-- Begin function|amdhsa_group_segment_fixed_size|amdhsa_private_segment_fixed_size|amdhsa_next_free_vgpr|NumVgprs|ScratchSize|LDSByteSize|Occupancy/' $<

xattention-reports: $(BUILD_DIR)/xattention_s$(XATTENTION_STRIDE)_tau$(XATTENTION_TAU_PERMILLE).s
	mkdir -p $(XATTENTION_EVIDENCE_DIR)
	awk '/-- Begin function/ { selected = ($$0 ~ /xattention_(pack_keys|rank|consumer|flash_consumer)_kernel/) } selected && $$0 ~ /-- Begin function|v_wmma|v_exp|v_fma|global_load|global_store/' $< > $(XATTENTION_EVIDENCE_DIR)/isa.txt
	awk '/-- Begin function/ { selected = ($$0 ~ /xattention_(pack_keys|rank|consumer|flash_consumer)_kernel/) } selected && $$0 ~ /-- Begin function|amdhsa_group_segment_fixed_size|amdhsa_private_segment_fixed_size|amdhsa_next_free_vgpr|NumVgprs|ScratchSize|LDSByteSize|Occupancy/' $< > $(XATTENTION_EVIDENCE_DIR)/resources.txt

$(BUILD_DIR)/linear_op_qual.s: $(LINEAR_IMPL) $(LINEAR_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S $(LINEAR_IMPL) -o $@

linear-isa: $(BUILD_DIR)/linear_op_qual.s
	grep -E "v_wmma_f32_16x16x16_bf16|v_fma|v_fmac|v_mac|v_dot" $(BUILD_DIR)/linear_op_qual.s

linear-resources: $(BUILD_DIR)/linear_op_qual.s
	grep -E "bf16_linear_wmma16_kernel|wave8_linear_kernelILb[01]ELj16EE|num_vgpr|group_segment_fixed_size|private_seg_size|LDSByteSize|ScratchSize|Occupancy" $(BUILD_DIR)/linear_op_qual.s

$(BUILD_DIR)/eager_ops.s: $(EAGER_IMPL) $(EAGER_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S $(EAGER_IMPL) -o $@

eager-isa: $(BUILD_DIR)/eager_ops.s
	grep -E "rmsnorm_kernel|layernorm_kernel|l2norm_kernel|embedding_gather_kernel|argmax_bf16_kernel" $(BUILD_DIR)/eager_ops.s

rmsnorm-prefill-static: $(BUILD_DIR)/eager_ops.s
	$(PYTHON) check_rmsnorm_prefill_static.py $<

rmsnorm-prefill-static-test:
	cd ../.. && $(PYTHON) -m unittest tools.r9700.test_check_rmsnorm_prefill_static

rmsnorm-prefill-benchmark: $(RMSNORM_PREFILL_TARGET) rmsnorm-prefill-static
	test -n "$(RMSNORM_PREFILL_JSON)"
	./$(RMSNORM_PREFILL_TARGET) --benchmark --out-json "$(RMSNORM_PREFILL_JSON)"

rmsnorm-k256-prefill-static: $(BUILD_DIR)/eager_ops.s
	$(PYTHON) check_rmsnorm_k256_prefill_static.py $<

rmsnorm-k256-prefill-static-test:
	cd ../.. && $(PYTHON) -m unittest tools.r9700.test_check_rmsnorm_k256_prefill_static

rmsnorm-k256-prefill-benchmark: $(RMSNORM_K256_PREFILL_TARGET) rmsnorm-k256-prefill-static
	test -n "$(RMSNORM_K256_PREFILL_JSON)"
	./$(RMSNORM_K256_PREFILL_TARGET) --benchmark --out-json "$(RMSNORM_K256_PREFILL_JSON)"

rmsnorm-decode-production-static: $(BUILD_DIR)/eager_ops.s
	$(PYTHON) check_rmsnorm_decode_static.py $<

rmsnorm-decode-production-static-test:
	cd ../.. && $(PYTHON) -m unittest tools.r9700.test_check_rmsnorm_decode_static

rmsnorm-decode-production-routing-test: $(RMSNORM_DECODE_ROUTING_TARGET)
	./$(RMSNORM_DECODE_ROUTING_TARGET)

rmsnorm-decode-production-regression: $(RMSNORM_DECODE_PRODUCTION_TARGET) \
		rmsnorm-decode-production-static rmsnorm-decode-production-routing-test
	./$(RMSNORM_DECODE_PRODUCTION_TARGET)

gated-rmsnorm-prefill-static: $(BUILD_DIR)/eager_ops.s
	$(PYTHON) check_gated_rmsnorm_prefill_static.py $<

gated-rmsnorm-prefill-static-test:
	cd ../.. && $(PYTHON) -m unittest tools.r9700.test_check_gated_rmsnorm_prefill_static

gated-rmsnorm-prefill-path-test: $(GATED_RMSNORM_PREFILL_TARGET)
	@tmp_dir=$$(mktemp -d); \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	./$(GATED_RMSNORM_PREFILL_TARGET) --check-output-path "$$tmp_dir/fresh.json"; \
	touch "$$tmp_dir/existing.json"; \
	! ./$(GATED_RMSNORM_PREFILL_TARGET) --check-output-path "$$tmp_dir/existing.json"; \
	ln -s missing "$$tmp_dir/dangling.json"; \
	! ./$(GATED_RMSNORM_PREFILL_TARGET) --check-output-path "$$tmp_dir/dangling.json"

gated-rmsnorm-prefill-benchmark: $(GATED_RMSNORM_PREFILL_TARGET) gated-rmsnorm-prefill-static
	test -n "$(GATED_RMSNORM_PREFILL_JSON)"
	./$(GATED_RMSNORM_PREFILL_TARGET) --benchmark --out-json "$(GATED_RMSNORM_PREFILL_JSON)"

gated-rmsnorm-k128-production-static: $(BUILD_DIR)/eager_ops.s
	$(PYTHON) check_gated_rmsnorm_k128_static.py $<

gated-rmsnorm-k128-production-static-test:
	cd ../.. && $(PYTHON) -m unittest tools.r9700.test_check_gated_rmsnorm_k128_static

silu-mul-split-prefill-static: $(BUILD_DIR)/eager_ops.s
	$(PYTHON) check_silu_mul_split_prefill_static.py $<

silu-mul-split-prefill-static-test:
	cd ../.. && $(PYTHON) -m unittest tools.r9700.test_check_silu_mul_split_prefill_static

silu-mul-split-prefill-benchmark: $(SILU_MUL_SPLIT_PREFILL_TARGET) silu-mul-split-prefill-static
	test -n "$(SILU_MUL_SPLIT_PREFILL_JSON)"
	./$(SILU_MUL_SPLIT_PREFILL_TARGET) --benchmark --out-json "$(SILU_MUL_SPLIT_PREFILL_JSON)"

$(BUILD_DIR)/gdn_ops.s: $(GDN_IMPL) $(GDN_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include -Wno-unused-command-line-argument \
		--offload-device-only -S $(GDN_IMPL) -o $@

gdn-isa: $(BUILD_DIR)/gdn_ops.s
	grep -E "causal_conv1d_silu_kernel|control_gates_kernel|recurrent_kernel|copy_fp32_kernel" $(BUILD_DIR)/gdn_ops.s

gdn-prefill-static: $(BUILD_DIR)/gdn_ops.s
	$(PYTHON) check_causal_conv_prefill_static.py $<

gdn-prefill-static-test:
	cd ../.. && $(PYTHON) -m unittest tools.r9700.test_check_causal_conv_prefill_static

gdn-prefill-benchmark: $(GDN_TARGET) gdn-prefill-static
	./$(GDN_TARGET) --prefill-benchmark

$(BUILD_DIR)/gdn_recurrence.s: $(GDN_RECURRENCE_IMPL) $(GDN_RECURRENCE_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include -Wno-unused-command-line-argument --offload-device-only -S $(GDN_RECURRENCE_IMPL) -o $@

gdn-recurrence-resources: $(BUILD_DIR)/gdn_recurrence.s
	grep -E "ordinary_kernel|snapshot_kernel|record_kernel|NumVgprs|ScratchSize|LDSByteSize|Occupancy" $(BUILD_DIR)/gdn_recurrence.s

$(GDN_GROUPED_HEADS_TARGET): $(GDN_GROUPED_HEADS_SOURCE) $(GDN_GROUPED_HEADS_IMPL) $(GDN_GROUPED_HEADS_HEADER) $(GDN_RECURRENCE_IMPL) $(GDN_RECURRENCE_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include $(GDN_GROUPED_HEADS_SOURCE) $(GDN_GROUPED_HEADS_IMPL) \
		$(GDN_RECURRENCE_IMPL) $(HIP_DEVICE_IMPL) $(HIP_ARENA_IMPL) $(CORE_DTYPE_IMPL) \
		$(CORE_TENSOR_IMPL) $(LDFLAGS) -o $@

$(BUILD_DIR)/gdn_grouped_heads.s: $(GDN_GROUPED_HEADS_IMPL) $(GDN_GROUPED_HEADS_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S \
		$(GDN_GROUPED_HEADS_IMPL) -o $@

.PHONY: gdn-grouped-heads-qual gdn-grouped-heads-benchmark gdn-grouped-heads-static gdn-grouped-heads-path-test
gdn-grouped-heads-qual: $(GDN_GROUPED_HEADS_TARGET)
	$(GDN_GROUPED_HEADS_TARGET)

gdn-grouped-heads-benchmark: $(GDN_GROUPED_HEADS_TARGET)
	test -n "$(GDN_GROUPED_HEADS_JSON)"
	$(GDN_GROUPED_HEADS_TARGET) --out-json "$(GDN_GROUPED_HEADS_JSON)"

gdn-grouped-heads-static: $(BUILD_DIR)/gdn_grouped_heads.s
	python3 check_gdn_grouped_heads_static.py $(BUILD_DIR)/gdn_grouped_heads.s

gdn-grouped-heads-path-test: $(GDN_GROUPED_HEADS_TARGET)
	@tmp_dir=$$(mktemp -d); trap 'rm -rf "$$tmp_dir"' EXIT; \
		$(GDN_GROUPED_HEADS_TARGET) --check-output-path "$$tmp_dir/fresh.json"; \
		touch "$$tmp_dir/existing.json"; \
		! $(GDN_GROUPED_HEADS_TARGET) --check-output-path "$$tmp_dir/existing.json"; \
		ln -s missing "$$tmp_dir/dangling.json"; \
		! $(GDN_GROUPED_HEADS_TARGET) --check-output-path "$$tmp_dir/dangling.json"

.PHONY: gdn-affine-chunk-static gdn-affine-chunk-static-test gdn-affine-chunk-benchmark

$(BUILD_DIR)/gdn_affine_chunk.s: $(GDN_AFFINE_CHUNK_IMPL) $(GDN_AFFINE_CHUNK_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include -Wno-unused-command-line-argument \
		--offload-device-only -S $(GDN_AFFINE_CHUNK_IMPL) -o $@

gdn-affine-chunk-static: $(BUILD_DIR)/gdn_affine_chunk.s
	$(PYTHON) check_gdn_affine_chunk_static.py $<

gdn-affine-chunk-static-test:
	cd ../.. && $(PYTHON) -m unittest tools.r9700.test_check_gdn_affine_chunk_static

gdn-affine-chunk-benchmark: $(GDN_AFFINE_CHUNK_TARGET) gdn-affine-chunk-static
	test -n "$(GDN_AFFINE_CHUNK_JSON)"
	./$(GDN_AFFINE_CHUNK_TARGET) --benchmark --out-json "$(GDN_AFFINE_CHUNK_JSON)"

$(BUILD_DIR)/sampling_ops.s: $(SAMPLING_IMPL) $(SAMPLING_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include -Wno-unused-command-line-argument --offload-device-only -S $(SAMPLING_IMPL) -o $@

sampling-isa: $(BUILD_DIR)/sampling_ops.s
	grep -E "sampling_partial_kernel|sampling_finalize_kernel|sampling_small_kernel|num_vgpr|group_segment_fixed_size|private_seg_size" $(BUILD_DIR)/sampling_ops.s

$(BUILD_DIR)/rope_ops.s: $(ROPE_IMPL) $(ROPE_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -I../../include -Wno-unused-command-line-argument --offload-device-only -S $(ROPE_IMPL) -o $@

rope-resources: $(BUILD_DIR)/rope_ops.s
	grep -E "rope_kernel|NumVgprs|ScratchSize|LDSByteSize|Occupancy" $(BUILD_DIR)/rope_ops.s

$(BUILD_DIR)/kv_op_qual.s: $(KV_ATTENTION_SOURCE) $(KV_ATTENTION_HEADER) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S $(KV_ATTENTION_SOURCE) -o $@

$(BUILD_DIR)/split512_attention.s: $(KV_ATTENTION_SOURCE) $(KV_ATTENTION_HEADER) \
		$(SPLIT512_STATIC_CHECK) Makefile
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -Wno-unused-command-line-argument --offload-device-only -S \
		$(KV_ATTENTION_SOURCE) -o $@

split512-attention-isa: $(BUILD_DIR)/split512_attention.s
	$(PYTHON) $(SPLIT512_STATIC_CHECK) isa $<

split512-attention-resources: $(BUILD_DIR)/split512_attention.s
	$(PYTHON) $(SPLIT512_STATIC_CHECK) resources $<

run: all
	./$(HIP_DEVICE_TARGET)
	./$(PROBE_TARGET)
	./$(KV_TARGET)
	./$(LINEAR_TARGET)
	./$(EAGER_TARGET)
	./$(GDN_TARGET)
	./$(GDN_RECURRENCE_TARGET)
	./$(SAMPLING_TARGET)
	./$(ROPE_TARGET)
	./$(SERVE_ENV_TARGET)

# CPU-only toolchain/capture audit. The named retained SQLite databases are opened read-only; the
# sanitizer capability probes compile temporary objects but never load HIP or launch a kernel.
# Override an evidence path explicitly when validating a newly retained capture.
rocm-tool-audit: $(ROCM_AUDIT)
	$(PYTHON) $(ROCM_AUDIT) \
		--trace-db $(TRACE_EVIDENCE_DB) \
		--pmc-db $(PMC_EVIDENCE_DB) \
		--expected-trace-command "$(TRACE_EVIDENCE_COMMAND)" \
		--expected-pmc-command "$(PMC_EVIDENCE_COMMAND)" \
		--zero-probe SQ_INSTS_WAVE32_VALU=$(VALU_EVIDENCE_DB) \
		--zero-probe SQ_INSTS_WAVE32_LDS=$(LDS_EVIDENCE_DB) \
		--zero-probe SQ_INSTS_VALU=$(GENERIC_VALU_EVIDENCE_DB) \
		--zero-probe TCP_REQ=$(TCP_EVIDENCE_DB) \
		--zero-probe TCP_REQ_MISS=$(TCP_EVIDENCE_DB) \
		--zero-probe GL2C_HIT=$(GL2C_EVIDENCE_DB) \
		--zero-probe GL2C_MISS=$(GL2C_EVIDENCE_DB) \
		--output $(ROCM_AUDIT_JSON)

rocm-tool-audit-test: $(ROCM_AUDIT) $(ROCM_AUDIT_TEST)
	$(PYTHON) $(ROCM_AUDIT_TEST)

# Dispatch/memory trace recipe for a named, reproducible small correctness workload. Counter
# collection is intentionally separate because profiler queue interception invalidates latency
# selection. Override PROFILE_DIR to retain an experiment artifact outside the checkout default.
profile-trace: $(KV_TARGET)
	mkdir -p $(PROFILE_DIR)
	$(ROCPROFV3) --runtime-trace --kernel-trace --memory-copy-trace --stats --summary \
		-d $(PROFILE_DIR) -- ./$(KV_TARGET) --context 66 --iterations 1

# Focused post-parallel-softmax decode attribution. The selected variant here is only the current
# latency challenger: the persistent group/layout remains gated on paired PPL and exact tokens.
profile-attention-trace: $(KV_TARGET)
	mkdir -p $(PROFILE_DIR)
	$(ROCPROFV3) --runtime-trace --kernel-trace --memory-copy-trace --stats --summary \
		-d $(PROFILE_DIR) -- ./$(KV_TARGET) --context 4096 --rows 1 --iterations 5 \
		--timing-only --group 16 --key-order token --value-order feature --scale-order feature

# Attribution-only PMU pass for the page-boundary workload. These are the counters that return
# validated nonzero data on the ROCm 10/gfx1201 baseline; the catalog's VALU/LDS/cache events are
# omitted because isolated target-specific probes also return only zero. Do not use an intercepted
# queue's timing as the latency-selection source.
profile-pmc: $(KV_TARGET)
	mkdir -p $(PROFILE_DIR)
	$(ROCPROFV3) --pmc SQ_BUSY_CYCLES SQ_WAVES -f rocpd \
		-d $(PROFILE_DIR) -- ./$(KV_TARGET) --context 4096 --iterations 1 \
		--timing-only --group 16 --key-order token --value-order feature --scale-order feature

clean:
	rm -f $(PROBE_TARGET) $(KV_TARGET) $(HIP_DEVICE_TARGET) $(HIP_DEVICE_SANITIZE_TARGET) $(LINEAR_TARGET) $(EAGER_TARGET) $(RMSNORM_K256_PREFILL_TARGET) $(RMSNORM_DECODE_PRODUCTION_TARGET) $(RMSNORM_DECODE_ROUTING_TARGET) $(GATED_RMSNORM_PREFILL_TARGET) $(SILU_MUL_SPLIT_PREFILL_TARGET) $(GDN_TARGET) $(GDN_RECURRENCE_TARGET) $(SAMPLING_TARGET) $(ROPE_TARGET) $(SERVE_ENV_TARGET) $(INT4_WMMA_TARGET) $(Q4_HARDWARE_PEAK_TARGET) $(W8A8_WMMA_TARGET) $(Q4G64_LINEAR_TARGET) $(Q4_TENSOR_A4_TARGET) $(Q4_TENSOR_A8_TARGET) $(BUILD_DIR)/r9700_qual.s $(BUILD_DIR)/kv_append.s $(BUILD_DIR)/linear_op_qual.s $(BUILD_DIR)/w8a8_wmma_linear_qual.s $(BUILD_DIR)/q4g64_linear.s $(BUILD_DIR)/q4_hardware_peak_probe.s $(BUILD_DIR)/eager_ops.s $(BUILD_DIR)/gdn_ops.s $(BUILD_DIR)/gdn_recurrence.s $(BUILD_DIR)/sampling_ops.s $(BUILD_DIR)/rope_ops.s $(BUILD_DIR)/int4_wmma_qual.s
