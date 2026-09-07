# This is a closed HIP production-core boundary. Everything in the archive is compiled for the
# one R9700 code object and owns no alternate runtime, tracing API, launch protocol, or generic
# backend state.
function(ninfer_r9700_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3/export
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_8_27b/export
    ${PROJECT_SOURCE_DIR}/third_party)
endfunction()

add_library(ninfer_r9700_core STATIC
  core/device.hip
  core/arena.hip
  core/dtype.cpp
  core/tensor.cpp
  core/layout.cpp
  core/paged_kv_cache.cpp
  core/fp8_int4_paged_kv_cache.cpp
  core/cyclic_kv_cache.cpp
  core/gdn_replay_records.cpp
  core/linear_attention_state.cpp
  core/decode_graph.hip
  runtime/engine/request_memory.cpp
  targets/qwen3/impl/runtime/prefix_identity.cpp
  targets/qwen3/impl/runtime/kv_ram_cache.cpp
  targets/qwen3/impl/state/decoder_state.cpp
  targets/qwen3/impl/state/decoder_state_transaction.hip
  targets/qwen3_8_27b/impl/r9700_full_attention.hip
  ops/r9700/eager/eager_tensor_ops.cpp
  ops/r9700/eager/eager_ops.hip
  ops/r9700/gdn/causal_conv1d_silu.cpp
  ops/r9700/gdn/gdn_gating.cpp
  ops/r9700/gdn/gdn_ops.hip
  ops/r9700/gdn_recurrence/gated_delta_net.hip
  ops/r9700/gdn_recurrence/gdn_prefill_scale_sidecar.hip
  ops/r9700/gdn_recurrence/gdn_replay_fold.cpp
  ops/r9700/gdn_recurrence/gdn_replay_fold.hip
  ops/r9700/sampling/sampling.hip
  ops/r9700/rope/rope.hip
  ops/r9700/vision/vision_attention.hip
  ops/r9700/vision/vision_pos_embed.hip
  ops/r9700/dflash/dflash2_path_select.cpp
  ops/r9700/dflash/dflash2_path_select.hip
  ops/r9700/dflash/bidirectional_gqa_attention.cpp
  ops/r9700/dflash/bidirectional_gqa_attention.hip
  ops/r9700/dflash/kv_cache_append_prefix.cpp
  ops/r9700/dflash/kv_cache_append_prefix.hip
  ops/r9700/dflash/grouped_dynamic_conv.cpp
  ops/r9700/dflash/grouped_dynamic_conv.hip
  ops/r9700/dflash/swa.cpp
  ops/r9700/dflash/swa.hip
  ops/r9700/speculative/mtp_round.hip
  ops/r9700/speculative/speculative_round.hip
  ops/r9700/schedule/scalar_schedule_tensor_ops.cpp
  ops/r9700/schedule/scalar_schedule_ops.hip
  ops/r9700/kv/fp8_int4_kv_append.hip
  ops/r9700/kv/fp8_int4_kv_compact.hip
  ops/r9700/kv/fp8_int4_kv_trace.hip
  ops/r9700/kv/fp8_int4_kv_attention.hip
  ops/r9700/linear/linear_tensor_op.cpp
  ops/r9700/linear/fp8_activation.hip
  ops/r9700/linear/linear_execution.hip
  ops/r9700/linear/r9700_linear.hip)
if(NINFER_R9700_XATTENTION_QUALIFICATION)
  target_sources(ninfer_r9700_core PRIVATE
    ops/r9700/kv/fp8_int4_kv_xattention.hip)
  target_compile_definitions(ninfer_r9700_core PRIVATE
    NINFER_R9700_XATTENTION_QUALIFICATION=1
    NINFER_R9700_XATTENTION_STRIDE=${NINFER_R9700_XATTENTION_STRIDE}
    NINFER_R9700_XATTENTION_TAU_PERMILLE=${NINFER_R9700_XATTENTION_TAU_PERMILLE})
endif()
ninfer_r9700_internal_includes(ninfer_r9700_core)
set_target_properties(ninfer_r9700_core PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_compile_options(ninfer_r9700_core PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<NOT:$<CONFIG:Release>>>:-gline-tables-only>)
target_link_libraries(ninfer_r9700_core PUBLIC
  hip::host
  roc::hipblaslt
  rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library)
add_library(ninfer::r9700_core ALIAS ninfer_r9700_core)

add_executable(ninfer_r9700_roctx_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/roctx_range_qual.cpp)
ninfer_r9700_internal_includes(ninfer_r9700_roctx_qual)
target_link_libraries(ninfer_r9700_roctx_qual PRIVATE
  rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library)

add_library(ninfer_runtime_capacity STATIC
  runtime/engine/kv_capacity.cpp)
ninfer_r9700_internal_includes(ninfer_runtime_capacity)

# Generic .ninfer framing/binding/final materialization has no model or weight-format execution
# ownership. Its direct-I/O event ring and final H2D copies use the canonical HIP core only.
add_library(ninfer_artifact STATIC
  artifact/reader.cpp
  artifact/storage_layouts.cpp
  artifact/binder.cpp
  artifact/materializer.cpp
  artifact/typed_binding.cpp)
ninfer_r9700_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_r9700_core hip::host)

# The R9700 target binder is a closed compileable boundary over the registered evaluation
# inventories. It performs no runtime repacking: every matrix view points directly into artifact
# storage.
add_library(ninfer_r9700_target_bindings STATIC
  targets/qwen3/impl/frontend/resources.cpp
  targets/qwen3/impl/vision/bindings.cpp
  targets/qwen3_8_27b/impl/package_identity.cpp
  targets/qwen3_8_27b/impl/load/bindings.cpp
  targets/qwen3_8_27b/impl/variant.cpp)
if(NINFER_R9700_XATTENTION_QUALIFICATION)
  target_compile_definitions(ninfer_r9700_target_bindings PRIVATE
    NINFER_R9700_XATTENTION_QUALIFICATION=1)
endif()
ninfer_r9700_internal_includes(ninfer_r9700_target_bindings)
target_link_libraries(ninfer_r9700_target_bindings PUBLIC ninfer_artifact hip::host)

# CPU text/frontend support used by the one exact Qwen runtime. Keeping utf8proc in this closed
# target avoids depending on the removed legacy Engine graph for tokenizer linkage.
add_library(ninfer_r9700_text STATIC
  text/unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)
ninfer_r9700_internal_includes(ninfer_r9700_text)

find_package(PkgConfig REQUIRED)
if(NINFER_BUILD_APPS)
  pkg_check_modules(NINFER_LIBCURL REQUIRED IMPORTED_TARGET libcurl)

  # Product-side acquisition owns paths, data URIs, and HTTP. Target/front-end code consumes only
  # already-owned bytes and cannot reach this library.
  add_library(ninfer_media_acquire STATIC
    product/media_acquire/acquire.cpp)
  ninfer_r9700_internal_includes(ninfer_media_acquire)
  target_link_libraries(ninfer_media_acquire PRIVATE PkgConfig::NINFER_LIBCURL)

  add_library(ninfer_product_prompt_input STATIC
    product/prompt_input/prompt_input.cpp)
  ninfer_r9700_internal_includes(ninfer_product_prompt_input)
  target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire)

  add_library(ninfer_product_load_progress STATIC
    product/load_progress/load_progress.cpp)
  ninfer_r9700_internal_includes(ninfer_product_load_progress)
endif()

# Media decode consumes already-owned bytes and is CPU-only. It is part of the exact Vision
# frontend closure, not an acquisition or target-execution backend.
pkg_check_modules(NINFER_FFMPEG REQUIRED IMPORTED_TARGET
  libavcodec
  libavformat
  libavutil
  libswscale)
add_library(ninfer_media_decode STATIC
  media/decode/decode.cpp)
ninfer_r9700_internal_includes(ninfer_media_decode)
target_link_libraries(ninfer_media_decode PUBLIC PkgConfig::NINFER_FFMPEG)

# Sole exact family-runtime instantiation. runtime.hip selects the Qwen3.8-27B R9700 Variant at
# compile time; the remaining sources are the family-owned frontend/state/control definitions that
# the instantiated planner and Program consume. This is a real linkable runtime boundary, not a
# diagnostic syntax-only TU and not yet the public Engine.
add_library(ninfer_r9700_qwen3_8_27b_runtime STATIC
  targets/qwen3_8_27b/impl/runtime.hip
  targets/qwen3/impl/frontend/chat_template.cpp
  targets/qwen3/impl/frontend/digest.cpp
  targets/qwen3/impl/frontend/encoded_history_cache.cpp
  targets/qwen3/impl/frontend/frontend.cpp
  targets/qwen3/impl/frontend/processor.cpp
  targets/qwen3/impl/frontend/tokenizer.cpp
  targets/qwen3/impl/state/round_state.cpp
  targets/qwen3/impl/runtime/visual_scatter.cpp
  targets/qwen3/impl/vision/control.cpp)
if(NINFER_R9700_XATTENTION_QUALIFICATION)
  target_compile_definitions(ninfer_r9700_qwen3_8_27b_runtime PUBLIC
    NINFER_R9700_XATTENTION_QUALIFICATION=1
    NINFER_R9700_XATTENTION_STRIDE=${NINFER_R9700_XATTENTION_STRIDE}
    NINFER_R9700_XATTENTION_TAU_PERMILLE=${NINFER_R9700_XATTENTION_TAU_PERMILLE})
endif()
ninfer_r9700_internal_includes(ninfer_r9700_qwen3_8_27b_runtime)
set_target_properties(ninfer_r9700_qwen3_8_27b_runtime PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_qwen3_8_27b_runtime PUBLIC
  ninfer_r9700_target_bindings
  ninfer_r9700_core
  ninfer_r9700_text
  ninfer_media_decode
  ninfer_runtime_capacity
  hip::host)

# Force every runtime archive member into one physical planner executable. This prevents an
# unresolved schedule dependency from being hidden merely because a small main does not call that
# Program method. The observable route then constructs/finalizes every C=1..4 ordinary, Vision,
# MTP, and DFlash planner profile on the selected gfx1201 device.
add_executable(ninfer_r9700_runtime_planner_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/runtime_planner_qual.cpp)
ninfer_r9700_internal_includes(ninfer_r9700_runtime_planner_qual)
set_target_properties(ninfer_r9700_runtime_planner_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201
  LINKER_LANGUAGE HIP)
target_link_libraries(ninfer_r9700_runtime_planner_qual PRIVATE
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,ninfer_r9700_qwen3_8_27b_runtime>"
  ninfer_r9700_target_bindings
  ninfer_r9700_core
  ninfer_r9700_text
  ninfer_media_decode
  ninfer_runtime_capacity
  hip::host)

# Public Engine ownership over the one exact Qwen3.8-27B R9700 runtime. The target registry and
# package adapter live here because they select and construct the only supported product; model
# mathematics and scheduling remain in the exact runtime archive above.
find_package(Threads REQUIRED)
add_library(ninfer_engine STATIC
  runtime/contract/sampling.cpp
  runtime/engine/admission_policy.cpp
  runtime/engine/public_types.cpp
  runtime/engine/engine.cpp
  targets/registry.cpp
  targets/qwen3_8_27b/impl/package.cpp)
ninfer_r9700_internal_includes(ninfer_engine)
set_target_properties(ninfer_engine PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_engine PUBLIC
  ninfer_r9700_qwen3_8_27b_runtime
  ninfer_r9700_target_bindings
  ninfer_artifact
  ninfer_r9700_core
  ninfer_r9700_text
  ninfer_media_decode
  ninfer_runtime_capacity
  Threads::Threads
  hip::host)
add_library(ninfer::engine ALIAS ninfer_engine)

if(NINFER_BUILD_APPS)
  # HTTP protocol/product adapter. It depends on the one public Engine and owns no model
  # mathematics, target state, or device backend selection.
  add_library(ninfer_serve STATIC
    serve/openai_schema.cpp
    serve/responses_schema.cpp
    serve/anthropic_schema.cpp
    serve/response_store.cpp
    serve/tool_call_parser.cpp
    serve/translate.cpp
    serve/console_log.cpp
    serve/generation_service.cpp
    serve/request_log.cpp
    serve/http_server.cpp
    serve/responses_http.cpp
    serve/serve_options.cpp)
  ninfer_r9700_internal_includes(ninfer_serve)
  target_include_directories(ninfer_serve PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  target_link_libraries(ninfer_serve
    PUBLIC ninfer_engine Threads::Threads
    PRIVATE ninfer_media_acquire hip::host)
endif()

# Force the complete public Engine archive through the native HIP link. The observable checks use
# validation paths that require no real checkpoint, while constructing DeviceContext proves the
# selected physical gfx1201 runtime and exception teardown. Real load/inference remains a separate
# gate requiring the complete BF16-derived product artifact.
add_executable(ninfer_r9700_engine_boundary_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/engine_boundary_qual.cpp)
ninfer_r9700_internal_includes(ninfer_r9700_engine_boundary_qual)
set_target_properties(ninfer_r9700_engine_boundary_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201
  LINKER_LANGUAGE HIP)
target_link_libraries(ninfer_r9700_engine_boundary_qual PRIVATE
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,ninfer_engine>"
  ninfer_r9700_qwen3_8_27b_runtime
  ninfer_r9700_target_bindings
  ninfer_artifact
  ninfer_r9700_core
  ninfer_r9700_text
  ninfer_media_decode
  ninfer_runtime_capacity
  Threads::Threads
  hip::host)

# Sole-target selection and startup device-memory/capacity planning. This compile-only object
# boundary checks the registry against native HIP and the Qwen3.8-27B R9700 package identity
# without claiming that the still excluded package schedule and Engine are linkable.
add_library(ninfer_r9700_registry OBJECT
  targets/registry.cpp)
ninfer_r9700_internal_includes(ninfer_r9700_registry)
target_link_libraries(ninfer_r9700_registry PRIVATE hip::host)

# This executable has the physical-device ownership oracle used while replacing the retired
# build. It links the archive, so the CMake path verifies HIP device-link ownership in
# addition to compiling every production-core translation unit.
option(NINFER_BUILD_R9700_CORE_QUALIFIER "Build the R9700 production-core qualifier" ON)
if(NINFER_BUILD_R9700_CORE_QUALIFIER)
  add_executable(ninfer_r9700_core_qual
    ${PROJECT_SOURCE_DIR}/tools/r9700/hip_device_qual.hip)
  ninfer_r9700_internal_includes(ninfer_r9700_core_qual)
  set_target_properties(ninfer_r9700_core_qual PROPERTIES
    HIP_ARCHITECTURES gfx1201)
  target_link_libraries(ninfer_r9700_core_qual PRIVATE ninfer_r9700_core hip::host)
endif()

# Synthetic .ninfer framing/binding/direct-materialization test. It deliberately binds tensors in
# a different order from the file and verifies exact R9700 H2D bytes, retained resources, plan
# offsets, and staging statistics without involving a checkpoint converter or target runtime.
add_executable(ninfer_r9700_artifact_qual
  ${PROJECT_SOURCE_DIR}/tests/test_artifact_materialization.cpp)
ninfer_r9700_internal_includes(ninfer_r9700_artifact_qual)
target_include_directories(ninfer_r9700_artifact_qual PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(ninfer_r9700_artifact_qual PRIVATE ninfer_artifact hip::host)

# Isolated first-slice row-scaled-E4M3 gate_up execution. It owns fixed descriptors and one
# installed top heuristic but is deliberately absent from target/Program dispatch until its
# physical numerical and speed gates pass.
add_executable(ninfer_r9700_fp8_gate_up_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/fp8_gate_up_qual.hip)
file(SHA256 ${PROJECT_SOURCE_DIR}/tools/r9700/fp8_gate_up_qual.hip
  NINFER_R9700_FP8_GATE_UP_QUAL_SOURCE_SHA256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${PROJECT_SOURCE_DIR}/tools/r9700/fp8_gate_up_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_fp8_gate_up_qual)
target_compile_definitions(ninfer_r9700_fp8_gate_up_qual PRIVATE
  NINFER_SOURCE_ROOT="${PROJECT_SOURCE_DIR}"
  NINFER_QUAL_SOURCE_SHA256="${NINFER_R9700_FP8_GATE_UP_QUAL_SOURCE_SHA256}")
set_target_properties(ninfer_r9700_fp8_gate_up_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_fp8_gate_up_qual PRIVATE
  ninfer_r9700_core hip::host roc::hipblaslt)
add_custom_command(TARGET ninfer_r9700_fp8_gate_up_qual POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -DINPUT=$<TARGET_FILE:ninfer_r9700_fp8_gate_up_qual>
          -DOUTPUT=$<TARGET_FILE:ninfer_r9700_fp8_gate_up_qual>.sha256
          -P ${PROJECT_SOURCE_DIR}/tools/cmake/write_sha256.cmake
  VERBATIM)

# Exact-shape catalog probe for the retained FP8 gate/up projection. This is intentionally
# disconnected from production dispatch: it enumerates every installed hipBLASLt algorithm before
# a bounded, independently checked timing qualifier admits any non-default algorithm.
add_executable(ninfer_r9700_fp8_gate_up_catalog_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/fp8_gate_up_catalog_qual.hip)
set_target_properties(ninfer_r9700_fp8_gate_up_catalog_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_fp8_gate_up_catalog_qual PRIVATE
  hip::host roc::hipblaslt)

# The two pending post-gate/up decisions instantiate the same hardened qualifier contract and
# prepared Linear execution owner at two explicit Qwen shapes. Compile-time identities keep the
# runnable source direct and avoid generated/source-rewritten variants.
function(ninfer_add_r9700_fp8_projection_qual target qualification rows middle last)
  add_executable(${target} ${PROJECT_SOURCE_DIR}/tools/r9700/fp8_gate_up_qual.hip
    ${PROJECT_SOURCE_DIR}/src/ops/r9700/linear/linear_execution.hip)
  ninfer_r9700_internal_includes(${target})
  target_compile_definitions(${target} PRIVATE
    NINFER_FP8_FIXED_PROJECTION_ROWS=${rows}
    NINFER_SOURCE_ROOT="${PROJECT_SOURCE_DIR}"
    NINFER_QUAL_SOURCE_SHA256="${NINFER_R9700_FP8_GATE_UP_QUAL_SOURCE_SHA256}"
    NINFER_QUAL_SCHEMA="ninfer.r9700.fp8_projection_qualification.v1"
    NINFER_QUAL_ARTIFACT_TYPE="ninfer_r9700_fp8_projection_comparison"
    NINFER_QUAL_SCHEMA_VERSION=1
    NINFER_QUAL_HAS_ID=1
    NINFER_QUALIFICATION_ID="${qualification}"
    NINFER_ORACLE_MIDDLE_ROW=${middle}
    NINFER_ORACLE_LAST_ROW=${last})
  set_target_properties(${target} PROPERTIES HIP_ARCHITECTURES gfx1201)
  target_link_libraries(${target} PRIVATE ninfer_r9700_core hip::host roc::hipblaslt)
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DINPUT=$<TARGET_FILE:${target}>
            -DOUTPUT=$<TARGET_FILE:${target}>.sha256
            -P ${PROJECT_SOURCE_DIR}/tools/cmake/write_sha256.cmake VERBATIM)
endfunction()

ninfer_add_r9700_fp8_projection_qual(
  ninfer_r9700_fp8_attention_qk_gate_value_qual attention_qk_gate_value 7168 3584 7167)
ninfer_add_r9700_fp8_projection_qual(
  ninfer_r9700_fp8_gdn_query_key_qual gdn_query_key 4096 2048 4095)

add_library(ninfer_r9700_hip_code_object_capture SHARED
  ${PROJECT_SOURCE_DIR}/tools/r9700/hip_code_object_capture.cpp)
target_link_libraries(ninfer_r9700_hip_code_object_capture PRIVATE hip::host ${CMAKE_DL_LIBS})

# Observable page-aligned explicit/automatic capacity behavior used by the registry preflight and
# post-load resolution passes.
add_executable(ninfer_r9700_kv_capacity_qual
  ${PROJECT_SOURCE_DIR}/tests/test_kv_capacity.cpp)
ninfer_r9700_internal_includes(ninfer_r9700_kv_capacity_qual)
target_link_libraries(ninfer_r9700_kv_capacity_qual PRIVATE ninfer_runtime_capacity)

# Sole registered model-to-target mapping; this deliberately does not instantiate the excluded
# package schedule.
add_executable(ninfer_r9700_registry_qual
  ${PROJECT_SOURCE_DIR}/tests/test_target_registry.cpp)
ninfer_r9700_internal_includes(ninfer_r9700_registry_qual)

# Metadata-complete target binding qualification. The companion Python tool creates sparse valid
# and wrong-format files, so this verifies all 1,124 descriptors and placements without pretending
# that zero-filled sparse matrix payloads are a numerical model artifact.
add_executable(ninfer_r9700_target_binding_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/target_binding_qual.cpp)
ninfer_r9700_internal_includes(ninfer_r9700_target_binding_qual)
target_link_libraries(ninfer_r9700_target_binding_qual PRIVATE
  ninfer_r9700_target_bindings ninfer_artifact hip::host)

# Physical state-owner qualifier: fixed request backing, typed growing-cache page mappings and
# spill/restore, canonical Qwen all-layer append/compact publication transactions, plus BF16
# cyclic and BF16/FP32 linear-attention state images.
add_executable(ninfer_r9700_state_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/hip_persistent_state_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_state_qual)
set_target_properties(ninfer_r9700_state_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_state_qual PRIVATE ninfer_r9700_core hip::host)

# Qwen3.8 DFlash2 state remains a target-private BF16 cyclic image. This qualifier compiles the
# HIP family-state bridge and checks exact lane capture/restore without introducing a fourth
# growing-cache format or borrowing the Text/MTP FP8-K/INT4-V owner.
add_executable(ninfer_r9700_dflash_state_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/dflash_state_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_dflash_state_qual)
target_include_directories(ninfer_r9700_dflash_state_qual PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_8_27b/export)
set_target_properties(ninfer_r9700_dflash_state_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_dflash_state_qual PRIVATE ninfer_r9700_core hip::host)

add_executable(ninfer_r9700_dflash2_select_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/dflash2_select_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_dflash2_select_qual)
set_target_properties(ninfer_r9700_dflash2_select_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_dflash2_select_qual PRIVATE ninfer_r9700_core hip::host)

add_executable(ninfer_r9700_bidirectional_gqa_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/bidirectional_gqa_attention_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_bidirectional_gqa_qual)
set_target_properties(ninfer_r9700_bidirectional_gqa_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_bidirectional_gqa_qual PRIVATE ninfer_r9700_core hip::host)

add_executable(ninfer_r9700_grouped_dynamic_conv_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/grouped_dynamic_conv_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_grouped_dynamic_conv_qual)
set_target_properties(ninfer_r9700_grouped_dynamic_conv_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_grouped_dynamic_conv_qual PRIVATE ninfer_r9700_core hip::host)

add_executable(ninfer_r9700_swa_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/swa_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_swa_qual)
set_target_properties(ninfer_r9700_swa_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_swa_qual PRIVATE ninfer_r9700_core hip::host)

# Exact DFlash BF16 state mutation. The live Qwen3.8 DFlash2 cyclic route and the family Full
# paged view are checked without borrowing the asymmetric Text/MTP cache transaction owner.
add_executable(ninfer_r9700_kv_cache_append_prefix_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/kv_cache_append_prefix_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_kv_cache_append_prefix_qual)
set_target_properties(ninfer_r9700_kv_cache_append_prefix_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_kv_cache_append_prefix_qual PRIVATE
  ninfer_r9700_core hip::host)

# Host-RAM spill/restore qualification for canonical FP8-K/INT4-V Text and MTP caches. It
# exercises fragmented physical mappings, asynchronous D2H-to-H2D ordering, semantic mismatch
# rejection before destination writes, and unpinned FIFO eviction.
add_executable(ninfer_r9700_kv_ram_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/kv_ram_cache_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_kv_ram_qual)
set_target_properties(ninfer_r9700_kv_ram_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_kv_ram_qual PRIVATE ninfer_r9700_core hip::host)

# Public sampling qualifier: the exact counter RNG, mixed greedy/stochastic rows, filtering,
# penalties, and token-count state transition are checked at the real padded vocabulary.
add_executable(ninfer_r9700_sampling_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/sampling_op_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_sampling_qual)
set_target_properties(ninfer_r9700_sampling_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_sampling_qual PRIVATE ninfer_r9700_core hip::host)

# Full speculative-round transition qualification: exact preparation, chain/tree acceptance,
# selector-q residuals, state publication, and hidden/token movement over the sole B<=4/W<=16
# product domain.  The executable uses an independent host oracle and caller-owned workspace.
add_executable(ninfer_r9700_speculative_round_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/speculative_round_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_speculative_round_qual)
set_target_properties(ninfer_r9700_speculative_round_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_speculative_round_qual PRIVATE ninfer_r9700_core hip::host)

# Public RoPE qualifier: all live Qwen3.8 Text, DFlash, and Vision geometries are checked
# directly against an independent represented-BF16/FP64 formula on physical wave32 gfx1201.
add_executable(ninfer_r9700_rope_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/rope_op_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_rope_qual)
set_target_properties(ninfer_r9700_rope_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_rope_qual PRIVATE ninfer_r9700_core hip::host)

# Vision position interpolation at the registered D=1152/R=2304 geometry, checked against a
# direct represented-input FP64 oracle on both wave-tiled and CTA execution routes.
add_executable(ninfer_r9700_vision_pos_embed_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/vision_pos_embed_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_vision_pos_embed_qual)
set_target_properties(ninfer_r9700_vision_pos_embed_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_vision_pos_embed_qual PRIVATE ninfer_r9700_core hip::host)

# Segmented non-causal Vision attention at D=72/H=16. Packed and uniform entries, padded QKV
# token strides, input immutability, and BF16 output are checked against one direct FP64 oracle.
add_executable(ninfer_r9700_vision_attention_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/vision_attention_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_vision_attention_qual)
set_target_properties(ninfer_r9700_vision_attention_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_vision_attention_qual PRIVATE ninfer_r9700_core hip::host)

# Public Qwen3.8 Gated DeltaNet recurrence qualifier: ordinary publication, masked snapshots,
# sequential records, and parent-indexed tree records against an independent host FP64 oracle.
add_executable(ninfer_r9700_gdn_recurrence_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/gdn_recurrence_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_gdn_recurrence_qual)
set_target_properties(ninfer_r9700_gdn_recurrence_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_gdn_recurrence_qual PRIVATE ninfer_r9700_core hip::host)

# All-layer ReplaySSM publication qualification at the registered Qwen3.8 geometry. Physical
# record rows target permuted absolute slots and exercise packed-prefix, explicit-path, exact
# no-op, read-only record, and BF16 convolution-tail contracts against an independent host FP64
# recurrence.
add_executable(ninfer_r9700_gdn_replay_fold_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/gdn_replay_fold_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_gdn_replay_fold_qual)
set_target_properties(ninfer_r9700_gdn_replay_fold_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_gdn_replay_fold_qual PRIVATE ninfer_r9700_core hip::host)

# Semantic eager-Op qualification. The executable links the public Tensor/Weight API and the
# native HIP eager kernels, then checks the exact and floating-point contracts on physical gfx1201.
add_executable(ninfer_r9700_eager_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/eager_op_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_eager_qual)
set_target_properties(ninfer_r9700_eager_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_eager_qual PRIVATE ninfer_r9700_core hip::host)

# Public causal-convolution qualification at the real Qwen3.8 GDN geometry. Both disjoint and
# exact in-place history transitions are checked directly against the independent FP64 oracle.
add_executable(ninfer_r9700_gdn_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/gdn_op_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_gdn_qual)
set_target_properties(ninfer_r9700_gdn_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_gdn_qual PRIVATE ninfer_r9700_core hip::host)

# Target-leaf qualification: the Qwen3.8 D256/Hq24/Hkv4 bridge consumes the canonical typed
# cache view and publication state, then runs the admitted finite WMMA/score-streaming crossover.
add_executable(ninfer_r9700_full_attention_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/full_attention_leaf_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_full_attention_qual)
set_target_properties(ninfer_r9700_full_attention_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_full_attention_qual PRIVATE ninfer_r9700_core hip::host)

# Retained numerical and physical regression harness for the admitted split-512 long-context
# production route. It calls the semantic entry point directly to compare both implementations.
add_executable(ninfer_r9700_split512_attention_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/split512_attention_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_split512_attention_qual)
target_include_directories(ninfer_r9700_split512_attention_qual PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_compile_definitions(ninfer_r9700_split512_attention_qual PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
set_target_properties(ninfer_r9700_split512_attention_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_split512_attention_qual PRIVATE ninfer_r9700_core hip::host)

# Qualification-only prefill XAttention evaluator. In an ordinary build it owns the evaluator
# implementation directly. In the private runtime-qualification build it consumes the single
# implementation already compiled into the core so the matched PPL route and evaluator cannot
# diverge or introduce duplicate device-code owners.
add_executable(ninfer_r9700_xattention_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/xattention_prefill_qual.hip)
if(NOT NINFER_R9700_XATTENTION_QUALIFICATION)
  target_sources(ninfer_r9700_xattention_qual PRIVATE
    ops/r9700/kv/fp8_int4_kv_xattention.hip)
endif()
ninfer_r9700_internal_includes(ninfer_r9700_xattention_qual)
target_include_directories(ninfer_r9700_xattention_qual PRIVATE ${PROJECT_SOURCE_DIR}/tests)
set_target_properties(ninfer_r9700_xattention_qual PROPERTIES HIP_ARCHITECTURES gfx1201)
target_compile_definitions(ninfer_r9700_xattention_qual PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
  NINFER_R9700_XATTENTION_STRIDE=${NINFER_R9700_XATTENTION_STRIDE}
  NINFER_R9700_XATTENTION_TAU_PERMILLE=${NINFER_R9700_XATTENTION_TAU_PERMILLE})
target_link_libraries(ninfer_r9700_xattention_qual PRIVATE ninfer_r9700_core hip::host)

# Target Variant GDN composition: direct W8G32 row-split projections feed the native mixed-width
# snapshot and tree-record convolution leaves with caller-owned workspace and real Qwen3.8 shapes.
add_executable(ninfer_r9700_target_variant_gdn_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/target_variant_gdn_qual.cpp)
ninfer_r9700_internal_includes(ninfer_r9700_target_variant_gdn_qual)
set_target_properties(ninfer_r9700_target_variant_gdn_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201
  LINKER_LANGUAGE HIP)
target_link_libraries(ninfer_r9700_target_variant_gdn_qual PRIVATE
  ninfer_r9700_target_bindings hip::host)

# Exact MTP next-round state transition for the complete fixed product domain K=1..5/B=1..4.
# The qualifier checks every output element, pitched padding, input immutability, and public
# shape/stride rejection against an independent host integer oracle on physical gfx1201.
add_executable(ninfer_r9700_mtp_round_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/mtp_round_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_mtp_round_qual)
set_target_properties(ninfer_r9700_mtp_round_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_mtp_round_qual PRIVATE ninfer_r9700_core hip::host)

# Native scalar/scheduling utilities used by the Vision, perplexity, and DFlash paths. Floating
# outputs are checked against independent FP64 formulas and the fixed W/B transforms against exact
# host oracles over the complete scheduling domain, including the real DFlash feature geometry.
add_executable(ninfer_r9700_scalar_schedule_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/scalar_schedule_ops_qual.hip)
ninfer_r9700_internal_includes(ninfer_r9700_scalar_schedule_qual)
set_target_properties(ninfer_r9700_scalar_schedule_qual PROPERTIES
  HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_scalar_schedule_qual PRIVATE ninfer_r9700_core hip::host)

# Retired backend tests and microbenchmarks do not enter this graph. The sole Qwen3.8 runtime,
# public Engine, and product applications above are native HIP link owners;
# real-model execution is still gated on the complete BF16-derived R9700 artifact.
