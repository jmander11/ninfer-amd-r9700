find_package(Python3 REQUIRED COMPONENTS Interpreter)

function(ninfer_add_r9700_test name)
  cmake_parse_arguments(arg "" "" "SOURCES;LIBRARIES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3/export
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3/impl
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_8_27b/export
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_8_27b/impl
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
  target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# The public headers must remain independent of repository-private and device headers.
add_executable(ninfer_public_api_test test_public_api.cpp)
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)

# Independent host authority for the only growing-cache representation.
add_executable(ninfer_fp8_int4_kv_oracle_test ops/test_fp8_int4_kv_oracle.cpp)
target_include_directories(ninfer_fp8_int4_kv_oracle_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(ninfer_fp8_int4_kv_oracle_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
  $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
add_test(NAME ninfer_fp8_int4_kv_oracle_test COMMAND ninfer_fp8_int4_kv_oracle_test)

add_executable(ninfer_xattention_oracle_test ops/test_xattention_oracle.cpp)
target_include_directories(ninfer_xattention_oracle_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(ninfer_xattention_oracle_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
  $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
add_test(NAME ninfer_xattention_oracle_test COMMAND ninfer_xattention_oracle_test)

add_test(NAME ninfer_fp8_gate_up_comparison_report_test
  COMMAND ${Python3_EXECUTABLE} -m unittest -q
    tools.bench.test_validate_fp8_gate_up_comparison)
set_tests_properties(ninfer_fp8_gate_up_comparison_report_test PROPERTIES
  LABELS "host"
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(NAME ninfer_fp8_gate_up_hardware_proof_host_test
  COMMAND ${Python3_EXECUTABLE} -m unittest -q
    tools.bench.test_prepare_fp8_gate_up_hardware_proof
    tools.bench.test_validate_fp8_gate_up_hardware_proof)
set_tests_properties(ninfer_fp8_gate_up_hardware_proof_host_test PROPERTIES
  LABELS "host"
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# Model-independent framing and supported host/runtime policies.
ninfer_add_r9700_test(ninfer_artifact_reader_test
  SOURCES test_ninfer_artifact_reader.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_r9700_test(ninfer_tensor_test
  SOURCES test_tensor.cpp
  LIBRARIES ninfer_r9700_core)
ninfer_add_r9700_test(ninfer_admission_policy_test
  SOURCES test_admission_policy.cpp
  LIBRARIES ninfer_engine ninfer_r9700_core)
ninfer_add_r9700_test(ninfer_sampling_defaults_test
  SOURCES test_sampling_defaults.cpp
  LIBRARIES ninfer_engine)
ninfer_add_r9700_test(ninfer_gdn_replay_records_test
  SOURCES test_gdn_replay_records.cpp
  LIBRARIES ninfer_r9700_core)
ninfer_add_r9700_test(ninfer_r9700_linear_prefill_dispatch_test
  SOURCES test_r9700_linear_prefill_dispatch.cpp)
ninfer_add_r9700_test(ninfer_r9700_a8q4_scalar_base_bounds_test
  SOURCES ${PROJECT_SOURCE_DIR}/tools/r9700/test_a8q4_scalar_base_bounds.cpp
  LIBRARIES ninfer_r9700_core)
set_tests_properties(ninfer_r9700_a8q4_scalar_base_bounds_test PROPERTIES LABELS "host")

ninfer_add_r9700_test(ninfer_r9700_fp8_activation_contract_test
  SOURCES test_r9700_fp8_activation_contract.cpp
  LIBRARIES ninfer_r9700_core)
ninfer_add_r9700_test(ninfer_r9700_fp8_execution_state_contract_test
  SOURCES test_r9700_fp8_execution_state_contract.cpp
  LIBRARIES ninfer_r9700_target_bindings ninfer_r9700_core)

# Family CPU behavior retained by the sole Qwen3.8-27B target.
ninfer_add_r9700_test(ninfer_qwen3_runtime_mechanisms_test
  SOURCES targets/qwen3/test_runtime_mechanisms.cpp
  LIBRARIES ninfer_engine ninfer_r9700_core)
ninfer_add_r9700_test(ninfer_qwen3_score_index_test
  SOURCES targets/qwen3/test_score_index.cpp
  LIBRARIES ninfer_engine)

# Product CLI and external serving contracts.
if(TARGET ninfer_serve)
  foreach(contract IN ITEMS
      openai_schema
      responses_schema
      response_store
      anthropic_schema
      tool_call_parser
      serve_options
      request_log)
    ninfer_add_r9700_test(ninfer_${contract}_test
      SOURCES test_${contract}.cpp
      LIBRARIES ninfer_serve)
  endforeach()
  ninfer_add_r9700_test(ninfer_http_error_handler_test
    SOURCES test_http_error_handler.cpp
    LIBRARIES ninfer_serve)
  target_include_directories(ninfer_http_error_handler_test PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
endif()

if(TARGET ninfer)
  ninfer_add_r9700_test(ninfer_cli_options_test
    SOURCES test_cli_options.cpp ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp
    LIBRARIES ninfer_engine)
  target_include_directories(ninfer_cli_options_test PRIVATE ${PROJECT_SOURCE_DIR}/apps)
endif()

ninfer_add_r9700_test(ninfer_bench_support_test
  SOURCES
    test_ninfer_bench_support.cpp
    ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_8_27b/ninfer_bench_support.cpp
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_bench_support_test PRIVATE
  ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_8_27b)
target_compile_definitions(ninfer_bench_support_test PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# Foundational ISA/codec/linear qualifications are tool-owned but use the production core where
# an implementation exists. Their CMake targets keep the ordinary test graph self-contained.
add_executable(ninfer_r9700_instruction_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/r9700_qual.hip)
target_include_directories(ninfer_r9700_instruction_qual PRIVATE
  ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests)
set_target_properties(ninfer_r9700_instruction_qual PROPERTIES HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_instruction_qual PRIVATE hip::host)

add_executable(ninfer_r9700_kv_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/kv_op_qual.hip)
target_include_directories(ninfer_r9700_kv_qual PRIVATE
  ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests)
set_target_properties(ninfer_r9700_kv_qual PROPERTIES HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_kv_qual PRIVATE ninfer_r9700_core hip::host)

add_executable(ninfer_r9700_linear_qual
  ${PROJECT_SOURCE_DIR}/tools/r9700/linear_op_qual.hip)
target_include_directories(ninfer_r9700_linear_qual PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
set_target_properties(ninfer_r9700_linear_qual PROPERTIES HIP_ARCHITECTURES gfx1201)
target_link_libraries(ninfer_r9700_linear_qual PRIVATE ninfer_r9700_core hip::host)

# The registry probe is declared with the core qualifiers and becomes a linked observable test
# only when the public Engine target is available.
target_link_libraries(ninfer_r9700_registry_qual PRIVATE ninfer_engine)

set(ninfer_r9700_valid_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-candidate-valid.ninfer)
set(ninfer_r9700_invalid_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-candidate-invalid-format.ninfer)
set(ninfer_r9700_q4_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-q4-evaluation-valid.ninfer)
set(ninfer_r9700_fp8_q4_hybrid_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-fp8-q4-hybrid-evaluation-valid.ninfer)
set(ninfer_r9700_q4_w8_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-q4-w8-n16k16-evaluation-valid.ninfer)
set(ninfer_r9700_q4_w8_mse_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-q4-w8-mse-n16k16-evaluation-valid.ninfer)
set(ninfer_r9700_w8_bf16_embedding_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-w8-bf16-embedding-evaluation-valid.ninfer)
set(ninfer_r9700_w8_bf16_attention_vo_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-w8-bf16-attention-vo-evaluation-valid.ninfer)
set(ninfer_r9700_w8_bf16_attention_qk_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-w8-bf16-attention-qk-evaluation-valid.ninfer)
set(ninfer_r9700_w8_bf16_gdn_qk_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-w8-bf16-gdn-qk-evaluation-valid.ninfer)
set(ninfer_r9700_w8_mse_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-w8-mse-evaluation-valid.ninfer)
set(ninfer_r9700_q4_dflash2_q4_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-q4-dflash2-q4-evaluation-valid.ninfer)
set(ninfer_r9700_q4_w8_mse_dflash2_q4_sparse_artifact
  ${CMAKE_CURRENT_BINARY_DIR}/r9700-q4-w8-mse-n16k16-dflash2-q4-evaluation-valid.ninfer)
add_test(NAME ninfer_r9700_target_binding_valid_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --out ${ninfer_r9700_valid_sparse_artifact})
add_test(NAME ninfer_r9700_target_binding_invalid_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --wrong-token-format --out ${ninfer_r9700_invalid_sparse_artifact})
add_test(NAME ninfer_r9700_target_binding_q4_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --profile q4 --out ${ninfer_r9700_q4_sparse_artifact})
add_test(NAME ninfer_r9700_target_binding_fp8_q4_hybrid_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --profile fp8-q4-hybrid --out ${ninfer_r9700_fp8_q4_hybrid_sparse_artifact})
add_test(NAME ninfer_r9700_target_binding_q4_w8_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --profile q4-w8 --out ${ninfer_r9700_q4_w8_sparse_artifact})
add_test(NAME ninfer_r9700_target_binding_q4_w8_mse_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --profile q4-w8-mse --out ${ninfer_r9700_q4_w8_mse_sparse_artifact})
add_test(NAME ninfer_r9700_target_binding_w8_bf16_embedding_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --profile w8-bf16-embedding --out ${ninfer_r9700_w8_bf16_embedding_sparse_artifact})
add_test(NAME ninfer_r9700_target_binding_w8_bf16_attention_vo_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --profile w8-bf16-attention-vo
    --out ${ninfer_r9700_w8_bf16_attention_vo_sparse_artifact})
add_test(NAME ninfer_r9700_target_binding_w8_bf16_attention_qk_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --profile w8-bf16-attention-qk
    --out ${ninfer_r9700_w8_bf16_attention_qk_sparse_artifact})
add_test(NAME ninfer_r9700_target_binding_w8_bf16_gdn_qk_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --profile w8-bf16-gdn-qk
    --out ${ninfer_r9700_w8_bf16_gdn_qk_sparse_artifact})
add_test(NAME ninfer_r9700_target_binding_w8_mse_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --profile w8-mse --out ${ninfer_r9700_w8_mse_sparse_artifact})
add_test(NAME ninfer_r9700_target_binding_q4_dflash2_q4_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --profile q4-dflash2-q4 --out ${ninfer_r9700_q4_dflash2_q4_sparse_artifact})
add_test(NAME ninfer_r9700_target_binding_q4_w8_mse_dflash2_q4_fixture
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/r9700/make_sparse_r9700_candidate.py
    --replace --profile q4-w8-mse-dflash2-q4
    --out ${ninfer_r9700_q4_w8_mse_dflash2_q4_sparse_artifact})
set_tests_properties(
  ninfer_r9700_target_binding_valid_fixture
  ninfer_r9700_target_binding_invalid_fixture
  ninfer_r9700_target_binding_q4_fixture
  ninfer_r9700_target_binding_fp8_q4_hybrid_fixture
  ninfer_r9700_target_binding_q4_w8_fixture
  ninfer_r9700_target_binding_q4_w8_mse_fixture
  ninfer_r9700_target_binding_w8_bf16_embedding_fixture
  ninfer_r9700_target_binding_w8_bf16_attention_vo_fixture
  ninfer_r9700_target_binding_w8_bf16_attention_qk_fixture
  ninfer_r9700_target_binding_w8_bf16_gdn_qk_fixture
  ninfer_r9700_target_binding_w8_mse_fixture
  ninfer_r9700_target_binding_q4_dflash2_q4_fixture
  ninfer_r9700_target_binding_q4_w8_mse_dflash2_q4_fixture
  PROPERTIES FIXTURES_SETUP r9700_target_binding)
add_test(NAME ninfer_r9700_target_binding_qual
  COMMAND ninfer_r9700_target_binding_qual
    ${ninfer_r9700_valid_sparse_artifact}
    ${ninfer_r9700_invalid_sparse_artifact}
    ${ninfer_r9700_q4_sparse_artifact}
    ${ninfer_r9700_q4_w8_sparse_artifact}
    ${ninfer_r9700_q4_w8_mse_sparse_artifact}
    ${ninfer_r9700_w8_bf16_embedding_sparse_artifact}
    ${ninfer_r9700_w8_bf16_attention_vo_sparse_artifact}
    ${ninfer_r9700_w8_bf16_gdn_qk_sparse_artifact}
    ${ninfer_r9700_w8_bf16_attention_qk_sparse_artifact}
    ${ninfer_r9700_w8_mse_sparse_artifact}
    ${ninfer_r9700_q4_dflash2_q4_sparse_artifact}
    ${ninfer_r9700_q4_w8_mse_dflash2_q4_sparse_artifact}
    ${ninfer_r9700_fp8_q4_hybrid_sparse_artifact})
set_tests_properties(ninfer_r9700_target_binding_qual PROPERTIES
  FIXTURES_REQUIRED r9700_target_binding
  LABELS "r9700"
  RESOURCE_LOCK r9700_device
  TIMEOUT 300)

# Register the production-owned gfx1201 qualifications. These are the device tests: each owns an
# independent exact or FP64 oracle and runs the same HIP implementation linked into the Engine.
set(ninfer_r9700_qualifiers
  ninfer_r9700_core_qual
  ninfer_r9700_artifact_qual
  ninfer_r9700_kv_capacity_qual
  ninfer_r9700_registry_qual
  ninfer_r9700_instruction_qual
  ninfer_r9700_kv_qual
  ninfer_r9700_linear_qual
  ninfer_r9700_state_qual
  ninfer_r9700_dflash_state_qual
  ninfer_r9700_dflash2_select_qual
  ninfer_r9700_bidirectional_gqa_qual
  ninfer_r9700_grouped_dynamic_conv_qual
  ninfer_r9700_swa_qual
  ninfer_r9700_kv_cache_append_prefix_qual
  ninfer_r9700_kv_ram_qual
  ninfer_r9700_sampling_qual
  ninfer_r9700_speculative_round_qual
  ninfer_r9700_rope_qual
  ninfer_r9700_vision_pos_embed_qual
  ninfer_r9700_vision_attention_qual
  ninfer_r9700_gdn_recurrence_qual
  ninfer_r9700_gdn_replay_fold_qual
  ninfer_r9700_eager_qual
  ninfer_r9700_gdn_qual
  ninfer_r9700_full_attention_qual
  ninfer_r9700_xattention_qual
  ninfer_r9700_target_variant_gdn_qual
  ninfer_r9700_mtp_round_qual
  ninfer_r9700_scalar_schedule_qual
  ninfer_r9700_runtime_planner_qual
  ninfer_r9700_engine_boundary_qual)
foreach(qualifier IN LISTS ninfer_r9700_qualifiers)
  if(TARGET ${qualifier})
    add_test(NAME ${qualifier} COMMAND ${qualifier})
    set_tests_properties(${qualifier} PROPERTIES
      LABELS "r9700"
      RESOURCE_LOCK r9700_device
      TIMEOUT 300)
  endif()
endforeach()

if(TARGET ninfer_r9700_runtime_planner_qual)
  add_test(NAME ninfer_r9700_request_lane_cap_host
    COMMAND ninfer_r9700_runtime_planner_qual --host-request-lane-cap)
  set_tests_properties(ninfer_r9700_request_lane_cap_host PROPERTIES LABELS "host")
  add_test(NAME ninfer_r9700_split512_routing_host
    COMMAND ninfer_r9700_runtime_planner_qual --host-split512-routing)
  set_tests_properties(ninfer_r9700_split512_routing_host PROPERTIES LABELS "host")
endif()
