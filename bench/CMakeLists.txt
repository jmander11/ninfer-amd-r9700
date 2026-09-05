# Whole-product R9700 benchmarks only. Kernel candidates are qualified and timed by the focused
# physical executables under tools/r9700; retired backend microbenchmarks do not enter this graph.
add_executable(ninfer_bench
  targets/qwen3_8_27b/ninfer_bench.cpp
  targets/qwen3_8_27b/ninfer_bench_support.cpp)
target_include_directories(ninfer_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_compile_definitions(ninfer_bench PRIVATE NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
set_target_properties(ninfer_bench PROPERTIES
  HIP_ARCHITECTURES gfx1201
  LINKER_LANGUAGE HIP)
target_link_libraries(ninfer_bench PRIVATE
  ninfer_engine
  hip::host
  rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library)

# Target-private MTP round benchmark over the same exact Program that the public Engine owns. It
# requires a real artifact and therefore builds now but is executed only at the real-model gate.
add_executable(ninfer_qwen3_8_27b_mtp_round_bench
  targets/qwen3_8_27b/target_mtp_round_bench.cpp)
target_include_directories(ninfer_qwen3_8_27b_mtp_round_bench PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_8_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3/export)
set_target_properties(ninfer_qwen3_8_27b_mtp_round_bench PROPERTIES
  HIP_ARCHITECTURES gfx1201
  LINKER_LANGUAGE HIP)
target_link_libraries(ninfer_qwen3_8_27b_mtp_round_bench PRIVATE
  ninfer_engine
  ninfer_r9700_qwen3_8_27b_runtime
  ninfer_r9700_core
  hip::host)
