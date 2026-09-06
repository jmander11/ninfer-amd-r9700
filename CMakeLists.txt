cmake_minimum_required(VERSION 3.28)

# NInfer is being rebuilt as one HIP/gfx1201 product. Set this before HIP compiler
# discovery: accepting a generic host-selected code object would make a successful
# configure unrelated to the only supported R9700 deployment.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1201 CACHE STRING "HIP architectures to build")
endif()
if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1201")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_HIP_ARCHITECTURES=gfx1201; got "
    "'${CMAKE_HIP_ARCHITECTURES}'")
endif()

# The real-model cache accuracy gate requires separately built G16 and G32 scorers. This temporary
# configure-time qualification value is compiled through the complete runtime/Engine graph; it is
# deliberately not a runtime, CLI, or artifact selector. Once the real PPL/exact-token gate chooses
# one group, delete this cache variable and the losing codec branch.
set(NINFER_R9700_KV_VALUE_GROUP "16" CACHE STRING
  "R9700 FP8-K/INT4-V value group for separately built qualification binaries")
set_property(CACHE NINFER_R9700_KV_VALUE_GROUP PROPERTY STRINGS 16 32)
if(NOT NINFER_R9700_KV_VALUE_GROUP STREQUAL "16" AND
   NOT NINFER_R9700_KV_VALUE_GROUP STREQUAL "32")
  message(FATAL_ERROR
    "NINFER_R9700_KV_VALUE_GROUP must be 16 or 32 for the pending real-model gate; got "
    "'${NINFER_R9700_KV_VALUE_GROUP}'")
endif()
add_compile_definitions(
  NINFER_R9700_KV_VALUE_GROUP=${NINFER_R9700_KV_VALUE_GROUP})

# A8 is the production-style execution profile for every Q4-bearing artifact: it materially
# reduces activation error while retaining native IU4 WMMA and large speed margins over W8.
# A4 remains available only as a separately built evaluator over identical persistent bytes.
# This value is reported in scorer/benchmark output and is not a runtime or artifact selector.
set(NINFER_R9700_Q4_ACTIVATION_BITS "8" CACHE STRING
  "R9700 Q4 evaluation activation width for separately built scorer binaries")
set_property(CACHE NINFER_R9700_Q4_ACTIVATION_BITS PROPERTY STRINGS 4 8)
if(NOT NINFER_R9700_Q4_ACTIVATION_BITS STREQUAL "4" AND
   NOT NINFER_R9700_Q4_ACTIVATION_BITS STREQUAL "8")
  message(FATAL_ERROR
    "NINFER_R9700_Q4_ACTIVATION_BITS must be 4 or 8; got "
    "'${NINFER_R9700_Q4_ACTIVATION_BITS}'")
endif()
add_compile_definitions(
  NINFER_R9700_Q4_ACTIVATION_BITS=${NINFER_R9700_Q4_ACTIVATION_BITS})

# Private build-only switch for source-matched scalar-base qualification. It is
# deliberately absent from the CLI/artifact/runtime contracts.
option(NINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION
  "Route selected Q4 prefill calls through the scalar-base qualifier" OFF)
if(NINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION)
  add_compile_definitions(NINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION=1)
endif()

# The selected W8G32 profile uses dynamic A8G32 only at measured winning shapes/token extents and
# retains exact represented-BF16 activation below each crossover. The explicit A16 build is the
# matched quality/performance control over identical persistent W8G32 bytes.
set(NINFER_R9700_W8_ACTIVATION_BITS "8" CACHE STRING
  "R9700 W8 activation width for separately built quality evaluator binaries")
set_property(CACHE NINFER_R9700_W8_ACTIVATION_BITS PROPERTY STRINGS 8 16)
if(NOT NINFER_R9700_W8_ACTIVATION_BITS STREQUAL "8" AND
   NOT NINFER_R9700_W8_ACTIVATION_BITS STREQUAL "16")
  message(FATAL_ERROR
    "NINFER_R9700_W8_ACTIVATION_BITS must be 8 or 16; got "
    "'${NINFER_R9700_W8_ACTIVATION_BITS}'")
endif()
add_compile_definitions(
  NINFER_R9700_W8_ACTIVATION_BITS=${NINFER_R9700_W8_ACTIVATION_BITS})

# The selected attention profile uses native FP8-Q/FP8-K WMMA for ordinary causal T1 at
# context>=64 and T2 at context>=320. Zero builds the represented-BF16-Q score-streaming control
# for isolated PPL comparison; neither value is exposed as an artifact or runtime option.
set(NINFER_R9700_FP8_QK_WMMA "1" CACHE STRING
  "Enable the qualified R9700 FP8-Q/K WMMA attention context/row crossover")
set_property(CACHE NINFER_R9700_FP8_QK_WMMA PROPERTY STRINGS 0 1)
if(NOT NINFER_R9700_FP8_QK_WMMA STREQUAL "0" AND
   NOT NINFER_R9700_FP8_QK_WMMA STREQUAL "1")
  message(FATAL_ERROR
    "NINFER_R9700_FP8_QK_WMMA must be 0 or 1; got "
    "'${NINFER_R9700_FP8_QK_WMMA}'")
endif()
add_compile_definitions(NINFER_R9700_FP8_QK_WMMA=${NINFER_R9700_FP8_QK_WMMA})

# XAttention remains qualification-only. The stride/tau values select the standalone evaluator;
# the OFF-by-default qualification option additionally compiles that fixed profile through a
# private Text-prefill runtime/PPL build. The ordinary product build sees neither implementation
# nor selector.
set(NINFER_R9700_XATTENTION_STRIDE "16" CACHE STRING
  "R9700 XAttention qualification stride")
set_property(CACHE NINFER_R9700_XATTENTION_STRIDE PROPERTY STRINGS 8 16)
if(NOT NINFER_R9700_XATTENTION_STRIDE STREQUAL "8" AND
   NOT NINFER_R9700_XATTENTION_STRIDE STREQUAL "16")
  message(FATAL_ERROR
    "NINFER_R9700_XATTENTION_STRIDE must be 8 or 16; got "
    "'${NINFER_R9700_XATTENTION_STRIDE}'")
endif()
set(NINFER_R9700_XATTENTION_TAU_PERMILLE "1000" CACHE STRING
  "R9700 XAttention qualification tau in integer permille")
if(NOT NINFER_R9700_XATTENTION_TAU_PERMILLE MATCHES "^[0-9]+$" OR
   NINFER_R9700_XATTENTION_TAU_PERMILLE LESS 1 OR
   NINFER_R9700_XATTENTION_TAU_PERMILLE GREATER 1000)
  message(FATAL_ERROR
    "NINFER_R9700_XATTENTION_TAU_PERMILLE must be an integer in [1,1000]; got "
    "'${NINFER_R9700_XATTENTION_TAU_PERMILLE}'")
endif()
option(NINFER_R9700_XATTENTION_QUALIFICATION
  "Build the private R9700 Text-prefill XAttention qualification route" OFF)
if(NINFER_R9700_XATTENTION_QUALIFICATION AND
   (NOT NINFER_R9700_XATTENTION_STRIDE STREQUAL "16" OR
    NOT NINFER_R9700_XATTENTION_TAU_PERMILLE STREQUAL "900"))
  message(FATAL_ERROR
    "The private Text-prefill XAttention route is the selected B128/S16/tau900 qualification "
    "profile; use the standalone evaluator for other stride/tau experiments")
endif()

# CMake intentionally rejects the hipcc wrapper as a HIP compiler. Resolve the active ROCm
# Clang through the versioned /opt/rocm selector before `project()` so an ordinary configure uses
# the same selected ROCm installation as the standalone qualification tools.
if(NOT DEFINED CMAKE_HIP_COMPILER)
  find_program(NINFER_ROCM_CLANGXX NAMES clang++
    HINTS /opt/rocm/llvm/bin
    NO_DEFAULT_PATH REQUIRED)
  set(CMAKE_HIP_COMPILER "${NINFER_ROCM_CLANGXX}" CACHE FILEPATH
    "ROCm Clang used for the sole gfx1201 HIP product" FORCE)
endif()

project(ninfer LANGUAGES C CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# HIP's compiler package supplies the runtime link target and selected ROCm include
# roots. The production core links it directly without a source-level compatibility shim.
find_package(hip REQUIRED CONFIG)
find_package(hipblaslt REQUIRED CONFIG)
find_package(rocprofiler-sdk-roctx REQUIRED CONFIG)
get_target_property(NINFER_AMDHIP64_LOCATION hip::amdhip64 IMPORTED_LOCATION_RELEASE)
if(NOT NINFER_AMDHIP64_LOCATION)
  get_target_property(NINFER_AMDHIP64_LOCATION hip::amdhip64 IMPORTED_LOCATION)
endif()
if(NOT NINFER_AMDHIP64_LOCATION)
  message(FATAL_ERROR "ROCm HIP runtime target has no imported shared-library location")
endif()
get_filename_component(NINFER_HIP_RUNTIME_DIR "${NINFER_AMDHIP64_LOCATION}" DIRECTORY)
# hip::host supplies link dependencies but does not give local build executables an rpath on
# this ROCm installation. Keep the executable tied to the same selected runtime that CMake found.
set(CMAKE_BUILD_RPATH "${NINFER_HIP_RUNTIME_DIR}")

# The sole R9700 Engine and product applications are native HIP owners. Tests use the same
# production targets and physical gfx1201 qualification executables; there is no compatibility
# backend or alternate device graph.
option(NINFER_BUILD_APPS "Build the R9700 CLI, server, and PPL applications" ON)
option(NINFER_BUILD_BENCHMARKS "Build the R9700 whole-Engine benchmarks" OFF)
include(CTest)

add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
