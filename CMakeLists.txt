cmake_minimum_required(VERSION 3.28)

# Select gfx1201 before HIP compiler discovery; other architectures are unsupported.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1201 CACHE STRING "HIP architectures to build")
endif()
if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1201")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_HIP_ARCHITECTURES=gfx1201; got "
    "'${CMAKE_HIP_ARCHITECTURES}'")
endif()

# Build separate G16/G32 scorers for cache-quality comparison. This setting applies to the
# runtime and Engine at compile time. Remove it and the losing codec after PPL/token selection.
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

# A8 is the baseline Q4 activation codec. The selected gate/up-only large-prefill
# override below uses qualified A4 kernels; decode and other Q4 shapes remain A8.
# Uniform A4 remains a separately built evaluator over identical persistent bytes.
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

set(NINFER_R9700_Q4_PREFILL_A4_FAMILIES "0" CACHE STRING
  "Q4 prefill A4 policy: 0=selected uniform A8; evaluators: 1=gate/up, 2=MLP, 3=projections, 4=attention input, 5=gate/up+attention input")
set_property(CACHE NINFER_R9700_Q4_PREFILL_A4_FAMILIES PROPERTY STRINGS 0 1 2 3 4 5)
if(NOT NINFER_R9700_Q4_PREFILL_A4_FAMILIES MATCHES "^[0-5]$")
  message(FATAL_ERROR "NINFER_R9700_Q4_PREFILL_A4_FAMILIES must be in [0,5]")
endif()
if(NOT NINFER_R9700_Q4_PREFILL_A4_FAMILIES STREQUAL "0" AND NOT NINFER_R9700_Q4_ACTIVATION_BITS STREQUAL "8")
  message(FATAL_ERROR "Mixed prefill A4 families require Q4 activation bits 8")
endif()
add_compile_definitions(
  NINFER_R9700_Q4_PREFILL_A4_FAMILIES=${NINFER_R9700_Q4_PREFILL_A4_FAMILIES})

# Build-time DFlash A/B: 0 uses the WMMA control; 1 enables the small-token N34816/K5120
# candidate cells selected by the two direct screens.
set(NINFER_R9700_DFLASH_SMALL_T_CANDIDATE "0" CACHE STRING
  "Enable the exact qualified R9700 DFlash small-token Linear candidate")
set_property(CACHE NINFER_R9700_DFLASH_SMALL_T_CANDIDATE PROPERTY STRINGS 0 1)
if(NOT NINFER_R9700_DFLASH_SMALL_T_CANDIDATE STREQUAL "0" AND
   NOT NINFER_R9700_DFLASH_SMALL_T_CANDIDATE STREQUAL "1")
  message(FATAL_ERROR
    "NINFER_R9700_DFLASH_SMALL_T_CANDIDATE must be 0 or 1; got "
    "'${NINFER_R9700_DFLASH_SMALL_T_CANDIDATE}'")
endif()
add_compile_definitions(
  NINFER_R9700_DFLASH_SMALL_T_CANDIDATE=${NINFER_R9700_DFLASH_SMALL_T_CANDIDATE})

# Build-time DFlash MLP-down A/B for N5120/K17408/T5 only.
set(NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE "0" CACHE STRING
  "Enable the exact qualified R9700 DFlash MLP-down T5 Linear candidate")
set_property(CACHE NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE PROPERTY STRINGS 0 1)
if(NOT NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE STREQUAL "0" AND
   NOT NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE STREQUAL "1")
  message(FATAL_ERROR
    "NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE must be 0 or 1; got "
    "'${NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE}'")
endif()
add_compile_definitions(
  NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE=${NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE})

# Qualification-only Text P129 WMMA tail candidate. Default zero retains the dense P129 route; one
# admits only the exact Text P129 tail overwrite. The DFlash W5/W6 batched-WMMA route is a
# production route and is not gated by this flag.
set(NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE "0" CACHE STRING
  "Enable the exact R9700 Text P129 WMMA tail candidate")
set_property(CACHE NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE PROPERTY STRINGS 0 1)
if(NOT NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE STREQUAL "0" AND
   NOT NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE STREQUAL "1")
  message(FATAL_ERROR
    "NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE must be 0 or 1; got "
    "'${NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE}'")
endif()
add_compile_definitions(
  NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE=${NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE})

# Qualification-only normalization association selector. Zero retains the snapshot/replay
# 128-element shared-memory tree; one makes those verify paths use the ordinary wave32/FMA
# association. Ordinary execution is unchanged in both builds.
set(NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE "0" CACHE STRING
  "Enable canonical ordinary-wave Q/K normalization in R9700 GDN verify paths")
set_property(CACHE NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE PROPERTY STRINGS 0 1)
if(NOT NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE STREQUAL "0" AND
   NOT NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE STREQUAL "1")
  message(FATAL_ERROR
    "NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE must be 0 or 1; got "
    "'${NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE}'")
endif()
add_compile_definitions(
  NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE=${NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE})


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

# The build host is shared with GPU work and other agents: Ninja never runs more than
# NINFER_BUILD_JOBS compile or link jobs, whatever -j a caller passes (bare Ninja uses cores+2).
set(NINFER_BUILD_JOBS 8 CACHE STRING "Maximum concurrent Ninja compile/link jobs (1..8)")
if(NOT NINFER_BUILD_JOBS MATCHES "^[1-8]$")
  message(FATAL_ERROR "NINFER_BUILD_JOBS must be 1..8")
endif()
set_property(GLOBAL APPEND PROPERTY JOB_POOLS ninfer_jobs=${NINFER_BUILD_JOBS})
set(CMAKE_JOB_POOL_COMPILE ninfer_jobs)
set(CMAKE_JOB_POOL_LINK ninfer_jobs)

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

find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(NINFER_ZSTD REQUIRED IMPORTED_TARGET libzstd)
include(cmake/xgrammar.cmake)
# After the third-party xgrammar target, so its sources keep their upstream diagnostics.
include(cmake/warnings.cmake)
# The vendored CPU compiler/matcher participates in host lifetime qualification.
if(NINFER_SANITIZE)
  target_compile_options(ninfer_token_grammar PRIVATE ${ninfer_sanitizers}
    -fno-omit-frame-pointer -fno-sanitize=vptr)
endif()

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
