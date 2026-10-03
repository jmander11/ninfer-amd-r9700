# Compiler diagnostics and sanitizers for project-owned C++ and HIP.
#
# Included after third-party targets are declared, so the directory-scoped options below apply
# only to NInfer targets (src/, apps/, tests/, bench/). Policy and rationale:
# docs/maintainer/code-quality.md.

option(NINFER_WARNINGS_AS_ERRORS "Treat compiler warnings in project code as errors" ON)
set(NINFER_SANITIZE "" CACHE STRING
  "Host sanitizers for project code, e.g. address,undefined (use a separate build tree)")
# Sanitizer instrumentation makes GCC's flow-based warnings (such as -Warray-bounds) report paths
# that do not exist, so an instrumented tree reports warnings without failing; the
# uninstrumented tree is the warning gate.
if(NINFER_SANITIZE AND NINFER_WARNINGS_AS_ERRORS)
  message(STATUS "NINFER_SANITIZE is set: compiler warnings are not errors in this tree")
  set(NINFER_WARNINGS_AS_ERRORS OFF)
endif()

# Each flag targets a defect class both toolchains (GCC 13 for host C++, ROCm clang for HIP)
# report reliably: shadowed locals, missing virtual destructors, hidden overloads, unannotated
# switch fallthrough, unhandled enumerators, format-string mismatch, unused code, and member
# initializer order (-Wreorder, in -Wall). Not enabled:
# - conversion warnings: kernel index arithmetic is checked by clang-tidy's
#   bugprone-implicit-widening-* instead;
# - -Wnull-dereference: GCC 13 infers null paths inside libstdc++ shared_ptr and streambuf after
#   inlining; clang-tidy's path-sensitive clang-analyzer-core.NullDereference covers the class;
# - -Wdouble-promotion: host code only, where it fires on float varargs;
# - -Wmissing-field-initializers: aggregate initialization that relies on default member
#   initializers is the intended idiom.
set(ninfer_common_warnings
  -Wall
  -Wextra
  -Wshadow
  -Wnon-virtual-dtor
  -Woverloaded-virtual
  -Wimplicit-fallthrough
  -Wformat=2
  -Wmisleading-indentation
  -Wno-missing-field-initializers)
set(ninfer_host_warnings ${ninfer_common_warnings})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND ninfer_host_warnings -Wduplicated-cond -Wduplicated-branches -Wlogical-op)
endif()
# HIP translation units compile host and gfx1201 device code with ROCm clang. The host pass does
# not count uses inside kernel bodies, so it reports every device-only helper and constant as
# unused; the device pass still reports device functions and constants that are really unused.
set(ninfer_hip_warnings ${ninfer_common_warnings}
  "SHELL:-Xarch_host -Wno-unused-function"
  "SHELL:-Xarch_host -Wno-unused-const-variable")
if(NINFER_WARNINGS_AS_ERRORS)
  list(APPEND ninfer_host_warnings -Werror)
  list(APPEND ninfer_hip_warnings -Werror)
endif()

add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:${ninfer_host_warnings}>"
  "$<$<COMPILE_LANGUAGE:HIP>:${ninfer_hip_warnings}>")

if(NINFER_SANITIZE)
  string(REPLACE "," ";" ninfer_sanitizers "${NINFER_SANITIZE}")
  list(TRANSFORM ninfer_sanitizers PREPEND "-fsanitize=")
  # gfx1201 has no xnack, so ROCm clang cannot instrument device code; HIP translation units
  # instrument their host compilation only. Device checks: tools/r9700/gpu_check.
  # SHELL: keeps each -Xarch_host pair intact through CMake's option de-duplication.
  set(ninfer_hip_sanitize_flags)
  foreach(flag IN LISTS ninfer_sanitizers ITEMS -fno-omit-frame-pointer)
    list(APPEND ninfer_hip_sanitize_flags "SHELL:-Xarch_host ${flag}")
  endforeach()
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX>:${ninfer_sanitizers};-fno-omit-frame-pointer>"
    "$<$<COMPILE_LANGUAGE:HIP>:${ninfer_hip_sanitize_flags}>")
  add_link_options(${ninfer_sanitizers})
endif()
