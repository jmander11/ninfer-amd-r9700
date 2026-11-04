# CPU-only grammar dependency, with its selected upstream provenance and native adaptations
# recorded in third_party/xgrammar/README.ninfer.md. No Python or device runtime is linked.
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/xgrammar" "${CMAKE_CURRENT_BINARY_DIR}/xgrammar")
