#include "targets/qwen3_8_27b/impl/variant.h"

// The exact Qwen3.8-27B R9700 package is the sole owner of the shared family runtime
// instantiation. Keeping these selections in one target-private HIP translation unit makes the
// compiled schedule unambiguous and prevents package.cpp from depending on an accidental test or
// diagnostic instantiation.
#define NINFER_QWEN3_VARIANT ::ninfer::targets::qwen3_8_27b::detail::Variant
#define NINFER_QWEN3_RUNTIME_NS qwen3_8_27b_r9700
#include "targets/qwen3/impl/runtime/instantiate.h"
