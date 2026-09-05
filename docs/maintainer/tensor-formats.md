# Persistent tensor numeric formats

The artifact registry contains three direct formats and four signed grouped-integer formats. A
format defines represented values; a storage layout defines byte order and padding.

## Direct formats

- `BF16`: exact IEEE bfloat16 word, two bytes.
- `FP32`: exact IEEE binary32 word, four bytes.
- `I32`: exact signed two's-complement 32-bit word, four bytes.

Direct encoders require the matching source dtype and do not cast implicitly.

## Grouped integer formats

| Format | bits | group | code interval | scale |
|---|---:|---:|---:|---|
| `Q4G64_F16S` | 4 | 64 | -8..7 | one FP16 multiplier |
| `Q5G64_F16S` | 5 | 64 | -16..15 | one FP16 multiplier |
| `Q6G64_F16S` | 6 | 64 | -32..31 | one FP16 multiplier |
| `W8G32_F16S` | 8 | 32 | -127..127 | one FP16 multiplier |

For a represented element `q` in a group with stored scale `s`, the logical value is
`float(q) * float(s)`. W8 deliberately excludes -128 so symmetric nearest-even quantization has one
unambiguous magnitude bound. Source conversion rejects nonfinite values, chooses the group scale
from the selected recipe, rounds ties to even, clamps to the registered code interval, and stores
the final FP16 scale that the oracle uses for reconstruction.

The current provisional R9700 recipe uses W8G32 for matrices and preserves direct BF16, FP32, and
I32 tensors. This is a candidate implementation profile, not a final numerical promise. Each HIP
Op is checked against an independent oracle that decodes the signed code with the exact stored FP16
scale. Final recipe selection requires paired real-source PPL, exact greedy-token, and performance
evidence.

The growing attention cache is not a persistent tensor numeric format. It is runtime state with
three typed planes: FP8 E4M3FN keys, signed INT4 values, and FP16 value scales, as specified in
`paged-kv-cache.md`.
