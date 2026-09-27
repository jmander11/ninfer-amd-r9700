# Persistent tensor numeric formats

The artifact registry contains three direct formats, four signed grouped-integer formats, one
row-scaled E4M3 format and one grouped codebook format. A format defines represented values; a
storage layout defines byte order and padding.

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

## Row-scaled E4M3

`F8E4M3_ROW_F32S`: one OCP E4M3FN code per element and one finite, nonnegative FP32 multiplier per
row; the logical value is `decode_e4m3fn(code) * row_scale`.

## Grouped codebook: `FP8LUT4`

Four-bit sign-magnitude codes (bit 3 sign, bits 0..2 magnitude index `j`), one group code byte per
32 columns and one finite, nonnegative FP32 multiplier `R` per row. Group code `b` selects
`E = (b >> 3) - 26` and `m = b & 7`; its eight magnitudes are the E4M3FN round-to-nearest-even
(saturating at 448) of `n_j * (8 + m) * 2^(E - 7)` with the fixed base
`n = {0, 13, 27, 41, 56, 74, 94, 120}` (sixteenths: a zero level plus a Lloyd fit of
group-normalized weights). The logical value is `+-magnitude(b, j) * R`. Every decoded magnitude
is an exact E4M3 value, so the Linear folds each group's scale into the FP8 weight operand and
applies only `R` and the per-token activation scale after the K sum.

Source conversion (`tools/convert/qwen3_8_27b_r9700/fp8lut4_codec.py`) sets `R = max|row| / 120` and,
for every group, chooses `(E, m)` among `E` in `{e0 - 1, e0, e0 + 1}` (e0 from the group maximum)
and all eight `m` by minimum decoded squared error, each element taking its nearest magnitude. On
the Qwen3.8 BF16 weights this has 0.82x the relative L2 error of Q4G64 absmax at the same 4.25
bits per weight. FP8LUT4 Linears consume per-token E4M3 activation images (codes, FP32 token scales,
one status word per token; a flagged token's outputs are the canonical BF16 NaN).

The growing attention cache is not a persistent tensor numeric format. It is runtime state with
three typed planes: FP8 E4M3FN keys, signed INT4 values, and FP16 value scales, as specified in
`paged-kv-cache.md`.
