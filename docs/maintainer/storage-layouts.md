# Persistent storage layouts

NInfer v2 registers four tensor layouts and one resource encoding. Payload offsets are relative to
the container payload and tensor starts are 256-byte aligned.

## `contiguous-le-v1`

This layout accepts `BF16`, `FP32`, and `I32`, rank 0 through 16, with positive dimensions. Logical
row-major elements are stored as their exact little-endian words without padding inside the object.
The encoded byte count is `product(shape) * word_bytes`.

## `row-split-k128-v1`

This layout accepts rank-two grouped signed-integer matrices `[N,K]` in `Q5G64_F16S`,
`Q6G64_F16S`, or `W8G32_F16S`. It pads K upward to 128 and stores three planes:

1. low code plane at offset zero;
2. optional high-bit plane at the next 256-byte boundary;
3. little-endian FP16 scale plane at the next 256-byte boundary after the aligned high plane.

For each row and group, low nibbles pack consecutive signed two's-complement codes with the even
element in bits 0..3. Q5/Q6 high bits are bit-packed in logical element order. W8 stores one signed
byte per element. Each group has one finite, nonnegative FP16 multiplier. Padding codes are zero and
their groups retain a valid scale.

The registered group and plane widths are:

| Format | group | low bytes/group | high bytes/group | scale bytes/group |
|---|---:|---:|---:|---:|
| `Q5G64_F16S` | 64 | 32 | 8 | 2 |
| `Q6G64_F16S` | 64 | 32 | 16 | 2 |
| `W8G32_F16S` | 32 | 32 | 0 | 2 |

The logical reconstruction is `float(code) * float(fp16_scale)` at each element. The R9700
candidate uses W8G32 for every persistent quantized matrix and consumes these planes directly.

## `r9700-q4g64-n16-k16-v1`

This Q4-only rank-two layout requires `N` divisible by 16 and pads K upward to 128. It preserves
the exact signed-Q4 codes, FP16 group scales, plane sizes, and reconstruction semantics while
ordering each code plane as `[N/16][Kpad/64][4 K16 pairs][16 rows][8 bytes]` and the scale plane as
`[N/16][Kpad/64][16 rows]`. Thus code pair `(row, group, pair)` is the little-endian u64 at
`((((row/16)*groups+group)*4+pair)*16+(row%16))`; its scale is the u16 at
`(((row/16)*groups+group)*16+(row%16))`. The code plane starts at zero and the scale plane begins at
the next 256-byte boundary. This is the sole accepted persistent layout for `Q4G64_F16S`; legacy
row-split Q4 descriptors are rejected. Conversion or the explicit offline transcoder writes this
order once, and loading performs no repack.

## `raw-bytes-v1`

Resources are exact nonempty byte strings with alignment one. The target inventory uses it for the
six frontend tokenizer/template/preprocessor resources. Resource bytes are retained on the host;
tensor bytes are materialized to the device arena.
