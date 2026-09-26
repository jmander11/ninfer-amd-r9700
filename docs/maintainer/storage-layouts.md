# Persistent storage layouts

NInfer v2 registers seven tensor layouts and one resource encoding. Payload offsets are relative to
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

## `r9700-w8g32-n16-k16-v1`

This W8-only rank-two layout requires N divisible by 16 and pads K upward to 128.
It preserves each signed W8 code and each exact FP16 G32 scale word. Codes are
ordered `[N/16][Kpad/32][2 K16 blocks][16 rows][16 bytes]`; scales are
`[N/16][Kpad/32][16 rows]`. The code byte at logical `(row,k)` has offset
`(((((row/16)*groups+k/32)*2+(k%32)/16)*16+row%16)*16+k%16)`.
Its scale word is at `((row/16)*groups+k/32)*16+row%16` in the scale plane.
The code plane starts at zero, the scale plane at the next 256-byte boundary;
plane sizes and total encoded bytes equal row-split W8. This is a storage-only
permutation, not a numerical-format or quantization-recipe change.

The selective-protected canonical-Q4 DFlash companion requires this layout for
`text/output_head` only. All its consumers use the same resident head; no loader
repack, duplicate weight, or runtime layout selector is permitted. Other W8
objects and evaluation profiles retain their separately specified row-split layout.

## `row-scaled-k128-v1`

This layout accepts rank-two `F8E4M3_ROW_F32S` matrices `[N,K]`. It pads K upward to 128 and
stores the row-major E4M3FN code plane `[N][Kpad]` at offset zero (padding codes zero), then the
little-endian FP32 row multipliers `[N]` at the next 256-byte boundary.

## `r9700-cb4g32-n16k64-v1`

This layout accepts rank-two `CB4G32_F32S` matrices `[N,K]` with `N` divisible by 16 and pads K
upward to 128. The code and group planes are ordered in N16 x K64 tiles: tile index
`(r / 16) * (Kpad / 64) + k / 64`, and within a tile slot `16 * ((k % 64) / 32) + r % 16`.

1. Code plane at offset zero: 512 bytes per tile, 16 bytes per slot holding the 32 four-bit codes
   of that row's 32-column group, low nibble for the even column.
2. Group plane at the next 256-byte boundary: 32 bytes per tile, one group code byte per slot.
3. FP32 row multipliers `[N]` at the next 256-byte boundary.

A wave of the small-T kernel reads one contiguous 512-byte code tile per 64-column step; the
prefill kernel stages 4 KiB of contiguous tiles per 128-row slab. Plane sizes equal a row-major
layout; the tiling is a storage permutation only.

## `raw-bytes-v1`

Resources are exact nonempty byte strings with alignment one. The target inventory uses it for the
six frontend tokenizer/template/preprocessor resources. Resource bytes are retained on the host;
tensor bytes are materialized to the device arena.
