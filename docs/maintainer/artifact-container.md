# Artifact container

`.ninfer` v2 is the sole C++ product artifact. The container owns framing, an exact JSON object
directory, aligned payload ranges, and the `(model_id, weights_id)` identity. Target code owns the
meaning and complete inventory of those objects.

## Framing

The file begins with the eight bytes `NINFER\0\2`, followed by a little-endian unsigned 64-bit
directory byte count, then UTF-8 JSON. The payload begins at the next 4096-byte boundary. Every
object offset is relative to that payload start. Readers reject v1 magic, malformed or extra JSON
members, duplicate names, overlapping ranges, invalid alignment, out-of-file ranges, unknown
formats/layouts/encodings, and encoded sizes inconsistent with the descriptor.

The root has exactly:

```json
{"identity":{"model_id":"...","weights_id":"..."},"objects":[...]}
```

A tensor entry has exactly `name`, `kind`, `shape`, `format`, `layout`, `offset`, and `bytes`. A
resource entry has exactly `name`, `kind`, `encoding`, `offset`, and `bytes`. Names are nonempty and
unique; shapes contain positive unsigned dimensions; object byte counts are positive.

## Registered vocabulary

- Numeric formats: `BF16`, `FP32`, `I32`, `Q4G64_F16S`, `Q5G64_F16S`, `Q6G64_F16S`, and
  `W8G32_F16S`.
- Tensor layouts: `contiguous-le-v1`, `row-split-k128-v1`,
  `r9700-q4g64-n16-k16-v1`, and `row-scaled-k128-v1`.
- Resource encoding: `raw-bytes-v1`.
- Provisional evaluation identities: `qwen3.8-27b/r9700-int-candidate`,
  `qwen3.8-27b/r9700-w8g32-mse-eval`,
  `qwen3.8-27b/r9700-q4g64-n16k16-eval`, `qwen3.8-27b/r9700-q4-w8-n16k16-eval`,
  `qwen3.8-27b/r9700-q4-w8-mse-n16k16-eval`,
  `qwen3.8-27b/r9700-w8-bf16-embed-eval`,
  `qwen3.8-27b/r9700-w8-bf16-attn-qk-eval`,
  `qwen3.8-27b/r9700-w8-bf16-attn-vo-eval`, and
  `qwen3.8-27b/r9700-w8-bf16-gdn-qk-eval`.

There is no compatibility format registry or runtime weight repacking. The R9700 candidate binds
its grouped integer matrices directly to the HIP kernels. Its name remains provisional until the
real BF16-source quality, capacity, and whole-inference Pareto gates select the final recipe.

## Planning and materialization

The binder consumes every target-owned object exactly once and produces host-resource and
device-tensor placements. Device placements are 256-byte aligned. Materialization allocates one
device arena, retains only requested resources on the host, coalesces aligned direct-I/O spans,
and uses pinned staging slots plus the owning HIP load stream. It verifies that planned offsets and
sizes equal the reader descriptors before copying and publishes completion only after the stream is
synchronized.

No object is implicitly converted during loading. `Q4G64_F16S` accepts only
`r9700-q4g64-n16-k16-v1`; old row-split Q4 artifacts are rejected and may be migrated only by the
one-shot offline transcoder before startup. A descriptor's format and layout are immutable
persistent semantics. Exact byte layout is defined in `storage-layouts.md`; numeric reconstruction
is defined in `tensor-formats.md`; the current target inventory and conversion gate are defined in
`r9700-integer-artifact-candidate.md`.

The streaming writer stages an incomplete artifact under an exclusive hidden name in the
destination directory. After every planned payload is complete, it closes the staging file and
atomically hard-links it to the requested path without replacement. A concurrent creator wins
rather than being overwritten, and an unsuccessful writer never publishes a partial artifact at
the requested path. Successful destination creation is the unambiguous commit point: a later
staging-unlink fault cannot convert success into failure or cause deletion of the destination.
Staging cleanup is best-effort; repeated `finish()` or `close()` calls retry it, but a persistent
filesystem cleanup fault may leave a hidden hard link that can be removed separately after the
fault clears.

The Text/MTP cache group and plane orders describe runtime state and are not `.ninfer` identity or
object metadata. G16/G32 qualification uses separately compiled runtime profiles and may reuse the
same explicit weight artifact unless the weight recipe is also being compared.

## Qualification

The artifact checks cover exact framing/directory parsing, direct and grouped-integer layout sizes,
duplicate/overlap/range rejection, complete 1,124-object target binding, exact materialized device
bytes, retained frontend resources, and accounting. The complete indexed 18-shard BF16 source is
present locally; real-source conversion and final identity selection still require their explicit
conversion and quality evidence.
