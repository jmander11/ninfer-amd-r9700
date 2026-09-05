# Qwen3.8-27B R9700-candidate implementation-profile diagnostic

This target-private PyTorch path exercises Text, Vision, MTP, sampling, state, and weight residency
over a native `.ninfer` artifact. It is independent from the C++ Engine implementation, but it is
an implementation-profile diagnostic rather than an independent numerical oracle: it decodes the
same stored W8 values, uses PyTorch BF16/FP32 operations, SDPA, FLA, and PyTorch's FP8 casts, and
implements the production-shaped INT4 cache codec with tensor operations.

It does not need the original Hugging Face checkpoint at inference time. `Frontend` materializes the
tokenizer, chat template, generation defaults, and image/video processor resources embedded in the
artifact, then delegates those functions to Transformers.

## Run

Install the target dependencies from `requirements.txt`, then run:

```bash
python3 \
  -m tools.reference.qwen3_8_27b \
  --weights out/qwen3_8_27b_r9700_candidate.ninfer \
  --prompt "请简短介绍一下你自己。" --decode 512
```

The input is exactly one of `--prompt`, `--ids`, or `--messages`. Structured messages may contain
images and videos in the normal Transformers format. Thinking is enabled by default and can be
disabled with `--no-thinking`.

MTP is disabled by default. Enable one to five draft positions with
`--mtp-draft-tokens 1..5`; `--draft-head` selects the artifact's optimized proposal head. Target
verification always uses the full output head. The CLI reports round counts, per-position accepted
drafts, fallback steps, timing, memory planning, and peak device allocation. On ROCm, PyTorch
exposes the AMD device through its `cuda` API and the default device spelling remains `cuda`.

Important runtime controls include:

- `--gpu-memory auto|24GiB` and `--headroom 2GiB`;
- `--prefill-chunk N`;
- `--greedy` or sampling overrides for temperature, top-p, top-k, and penalties;
- `--vision-attention-limit N`;
- `--activation-dump DIR --dump-level layer|op`.

The cache is fixed to FP8 E4M3FN keys plus signed INT4-G16 values and FP16 value scales; there is no
cache-format selector or key scale. The provisional artifact binds every persistent quantized
matrix as W8G32. Its Text matrices retain the decoded/packed/streamed residency plan, while Vision
decodes large matrices one at a time and releases its weight store before Text weight preparation.
Multimodal MTP uses the composed Vision embedding for the shifted input, including at prefill chunk
boundaries.

Use this path to diagnose artifact binding, the represented W8 implementation profile, schedule,
state, and library-kernel behavior. Agreement with it does not independently prove the W8 decode,
FP8-key codec, INT4-value codec, model logits, PPL, or exact tokens. The dependency-light converter
codec fixtures and the R9700 Op qualifiers own independent scalar represented-value and kernel
checks. `tools/reference/qwen3_8_27b_bf16/` is the independent checkpoint-direct authority for the
real-model PPL and exact-token gate; it requires the complete original BF16 checkpoint.
