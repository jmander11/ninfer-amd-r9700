# Credits and licenses

NInfer AMD is an AMD HIP port of [Neroued/NInfer](https://github.com/Neroued/ninfer). Credit for the original engine and shared inference features belongs to its upstream contributors. This repository adds the R9700 kernels, LQ4 execution, cache formats and hardware-specific tuning. Port history and the upstream baseline are recorded in `maintainer/upstream-sync.md`.

The engine source is distributed under Apache-2.0; see the root `LICENSE` and existing source notices. Model weights are separate artifacts.

## Model sources

- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B): target model, tokenizer and media resources from the Qwen team. The supplied source model card declares Apache-2.0 and includes a `LICENSE`.
- [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2): draft model, also declared Apache-2.0 in the supplied model card. The local checkpoint identifies itself as a mirror of this source.
- [DFlash](https://github.com/z-lab/dflash): the upstream speculative-decoding work associated with the draft model.

LQ4 is this port's quantized artifact format. It is not an official Qwen release. A distributed artifact should carry its source model/draft notices, conversion receipt, checksum and recipe description alongside the weights. Engine licensing does not replace those notices. See `model-artifact.md` for the selected file.

## Included and build-time dependencies

Vendored code retains its notices in the following files:

| Dependency | License file |
|---|---|
| cpp-httplib | `../third_party/cpp-httplib/LICENSE` |
| nlohmann/json | `../third_party/nlohmann/LICENSE.MIT` |
| utf8proc and Unicode data | `../third_party/utf8proc/LICENSE.md` |

The CPU [XGrammar](https://github.com/mlc-ai/xgrammar) sources and DLPack headers are vendored under `../third_party/xgrammar` (both Apache-2.0). Selected source commits and native adaptations are recorded in `../third_party/xgrammar/README.ninfer.md`; preserve their included license and notice files when redistributing sources. ROCm, FFmpeg, libcurl and zstd are external build/runtime dependencies with their own notices.

## Workflow references

The host workflow follows GitHub's [Python build and test documentation](https://docs.github.com/en/actions/tutorials/build-and-test-code/python). The localhost Compose override uses [Compose's override tag](https://docs.docker.com/reference/compose-file/merge/#replace-value).

Git source archive behavior is documented in the [git-archive manual](https://git-scm.com/docs/git-archive#ATTRIBUTES).
