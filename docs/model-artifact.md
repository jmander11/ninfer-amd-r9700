# Selected model artifact

The supported runtime file is `qwen3.8-27b-r9700-lq4.ninfer`, with embedded identity `qwen3.8-27b/r9700-lq4`. It includes the target model, DFlash2 companion and model resources. The file is 16,752,796,160 bytes (16.75 GB decimal); this is disk size, not the total GPU memory requirement.

**A public download has not been published yet.** The engine repository does not include weights or download them automatically. The quickstart requires this converted artifact. A Transformers checkpoint, GGUF file or another `.ninfer` profile cannot be substituted by renaming it.

The selected checksum and conversion settings are in `model-artifact.json`. Check a local copy before use:

```sh
sha256sum /path/to/qwen3.8-27b-r9700-lq4.ninfer
```

Place it anywhere readable by the CLI/server. For Compose, set `NINFER_MODELS_DIR` to its parent directory and `NINFER_MODEL` to its relative filename (or relative subdirectory plus filename). The mount is read-only.

Maintainers building the artifact from source should use the exact conversion recipe in `maintainer/qwen3.8-27b-artifact.md`, with the BF16 checkpoint, calibration inputs and conversion-base/DFlash prerequisites supplied explicitly. Conversion is only needed if you do not already have the selected artifact. Source credits and licensing are in `acknowledgements.md`.
