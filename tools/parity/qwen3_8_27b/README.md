# Qwen3.8-27B R9700 Vision parity tool

This tool compares the independent `.ninfer` Python reference with the source BF16 Vision tower at
matching semantic boundaries.

After schema-v7 selection, prepare the selected artifact's one-image diagnostic with:

```bash
bash profiles/bench/post-terminal-selected-vision-prepare-20260905/prepare.sh
bash profiles/bench/post-terminal-selected-vision-20260905/commands.sh
```

Preparation resolves the exact winner, freezes the committed image through its selected artifact
frontend on CPU, and binds the selected build, cache group, execution profile, chunk, artifact,
two explicit Python environments, and the validated 18-shard BF16 checkpoint receipt. The GPU
command compares the selected artifact and source BF16 tower at block 0/13/26 and merger. The
`ninfer_vision_bf16_comparison_v3` report retains exact finite shapes and numerical differences;
its completion authority is deliberately diagnostic-only and has no invented error threshold.
