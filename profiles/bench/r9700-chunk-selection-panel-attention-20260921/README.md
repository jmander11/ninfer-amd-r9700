# Shared prefill chunk selection

Prepared for qualified bounded-panel attention builds (not runnable until that route passes
its independent oracle/static and whole8K admission). The paused predecessor campaign remains
unchanged evidence. Current production builds on the R9700, C1, ordinary prefill. Compare the three existing N16
weight recipes across G16/G32 and dense/B128-S16-tau900 attention. Measure chunks 1024, 2048,
4096 and 8192 at 8K; the existing selector chooses two global finalists for all twelve 32K cells.
Each point uses three repetitions and one warmup. Profiled timings are excluded.

Run from the repository root, with exclusive use of the R9700:

```bash
bash profiles/bench/r9700-chunk-selection-panel-attention-20260921/commands.sh preflight
bash profiles/bench/r9700-chunk-selection-panel-attention-20260921/commands.sh freeze
bash profiles/bench/r9700-chunk-selection-panel-attention-20260921/commands.sh prepare
bash profiles/bench/r9700-chunk-selection-panel-attention-20260921/commands.sh screens
bash profiles/bench/r9700-chunk-selection-panel-attention-20260921/commands.sh finalists
bash profiles/bench/r9700-chunk-selection-panel-attention-20260921/commands.sh select
```

`preflight` is read-only. `freeze` publishes input identities once; it must not be repeated.
The runner validates existing reports before skipping them on `screens`/`finalists` resume.
Failed outputs remain evidence; inspect the failure before resuming. No input may change after
freeze. A changed build or artifact needs a separate campaign.

Selection is published to
`profiles/bench/prefill-chunk-selection-panel-attention-20260921.json` only after the
existing selector validates all required measured cells. This selects execution geometry;
weight quality and final artifact selection remain subsequent decisions.
