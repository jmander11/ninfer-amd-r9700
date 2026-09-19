# R9700 all-Q4 C2-C4 paired-WMMA direct qualification design

This source/static-only package prepares a bounded qualifier for ordinary compact decode widths
T=2,3,4. It covers the 48 GDN projection pairs (`N=4096+12288,K=5120`) and the 16 full-attention
projection pairs (`N=7168+7168,K=5120`) per round.

The three arms are the complete incumbent (two quantize-plus-WMMA Linear calls and, for attention,
four extracts), a diagnostic with one shared A8G64 quantization followed by two unchanged WMMA
calls, and the challenger with one shared quantization plus one combined-row native-IU4 WMMA grid.
The attention challenger writes Q6144, K1024, Gate6144, and V1024 directly.

The qualifier requires complete three-arm BF16 bit parity, an independent represented-input and
represented-weight FP64 oracle, nonfinite-status poisoning, output canaries, malformed extent,
alignment and alias rejection, exact gfx1201 native IU4 WMMA, wave32, and zero LDS, scratch and
spills. Timing uses twelve order-balanced trials across three disjoint weight-pair allocations;
the report weights direct savings by 48 GDN and 16 attention calls.

The completed selector-free C2-C4 production confirmation is bound by exact identity in
`whole-baselines.json`. Its two runs per width derive exact 256-round medians of 62.185710 ms,
69.072337 ms, and 79.071854 ms for C2, C3, and C4. The fail-closed preflight re-derives those
values from the bound authority, verifies every source identity, performs a disposable host/device
build, and checks the exact static receipt without opening the GPU:

```
bash profiles/bench/r9700-a8q4-pair-wmma-c2c4-design-20260919/commands.sh --preflight
```

After independent wrapper review and ledger authorization, the sole create-only measurement is:

```
bash profiles/bench/r9700-a8q4-pair-wmma-c2c4-design-20260919/commands.sh --measure
```

The qualifier selects HIP device 0, requires its exact R9700/gfx1201/wave32 identity, derives the
power node from that device's PCI identity, and requires `auto` both before and after measurement.
The retained report records the exact qualifier argv, device name/architecture/PCI/wave size,
HIP runtime and driver versions, both power observations, and the assembly-bound static receipt
with its exact VGPR/SGPR/LDS/scratch/spill resources and digests.
The direct gate requires every balanced cold allocation to be non-regressing and a credible
whole-round gain of at least 1% at every width. Passing authorizes only a subsequent selector-off/on
whole-inference C2-C4 A/B.

Prepared sources:

- `tools/r9700/a8q4_pair_wmma_c2c4_qual.hip`
- `tools/r9700/check_a8q4_pair_wmma_c2c4_static.py`

The fresh create-only output is `qualification.json`. The first authorized invocation stopped
before static validation, HIP device selection, or timing because the baseline stream reader
mistook a successful iterator read for failure. `attempt-1-read-failure.json` retains that result.
The repaired reader and rebound package require independent re-review before another invocation.
