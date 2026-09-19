# BF16 GDN control T1 direct qualification

This reviewed package compares the exact Qwen3.8-27B decode boundary at T=1: two BF16
`[48,5120]` projections followed by the GDN control formula, one combined 96-row projection plus
the same control kernel, and one 48-CTA dual-projection/control fusion. Every route rounds both
projection results to BF16 before the control formula.

Run from the repository root. Preflight is read-only and requires a fresh report/artifact state,
the recorded source identities, and the R9700 power profile `auto`:

```bash
bash profiles/bench/r9700-bf16-gdn-control-t1-20260919/commands.sh --preflight
```

Only after independent package review, launch the create-only physical measurement explicitly:

```bash
bash profiles/bench/r9700-bf16-gdn-control-t1-20260919/commands.sh --measure
```

The measure action compiles the qualifier, emits normal gfx1201 assembly, runs the static checker,
and writes `report.json`. The report admits a route only when its observed per-layer median saving
multiplied by 48 layers is at least 0.2 ms/token. Operator admission does not authorize production
routing; a material winner still requires a whole-model gate.
