# Three-route production confirmation

Fresh selector-free confirmation of BF16 GDN control at T1, all-Q4 attention
projection at T1, and all-Q4 paired GDN/attention projection at T2–4. All other
candidate routes remain disabled. This is a production composition check, not
another A/B or a physical bandwidth claim.

From the repository root, run `bash profiles/bench/r9700-three-route-production-confirmation-20260920/commands.sh --prepare`,
review the resulting plan, then use `--preflight` and `--measure`. Preparation
creates a fresh Release/gfx1201 build and plan; neither preparation nor
measurement overwrites existing output. Preflight writes nothing and launches
no GPU work. Keep competing GPU applications stopped for the complete run.

Measurement serializes one ordinary Device-Graph P8192+G256 process at each of
C1, C2, C3, C4, each with one warmup and one measured repetition, chunk 4096,
the fixed FP8-K/INT4-V cache, and exact retained tokens for every lane. The CLI
uses `--spec mtp --draft-tokens 0`, which must report `spec: none`.

C1 decode time must be no more than 1.005 times the faster median of the two
admitted singleton candidate campaigns. C2–4 must be no more than 1.02 times
their respective admitted candidate medians. These are regression ceilings,
not a prediction that the separately measured gains add. All baselines and
thresholds are written into plan.json before measurement.

The runner requires auto power and low PCI-bound VRAM before every process,
allows at most 30 seconds for post-process teardown, and records every drain
sample. Identity or telemetry failure stops the campaign. Results include
process receipts, stdout/stderr, benchmark reports, summary or failure receipt,
and a complete SHA256 closure. Low pre/post VRAM cannot prove that no external
application used the GPU during inference.
