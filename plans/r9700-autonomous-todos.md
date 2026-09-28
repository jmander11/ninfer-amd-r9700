# R9700 execution ledger

Status: live work ledger. Every unchecked item is assigned work unless the user narrows or cancels
it. Completed work and its evidence belong in `docs/performance.md`; product state is in
`AGENTS.md` and `README.md`. The previous campaign ledger (Q4/W8 recipe selection, XAttention and
DFlash-recipe chains, terminal selection and cutover) was retired on 2026-09-27 when the GPTQ
FP8LUT4 artifact was admitted; it remains in git history.

## Standing constraints

- GPU work holds `flock --exclusive /ssdpool2nvme/local_llm/.ninfer-coordination/gpu.lock`
  through all child work (shared with the `ninfer-amd-r9700-2` agent); builds use at most 8
  jobs, one build per agent.
- Precision: QK stays BF16 and PV FP16 (no FP8 QK/PV, no Sage-style attention); changes that cost
  PPL are measured last, left uncommitted and reported for the user's decision.
- Clocks, power and voltage are unchanged until the user starts `CLOCKS-README`.
- Speed claims are interleaved or same-session A/Bs at the level claimed; numerical changes pass
  the owning FP64 oracle, and weight or attention-arithmetic changes also pass NIAH.

## Live items

- [x] `LONGCTX-PREFILL` Long-context (32K-240K) prefill, closed 2026-09-27: 128K attribution
  (attention 47%, FP8LUT4 GEMMs 44%), attention ablations, paired-wave split rejected at Layer 0,
  staging latency unhideable on gfx1201 at the 240-VGPR budget, chunk 4096 mixed (default 2048).
  No mechanism with a credible whole-prefill bound remains; `docs/performance.md` has the evidence.
- [ ] `QUALITY-GAINS` Cheap quality improvements measured against the BF16 8K/32K admission
  cells: the Q4G64 target output head as (GPTQ) FP8LUT4 at the same 4.25 bits/weight,
  attention-role GPTQ with per-role damping (full attention GPTQ failed 8K multikey NIAH), GPTQ
  activation ordering, a larger usage calibration set. Decode is not a lever here: Q4G64 and
  FP8LUT4 are the same size and the Q4 drafter/head kernels already read at ~600 GB/s.
- [ ] `CLOCKS-README` [user-gated; `LONGCTX-PREFILL` is closed] Matched README benchmarks at stock
  clocks and at the user's undervolted/higher-clock setting.
