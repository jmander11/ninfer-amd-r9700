# R9700 execution ledger

Status: live work ledger. Every unchecked item is assigned work unless the user narrows or cancels
it. Completed work and its evidence belong in `docs/performance.md`; product state is in
`AGENTS.md` and `README.md`. The previous campaign ledger (Q4/W8 recipe selection, XAttention and
DFlash-recipe chains, terminal selection and cutover) was retired on 2026-09-27 when the GPTQ
FP8LUT4 artifact was admitted; it remains in git history.

## Standing constraints

- GPU work holds `flock --exclusive /ssdpool2nvme/local_llm/.ninfer-coordination/gpu.lock`
  through all child work (shared with the `ninfer-amd-r9700-2` agent); builds use 12 jobs
  (maximum 14), one build per agent.
- Precision: QK stays BF16 and PV FP16 (no FP8 QK/PV, no Sage-style attention); changes that cost
  PPL are measured last, left uncommitted and reported for the user's decision.
- Clocks, power and voltage are unchanged until the user starts `CLOCKS-README`.
- Speed claims are interleaved or same-session A/Bs at the level claimed; numerical changes pass
  the owning FP64 oracle, and weight or attention-arithmetic changes also pass NIAH.

## Live items

- [ ] `LONGCTX-PREFILL` Long-context (32K-240K) prefill. Start from a fresh attribution of the
  production artifact at long context, then pursue every mechanism with a credible whole-prefill
  bound: dense prefill attention (bounded by one LDS fragment per WMMA at ~116 of ~140 TFLOP/s),
  the FP8LUT4 prefill GEMM decode/staging (~20%), chunk size at long context, and the chunked GDN
  serial chain. Exclusions already measured are listed in `docs/performance.md`.
- [ ] `DECODE-BYTES` Decode is bandwidth-bound (T1 projections at 500-555 GB/s of ~612 GB/s); the
  remaining lever is fewer bytes: the draft/output head and the DFlash drafter matrices, still
  Q4G64, as FP8LUT4 (converter plus DFlash acceptance and exact-token checks).
- [ ] `QUALITY-GAINS` Cheap quality improvements measured against the BF16 8K/32K admission
  cells: attention-role GPTQ with per-role damping (full attention GPTQ failed 8K multikey NIAH),
  GPTQ activation ordering, a larger usage calibration set.
- [ ] `CLOCKS-README` [user-gated; after `LONGCTX-PREFILL`] Matched README benchmarks at stock
  clocks and at the user's undervolted/higher-clock setting.
