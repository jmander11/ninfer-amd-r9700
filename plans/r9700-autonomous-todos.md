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
  cells. Done 2026-09-28: the GPTQ FP8LUT4 output head (-0.006 to -0.008 nats/token, promoted
  into `r9700-fp8lut4`); GPTQ on attention gate/value and output (-0.0013 to -0.0028, NIAH
  passes; query/key stays independent). Rejected: group-wise GPTQ activation ordering (+0.0003 to
  +0.0043). Remaining: a larger usage calibration set. Decode
  is not a lever here: Q4G64 and FP8LUT4 are the same size and the Q4 drafter kernels already
  read at ~600 GB/s.
- [ ] `ATTN-MIDROWS` Prefill chunks of 9..127 rows (short follow-up turns and tool results,
  and prompt tails with `P mod 2048` in 9..127) fall between the packed decode route (1..8 rows)
  and dense prefill (>=128 rows) onto `fused_attention_causal_kernel`, one query row per CTA:
  34 ms/layer, 547 ms per request at 22K context. Route them through a qualified long-context
  kernel (dense prefill partial tile or 8-row packed slices, whichever is faster), FP64 oracle,
  TTFT A/B. Evidence: `profiles/bench/r9700-decode-study-20260928/`.
- [ ] `DFLASH-BATCHED-DRAFT` At C2..C4 the DFlash drafter runs batched only for `k == 4`
  (`lockstep_dflash4`); other K loop per request (C4 K5 ~6.4 vs ~2 ms/round). Batch every K,
  add the missing small-batch drafter projection cells (T14/16/21), extend the fused GDN
  pair conv/record kernel beyond batch 1 if C>1 falls back, and confirm adaptive picks the best
  K afterwards. C1..C4 decode A/B on the study corpus.
- [ ] `DFLASH-K3-ROUTES` C1 K3 (W4) misses the fused GDN pair conv/record and the T4 drafter
  projections (+2.4 ms/round vs K4); add the W4/T4 cells.
- [ ] `CLOCKS-README` [user-gated; `LONGCTX-PREFILL` is closed] Matched README benchmarks at stock
  clocks and at the user's undervolted/higher-clock setting.
