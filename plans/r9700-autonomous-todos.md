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
- [x] `ATTN-MIDROWS` Closed 2026-09-28: 9..127-row causal chunks run the dense prefill tile split
  over context chunks with the packed route's FP32 merge (19 rows at 22K: 34 -> 0.24 ms/layer;
  19-token follow-up TTFT at 22K 818 -> 240 ms); `docs/performance.md` has the evidence.
- [x] `DFLASH-BATCHED-DRAFT` Closed 2026-09-28: every chain K drafts batched at C2..C4, with
  T14/T16/T21 small-batch cells; corpus aggregate +7.6% / +5.1% / +4.0% at C2 / C3 / C4. The
  fused GDN pair stays batch-1 (C>1 split path is weight-bound at the same cost).
- [x] `DFLASH-K3-ROUTES` Closed 2026-09-28: T4 drafter cells and the W4 FP8LUT4 GDN pair;
  C1 K3 33.1 -> 30.6 ms/round.
- [x] `DECODE-BANDWIDTH` Closed 2026-09-29 (`docs/performance.md`, Decode step attribution):
  weight streaming runs at ~95% of the read peak; the four small-kernel candidates (quantization
  fusion, GDN record and front, narrow drafter projections) and an overlapped replay fold gave
  no admissible gain. Follow-ups done: the tool-mask exchange is a host mailbox (+0.7% with
  tools) and the Q4G64 token embedding is stored row-split (gathers ~50 -> ~7 us).
- [x] `NVFP4-MATCH` Closed 2026-09-30: the weight bar is "no worse than the 5090 NVFP4 build",
  measured with the frozen 5090 scorer on the BF16-source cells (NVFP4 decode is only ~0.004-0.005
  nats/token behind production, prefill ~0.01-0.02). Protections 26 -> 21 (16.75 GB; C1 decode
  +2.3%, C4 unchanged); 3-bit MLP gate/up is not admissible (all layers, the 12 lowest-error
  late layers and 12 mid layers each fall behind NVFP4 on decode or break a severe cap).
  `docs/performance.md` has the evidence.
- [ ] `CLOCKS-README` [user-gated; `LONGCTX-PREFILL` is closed] Matched README benchmarks at stock
  clocks and at the user's undervolted/higher-clock setting.
- [x] `MIXED-PREFILL-DECODE` [changed the product contract] Done 2026-09-29.
  `--mixed-forward N --mixed-forward-rounds D` lets a DFlash text owner's slice fill an N-column
  forward (a multiple of 256) beside every D-th decode round's verify columns (a mixed round);
  other owners time-share with N-token steps. The default (`auto`, 2026-09-30) is N = 1024 under
  DFlash with C > 1 and prefill-first otherwise. The decode-share vs prefill-throughput Pareto frontier, the serve
  stall check, and the kernel fixes that made mixed rounds cheap are in `docs/performance.md`
  "Mixed prefill/decode frontier". Architecture: `concurrent-inference-architecture.md` §1.2,
  §2.5, §7.3, §8.9.
