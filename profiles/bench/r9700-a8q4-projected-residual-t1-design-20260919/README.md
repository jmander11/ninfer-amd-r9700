# R9700 all-Q4 T1 projected-residual direct qualification design

This create-only package qualifies one bounded ordinary-decode candidate at the complete
projection boundary. It compares the production native-IU4 Q4 Linear followed by the production
BF16 residual-add Op against a single native-IU4 kernel that preserves the same two represented
boundaries in registers:

```
delta = BF16(dot)
residual = BF16(FP32(residual) + FP32(delta))
```

The exact shapes are `N=5120,K=6144,T=1` (16 full-attention plus 48 GDN output projections) and
`N=5120,K=17408,T=1` (64 MLP-down projections). Both therefore carry 64 calls per ordinary C1
round. Mixed-weight and `T>1` routes are outside this package.

The gate requires complete BF16 residual bit parity, an independent FP64 oracle that decodes the
actual signed Q4N16K16 codes and FP16 G64 row scales, nonfinite poisoning, canaries, malformed and
alias rejection, and exact gfx1201 native-IU4 ISA. Timing is jointly arm-order and allocation
balanced over three disjoint weight copies: each arm occupies each order position twice on every
copy. An 80 MiB device cache scrub runs before every timed arm but outside
its HIP-event interval. Every allocation must remain within 1% of the incumbent, and the two shape
savings weighted by 64 calls each must total at least `0.2 ms/token`.

Run the non-GPU disposable compile and static check from the repository root:

```
bash profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919/commands.sh --preflight
```

Only after independent review and ledger authorization, run the sole create-only measurement:

```
bash profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919/commands.sh --measure
```

Passing authorizes only a selector-off/on whole C1 A/B. It does not authorize product routing.
The entire measurement is create-only under `attempt-1/`; a failed attempt is retained and cannot
be overwritten by a rerun. Its output is `attempt-1/qualification.json`; no GPU measurement has
been run.
