# R9700 all-Q4 T1 attention paired-projection direct gate

This create-only package qualifies one bounded challenger before any product integration:
two complete `N=7168,K=5120` Q4 linears plus the four incumbent extracts versus one shared
A8G64 quantization and one combined 14336-row native-IU4 kernel that writes Q6144, K1024,
Gate6144, and V1024 directly.

The gate requires complete BF16 bit parity, an independent represented-input/weight FP64 oracle,
canaries, nonfinite-status poisoning, malformed/extent/alignment/alias rejection, exact gfx1201
native-IU4 ISA with wave32 and no LDS/scratch/spills, and an allocation/order-balanced cold timing
screen over three disjoint weight-pair allocations. Each allocation must win by at least 1%.

Whole-decode admission uses the retained selector-free base-decode median 27.0281083 tok/s
(36.99851 ms/token) and the 16 all-Q4 full-attention layers. The direct saving times 16 must imply
at least 1% of that whole-token time. Passing admits only a selector-off/on whole-inference A/B;
it does not promote the route.

The bound plan is `plan.json`; it contains the exact source identities, compile contract,
admission rule, invocations, and output path. There are no deferred input identities.

Run the non-GPU, non-mutating check from the repository root with:

```
bash profiles/bench/r9700-attention-q4-pair-t1-direct-20260919/commands.sh --preflight
```

Independent review must return SHIP before the ledger authorizes the sole measurement command:

```
bash profiles/bench/r9700-attention-q4-pair-t1-direct-20260919/commands.sh --measure
```

Measurement requires a fresh `qualification.json` path and `auto` power, recompiles the bound
sources, retains the static receipt under `build/static-receipt.txt`, and writes the create-only
`qualification.json` result. Passing admits a later whole-inference A/B only.
