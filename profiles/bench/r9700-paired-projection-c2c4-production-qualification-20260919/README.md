# R9700 C2-C4 paired-projection production qualification

This create-only package qualifies the exact production raw symbol shared by the semantic
full-attention and GDN input-projection Ops. It covers T=2,3,4 with canonical Q4G64_F16S
Q4N16K16 weights only and compares both boundaries against two complete production Linear calls
(plus the four attention extracts), a shared-quantization diagnostic, and an independent
represented-input/weight FP64 oracle.

Run the non-GPU build and exact-symbol static gate from the repository root:

```
bash profiles/bench/r9700-paired-projection-c2c4-production-qualification-20260919/commands.sh --preflight
```

After independent review, the sole create-only GPU qualification is:

```
bash profiles/bench/r9700-paired-projection-c2c4-production-qualification-20260919/commands.sh --measure
```

Passing authorizes only preparation of the selector-off/on whole C2/C3/C4 A/B. It does not
authorize default routing. The fixed build directory and report are both create-only; any failed
attempt remains retained and blocks an in-place rerun.
