# R9700 T1 all-Q4 attention projection production qualification

This package directly qualifies the production raw symbol used by the semantic full-attention
projection Op. It compares two complete N7168/K5120 production Q4 linears plus four extracts with
one shared A8 quantization and one combined direct-scatter grid producing Q6144, K1024, Gate6144,
and V1024.

The non-GPU source/build/static check is:

```
bash profiles/bench/r9700-attention-projection-t1-production-qualification-20260919/commands.sh --preflight
```

After independent review, the sole create-only GPU qualification is:

```
bash profiles/bench/r9700-attention-projection-t1-production-qualification-20260919/commands.sh --measure
```

Passing authorizes only the prepared selector-off/on whole C1 A/B. It does not authorize default
production routing.
