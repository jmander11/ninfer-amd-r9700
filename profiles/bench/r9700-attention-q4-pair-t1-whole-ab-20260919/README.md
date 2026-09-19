# R9700 exact-T1 all-Q4 attention projection whole A/B

`--prepare` is fail-closed on the direct production-symbol qualification, builds selector-off and
selector-on binaries, and binds their exact identities and live selector symbol receipts. It will
not succeed before that direct GPU receipt exists.

```
bash profiles/bench/r9700-attention-q4-pair-t1-whole-ab-20260919/commands.sh --prepare
bash profiles/bench/r9700-attention-q4-pair-t1-whole-ab-20260919/commands.sh --preflight
bash profiles/bench/r9700-attention-q4-pair-t1-whole-ab-20260919/commands.sh --measure
```

The measurement is six order-balanced C1 P8192+G256 processes with exact retained public-token
parity. Results are create-only. Passing requires every paired candidate/control time below one,
the paired mean plus two standard errors below one, and median ratio at most 0.99.
