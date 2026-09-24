# A8 numerical bound and decode follow-up

Scope: justify/requalify the small-batch and down A8 numerical profile, inspect
C1 DFlash for a new material mechanism, and diagnose adaptive C4 versus fixedK4.
Starting checkpoint41b4bdb5; installed cap26 Q4-head mixed gate/up-A4 model and
Q4/BF16-codebook DFlash companion unchanged. Prefill tuning remains paused.
Heavy jobs serial on R9700/gfx1201/device0/PCI0000:13:00.0, auto power, C<=4,
ROCm10; memory20Ghigh/24Gmax/no swap, build cap14 (normally4).

Prior matched C1 codeP4096/G128 ordinary30.16/K5 96.05tok/s, C4 fixedK4 161.66,
repaired adaptive154.82 before the last K5 output extension. Reuse unchanged C1
attribution; fresh current-build adaptive timing/trace will resolve actual K
residence and policy overhead before proposing any policy change.

Numerical policy is not PPL: retain original BF16-input/stored-Q4 FP64 oracle,
exact independent codec and represented-A8 arithmetic diagnostics. The old
N4096/K5120/T6 2.034% result remains evidence, not a kernel bug established by
crossing an unexplained2% cutoff. Any replacement criterion must follow a
documented error derivation, cover the complete affected domain, reject corruption,
and preserve separate model-quality gates. No production precision change planned.

## Numerical derivation: exact-int/group-FMA/BF16 profile

For each original BF16 input x and stored signedQ4/FP16 weight W, the independent
public oracle remains y=FP64(W*x), with no private casts. The independent host
A8 codec supplies xhat; the supplementary represented-input FP64 calculation
gives q=FP64(W*xhat). Its G=K/64 exact integer group terms are t_g; S=sum|t_g|.
Let gamma(n,u)=n*u/(1-n*u), u32=2^-24, u64=2^-53, uBF16=2^-8.
Inflate computed S by1/(1-gamma(G,u64)); let R=gamma(G,u64)*S and
A=gamma(G,u32)*S. The arithmetic budget about q is
E=A+R+uBF16*(|q|+R+A). The complete public-output budget is |q-y|+E.
Require both bounds per output and in per-token L2 norm. Thus quantization
allowance cannot conceal an arithmetic defect. Bounds receive outward nextafter
rounding; zero terms require exact zero. Nonfinite outputs/references fail.

Justification specific to these admitted routes: signedINT4 and signedA8 G64
integer dot is at most65024 and exactly representable in INT32/FP32. Two finite
FP16 scales multiply exactly in FP32 (at most22 significant bits); one FP32 FMA
per group supplies the gamma_G bound. All nonzero scaled terms and intermediate
FP32 sums lie on the2^-48 lattice; for the real K<=17408 shapes the largest
absolute sum is below8e16. Neither FP32/BF16 underflow nor overflow occurs in
this finite profile, so normal RNE bounds apply, including ties. FP64 terms
have at most38 significant bits and are exact before their accumulation.

This is a correctness envelope for the specified A8 implementation profile,
not a new bound on model-quality loss. Original public relativeRMS/gross errors
and the old2%/10% pass/fail remain diagnostic fields. No production arithmetic,
precision, weights, PPL policy or oracle formula changes. Qualification checks
the whole affected small-batch/down shape domain and deliberately corrupts real
outputs (sign, zero,12.5% gain) to ensure the envelope rejects implementation errors.

Full requalification: `small-batch-qualification.json` passes all32 cells and
`down-qualification.json` passes both cells; five finite/graph fixtures per cell,
19 malformed cases per cell, exact codec/control/eager/graph, poison and guards.
All102 output corruptions rejected. Max public/arithmetic norm-budget fractions
0.88149453/0.45583670. Original N4096/T6 public RMS2.0340653% remains explicitly
false under `historical_2pct_rms_10pct_gross_pass`; new criterion does not claim
that accuracy improved. No production numerical change and no PPL rerun needed.
