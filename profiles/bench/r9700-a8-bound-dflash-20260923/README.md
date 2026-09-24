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
FP32 sums lie on the2^-48 lattice; for the real K<=25600 shapes the largest
absolute sum is below1.2e17. Neither FP32/BF16 underflow nor overflow occurs in
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

## C1 draft projection Layer0

Public Linear, BF16 inputs→A8G64, stored signedQ4/FP16 N16K16/G64, N5120,
K17408 and K25600, T5/6; same independent public oracle and v3 error profile.
Cold complete-Op baseline (80MiB scrub, three weight allocations,24 alternating
events) is0.32082/0.33070ms down and0.45442/0.46956ms feature projection.
Minimum weight traffic is47,349,760 and69,632,000bytes respectively, plus
BF16 inputs/output and fresh A8 codec/workspace traffic; useful MACs=N*K*T.
Current unchanged C1 trace has five generic draft down calls at0.26015ms each
and one feature projection at0.39272ms per46.61ms kernel-service round. The same
down shape already runs at0.09328ms in target verify using successor loading.
Mechanism: transfer the qualified latency-hiding successor pipeline to these
public shape-owned Linear routes, rather than marking draft calls as target verify.
Five down savings plus a conservative25% feature saving predict about0.93ms/round
(2% of kernel service); cold complete-Op timings support testing the bundle.
Not a throughput guarantee: require full affected oracle/ISA checks, cold A/B,
then matched unprofiled C1 inference. No new arithmetic, allocation or weight format.

Candidate qualification: all four complete public-Op cells pass full FP64/A8
bounds, exact generic/codec/eager/graph, poison and guards. Prior32 kernel
instruction/resource streams are exact; four new kernels use62/63VGPR,22SGPR,
wave32 nativeIU4 and no LDS/private scratch. Cold complete-Op control→candidate:
downT5 0.32068→0.17750ms, T6 0.33074→0.17848ms;
featureT5 0.45510→0.24746ms, T6 0.47046→0.24768ms (44.6–47.4% lower latency).
Matched C1 inference: K5 95.9727→97.9506tok/s (+2.06%), K4 83.1713→84.5524
(+1.66%), warm1/reps3, codeP4096/G128, chunk2048, auto. Every repetition exact
same-C ordinary tokens; unchanged25/30 speculative request-lane rounds forK5/K4.
Both candidate repetition ranges exceed their control ranges. Promote the
shape-owned pipeline bundle; remove the superseded caller-specific down route.

## Adaptive C4 diagnosis

Fresh matched warm1/reps3: fixedK4 161.7119 aggregate tok/s versus adaptive
156.5149 (3.32% fixed advantage), all repetitions exact ordinary tokens.
Adaptive request-lane K3/K4/K5 histograms are3/123/4,2/129/1,2/129/1;
fixedK4 is132 K4 request-lane rounds each time. Histograms are not graph counts.

Matched selected-region traces: both first execute31 C4/K4 graphs. Kernel service
per steady round is83.3766ms fixed and83.7190ms adaptive (0.41% difference).
Fixed then runs2 C3/K4 and4 C1/K4 rounds (311.588ms summed kernel service);
adaptive runs C3/K5→C2/K3→C1/K5→C1/K3 (382.549ms). Tail difference70.961ms
dominates the10.615ms steady difference. Profiling is attribution, not selection.
The C1/C2 K3/W4 attention fallback costs45.445/93.125ms; five redundant compact
copies are only about8us/round and do not explain the gap. Larger append storage
is necessary after K5→K4 because up to six context rows may remain pending.

This is a supported tail-policy limitation, not a numerical bug or proof of a
faulty steady-state chooser. The current policy intentionally clamps physicalK
to remaining logical draft budget; existing tests require it. Per-round extents
are not in this trace, so budget narrowing versus cold smaller-batch exploration
cannot be fully separated. Do not banK3 from this sample or rewrite the policy
to fit it. A future policy experiment may separate physical captured width from
logical output budget, retain budget-clipped expected yield and independently
legal context/capture capacity, then qualify tail transitions. That is not part
of this bounded implementation. Use fixedK4 for this measured concurrent workload.

The first C1 baseline benchmark completed successfully, but postprocessing used
a nonexistent `final-c1-k0` authority path. Corrected to the explicit preceding
`r9700-compact-followup-20260923/baseline-c1-k0` report; recovered the retained
measurement without rerunning it. Its scope counters were not retained; do not
infer zero throttling from missing counters. All three repetitions match tokens.

## Final verification and reuse

Removed the superseded caller-specific down API, kernel and qualifiers, including
unused post-mixer leaf arguments. Ordinary Linear now owns all36 admitted shapes.
Old target-down T5/T6 instructions/resources match the new public kernels exactly
(439/480 instructions; objdump padding ellipsis is not an instruction). Final
linked projection code object is byte-identical to the qualified/timed candidate.
Final `--draft-only` requalification passes all four cells through the delivered
public boundary. CLI/server/PPL/bench rebuild and execution-state/benchmark host
tests pass; the capacity test uses prefill1 and every C1–4 captured/append width
so large prefill cannot hide underallocation. Two Python matrix tests pass.

For another explicitly chosen measurement, use a fresh label/output and the
current build; do not overwrite this retained package or relaunch an unchanged
experiment merely to reproduce it. Heavy jobs remain serial. Example:

```sh
cmake --build build-r9700 --parallel 4 --target ninfer_bench ninfer_r9700_a8q4_small_batch_projection_qual
build-r9700/src/ninfer_r9700_a8q4_small_batch_projection_qual --out-json FRESH.json --draft-only
systemd-run --user --scope -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=0 \
  env PYTHONPATH=. /home/battlefront/.local/bin/python3.11 \
  profiles/bench/r9700-a8-bound-dflash-20260923/run_decode.py \
  --binary build-r9700/bench/ninfer_bench --label FRESH --concurrency 1 --draft 5
```

The numerical source checkpoint is667a3e32; later promotion consolidates owners,
not the public oracle or error formula. Baseline/candidate/final binaries and raw
trace databases remain local; compact reports/receipts and decision evidence are
tracked. No weight file or model precision was changed.

Final confirmation in `final-summary.json`: C1 K5/K4 98.0343/84.7129tok/s
versus95.9727/83.1713 (+2.15%/+1.85%); C4 K5 149.0754 versus144.8199
(+2.94%). C4 fixedK4 161.7356 and adaptive156.2805 are essentially unchanged
from161.7119/156.5149; adaptive repetition ranges overlap. All15 final repetitions
exact same-C ordinary tokens, no resource-limit hits or recorded CPU throttling.
Final policy recommendation on this workload remains C1 K5 and concurrent fixedK4.
This closes the bounded pass, not the hardware optimization ceiling. The potential
adaptive physical-width/logical-budget policy experiment is documented for future
work but is not an unfinished implementation of this request.
