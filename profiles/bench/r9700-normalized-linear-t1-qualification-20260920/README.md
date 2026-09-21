# T1 normalized-linear direct qualification

Status: executable package prepared for independent review; review must precede measurement.
Passing admits preparation of the existing C1 whole-inference A/B gate only.

Prepared sources are `tools/r9700/a8q4_normalized_linear_t1_qual.hip`,
`tools/r9700/check_a8q4_normalized_linear_t1_static.py`, and this package's
`analyze.py`. The qualifier accepts `--out-json NEW.json`. The checker accepts
`ASSEMBLY --binary QUALIFIER --embedded-dir NEW_DIRECTORY`, and the analyzer accepts
the resulting qualification JSON. The only launch entry is this package's `commands.sh`.

## Decision and measured boundary

The exact decision point is BF16 RMSNorm followed by signed-A8G64 × signed-Q4G64
gate/up Linear, T=1, N=34816, K=5120, Q4N16K16 persistent codes and FP16 scales.
There are 64 such base-Text MLP boundaries per decoded token. The fused preparation
may remove the normalized BF16 global materialization and one launch, while retaining
the explicit BF16 normalization seam and the existing native IU4 dot8 consumer.
No claimed saving is multiplied by 128.

Control invokes public `ops::rmsnorm` then public `ops::linear` with caller-owned
normalized storage and serialized A8 workspace. Candidate invokes public
`ops::normalized_linear`. Both link the same `ninfer_r9700_core` archive and use
the A8 compile profile. No private candidate launcher or replacement GEMV may be
timed. Capture the complete public call sequence in each arm's Device Graph before
recording events; time one complete replay, including normalization, status reset,
activation encoding, and matrix projection. Scrubbing and host transfers are outside
the timed interval. An isolated preparation time is diagnostic only.

## Independent numerical authority

For represented BF16 x and normalization weight w, compute the entire RMSNorm in
naive FP64: inv = 1/sqrt(sum(x[k]^2)/5120 + eps), then x[k]*inv*(1+w[k])
for unit-offset mode or x[k]*inv*w[k] otherwise. Explicitly round that result once
to BF16 before applying the registered A8G64 codec. Do not copy the production
reduction tree, reciprocal-square-root instruction, or intermediate FP32 casts.

The independent exact A8 codec uses the represented BF16 seam values, group size 64,
FP16 RNE scale=max(abs(group))/127 (minimum FP16 subnormal if positive underflow),
RNE integer codes clipped to [-127,127], and zero codes/scale for all-zero groups.
Check packed low/high nibble planes, scales, and status exactly against that codec.
Decode the signed Q4 code and stored FP16 scale by independent Q4N16K16 indexing;
evaluate and retain the complete represented dot in FP64. Apply the named
`normalized_linear_a8q4_fp64_rel_l2_1e-2_bf16_steps_2_v1` criterion: relative L2 error at most
`1e-2`, with a gross pointwise cap of two BF16 steps from the independently rounded FP64 result.
Record maximum relative L2, absolute error, and BF16 steps. Check both arms directly. Pairwise
output equality is additional evidence, never a substitute for the independent oracle.

Use a structured signed finite input and nonuniform norm weights at the real shape,
unit-offset true and false, zero input, and a finite tiny input covering FP16-scale
underflow. Include nonfinite input and norm-weight cases, requiring every output
to be NaN under the codec-failure contract, followed by restored finite data on the
same allocation and captured graph. Seed stale status before finite replay; graph
replay must clear it and reproduce eager output. Three finite/nonfinite/finite graph
replays exercise state reset, not merely three repetitions of clean data.

Canaries surround output and aligned workspace; preserve input and norm-weight
bits. Reject unsupported dtypes, wrong norm extent, output mismatch, invalid epsilon,
short packed-code/scale/workspace spans, every public plane's forbidden one-byte misalignment, and aliases among input,
norm, output, weight planes, and workspace. Positive T>1 is supported composition,
so it is not a malformed-input test. Use one T2 fallback test if that public contract
is introduced with the implementation; its two token rows must be materially distinct and receive
separate normalization, codec, and projection oracles. Do not run a token-count performance sweep.

## Timing and physical evidence

Allocate three disjoint copies of identical weight bytes. For each copy capture a
control graph and a candidate graph with stable storage. Warm both arms twice per
copy. Collect 24 adjacent paired samples: copy=i%3; control first for groups of
three samples alternating with candidate first. This yields eight paired samples,
four with either arm first, per physical allocation. Scrub 80 MiB before every arm.

Retain every pair's copy, order, control_ms, and candidate_ms. The admission criteria
are all required:

- Every copy's median candidate latency is strictly less than its median control.
- Across all paired ratios candidate_ms/control_ms, mean + 2*sample_SE is <1.
- `64*(median(control_ms)-median(candidate_ms)) >= 0.2 ms/token`.
- Every event is positive and finite; no profiler interception; physical PCI
  0000:13:00.0 identifies AMD 1002:7551; selected HIP device is that PCI function,
  gfx1201 wave32; power is auto before HIP, after correctness, and after timing.

Inspect source assembly and the executable's actual embedded gfx1201 code object.
The preparation kernel should be wave32 and spill/scratch-free; record actual
VGPR/SGPR/LDS, allowing the shared reduction storage required by RMSNorm. Do not copy
the old projected-residual zero-LDS ceiling onto a reduction kernel. The unchanged
consumer must contain native `v_dot8_i32_iu4`, execute wave32, and retain the existing
qualified no-spill/no-scratch resource profile. Inspect both symbols independently:
the preparation kernel has no reason to contain an integer dot instruction.

Bind only the necessary source files, compile profile, linked core archive, qualifier
binary, and static/embedded receipts. Record complete raw samples and strict JSON
results in a fresh explicit attempt directory. Do not use a plan hash or prior report
as evidence that the linked candidate executed. Verify the public host symbol and
the exact preparation plus consumer embedded symbols. The package must not build or
run a GPU command merely on import or during its host preflight.

## Before executable launch

The CMake qualifier links the current `build-r9700` production core with A8 and
`NINFER_R9700_NORMALIZED_LINEAR_T1_CANDIDATE=0`. It calls the candidate public Op directly;
the selector belongs only to later whole-model caller routing. Preparation fails if the
explicit qualifier target needs rebuilding. Build separately before sealing the plan.

From the repository root, create the plan exactly once:

```bash
bash profiles/bench/r9700-normalized-linear-t1-qualification-20260920/commands.sh --prepare
```

After independent implementation, qualifier, and package review report SHIP:

```bash
bash profiles/bench/r9700-normalized-linear-t1-qualification-20260920/commands.sh --preflight
bash profiles/bench/r9700-normalized-linear-t1-qualification-20260920/commands.sh --measure
```

Preflight does not initialize HIP, rebuild, change the package, or create `attempt-1`.
It compiles device assembly in a temporary directory with the exact CMake compile
profile and inspects both embedded kernels and public host symbols. Measurement repeats
these checks with retained process receipts, invokes the qualifier once, independently
recomputes its raw timing samples, and writes create-only `closure.json` and
`result.sha256`, including embedded code objects. Verify the manifest from `attempt-1`.
All source, build, script, profile, and PCI identities must still match immediately
before and after qualification. Any failure stops the package; preserve it and prepare
a new explicitly named package if a fix is needed. No automatic retry is permitted.
