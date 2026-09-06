# R9700 codec and gfx12 qualification

This native-HIP suite qualifies the gfx1201 hardware assumptions and the R9700 Op implementations
linked by `ninfer_r9700_core`. The standalone Makefile keeps focused oracle, ISA/resource, timing,
and profiling work independent of the complete product build.

It selects exactly one discrete AMD `gfx1201` device with PCI device ID `1002:7551`. If a host has
more than one such card, set `NINFER_R9700_PCI_BUS_ID` to the full PCI address reported by HIP (for
example `0000:03:00.0`). An ordinal alone is never accepted as the device identity.

Build and run with the installed ROCm toolchain:

```bash
make -C tools/r9700
tools/r9700/build/r9700_qual
tools/r9700/build/kv_op_qual
build-r9700/src/ninfer_r9700_roctx_qual
make -C tools/r9700 isa
```

`make -C tools/r9700 hbm` builds the native `gfx1201` sustained-memory probe. Run
`tools/r9700/build/hbm_bandwidth_probe`; it selects the same exact discrete `1002:7551` R9700,
uses a working set larger than last-level cache, and reports read, write, copy, and HIP D2D rates
against the card's 640 GB/s advertised bus bandwidth. This is a hardware-bound measurement, not
an end-to-end inference result or a substitute for profiling a named production Op.

`make -C tools/r9700 q4-hardware-peak` compile-qualifies the ceilings needed to interpret the
production A8Q4G64 prefill route. The IU4 saturation control carries eight independent accumulator
chains and exactly eight static signed/signed, unclamped `v_wmma_i32_16x16x32_iu4` instructions.
Two native-IU8 register-only controls each contain exactly eight signed/signed, unclamped
`v_wmma_i32_16x16x16_iu8` instructions: the topology-matched control has two accumulator chains
with four dependent K16 instructions per chain for one useful M16xN32xK64 fragment, while the
saturation control has eight independent chains. Both intentionally exclude Q4 unpack, LDS
staging, and repeated operand loads so dependency cost can be separated from the opcode ceiling.
The matched streaming probe reads separate packed Q4 code and FP16 scale planes at the exact
represented G64 weight ratio of 32:2 bytes, with a working set at least four times L2. The static
gate proves the instruction/dependency topology, one reused register operand pair, zero LDS and
scratch, bounded VGPRs, the stream's two vector load sites, and its exact 16:1 aggregate address
pattern. Schema v2 reports runtime kernel resources and distinguishes issued operations from useful
W4A8 operations: split-IU4 useful work is half its issued operation count, topology-matched IU8
issued and useful counts are equal, and the non-production IU8 saturation control has no useful
W4A8 rate. Run
`make -C tools/r9700 q4-hardware-peak-run Q4_HARDWARE_PEAK_JSON=<fresh-path>` only under the `auto`
power profile. The executable checks `auto` before allocation and after timing, selects exactly one
discrete `1002:7551` gfx1201 card, and creates its raw JSON with exclusive-create semantics. Reconcile
its measured issued/useful IU4 and IU8 TOPS and represented-read GB/s with
`a8q4_prefill_traffic.py`; none of these isolated ceilings establishes production-kernel
utilization or stall freedom.

The current retained physical run is
`profiles/bench/r9700-q4-hardware-peak-iu8-20260905.json`, SHA-256
`1e4f5a5ac863a1939e560794758e335692998121bc5172a6f5696f1af8653756`. Under `auto`, its medians
are `803.521803` issued-IU4 TOPS, `401.760902` useful split-W4A8 TOPS, `390.247576`
topology-matched IU8 TOPS, `397.004351` saturation-control IU8 TOPS, and `633.282493 GB/s`
represented code/scale reads. Because each multiply-accumulate is counted as two operations, the
comparable useful matrix rates are `200.880451` TMAC/s for split-IU4 and `195.123788` TMAC/s for
topology-matched IU8. The IU8/IU4 ratio is `0.97134284`, a `2.86572%` IU8 deficit; topology-matched
IU8 reaches `98.2981%` of its eight-independent-chain opcode ceiling, so the four-deep dependency
topology costs only `1.70194%` in this register-only control. The independently supplied
`148/152.5 = 0.97049180` ratio differs by only `0.08510` percentage points, although its absolute
rates are about 31.8% lower and therefore are not interchangeable ceilings. The JSON's
`compute_units=32` is the raw HIP
`multiProcessorCount` scheduler value on this runtime, not a revision of the R9700's architectural
64-CU count. The retained schema-v1 predecessor is
`profiles/bench/r9700-q4-hardware-peak-20260904.json`. Against the production M64xN128 weighted
P2048 aggregate, `101.082` tera-issued-operation-equivalents over `1.109850230 s` is `91.077154`
issued TOPS, or `11.3347%` of the current isolated IU4 issue ceiling. Its `842.961535 GB`
source-request model similarly evaluates to `759.527288 GB/s`,
but that is not physical HBM traffic: the model counts represented requests whose physical
transactions depend on cache, coalescing, LDS reuse, and execution overlap. It therefore cannot be
used to claim `119.938%` physical bandwidth or derive a cache-hit fraction from the isolated stream
ceiling. The result proves that this card and toolchain can approach the nominal memory bound and
that production is far below the isolated matrix-issue bound; final production stall and memory-use
claims still require matched hardware-counter profiling.

`make -C tools/r9700 int4-wmma` builds and runs the standalone signed-INT4 matrix-instruction
feasibility qualifier. It owns literal wave32 register maps for both dense gfx12 `16x16x16` and
`16x16x32` signed IU4 forms rather than delegating layout to a library. For either form, lane
`l` loads A row `l%16` and B column `l%16`; lanes 0--15 own the low K half, lanes 16--31 own the
high K half, and accumulator element `j` owns `C[(l/16)*8+j,l%16]`. Sparse asymmetric A and B
matrices plus an asymmetric I32 accumulator seed compare all 256 outputs exactly with a separate
host signed-nibble/I32 GEMM oracle.

On the ROCm 10 physical R9700 baseline, both forms pass. The 256-tile shared-packed-input fixture
measures WMMA16/WMMA32/straight vector-dot nanoseconds per tile of 21.16/18.33/132.95 at K64,
26.47/23.84/148.87 at K256, 81.05/65.82/777.53 at K1024, and 242.77/205.20/2666.79 at K5120.
Thus WMMA32 is 7.25x, 6.25x, 11.81x, and 13.00x faster than this deliberately straightforward
packed-INT4 vector baseline. `make -C tools/r9700 int4-wmma-isa` proves native
`v_wmma_i32_16x16x16_iu4` and `v_wmma_i32_16x16x32_iu4` emission with signed A/B controls;
`int4-wmma-resources` reports 35/55 VGPR, zero LDS and scratch, and occupancy 16. This establishes
a viable low-level primitive for the registered Q4G64 evaluation identities. By itself it does
not qualify activation quantization, grouped-scale composition, a complete Linear Op, real-model
accuracy, or select a product weight recipe.

`make -C tools/r9700 q4g64-linear` builds and runs the complete Q4G64 Linear qualifier used by the
evaluation dispatch. The raw boundary consumes represented BF16 activations, writes canonical
signed A4G64 codes plus FP16 scales into caller-owned storage, and executes Q4G64 weights with
native signed `16x16x32` wave32 WMMA. Both planes retain the persistent row-split K128 alignment while their
scales remain G64; the exact codec gate covers a partial group and the group-complete K192 to K256
tail. Nonfinite input and finite-source FP16-scale overflow publish device status; the ordered
consumer converts any nonzero status into exact BF16 NaN outputs without host readback or another
kernel launch, so invalid activation codes cannot be silently consumed as zeros. Malformed
extents and aliases reject, and the route performs no allocation or runtime weight repack.

On the ROCm 10 physical R9700, exact activation-image checks and every output of the independent
FP64 A4-times-Q4 formula pass for `[T,5120] x [7168,5120]` at T=1..8,16,32,64,128. The separately
computed represented-BF16-times-decoded-Q4 reference reports a maximum activation-quantization
error of 0.212891, below its per-output decoded-weight error bound (maximum 0.732621). Interleaved
complete-candidate medians are 0.05356--0.07826 ms for decode T=1..8 and
0.09961/0.11374/0.13436/0.24254 ms for prefill T=16/32/64/128. The current W8 path at those same
synthetic shapes measures 0.12623--0.57885 ms and 1.14337/2.21130/4.42516/8.73436 ms; the complete
Q4 candidate is 2.36x--36.01x faster in this isolated shape test. The straightforward Q4 vector
route is 5.90x--151.73x slower than the complete WMMA candidate. `q4g64-linear-isa` proves the
native signed IU4 opcode; `q4g64-linear-resources` reports quantize/vector/WMMA at 11/19/55 VGPR,
260/1024/0 bytes LDS, zero scratch, and occupancy 16. These measurements admit Q4G64 to the
registered evaluation dispatch and the real BF16-source PPL/token and whole-inference recipe gate;
they do not select it as the production recipe or change W8 dispatch.

The same `q4g64-linear` run also qualifies the production-style default activation execution
profile over the identical persistent Q4G64 bytes; A4 remains an explicit evaluator. Canonical
signed A8G64 codes use one unsigned low-nibble
plane and one signed high-nibble plane, with the exact identity `a8 = low + 16 * high`. Each G64
dot uses native unsigned-A/signed-W and signed-A/signed-W IU4 WMMA pairs, recombines in I32, then
composes the FP16 activation and weight scales in FP32 before one BF16 output rounding. The route
has caller-owned K128-aligned workspace and the same finite/RNE, tail, status, malformed-extent,
and alias gates as A4; it changes neither Q4 artifact identity nor persistent bytes.

On the same ROCm 10 R9700, every A8 activation byte and every `[7168,5120]` output passes its
independent exact-code/FP64 formula oracle at T=1..8,16,32,64,128 with zero BF16-step error. Four
nonzero lanes rotate by G64 group so every one of the 64 packed K-lane/WMMA fragment positions is
covered while the oracle still evaluates the complete represented-input-times-decoded-weight
formula. Against the represented-BF16-times-decoded-Q4 formula, maximum error falls from A4's
0.212891 to 0.015625. Interleaved complete-Op medians are 0.05874--0.08699 ms for A8 decode and
0.11070/0.12538/0.15366/0.30146 ms at T=16/32/64/128. A8 costs 8.7--24.3 percent over A4 across
the tested shapes, while remaining faster than the current W8 route at every shape.
ISA/resource evidence is native `v_wmma_i32_16x16x32_iu4`, 64 VGPR, zero LDS/scratch, and
occupancy 16 for A8Q4 WMMA; the selected wave32 A8 quantizer uses 10 VGPR and zero LDS/scratch. The
complete durable result, including all five per-route samples from alternating forward/reverse
route rounds plus both A4 and A8 timings and hardware/toolchain identity, is
`profiles/bench/r9700-a8q4g64-linear-qualification.json`. The same command directly qualifies the
compile-selected public Tensor/WorkspaceArena branch and exact workspace rewind in both builds;
those records are `profiles/bench/r9700-q4-tensor-dispatch-a4.json` and
`profiles/bench/r9700-q4-tensor-dispatch-a8.json`. The artifact-wide shape qualifier covers all
17 all-Q4 and six mixed-artifact Q4 shapes plus the five additional unique shapes used by the 32
DFlash Q4 matrices at T=1..8/32/128. Every one of 220 extents passes the exact codec and sampled
independent FP64 formula with zero BF16 steps. Seven-trial interleaved timing shows the wave32 and
64-thread quantizers are effectively tied at the narrowest points and favors wave32 by up to
2.034% for all-Q4 and 1.436% for DFlash. Its complete inventory and medians are
`profiles/bench/r9700-a8q4g64-artifact-shape-sweep.json`. Real-model A8 PPL/token evidence remains
a separate gate.

Existing base overlaps account for both `[5120,17408]` and `[34816,5120]`; the five added unique
shapes cover feature projection `[5120,25600]`, fused QKV `[6144,5120]`, attention output
`[5120,4096]`, the two convolution projections `[1280,5120]`, and selector hidden projection
`[256,5120]`.

`python3 tools/r9700/a8q4_prefill_traffic.py` is a host-only development calculator for the
T=4,096 quantized-prefill bottleneck. Its explicit A8Q4G64 model compares the current one-wave
M16/N16 global-load contract with M32/N64 and M64/N64 cooperative-LDS tiles at the three dominant
traced shapes. Its separate signed-A8G32/W8G32 model compares the one-wave route with the
M64/N64, sixteen-wave, 4,352-byte-LDS challenger at the four dominant mixed Text shapes. At those
four W8 shapes the challenger preserves issued wave count while reducing modeled requested bytes
to 17.05--17.17% of the old route. The report counts represented-code, FP16-scale, per-wave status,
and BF16-output requests; logical operations; padded/tail IU4 or IU8 issue; arithmetic intensity;
and static LDS. These are source/ISA-requested bytes, not HBM-byte or achieved-cache claims.
Optional `--bandwidth-gbps`, `--iu4-tops`, and `--iu8-tops` add scenarios using caller-supplied
requested-byte service and instruction rates; an HBM bus rating is not a valid requested-byte
bound when caches serve loads. The calculator has no implicit hardware defaults. Run
`python3 tools/r9700/test_a8q4_prefill_traffic.py` for its focused tile-accounting checks.

The selected T=1 A8Q4 native-dot8 route owns seven exact full-K Q4 matrix tuples. They were
identified from ordinary decode; `[5120,17408]` and `[34816,5120]` also occur in DFlash2, so
dispatch follows the Linear shape contract rather than caller identity. Every other T=1 tuple,
logical-K tail, and wider token extent retains WMMA. The retained static gate requires exactly
sixteen mixed-sign `v_dot8_i32_iu4` sites, no WMMA, at most 64 VGPR, and zero
LDS/private/scratch. The production regression uses dense varied activations, nonuniform signed
W4 codes in exact N16/K16 storage, nonuniform FP16 scales, full-output WMMA parity, spread-row
complete-K FP64 checks, status poisoning, full rewrites, and output canaries.

```sh
make -C tools/r9700 a8q4-decode-dot8-static
make -C tools/r9700 a8q4-decode-dot8-regression
```

The completed operator gate passed every tuple with a `14.1025415618 ms/token` robust weighted
saving lower. The source-matched C1/P8192+G256 ordinary Device Graph gate then reduced median decode
from `16.9262812925 s` (`15.1244089340 tok/s`) to `12.5301623450 s`
(`20.4307009719 tok/s`). Its robust candidate/control upper ratio was `0.7427031671`, robust saving
lower was `17.0000856399 ms/token`, prefill ratio upper was `1.0016298424`, and all generated tokens,
configuration, environment, artifact, and workspace values matched. The immutable whole report is
`profiles/bench/r9700-a8q4-t1-dot8-whole-p8192-g256-full-20260905.json`, SHA-256
`19278ca8c8df5d417bbd5d36ce1689760e3cb6dbe606c3a597a6b2f0b847fee1`. The former private build
selector and terminal A/B tooling were removed after promotion.

The G16 and G32 S16/tau900 XAttention build variants each expose a production-scale admission
control at context 8,192/T=4,096. Run each configured binary with `--benchmark
--production-scale --iterations 10 --out-json FRESH.json`, then validate it with `python3
tools/r9700/check_xattention_production_report.py --value-group 16 --executable BINARY FRESH.json` (or value group
32 for the G32 build). The validator rehashes the executable and every source from one exact source
root; `--executable` must resolve to the exact path recorded by the report and still have its
recorded bytes. Production-scale mode fails before device construction unless card2 is at
the exact `auto` power profile and the output path is unused, completes its numerical and rejection
gate before timing, rechecks the power profile after both phases, and exclusively publishes a
schema-v6 report containing raw samples plus executable, source, device, runtime, and power
identity. Ordinary benchmark-only schema-v5 reports remain diagnostic and are not production-scale
admission evidence.

The cooperative prefill CTAs are now production linear-Op routes at only their physically admitted
tuple predicates. A8Q4G64 uses a sixteen-wave 64-token-by-64-row CTA, stages one K64 group in a
6,400-byte fragment-major LDS tile, and issues the native low/high signedness pair of
`v_wmma_i32_16x16x32_iu4`. A8W8G32 is a separate sixteen-wave 64-token-by-64-row implementation,
stages one signed-I8 G32 group in 4,352 bytes of fragment-major LDS, and issues two native signed
`v_wmma_i32_16x16x16_iu8` operations per group. Both retain no scratch and bounded VGPR use.

The qualified Stage-1 memory pipeline is the production body for both cooperative CTAs. Each
barrier uses workgroup-local release, signal/wait, and acquire; the former broad `global_inv`
body is retained only under an explicitly named qualification entry for direct regression. Run
the exact bit-parity/status/tail matrices with:

```sh
make -C tools/r9700 a8q4-prefill-cta-lds-scope-regression
make -C tools/r9700 w8a8-prefill-cta-lds-scope-regression
```

Then pass the compiled production-entry assembly and metadata explicitly to its offline,
fail-closed static gate (both inputs may name the same device-only `.s` file):

```sh
make -C tools/r9700 prefill-cta-static-q4 \
  PREFILL_CTA_ASSEMBLY=/path/to/q4-production.s \
  PREFILL_CTA_METADATA=/path/to/q4-production.s
make -C tools/r9700 prefill-cta-static-w8 \
  PREFILL_CTA_ASSEMBLY=/path/to/w8-production.s \
  PREFILL_CTA_METADATA=/path/to/w8-production.s
```

`--mode lds-scope` selects only the promoted production symbol. For that exact selected mangled
Q4 or W8 specialization, the gate requires
the native IU4/IU8
WMMA instruction count, exactly two `s_barrier_signal` plus two `s_barrier_wait` operations, no
monolithic `s_barrier`, no `global_inv`, the selected LDS/VGPR ceilings, and zero private,
scratch, and flat-scratch storage. Symbol absence or duplication fails closed. The separate
`prefill-cta-static-incumbent-diagnostic` target expects the qualification-only former
incumbent's two
`global_inv` instructions and prints `admission=false`; it cannot admit a challenger. This is
only a static prerequisite. The regression above requires bit-exact output between production
and the former incumbent over the complete qualified Cartesian matrix plus the exhaustive
padded-tail/status checks. Stage 2 ping-pong staging is intentionally not part of this route.

The immutable pre-promotion schema-v1 reports are retained at
`profiles/bench/r9700-a8q4-prefill-cta-lds-scope-ab-20260904.json` and
`profiles/bench/r9700-w8a8-prefill-cta-lds-scope-ab-20260904.json`. They cover every admitted
shape at T=1024/2048/4096/8192 with one warmup and seven balanced forward/reverse samples per
route. Their weighted T=2048 challenger/incumbent ratios are respectively 0.9857744040 and
0.9431145048; every individual tuple is faster, bit-exact, and within the independent oracle's
zero-BF16-step result. Validate those reports only against their recorded executable and retained
pre-promotion source tree:

```sh
make -C tools/r9700 a8q4-prefill-cta-lds-scope-report \
  PREFILL_CTA_LDS_SCOPE_OUT=/retained/path/q4-lds-scope-ab.json
make -C tools/r9700 w8a8-prefill-cta-lds-scope-report \
  PREFILL_CTA_LDS_SCOPE_OUT=/retained/path/w8-lds-scope-ab.json
```

The schema-v1 validator independently reopens and hashes the exact executable and four canonical
source roles, checks the live `auto` state, reconstructs all balanced samples, medians, ratios,
the complete Cartesian inventory, and the weighted T=2048 decision. Passing this operator A/B is
admission provenance, not post-promotion whole-model performance evidence. Because production
source changed at promotion, current-root revalidation is expected to fail closed; use an exact
retained source root matching the report when renewed validation is required.

The selected Q4 prefill CTA is the M64xN128 persistent-N2 route. Its 16 waves retain the M64
token tile, but each wave owns two N16 fragments. Two 8,576-byte LDS banks alternate the current
and next activation/weight tiles, so the two independent IU4 accumulator chains share activation
traffic while four next-group code dwords overlap current WMMA. It is the sole production CTA for
the exact eight qualified tuples at
T=1024/2048/4096/8192; M64xN64 remains only as a directly named regression control, and the
one-wave route retains all other supported extents. Run the production static gate and direct
control regression with:

```sh
make -C tools/r9700 a8q4-prefill-cta-m64n64-regression
make -C tools/r9700 prefill-cta-static-q4 \
  PREFILL_CTA_ASSEMBLY=/path/to/current-linear.s \
  PREFILL_CTA_METADATA=/path/to/current-linear.s
```

The production static gate requires its unique selected symbol, eight native IU4 WMMA instructions,
two LDS signal/wait pairs, no monolithic barrier or `global_inv`, exactly 17,152 bytes of LDS,
at most 96 VGPR, and zero private/scratch storage. The immutable pre-promotion physical report is
`profiles/bench/r9700-a8q4-prefill-cta-n128-ab-20260904.json` (SHA-256
`9f6a49e725bc3f13ade5775332b95848a36ad796cb4ff25ea9b55ae4838f674e`). It covers the full
8-shape by 4-token-extent matrix, selected N128 at weighted T2048 ratio `0.8319113784`, and records
92 VGPR, 8,576-byte LDS, and zero scratch for the then-challenger single-bank N128 route. The later
ping/pong evidence below is the current production authority.

The qualification-only M128xN128 challenger doubles the token tile to 128 while retaining two
N16 fragments per wave. Its 32-wave workgroup stages exactly 12,800 bytes and halves requested
weight traffic across token tiles without changing the represented A8-low/A8-high, signed-Q4,
INT32 recombination, FP32 group-order accumulation, or final BF16 rounding boundaries. Production
remains M64xN128 until physical admission. Build and statically gate it, then run the bounded
numerical and physical comparisons only under the normal serialized GPU procedure:

```sh
make -C tools/r9700 prefill-cta-static-q4-m128n128 \
  PREFILL_CTA_ASSEMBLY=tools/r9700/build/w8a8_wmma_linear_qual.s \
  PREFILL_CTA_METADATA=tools/r9700/build/w8a8_wmma_linear_qual.s
make -C tools/r9700 a8q4-prefill-cta-m128n128-regression
make -C tools/r9700 a8q4-prefill-cta-m128n128-benchmark \
  PREFILL_CTA_M128N128_OUT=/fresh/path/q4-m128n128-ab.json
make -C tools/r9700 a8q4-prefill-cta-m128n128-report \
  PREFILL_CTA_M128N128_OUT=/fresh/path/q4-m128n128-ab.json
```

The static gate requires the unique M128xN128 symbol, eight native IU4 WMMA instructions, two
workgroup signal/wait pairs, no monolithic barrier or `global_inv`, exactly 12,800-byte LDS,
at most 96 VGPR, an exact 1,024-thread maximum workgroup, and zero private/scratch storage. The
schema-v1 A/B covers all eight production shapes at T=1024/2048/4096/8192 after tail/status/oracle
qualification, retains seven balanced forward/reverse samples, rejects any tuple above 1.01x,
and requires at least 150 ms weighted T2048 saving before it can be considered for promotion.
An M128xN128 performance rejection is still terminal evidence: after every numerical, resource,
power, executable, and source gate has remained valid, the benchmark exclusively publishes all
32 raw/median rows, the exactly derived failing-tuple inventory, and
`qualification_rejected_production_unchanged`, then returns nonzero. The validator accepts that
envelope as a valid rejected qualification record without treating it as admission.

The Q4 ping/pong route keeps the single-bank M64xN128 arithmetic but
prefetches the four K-group payload dwords owned by each lane into registers while native IU4
WMMA consumes the current LDS bank. It then publishes those payloads and scales into the alternate
bank with one workgroup-scoped handoff. Full production tiles take this exact fast path; the same
entry deliberately routes logical-K, row, and token tails through the named single-bank regression
control so the bounded status/alignment/tail matrix remains complete.

```sh
make -C tools/r9700 prefill-cta-static-q4 \
  PREFILL_CTA_ASSEMBLY=build/w8a8_wmma_linear_qual.s \
  PREFILL_CTA_METADATA=build/w8a8_wmma_linear_qual.s
make -C tools/r9700 a8q4-prefill-cta-pingpong-regression
make -C tools/r9700 a8q4-prefill-cta-pingpong-benchmark \
  PREFILL_CTA_PINGPONG_OUT=/fresh/path/q4-pingpong-ab.json
make -C tools/r9700 a8q4-prefill-cta-pingpong-report \
  PREFILL_CTA_PINGPONG_OUT=/fresh/path/q4-pingpong-ab.json
```

The production static gate requires the unique selected symbol, eight native IU4 instructions, exactly
17,152 bytes of LDS, at most 96 VGPR, occupancy 16, a 512-thread maximum workgroup, and zero
private/scratch storage. The no-clobber physical A/B records seven balanced forward/reverse samples
for every eight-shape by four-token-extent row. Admission requires no individual row regression
and at least 1.5x weighted T2048 speedup; either outcome is published as terminal qualification
evidence before the executable returns success or rejection.

The retained run passed the exact oracle and every tuple was nonregressing, but weighted P2048
improved only from `1126.778368` to `914.413888 ms` (`0.8115295`, `1.23224x`, saving
`212.364480 ms`), below the predeclared `1.5x` gate. Its terminal report is
`profiles/bench/r9700-a8q4-prefill-cta-pingpong-ab-20260904.json`, SHA-256
`06a2846e1ae5ba90989dcb401e578a1b96479a8e472b7aa0d5befdcbcf6281f7`.
That fixed operator gate remains a valid rejection at its claimed scope. A later matched
whole-P2048 gate nevertheless selected its aggregate operator saving and promoted ping/pong. The
subsequent source-matched scalar-base/U32-voffset qualification improved the exact 48/64/64
operator aggregate by a robust-lower `21.057251555 ms` and whole P2048 by a robust-lower
`20.306733144 ms`, with exact token identity and unchanged workspace/environment. Scalar-base
addressing is therefore part of the sole production ping/pong route for the exact eight-shape by
P=1,024/2,048/4,096/8,192 predicate. It reports
`q4_prefill_cta_profile=m64n128-pingpong-n16-k16-scalar-base-production`; every other valid direct
tuple delegates to the predicated, `size_t`-addressed M64xN128 fallback. There is no build or runtime
selector or duplicate full-tile production kernel. The retained scalar operator and whole reports
are `profiles/bench/r9700-a8q4-n16k16-scalar-base-product-p2048-ab-20260905.json` and
`profiles/bench/r9700-scalar-base-production-p2048-c1-ab-20260905.json`.

Two M64xN256 Q4 prefill experiments are terminal rejections and their executable paths have been
removed. The 16-wave/512-thread variant passed numerical qualification and improved the weighted
P2048 operator aggregate by `1.20609x`, but missed its predeclared `1.5x` gate. Its immutable report
is `profiles/bench/r9700-a8q4-prefill-cta-m64n256-ab-20260904.json`, SHA-256
`e1d611ff2a74ebad9c300297c93ea1722a74c57d01c71a75836987fe7fe3ba3b`.

The trace-selected 32-wave/1,024-thread MLP-down-only variant compiled to eight IU4 WMMAs, 87 VGPR,
25,856 bytes LDS, and zero private/scratch storage, but was slower at every qualified extent. At
T2048 it measured `3.419854 ms` versus `3.379034 ms` for production (ratio `1.0120804`); its
immutable report is `profiles/bench/r9700-a8q4-m64n256-w32-mlp-down-ab-20260904.json`, SHA-256
`a41e2a2ad65809ade16b629e698244834d2c2da065335f8fab62869813aaef5b`. Production remains the
M64xN128 ping/pong route; no adjacent N256 or cache-hint sweep follows these rejections.

The production-geometry next-G64 scale-prefetch qualification is also terminally rejected and its
API, kernel specialization, checker mode, and harness modes have been removed. Although static
inspection passed at eight IU4 WMMAs, 89 VGPR, 17,152-byte LDS, occupancy 16, and zero scratch,
all 12 MLP-down/GDN value-Z/GDN output cells lost. Trace-call-weighted P2048 regressed from
`375.835487` to `391.499994 ms` (`1.041679x`). Its immutable report is
`profiles/bench/r9700-a8q4-prefill-cta-scale-prefetch-ab-20260904.json`, SHA-256
`aaeeaf16a0040d5a23046d40fff532f34805124cdd3e5bbeb5d0a721962ad8ef`. Production continues to
load each next activation/weight scale after the current group compute before publishing the next
LDS bank.
The qualification-only signed-A4G64 x signed-Q4G64 M64xN128 route tests a private activation
profile change rather than an equivalent implementation. It uses one packed signed-A4 plane and
two K32 IU4 operations per output fragment/G64, versus the production A8 split plane's four. The
M64xN128 kernel compiles to four static signed/signed IU4 instructions per loop, 84 VGPR, 6,528
bytes LDS, occupancy 16, and zero private/scratch. Its complete path also halves activation-code
workspace and global reads. Production remains A8 regardless of the operator result until separate
PPL, exact-token, and whole-inference gates select the profile.

```sh
make -C tools/r9700 build/a4q4_prefill_qual -B -j1
make -C tools/r9700 prefill-cta-static-q4-a4-m64n128 \
  PREFILL_CTA_ASSEMBLY=build/w8a8_wmma_linear_qual.s \
  PREFILL_CTA_METADATA=build/w8a8_wmma_linear_qual.s
make -C tools/r9700 a4q4-prefill-regression
make -C tools/r9700 a4q4-prefill-benchmark \
  A4Q4_PREFILL_OUT=/fresh/path/a4q4-m64n128-vs-a8-production.json
```

The harness compares complete quantize-plus-matrix calls on identical BF16/Q4 operands. It
requires the exact independent A4 codec image, bit-exact output against the signed-A4 one-wave
reference, separate A4/A8 FP64 represented-code oracles within two BF16 steps, and raw seven-sample
timings over all eight production shapes and four chunk extents. The retained result failed the
operator gate: weighted P2048 improved only from `1121.674698` to `1006.050599 ms` (`1.11493x`),
and N1024 regressed at all four extents by `1.21819x` through `1.35600x`. Its report is
`profiles/bench/r9700-a4q4-prefill-m64n128-vs-a8-production-ab-20260904.json`, SHA-256
`2ac938b230197384a476bd57ccfd39d72a1b73c03bee82c58bf3be9df7de5b1f`. Production remains A8;
the already-rejected A4 quality profile was not reevaluated.

The production low-context dense attention route is the selected three-stage full-score GQA6 path
in `fp8_int4_kv_attention.hip`. It accepts only an initial-prefix P=128..4096 call with token-fastest
FP8 K plus feature-fastest signed INT4 V and FP16 scales. QK uses Bk16 below P512 and the selected
Bk32 schedule from P512 through P4096; each tile is shared across the six query heads of one KV
head and writes FP32 scores. A separate maximum pass validates each causal
vector; PV forms FP32 probabilities and reuses each direct INT4/FP16-scale V tile across the same
six heads. The caller-owned score/max workspace is reused across layers at its planner-stable arena
address. P<128, later chunks, tree attention, and other layouts retain their prior routes.

After the serialized campaign permits a new build/GPU run, execute the independent FP64 harness:

```sh
make -C tools/r9700 dense-prefill-attention
```

It covers the production G16/G32 full-score route at P128/512/1024/2048/4096 with fragmented
physical pages, arbitrary device causal positions, and the exact stored cache planes.
`dense-prefill-attention-full-score-isa` prints both selected QK kernels plus maximum and PV. Invoke
`dense-prefill-attention-full-score-static` with the exact selected mangled symbol and
`DENSE_ATTN_STAGE=qk_bk16`, `qk_bk32`, `maximum`, or `pv`. The checker requires exactly sixteen
BF16 WMMAs for Bk16 and 32 for Bk32, and forbids every WMMA opcode in maximum/PV. The emitted
gfx1201 resources are Bk16 QK 29 VGPR/8,296-byte LDS/occupancy 15, Bk32 QK 97 next-free
VGPR/16,488-byte LDS/occupancy 11, maximum 17 VGPR/64-byte LDS/occupancy 16, and PV 116
VGPR/occupancy 12 with 9,208-byte G16 or 8,952-byte G32 LDS. Every stage is wave32 WGP mode with
zero private/scratch/flat-scratch and zero register spills; PV must contain native FP32 exp and
FP32 FMA/FMAC.

The full-score route uses a caller-owned reusable FP32 workspace of
`(24*P*P + 24*P)*4` bytes (384.1875 MiB at P2048). Its 192-thread QK stage launches one Bq16 CTA
per KV head/query tile/key tile; Bk32 halves the key-tile workgroups and reuses each represented
query fragment across two K16 WMMA fragments. Six query-head waves share each decoded FP8-K tile; its
maximum stage validates and reduces each causal FP32 score vector; its 256-thread PV stage stages
FP32 probabilities plus each signed-INT4/FP16-scale V tile once for all six GQA heads. PV retains
FP32 exponential, denominator, and numerator arithmetic and contains no WMMA. Inspect all staged
ISA with `dense-prefill-attention-full-score-isa`; invoke
`dense-prefill-attention-full-score-static` with `DENSE_ATTN_STAGE=qk_bk16`, `qk_bk32`, `maximum`,
or `pv` and the
exact corresponding non-active mangled symbol. The retained selector timed the three-launch Op as
one HIP-event sample after every stage and the independent FP64 oracle passed.

For G16 and G32 at P128/512/1024/2048/4096, the retained pre-promotion report compared the complete
incumbent fused call with Bq4/Bq8/Bq16 after independent FP64 qualification. It timed one complete Op
call per event sample using seven rotating forward/reverse route pairs, preserving all fourteen
raw samples and each route's real CTA count. Bq16 won every measured cell. Its exact
selected/incumbent median ratios are G16 0.561224173/0.566735009/0.504308119/0.437316979/0.416703457
and G32 0.546212923/0.554317816/0.495417690/0.429230227/0.417418378 in increasing P order. The
retained report is `profiles/bench/r9700-dense-prefill-attention-ab.json`, SHA-256
`d7a0f50a6ae91448583b07d7477d2926b950b65eeadd3ac924f28a8354ed95a6`. The selected full-score report
is `profiles/bench/r9700-dense-prefill-full-score-ab-20260904.json`, SHA-256
`0f80353256105a6b759b1f906f8cfc1cee1ef204870c83f1d3dff7ca0d69dcca`. At P2048, full-score/fused
Bq16 medians are 14.669395/61.569540 ms for G16 and 14.819379/61.336494 ms for G32.
The selected crossover report is
`profiles/bench/r9700-dense-full-score-bk16-bk32-crossover-ab-20260904.json`, SHA-256
`7941bb518ce2a2287e06403952ad9d90f2ada115550c1967e474906f5a600a2f`. It selects Bk16 at P128 and
Bk32 at P512/1024/2048/4096 for both G16 and G32; production keeps Bk16 for the intervening
unmeasured P129..511 fallback. The focused qualifier now covers production correctness only.

The subsequently isolated dense P2048 FP8-Q/Bk32 route passed its exact FP8-Q panel, independent
FP64 score and complete-attention oracles, workspace overwrite/liveness, and static resource gates,
but failed its fixed physical threshold. Candidate/incumbent medians were 2.983591080/3.442310095 ms
(`0.8667409378x`); 16 calls total 47.73745728 ms, only 7.33950424 ms matched saving rather than the
required 15 ms (candidate ceiling 32.439611 ms). Production remains the BF16-WMMA Bq16/Bk32 route,
and all FP8-Q/Bk32 candidate modes and checker surfaces are removed. The immutable design report is
`profiles/bench/r9700-dense-full-score-fp8-q-bk32-static-design-20260904.json` (SHA-256
`aed480c787313c280aefc83d8cdf943a6b81711872786677f905f3d4036cd521`); the terminal physical report
is `profiles/bench/r9700-dense-full-score-fp8-q-bk32-ab-20260904.json` (SHA-256
`ef0d6557a59145c3f632b96b24f41c922369d9f42fa48ab2c7b24d5b1c7e7f30`).

The immutable pre-promotion A8Q4 schema-v3 report is
`profiles/bench/r9700-a8q4g64-prefill-cta-qualification.json` (SHA-256
`b3c78cab21dfc0f97521269245b8442009048aadcd9db5ffd947a4d7d3baf889`). It passed the exact
eight-shape by T=1,024/2,048/4,096/8,192 matrix, including all six reachable base-Text tuples and
two bulk-MTP execution views, with zero maximum BF16 steps and `2.5261x` through `8.1587x`
speedup. The corresponding W8 report is
`profiles/bench/r9700-w8a8g32-prefill-cta-qualification.json` (SHA-256
`1547f7d62fae4077f32f203b61831fc97b4f98bffcc930c6a29de7284f5c40c0`). It passed the four
mixed-Text tuples `[7168,5120]`, `[12288,5120]`, `[5120,6144]`, and `[5120,17408]` at the same
four extents with zero maximum BF16 steps and `14.8205x` through `18.4695x` speedup. Both gates
also passed exhaustive logical-K193/padded-K256 T129/N65 tail, packed-code alignment, poisoned
padding, and nonzero-status rejection checks.

Production predicates exactly match those measured Cartesian sets. Decode, partial chunks,
Vision, DFlash, and unqualified Text/MTP shapes continue to use the incumbent one-wave route. The
reports bind the then-qualified executable and source bytes and remain immutable admission
provenance; no fresh admission-report producer remains after promotion. Rebuilt whole-model
prefill evidence is the next performance authority.

`ninfer_r9700_roctx_qual` is built by the main CMake tree. It protects the common runtime tracing
facade's stable names, categories, and nested RAII lifetime while proving that destructors leave
the native ROCtx push/pop stack balanced. ROCtx carries only string labels, so the facade encodes
the category, registered name, and numeric payload in one fixed allocation-free label.

Capture the named page-boundary trace with `make -C tools/r9700 profile-trace`. It records HIP
runtime, dispatch, and memory-copy activity for the D256/Hq24/Hkv4 66-token workload in
`profiles/r9700` by default. Collect the validated gfx1201 counters in a separate kernel-specific
pass with `make -C tools/r9700 profile-pmc` because profiler interception is not a latency
benchmark. That active recipe records only SQ busy cycles and wave counts, the two events that
produce nonzero per-dispatch data on rocprofiler-sdk 1.3.5 from ROCm 10.

Use a new explicit directory for each physical capture; never select a database by glob or
modification time:

```bash
make -C tools/r9700 PROFILE_DIR=../../profiles/r9700/<named-trace> profile-trace
make -C tools/r9700 PROFILE_DIR=../../profiles/r9700/<named-pmc> profile-pmc
```

After naming the emitted database paths, `make -C tools/r9700 rocm-tool-audit` performs the
GPU-independent capability audit over the retained captures. Override `TRACE_EVIDENCE_DB`,
`PMC_EVIDENCE_DB`, the paired `TRACE_EVIDENCE_COMMAND`/`PMC_EVIDENCE_COMMAND`, or one of the
explicit zero-probe database variables to audit a replacement capture. It validates target
identity, database schema, exact workload command, required trace rows,
nonzero SQ counters, identically-zero unsupported counters, and database hashes. It also records
the installed tool versions and performs compile-only device-ASan target probes. The atomic JSON
result is `profiles/r9700/rocm10-tool-capability-audit.json`; run
`make -C tools/r9700 rocm-tool-audit-test` for its dependency-free SQLite parser test. Neither
target loads HIP or launches a kernel.

The retained page-boundary trace database has SHA-256
`4db8dd6a73886a06f23b32ecd615b37c7b592d3c66ad2b9644779688d3af4f21` and contains 1,024 kernel
dispatches, 224 memory copies, and 3,130 runtime regions. The paired PMU database has SHA-256
`b90405a725ecaa64007c8078b5c06cfffc9f8a8daa593ee0f34bf036b0e0a05f`; SQ busy cycles are nonzero
in 32,586 of 32,768 samples and SQ waves in 14,960 of 32,768 samples. Both databases record the
exact `./build/kv_op_qual --context 66 --iterations 1` workload and an
`AMD Radeon AI PRO R9700`/`gfx1201` agent.

The installed SDK catalog and amdgpu `soc24_enum.h` agree on the gfx1201 selector numbers for
generic and wave32 VALU/LDS, TCP request/miss, and GL2C hit/miss events. This rules out stale names
or a project-side selector translation. Nevertheless, the recorded isolated generic VALU pass is
zero for all 57 dispatches, and isolated `SQ_INSTS_WAVE32_VALU`/`SQ_INSTS_WAVE32_LDS` plus separate
TCP and GL2C probes are zero for all 93 dispatches each despite known matching instructions and
traffic. Multiplexing and counter-group conflicts are therefore also ruled out. The associated
databases are retained under `profiles/r9700/rocm10-gfx12-pmc-probe`; none of those zero events is
usable kernel evidence. ROCm Compute Profiler 3.8.0 has no gfx1200/gfx1201 SoC class, profile set,
or analysis configuration, so constructing a local report by copying the gfx115x definitions
would apply the wrong architecture-specific formulas rather than fix collection.

The local build derives one ROCm root from the selected `HIPCC` and puts that release's headers and
libraries ahead of stale distro HIP 5 development files currently installed under `/usr`. This is
isolation for the qualification tool, not a compatibility layer.

`make -C tools/r9700 sanitize` rebuilds the HIP core/cache ownership qualifier with host Address
Sanitizer, then runs the same arena, graph, transfer, typed cache mapping, RAM restore, and
append/compact transaction gates. The R9700 agent reports XNACK disabled. ROCm 10 Clang warns that
device AddressSanitizer is unsupported for plain `gfx1201` and rejects `gfx1201:xnack+` as an
invalid target ID; its driver expansion applies AddressSanitizer only to the host compilation.
The build therefore explicitly disables device instrumentation rather than generating an invalid
code object. This is a focused host ownership check, not a device-bounds claim or performance
build, and no project-side compiler flag can provide the missing XNACK hardware target.

The CPU-only audit pins this classification to HIP 7.15.26333, AMD Clang 23.0.0git commit
`8f497e0992fb7513f7f78a6f6b6f1056c375e961`, rocprofiler-sdk 1.3.5 commit
`6b0e43f341195e203754e08f850e437ff2fc09f9`, ROCm Compute Profiler 3.8.0 commit `16adc4d`, and
ROCm-GDB 16.3. Compiling `gfx1201` with `-fsanitize=address` succeeds only after Clang warns that
device instrumentation is ignored; compiling the same probe for `gfx1201:xnack+` fails because
that target ID is invalid. The full compiler commands and outputs are retained in the audit JSON.
ROCm System Profiler 1.8.0 is installed, but its optional CPU PAPI source is disabled while
`/proc/sys/kernel/perf_event_paranoid` is 4; this does not affect rocprofv3 GPU tracing.

The executable checks:

- GPU OCP E4M3 conversion against canonical encodings and the HIP host conversion;
- GPU E4M3 decode against an independent exact bit decoder;
- direct wave32 `FP8 x FP8 -> FP32` gfx12 WMMA with an all-ones `16x16x16` product;
- an owned raw-register asymmetric `16x16x16` WMMA fragment map, including nonuniform A, B,
  and FP32 accumulator seed operands, checked exactly against the independent logical oracle;
- symmetric signed INT4-G32 packing (`-7..7`) with an FP16 stored scale and exact GPU
  dequantization;
- ROCm 10 baseline latency and effective minimum-payload throughput for standalone FP8 encode/decode and INT4-G32
  encode/decode.

Useful options are `--elements N` and `--iterations N`. The element count must be a multiple of 32.
Defaults are 67,108,864 elements and 100 timed iterations so the working set exceeds the R9700's
last-level cache. Standalone timings are qualification data only and must not select production
kernels.

V uses the canonical adjacent-pair packing: byte `i` holds element `2i` in its low nibble and
element `2i + 1` in its high nibble. The standalone codec timings are bring-up evidence only;
production speed decisions require the complete fused attention consumer.

`kv_op_qual` extends the codec probe into a correctness-first asymmetric KV Op at the real text
attention shape `D=256`, `Hq=24`, `Hkv=4`. It qualifies G16 and G32 with both feature-fastest and
token-fastest physical planes, fragmented 64-token page tables, append across positions 63/64/65,
bit-exact speculative-tree copy/compaction, and a complete cached QK/softmax/PV consumer checked at
every output against an independent host FP64 formula and the host-only canonical codec authority.
Generated K, V, and Q fixtures are explicitly rounded to represented BF16 before either path sees
them, and the device append codec consumes actual BF16 storage. Its append dispatch links the
production R9700 source at `src/ops/r9700/kv/fp8_int4_kv_append.hip`; the executable also reports
ROCm 10 baseline
whole-append and whole-attention latency. Use `--context N`, `--rows T` (any positive U32 extent),
and `--iterations N`; defaults are 256, one, and 20. Rows model a causal staged-prefill suffix:
each has distinct represented BF16 Q and a separate device-I32 position while sharing the paged
typed cache. The raw launcher partitions rows only when required by gfx1201's grid-Y limit; it
does not allocate or materialize scores.
The streamed fused A3 route has no context-sized score buffer, so this qualification
accepts contexts through 32,768 tokens; use a separate repeat timing sweep after its numerical
gate succeeds.

The CMake-built `ninfer_r9700_full_attention_qual` exercises the production Qwen3.8 target leaf
over its generation-bound typed cache. In addition to causal and pending-publication checks, it
qualifies packed-tree rows at context 257 with prefix 253, positions 253..256, and sibling-skipping
masks 1/3/5/13 against an independent FP64 oracle. The same expected output is required from a
rowwise prefix array and the schedule's zero-stride per-sequence scalar broadcast. It also proves
that malformed device tree metadata poisons only the affected row and that the unadmitted FP8-Q
WMMA challenger rejects tree masks rather than silently applying causal attention.
An optional device-I32 active-row scalar supports fixed-width captured MTP panels: null means all
rows, values from zero through T execute only that row prefix and write exact positive zero to the
tail before reading its query/position/tree/cache inputs, and negative or over-T values poison the
whole invocation with NaN. The qualifier checks zero, partial, full, negative, and over-T counts,
a valid invocation immediately after each error mode, and a partial packed-tree panel whose
inactive rows contain deliberately malformed metadata. Those count changes replay one captured
fixed-address Device Graph rather than rebuilding or recapturing the attention launch.

`ninfer_r9700_split512_attention_qual` is the retained direct admission and regression harness for
the production long-context split-512 decode leaf. Its cache fixture consists of deterministic
host-owned FP8 key bytes, packed signed-INT4
value bytes, binary16 scale words, and a fragmented page table, and its independent FP64 formula
decodes those stored representations directly. The bounded matrix covers contexts
1/63/64/65/511/512/513/1025, T=1 and T=4, G16 and G32, active-row prefixes, row-local invalid
causal/tree metadata, a nonzero page-table-row selection, malformed row selectors and physical-page
mappings, and both T=1 and T=4 fixed-address Device Graph replay. T=1 additionally checks shortened
and invalid row positions; its oracle includes the challenger's explicit private E4M3 query cast,
while T=4's oracle consumes represented BF16 queries.
The edge matrix checks every output element. At the real 8,192- and 32,768-token decode contexts,
the same independent oracle evaluates complete-context score and softmax streams for fixed samples
of rows and GQA heads, then checks representative output features for both T=1 and T=4 in G16 and
G32; this bounds host reference time without truncating the long-context attention formula.
Use the standalone Makefile targets `split512-attention-isa` and
`split512-attention-resources` as failing gfx1201 gates before a physical timing comparison. They
require native FP8 WMMA in the retained T=1 QK and native BF16 WMMA in every T=4 QK
specialization, zero private storage and scratch in every participating kernel, and enforce the
resource ceilings for T=1 QK (0-byte LDS/24 VGPR), T=4 QK (9220-byte LDS/33 VGPR), partial
(6224-byte LDS/26 VGPR), and merge (2124-byte LDS/25 VGPR). Building the qualifier and passing
these static gates remains a required regression gate for the admitted leaf. `--workspace-only`
is CPU-only and proves the exact 262,144-token capacities
(37,847,040 bytes at T=1 and 151,388,160 bytes at T=4), maximum-plus-one/overflow rejection,
without constructing a device context.

The split-512 admission command fails before device construction unless the power profile is
`auto` and the output path does not exist. It runs the complete numerical, rejection, and fixed-
address Device Graph suite first, then alternates complete incumbent and split-leaf measurements
at T=1/4, context 8,192/32,768, G16/G32, and every active T=4 prefix 1..4. The report retains raw
samples and recomputed means/speedups together with exact executable/source-tree bytes, gfx1201
wave/device/runtime identity, live and recorded power state, and structured native-context
workspace boundaries. Schema v2 requires exactly 20 iterations and validation against the same
canonical executable path and source tree. The retained 2026-09-04 report passed this gate before
promotion; reruns must use a fresh output path because the report writer is deliberately
non-clobbering.

```bash
test "$(tr -d '\n' </sys/class/drm/card2/device/power_dpm_force_performance_level)" = auto
OUT=profiles/bench/r9700-split512-attention-admission-20260904.json
test ! -e "$OUT"
build-r9700/src/ninfer_r9700_split512_attention_qual \
  --benchmark --iterations 20 --out-json "$OUT"
python3 tools/r9700/check_split512_report.py \
  --executable build-r9700/src/ninfer_r9700_split512_attention_qual "$OUT"
```

A later grouped-KV-head T=1 PV experiment was rejected at the whole-runtime screen. The standalone
mechanism report, `profiles/bench/r9700-split512-grouped-pv-qualification-v2-20260906.json`
(SHA-256 `f690accdc47bd85096aa412aa415e3f55446e8d9b737b9928cef671b1054ea6d`), is screen-only:
it establishes that V reuse can accelerate its reduced fixed fixture, not that the production leaf
is faster. The product-integrated direct qualifier then passed the complete numerical, rejection,
and Device Graph suite in `profiles/bench/r9700-split512-grouped-t1-direct-20260906.json`
(SHA-256 `958b38efaced4b671222c32de93ce72264df6a58aa6fbb5797f0738e965ba65b`). Despite that exact
correctness, the C1/P8192+G32 whole screen rejected the route: candidate/control decode ratio
`1.0038023342`, saving `-0.141697625 ms/token`, and prefill ratio `1.0033795393`, with exact
generated tokens. Its report is
`profiles/bench/r9700-split512-grouped-t1-whole-p8192-g32-screen-v2-20260906.json` (SHA-256
`bcd24621462dbb404ecc77b79a6b324f0c4d4b67a7e0afdcbacc69281cd1d9f2`). No full gate was run;
the challenger, selector, and temporary qualification tooling were removed. The retained
per-query-head split-512 producer remains production.

The separate `kv_op_qual --timing-only` mode preserves device append, raw-ABI rejection,
fragmented-map, and timing work but omits its repeated host FP64/QK/PV output comparisons. It is
allowed only after that identical shape/layout route has passed a recorded full-oracle run, and
exists to prevent CPU reference work from contaminating its context/T latency matrix.
`kv_op_qual --transaction-timing-only` is the narrower repeat recipe for the production-shaped boundary: it
times the T-row suffix A2 append, selected A3 attention, and their same-stream ordered pair while
omitting standalone QK/PV and the unselected WMMA challenger timings. The fixture still builds the
cache through the production codec and retains the transaction/error/compaction checks.

The post-ROCm-10 append-fusion decision used G16/token-K/feature-V/feature-scale, the provisional
whole-attention leader, on the physical `gfx1201`. Direct full-oracle runs passed first at 1K for
T=1, 8, 9, 17, and 128; the selected represented-BF16-Q score-streaming output had maximum absolute
error at most `3.47e-8` against the independent FP64 attention formula. The already-qualified
page-boundary corpus covers every decode width T=1..8. Three uncontended event-timing repeats then
gave these medians; the last value assumes an impossible fusion that makes the complete append
free, so it is a strict upper bound rather than a claimed speedup:

| T | Context | suffix A2 (ms) | A3 (ms) | maximum possible reduction |
|---:|---:|---:|---:|---:|
| 1 | 1,024 | 0.0052 | 0.6693 | 0.77% |
| 1 | 4,096 | 0.0053 | 2.5978 | 0.20% |
| 1 | 8,192 | 0.0052 | 5.1744 | 0.10% |
| 1 | 32,768 | 0.0057 | 20.6455 | 0.03% |
| 8 | 1,024 | 0.0051 | 0.7498 | 0.68% |
| 8 | 4,096 | 0.0049 | 2.9246 | 0.17% |
| 8 | 8,192 | 0.0053 | 5.7520 | 0.09% |
| 8 | 32,768 | 0.0064 | 22.0522 | 0.03% |
| 128 | 1,024 | 0.0366 | 9.1854 | 0.40% |
| 128 | 4,096 | 0.0378 | 36.5133 | 0.10% |
| 128 | 8,192 | 0.0388 | 74.4051 | 0.05% |
| 128 | 32,768 | 0.0416 | 301.5830 | 0.01% |

T=2..7 at 1K remained below a 0.80% free-append bound; staged-prefill T=9 and T=17 at
1K/4K/8K/32K remained below 0.68%. Directly timed ordered A2+A3 pairs showed no repeatable positive
delta beyond normal clock variance. A simple single-kernel merge is also invalid: the independent
T×Hkv append workgroups must complete their exact three-plane writes and status publication before
the T×Hq attention readers, while an ordinary HIP launch has no inter-workgroup barrier. A
cooperative barrier cannot admit the T=128 grid, and making every GQA reader encode its own suffix
would duplicate the stored-format work sixfold and bypass the transaction's single-writer/status
ownership. The sub-percent zero-cost bound cannot justify that larger private profile. Production
therefore retains one ordered A2 launch followed by the already-fused paged QK + online FP32
softmax + exact INT4×FP16-scale PV A3 launch. Direct gfx1201 metadata inspection keeps the selected
A2 at 13 VGPR, zero LDS/scratch, occupancy 16 and A3 at 23 VGPR, 52-byte LDS, zero scratch,
occupancy 16; no speculative fused code or unsupported profiler counter informed the rejection.

For a kernel-specific profiler pass, all four of `--group`, `--key-order`, `--value-order`, and
`--scale-order` select exactly one already-compiled variant. The default remains the complete
sixteen-variant qualification matrix. The checked-in PMU recipe names the current provisional
timing leader explicitly and profiles it at a 4,096-token context; changing that recipe requires a
new complete unprofiled timing matrix and prior full-oracle evidence for the replacement route.

Short causal tails are first-class qualification inputs: `--context 1`, `63`, `64`, and `65` run
the same independent codec, QK, FP32-PV, fused-attention, compact, and invalid-mapping checks. The
63/64/65 cross-page append path is covered by every run at 66 tokens or more.

The qualification also invokes the owned vector FP8-K QK baseline from
`src/ops/r9700/kv/fp8_int4_kv_attention.hip`. It writes one FP32 score per logical context token,
uses the selected closed K-plane layout and page table directly, and is checked score-by-score
against a host FP64 formula. The same source also contains a raw gfx12 wave32 WMMA challenger:
six Q heads belonging to one KV head occupy the first six rows of a 16x16 tile, sixteen context
tokens occupy the columns, and ten unused rows are zero. It converts represented BF16 Q to OCP
E4M3 at the declared private QK boundary, is checked against an independent FP8-Q host formula,
and its emitted `v_wmma_f32_16x16x16_fp8_fp8` appears in the candidate's own ISA. The vector and
WMMA timings do not select the final fused A3 implementation.

The append source receives a zeroed device status word. It reports nonfinite BF16 K/V input,
unrepresentable finite FP16 V scale, a device position that differs from the authorized contiguous
suffix, and invalid logical or physical page addresses. A nonzero status invalidates the entire
append transaction and must prevent its cache commit. The qualifier exercises each error path in
addition to its normal exact-plane comparison. The canonical decoder-state owner can stage its
checked host U32 positions once or consume an existing device I32 panel directly; in both cases the
kernel validates `position[i] == old_frontier+i`. Valid page-table IDs and non-overlapping output
planes remain cache-owner preconditions enforced by the paged-cache allocator.

Every normal exact-plane run also injects a codec-edge corpus: signed K zero, finite K saturation,
INT4 quotient ties, signed V zero, and a finite BF16 V group whose FP16 scale canonically
underflows to positive zero. The same run rejects an unrepresentable scale, empty raw A2/A3
transactions, invalid mappings, and negative or out-of-range device causal positions. Those rejection
checks are raw-Op evidence; `ninfer_r9700_state_qual` separately qualifies the canonical
all-layer transaction and publication contract.

`hip_device_qual` checks only the HIP device/arena/graph substrate. Canonical cache evidence is
split by ownership: `ninfer_r9700_state_qual` covers all-layer host/device append, compact
publication, invalid device-position frontier preservation, and publication-local poisoning;
`ninfer_r9700_kv_ram_qual` covers exact three-plane capture/restore and fingerprint
rejection, and `ninfer_r9700_full_attention_qual` covers the target attention consumer.

`ninfer_r9700_kv_cache_append_prefix_qual` covers the separate DFlash BF16 state mutation. It
checks bit-exact D128/Hkv8 device-count prefix writes, inert physical tails, wraparound in the live
2048-token DFlash2 cyclic state, lane selection, fragmented Full-cache page/table-row selection,
unchanged represented inputs, and dynamic count replay through a HIP Graph. It does not publish a
frontier and never enters the asymmetric FP8-K/INT4-V Text/MTP transaction contract.

`ninfer_r9700_mtp_round_qual` covers the exact MTP next-round state transition over every fixed
product shape K=1..5 and B=1..4. It compares alignment IDs, next proposal extents, AR positions,
MRoPE positions, and validity columns with an independent host integer oracle, including
budget/context exhaustion, zero/full acceptance, nontrivial row pitch, untouched pitch padding,
and immutable inputs. The public Op and implementation are native HIP members of the closed
gfx1201 archive; the retired wrapper, launcher, kernel, and duplicate legacy test are removed.

`ninfer_r9700_speculative_round_qual` covers the complete chain and packed-tree acceptance owner.
It exhausts K=1..8, W=2..16, and B=1..4, then checks mixed greedy/stochastic rows at both the
512-token single-workgroup boundary and the full 248320-token Qwen3.8 vocabulary. An independent
host oracle rebuilds the represented-BF16 top-20 distribution in FP64, applies penalties and the
round-local path overlay, uses the exact counter RNG, and evaluates selector-q Leviathan residuals
and SpecInfer child membership. Exact comparisons cover every accepted token, count, anchor,
length, accepted column, fold path, optional token-count publication, immutable input, and BF16
hidden selection bit. A captured full-vocabulary-style multi-stage chain replays at fixed addresses
with dynamic input state. The selected caller-owned partial/distribution pipeline uses no hidden
allocation or host readback. Retained pre-cap physical R9700 timings over 20 unprofiled ROCm events
measured 2.01 ms for chain V=248320/K=8/B=8 and 0.59 ms for the mixed greedy/stochastic product
tree V=248320/W=12/B=8; those B=8 measurements are historical, not supported-product evidence.
The active qualifier covers B=1..4. LLVM metadata reports 20 VGPR/1440 bytes LDS for the partial kernel and 17
VGPR/224 bytes LDS for finalization, with wave32 and no private scratch in either. The retired
wrapper, launcher, kernel, and duplicate legacy test are removed.

`ninfer_r9700_full_attention_qual` exercises the target-private Qwen3.8 attention leaf linked into
the closed HIP archive, rather than calling the raw A3 contract directly. On physical wave32
gfx1201, the real D256/Hq24/Hkv4 leaf passed an independent represented-BF16/FP8-K/INT4-V FP64
oracle for T=1..9, T=17, and T=128 over a fragmented 257-token page table using the
G16/token-K/feature-V/feature-scale route family. The compile-selected T1 WMMA path has maximum
absolute error 3.83e-4; the score-streaming control is 5.13e-8. The staged prefill T=9/17/128
target-leaf latencies were 0.213/0.453/2.513 ms over 50 ROCm-event iterations
after 20 warmups. A paired seven-sample physical diagnostic at T=4 measured median whole-leaf
latencies of 0.172 ms with the fixed all-row specialization and 0.171 ms with a device full-count;
this close single-shape observation is not a route-selection claim. It rejects null,
open, poisoned, empty, and beyond-mapping publication states before launch. Device row frontiers of
zero and frontier+1 produce NaN in exactly those rows while an adjacent valid row still matches the
FP64 oracle. LLVM metadata for this score-streaming G16 layout reports 23 VGPR for the existing
fixed-all-rows specialization and 24 VGPR for the device-count specialization; both use 52 bytes
LDS, no private scratch, and occupancy 16. This binds the streaming half of the selected attention
family to target state; the compile-isolated scorer remains the model-level private-FP8-Q control.

The streamed fused A3 candidate derives paged BF16-Q/FP8-K scores, maintains online FP32 softmax
state, and dequantizes each signed INT4 V using its stored FP16 group scale without a private
probability cast. A corrupt raw page-table ID returns NaN output rather than dereferencing beyond
the cache; production callers still use only allocation-verified page tables. At D256 its score
reduction is wave32-local followed by an eight-partial LDS merge, rather than a full-block tree;
the change is oracle-qualified but not yet a published speed selection.

The qualifier also measures the selected decode route that uses the owned raw FP8 WMMA QK kernel, one
reusable FP32 score workspace, parallel stable FP32 softmax, and the exact FP32-probability vector
PV route. It is compared directly with the represented-BF16-input FP64 attention oracle; its
private FP8-Q cast is never treated as an oracle input. Repeated 4K measurements favor this route
for T=1..2 and the score-streaming route for T>=3. A serialized short-context sweep selects WMMA
for T1 at context>=64 and T2 at context>=320; shorter contexts stream. At the target qualifier's
257-token frontier, selected T1 is 0.104 ms versus 0.191 ms for the separately compiled streaming
control, while T2 correctly remains streaming. The classifier, raw sweep, represented-input FP64
oracle, and production-boundary results are retained in
`profiles/bench/r9700-fp8-qk-wmma-crossover.json`; the PPL scorer reports whether the compile-
isolated WMMA profile is enabled.

A fresh post-softmax physical-gfx1201 audit used full oracles at 65-token T=8, 4K T=1/2/3, and
32K T=1. The WMMA maximum absolute errors were respectively 1.84e-3, 4.64e-5, 6.43e-5,
6.44e-5, and 3.27e-6. An unprofiled 50-event 4K sweep measured WMMA 0.866/1.709/2.597/3.432 ms
at T=1/2/3/4 versus score streaming 2.575/2.605/2.596/2.608 ms, making T=3 a tie rather than a
stable reason to widen the selected T<=2 width ceiling. The focused profiler trace is stored under
`profiles/r9700/rocm10-post-softmax-wmma-trace-4k`; its intercepted-queue durations are attribution
only. In that trace, QK WMMA averaged 26.1 us, parallel softmax 7.7 us, and exact vector PV
818.8 us, so softmax launch/reduction tuning cannot materially change this candidate. Fresh LLVM
metadata reports QK WMMA at 24 VGPR/0 LDS/0 scratch, softmax at 23 VGPR/76 bytes LDS/0 scratch,
and the selected G16 feature-V/feature-scale PV at 15 VGPR/0 LDS/0 scratch; all three report
occupancy 16. The QK image contains `v_wmma_f32_16x16x16_fp8_fp8`. Reproduce the attribution and
metadata with `make -C tools/r9700 profile-attention-trace` and
`make -C tools/r9700 kv-resources` while keeping profiler timings out of route selection.

`eager_op_qual` links the same native-HIP public Tensor/Weight eager Op boundary as
`ninfer_r9700_core`. It covers exact position and scalar state transitions, flat and structured
BF16 movement, FP32/BF16 conversion, residual and normalization routes, activations, dense and
integer-codec embedding, and deterministic argmax. The real D5120/T1..8 and padded-vocabulary
corpus is checked against independent exact-bit or FP64 oracles; a smaller structural corpus also
checks fused/unfused residual-RMSNorm identity, L2 diagnostic intermediates, lane/path movement,
Q6G64/W8G32 decode, and exact MTP FC-input pack plus Q/K/gate/V attention split at T=1..8. The
W8 route additionally owns complete `[248320,5120]` device code/scale extents, populates and
checks IDs through 248319 bit-exactly against signed-code times stored-FP16-scale, and rejects a
short payload and incorrect K padding. Its 1,288.36 MiB fixture measured 0.013 ms for the eight-row
gather in the qualifying ROCm-event observation; that timing describes the check, not an embedding
route selection. MTP cases also reject malformed logical shapes and partial byte-range aliases. Build and run the
standalone form with
`make -C tools/r9700 build/eager_op_qual`.

`rmsnorm_prefill_qual` retains the direct regression boundary for the production K5120 prefill
route selected at T>=128. One 256-thread workgroup assigns each of its eight wave32 waves
to one token; lane zero retains the incumbent feature-ascending FP32 `fmaf` chain while 128-bit
loads reduce fetch instruction overhead, and the whole wave writes the represented BF16 row.
The qualifier requires bit-exact incumbent parity for T1/2/7/8/9 through T2048, both unit-offset
modes, three eps values, and the independent FP64 criterion before timing. Run the CPU synthetic
static-check tests with `make -C tools/r9700 rmsnorm-prefill-static-test`; the real assembly gate
is `rmsnorm-prefill-static`. A fresh no-clobber physical report is produced only by explicitly
setting `RMSNORM_PREFILL_JSON` and invoking `rmsnorm-prefill-benchmark`. The retained admission
report passed the no-regression envelope from T128 upward and the 1.5x P2048 gate; other feature
widths and smaller row counts retain the incumbent kernel.

The later qualification-only residual-add to K5120 token8 RMSNorm fusion has been removed after
terminal physical rejection. Although its exact residual/output, complete independent-oracle,
dead-delta alias, rewrite, status, and 18-VGPR/zero-scratch static gates passed, every
T=1,024/2,048/4,096/8,192 timing cell lost. P2048 changed from `0.188480005` to
`0.554238975 ms` per pair (`2.94057178x`). The immutable report is
`profiles/bench/r9700-residual-rmsnorm-k5120-prefill-ab-20260904.json`, SHA-256
`6ed367d0438f0f7f8dbfe4bd01f7da126a6857c1953da8f1ed6e294eec20c134`. There is no live
candidate command or checker: production retains the separate residual-add plus selected RMSNorm
pair, and this result closes adjacent fusion variants.

`rmsnorm_k256_prefill_qual` retains the direct regression boundary for the production K256
query/key-normalization rows. One wave owns each row; lane zero retains the incumbent's exact
feature-ascending FP32 `fmaf` chain over all 256 features, and the wave writes the BF16 result.
Exact incumbent parity for both gain modes, three eps values, and row tails around wave/block
boundaries plus an independent FP64 formula precede balanced raw timing at
128/512/2048/8192/49152 rows. Admission requires at least 1.5x at 2048 rows and no measured
T>=128 regression. The immutable report
`profiles/bench/r9700-rmsnorm-k256-token8-ab-20260904.json` passed every gate; at 2048 rows the
median fell from 0.093921 to 0.041240 ms (2.2774x), and the real 49152-row query extent improved
from 1.101083 to 0.095321 ms (11.5513x). Production selects token8 for K256/T>=128 and retains the
incumbent below that boundary and for all other widths. Run
`make -C tools/r9700 rmsnorm-k256-prefill-static-test` and
`make -C tools/r9700 rmsnorm-k256-prefill-static`; create a fresh auto-profile report with:

```bash
make -C tools/r9700 rmsnorm-k256-prefill-benchmark \
  RMSNORM_K256_PREFILL_JSON=../../profiles/bench/EXPLICIT-FRESH-rmsnorm-k256-token8.json
```

The canonical ordinary-decode RMSNorm route covers exactly K5120 and rows 1 through 4, the
complete supported concurrency domain. One 256-thread CTA owns each row:
every lane accumulates 20 represented BF16 values in FP32, wave32 shuffles reduce each wave, and
eight LDS partials complete the CTA reduction before BF16 publication. The numerical gate compares
the complete result directly with an independent CPU FP64 formula across ordinary, zero, and
mixed-magnitude inputs, both gain modes, three epsilon values, every selected row count, and two
Device Graph replays. K5120 rows 5 through 127 retain the generic route, K5120 rows at or above
128 retain the token8 prefill route, and every other feature width retains its existing specialized
or generic fallback. Check the host selection boundary and gfx1201 resources without GPU execution
with:

```bash
make -C tools/r9700 rmsnorm-decode-production-routing-test \
  rmsnorm-decode-production-static-test rmsnorm-decode-production-static
```

The immutable operator report is
`profiles/bench/r9700-rmsnorm-k5120-rows4-qualification-8192i-20260906.json`, SHA-256
`e58b2e56980fb083548f1c357c26fc98a8a5bae27b1723ccf712451eaf4b8103`. All four rows passed; the
minimum robust ordinary-round saving lower bound was `11.7782198046 ms`.

The four-pair source-matched C1/P8192+G256 whole gate reduced median decode from
`12.521242570 s` (`20.44525522 tok/s`) to `9.467485694 s` (`27.03991411 tok/s`). Its robust
candidate/control upper ratio was `0.7598847464`, robust saving lower was
`11.72510638 ms/token`, and prefill upper ratio was `1.0029548902`; every generated-token vector
was exact. The immutable whole report is
`profiles/bench/r9700-rmsnorm-rows4-whole-p8192-g256-full-20260906.json`, SHA-256
`3f5c7a678f29b09537b46ebf7692e6f8d9b422e08f9ffe656367815932b2d0f6`.
The fresh selector-free production smoke measured `27.05729956 tok/s` and retained all 257
generated token IDs. Its report is
`profiles/bench/r9700-rmsnorm-production-final-p8192-g256-c1-20260906.json`, SHA-256
`b05db0068a4f1c73ce9c2092443b42f9f48b0fdb80ff8b3335db60cd5bdca74b`; the executable SHA-256 is
`a7c9303bd213fa3dbdb29ca0cee73addef1de8ab6a6e6b231509e25239776425`. The extracted loaded
gfx1201 object SHA-256 is `c3dcad45559a112f42f07b1d7e87fbd1d494d1d024083efc0498500ee678cd72`;
the selected kernel uses 17 VGPR, 32 bytes LDS, wave32, occupancy 16, and zero scratch/spills.

A direct GPU regression of the canonical route against its independent FP64 oracle, including
output guards and Device Graph replay, is available with:

```bash
make -C tools/r9700 rmsnorm-decode-production-regression
```

`gated_rmsnorm_prefill_qual` retains the direct regression boundary for the production K6144
token8 GDN output normalization selected at T>=64, the lowest physically measured extent. Each
wave owns one token, while lane zero retains the incumbent's exact
feature-ascending FP32 `fmaf` chain; vector loads change only instruction formation, not reduction
association. The qualifier requires BF16-bit-exact incumbent parity at token tails around eight,
an independent FP64 complete-formula criterion, the incumbent status and supported input/output
alias boundary, and then balanced
seven-sample timing at T64/128/512/2048/8192. Admission requires at least 1.5x at P2048 and no
median regression at any T>=128. The retained pre-promotion report
`profiles/bench/r9700-gated-rmsnorm-k6144-token8-ab-20260904-r3.json` passed those gates and also
measured a 1.7874x win at T64; P2048 improved from 1.150442 to 0.155001 ms (7.4222x). K6144/T>=64
therefore selects token8, while smaller rows and other feature widths retain the incumbent. Run
the host static fixtures with
`make -C tools/r9700 gated-rmsnorm-prefill-static-test`, inspect compiled assembly with
`make -C tools/r9700 gated-rmsnorm-prefill-static`, and create a fresh report only with:

```bash
make -C tools/r9700 gated-rmsnorm-prefill-benchmark \
  GATED_RMSNORM_PREFILL_JSON=../../profiles/bench/EXPLICIT-FRESH-gated-rmsnorm-token8.json
```

The production K128 ordinary-GDN normalization assigns one independently normalized row to each
wave and exactly eight rows to a 256-thread CTA at the admitted flattened extents `48*T` for
T=1024/2048/4096/8192. The immutable pre-promotion report
`profiles/bench/r9700-gated-rmsnorm-k128-rows8-ab-20260904.json` (SHA-256
`bdada371d626a6f4398ac350f5aaebdf3318d1743fb507e32f626b3c062e2e5c`) passed the independent
FP64 complete formula, incumbent BF16-bit parity, poison rewrite, alias, and malformed-input gates;
its P2048 median fell from 1.16636395 to 0.143720999 ms (ratio 0.123221397), and every qualified
extent won. The one-shot qualification executable was removed after promotion. Inspect the
retained production symbol with `make -C tools/r9700 gated-rmsnorm-k128-production-static` and run
its CPU parser fixtures with `make -C tools/r9700 gated-rmsnorm-k128-production-static-test`.

The P2048 ordinary-GDN output gated-RMSNorm-to-A8G64 fusion is terminally rejected. It passed
complete correctness and emitted-resource gates, but its `9.421392202 ms` aggregate preparation
missed the `5.699527 ms` ceiling and its `3.050880432 ms` complete 48-call saving missed the
required `5 ms`. The immutable report is
`profiles/bench/r9700-gdn-output-gated-rmsnorm-a8-fusion-p2048-ab-20260904.json`, SHA-256
`8396c5e76b4166405acab70abce11f7f7137e264a1eb61eec3b25d0a098ba962`. There is no live
qualification command or checker; production retains the separate normalization, A8 preparation,
and Q4 output matrix.

The production Text-MLP fusion is selected only for row-scaled-E4M3 gate/up plus Q4G64 down at
T=2,048 in a main Text layer. It preserves the explicit BF16 rounding boundary between FP32
SiLU-multiply and the existing signed-A8G64 codec, then invokes the unchanged production M64N128
Q4 matrix. All other profiles and widths use the ordinary split boundary. The one-shot timing
executable was removed after direct and matched-whole admission; its immutable report remains at
`profiles/bench/r9700-fused-silu-a8q4-down-p2048-ab-20260904.json`. To inspect the actual
production kernel, compile `src/ops/r9700/linear/r9700_linear.hip` for gfx1201 to assembly and run:

```bash
python3 tools/r9700/check_fused_silu_a8q4_static.py \
  --assembly tools/r9700/build/fused_silu_a8q4_down.s \
  --source src/ops/r9700/linear/r9700_linear.hip \
  --output tools/r9700/build/fused_silu_a8q4_down_static.json
```

The checker binds the exact production symbol and requires at most 24 VGPR, zero private/LDS,
four BF16 input-load sites, one FP16 scale-store site, two packed-code store sites, the native
exponential sequence, and no BF16 intermediate or scratch/flat-memory traffic.
The retained production result is
`profiles/bench/r9700-fused-silu-a8q4-down-static-20260904.json`, SHA-256
`aea9061d6bb27fb9e0fec9508b242816cf5cf69118fe1f36b5616a3fcdbf5336`.

The bounded P2048 gate/up algorithm qualification is closed. All eight results returned by the
zero-workspace production descriptor were correct, but rank 1's complete median was
4.186669827 ms versus rank 0's 4.189990520 ms, projecting only 0.212524414 ms saving across 64
calls; its matrix route was slightly slower, and ranks 2 through 7 were slower. Production keeps
rank 0 fingerprint `e0e00100000000000000000000000000` and zero workspace. The one-shot harness
was removed; the immutable terminal report is
`profiles/bench/r9700-fp8-gate-up-algorithm-p2048-ab-20260904.json`, SHA-256
`5dd95d8f01d911e34e437175e028ed973ce025361ba61e9a269239f97c32de31`. This result does not
authorize a broader catalog, workspace, or shape sweep.

`silu_mul_split_prefill_qual` retains the generic-route regression boundary for the production
two-dimensional Text MLP split view: gate/up each have logical shape `[17408,T]`, element strides
`[1,34816,34816*T,34816*T]`, and the
output is contiguous. Grid X owns feature tiles and grid Y owns tokens, eliminating the generic
strided kernel's per-element coordinate division while preserving its FP32 exponential,
multiplication, and BF16 result. Exact incumbent parity and an independent FP64 formula cover
T1/2/3/7/8/9/17/128 before balanced raw timing at T128/512/2048/4096. The immutable physical
report `profiles/bench/r9700-silu-mul-split17408-2d-ab-20260904.json` passed exact/oracle checks
and every timing row; P2048 improved from 1.009159 to 0.374680 ms (2.6934x). Production therefore
selects the 2D route only for that exact layout at T>=128; smaller rows and all other layouts keep
the generic strided route. Run the host checker fixtures with
`make -C tools/r9700 silu-mul-split-prefill-static-test`, inspect actual gfx1201 assembly with
`make -C tools/r9700 silu-mul-split-prefill-static`, and create a report only under `auto` with:

```bash
make -C tools/r9700 silu-mul-split-prefill-benchmark \
  SILU_MUL_SPLIT_PREFILL_JSON=../../profiles/bench/EXPLICIT-FRESH-silu-mul-split17408-2d.json
```

`linear_op_qual` owns the public Tensor/Weight linear boundary and its raw matrix kernels. It uses
the real Qwen3.8 full-attention query/key projection shape `[7168,5120]`, decode widths `T=1..8`,
and prefill widths through T128. Its BF16-weight route and candidate W8G32-F16-scale route both
consume BF16 activations, accumulate FP32, and round once to BF16 output. W8 consumes the
provisional converter's K128-padded code and FP16-scale planes as independent direct views, so a
target row slice requires no repacking. Synthetic ties, signed codes, K padding, exact plane
extents, overlap/malformed rejection, a direct row view, arbitrary prefill tails T9/15/17, and the
real-shape executions compare directly against independent host FP64 formulas over represented
BF16 inputs and decoded W8 bytes. Sparse represented-weight fixtures also exercise the complete
`[248320,5120]` output head and `[131072,5120]` draft head at T=1/3 without allocating host weight
matrices. Every output bit matches the independent FP64 formula; short code/scale planes and wrong
padding reject. The full output head composes with valid-vocabulary argmax and selects the exact
winner at a one-BF16-ULP margin. Fixture peaks were 1,289.73/680.78 MiB; single-event Linear
observations were 5.973/17.852 ms for output T=1/3 and 3.153/9.428 ms for draft T=1/3. These event
timings are bring-up data; they do not select a production recipe. Build and run it with
`make -C tools/r9700 build/linear_op_qual`; inspect emitted arithmetic ISA with
`make -C tools/r9700 linear-isa`.

On the ROCm 10 baseline, the fixed raw-Op decode dispatch is intentionally narrow: at
`[7168,5120]`, baseline blocks win BF16 `T=1..3` and W8G32 `T=1..4`; the wave32/LDS route wins
BF16 `T=4..8` and W8G32 `T=5..8`. The cooperative kernels use 1 KiB LDS, no private scratch, and
15--16 VGPRs; baseline kernels use the same 1 KiB reduction LDS, no scratch, and 8 VGPRs.

The prefill contract accepts every `T>=9`, including incomplete 16-token tiles; the fixed
`T=16/32/64/128` values are the real-shape timing points, not an API restriction. Decode selection
is unchanged. BF16 uses the native wave32 `16x16x16 BF16 x BF16 -> FP32` gfx1201 WMMA route; the literal raw
fragment map is the asymmetric map independently qualified by `r9700_qual`, and it preserves
represented BF16 operands, FP32 accumulation, and one BF16 output rounding. At the same real
shape, a full FP64 oracle at T16 and spread direct-FP64 samples at T32/64/128 report zero observed
BF16 error for the decode-style baseline, 16-wave LDS candidate, WMMA candidate, and fixed route.
The W8G32 route remains a scale-exact FP32 FMA path: no WMMA operand form preserves its signed
codes plus per-G32 FP16 scales, so a 16-wave, 1 KiB-LDS reuse route is selected after its matching
oracle gate. A 50-event sweep records BF16 WMMA 0.203/0.267/0.698/2.418 ms and W8G32 wave16
1.119/2.225/4.371/8.751 ms at T16/32/64/128. Inspect opcode and resource metadata with
`make -C tools/r9700 linear-isa` and `make -C tools/r9700 linear-resources`; WMMA is 29 VGPR,
0 LDS, 0 scratch, while the 16-wave candidates retain 1 KiB LDS and no scratch. These figures are
qualifier metadata, not a complete-model or artifact-recipe decision.

`w8a8_wmma_linear_qual` evaluates the separately compiled lossy-activation profile through the
production Linear Tensor/WorkspaceArena ABI. It quantizes each represented BF16 token's K
dimension to signed A8 with
one represented FP16 scale per G32, using caller-owned code/scale/status workspace. This aligns
exactly with each stored W8G32 scale: two native signed `16x16x16 INT8 x INT8 -> INT32` gfx1201
WMMA operations form one G32 dot, which is multiplied by the activation and weight scales and
accumulated in FP32 before one BF16 output rounding. The activation codec matches an independent
host RNE implementation byte for byte over the complete T128 real-shape input. The complete
represented formula passes spread FP64 checks at `[7168,5120]` for T1--8/16/32/64/128 with zero
observed BF16 error. LLVM emits native `v_wmma_i32_16x16x16_iu8`; the quantizer is 10 VGPR and the
matrix kernel 83 VGPR, both with zero LDS/scratch and reported occupancy 16.

An interleaved five-round ROCm 10 sweep covers all 13 unique W8 shapes and all 256 occurrences in
the mixed artifact. The adaptive evaluator selects A8 from T3 for `[7168,5120]`, `[12288,5120]`,
`[14336,5120]`, and `[5120,17408]`; T4 for `[4608,4608]`, `[5120,4608]`, `[248320,5120]`,
`[5120,6144]`, and `[5120,10240]`; T32 for `[34816,5120]`; and a conservative T64 for the three
1152-row Vision shapes because their T32 margins were only 1.07--1.23x. The main
`[7168,5120]` route loses at T1/T2, wins 1.32x at T3 and 1.53--1.75x at T4--8, and wins
3.37/5.55/4.28/3.14x at T16/32/64/128. Unknown shapes remain exact. Synthetic output error is
material, so this result admits the adaptive profile to matched real-model quality gates but does
not select it. `make -C tools/r9700 w8a8-wmma` regenerates the complete raw evidence;
`w8a8-wmma-tensor` checks the production boundary without repeating the sweep. ISA/resources use
`w8a8-wmma-isa` and `w8a8-wmma-resources`.

`gdn_op_qual` owns the native gfx1201 Gated DeltaNet semantic kernels. Its feature-fastest
convolution weight/history indexing matches Tensor `[channels,4]` and `[channels,3]` layout rather
than a private channel-major reinterpretation. The public `causal_conv1d_silu` Tensor boundary is
qualified at the real C=10240 geometry for decode T=1..8 and prefill
T=17/63/64/65/128/1024/2048, including both sides of the production crossover, with exact in-place
history and disjoint immutable-input history. Partial aliases and malformed shapes are rejected.
The same physical oracle covers control/recurrent execution plus the fixed Qwen3.8
projection-convolution geometry C10240 with mixed-width B=3 snapshot publication and tree-parent
replay records. It checks exact BF16 projection records, output-gate copies, invalid-tail zeros,
snapshot slots, immutable replay checkpoints, and FP64 convolution/recurrent outputs. Build and run with
`make -C tools/r9700 build/gdn_op_qual` and `tools/r9700/build/gdn_op_qual`.

The same owner selects its two-dimensional causal-convolution prefill route with token tile 4 at
the conservative stable crossover T>=64 and preserves the serial incumbent below it. Both routes
retain the incumbent FP32 FMA order and publish the exact final three represented input rows;
short T1/T2 histories use the incumbent, while tiled T>=3 state publication uses a same-stream
device copy. The physical tile-4 candidate loses at T16 and only marginally wins T32, so neither
point is selected. It wins every selected measured extent: candidate/incumbent medians are
0.02072/0.02680 ms at T64, 0.02472/0.04248 at T128, 0.03268/0.07220 at T256,
0.07836/0.249079 at T1024, 0.16376/0.659199 at T2048, 0.308919/1.29856 at T4096, and
0.60952/2.57596 at T8192. `make -C tools/r9700 gdn-prefill-static` requires the selected and
retained direct-regression tile symbols to be wave32/WGP kernels with at most 240 VGPR, occupancy
at least six, and zero LDS/private/scratch; the selected tile-4 kernel has 21 VGPR, occupancy 16,
and zero LDS/private/scratch. `make -C tools/r9700 gdn-prefill-benchmark` keeps separate direct
incumbent and tile boundaries, checks bit-exact output/state parity through P2048, then prints the
nine-sample median sweep.

The separate fixed-P2048 projection/convolution direct-scatter experiment passed its direct gate
at 0.288159013 versus 0.559597015 ms per layer, but its isolated whole run after the accepted MLP
fusion improved only from 1.178537057 to 1.168570885 seconds: 9.966172 ms and 1.00852852x, below
the fixed 12 ms and 1.01x admission gates. The terminal whole report is
`profiles/bench/r9700-gdn-prefill-projection-conv-direct-scatter-production-p2048-c1-20260904.json`
(SHA-256 `8ceaa1d948e0a76d29765a4255dd806a2bc3ffeba4b5ec009fc39f8b893e7c2f`). Production retains the
ordinary projection-copy, causal-convolution, and extract sequence; the rejected candidate's
API, kernels, executable, Make targets, checker, validator, and tests no longer exist.

`gdn_recurrence_qual` owns the public Tensor-level Qwen3.8 Gated DeltaNet recurrence boundary at
Hq16/Hv48 and K=V=128. Its independent represented-BF16/FP64 oracle checks ordinary distinct-state
publication through normalized T=128 and non-normalized T=64, masked snapshot slots and exact zero
tails, immutable sequential replay records, and parent-indexed tree replay records. Valid K, V, g,
and beta records must be bit-exact and invalid suffixes remain untouched. Tree state is caller-arena
storage sized by the public capacity query; ordinary and snapshot paths allocate no hidden
workspace.

The ordinary-prefill speed decision uses the complete public boundary. At T=128 it crosses a
minimum 10,534,912 represented bytes and performs 704,643,072 useful state-recurrence FLOPs before
the comparatively small normalization/control terms. The bring-up route used one 1024-thread CTA
per value head and a workgroup reduction for each Q/K norm; its three-run median latencies at
T=16/32/64/128 were 0.042276/0.074826/0.138775/0.265752 ms. The selected route splits every value
head into four independent 32-row, 256-thread CTAs, giving the 64-CU R9700 enough blocks to hide the
serial recurrence latency. One wave loads all 128 Q/K elements, performs the FP32 norm reduction,
and publishes the two scales; block-wide synchronization falls from 19 to 3 points per token.
Uniform alpha/beta and per-row V values are explicitly broadcast rather than recomputed or
reloaded by every lane. A physical row-tile sweep rejected two and eight tiles; at the real T=128
point their medians were 0.196199 and 0.193988 ms, versus 0.188758 ms for four tiles. Four tiles also
won at T=2048 and T=4096, so production has one route rather than an unsupported crossover. The
seven-run selected medians at T=16/32/64/128 are 0.033244/0.054612/0.099263/0.188758 ms, reductions
of 21.37/27.02/28.47/28.97 percent. The same public timing fixture records T=1 through T=4096.

Build and run with `make -C tools/r9700 build/gdn_recurrence_qual`. Compiler metadata reports the
selected normalized ordinary kernel at 56 VGPR, 1040 bytes LDS, zero private scratch, and occupancy
16; snapshot/tree remain at 58/66 VGPR and at most 1536 bytes LDS. Reproduce with
`make -C tools/r9700 gdn-recurrence-resources`. The oracle passes with maximum BF16-output absolute
error 7.63e-6, relative L2 error 0.001798, and FP32-state absolute error 8.42e-9. Unprofiled event
timing was decisive, so no unavailable or intercepting PMU counter was used for selection.

`gdn_affine_chunk_qual` retains the rejected three-stage FP32 affine-chunk experiment for the
ordinary normalized recurrence. It builds one dense transform and addition per value-head chunk,
propagates the small sequence of chunk boundaries, overwrites each addition with its exact input
boundary, and then replays chunks independently to publish every BF16 output plus the final FP32
state. The C32/C64/C128 sweep keeps production dispatch unchanged; C64 uses about 192 MiB of
caller-owned workspace at P2048. The independent sequential FP64 oracle covers complete output and
state publication through P2048. Its retained no-clobber report at
`profiles/bench/r9700-gdn-affine-chunk-rejected-20260904-r2.json` has SHA-256
`f574df9e6e1207b8177e527289a6bc804fb978960415b366405b5bf7ad5fb7fe`: C64 measured
5.622350 ms at P2048 versus 2.892155 ms for the incumbent, only 0.514403x as fast. It is therefore
closed as rejected and remains disconnected from production. `make -C tools/r9700
gdn-affine-chunk-static-test` checks the
fail-closed assembly parser, while `make -C tools/r9700 gdn-affine-chunk-static` requires all three
wave32/WGP stages to use 256-thread workgroups, zero private scratch, no matrix opcodes, at most 240
VGPR and 43,520 bytes LDS, and occupancy of at least six. A physical run is intentionally separate:
`make -C tools/r9700 gdn-affine-chunk-benchmark GDN_AFFINE_CHUNK_JSON=PATH` writes a fresh report only
once, with all nine raw samples and medians for every row. It records an explicit admitted or
rejected disposition and returns nonzero unless C64 is at most 1.247 ms at P2048 and at least 2.25
times faster than the incumbent, the bounded gate for saving at least 75 ms over the 48 model calls.

`gdn_grouped_heads_qual` is a qualification-only ordinary-prefill challenger. One 256-thread CTA
owns one 32-row tile for all three value heads sharing a normalized Q/K head, reducing the grid
from 192 to 64 CTAs and eliminating two thirds of duplicate Q/K loads, norm reductions, and
workgroup handoffs. Each value head retains the incumbent key-order FP32 dot, delta, state update,
query dot, BF16 output rounding, and final distinct-state publication. Production is unchanged.
The complete independent FP64 oracle covers T3/64/128. `make -C tools/r9700
gdn-grouped-heads-static` requires the exact symbol, 1,056 bytes LDS, at most 128 VGPR, occupancy
at least eight, four LDS-scoped barrier pairs, a 256-thread maximum, zero private/scratch, and no
matrix or global-cache invalidation instruction; current assembly uses 116 VGPR and occupancy 12.
The retained no-clobber physical report is
`profiles/bench/r9700-gdn-grouped-heads-ab-20260904.json` (SHA-256
`014c442f916466a8d460783bb9f18fc7d5ee4a69663827faf3536d4ab43f6bd4`). The challenger lost every
T128/512/2048/4096 row; at P2048 it measured 4.105878 ms versus the 2.962798 ms incumbent, only
0.721599x as fast. The saved Q/K work did not compensate for reduced cross-CTA latency hiding, so
the branch is rejected and production remains unchanged.

The rejected ordinary-recurrence LDS-scope experiment retained the incumbent normalized M32-row/
256-thread geometry, 192-CTA grid, key-order FP32 recurrence, and thread-private state while
narrowing synchronization for CTA-private Q/K, norm-scale, and control exchange. Its retained
direct report
`profiles/bench/r9700-gdn-ordinary-lds-scope-ab-20260904.json` (SHA-256
`4927845416d01aa6267d72df89342e91ab30ef0b9416bdf221d0116230acb444`) records a win at every row.
P2048 improved from 3.113205 to 2.761526 ms (`1.127350x`), but the matched whole-P2048 cutover
improved only from 1.202762421 to 1.193522512 seconds: 9.239909 ms and 1.007741x, below the fixed
10 ms admission gate. The immutable whole report is
`profiles/bench/r9700-gdn-ordinary-lds-production-p2048-c1-20260904.json` (SHA-256
`0af07a8340f310b7f89a800bc90864377fb76ac5e45c2e22f8dd74d580a6a69f`). Production therefore
uses the incumbent ordinary kernel, and the rejected challenger API, kernel, checker, and one-shot
benchmark surface have been removed.

`sampling_op_qual` owns the native HIP implementation of the public sampling contract. Large
vocabularies use one exact 512-token partial top-20 stage followed by a bounded wave32 final merge;
each partial selects within eight waves and merges their shortlists in wave zero, avoiding a
45-barrier full sorting network while retaining the same total 64-bit ordering key;
the transient key plane comes only from the caller's `WorkspaceArena`. The qualifier runs the
full padded Qwen vocabulary of 248320 tokens at every product B=1..4 and compares selected IDs plus the
entire token-count publication image with an independent host ordering/FP64 probability oracle.
Its cases combine greedy and positive-temperature rows, exact BF16 ties across partials,
presence/frequency penalties, top-k clamping, top-p/min-p filters, and counter keys separated by
seed, signed logical position, and purpose. A separate 257-token route checks the zero-workspace
path. On the retained pre-cap physical ROCm 10/gfx1201 baseline, the selected route measured
0.58 ms for 50 unprofiled B=8 events versus 0.67--0.74 ms for the exact sorting-network baseline;
this historical B=8 timing is not supported-product evidence. The route uses 620800 caller-owned
bytes. LLVM reports the partial/final kernels at 20/16 VGPR and 1440/304 bytes LDS
with no private scratch and occupancy 16; reproduce with `make -C tools/r9700 sampling-isa`.

`rope_op_qual` owns the native HIP implementation of the public RoPE contract. It checks the sole
Qwen3.8 Text D256/R64/Hq24/Hkv4 1-D and MRoPE routes, DFlash D128/R128/Hq32/Hkv8, and Vision
D72/R72/H16 directly against an independent FP64 formula from represented BF16 inputs. The corpus
covers split and combined wave32 launch shapes, padded token strides, exact preservation outside
the rotary span, the single-tensor form, and rejection of the retired 16/2 geometry, wrong theta,
and aliased storage. The selected short-token route assigns four heads per CTA; physical T=1..8
A/B measurements beat the one-CTA-per-token route while avoiding redundant per-head coefficient work.
The Text kernels use 18 VGPR, 256 bytes LDS, no scratch, and report occupancy 16. Build and run it
with `make -C tools/r9700 build/rope_op_qual`; reproduce compiler resource metadata with
`make -C tools/r9700 rope-resources`.

`dflash2_select_qual` owns the native HIP DFlash2 selector boundary over BF16 predecessor and
successor codebooks. It exercises the full 248320-row codebook geometry at T=2/B=2 and delegates
BF16, directly represented signed-W8G32/FP16-scale, and signed-Q4G64/FP16-scale hidden projections
to the canonical R9700 Linear Op. Its independent FP64 oracle evaluates the complete projection
and selector
formula, including represented BF16 boundaries, exact top-k and tie ordering, identity and mapped
draft-head IDs, chain probabilities, and every packed-tree output. It also rejects short,
overlapping, and unexpected W8 planes. The physical qualifier also executes direct Q4G64 hidden
projection through both chain and tree entry points with adaptive A8G64 workspace, mapped and
identity token domains, exact output IDs, caller-owned arena accounting, and malformed Q4 plane
rejection. Build it through `ninfer_r9700_dflash2_select_qual` in the R9700 CMake tree and run the
resulting executable directly on gfx1201.

`bidirectional_gqa_attention_qual` qualifies the separate DFlash Full BF16 paged-attention
boundary at D128/Hq32/Hkv8. Its independent FP64 oracle covers chain W=5 and tree W=12 with B=2,
mixed context and valid widths, non-monotonic page mappings, distinct table rows, query-key
visibility, and exact-zero inactive tails. The direct wave32 online-softmax kernel requires no
transient workspace; LLVM reports 31 VGPR, 42 SGPR, no LDS/private scratch or spills. Build and
run the `ninfer_r9700_bidirectional_gqa_qual` target in the R9700 CMake tree.

`grouped_dynamic_conv_qual` reaches the public DFlash2 grouped-convolution boundary at its exact
D5120/G320/kernel-2 geometry. It independently evaluates the complete projection and convolution
formula in FP64 from represented BF16 activations and represented BF16, exact signed W8G32, or
exact signed Q4G64 weights with their stored FP16 scales. The physical cases cover BF16 T1/B1,
W8 T2/B2, adaptive-A8 Q4 T2/B2, both preparation phases, the zero-padded block boundary, and a
separate T4/B2 finish. The
selected implementation delegates the large `[1280,5120]` projection directly to the qualified
native Linear Op, materializes only the caller-owned BF16 projection, and fuses phase-0 convolution
with phase-1 stash extraction in one gfx1201 kernel. ROCm-event measurements were 0.0615 ms for
BF16 T1/B1, 0.0939 ms for W8 T2/B2, and 0.0741 ms for Q4 T2/B2; maximum absolute oracle error was
1.59e-5. Build and run the
`ninfer_r9700_grouped_dynamic_conv_qual` target in the R9700 CMake tree. The prepare/finish kernels
use 13/10 VGPR, no LDS, and no private scratch.

`swa_qual` qualifies the DFlash2 symmetric cyclic SWA boundary at D128/Hq32/Hkv8 for both admitted
W2048 and W4096 windows. Its represented-BF16/FP64 oracle covers wraparound, the exact distance
`W-1` inclusion boundary, mixed lanes and context lengths, inactive physical tails, the
workspace-free short-envelope route, split-KV, and a saturated T16/B4 route. The selected wave32
kernel assigns the four grouped query heads to four waves in each KV-head workgroup. Long sparse
rounds partition context into a finite 32/16/8/4/2-way schedule based on `T*B`, then merge BF16
partial numerators with FP32 softmax statistics from caller-owned workspace; short envelopes and
T*B greater than 48 stream directly without workspace. An unprofiled split sweep rejected eight
splits (0.303/0.158 ms at W4096 T3/B2 and W2048 T1) and a fixed 32-way route (0.178/0.063 ms): the
selected finite schedule measured 0.173 ms at W4096 T3/B2, 0.063 ms at the real W2048 T1 point,
0.097 ms at W2048 T3/B2, and 0.053 ms for direct W2048 T16/B4. The original unsplit bring-up route
measured 2.11/1.17 ms at the first two points. Maximum absolute oracle error was 1.25e-3. Build and
run the `ninfer_r9700_swa_qual` target in the R9700 CMake tree. Direct/partial/reduce use 32/32/8
VGPR respectively, with no LDS or private scratch.
