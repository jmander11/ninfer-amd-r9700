# Sequential GDN exact-tree staging challenger

Public owner: `gated_delta_net_replay_record`, sequential record only. Input BF16
Q/K/V, FP32 controls and initial state; Hq16/Hv48/D128, C1..4, W4/5/6.
The existing independent FP64 recurrence and criteria remain unchanged. Raw replay
records must remain exact, input state immutable, invalid record suffix untouched,
invalid output suffix zero. Supplementary output parity uses unchanged snapshot
normalization, not a replacement numerical oracle.

Final dispatch: exact-tree at actual batch1/4, original tree at batch2/3 after
whole tests showed no middle-batch win. `gdn-batch-selected.json` qualifies all12
B1..4 W4/5/6 cells; both emitted bodies match their respective qualified controls.
The earlier six-cell results below describe the exact-tree candidate, not blanket
promotion across batch sizes. See package README for whole selection evidence.

Layer 0: final C4 record(false) owns 5.17% of kernel service; C3/C4 attributed
latencies are 52.449/78.182 us per call. These are not unprofiled admission timings.
Each call reads C*48*128*128*4 bytes of checkpoint state (3 MiB per request),
plus Q/K/V/controls, and emits outputs/raw records. It performs the actual recurrent
state-key and state-query dots and rank-one update; it is not a snapshot-copy kernel.
The challenger changes synchronization only, not those bytes or useful arithmetic.
The old normalization invokes approximately twenty CTA barriers per token; a
first-wave exact reduction replaces its shared-memory tree with shuffles and one
publishing barrier. A 20–30% Op gain would save 1.0–1.55% of kernel service, not
a promised whole-inference gain. Final unprofiled Op and isolated whole comparisons
are complete in `gdn-final.json` and the package README; cold C4 Op results are
mixed, while isolated C1/C4 whole tests improve with separated repetition ranges.

The first wave squares indices l,l+32,l+64,l+96 separately, combines
`(s0+s2)+(s1+s3)`, then reduces offsets16,8,4,2,1. Rounded intrinsics alone did not
prevent compiler contraction in the initial version. The final register-only
materialization barrier makes square results observable to the compiler before
tree additions; emitted ISA confirms separate squares/adds and the old association. The
existing wave-QK experiment has a different four-FMA tree and is not enabled.
Snapshot, ordinary, tree record, replay-fold and all public interfaces are unchanged.
Inspect the emitted tree, barriers, VGPR/SGPR/LDS/scratch before performance admission.

Build both screen binaries from the same source, serially; do not rebuild the core
until the baseline helper is linked against the saved baseline archive:

```bash
PYTHONPATH=. /home/battlefront/.local/bin/python3.11 profiles/bench/r9700-remaining-candidates-20260924/build_screen.py profiles/bench/r9700-remaining-candidates-20260924/gdn_screen.hip gdn-control --core profiles/bench/r9700-remaining-candidates-20260924/baseline-core.a
profiles/bench/r9700-remaining-candidates-20260924/gdn-control profiles/bench/r9700-remaining-candidates-20260924/gdn-baseline.json
# Primary performs the serialized candidate core rebuild, then:
PYTHONPATH=. /home/battlefront/.local/bin/python3.11 profiles/bench/r9700-remaining-candidates-20260924/build_screen.py profiles/bench/r9700-remaining-candidates-20260924/gdn_screen.hip gdn-final
profiles/bench/r9700-remaining-candidates-20260924/gdn-final profiles/bench/r9700-remaining-candidates-20260924/gdn-final.json
build-r9700/src/ninfer_r9700_gdn_replay_fold_qual
```

The current twelve screen cells (originally six C1/C4 cells) qualify ragged and
dense public outputs/records plus snapshot
state against FP64, exact snapshot output, graph replay, and W2/W16 public boundaries.
Warm and 80-MiB-scrub-cold timings are separate 16-sample graph/event measurements.
The existing replay-fold qualifier establishes downstream actual publication state;
the screen does not claim its snapshot check directly executes replay-fold.
Physical PCI identity and auto power are checked before initialization and after
qualification/timing. Whole-engine admission belongs to the primary agent.
These are retained command identities: use fresh output names for any new run.
`gdn-candidate.json` is historical contracted-prototype evidence, not final admission.
