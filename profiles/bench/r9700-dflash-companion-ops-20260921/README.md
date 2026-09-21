# DFlash represented-W8 public-Op qualification

This package qualifies the W8 companion's existing public Ops; it does not select a recipe,
run an inference-speed comparison, change a kernel, or admit production DFlash.

The fresh build is `build-r9700-dflash-companion-ops-20260921`, Release/gfx1201,
Q4-A8/W8-A8, G16. The changed operators do not depend on the Text KV group.
Compile only (no qualifier launch):

```
cmake --build build-r9700-dflash-companion-ops-20260921 --target ninfer_r9700_linear_op_qual ninfer_r9700_dflash2_select_qual ninfer_r9700_grouped_dynamic_conv_qual -j2
bash profiles/bench/r9700-dflash-companion-ops-20260921/commands.sh preflight
```

Only after independent review and serial GPU ownership:

```
bash profiles/bench/r9700-dflash-companion-ops-20260921/commands.sh run
```

Execution is create-only at `physical/`. Failed or partial logs are retained; there is no resume
or overwrite. Preflight is CPU-only and rejects stale builds. Run binds physical R9700 device 0,
requires auto power and idle/low residency between independent processes, and records the exact
executables, source/configuration inputs, commands, raw outputs and complete outcome counts.

Proposal and pending-context append use W=K+1 rows, so W5/W6 at C1..4 exercises T=5/10/15/20
and 6/12/18/24. Feature [5120,25600] and context-QKV [6144,5120] additionally use the selected
2048-row prefill extent. Output [5120,4096] and gate/up [34816,5120] remain BF16×W8 at these
widths; down [5120,17408] uses the already selected private A8 arithmetic. Linear uses represented
public BF16 input and independently decoded signed W8/FP16-scale weights in full-K FP64 dot
products. It samples nine output channels for every small-T row, and seven varied rows at T2048;
this is explicitly sampled oracle coverage, not exhaustive validation of every large output.
Every device output is checked finite. Fixed error bounds are .001+.0079×|oracle| for exact W8
and .01+.03×|oracle| for adaptive A8W8; failures do not authorize relaxing these bounds.

Grouped convolution uses W5/W6 × C1..4 with its complete logical FP64 projection/convolution
oracle. Selector uses K4/K5 × C1..4, the actual 131072-row optimized logit domain and unchanged
248320-row BF16 codebooks, including mapped token IDs above the optimized domain. Its full
logical projection and Markov scores remain FP64 in the oracle; private BF16 staging is not
copied into the reference. Discrete path/support and greedy proposal distributions compare
exactly. All three public Ops use caller-owned scratch and check eager execution plus two graph
replays against the independent oracle, poisoning scratch/output before replay.

Remaining after Op qualification: materialize each append recipe from the selected base,
qualify real artifact binding and whole-model graph/determinism/output behavior, then evaluate
K4/W5 and K5/W6 quality, acceptance, capacity, and speed through the selected-DFlash campaign.
