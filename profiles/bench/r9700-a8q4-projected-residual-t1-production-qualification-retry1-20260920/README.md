# Projected residual T1 production qualification retry1

The original production package's attempt-1 failed before qualification because DRM
card numbering changed: card2 identified the integrated GPU, while the R9700 was
card1. That failed package and closure remain immutable diagnostic evidence. This
retry binds the corrected qualifier and rebuilt production executable; the power
check resolves the established R9700 PCI identity and still requires the HIP device
to match it. Numerical, static, timing, and admission criteria are unchanged.

This create-only gate follows the accepted standalone retry2 result. The candidate
calls the public `ninfer::ops::projected_residual_t1` entry through canonical
Tensor/Weight/DeviceSpan values and uses its public workspace-capacity query. It runs
the CMake target `build-r9700/src/ninfer_r9700_a8q4_projected_residual_t1_qual`, linked
against `ninfer_r9700_core`. The plan binds this executable, production archive,
CMake cache, and compile-command database; preflight and measurement do not rebuild
or substitute a separately linked qualifier. The ISA gate inspects device assembly
from production `r9700_linear.hip` and safely extracts the exact kernel's gfx1201
code object from the bound executable. Embedded disassembly and metadata must show
native IU4 dot8, wave32, bounded registers, and zero LDS/scratch/spills. The receipt
also retains the CMake link command naming the production archive.

For N5120 and K6144/K17408, it checks independently decoded represented A8/Q4 inputs
against an FP64 oracle (at most two BF16 steps), bit-exact residual output against
unfused projection plus residual add, nonfinite poisoning, canaries, malformed
alignment/alias rejection, and three captured-graph replays against eager output.
The public contract explicitly includes the represented A8 activation boundary and
the BF16 projection seam. Graph replay uses unchanged addresses and resets residual
state outside capture before each replay.

Timing is unprofiled HIP event timing with three disjoint weight allocations,
80 MiB scrub before each arm, and twelve balanced samples per arm. Each allocation
must remain within 1.01x; the summed 64-layer bound must save at least 0.2 ms/token.
Power must read `auto` before HIP initialization, after each shape's numerical and
graph checks before timing, and after timing. The pre-initialization sysfs device is
also checked against the HIP device PCI identity.

Run from the repository root with the selected Python 3.11 environment:

```bash
bash profiles/bench/r9700-a8q4-projected-residual-t1-production-qualification-retry1-20260920/commands.sh --preflight
bash profiles/bench/r9700-a8q4-projected-residual-t1-production-qualification-retry1-20260920/commands.sh --measure
```

Preflight compiles source assembly and checks the bound executable's serializer/ISA
in a temporary directory, initializes
no HIP device, and does not modify this package or prior evidence. Measurement
creates `attempt-1` exclusively and refuses reruns. Every process retains stdout,
stderr, argv, exit code, and timestamps. Normal failures and catchable signals seal
partial output in `closure.json` and `result.sha256`; SIGKILL or power loss cannot be
sealed by a running harness. A passing strict JSON report admits whole C1 A/B only.
It does not promote routing or establish whole-model speed.
