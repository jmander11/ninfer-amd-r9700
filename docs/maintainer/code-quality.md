# Code-quality gates

This is the authority for NInfer's static and dynamic quality tooling on the R9700: what each
gate catches, where its configuration and version pin live, how to run it, and how a finding may
be suppressed. Every tool is open source and version-pinned or part of the selected ROCm
toolchain, so a finding is a property of the code, not of the contributor's machine.

## Gates

| Gate | Catches | Configuration and pin | Run |
|---|---|---|---|
| Compiler diagnostics | shadowing, missing virtual destructors, hidden overloads, unannotated fallthrough, format mismatches, duplicated conditions/branches (GCC), member-initializer order | [`cmake/warnings.cmake`](../../cmake/warnings.cmake); GCC 13 for host C++, ROCm clang for HIP | every build |
| clang-tidy (ROCm LLVM) | use-after-move, unchecked `optional` access, dangling references, implicit widening of `int` index products in host code, slicing, exception escape from destructors and `noexcept`, empty catch, needless copies and no-op moves, path-sensitive analyzer findings | [`.clang-tidy`](../../.clang-tidy); `/opt/rocm/lib/llvm/bin/clang-tidy` | [`scripts/run-clang-tidy.py`](../../scripts/run-clang-tidy.py) |
| clang-format 22 | C++/HIP layout drift | [`.clang-format`](../../.clang-format); version in [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) | pre-commit |
| ruff (lint + format) | undefined names, late-binding closures, silent `zip` truncation, unchecked `subprocess`, naive datetimes, dead code, import order | [`ruff.toml`](../../ruff.toml); version in `.pre-commit-config.yaml` | pre-commit |
| shellcheck | unquoted expansions, undeclared locals leaking to global scope, dead variables | version in `.pre-commit-config.yaml` | pre-commit |
| typos | misspelled identifiers, comments, and documentation | [`_typos.toml`](../../_typos.toml); version in `.pre-commit-config.yaml` | pre-commit |
| File hygiene | merge markers, large files, private keys, broken JSON/TOML/YAML, CRLF, trailing whitespace, missing final newline, shebang/executable-bit mismatch | `.pre-commit-config.yaml` | pre-commit |
| Device checks (gpu_check) | device out-of-bounds access, LDS (shared-memory) races, reads of unwritten device memory | [`tools/r9700/gpu_check`](../../tools/r9700/gpu_check) | [`scripts/gpu-check.sh`](../../scripts/gpu-check.sh) |
| AddressSanitizer / UBSan | host heap/stack overflow, use-after-free, leaks, signed overflow, misaligned and invalid casts | `NINFER_SANITIZE` in `cmake/warnings.cmake` | separate tree, below |

## Device checks

NVIDIA's compute-sanitizer has no ROCm counterpart for gfx1201: ROCm clang drops
`-fsanitize=address` for every offload target without `xnack+`, and the R9700 has no XNACK.
`tools/r9700/gpu_check` provides the three checks the upstream gates use, through one
`LD_PRELOAD` shim (`NINFER_GPU_CHECK=memcheck|initcheck|racecheck`):

| Mode | compute-sanitizer analogue | Mechanism | Build | Finding |
|---|---|---|---|---|
| `memcheck` | memcheck (global memory) | every `hipMalloc` becomes a HIP virtual-memory mapping whose 256-byte-rounded end abuts an unmapped 4 KiB guard page, with a guard page before it | ordinary `build-r9700` | GPU memory fault naming the kernel and address |
| `initcheck` | initcheck | every `hipMalloc` is filled with `NINFER_GPU_CHECK_POISON` (default `0xff`: NaN in every float format, -1 in integers) | ordinary `build-r9700` | the test's oracle fails |
| `racecheck` | racecheck (shared memory) | the HIP compiler launcher instruments optimized gfx1201 bitcode: every LDS load/store (and flat access that resolves to LDS) checks a 64-bit shadow word per 4-byte LDS word, and every workgroup barrier advances the wave's barrier interval | `build-r9700-racecheck` (`hip_racecheck_launcher.py` as `CMAKE_HIP_COMPILER_LAUNCHER`) | exit status 86 with one report line per source site and hazard kind |

```bash
scripts/gpu-check.sh memcheck                                     # the gpucheck set
scripts/gpu-check.sh initcheck
scripts/gpu-check.sh racecheck
scripts/gpu-check.sh racecheck -- -R ninfer_r9700_sampling_qual   # one qualifier
```

The script builds the tools (`make -C tools/r9700/gpu_check`), builds the `gpucheck` CTest set in
its tree, and runs it under the shared GPU lock. Rebuilding the tools cleans the race-check tree,
because the launcher's instrumenter and device runtime are not Ninja dependencies.

Race-check semantics. A hazard is two different waves of one workgroup touching overlapping bytes
of one LDS word in the same barrier interval, at least one access a plain store
(read-after-write, write-after-read, write-after-write). Lanes of one wave execute in lockstep and
are not compared. LDS atomics are synchronization and are not checked. Accesses beyond the
launch's group segment are reported as out of bounds. Each workgroup instance clears its shadow
slot and draws a fresh tag, so stale shadow from an earlier launch never matches. Limits, all of
which can only hide a hazard, never invent one, except the multi-reader case noted last:

- the gfx12 split barrier is modelled at `s_barrier_wait`, so an access between a wave's
  `s_barrier_signal` and its wait is attributed to the interval before the barrier;
- ordering established without a workgroup barrier (release/acquire through LDS or global
  atomics) is not modelled, so code synchronized that way reports hazards that must be read
  against its protocol;
- per-word tracking keeps the most recent writer and the first reader plus a several-readers
  flag; a write after reads by two waves is reported even if the overlapping bytes were read only
  by the writing wave;
- more than `NINFER_RACECHECK_SLOTS` (default 2048) concurrently resident workgroups share
  shadow slots and lose their history (missed hazards only); a launch with more LDS than
  `NINFER_RACECHECK_LDS_BYTES` (default 65536) is reported once and its excess is not checked.

Every instrumented kernel gains one 4-byte LDS word (the workgroup nonce) and a barrier at entry,
so a kernel already at the 64 KiB LDS limit cannot launch in the race-check tree. Exit status 87
means the checker itself failed (shadow mapping or a HIP call), not that a hazard was found.

Memcheck catches overruns past the 256-byte-rounded end and underruns beyond the slack before it
at page granularity; it does not check LDS, and a GPU memory fault can leave the device needing a
few seconds to recover, so rerun the next test rather than chaining after a fault. Initcheck
catches unwritten device memory only through the oracle's verdict; it does not name the reading
kernel.

The `gpucheck` label is the set these gates run: every registered R9700 qualifier (31). On
2026-10-02 the set took about 6 minutes under memcheck or initcheck and 11 minutes under racecheck
(speculative round 407 s against 172 s plain), all passing after the fixes recorded in
[upstream-sync.md](upstream-sync.md). The race-check tree sets `NINFER_R9700_QUALIFIER_TIMEOUT` to
3600 s. A kernel without LDS (for example bidirectional DFlash attention) reports zero checked
workgroups.

## Running the static gates

```bash
python3.11 -m pip install pre-commit==4.6.2   # the hook runner, once per machine
git config core.hooksPath .githooks           # the commit hook, once per checkout
pre-commit run --all-files                    # every commit-stage hook over the whole tree
./scripts/run-clang-tidy.py --changed         # clang-tidy on lines changed vs. origin/HEAD
./scripts/run-clang-tidy.py                   # whole tree
./scripts/run-clang-tidy.py src/serve/foo.cpp # named files; a header selects its includers
```

The hook ([`.githooks/pre-commit`](../../.githooks/pre-commit)) runs the tools in
`.pre-commit-config.yaml` on the staged files, stages the formatters' and `ruff --fix` rewrites
into the commit, runs the checks once more, and fails only when a check still fails. A file that
also has unstaged edits is not re-staged; stage or stash those edits and commit again.

clang-tidy is not in the commit hook: it needs a configured `build-r9700` tree. The runner reads
that tree's `compile_commands.json` directly, because HIP translation units are already ROCm
clang commands; host C++ commands come from GCC and only gain `-Wno-unknown-warning-option`.
`--changed` analyzes every translation unit that is or includes a changed file and reports only
diagnostics on changed lines. The two widening checks apply to host translation units only:
device index arithmetic is 32-bit by design and bounded by each kernel's shape contract, which its
host launcher validates under those checks. Header changes are mapped to their includers through
Ninja's dependency log, so build the tree first.

Compiler diagnostics need no separate step; the policy applies only to NInfer targets, and
third-party sources keep their upstream flags.

Host sanitizers use a separate tree. HIP translation units instrument their host compilation only
(`-Xarch_host`), because gfx1201 device code cannot be instrumented:

```bash
cmake -S . -B build-r9700-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DNINFER_SANITIZE=address,undefined
cmake --build build-r9700-asan
ASAN_OPTIONS=protect_shadow_gap=0 ctest --test-dir build-r9700-asan -LE r9700
```

## Current state

The tools landed with the 2026-10-02 upstream sync, and on the same day the tree was brought clean
under every gate: one tree-wide clang-format/ruff format commit (listed in
[`.git-blame-ignore-revs`](../../.git-blame-ignore-revs); run
`git config blame.ignoreRevsFile .git-blame-ignore-revs` once per checkout), then the clang-tidy
and ruff findings fixed or suppressed at the line with their reasons. Every gate is a hard gate:
the commit hook is meant to be enabled in every checkout, and whole-tree
`./scripts/run-clang-tidy.py` must report 0 diagnostics, with `--changed` as the fast check while
editing.

| Gate | State |
|---|---|
| Compiler diagnostics | clean; warnings are errors (`NINFER_WARNINGS_AS_ERRORS=ON`) |
| clang-tidy | clean over the whole tree |
| pre-commit (clang-format 22, ruff lint and format, shellcheck, typos, file hygiene) | clean with `--all-files` |
| Device checks (`gpucheck`, 31 qualifiers) | clean under memcheck, initcheck, and racecheck |

Byte-pinned candidate qualifiers under `tools/r9700/` whose source text static checkers compare
are excluded from clang-format (the list is in `.pre-commit-config.yaml`); clang-tidy still
covers them.

## Suppressions

A finding is fixed unless the flagged code is correct and the reason can be stated. Suppress the
single instance, at the line, with the reason:

- clang-tidy: `// NOLINT(check-name): reason` or `// NOLINTNEXTLINE(check-name): reason`;
- GCC and clang: `#pragma GCC diagnostic push` / `ignored "-W..."` / `pop` around the smallest
  region, with a comment;
- ruff: `# noqa: CODE  reason`; typos: an entry in `_typos.toml` for real vocabulary.

Bare `NOLINT`, `# noqa` without a code, file-wide suppressions, and removing a check or warning to
make a change pass are not accepted. A check that is wrong for the whole codebase is removed in
`.clang-tidy`, `cmake/warnings.cmake`, or `ruff.toml` with the reason recorded there.

## Changing a tool or its version

Bump the pin, run the tool over the whole tree, and commit the mechanical result separately from
behavioral changes. clang-tidy follows the ROCm toolchain's LLVM; a check that appears with a new
LLVM release is admitted or disabled in `.clang-tidy` with its reason, as `bugprone-signed-bitwise`
(LLVM 23) is.
