# Contributing to NInfer AMD

Bug reports, documentation fixes and measured improvements for Qwen3.8-27B on the Radeon AI PRO R9700 are welcome. This is a single-GPU, `gfx1201` project; other models and hardware are outside its current scope.

## Before you start

For a bug report, include the command, GPU, ROCm version, artifact identity and relevant startup or request logs. A short reproducer is more useful than a large log dump. Remove private prompts and credentials before sharing logs.

Discuss changes to the architecture, artifact format, numerical behavior or public HTTP contracts before implementing them. Small fixes can go straight to a pull request. Target `experimental`; the maintainer promotes qualified changes to `master` using `docs/maintainer/merging-to-master.md`.

Read the applicable rules in `AGENTS.md`. It defines module ownership, supported behavior and numerical requirements. Follow the existing implementation rather than adding compatibility paths or abstractions for unsupported configurations.

## Validate your change

Choose checks that exercise the behavior you changed:

- Documentation: check commands, references and `git diff --check`.
- CLI or HTTP behavior: build the affected application and run its option/schema tests; update the corresponding user guide.
- Artifact or converter changes: run the affected framing, codec and binding tests.
- HIP arithmetic: compare supported shapes against the independent mathematical oracle, then qualify the affected production route on the R9700.
- Performance: measure the scope you claim. Kernel timings alone do not establish an inference speedup. Report hardware, toolchain, workload and variation.

Commands and conventions are in `tests/README.md`, `bench/README.md` and `docs/maintainer/op-development.md`. The public CPU workflow checks formatting and Python benchmark/report contracts. Native builds and GPU/model qualification run separately on the maintainer's R9700.

## Open a pull request

Explain the problem, resulting behavior and any design choice a reviewer needs to understand. Include verification commands and summarized results, plus relevant checks you could not run. Keep unrelated changes separate.

You are responsible for understanding and reviewing every submitted line, including AI-assisted code. Disclose substantial AI assistance and the model used. Generated code needs the same correctness and performance evidence as other code.

Review time is limited. Changes outside the supported scope, without relevant validation, or requiring substantial redesign may be closed with an explanation.
