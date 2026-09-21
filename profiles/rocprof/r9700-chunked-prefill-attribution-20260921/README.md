# Appended-chunk prefill attribution

This package prepares one C1, P8192, chunk1024, spec-none run of the current dense
G16 benchmark with the exact all-Q4 N16/K16 artifact and bench corpus. It uses one
repetition, zero warmups, and only the measured region. No trace has been executed
by package preparation.

The fresh unprofiled P8192/chunk1024 observation was 184.864 tok/s (44.3138 s),
versus roughly 1904 tok/s for P2048. P8192/chunk2048 subsequently measured
194.8000609 tok/s. These observations motivate attribution of appended-chunk
attention; they do not establish which kernel dominates. The retained ROCTX
markers and kernel durations will identify that owner and quantify its share.
Profiler-intercepted times are attribution evidence only, never speed admission.

```sh
bash profiles/rocprof/r9700-chunked-prefill-attribution-20260921/commands.sh preflight
bash profiles/rocprof/r9700-chunked-prefill-attribution-20260921/commands.sh run
```

Preflight is nonmutating and launches no subprocess or GPU work. It validates the
benchmark/corpus hashes and artifact/receipt identity against the frozen chunk
campaign `inputs.json`, checks profiler availability, and reads AMD R9700 PCI
identity and `auto` power at PCI 0000:13:00.0. It does not bind unrelated frozen
tools. The run stage additionally requires less than 1 GiB resident VRAM and zero
reported GPU activity, probes HIP device 0's PCI identity, then rechecks
auto power before creating its output namespace. The HIP probe itself can transiently
raise the activity counter, so idle is checked before that probe. Coordinate with the paused
matrix campaign: do not resume it until this trace finishes.

The create-only `run/` directory contains `launch.json`, `benchmark-report.json`,
`raw/chunked-prefill_results.db`, `execution.json`, and `closure.json`. rocprofv3
uses `--selected-regions -f rocpd --marker-trace --kernel-trace --memory-copy-trace`.
The closure reuses `tools/bench/analyze_whole_profile.py`, checks eight text chunks
and no MTP prefill ranges, and retains operator/stage attribution, top kernels,
active wall-time union, copy categories, and the analyzer's completeness flags.
Incomplete attribution remains visibly incomplete; the script does not infer an
owner absent from the trace.

Completed result: eight Text chunks, no MTP ranges, 43838.812 ms Text-prefill wall.
The 112 serial causal-attention calls total 39087.580 ms (89.16% of wall), versus
3237.931 ms for Q4 CTAs. The 0.294 ms unattributed kernel duration remains marked
incomplete; it does not change the dominant-owner conclusion. This supports the
bounded appended-chunk tiled-attention extension, not a performance-admission claim.
The retained run must not be repeated.

Existing output namespaces are never reused or overwritten. Failed profiler runs
retain partial evidence and cannot publish closure. If profiling succeeded but
analysis failed, `commands.sh analyze` retries only analysis while requiring an
absent closure and the exact successful execution receipt. No automatic cleanup,
model download, build, or power/environment modification occurs.

CPU checks:

```sh
/home/battlefront/.local/bin/python3.11 profiles/rocprof/r9700-chunked-prefill-attribution-20260921/test_trace.py
```
