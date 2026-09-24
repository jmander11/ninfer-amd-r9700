# Offline C4/K4 owner review

Initial review before the paired-QK prototype. The later unused-query-load
finding, bounded admission and measurements are recorded in README.md; they
supersede the tentative attention decision below, not the overhead/FP8 bounds.

Source: `baseline-c4-k4-trace/raw/r2d2/1423243_results.db`, current gfx1201
compact-cap26 model, chunk2048, P4096/G128. Attribution only, not timing admission.

- 37 graph rounds: span3476.064ms, kernel union2885.561ms, uncovered590.503ms.
  Inter-round span113.985ms, uncovered31.476ms. The memory-copies table is empty
  despite requesting copy tracing; uncovered time is not proved hardware idle.
- Main graph4 has31 rounds. Of67,766 noninitial kernel gaps,64,250 are5–12us
  (mean7.44us). Profiled output142.577tok/s versus retained unprofiled~162tok/s
  demonstrates material tracing perturbation. Do not budget590ms as removable
  host overhead. Graph-launch API duration is submission, not execution.
- Clipped inter-round APIs: event synchronization84.773ms, memcpy23.176ms,
  graph launch2.408ms. Synchronization overlaps device work; these are not
  independent savings. Even deleting all31.476ms uncovered host time would
  save less than0.9% of3.591s. Pinned/coalesced status copies are not justified
  as a material standalone optimization by this trace.
- Graph4 FP8 quantization8.095ms, poison1.325ms, matmul114.943ms. Removing all
  FP8 preparation saves only0.261ms/round. Shared preparation between protected
  projections does not clear a material whole-round bound. Matmul itself is
  ~4.45% of kernel service; no identified algorithm defect yet.
- Graph4 attention: PV362.142ms (14.02% kernel service), QK152.399ms (5.90%),
  softmax16.840ms (0.65%). Prior rejected PV row-sharing/stage64/page-pointer
  mechanisms remain excluded. Successful stage32 is already installed.

A distinct QK possibility, not an admitted candidate: pack two causal query
rows' six heads per KV head into twelve of WMMA's sixteen M lanes. Current
mapping uses six lanes per query-row tile. W5 could reduce QK tiles5→3,
W6 6→3 while preserving each dot's K-order and represented FP8 operands.
At W5, even ideal40% QK reduction saves only1.97ms/round (~2% of traced span),
before extra indexing/tail cost. Reevaluate only after current projection
promotion changes the whole-round bound; do not implement from this estimate
alone. This is not the previously rejected PV row-sharing mechanism.
