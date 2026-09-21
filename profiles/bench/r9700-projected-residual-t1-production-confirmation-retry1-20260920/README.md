# Selector-free projected-residual production confirmation

Preparation binds an explicitly supplied existing production build and a fresh
linked public-Op qualification after selector removal. It never configures, builds, or runs
an inference/kernel workload. Preparation and `--preflight` both execute the bound helper's
read-only `--device-info` query to validate its supported interface and physical HIP identity.
They check auto power and low VRAM before HIP initialization, then drain and recheck.
Preflight writes no files; neither command repairs inputs.
`plan.json` and `attempt-1` are create-only. Review the prepared plan before `--measure`.

Run from the repository root with:

```bash
bash profiles/bench/r9700-projected-residual-t1-production-confirmation-retry1-20260920/commands.sh --prepare --build-dir /absolute/qualified/build --qualification-attempt /absolute/fresh/public-op/attempt-1
bash profiles/bench/r9700-projected-residual-t1-production-confirmation-retry1-20260920/commands.sh --preflight
bash profiles/bench/r9700-projected-residual-t1-production-confirmation-retry1-20260920/commands.sh --measure
```

The build must provide `ninfer_bench`, `ninfer_r9700_target_variant_projected_residual_qual`,
and `ninfer_r9700_gdn_qual`. Its cache and core archive must be the same files bound by the
fresh passing public-Op qualification. The retired selector must be absent from every
CMake cache key and every benchmark config field.
The fresh qualification must report `qualified_for_selector_free_production_confirmation`,
decision `prepare_selector_free_whole_c1_confirmation`, and
`production_routing_authorized: false`; the historical whole-A/B status is not accepted.

Three ordinary Device Graph C1 P8192/G256 runs use the exact Q4 artifact, chunk 4096,
fixed FP8-K/INT4-V cache, no speculative decoding, one warmup and one measured repetition.
All 257 public tokens must equal the retained three-route C1 authority. Every run must
take no more than 1.01 times the admitted candidate median, and the confirmation median
must remain at least 1% faster than the retained control median. This is a conservative
production reproduction gate, not a new randomized performance comparison.

Each run binds the physical R9700 PCI function, HIP ordinal/architecture/wavefront,
runtime/driver versions, `auto` power, low VRAM before initialization, and bounded VRAM
drain after each child. Receipts preserve errors and partial results on failure. No
sudo, power writes, reset, downloads, or automatic retries occur.

The audited whole A/B attempt and its direct production qualification remain historical
evidence. Their retained manifests and plan identities are verified without requiring
superseded source files or old binaries to match the promoted implementation. The fresh
public-Op qualification and current build do require current bound inputs to match.
The retained whole-A/B campaign supplies read-only helpers for manifests, device receipts,
token geometry, and telemetry; its script identity is bound into this plan.

This retry retains the original confirmation's failed plan, attempt manifest, and failure.
That attempt stopped before inference because an old helper binary lacked `--device-info`.
The owner explicitly rebuilt the helper. Both preparation and preflight now exercise the
interface before creating any measurement attempt. The numerical qualifier, benchmark,
core archive, model artifact, and performance requirements are unchanged.
