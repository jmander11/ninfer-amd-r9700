# R9700 final artifact cutover audit

This is a decision-gated cutover sequence, not a second live selection ledger. Do not edit a
product identity, converter identity, admitted profile, default, or artifact filename until gate 0
below is complete. The retained schema-v7 base Pareto record selects only the base artifact/cache/
execution tuple; it carries no downstream-readiness or MTP-head status and is not by itself cutover
authority. The eventual schema-v3 DFlash selection record is likewise a companion decision, not
permission to promote the base early. DFlash2 is the preferred and required speculative production
path. MTP is retained as an already-working backend and remains in regression/parity coverage, but
no new MTP feature, head-precision experiment, trace, or performance optimization is a cutover
prerequisite.
The cutover must replace the evaluation product surface in one change; it must not add final
identity aliases beside the old identities.

Use the planned pair corresponding to the selected recipe throughout the cutover:

```text
all-Q4 base weights_id       = r9700-q4g64-n16k16
all-Q4 DFlash weights_id     = r9700-q4g64-n16k16-dflash2-q4g64
all-Q4 base filename         = qwen3.8-27b-r9700-q4g64-n16k16.ninfer
all-Q4 DFlash filename       = qwen3.8-27b-r9700-q4g64-n16k16-dflash2-q4g64.ninfer
mixed base weights_id        = r9700-q4-w8-mse-n16k16
mixed DFlash weights_id      = r9700-q4-w8-mse-n16k16-dflash2-q4g64
mixed base filename          = qwen3.8-27b-r9700-q4-w8-mse-n16k16.ninfer
mixed DFlash filename        = qwen3.8-27b-r9700-q4-w8-mse-n16k16-dflash2-q4g64.ninfer
four-role base weights_id    = r9700-q4g64-f8e4m3-four-role-n16k16
four-role DFlash weights_id  = r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4g64
four-role base filename      = qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16.ninfer
four-role DFlash filename    = qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4g64.ninfer
```

These names are reserved by this plan, not admitted product identities or existing final
artifacts. They become authoritative only after the retained decisions select the recipe and fresh
source conversion materializes and validates both files. The base identity denotes the selected
all-Q4G64/A8G64, source-MSE Q4G64/W8G32+A8G64, or authority-selected four-role
rowwise-FP8/all-other-Q4G64 recipe; the DFlash identity denotes that same base plus the selected
Q4G64/BF16-codebook companion. Cache G16/G32 is compile-time product state, not artifact metadata,
so it must not be encoded in either `weights_id`.

## Required sequence

0. Admit the evaluation winner before any cutover mutation. The following distinct records must
   exist, reopen successfully, and bind the same schema-v7 winner artifact, cache group, selected
   chunk, and selected execution profile/executable bytes, except that the low-context floor is
   deliberately measured on that winner recipe/group/chunk's separately bound dense-control
   profile and executable when XAttention wins:

   The BF16 source acquisition prerequisite is already satisfied by
   `profiles/ppl/r9700-bf16-source-checkpoint-preflight-20260905.json` (SHA-256
   `4c605982c84fbfc8803fea24ccdd0230277f4f20c8547a38098f6f440bf99fa9`): the target-owned
   protocol accepted the exact config/index and all 18 named nonempty shards. This makes the
   following conversion and quality gates runnable; it does not satisfy them.

   - `profiles/bench/prefill-chunk-selection-20260905.json`, produced from the complete twelve-way
     8K screen and 32K finalist confirmation;
   - `profiles/ppl/terminal-quality-recovery-20260905/quality-authorities.json`, containing all six
     native schema-v6 quality authorities. Each must bind its candidate to the deterministic
     complete 18-shard BF16-source reference at the selected chunk; retained historical scorer
     output is not a substitute;
   - `profiles/ppl/post-terminal-selected-exact-token-20260905/admission.json`, prepared from the
     schema-v7 winner by `post-terminal-exact-token-prepare-20260905`. It must reopen the selected
     artifact/scorer, candidate-local quality authority, selected chunk/group/profile, and validated
     18-shard BF16 source, then pass exact same-route graph/eager comparisons. MTP3/ordinary and
     MTP3/MTP4 checks are optional diagnostic exact-token/state/graph regressions, not selection
     prerequisites. Candidate-to-BF16 argmax differences remain diagnostic;
   - `profiles/bench/pareto-result-post-promotion-20260905.json`, whose bound schema-v4 input
     contains all twelve exact C=1..4 capacity outcomes and a C=1..4 whole matrix only for each
     capacity-eligible profile. Dense/XAttention capacity eligibility must match within every
     recipe/cache-group pair before whole acquisition begins. The eligible whole rows must pass
     spec-none ordinary fresh-request timing rows; MTP3 does not enter base ranking. BF16 argmax
     flips in PPL remain diagnostic, and DFlash is the required downstream speculative gate;
   - `profiles/bench/low-context-prefill-evaluation-20260905.json`, recomputed from the selected
     dense-control C1 ladder with `passes_p2048_gate=true` at the user's 2,000 tok/s floor. A valid
     below-floor diagnostic may guide profiling but is not production admission;
   - `profiles/bench/post-terminal-niah-20260905/admission.json`, recomputed by
     `tools/bench/validate_selected_niah.py` from the fresh selected-route five-position 64K NIAH
     run; and
   - `profiles/bench/post-terminal-selected-vision-20260905/admission.json`, recomputed by
     `tools/bench/validate_selected_vision_diagnostic.py` from the schema-v7 winner's committed
     one-image/no-thinking source-BF16 diagnostic. It must complete with the exact finite
     block-0/13/26/merger shapes and route/source identities, but remains diagnostic and imposes no
     invented numerical-error threshold; and
   - `profiles/bench/selected-hardware-use-20260905.json`, recomputed by the exact current
     `profiles/bench/post-terminal-selected-hardware-use-20260905` package from the selected-route
     trace/static schedule and conditional IU8/FP8 proofs. Existing MTP graph/eager/token,
     acceptance, and selected Q4-wave dispatch checks remain regression evidence only. No selected
     recipe carries a standalone MTP shortlist-head, alternate-precision, or mixed MTP-bulk
     optimization proof, and none of those retired proofs is a finalization or cutover gate; and
   - `profiles/bench/selected-dflash-20260905/selection.json`, a passed schema-v3
     `ninfer_r9700_dflash_selection` record whose `selected_base` is the same schema-v7 authority
     and whose companion conversion receipt and selected K/W are exact.

   The shared runner already admits the three DFlash presets for a hybrid companion only under its
   exact `--require-fp8-hybrid` contract, validates the companion conversion report and
   `base.authority` back to the selected hybrid receipt, retains the hybrid-width workspace proof,
   and covers the exact C1 shortlist plus ordered C1..4 capacity/Pareto cases while rejecting other
   presets and non-hybrid companions. If the four-role recipe wins, the remaining blockers are the
   selected base authority, materialization of its currently absent companion, regenerated
   selected-DFlash preparation, and explicitly authorized physical evidence; do not weaken or
   bypass the flag.

   The selected physical mode in
   `profiles/bench/post-terminal-focused-verification-20260905` must also publish
   `gpu-verification.json` and pass against that same tuple before step 2. Its selected-artifact
   graph/eager, MTP/ordinary, binder, cache, Text/Vision, and external-schema checks are the final
   pre-cutover correctness/exact-execution boundary; a successful CPU-only preparation or an
   earlier candidate qualification is not a substitute.

   A missing record, a changed bound hash, failed quality or
   whole/token parity, failed selected exact-token admission, below-floor low-context prefill,
   failed selected-hardware/focused
   verification, missing/invalid selected Vision diagnostic completion, or failed NIAH stops the cutover. Steps
   2--10 must not begin, and evaluation artifacts must not be renamed or copied under reserved
   final names. At the current snapshot the
   selected-chunk, six-quality, schema-v7, selected exact-token, low-context-admission,
   selected-hardware, focused physical verification, NIAH-admission, selected-Vision completion,
   and DFlash-selection outputs
   are not yet published or passed, so cutover is blocked.

   Before running gate 0 and before creating the isolated staging tree, exercise exactly the
   schema-v7 winner's source-to-plan path without creating an artifact or initializing a GPU. The
   producer dispatches to exactly one of the all-Q4, source-MSE mixed, or authority-bound four-role
   converters from the winner identity; callers do not select a converter branch themselves:

   ```bash
   export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   /ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python -m tools.bench.prepare_selected_converter_preflight \
     --selection profiles/bench/pareto-result-post-promotion-20260905.json \
     --model /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
     --draft-ranking out/qwen3_8_27b_draft_ranking.i64 \
     --prospective-artifact /ssdpool2nvme/local_llm/.ninfer-r9700-cutover-staging-20260905/qwen3.8-27b-r9700-selected.ninfer \
     --out profiles/bench/selected-converter-preflight-20260905.json
   ```

   The winner-only receipt binds the selection and selected evaluation artifact, converter and
   inventory implementation, exact source/index/shard and frontend-resource identities, ranking,
   recipe, format inventory, complete object-plan digest, and absent prospective destination.
   Gate 0 reopens and recomputes this receipt. The prospective path is validated for the four-role
   capacity requirement but is not created by any branch. The JSON must retain 18 source shards,
   1,199 BF16 source tensors, six frontend resources, 1,124 planned objects, and the validated
   ranking/provenance hashes. The selected plan contains either 439 Q4G64 matrices; 183 source-MSE
   Q4G64 plus 256 source-MSE W8G32 matrices; or the authority-fixed 144 rowwise-FP8 matrices plus
   295 all-other Q4G64 matrices. This preflight writes only its receipt; it writes no artifact and
   does not select or promote the evaluation identity.

   After that converter preflight and before step 2, one read-only cutover-admission command must
   reopen the records above with their
   owning validators, recompute the schema-v7 winner, and publish one create-only
   `profiles/bench/final-artifact-cutover-admission-20260905.json` receipt. The receipt must retain
   each input path and SHA-256, the selected evaluation artifact and executable identities, cache
   group, attention profile, prefill chunk, all twelve exact C=1..4 capacity outcomes, and whole
   matrices exactly for capacity-eligible profiles. It must
   require the same selected tuple in the exact-token, NIAH, selected-Vision, focused,
   hardware-use, and DFlash records. For low-context only, it must require the same winner recipe, artifact bytes,
   cache group, and chunk but require the unique schema-v7 dense-control profile/executable rather
   than the selected XAttention profile/executable when those differ. It must retain MTP regression
   coverage without requiring shortlist-head, alternate-precision,
   bulk-W8, or other MTP optimization proof for cutover. It must require the loaded-ELF native-FP8
   proof only for a four-role winner, reject that proof on an inapplicable winner, and require the
   DFlash companion decision to name the same base
   selection and its independently validated conversion receipt. Publication must use an exclusive
   same-directory atomic operation, revalidate the published bytes, and fail if the output name is
   occupied, including by a dangling symlink. A shell success status or the independent existence
   of all inputs is not a substitute for this joined receipt.

   The executable joined gate is prepared at
   `profiles/bench/final-artifact-cutover-admission-prepare-20260905`: it invokes the owning
   validators and publishes no receipt while any physical authority is absent or fails fresh
   revalidation. Its pre-decision closure excludes the obsolete standalone MTP shortlist-head and
   bulk-optimization requirements while retaining MTP regression in the whole and selected-trace
   evidence. The final receipt itself remains
   absent and is an explicit blocker, not permission
   to begin editing once only some individual reports appear. The preparation package is
   closure-bound to the validators it invokes; validation performs no conversion, benchmark, GPU,
   or product mutation.

   Steps 2--9 are staging operations after that receipt, not incremental edits to the admitted
   product tree. Step 10 is the only integration boundary.

1. Freeze selection provenance. Retain the schema-v7 three-recipe base Pareto decision and schema-v3 DFlash
   selection decision, exact quality/capacity/whole matrix manifests (the whole rows retain the
   separately timed prefill/decode phases), all raw report hashes, the complete non-dominated
   frontier, normalized objectives, all three per-recipe winners, the exact hybrid conversion
   receipt when applicable, exact retained capacity-failure provenance for excluded profiles, and
   the `terminal_production_selection` rule
   `global_maximin_whole_then_capacity_then_quality_then_canonical_v1`, its normalized objectives,
   and decisive tie-break stage used to select the artifact/cache/execution tuple,
   selected DFlash K/W plus its shortlist/capacity/parity/determinism/generated-quality inputs,
   selected evaluation-artifact hashes, benchmark executable hashes, GPU/toolchain identity, and the
   rejected candidates with their failure evidence. Quality is an admission gate; terminal ranking
   maximizes worst matched throughput, then capacity, then remaining declared-tier quality budget,
   and uses canonical artifact/static identity only for a complete measured tie. The decision uses
   retained per-cell means exactly and retains raw repetition spread as evidence, not a tolerance.
   Do not rewrite an evaluation report to claim that it measured the later final identity.

2. Stage the converter authority in an isolated cutover tree. Do not edit the currently admitted
   product tree in place. Record the exact pre-cutover source state and create a fresh, private
   staging tree only after the joined cutover-admission receipt passes. Within that staging tree,
   make
   `tools/convert/qwen3_8_27b_r9700/convert.py` and `inventory.py` describe the selected recipe
   under its non-evaluation recipe id and production conversion
   status. Make the retained DFlash converter accept only that final base identity and emit only
   selected DFlash identity; collapse `dflash2_q4_inventory.py` to the selected base inventory.
   Set the final conversion report's `weight_recipe_selected` to true, retain the fresh final
   artifact path/byte count/SHA-256 computed after atomic close, and bind the selection-report
   path/hash and selected evaluation artifact path/hash. Remove the losing recipe, BF16-role, and
   evaluation-only converter entry points and their inventory tests rather than leaving tools
   that emit artifacts the product no longer accepts.

   Validate the DFlash extension from the exact selected evaluation base with the same no-output
   discipline (the optional `--out` is checked and reported but is not created):

   ```text
   /ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python -m tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 --base "${SELECTED_EVAL_BASE}" --dflash-model /ssdpool2nvme/local_llm/models/qwen3.8-27b-dflash2 --out "${FINAL_DFLASH_ARTIFACT}" --preflight-only
   ```

   The report must bind the base identity/size/SHA-256, DFlash config/README/safetensors hashes,
   81 BF16 source tensors fused into 66 appended objects, both BF16 selector codebooks, the
   complete 1,190-object plan, projected file/device bytes, and the pending/final report paths.
   When the hybrid branch wins, the DFlash preflight and completed conversion must additionally
   bind the hybrid base's conversion-receipt path/SHA-256, recipe/selection/object-plan identity,
   and source index/ranking hashes. It must append to the exact hybrid base; neither existing
   all-Q4 nor mixed companion may be reused. The projected hybrid companion is 22,763,026,944
   file bytes and 22,750,001,152 device-arena bytes, but those are planning bounds only: fresh
   DFlash K/W quality, C=1..4 capacity, and whole evidence remain mandatory.
   No DFlash capacity impact may be derived from the historical hybrid C1--C4 slack: that report
   omitted MTP plus optimized-head materialization and its apparent slack is invalid. The appended
   companion's selected-K/W C=1..4 capacity outcomes must be measured against its exact loaded
   feature set after base selection; retained memory-admission failures are exclusions rather than
   fabricated capacities.

3. Materialize new artifacts from the isolated staging tree into fresh absent staging paths; never
   rename or overwrite the evaluation files and do not publish either reserved final filename yet.
   `weights_id` is embedded in the
   NInfer v2 directory and therefore changes the artifact bytes and SHA-256. Convert the final base
   from the complete BF16 source, then convert the final DFlash companion from that final base and
   the bound DFlash source. Verify the complete ordered object inventory, formats, shapes, layouts,
   offsets, encoded sizes, and payload bytes against the selected evaluation artifacts. The base
   must have 1,124 objects and the companion must add exactly the registered 66 DFlash objects.
   Retain fresh final conversion reports and hashes; do not edit the old conversion reports.

4. Collapse target admission in the isolated staging tree. In
   `src/targets/qwen3_8_27b/export/ninfer/targets/qwen3_8_27b/package.h`, replace the evaluation
   `WeightsProfile` enumeration with exactly the final base and final DFlash profiles. In
   `src/targets/qwen3_8_27b/impl/package_identity.cpp`, accept only the two canonical final
   identities and reject every old candidate/evaluation identity. Simplify
   `src/targets/qwen3_8_27b/impl/load/bindings.cpp` and
   `src/targets/qwen3_8_27b/impl/variant.cpp` to the selected base/DFlash formats and their A8
   workspace requirements. Do not retain an alias from either selected evaluation identity to a
   final profile.

5. Freeze the selected cache group. Remove `NINFER_R9700_KV_VALUE_GROUP` from the root
   `CMakeLists.txt` and bake the selected value into
   `src/targets/qwen3/impl/runtime/r9700_cache_profile.h`. Remove the losing production dispatch
   from `src/core/fp8_int4_paged_kv_cache.cpp`,
   `src/ops/r9700/kv/fp8_int4_kv_append.hip`,
   `src/ops/r9700/kv/fp8_int4_kv_attention.hip`, and
   `src/targets/qwen3_8_27b/impl/r9700_full_attention.hip`. Keep dual-group logic only in an
   explicitly qualification-only oracle if it remains useful. Update the cache runtime,
   persistence, benchmark-report, and independent codec tests to require the one selected group.
   This step is mandatory even if G16 wins: leaving a configure-time G32 product build would retain
   a second unsupported cache ABI.

6. Freeze selected arithmetic. Remove the product-wide A4 Q4 configuration knob and bake A8 into
   the Q4 route. If all-Q4 wins, remove W8 candidate/evaluator profile selection and the
   product-wide W8 activation-width knob; if mixed wins, retain only its selected adaptive-A8 W8
   route and remove the losing recipe paths. If the four-role hybrid wins, retain exactly its
   four authority-selected rowwise-FP8 roles and their all-other-Q4 fallback in both the base and
   byte-exact DFlash companion; do not collapse that branch into the all-Q4 recipe. Retain
   compile-isolated controls only as qualification tools that cannot enter the Engine target.
   Keep the already selected FP8-Q/K WMMA classifier fixed and remove its product-wide control
   build if no further final-artifact gate depends on it.

7. Update campaign and application surfaces in the isolated staging tree. Change
   `tools/bench/run_ninfer_bench_matrix.py`'s DFlash identity guard to the selected final DFlash
   identity, update `tools/bench/run_niah_check.py` and its fixtures to consume only the terminal
   winner's artifact/cache/execution tuple, and make post-selection benchmark/PPL helpers require
   that selected profile instead of presenting recipe, G16/G32, or attention route as product
   choices. The DFlash selection input must bind the same terminal winner. CLI, PPL, benchmark, and serving
   C++ applications already require an explicit artifact path and report the loaded canonical
   identity; they have no hardcoded W8 default to replace. Preserve that artifact-derived behavior
   and update only their examples/help text if the final filenames are shown.

8. Replace fixtures and qualifiers. Update `tools/r9700/make_sparse_r9700_candidate.py`,
   `tools/r9700/target_binding_qual.cpp`, `tools/r9700/runtime_planner_qual.cpp`,
   `tools/r9700/target_variant_gdn_qual.cpp`, `tests/CMakeLists.txt`, and
   `tests/test_target_registry.cpp` to construct and admit the two final
   profiles, validate the selected formats/workspaces, and explicitly reject every former
   evaluation identity. Update artifact converter tests to require production status, selected
   provenance, final recipe ids, and the final identities. Report-format tests must use neutral
   fixture identities rather than a production name.

9. Replace active documentation. Update `AGENTS.md`, `README.md`, `docs/README.md`, `docs/cli.md`,
   `docs/performance.md`, `docs/maintainer/artifact-container.md`,
   `docs/maintainer/paged-kv-cache.md`, `docs/maintainer/softmax-attention.md`,
   `docs/maintainer/storage-layouts.md`, `docs/maintainer/tensor-formats.md`,
   `docs/maintainer/qwen3.8-27b-artifact.md`,
   `docs/maintainer/qwen3.8-27b-model.md`, `tools/README.md`, `tools/bench/README.md`,
   `tools/ppl/README.md`, and `tools/r9700/README.md`. Rename or replace
   `docs/maintainer/r9700-integer-artifact-candidate.md` with the final artifact authority and
   remove the temporary autonomous ledger after all of its remaining work is closed. Preserve
   `docs/maintainer/r9700-overhaul-plan.md` and raw `profiles/` reports as historical evidence;
   their old identity strings describe what was actually measured and must not be rewritten.

10. Verify and publish the atomic cutover. First run every scan, build, and real-artifact check
    below against the isolated staging tree and its private artifact paths. Run exact repository
    scans proving that active code, converters,
    fixtures, tests, CLI examples, and current authorities contain neither a provisional/evaluation
    identity nor the losing cache/weight profile. Build a fresh product configuration with no cache
    group, Q4 activation-width, W8 activation-width, or attention-control selector. Run converter
    inventory/codec tests, artifact reader/materializer tests, target registry and sparse complete
    binder qualification, runtime planner/cache persistence tests, CLI/PPL/serve/benchmark help and
    schema tests, then load both real final artifacts on the R9700. Confirm ordinary inference from
    the base, ordinary plus selected-K/W DFlash inference from the companion, exact frontend
    resources, no runtime repacking, selected cache group in reports, and rejection of every old
    identity. Finally regenerate publication manifests/checksums from the final bytes.

    Only after all checks pass may the two final artifacts be copied to their reserved absent paths
    and a new artifact manifest be published. The manifest must bind both final artifact hashes and
    the staged source/build identity. Every destination must be rejected under lstat semantics before
    publication, and each file must be published without replacement. These new files remain inert
    while the current product admits only evaluation identities. Apply the complete staged
    source/documentation change to the product tree last, as one reviewed integration change; that is
    the consumer-visible cutover point. If staging or validation fails, remove only private files
    whose inodes were created by that staging invocation and leave the product tree unchanged. If
    artifact/manifest publication or final source integration fails, restore the recorded pre-cutover
    source state and remove only newly published final names after verifying their recorded inodes;
    never follow a symlink or delete an occupied namespace. Do not use partial identity aliases as a
    recovery path.

## Final focused verification commands

Run this set from the isolated staging tree only after steps 1--9 are complete. Before running it,
set `SELECTED_GROUP` and
`SELECTED_PREFILL_CHUNK` from the schema-v7 authority, replace
the `/absolute/path/to` prefixes while preserving the exact final filenames and weights identities,
and replace `FINAL_DFLASH_K` and `FINAL_DFLASH_W` in the DFlash commands with the selected integer
literals from the retained schema-v3 DFlash selection decision. K/W cannot be inferred from the
companion artifact and must not be selected from a filename, timestamp, or default.

```bash
TERMINAL_SELECTION=profiles/bench/pareto-result-post-promotion-20260905.json
mapfile -t SELECTED < <(python3 - "${TERMINAL_SELECTION}" <<'PY'
import json, sys
from pathlib import Path
from tools.ppl.pareto import validate_terminal_production_authority

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
terminal, _ = validate_terminal_production_authority(value)
print(terminal["winner_cache_profile"]["value_group"])
print(value["selected_prefill_chunk"])
print(terminal["winner_artifact"]["weights_id"])
PY
)
test "${#SELECTED[@]}" -eq 3
SELECTED_GROUP=${SELECTED[0]}
SELECTED_PREFILL_CHUNK=${SELECTED[1]}
SELECTED_EVAL_WEIGHTS_ID=${SELECTED[2]}
case "${SELECTED_EVAL_WEIGHTS_ID}" in
  r9700-q4g64-n16k16-eval)
    FINAL_BASE_WEIGHTS_ID=r9700-q4g64-n16k16
    FINAL_DFLASH_WEIGHTS_ID=r9700-q4g64-n16k16-dflash2-q4g64
    FINAL_QUALITY_TIER=capacity-speed
    FINAL_NLL_GATE=0.04879016416943205
    FINAL_BASE_ARTIFACT=/absolute/path/to/qwen3.8-27b-r9700-q4g64-n16k16.ninfer
    FINAL_DFLASH_ARTIFACT=/absolute/path/to/qwen3.8-27b-r9700-q4g64-n16k16-dflash2-q4g64.ninfer ;;
  r9700-q4-w8-mse-n16k16-eval)
    FINAL_BASE_WEIGHTS_ID=r9700-q4-w8-mse-n16k16
    FINAL_DFLASH_WEIGHTS_ID=r9700-q4-w8-mse-n16k16-dflash2-q4g64
    FINAL_QUALITY_TIER=accuracy
    FINAL_NLL_GATE=0.02
    FINAL_BASE_ARTIFACT=/absolute/path/to/qwen3.8-27b-r9700-q4-w8-mse-n16k16.ninfer
    FINAL_DFLASH_ARTIFACT=/absolute/path/to/qwen3.8-27b-r9700-q4-w8-mse-n16k16-dflash2-q4g64.ninfer ;;
  r9700-q4g64-f8e4m3-four-role-n16k16-eval)
    FINAL_BASE_WEIGHTS_ID=r9700-q4g64-f8e4m3-four-role-n16k16
    FINAL_DFLASH_WEIGHTS_ID=r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4g64
    FINAL_QUALITY_TIER=capacity-speed
    FINAL_NLL_GATE=0.04879016416943205
    FINAL_BASE_ARTIFACT=/absolute/path/to/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16.ninfer
    FINAL_DFLASH_ARTIFACT=/absolute/path/to/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4g64.ninfer ;;
  *) echo "unsupported terminal recipe: ${SELECTED_EVAL_WEIGHTS_ID}" >&2; exit 2 ;;
esac
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

case "${SELECTED_GROUP}" in
  16) FINAL_PROFILE=r9700-g16; FINAL_PPL_FLAG=--g16-ppl-bin; FINAL_WEIGHTS_FLAG=--g16-weights ;;
  32) FINAL_PROFILE=r9700-g32; FINAL_PPL_FLAG=--g32-ppl-bin; FINAL_WEIGHTS_FLAG=--g32-weights ;;
  *) echo "SELECTED_GROUP must be 16 or 32" >&2; exit 2 ;;
esac
case "${SELECTED_PREFILL_CHUNK}" in
  1024|2048|4096|8192) ;;
  *) echo "SELECTED_PREFILL_CHUNK must be 1024, 2048, 4096, or 8192" >&2; exit 2 ;;
esac
```

These are the host/build checks. The fresh configuration deliberately has no cache-group, Q4/W8
activation-width, or attention-profile selector. The CTest regex names current observable host
contracts; it does not run the device-labelled qualification programs.

```bash
cmake -S . -B build-r9700-final -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON \
  -DBUILD_TESTING=ON
cmake --build build-r9700-final --parallel 4

ctest --test-dir build-r9700-final --output-on-failure \
  -R '^ninfer_(public_api|fp8_int4_kv_oracle|xattention_oracle|artifact_reader|tensor|admission_policy|sampling_defaults|gdn_replay_records|r9700_linear_prefill_dispatch|r9700_fp8_activation_contract|r9700_fp8_execution_state_contract|qwen3_runtime_mechanisms|qwen3_score_index|cli_options|bench_support)_test$'

TEST_PYTHON=/absolute/path/to/python-with-pytest-torch-and-safetensors
# Invoke the absolute venv launcher directly; do not resolve its symlink to the base interpreter.
"${TEST_PYTHON}" -c 'import pytest, safetensors, torch'
NINFER_RUN_R9700_CODEC_TESTS=0 "${TEST_PYTHON}" -m pytest \
  tests/artifact/test_container.py \
  tests/artifact/test_layouts.py \
  tests/test_bench_matrix.py \
  tests/test_serve_corpus.py \
  tools/bench/test_post_terminal_focused_verification.py \
  tools/bench/test_assemble_dflash_selection.py \
  tools/bench/test_prepare_whole_profile.py \
  tools/bench/test_run_niah_check.py \
  tools/bench/test_run_ninfer_bench_matrix.py \
  tools/ppl/test_assemble_pareto.py \
  tools/ppl/test_exact_gate.py \
  tools/ppl/test_pareto.py \
  tools/convert/qwen3/common/test_conversion.py \
  tools/convert/qwen3_8_27b_r9700/test_build_draft_ranking.py \
  tools/convert/qwen3_8_27b_r9700/test_codec.py \
  tools/convert/qwen3_8_27b_r9700/test_convert_dflash2_q4.py \
  tools/convert/qwen3_8_27b_r9700/test_convert_q4_w8_mse.py \
  tools/convert/qwen3_8_27b_r9700/test_dflash2_q4_inventory.py \
  tools/convert/qwen3_8_27b_r9700/test_eval_inventories.py \
  tools/convert/qwen3_8_27b_r9700/test_vectorized_codec.py \
  tools/reference/qwen3_8_27b_bf16/test_kv_codec.py \
  tools/reference/qwen3_8_27b_bf16/test_protocol.py \
  tools/parity/qwen3_8_27b/test_vision.py

build-r9700-final/apps/ninfer --help
build-r9700-final/apps/ninfer-ppl --help
build-r9700-final/apps/ninfer-serve --help
build-r9700-final/bench/ninfer_bench --help
```

Run serving schema/unit contracts separately so a schema failure is not confused with a physical
model failure:

```bash
ctest --test-dir build-r9700-final --output-on-failure \
  -R '^ninfer_(openai_schema|responses_schema|response_store|anthropic_schema|tool_call_parser|serve_options|request_log|http_error_handler)_test$'
```

The following focused R9700 qualification set covers the final artifact/binder boundary, selected
cache representation and persistence, Text/Vision leaves, ordinary and speculative state, graph
and eager schedules, both still-selectable dense and XAttention implementations, and the public
Engine boundary. It intentionally excludes unrelated instruction micro-probes.

```bash
ctest --test-dir build-r9700-final --output-on-failure \
  -R '^ninfer_r9700_(core|artifact|kv_capacity|registry|target_binding|kv|linear|state|dflash_state|dflash2_select|bidirectional_gqa|grouped_dynamic_conv|swa|kv_cache_append_prefix|kv_ram|sampling|speculative_round|rope|vision_pos_embed|vision_attention|gdn_recurrence|gdn_replay_fold|eager|gdn|full_attention|xattention|target_variant_gdn|mtp_round|scalar_schedule|runtime_planner|engine_boundary)_qual$'
```

Exercise both real final artifacts explicitly. The base checks ordinary, MTP, and Vision product
surfaces; the companion checks ordinary loading plus the selected DFlash K/W route. DFlash is
text-only by contract.

```bash
build-r9700-final/apps/ninfer "${FINAL_BASE_ARTIFACT}" \
  --messages examples/cli/messages/text_chat_history.json \
  --prefill-chunk "${SELECTED_PREFILL_CHUNK}" --max-new 64
build-r9700-final/apps/ninfer "${FINAL_BASE_ARTIFACT}" \
  --messages examples/cli/messages/text_chat_history.json \
  --prefill-chunk "${SELECTED_PREFILL_CHUNK}" \
  --spec mtp --draft-tokens 3 --lm-head-draft --max-new 64
build-r9700-final/apps/ninfer "${FINAL_BASE_ARTIFACT}" \
  --messages examples/cli/messages/image_chart.json --vision \
  --prefill-chunk "${SELECTED_PREFILL_CHUNK}" --max-new 64

build-r9700-final/apps/ninfer "${FINAL_DFLASH_ARTIFACT}" \
  --messages examples/cli/messages/text_chat_history.json \
  --prefill-chunk "${SELECTED_PREFILL_CHUNK}" --max-new 64
build-r9700-final/apps/ninfer "${FINAL_DFLASH_ARTIFACT}" \
  --messages examples/cli/messages/text_chat_history.json \
  --prefill-chunk "${SELECTED_PREFILL_CHUNK}" \
  --spec dflash --draft-tokens FINAL_DFLASH_K \
  --dflash-verify-width FINAL_DFLASH_W --lm-head-draft --max-new 64
```

Run the source-BF16 PPL/token gate on the final base bytes in separate prefill and decode
campaigns. Splitting schedules avoids inventing a prefill-vs-decode tolerance: the decode campaign
still runs the supported graph/eager, MTP/ordinary, and draft-window exact execution comparisons.
Use the selected recipe's declared quality tier: `capacity-speed` for all-Q4 and four-role hybrid,
or `accuracy` for the source-MSE mixed recipe. The recipe switch above sets the corresponding
fixed NLL threshold; do not infer it from a filename or reuse another recipe's gate.

```bash
/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python tools/ppl/run.py \
  --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
  --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  "${FINAL_PPL_FLAG}" build-r9700-final/apps/ninfer-ppl \
  "${FINAL_WEIGHTS_FLAG}" "${FINAL_BASE_ARTIFACT}" \
  --profiles "bf16-reference,${FINAL_PROFILE}" \
  --quality-tier "${FINAL_QUALITY_TIER}" --gate "${FINAL_PROFILE}=${FINAL_NLL_GATE}" \
  --schedule prefill --prefill-chunk "${SELECTED_PREFILL_CHUNK}" --no-extras \
  --out profiles/ppl/final-base-prefill.json

/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python tools/ppl/run.py \
  --bf16-reference-ppl-bin tools/reference/qwen3_8_27b_bf16/ppl.py \
  --bf16-reference-weights /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  "${FINAL_PPL_FLAG}" build-r9700-final/apps/ninfer-ppl \
  "${FINAL_WEIGHTS_FLAG}" "${FINAL_BASE_ARTIFACT}" \
  --profiles "bf16-reference,${FINAL_PROFILE}" \
  --quality-tier capacity-speed --gate "${FINAL_PROFILE}=0.048790164" \
  --schedule decode --prefill-chunk "${SELECTED_PREFILL_CHUNK}" \
  --spec mtp --draft-tokens 3 \
  --execution-parity-max-abs-nll 0 \
  --out profiles/ppl/final-base-decode.json

/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python \
  -m tools.parity.qwen3_8_27b.vision \
  --weights "${FINAL_BASE_ARTIFACT}" \
  --model-dir /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
  --messages examples/cli/messages/image_chart.json \
  --output profiles/parity/qwen3.8-27b-final-vision-image-chart.json
```

Run the final base and companion end-to-end matrices from the same fresh executable. The base pair
provides matched phase and fresh-prompt whole-inference C=1..4 evidence. The DFlash matrix provides
matched phase/whole timing, ordinary-token parity, acceptance accounting, and repeated proposal-
trace determinism for the selected K/W. It is not necessary to repeat the pre-selection shortlist
or losing K/W candidates on the final identity.

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto \
  --prefill-chunk "${SELECTED_PREFILL_CHUNK}" \
  --bench build-r9700-final/bench/ninfer_bench --no-build \
  --weights "${FINAL_BASE_ARTIFACT}" \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-kv-value-group "${SELECTED_GROUP}" \
  --output-dir profiles/bench/final-base-pareto

python3 tools/bench/run_ninfer_bench_matrix.py --preset pareto-whole \
  --prefill-chunk "${SELECTED_PREFILL_CHUNK}" \
  --bench build-r9700-final/bench/ninfer_bench --no-build \
  --weights "${FINAL_BASE_ARTIFACT}" \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-kv-value-group "${SELECTED_GROUP}" \
  --output-dir profiles/bench/final-base-pareto-whole

python3 tools/bench/run_ninfer_bench_matrix.py --preset dflash-pareto \
  --prefill-chunk "${SELECTED_PREFILL_CHUNK}" \
  --dflash-draft-tokens FINAL_DFLASH_K \
  --dflash-verify-width FINAL_DFLASH_W \
  --bench build-r9700-final/bench/ninfer_bench --no-build \
  --weights "${FINAL_DFLASH_ARTIFACT}" \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --expected-kv-value-group "${SELECTED_GROUP}" \
  --output-dir profiles/bench/final-dflash-pareto
```

Finally exercise the external HTTP surface through a real base artifact with both MTP and Vision
resident. The schema tests above own malformed-input detail; this smoke owns live OpenAI,
Responses, Anthropic, streaming, state, token-count, and media behavior. Use a fresh absent output
directory, reject an occupied path including a dangling symlink, enable shell noclobber before
redirecting logs, and clean up only the server process and private output inode created by this
invocation on failure. The illustrative commands below are not by themselves a publication
protocol; the cutover implementation must wrap them with those ownership checks.

```bash
FINAL_SERVE_DIR=profiles/serve/final-contract
if [[ -e "${FINAL_SERVE_DIR}" || -L "${FINAL_SERVE_DIR}" ]]; then
  echo "refusing occupied final serve output: ${FINAL_SERVE_DIR}" >&2
  exit 1
fi
mkdir -- "${FINAL_SERVE_DIR}"
set -C
build-r9700-final/apps/ninfer-serve "${FINAL_BASE_ARTIFACT}" \
  --host 127.0.0.1 --port 18080 --model-id qwen3.8-27b-final \
  --max-context 8192 --kv-capacity auto --max-concurrency 2 \
  --prefill-chunk "${SELECTED_PREFILL_CHUNK}" \
  --spec mtp --draft-tokens 3 --lm-head-draft --vision \
  >"${FINAL_SERVE_DIR}/server.stdout.txt" \
  2>"${FINAL_SERVE_DIR}/server.stderr.txt" &
FINAL_SERVER_PID=$!
trap 'kill "${FINAL_SERVER_PID}" 2>/dev/null || true' EXIT
python3 tools/smoke/serve_contract.py \
  --base-url http://127.0.0.1:18080 --model qwen3.8-27b-final \
  >"${FINAL_SERVE_DIR}/contract.json"
kill "${FINAL_SERVER_PID}"
wait "${FINAL_SERVER_PID}" || true
trap - EXIT
set +C
```

Successful completion closes the focused final-suite gate only when the retained reports bind the
new final artifact and executable hashes, identify the selected cache group, contain no failed or
missing cells, and the publication manifest is then generated from those exact final bytes.

## Current hardcoded cutover points

The following are real promotion dependencies rather than generic occurrences of the word
"provisional":

- Target identity/profile: `package.h`, `package_identity.cpp`, `load/bindings.cpp`, `variant.cpp`.
- Converter identity/status: `inventory.py`, `q4_inventory.py`, `dflash2_q4_inventory.py`,
  `convert.py`, `convert_q4.py`, `convert_dflash2_q4.py`, and the other evaluation converter and
  inventory modules under `tools/convert/qwen3_8_27b_r9700/`.
- Product build/cache selection: root `CMakeLists.txt`, `r9700_cache_profile.h`, the four production
  cache/attention sources named in step 5, and benchmark/PPL expected-group validation.
- Product prefill-chunk selection: replace the generic 4096 defaults in `include/ninfer/types.h`,
  `apps/cli/options.h`, `src/serve/serve_options.h`, `apps/ppl/main.cpp`, and benchmark support with
  the schema-v7-selected value; update the corresponding CLI/serving/help authorities. Until that
  atomic promotion, every selection-bound command must pass the selected value explicitly.
- Physical/sparse qualification: `make_sparse_r9700_candidate.py`, `target_binding_qual.cpp`,
  `runtime_planner_qual.cpp`, `target_variant_gdn_qual.cpp`, `q4g64_linear_qual.hip`,
  `q4_tensor_dispatch_qual.hip`, `w8a8_wmma_linear_qual.hip`, `tests/CMakeLists.txt`, and
  `test_target_registry.cpp`.
- Campaign admission and final decisions: `tools/bench/run_ninfer_bench_matrix.py`,
  `tools/bench/assemble_dflash_selection.py`, `tools/bench/run_niah_check.py`,
  `tools/bench/run_serve_corpus.py`, `tools/bench/run_serve_concurrency.py`, and their focused tests.
  The current serve-corpus method fixes a dense build and 1024-token chunk; post-selection evidence
  must instead bind the schema-v7-selected attention profile and prefill chunk, while retained
  pre-selection reports continue to describe their original method.
- Converter contract tests: `qwen3/common/test_conversion.py`, `test_codec.py`, `test_convert_dflash2_q4.py`,
  `test_convert_q4_w8_mse.py`, `test_dflash2_q4_inventory.py`, and
  `test_eval_inventories.py`; keep `test_build_draft_ranking.py` and
  `test_vectorized_codec.py` only to the extent that they still exercise the final converter.
- User-facing current authorities: the files named in step 9.

There is no hidden CLI artifact default: `ninfer`, `ninfer-serve`, `ninfer-ppl`, and
`ninfer_bench` consume an explicit path and propagate the directory identity. The remaining unsafe
defaults are the root CMake cache group (currently G16), the target's admission of all evaluation
identities, and the generic 4096-token prefill chunk when a final validation command omits the
schema-v7-selected value. A selected final artifact would be rejected if only its filename changed,
and adding its new identity as an alias would leave the old evaluation lanes supported; both
outcomes violate the cutover contract. Neither recipe nor final prefill chunk may be hard-coded
before the C=1..4 decision.
