#!/usr/bin/env python3
"""Selected-chunk numerical quality campaign; stages never select a chunk themselves."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.bench.prefill_chunk_authority import validate_prefill_chunk_authority
from tools.ppl import run
from tools.ppl.quality_recovery_io import EXPECTED_AUTHORITIES, publish
from tools.reference.qwen3_8_27b_bf16.protocol import validate_checkpoint_files

PACKAGE = Path(__file__).resolve().parent
SELECTION = REPO / "profiles/bench/prefill-chunk-selection-panel-attention-20260921.json"
FROZEN_INPUTS = REPO / "profiles/bench/r9700-chunk-selection-panel-attention-20260921/inputs.json"
PYTHON = Path("/home/battlefront/.local/bin/python3.11")
SCORER = REPO / "tools/reference/qwen3_8_27b_bf16/ppl.py"
CHECKPOINT = Path("/ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16")
IDS = REPO / "tools/ppl/corpus.ids"
FINAL = PACKAGE / "quality-authorities-receipt-bound-n16k16.json"
PENDING = PACKAGE / "quality-authorities-receipt-bound-n16k16.pending.json"


def campaigns():
    for name, (weights_id, profile) in EXPECTED_AUTHORITIES.items():
        route = "dense" if profile == "dense" else "xattention"
        yield name, weights_id, profile, {
            group: REPO / f"build-r9700-selection-panel-{route}-g{group}-20260921/apps/ninfer-ppl"
            for group in (16, 32)
        }, PACKAGE / name.lower()


def reference_paths(chunk):
    if chunk == 4096:
        return (
            REPO / "profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json",
            REPO / "profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json",
        )
    return PACKAGE / f"bf16-chunk{chunk}-a/results.json", PACKAGE / f"bf16-chunk{chunk}-repeat.json"


def absent(paths):
    for path in paths:
        if os.path.lexists(path):
            raise ValueError(f"output already exists; preserve the previous attempt: {path}")


def validate_reference(chunk):
    authority, repeat = reference_paths(chunk)
    run.validate_bf16_repeat_comparison(repeat, authority)
    run.load_reused_bf16_cells(
        authority, bf16_weights=CHECKPOINT, bf16_scorer=SCORER,
        scorer_identity={"path": str(SCORER.resolve()), "bytes": SCORER.stat().st_size,
                         "sha256": run.file_sha256(SCORER)},
        ids=IDS, corpus_provenance=run.validate_corpus(IDS, 32768),
        lengths=[8192, 32768], skip="half", prefill_chunk=chunk, device=0,
    )


def verify_frozen_inputs(record):
    frozen = json.loads(FROZEN_INPUTS.read_text())
    if (frozen.get("artifact_type") != "ninfer_r9700_chunk_campaign_inputs"
            or frozen.get("schema_version") != 1):
        raise ValueError("unsupported frozen chunk campaign inputs")
    artifacts = {item["weights_id"]: item for item in frozen["artifacts"]}
    for weights_id in dict(EXPECTED_AUTHORITIES.values()):
        actual = run.inspect_candidate_artifact(REPO / f"out/qwen3.8-27b-{weights_id}.ninfer")
        if actual != artifacts.get(weights_id):
            raise ValueError(f"artifact differs from frozen chunk campaign: {weights_id}")
    binaries = {binary for *_, groups, _output in campaigns() for binary in groups.values()}
    for binary in binaries:
        if (not binary.is_file() or not os.access(binary, os.X_OK)
                or run.file_sha256(binary) != frozen["files"].get(str(binary.relative_to(REPO)))):
            raise ValueError(f"PPL scorer differs from frozen chunk campaign: {binary}")
    for source in record["sources"]:
        expected = dict(artifacts[source["weights_id"]])
        expected["file_size_bytes"] = expected.pop("bytes")
        if source["artifact"] != expected:
            raise ValueError("selected-chunk artifact evidence differs from frozen campaign")
        route = "dense" if source["xattention_profile"] == "dense" else "xattention"
        relative = (f"build-r9700-selection-panel-{route}-g{source['kv_value_group']}-20260921/"
                    "bench/ninfer_bench")
        benchmark = source["benchmark_executable"]
        if (benchmark["path"] != str(REPO / relative)
                or benchmark["sha256"] != frozen["files"].get(relative)):
            raise ValueError("selected-chunk benchmark evidence differs from frozen campaign")


def preflight():
    selection, record = validate_prefill_chunk_authority(SELECTION)
    verify_frozen_inputs(record)
    chunk = selection["selected_prefill_chunk"]
    run.validate_corpus(IDS, 32768)
    if chunk == 4096 or all(path.is_file() for path in reference_paths(chunk)):
        validate_reference(chunk)
        print(f"selected chunk {chunk}; BF16 reference and repeat validated")
    else:
        validate_checkpoint_files(CHECKPOINT)
        print(f"selected chunk {chunk}; fresh BF16 reference A/B and repeat required; "
              "reference stage requires an explicit Python 3.11 ROCm PyTorch interpreter")
    return selection


def common(chunk, python=PYTHON):
    return [str(python), str(REPO / "tools/ppl/run.py"),
            "--bf16-reference-ppl-bin", str(SCORER),
            "--bf16-reference-weights", str(CHECKPOINT), "--ids", str(IDS),
            "--schedule", "prefill", "--spec", "none", "--no-extras",
            "--prefill-chunk", str(chunk), "--device", "0"]


def quality_command(chunk, weights_id, profile, binaries, output):
    authority, repeat = reference_paths(chunk)
    mixed = weights_id == "r9700-q4-w8-mse-n16k16-eval"
    gate = "0.02" if mixed else "0.048790164169432"
    command = common(chunk) + [
        "--profiles", "bf16-reference,r9700-g16,r9700-g32",
        "--quality-tier", "accuracy" if mixed else "capacity-speed",
        "--gate", f"r9700-g16={gate}", "--gate", f"r9700-g32={gate}",
        "--expected-q4-activation-bits", "8", "--expected-w8-activation-bits", "8",
        "--expected-fp8-qk-wmma", "1", "--expected-xattention-profile", profile,
        "--reuse-bf16-campaign", str(authority), "--bf16-repeat-comparison", str(repeat),
        "--out", str(output),
    ]
    for group in (16, 32):
        command += [f"--g{group}-ppl-bin", str(binaries[group]),
                    f"--g{group}-weights", str(REPO / f"out/qwen3.8-27b-{weights_id}.ninfer")]
    if "four-role" in weights_id:
        command += ["--require-fp8-hybrid"]
    # run.py's default matrix is exactly 8192 and 32768 tokens.
    return command


def unchanged(selection):
    current, _ = validate_prefill_chunk_authority(SELECTION)
    if current != selection:
        raise ValueError("selected-chunk authority changed during this stage")


def reference(selection, python):
    chunk = selection["selected_prefill_chunk"]
    if chunk == 4096:
        validate_reference(chunk)
        return
    if python is None:
        raise ValueError("fresh BF16 requires --reference-python with Python 3.11 ROCm PyTorch")
    # Explicit dependency check occurs before any fresh GPU scoring or outputs.
    subprocess.run([str(python), "-c", "import sys; assert sys.version_info[:2] == (3, 11), "
                    "'fresh BF16 requires Python 3.11'; import torch; "
                    "assert torch.version.hip, 'fresh BF16 requires ROCm PyTorch'"], check=True)
    validate_checkpoint_files(CHECKPOINT)
    authority, repeat = reference_paths(chunk)
    first, second = authority.parent, PACKAGE / f"bf16-chunk{chunk}-b"
    probe = PACKAGE / f"bf16-chunk{chunk}-gdn-full-span.json"
    absent([first, second, repeat, probe])
    unchanged(selection)
    subprocess.run([str(python), str(SCORER.with_name("gdn_full_span_probe.py")),
                    "--device", "0", "--out-json", str(probe)], check=True)
    report = json.loads(probe.read_text(encoding="utf-8"))
    if (report.get("all_pass") is not True
            or report.get("geometry", {}).get("row_extents") != [4095, 4096]):
        raise ValueError(f"full-span GDN numerical qualification failed: {probe}")
    execution = report.get("provenance", {}).get("execution", {})
    recorded_python = execution.get("python_executable")
    if (not isinstance(recorded_python, str)
            or Path(recorded_python).resolve() != python.resolve()
            or execution.get("python_executable_sha256") != run.file_sha256(python)):
        raise ValueError(f"full-span GDN interpreter provenance differs: {probe}")
    for output in (first, second):
        unchanged(selection)
        subprocess.run(common(chunk, python) + ["--profiles", "bf16-reference",
                       "--quality-tier", "accuracy", "--allow-ungated", "--out", str(output)],
                       check=True)
    subprocess.run([str(PYTHON), str(REPO / "tools/ppl/compare_bf16_repeats.py"),
                    "--first", str(authority), "--second", str(second / "results.json"),
                    "--out", str(repeat)], check=True)
    validate_reference(chunk)
    unchanged(selection)


def quality(selection):
    from tools.ppl.assemble_pareto import _campaign_quality_candidate
    chunk = selection["selected_prefill_chunk"]
    validate_reference(chunk)
    absent([FINAL, PENDING, *(output for *_, output in campaigns())])
    failed = []
    for name, weights_id, profile, binaries, output in campaigns():
        unchanged(selection)
        completed = subprocess.run(quality_command(chunk, weights_id, profile, binaries, output))
        try:
            report = json.loads((output / "results.json").read_text())
            eligibility = []
            for group in (16, 32):
                cells, _ = _campaign_quality_candidate(report, weights_id, group, chunk)
                eligibility.append(all(cells[label]["eligible"] for label in ("8k", "32k")))
            expected_returncode = 0 if all(eligibility) else 1
            if completed.returncode != expected_returncode:
                raise ValueError("runner exit differs from replayed quality outcome")
            if not all(eligibility):
                print(f"{name}: retained measured quality exclusion; eligibility G16/G32={eligibility}")
        except (OSError, ValueError) as error:
            failed.append(f"{name}: {error}")
    unchanged(selection)
    if failed:
        raise ValueError("quality campaigns failed; inspect their reports: " + ", ".join(failed))


def publication(selection):
    from tools.ppl.assemble_pareto import _campaign_quality_candidate
    absent([FINAL, PENDING])
    entries = {}
    for name, weights_id, _, _, output in campaigns():
        path = output / "results.json"
        campaign = json.loads(path.read_text())
        identities = []
        eligibility = {}
        for group in (16, 32):
            cells, source = _campaign_quality_candidate(
                campaign, weights_id, group, selection["selected_prefill_chunk"])
            eligibility[str(group)] = all(cells[label]["eligible"] for label in ("8k", "32k"))
            identities.append({key: source[key] for key in (
                "weights_id", "sha256", "file_size_bytes", "conversion_receipt")})
        if identities[0] != identities[1]:
            raise ValueError(f"{name}: G16/G32 artifact identities differ")
        entries[name] = {"path": str(path), "sha256": run.file_sha256(path),
                         "artifact": identities[0], "eligibility_by_group": eligibility}
    unchanged(selection)
    payload = {"artifact_type": "ninfer_r9700_terminal_quality_authority_map",
               "schema_version": 2, "selected_prefill_chunk": selection["selected_prefill_chunk"],
               "selected_prefill_chunk_authority": {key: selection[key] for key in ("path", "sha256")},
               "concurrency": 1, "authorities": entries}
    with PENDING.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    publish(PENDING, FINAL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "reference", "quality", "publish"))
    parser.add_argument("--reference-python", type=Path)
    args = parser.parse_args()
    selection = preflight()
    if args.stage == "reference":
        reference(selection, args.reference_python)
    elif args.stage == "quality":
        quality(selection)
    elif args.stage == "publish":
        publication(selection)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
