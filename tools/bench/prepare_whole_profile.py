#!/usr/bin/env python3
"""Prepare one provenance-bound whole-inference rocprof command without executing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Sequence

from tools.bench.run_ninfer_bench_matrix import (
    MATRIX_SCHEMA_VERSION,
    PRODUCT_CONCURRENCIES,
    REPORT_SCHEMA_VERSION,
    R9700_KV_PLANE_LAYOUTS,
)
from tools.bench.validate_low_context_prefill import validate_ladder

SUPPORTED_PRESETS = frozenset({"pareto-whole", "dflash-pareto"})
SUPPORTED_XATTENTION_PROFILES = frozenset({"dense", "b128-s16-tau900"})
SUPPORTED_PREFILL_CHUNKS = frozenset({1024, 2048, 4096, 8192})
DISPATCH_COUNTERS = (
    "GL2C_HIT", "GL2C_MISS", "L2CacheHit", "TCP_REQ", "TCP_REQ_MISS",
    "GL2C_EA_RDREQ", "GL2C_EA_WRREQ", "SQ_WAVES",
)
ROCPROFV3 = Path("/opt/rocm/bin/rocprofv3")


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _replace_option(command: list[str], option: str, value: str) -> None:
    if option not in command:
        raise ValueError(f"source benchmark command lacks {option}")
    index = command.index(option)
    if index + 1 >= len(command):
        raise ValueError(f"source benchmark command has no value for {option}")
    command[index + 1] = value


def _identity_matches_file(identity: dict[str, Any], label: str) -> None:
    path = Path(str(identity.get("path", "")))
    expected_size = identity.get("file_size_bytes")
    expected_hash = identity.get("sha256")
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"{label} path or size changed since the matrix")
    if not isinstance(expected_hash, str) or file_sha256(path) != expected_hash:
        raise ValueError(f"{label} bytes changed since the matrix")


def _write_plan(
    out_dir: Path,
    *,
    kind: str,
    question: str,
    benchmark_command: list[str],
    power_profile_path: Path,
    source_matrix: dict[str, Any],
    artifact: dict[str, Any],
    bench: dict[str, Any],
    workload: dict[str, Any],
    kernel_include_regex: str | None,
    extra_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_data_dir = out_dir / (
        "rocprof-trace" if kind == "trace" else "rocprof-dispatch-pmc"
    )
    common = [str(ROCPROFV3), "--selected-regions", "-f", "rocpd", "-d", str(profile_data_dir)]
    if kind == "trace":
        if kernel_include_regex is not None:
            raise ValueError("kernel include regex applies only to dispatch-pmc")
        profiler_command = [
            *common, "--marker-trace", "--kernel-trace", "--memory-copy-trace",
            "--stats", "--summary", "--", *benchmark_command,
        ]
        required_power_profile = "auto"
    elif kind == "dispatch-pmc":
        if kernel_include_regex is None or not kernel_include_regex.strip():
            raise ValueError("dispatch-pmc requires a nonempty kernel include regex")
        profiler_command = [
            *common, "--marker-trace", "--kernel-trace",
            "--kernel-include-regex", kernel_include_regex.strip(),
            "--pmc", *DISPATCH_COUNTERS, "--", *benchmark_command,
        ]
        required_power_profile = "profile_standard"
    else:
        raise ValueError("profile kind must be trace or dispatch-pmc")

    if out_dir.exists():
        raise ValueError(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    power_before_path = out_dir / "power-profile-before.txt"
    power_after_path = out_dir / "power-profile-after.txt"
    payload = {
        "artifact_type": "ninfer_whole_profile_plan",
        "schema_version": 2,
        "status": "command_only_not_executed",
        "attribution_question": question.strip(),
        "profile_kind": kind,
        "measured_region": "ninfer_bench_measured",
        "source_matrix": source_matrix,
        "artifact": artifact,
        "benchmark_executable": bench,
        "workload": workload,
        "required_power_profile": {
            "value": required_power_profile,
            "sysfs_path": str(power_profile_path),
            "before_evidence": str(power_before_path),
            "after_evidence": str(power_after_path),
        },
        "kernel_include_regex": (
            kernel_include_regex.strip() if kernel_include_regex is not None else None
        ),
        "counters": list(DISPATCH_COUNTERS) if kind == "dispatch-pmc" else [],
        "benchmark_command": benchmark_command,
        "profiler_command": profiler_command,
        "interpretation": (
            "Attribution only. Use the source matrix's unprofiled timing for selection; "
            "profile only the selected ROCTx region."
        ),
    }
    if extra_provenance:
        payload.update(extra_provenance)
    (out_dir / "plan.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    quoted_profile_path = shlex.quote(str(power_profile_path))
    quoted_required_profile = shlex.quote(required_power_profile)
    quoted_before_path = shlex.quote(str(power_before_path))
    quoted_after_path = shlex.quote(str(power_after_path))
    before_guard = (
        f'test ! -e {quoted_before_path}\n'
        f'test ! -e {quoted_after_path}\n'
    )
    if kind == "dispatch-pmc":
        before_guard += (
            f'POWER_PROFILE_VALUE="$(cat {quoted_profile_path})"\n'
            f'test "$POWER_PROFILE_VALUE" = auto\n'
            f'sudo -v\n'
            f'restore_auto() {{\n'
            f'  printf "%s\\n" auto | sudo tee {quoted_profile_path} >/dev/null\n'
            f'  test "$(cat {quoted_profile_path})" = auto\n'
            f'}}\n'
            f'trap restore_auto EXIT\n'
            f'printf "%s\\n" profile_standard | sudo tee {quoted_profile_path} >/dev/null\n'
            f'POWER_PROFILE_VALUE="$(cat {quoted_profile_path})"\n'
            f'(set -C; printf "%s\\n" "$POWER_PROFILE_VALUE" > {quoted_before_path})\n'
            f'test "$POWER_PROFILE_VALUE" = profile_standard'
        )
    else:
        before_guard += (
            f'POWER_PROFILE_VALUE="$(cat {quoted_profile_path})"\n'
            f'(set -C; printf "%s\\n" "$POWER_PROFILE_VALUE" > {quoted_before_path})\n'
            f'test "$POWER_PROFILE_VALUE" = {quoted_required_profile}'
        )
    after_guard = (
        ('restore_auto\ntrap - EXIT\n' if kind == "dispatch-pmc" else '') +
        f'POWER_PROFILE_VALUE="$(cat {quoted_profile_path})"\n'
        f'POWER_PROFILE_READ_RC=$?\n'
        f'(set -C; printf "%s\\n" "$POWER_PROFILE_VALUE" > {quoted_after_path})\n'
        f'POWER_PROFILE_WRITE_RC=$?\n'
        f'set -e\n'
        f'if (( PROFILE_RC != 0 )); then exit "$PROFILE_RC"; fi\n'
        f'test "$POWER_PROFILE_READ_RC" -eq 0\n'
        f'test "$POWER_PROFILE_WRITE_RC" -eq 0\n'
        f'test "$POWER_PROFILE_VALUE" = {"auto" if kind == "dispatch-pmc" else quoted_required_profile}'
    )
    (out_dir / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + before_guard + "\n"
        + "set +e\n"
        + shlex.join(profiler_command) + "\n"
        + "PROFILE_RC=$?\n"
        + after_guard + "\n",
        encoding="utf-8",
    )
    return payload


def prepare(
    matrix_dir: Path,
    out_dir: Path,
    *,
    concurrency: int,
    prompt_tokens: int,
    generated_tokens: int,
    kind: str,
    question: str,
    expected_weights_id: str,
    expected_kv_value_group: int,
    expected_xattention_profile: str,
    expected_prefill_chunk: int,
    kernel_include_regex: str | None = None,
    expected_dflash_draft_tokens: int | None = None,
    expected_dflash_verify_width: int | None = None,
) -> dict[str, Any]:
    matrix_dir = matrix_dir.resolve()
    out_dir = out_dir.resolve()
    if not question.strip():
        raise ValueError("a named attribution question is required")
    if concurrency not in PRODUCT_CONCURRENCIES:
        raise ValueError("concurrency must be in [1, 4]")
    if prompt_tokens <= 0 or generated_tokens <= 0:
        raise ValueError("prompt and generated token counts must be positive")
    if not expected_weights_id:
        raise ValueError("selected weights identity is required")
    if expected_kv_value_group not in (16, 32):
        raise ValueError("selected KV value group must be 16 or 32")
    if expected_xattention_profile not in SUPPORTED_XATTENTION_PROFILES:
        raise ValueError("selected XAttention profile is unsupported")
    if expected_prefill_chunk not in SUPPORTED_PREFILL_CHUNKS:
        raise ValueError("selected prefill chunk is unsupported")
    if (matrix_dir / "failures.json").exists():
        raise ValueError("source matrix has retained failures and is not profile-eligible")
    manifest_path = matrix_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("artifact_type") != "ninfer_bench_matrix_run"
        or manifest.get("schema_version") != MATRIX_SCHEMA_VERSION
        or manifest.get("preset") not in SUPPORTED_PRESETS
    ):
        raise ValueError(
            f"source must be a complete schema-v{MATRIX_SCHEMA_VERSION} whole-inference matrix"
        )
    if concurrency not in manifest.get("concurrency", []):
        raise ValueError("requested concurrency is absent from the source matrix")
    if manifest.get("expected_kv_value_group") != expected_kv_value_group:
        raise ValueError("source matrix is not the selected KV value group")
    if manifest.get("expected_kv_plane_layouts") != R9700_KV_PLANE_LAYOUTS:
        raise ValueError("source matrix has the wrong KV plane-layout identity")
    xattention_profile = manifest.get("expected_xattention_profile")
    if xattention_profile != expected_xattention_profile:
        raise ValueError("source matrix is not the selected XAttention profile")
    if manifest.get("selected_prefill_chunk") != expected_prefill_chunk:
        raise ValueError("source matrix is not bound to the selected prefill chunk")
    power_profile = manifest.get("power_profile")
    if not isinstance(power_profile, dict) or any(
        power_profile.get(key) != "auto"
        for key in ("required", "observed", "rechecked_after")
    ):
        raise ValueError("source whole-inference timing is not proven under auto power profile")
    power_profile_path = Path(str(power_profile.get("sysfs_path", "")))
    if not power_profile_path.is_absolute():
        raise ValueError("source matrix has no absolute power-profile sysfs path")

    artifact = manifest.get("artifact")
    bench = manifest.get("bench")
    if not isinstance(artifact, dict) or not isinstance(bench, dict):
        raise ValueError("source matrix lacks artifact or executable provenance")
    if artifact.get("weights_id") != expected_weights_id:
        raise ValueError("source matrix artifact is not the selected weights identity")
    _identity_matches_file(artifact, "artifact")
    _identity_matches_file(bench, "benchmark executable")

    matches = [
        row for row in manifest.get("commands", [])
        if isinstance(row, dict)
        and row.get("concurrency") == concurrency
        and "whole_inference" in str(row.get("suite", ""))
    ]
    if len(matches) != 1:
        raise ValueError("source matrix has no unique whole-inference command at concurrency")
    source_record = matches[0]
    source_report_path = Path(str(source_record.get("report", "")))
    source_report = _load_json(source_report_path)
    expected_xattention = (
        {"xattention_qualification": False}
        if xattention_profile == "dense"
        else {
            "xattention_qualification": True,
            "xattention_profile": "b128-s16-tau900",
            "xattention_find_block": 128,
            "xattention_stride": 16,
            "xattention_tau_permille": 900,
        }
    )
    if (
        source_report.get("artifact_type") != "ninfer_bench_report"
        or source_report.get("schema_version") != REPORT_SCHEMA_VERSION
        or source_report.get("config", {}).get("concurrency") != concurrency
        or source_report.get("load", {}).get("weights_id") != artifact.get("weights_id")
        or source_report.get("config", {}).get("kv_value_group") != expected_kv_value_group
        or source_report.get("config", {}).get("prefill_chunk") != expected_prefill_chunk
        or source_report.get("config", {}).get("kv_plane_layouts")
        != R9700_KV_PLANE_LAYOUTS
    ):
        raise ValueError("source whole-inference report identity is invalid")
    config = source_report["config"]
    if any(config.get(key) != value for key, value in expected_xattention.items()):
        raise ValueError("source whole-inference XAttention profile is invalid")
    if xattention_profile == "dense" and any(
        key in config for key in (
            "xattention_profile", "xattention_find_block", "xattention_stride",
            "xattention_tau_permille",
        )
    ):
        raise ValueError("dense source report retains XAttention profile fields")
    if manifest["preset"] == "dflash-pareto":
        if expected_dflash_draft_tokens is None or expected_dflash_verify_width is None:
            raise ValueError("selected DFlash K/W are required for a DFlash matrix")
        if (
            source_report["config"].get("draft_tokens") != expected_dflash_draft_tokens
            or source_report["config"].get("dflash_verify_width")
            != expected_dflash_verify_width
        ):
            raise ValueError("source matrix does not use the selected DFlash K/W")
    elif expected_dflash_draft_tokens is not None or expected_dflash_verify_width is not None:
        raise ValueError("DFlash K/W cannot be applied to a base whole-inference matrix")
    wanted = (prompt_tokens, generated_tokens)
    observed = {
        (test.get("n_prompt"), test.get("n_gen"))
        for test in source_report.get("tests", [])
        if isinstance(test, dict) and test.get("kind") == "whole"
    }
    if wanted not in observed:
        raise ValueError("requested whole-inference geometry is absent from the measured matrix")

    command = source_record.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("source matrix command is invalid")
    benchmark_command = list(command)
    chunk_index = benchmark_command.index("--prefill-chunk") if "--prefill-chunk" in benchmark_command else -1
    if chunk_index < 0 or chunk_index + 1 >= len(benchmark_command):
        raise ValueError("source benchmark command lacks --prefill-chunk")
    if benchmark_command[chunk_index + 1] != str(expected_prefill_chunk):
        raise ValueError("source benchmark command is not bound to the selected prefill chunk")
    _replace_option(benchmark_command, "--whole-pg", f"{prompt_tokens},{generated_tokens}")
    _replace_option(benchmark_command, "-r", "1")
    _replace_option(benchmark_command, "--output-file", str(out_dir / "benchmark-report.json"))
    if "--profile-measured" not in benchmark_command:
        benchmark_command.append("--profile-measured")

    return _write_plan(
        out_dir,
        kind=kind,
        question=question,
        benchmark_command=benchmark_command,
        power_profile_path=power_profile_path,
        source_matrix={
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "preset": manifest["preset"],
            "report": {"path": str(source_report_path), "sha256": file_sha256(source_report_path)},
        },
        artifact=artifact,
        bench=bench,
        workload={
            "concurrency": concurrency,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "spec": source_report["config"].get("spec"),
            "draft_tokens": source_report["config"].get("draft_tokens"),
            "dflash_verify_width": source_report["config"].get("dflash_verify_width"),
            "kv_value_group": expected_kv_value_group,
            "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
            "xattention_profile": xattention_profile,
            "prefill_chunk": expected_prefill_chunk,
        },
        kernel_include_regex=kernel_include_regex,
    )


def _canonical_evaluation(value: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(value))
    manifest = normalized.get("manifest")
    if isinstance(manifest, dict) and isinstance(manifest.get("path"), str):
        manifest["path"] = str(Path(manifest["path"]).resolve())
    selection = normalized.get("terminal_selection")
    if isinstance(selection, dict) and isinstance(selection.get("path"), str):
        selection["path"] = str(Path(selection["path"]).resolve())
    ladder = normalized.get("ladder")
    if isinstance(ladder, list):
        for row in ladder:
            report = row.get("report") if isinstance(row, dict) else None
            if isinstance(report, dict) and isinstance(report.get("path"), str):
                report["path"] = str(Path(report["path"]).resolve())
    return normalized


def prepare_low_context(
    manifest_path: Path,
    evaluation_path: Path,
    terminal_selection_path: Path,
    out_dir: Path,
    *,
    kind: str,
    question: str,
    expected_executable: Path,
    expected_artifact: Path,
    expected_weights_id: str,
    expected_kv_value_group: int,
    expected_prefill_chunk: int,
    kernel_include_regex: str | None = None,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("a named attribution question is required")
    if expected_kv_value_group not in (16, 32):
        raise ValueError("selected KV value group must be 16 or 32")
    if expected_prefill_chunk not in SUPPORTED_PREFILL_CHUNKS:
        raise ValueError("selected prefill chunk is unsupported")
    if not expected_weights_id:
        raise ValueError("selected weights identity is required")

    evaluation_path = evaluation_path.resolve(strict=True)
    evaluation_bytes = evaluation_path.read_bytes()
    evaluation = json.loads(evaluation_bytes.decode("utf-8"))
    if (
        not isinstance(evaluation, dict)
        or evaluation.get("artifact_type") != "ninfer_r9700_low_context_prefill_evaluation"
        or type(evaluation.get("schema_version")) is not int
        or evaluation["schema_version"] != 1
        or type(evaluation.get("minimum_p2048_tok_s")) not in (int, float)
    ):
        raise ValueError("low-context evaluation is not a schema-v1 evaluation authority")
    recomputed = validate_ladder(
        manifest_path,
        float(evaluation["minimum_p2048_tok_s"]),
        expected_executable,
        expected_artifact,
        terminal_selection_path,
        # Preparation validates the retained pre/post-auto authority. Each generated command
        # independently guards the live profile required by its trace or PMC mode.
        power_reader=lambda _path: "auto",
    )
    if _canonical_evaluation(evaluation) != _canonical_evaluation(recomputed):
        raise ValueError("low-context evaluation differs from a fresh validation of its inputs")

    manifest_path = manifest_path.resolve(strict=True)
    terminal_selection_path = terminal_selection_path.resolve(strict=True)
    manifest = _load_json(manifest_path)
    artifact = manifest["artifact"]
    bench = manifest["bench"]
    corpus_path = Path(str(manifest.get("corpus", ""))).resolve(strict=True)
    corpus_hash = file_sha256(corpus_path)
    corpus_size = corpus_path.stat().st_size
    corpus_tokens = manifest.get("corpus_tokens")
    if (
        manifest.get("corpus_sha256") != corpus_hash
        or type(corpus_tokens) is not int
        or corpus_tokens <= 0
    ):
        raise ValueError("low-context authority has invalid corpus provenance")
    if (
        artifact.get("weights_id") != expected_weights_id
        or manifest.get("expected_kv_value_group") != expected_kv_value_group
        or manifest.get("selected_prefill_chunk") != expected_prefill_chunk
        or manifest.get("expected_xattention_profile") != "dense"
    ):
        raise ValueError("low-context authority is not the explicitly selected dense route")

    records = [
        record for record in manifest.get("commands", [])
        if isinstance(record, dict)
        and record.get("suite") == "low_context_prefill"
        and record.get("case") == "prefill_p2048_dense_none"
        and record.get("concurrency") == 1
    ]
    if len(records) != 1:
        raise ValueError("low-context authority has no unique dense C1 P2048 command")
    source_record = records[0]
    source_report_path = Path(str(source_record.get("report", ""))).resolve(strict=True)
    source_report = _load_json(source_report_path)
    config = source_report.get("config")
    tests = source_report.get("tests")
    if (
        source_report.get("artifact_type") != "ninfer_bench_report"
        or type(source_report.get("schema_version")) is not int
        or source_report["schema_version"] != REPORT_SCHEMA_VERSION
        or not isinstance(config, dict)
        or type(config.get("concurrency")) is not int
        or config["concurrency"] != 1
        or type(config.get("prefill_chunk")) is not int
        or config["prefill_chunk"] != expected_prefill_chunk
        or type(config.get("kv_value_group")) is not int
        or config["kv_value_group"] != expected_kv_value_group
        or config.get("spec") != "none"
        or type(config.get("draft_tokens")) is not int
        or config["draft_tokens"] != 0
        or type(config.get("dflash_verify_width_requested")) is not int
        or config["dflash_verify_width_requested"] != 0
        or type(config.get("dflash_verify_width")) is not int
        or config["dflash_verify_width"] != 0
        or config.get("xattention_qualification") is not False
        or any(key in config for key in (
            "xattention_profile", "xattention_find_block", "xattention_stride",
            "xattention_tau_permille",
        ))
        or type(config.get("repetitions")) is not int
        or config["repetitions"] != 3
        or type(config.get("warmup")) is not int
        or config["warmup"] != 1
        or not isinstance(tests, list)
        or len(tests) != 1
        or not isinstance(tests[0], dict)
        or tests[0].get("kind") != "pp"
        or type(tests[0].get("n_prompt")) is not int
        or tests[0]["n_prompt"] != 2048
        or type(tests[0].get("n_gen")) is not int
        or tests[0]["n_gen"] != 0
    ):
        raise ValueError("low-context source report is not exact dense C1 spec-none P2048 3/1")

    source_command = source_record.get("command")
    if not isinstance(source_command, list) or not all(
        isinstance(item, str) for item in source_command
    ):
        raise ValueError("low-context source command is invalid")
    benchmark_command = list(source_command)
    if benchmark_command.count("-r") != 1 or benchmark_command.count("--output-file") != 1:
        raise ValueError("low-context source command has ambiguous mutable options")
    if "--profile-measured" in benchmark_command:
        raise ValueError("low-context source command is already profiler-instrumented")
    _replace_option(benchmark_command, "-r", "1")
    _replace_option(benchmark_command, "--output-file", str(out_dir.resolve() / "benchmark-report.json"))
    benchmark_command.append("--profile-measured")

    manifest_hash = file_sha256(manifest_path)
    report_hash = file_sha256(source_report_path)
    evaluation_hash = hashlib.sha256(evaluation_bytes).hexdigest()
    selection_hash = file_sha256(terminal_selection_path)
    bound_manifest = recomputed.get("manifest")
    bound_terminal = recomputed.get("terminal_selection")
    bound_reports = [
        row.get("report") for row in recomputed.get("ladder", [])
        if isinstance(row, dict) and row.get("prompt_tokens") == 2048
    ]
    if (
        not isinstance(bound_manifest, dict)
        or Path(str(bound_manifest.get("path", ""))).resolve() != manifest_path
        or bound_manifest.get("sha256") != manifest_hash
        or not isinstance(bound_terminal, dict)
        or Path(str(bound_terminal.get("path", ""))).resolve() != terminal_selection_path
        or bound_terminal.get("sha256") != selection_hash
        or len(bound_reports) != 1
        or not isinstance(bound_reports[0], dict)
        or Path(str(bound_reports[0].get("path", ""))).resolve() != source_report_path
        or bound_reports[0].get("sha256") != report_hash
    ):
        raise ValueError("low-context profiling inputs differ from the recomputed authority chain")
    if (
        file_sha256(evaluation_path) != evaluation_hash
        or file_sha256(manifest_path) != manifest_hash
        or file_sha256(source_report_path) != report_hash
        or file_sha256(terminal_selection_path) != selection_hash
        or file_sha256(corpus_path) != corpus_hash
        or corpus_path.stat().st_size != corpus_size
    ):
        raise ValueError("low-context profiling authority changed while preparing")
    _identity_matches_file(bench, "benchmark executable")
    _identity_matches_file(artifact, "artifact")
    if (
        file_sha256(evaluation_path) != evaluation_hash
        or file_sha256(manifest_path) != manifest_hash
        or file_sha256(source_report_path) != report_hash
        or file_sha256(terminal_selection_path) != selection_hash
        or file_sha256(corpus_path) != corpus_hash
        or corpus_path.stat().st_size != corpus_size
    ):
        raise ValueError("low-context profiling authority changed while preparing")

    power_profile_path = Path(str(manifest["power_profile"]["sysfs_path"]))
    return _write_plan(
        out_dir.resolve(),
        kind=kind,
        question=question,
        benchmark_command=benchmark_command,
        power_profile_path=power_profile_path,
        source_matrix={
            "path": str(manifest_path),
            "sha256": manifest_hash,
            "preset": "low-context-prefill",
            "report": {"path": str(source_report_path), "sha256": report_hash},
        },
        artifact=artifact,
        bench=bench,
        workload={
            "concurrency": 1,
            "prompt_tokens": 2048,
            "generated_tokens": 0,
            "spec": "none",
            "draft_tokens": 0,
            "dflash_verify_width": 0,
            "kv_value_group": expected_kv_value_group,
            "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
            "xattention_profile": "dense",
            "prefill_chunk": expected_prefill_chunk,
        },
        kernel_include_regex=kernel_include_regex,
        extra_provenance={
            "corpus": {
                "path": str(corpus_path),
                "sha256": corpus_hash,
                "file_size_bytes": corpus_size,
                "tokens": corpus_tokens,
            },
            "low_context_evaluation": {
                "path": str(evaluation_path),
                "sha256": evaluation_hash,
                "minimum_p2048_tok_s": evaluation["minimum_p2048_tok_s"],
                "observed_p2048_tok_s": evaluation["observed_p2048_tok_s"],
                "passes_p2048_gate": evaluation["passes_p2048_gate"],
            },
            "terminal_selection": recomputed["terminal_selection"],
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--matrix-dir", type=Path)
    source.add_argument("--low-context-manifest", type=Path)
    parser.add_argument("--low-context-evaluation", type=Path)
    parser.add_argument("--terminal-selection", type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--generated-tokens", type=int, default=256)
    parser.add_argument("--kind", required=True, choices=("trace", "dispatch-pmc"))
    parser.add_argument("--question", required=True)
    parser.add_argument("--expected-weights-id", required=True)
    parser.add_argument("--expected-kv-value-group", required=True, type=int, choices=(16, 32))
    parser.add_argument(
        "--expected-xattention-profile", required=True,
        choices=tuple(sorted(SUPPORTED_XATTENTION_PROFILES)),
    )
    parser.add_argument(
        "--expected-prefill-chunk", required=True, type=int,
        choices=tuple(sorted(SUPPORTED_PREFILL_CHUNKS)),
    )
    parser.add_argument("--kernel-include-regex")
    parser.add_argument("--expected-dflash-draft-tokens", type=int)
    parser.add_argument("--expected-dflash-verify-width", type=int)
    args = parser.parse_args(argv)
    try:
        if args.low_context_manifest is not None:
            if any(value is None for value in (
                args.low_context_evaluation, args.terminal_selection,
                args.executable, args.artifact,
            )):
                raise ValueError(
                    "low-context profiling requires --low-context-evaluation, "
                    "--terminal-selection, --executable, and --artifact"
                )
            if (
                args.concurrency != 1
                or args.prompt_tokens != 2048
                or args.generated_tokens != 0
                or args.expected_xattention_profile != "dense"
                or args.expected_dflash_draft_tokens is not None
                or args.expected_dflash_verify_width is not None
            ):
                raise ValueError(
                    "low-context profiling is fixed to dense C1 P2048 prefill-only spec-none"
                )
            payload = prepare_low_context(
                args.low_context_manifest,
                args.low_context_evaluation,
                args.terminal_selection,
                args.out,
                kind=args.kind,
                question=args.question,
                expected_executable=args.executable,
                expected_artifact=args.artifact,
                expected_weights_id=args.expected_weights_id,
                expected_kv_value_group=args.expected_kv_value_group,
                expected_prefill_chunk=args.expected_prefill_chunk,
                kernel_include_regex=args.kernel_include_regex,
            )
        else:
            if any(value is not None for value in (
                args.low_context_evaluation, args.terminal_selection,
                args.executable, args.artifact,
            )):
                raise ValueError("low-context authority options require --low-context-manifest")
            payload = prepare(
                args.matrix_dir, args.out, concurrency=args.concurrency,
                prompt_tokens=args.prompt_tokens, generated_tokens=args.generated_tokens,
                kind=args.kind, question=args.question,
                expected_weights_id=args.expected_weights_id,
                expected_kv_value_group=args.expected_kv_value_group,
                expected_xattention_profile=args.expected_xattention_profile,
                expected_prefill_chunk=args.expected_prefill_chunk,
                kernel_include_regex=args.kernel_include_regex,
                expected_dflash_draft_tokens=args.expected_dflash_draft_tokens,
                expected_dflash_verify_width=args.expected_dflash_verify_width,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"wrote {payload['profile_kind']} command plan to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
