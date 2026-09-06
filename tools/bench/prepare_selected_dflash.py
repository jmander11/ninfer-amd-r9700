#!/usr/bin/env python3
"""Prepare and advance the exact terminal-winner DFlash2 selection campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from tools.artifact.container import Artifact
from tools.bench.assemble_dflash_selection import _matrix, _records
from tools.bench.run_ninfer_bench_matrix import inspect_artifact, inspect_executable
from tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 import preflight, preflight_summary

REPO = Path(__file__).resolve().parents[2]
RESOLVER = REPO / "profiles/bench/post-terminal-focused-verification-20260905/resolve.py"
RUNNER = REPO / "tools/bench/run_ninfer_bench_matrix.py"
ASSEMBLER = REPO / "tools/bench/assemble_dflash_selection.py"
CONVERTER = REPO / "tools/convert/qwen3_8_27b_r9700/convert_dflash2_q4.py"
DFLASH_INVENTORY = REPO / "tools/convert/qwen3_8_27b_r9700/dflash2_q4_inventory.py"
SEMANTIC_AUTHORITIES = (
    REPO / "tools/bench/matrix_contract.py",
    REPO / "tools/bench/prefill_chunk_authority.py",
    REPO / "tools/bench/select_prefill_chunk.py",
    REPO / "tools/ppl/run.py",
    REPO / "tools/ppl/assemble_pareto.py",
    REPO / "tools/ppl/pareto.py",
    REPO / "tools/ppl/validate_fp8_hybrid_execution_gate.py",
)
DFLASH_SOURCE = Path("/ssdpool2nvme/local_llm/models/qwen3.8-27b-dflash2")
CONVERSION_PYTHON = Path("/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python")
CONVERSION_LD_LIBRARY_PATH = "/opt/rocm/lib:/opt/rocm/core-10.0/lib"
RUNTIME_AUTHORITIES = (
    REPO / "src/targets/qwen3_8_27b/impl/config.h",
    REPO / "src/targets/qwen3/impl/runtime/dflash_context_impl.h",
    REPO / "src/targets/qwen3/impl/runtime/layouts_impl.h",
)
BENCH_AUTHORITIES = (
    REPO / "bench/targets/qwen3_8_27b/ninfer_bench.cpp",
    REPO / "bench/targets/qwen3_8_27b/ninfer_bench_support.cpp",
    REPO / "bench/targets/qwen3_8_27b/ninfer_bench_support.h",
)
COMPANIONS = {
    "r9700-q4g64-n16k16-eval": ("all-q4", "r9700-q4g64-n16k16-dflash2-q4-eval",
                          "qwen3.8-27b-r9700-q4g64-n16k16-dflash2-q4-eval.ninfer"),
    "r9700-q4-w8-mse-n16k16-eval": ("mixed", "r9700-q4-w8-mse-n16k16-dflash2-q4-eval",
                               "qwen3.8-27b-r9700-q4-w8-mse-n16k16-dflash2-q4-eval.ninfer"),
    "r9700-q4g64-f8e4m3-four-role-n16k16-eval": (
        "four-role", "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval",
        "qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer"),
}


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _conversion_python_identity() -> dict:
    """Bind the one interpreter/environment capable of the ROCm conversion."""

    if (
        not CONVERSION_PYTHON.is_absolute()
        or not CONVERSION_PYTHON.is_file()
        or not os.access(CONVERSION_PYTHON, os.X_OK)
    ):
        raise ValueError("selected DFlash conversion Python is unavailable")
    pyvenv = CONVERSION_PYTHON.parent.parent / "pyvenv.cfg"
    if not pyvenv.is_file():
        raise ValueError("selected DFlash conversion Python lacks pyvenv.cfg")
    probe = (
        "import json, sys, torch, safetensors; "
        "print(json.dumps({'runtime_executable': sys.executable, "
        "'python_version': '.'.join(map(str, sys.version_info[:3])), "
        "'torch': torch.__version__, 'torch_hip': torch.version.hip, "
        "'safetensors': safetensors.__version__}))"
    )
    environment = {**os.environ, "LD_LIBRARY_PATH": CONVERSION_LD_LIBRARY_PATH}
    try:
        completed = subprocess.run(
            [str(CONVERSION_PYTHON), "-c", probe], capture_output=True, text=True,
            check=True, env=environment,
        )
        runtime = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ValueError(
            "selected DFlash conversion Python cannot import ROCm Torch and safetensors"
        ) from error
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {
            "runtime_executable", "python_version", "torch", "torch_hip", "safetensors",
        }
        or not all(isinstance(runtime.get(key), str) and runtime[key] for key in runtime)
        or runtime.get("runtime_executable") != str(CONVERSION_PYTHON)
    ):
        raise ValueError("selected DFlash conversion Python returned malformed identity")
    return {
        "launcher_path": str(CONVERSION_PYTHON),
        "launcher_sha256": sha(CONVERSION_PYTHON),
        "pyvenv_cfg": {"path": str(pyvenv), "sha256": sha(pyvenv)},
        "environment": {"LD_LIBRARY_PATH": CONVERSION_LD_LIBRARY_PATH},
        **runtime,
    }


def _base_migration_authority(artifact: dict) -> dict:
    receipt = artifact.get("conversion_receipt")
    from tools.ppl.run import validate_n16_receipt_summary
    validate_n16_receipt_summary(receipt, artifact.get("weights_id"))
    fields = ("recipe_id", "object_plan_sha256", "source_artifact_sha256",
              "source_receipt_sha256", "transcoder_sha256")
    if (not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str)
            or not isinstance(receipt.get("sha256"), str)
            or any(not isinstance(receipt.get(name), str) for name in fields)):
        raise ValueError("selected N16 base lacks its migration authority")
    result = {"receipt": {"path": receipt["path"], "sha256": receipt["sha256"]},
              **{name: receipt[name] for name in fields}}
    if artifact.get("weights_id") == "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
        hybrid = ("selection_sha256", "source_index_sha256", "source_ranking_sha256")
        if any(not isinstance(receipt.get(name), str) for name in hybrid):
            raise ValueError("selected hybrid base lacks its selection authority")
        result.update({name: receipt[name] for name in hybrid})
    else:
        if not isinstance(receipt.get("receipt_producer_sha256"), str):
            raise ValueError("selected N16 base lacks its receipt producer authority")
        result["receipt_producer_sha256"] = receipt["receipt_producer_sha256"]
    return result


def resolve(selection: Path) -> dict:
    spec = importlib.util.spec_from_file_location("ninfer_terminal_route", RESOLVER)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load terminal route resolver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve(selection)


def _common(plan: dict, preset: str, out: Path, k: int | None = None,
            w: int | None = None) -> list[str]:
    command = [
        "python3", str(RUNNER), "--preset", preset,
        "--prefill-chunk", str(plan["selected_prefill_chunk"]),
        "--bench", plan["benchmark"]["path"], "--no-build",
        "--weights", plan["companion"]["path"],
        "--expected-kv-value-group", str(plan["cache_group"]),
        "--expected-q4-activation-bits", "8", "--expected-w8-activation-bits", "8",
        "--expected-fp8-qk-wmma", "1", "--expected-xattention-profile",
        plan["text_prefill_attention_profile"], "--output-dir", str(out),
    ]
    if plan["recipe"] == "four-role":
        tool = plan.get("hybrid_width_tool")
        if not isinstance(tool, dict) or not isinstance(tool.get("path"), str):
            raise ValueError("selected four-role DFlash campaign lacks its hybrid planner")
        command.extend(["--require-fp8-hybrid", "--hybrid-width-tool", tool["path"]])
    if preset != "dflash-shortlist":
        if k is None or w is None:
            raise ValueError("fixed DFlash K/W are required after shortlist")
        command[4:4] = ["--dflash-draft-tokens", str(k), "--dflash-verify-width", str(w)]
        command[4:4] = [item for concurrency in (1, 2, 3, 4)
                        for item in ("--concurrency", str(concurrency))]
    return command


def _validate_companion(plan: dict) -> None:
    path = Path(plan["companion"]["path"]).resolve(strict=True)
    report_path = Path(plan["companion"]["conversion_report"]).resolve(strict=True)
    artifact = inspect_artifact(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recipe = report.get("dflash_recipe", {})
    source = report.get("dflash_source", {})
    expected_source = plan["dflash_source"]
    base = report.get("base", {})
    expected_authority = _base_migration_authority(plan["base_artifact"])
    checked = preflight(plan["base_artifact"]["path"], plan["dflash_source"]["path"])
    preflight_record = preflight_summary(checked)
    with Artifact.open(path) as opened:
        if (opened.identity != checked.output_identity or opened.objects != checked.objects
                or len(opened.objects) != 1190):
            raise ValueError("DFlash companion directory differs from exact conversion preflight")
    if (
        artifact["weights_id"] != plan["companion"]["weights_id"]
        or report.get("status") != "registered-evaluation-only"
        or report.get("base", {}).get("identity", {}).get("weights_id")
        != plan["base_artifact"]["weights_id"]
        or report.get("base", {}).get("sha256") != plan["base_artifact"]["sha256"]
        or base.get("bytes") != plan["base_artifact"]["file_size_bytes"]
        or Path(str(base.get("path", ""))).resolve()
        != Path(plan["base_artifact"]["path"]).resolve()
        or base.get("payload_copy") != "byte_exact"
        or base.get("authority") != expected_authority
        or base.get("authority") != preflight_record["base"]["authority"]
        or report.get("artifact", {}).get("sha256") != artifact["sha256"]
        or report.get("artifact", {}).get("bytes") != artifact["file_size_bytes"]
        or Path(str(report.get("artifact", {}).get("path", ""))).resolve() != path
        or recipe.get("matrix_format") != "Q4G64_F16S"
        or recipe.get("activation_profile") != "compile_selected_adaptive_A8G64"
        or recipe.get("selector_codebook_format") != "BF16"
        or recipe.get("objects") != 66 or recipe.get("source_tensors") != 81
        or recipe.get("format_counts") != {"BF16": 34, "Q4G64_F16S": 32}
        or recipe.get("format_encoded_bytes")
        != {"BF16": 254_814_720, "Q4G64_F16S": 954_654_720}
        or recipe.get("tensor_encoded_bytes") != 1_209_469_440
        or recipe.get("runtime_repack") is not False
        or source.get("safetensors_sha256") != expected_source["model_sha256"]
        or source.get("safetensors_bytes") != expected_source["model_bytes"]
        or source.get("config_sha256") != expected_source["config_sha256"]
        or source.get("readme_sha256") != expected_source["readme_sha256"]
        or path.stat().st_size != checked.projected_file_bytes
        or report.get("artifact", {}).get("projected_bytes") != checked.projected_file_bytes
        or report.get("artifact", {}).get("projected_device_arena_bytes")
        != checked.projected_device_arena_bytes
    ):
        raise ValueError("DFlash companion conversion does not match the terminal base")


def _companion_conversion_command(plan: dict, *, companion_exists: bool) -> list[str]:
    """Build the one selected winner's current-N16 companion conversion command."""

    command = [
        "env", f"LD_LIBRARY_PATH={plan['conversion_python']['environment']['LD_LIBRARY_PATH']}",
        plan["conversion_python"]["launcher_path"], "-m",
        "tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4",
        "--base", plan["base_artifact"]["path"],
        "--dflash-model", plan["dflash_source"]["path"],
        "--out", plan["companion"]["path"],
    ]
    if companion_exists:
        command.append("--finalize-report")
    else:
        command.extend(["--device", "cuda"])
    return command


def _publish_text(path: Path, payload: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_script(path: Path, commands: list[list[str]], tail: list[str] | None = None) -> None:
    tail = [] if tail is None else tail
    payload = (
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" +
        "\n".join(shlex.join(command) for command in commands) +
        ("\n" if commands else "") + "\n".join(tail) + "\n"
    )
    if os.path.lexists(path):
        if path.is_file() and path.read_text(encoding="utf-8") == payload:
            return
        raise ValueError(f"refusing to overwrite {path}")
    _publish_text(path, payload)


def _matrix_shell(command: list[str], output: Path) -> str:
    resumed = [*command, "--resume"]
    return (
        f"if [[ -L {shlex.quote(str(output))} ]]; then echo 'refusing symlink output' >&2; exit 1; "
        f"elif [[ -e {shlex.quote(str(output))} ]]; then "
        f"{shlex.join(resumed)}; else {shlex.join(command)}; fi"
    )


def _write_closure(path: Path, files: list[Path]) -> None:
    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite {path}")
    rows = []
    for item in files:
        try:
            name = item.relative_to(REPO)
        except ValueError:
            name = item
        rows.append(f"{sha(item)}  {name}\n")
    _publish_text(path, "".join(rows))


def _frontier_profiles(root: Path, prefill_chunk: int) -> list[tuple[int, int]]:
    _matrix(root, "dflash-shortlist", prefill_chunk=prefill_chunk)
    shortlist = json.loads((root / "dflash-shortlist.json").read_text())
    rows = shortlist.get("non_dominated_candidates")
    profiles = sorted({
        (row.get("draft_tokens"), row.get("verify_width_resolved"))
        for row in rows if isinstance(row, dict)
    }) if isinstance(rows, list) else []
    if (
        not profiles or len(profiles) != len(rows)
        or any(type(k) is not int or type(w) is not int for k, w in profiles)
    ):
        raise ValueError("shortlist has no exact unique numeric frontier")
    return profiles


def prepare(selection: Path, out: Path) -> dict:
    selection = selection.resolve(strict=True)
    out = out.resolve()
    if os.path.lexists(out):
        raise ValueError(f"output directory already exists: {out}")
    route = resolve(selection)
    base_id = route["artifact"]["weights_id"]
    if base_id not in COMPANIONS:
        raise ValueError("terminal base has no registered DFlash2 companion")
    recipe, companion_id, filename = COMPANIONS[base_id]
    companion = (REPO / "out" / filename).resolve()
    conversion = Path(str(companion) + ".conversion.json")
    source_tensor = DFLASH_SOURCE / "model.safetensors"
    source_config = DFLASH_SOURCE / "config.json"
    source_readme = DFLASH_SOURCE / "README.md"
    if not all(path.is_file() for path in (source_tensor, source_config, source_readme)):
        raise ValueError("fixed DFlash2 source tensor is unavailable")
    plan = {
        "artifact_type": "ninfer_r9700_selected_dflash_campaign_plan",
        "schema_version": 1, "status": "prepared_not_executed",
        "terminal_selection": {"path": str(selection), "sha256": sha(selection)},
        "winner": route["winner"], "base_artifact": route["artifact"],
        "benchmark": route["benchmark"], "cache_group": route["cache_profile"]["value_group"],
        "text_prefill_attention_profile": route["execution_profile"]["xattention_profile"],
        "dflash_attention_profiles": {"proposal": "dense", "target_verification": "dense"},
        "selected_prefill_chunk": route["selected_prefill_chunk"],
        "recipe": recipe,
        "hybrid_width_tool": route.get("hybrid_width_tool"),
        "companion": {"path": str(companion), "weights_id": companion_id,
                      "conversion_report": str(conversion)},
        "dflash_source": {
            "path": str(DFLASH_SOURCE), "model_bytes": source_tensor.stat().st_size,
            "model_sha256": sha(source_tensor), "config_sha256": sha(source_config),
            "readme_sha256": sha(source_readme),
        },
        "conversion_python": _conversion_python_identity(),
        "runtime_authorities": [
            {"path": str(path.resolve()), "sha256": sha(path)} for path in RUNTIME_AUTHORITIES
        ],
        "persistent_recipe": {"matrix": "Q4G64_F16S with dynamic A8G64 activations",
                              "non_matrix_and_selector_codebook": "source BF16",
                              "separate_runtime_state": "private fixed BF16",
                              "runtime_weight_repack": False},
        "maximum_concurrency": 4,
        "stages": ["companion", "shortlist", "frontier-capacity", "eligible-pareto", "schema-v3-selection"],
    }
    plan_path = out / "plan.json"
    commands: list[list[str]] = []
    companion_exists = companion.is_file()
    conversion_exists = conversion.is_file()
    if not companion_exists or not conversion_exists:
        if conversion_exists:
            raise ValueError("selected companion is missing but its conversion report exists")
        commands.append(_companion_conversion_command(
            plan, companion_exists=companion_exists
        ))
    else:
        _validate_companion(plan)
    out.mkdir(parents=True)
    _publish_text(plan_path, json.dumps(plan, indent=2) + "\n")
    shortlist = out / "shortlist"
    commands.insert(0, ["sha256sum", "--check", "--strict", str(out / "prepared.sha256")])
    tail = [
        shlex.join(["python3", "-m", "tools.bench.prepare_selected_dflash", "validate-plan",
                    "--plan", str(plan_path)]),
        _matrix_shell(_common(plan, "dflash-shortlist", shortlist), shortlist),
        shlex.join(["python3", "-m", "tools.bench.prepare_selected_dflash", "advance-capacity",
                    "--plan", str(plan_path)]),
        shlex.join(["bash", str(out / "capacity-commands.sh")]),
        shlex.join(["python3", "-m", "tools.bench.prepare_selected_dflash", "advance-pareto",
                    "--plan", str(plan_path)]),
        shlex.join(["bash", str(out / "pareto-and-select.sh")]),
    ]
    _write_script(out / "commands.sh", commands, tail)
    _write_closure(out / "prepared.sha256", [
        Path(__file__).resolve(), ASSEMBLER, RUNNER, CONVERTER, DFLASH_INVENTORY,
        RESOLVER, selection, plan_path, out / "commands.sh",
        source_tensor, source_config, source_readme, CONVERSION_PYTHON,
        CONVERSION_PYTHON.parent.parent / "pyvenv.cfg", *SEMANTIC_AUTHORITIES,
        *RUNTIME_AUTHORITIES,
        *BENCH_AUTHORITIES,
    ])
    return plan


def _load_plan(path: Path) -> tuple[dict, Path]:
    path = path.resolve(strict=True)
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("artifact_type") != "ninfer_r9700_selected_dflash_campaign_plan":
        raise ValueError("not a selected DFlash campaign plan")
    selection = Path(plan["terminal_selection"]["path"]).resolve(strict=True)
    route = resolve(selection)
    source_tensor = Path(plan["dflash_source"]["path"]) / "model.safetensors"
    source_config = Path(plan["dflash_source"]["path"]) / "config.json"
    source_readme = Path(plan["dflash_source"]["path"]) / "README.md"
    conversion_python = _conversion_python_identity()
    base_id = route["artifact"]["weights_id"]
    if base_id not in COMPANIONS:
        raise ValueError("terminal base has no registered DFlash2 companion")
    recipe, companion_id, filename = COMPANIONS[base_id]
    companion_path = (REPO / "out" / filename).resolve()
    expected_companion = {
        "path": str(companion_path), "weights_id": companion_id,
        "conversion_report": str(Path(str(companion_path) + ".conversion.json")),
    }
    if (
        plan.get("schema_version") != 1 or plan.get("status") != "prepared_not_executed"
        or sha(selection) != plan["terminal_selection"]["sha256"]
        or route["winner"] != plan["winner"]
        or route["artifact"] != plan["base_artifact"]
        or route["benchmark"] != plan["benchmark"]
        or route["cache_profile"]["value_group"] != plan["cache_group"]
        or route["execution_profile"]["xattention_profile"]
        != plan["text_prefill_attention_profile"]
        or route["selected_prefill_chunk"] != plan["selected_prefill_chunk"]
        or route.get("hybrid_width_tool") != plan.get("hybrid_width_tool")
        or plan.get("recipe") != recipe
        or plan.get("companion") != expected_companion
        or plan.get("dflash_attention_profiles")
        != {"proposal": "dense", "target_verification": "dense"}
        or plan.get("persistent_recipe") != {
            "matrix": "Q4G64_F16S with dynamic A8G64 activations",
            "non_matrix_and_selector_codebook": "source BF16",
            "separate_runtime_state": "private fixed BF16",
            "runtime_weight_repack": False,
        }
        or plan.get("maximum_concurrency") != 4
        or plan.get("stages") != [
            "companion", "shortlist", "frontier-capacity", "eligible-pareto",
            "schema-v3-selection",
        ]
        or Path(plan["dflash_source"]["path"]).resolve() != DFLASH_SOURCE.resolve()
        or sha(source_tensor) != plan["dflash_source"]["model_sha256"]
        or source_tensor.stat().st_size != plan["dflash_source"]["model_bytes"]
        or sha(source_config) != plan["dflash_source"]["config_sha256"]
        or sha(source_readme) != plan["dflash_source"]["readme_sha256"]
        or plan.get("runtime_authorities") != [
            {"path": str(authority.resolve()), "sha256": sha(authority)}
            for authority in RUNTIME_AUTHORITIES
        ]
        or plan.get("conversion_python") != conversion_python
    ):
        raise ValueError("terminal winner changed after DFlash preparation")
    _validate_companion(plan)
    if inspect_executable(Path(plan["benchmark"]["path"])) != plan["benchmark"]:
        raise ValueError("selected benchmark changed")
    return plan, path.parent


def advance_capacity(path: Path) -> None:
    plan, root = _load_plan(path)
    shortlist_root = root / "shortlist"
    profiles = _frontier_profiles(shortlist_root, plan["selected_prefill_chunk"])
    commands = [_common(plan, "dflash-capacity", root / f"capacity-k{k}-w{w}", k, w)
                for k, w in profiles]
    # A capacity exclusion is expected evidence, so continue only when the runner retained failures.
    shell = [["bash", "-c", _matrix_shell(command, root / f"capacity-k{k}-w{w}") + " || test -f "
              + shlex.quote(str(root / f"capacity-k{k}-w{w}/failures.json"))]
             for command, (k, w) in zip(commands, profiles, strict=True)]
    _write_script(root / "capacity-commands.sh", shell)
    frontier_path = root / "frontier.json"
    payload = json.dumps({"profiles": profiles}, indent=2) + "\n"
    if os.path.lexists(frontier_path):
        if not frontier_path.is_file() or frontier_path.read_text() != payload:
            raise ValueError("refusing to overwrite changed shortlist frontier")
    else:
        _publish_text(frontier_path, payload)


def advance_pareto(path: Path) -> None:
    plan, root = _load_plan(path)
    profiles = [tuple(row) for row in json.loads((root / "frontier.json").read_text())["profiles"]]
    if profiles != _frontier_profiles(root / "shortlist", plan["selected_prefill_chunk"]):
        raise ValueError("retained frontier differs from the recomputed shortlist")
    eligible = []
    for k, w in profiles:
        cap = root / f"capacity-k{k}-w{w}"
        manifest = _matrix(cap, "dflash-capacity", k, w,
                           prefill_chunk=plan["selected_prefill_chunk"], allow_failures=True)
        reports = _records(cap, manifest, "dflash-capacity", k, w,
                           plan["selected_prefill_chunk"], allow_missing=True)
        if not (cap / "failures.json").exists() and set(reports) == {1, 2, 3, 4}:
            eligible.append((k, w))
    if not eligible:
        raise ValueError("no shortlist-frontier K/W has complete C1..4 capacity")
    commands = [_common(plan, "dflash-pareto", root / f"pareto-k{k}-w{w}", k, w)
                for k, w in eligible]
    selection = root / "selection.json"
    assemble = ["python3", str(ASSEMBLER), "--base-selection",
                plan["terminal_selection"]["path"], "--conversion-report",
                plan["companion"]["conversion_report"], "--shortlist-dir", str(root / "shortlist")]
    for k, w in profiles:
        assemble += ["--capacity", str(k), str(w), str(root / f"capacity-k{k}-w{w}")]
    for k, w in eligible:
        assemble += ["--pareto", str(k), str(w), str(root / f"pareto-k{k}-w{w}")]
    assemble += ["--out", str(selection)]
    tail = [
        *(_matrix_shell(command, root / f"pareto-k{k}-w{w}")
          for command, (k, w) in zip(commands, eligible, strict=True)),
        shlex.join(assemble),
    ]
    _write_script(root / "pareto-and-select.sh", [], tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("prepare")
    first.add_argument("--selection", type=Path, required=True)
    first.add_argument("--out", type=Path, required=True)
    for name in ("validate-plan", "advance-capacity", "advance-pareto"):
        stage = sub.add_parser(name)
        stage.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare": prepare(args.selection, args.out)
        elif args.command == "validate-plan": _load_plan(args.plan)
        elif args.command == "advance-capacity": advance_capacity(args.plan)
        else: advance_pareto(args.plan)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
