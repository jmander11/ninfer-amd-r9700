#!/usr/bin/env python3
"""Prepare recipe-separated DFlash evaluation after terminal base/chunk selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from tools.bench.assemble_dflash_selection import (
    RECIPES, DFLASH_PRODUCTION_PROFILES, selected_base_route, benchmark_profile,
    recipe_evidence, capacity_cells, performance_cells, candidate_key,
)
from tools.bench.run_ninfer_bench_matrix import build_hybrid_shared_workspace_authority
from tools.convert.qwen3_8_27b_r9700.dflash2_q4_inventory import companion_weights_id

REPO = Path(__file__).resolve().parents[2]
DFLASH_SOURCE = Path("/ssdpool2nvme/local_llm/models/qwen3.8-27b-dflash2")
CONVERSION_PYTHON = Path("/home/battlefront/.local/bin/python3.11")
MODULE = "tools.bench.prepare_selected_dflash"


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _source_identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "files": {
        name: {"sha256": sha(path / name), "bytes": (path / name).stat().st_size}
        for name in ("model.safetensors", "config.json", "README.md")}}


def _conversion_python_identity(python: Path, pythonpath: str | None) -> dict:
    # CPU packages need not be a ROCm environment. Never query GPU availability.
    python = python.absolute()
    environment = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": pythonpath or ""}
    probe = ("import json,sys,torch,numpy,safetensors; "
             "assert sys.version_info[:2]==(3,11); "
             "assert torch.tensor([1.0],device='cpu').item()==1.; "
             "assert not torch.cuda.is_initialized(); "
             "print(json.dumps({'python':sys.version.split()[0], 'torch':torch.__version__, "
             "'numpy':numpy.__version__, 'safetensors':safetensors.__version__}))")
    result = subprocess.run([str(python), "-c", probe], check=True, capture_output=True,
                            text=True, env={**os.environ, **environment})
    return {"path": str(python), "sha256": sha(python), "environment": environment,
            "packages": json.loads(result.stdout)}


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


def _write_script(path: Path, lines: list[str]) -> None:
    payload = "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(REPO)) + "\n\n"
    payload += "\n".join(lines) + "\n"
    if os.path.lexists(path):
        if path.is_file() and not path.is_symlink() and path.read_text() == payload:
            return
        raise ValueError(f"refusing to overwrite {path}")
    _publish_text(path, payload)


def _python(plan: dict) -> list[str]:
    selected = plan["conversion_python"]
    return ["env", *(f"{key}={value}" for key, value in selected["environment"].items()),
            selected["path"]]


def _stage(plan: dict, root: Path, stage: str) -> str:
    return shlex.join([*_python(plan), "-m", MODULE, stage, "--plan", str(root / "plan.json")])


def _companion_conversion_command(plan: dict, recipe: str) -> list[str]:
    return [*_python(plan), "-m", "tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4",
            "--base", plan["route"]["base_artifact"]["path"], "--dflash-model",
            plan["dflash_source"]["path"], "--out", plan["recipes"][recipe]["path"],
            "--matrix-recipe", recipe, "--device", "cpu"]


def _common(plan: dict, recipe: str, preset: str, out: Path,
            k: int | None = None, w: int | None = None,
            concurrency: list[int] | None = None) -> list[str]:
    route = plan["route"]
    command = [*_python(plan), "-m", "tools.bench.run_ninfer_bench_matrix", "--preset", preset,
        "--prefill-chunk", str(route["selected_prefill_chunk"]), "--bench",
        plan["build"]["benchmark"]["path"], "--no-build", "--weights",
        plan["recipes"][recipe]["path"], "--expected-kv-value-group", str(route["cache_group"]),
        "--expected-q4-activation-bits", "8", "--expected-w8-activation-bits", "8",
        "--expected-fp8-qk-wmma", "1", "--expected-xattention-profile",
        route["text_prefill_attention_profile"], "--output-dir", str(out)]
    if route["hybrid_base_authority"] is not None:
        command += ["--require-fp8-hybrid", "--hybrid-width-tool", plan["planner"]["tool"]["path"]]
    if preset != "dflash-shortlist":
        candidate_key(recipe, k, w)
        if not concurrency or concurrency != sorted(set(concurrency)) or any(c not in (1,2,3,4) for c in concurrency):
            raise ValueError("explicit declared concurrency is required")
        command += ["--dflash-draft-tokens", str(k), "--dflash-verify-width", str(w)]
        command += [item for c in concurrency for item in ("--concurrency", str(c))]
    return command


def _matrix_shell(command: list[str], output: Path) -> str:
    destination = shlex.quote(str(output))
    return (f"if [[ -L {destination} || -e {destination} ]]; then "
            "echo 'refusing existing matrix: preserve prior evidence and use a fresh attempt' >&2; "
            f"exit 1; else {shlex.join(command)}; fi")


def _recipe_paths(root: Path, base_id: str) -> dict:
    return {recipe: {"path": str(root / recipe / "companion.ninfer"),
            "conversion_report": str(root / recipe / "companion.ninfer.conversion.json"),
            "weights_id": companion_weights_id(base_id, recipe)} for recipe in RECIPES}


def _planner(route: dict, bench: Path, tool: Path) -> dict:
    authority = build_hybrid_shared_workspace_authority(tool.resolve(strict=True), bench,
                    [route["selected_prefill_chunk"]], [5, 6])
    base = route["hybrid_base_authority"]
    if base is not None:
        for chunk, inventory in base["inventories_by_prefill_chunk"].items():
            if any(authority["inventories_by_prefill_chunk"][chunk][name] != inventory[name]
                   for name in ("ordinary", "mtp3")):
                raise ValueError("fresh planner differs from selected base width semantics")
    return authority


def prepare(selection: Path, out: Path, *, bench: Path, planner: Path,
            conversion_python: Path = CONVERSION_PYTHON, pythonpath: str | None = None,
            source: Path = DFLASH_SOURCE) -> dict:
    out = out.resolve()
    if os.path.lexists(out):
        raise ValueError(f"output directory already exists: {out}")
    route = selected_base_route(selection.resolve(strict=True))
    build = benchmark_profile(bench, route["cache_group"], route["text_prefill_attention_profile"])
    plan = {"artifact_type": "ninfer_r9700_selected_dflash_campaign_plan", "schema_version": 2,
            "status": "prepared_not_executed", "route": route, "build": build,
            "planner": _planner(route, bench, planner), "dflash_source": _source_identity(source),
            "conversion_python": _conversion_python_identity(conversion_python, pythonpath),
            "recipes": _recipe_paths(out, route["base_artifact"]["weights_id"])}
    out.mkdir(parents=True)
    for recipe in RECIPES:
        (out / recipe).mkdir()
    _publish_text(out / "plan.json", json.dumps(plan, indent=2) + "\n")
    lines = [_stage(plan, out, "validate-plan")]
    for recipe in RECIPES:
        lines += [shlex.join(_companion_conversion_command(plan, recipe))]
        shortlist = out / recipe / "shortlist"
        lines += [_matrix_shell(_common(plan, recipe, "dflash-shortlist", shortlist), shortlist)]
    for stage, script in (("advance-capacity", "capacity-commands.sh"),
                          ("advance-c1", "c1-commands.sh"),
                          ("advance-pareto", "pareto-and-select.sh")):
        lines += [_stage(plan, out, stage), shlex.join(["bash", str(out / script)])]
    _write_script(out / "commands.sh", lines)
    return plan


def _load_plan(path: Path) -> tuple[dict, Path]:
    path = path.resolve(strict=True)
    plan = json.loads(path.read_text())
    if (plan.get("artifact_type") != "ninfer_r9700_selected_dflash_campaign_plan"
            or plan.get("schema_version") != 2 or plan.get("status") != "prepared_not_executed"):
        raise ValueError("not a recipe-aware selected DFlash campaign plan")
    route = selected_base_route(Path(plan["route"]["terminal_selection"]["path"]))
    build = benchmark_profile(Path(plan["build"]["benchmark"]["path"]), route["cache_group"],
                              route["text_prefill_attention_profile"])
    python = plan["conversion_python"]
    if (route != plan["route"] or build != plan["build"]
            or _planner(route, Path(build["benchmark"]["path"]), Path(plan["planner"]["tool"]["path"])) != plan["planner"]
            or _source_identity(Path(plan["dflash_source"]["path"])) != plan["dflash_source"]
            or _conversion_python_identity(Path(python["path"]), python["environment"]["PYTHONPATH"]) != python
            or _recipe_paths(path.parent, route["base_artifact"]["weights_id"]) != plan["recipes"]):
        raise ValueError("selected base, evaluator, source, or CPU conversion environment changed")
    return plan, path.parent


def _evidence(plan: dict, root: Path) -> dict:
    evidence = {recipe: recipe_evidence(plan["route"], recipe,
        Path(plan["recipes"][recipe]["conversion_report"]), root / recipe / "shortlist")
        for recipe in RECIPES}
    if any(row["build"] != plan["build"] for row in evidence.values()):
        raise ValueError("shortlist differs from prepared evaluator")
    return evidence


def _cells(plan: dict, root: Path, evidence: dict) -> list[tuple]:
    return [(recipe, k, w, capacity_cells(plan["route"], evidence[recipe], k, w,
                    root / recipe / f"capacity-k{k}-w{w}"))
            for recipe in RECIPES for k, w in DFLASH_PRODUCTION_PROFILES]


def advance_capacity(path: Path) -> None:
    plan, root = _load_plan(path)
    _evidence(plan, root)
    lines = [_stage(plan, root, "validate-plan")]
    for recipe in RECIPES:
        for k, w in DFLASH_PRODUCTION_PROFILES:
            output = root / recipe / f"capacity-k{k}-w{w}"
            command = _common(plan, recipe, "dflash-capacity", output, k, w, [1,2,3,4])
            # Only retained, subsequently validated cell failures permit advancement.
            lines += ["{ " + _matrix_shell(command, output) + "; } || test -f "
                      + shlex.quote(str(output / "failures.json"))]
    _write_script(root / "capacity-commands.sh", lines)


def advance_c1(path: Path) -> None:
    plan, root = _load_plan(path)
    evidence = _evidence(plan, root)
    lines = [_stage(plan, root, "validate-plan")]
    for recipe, k, w, capacity in _cells(plan, root, evidence):
        if 1 in capacity["eligible_concurrency"]:
            output = root / recipe / f"c1-k{k}-w{w}"
            lines += [_matrix_shell(_common(plan, recipe, "dflash-pareto", output, k, w, [1]), output)]
    _write_script(root / "c1-commands.sh", lines)


def advance_pareto(path: Path) -> None:
    plan, root = _load_plan(path)
    evidence = _evidence(plan, root)
    cells = _cells(plan, root, evidence)
    survivors = []
    assembly = [*_python(plan), "-m", "tools.bench.assemble_dflash_selection",
                "--base-selection", plan["route"]["terminal_selection"]["path"]]
    for recipe in RECIPES:
        assembly += ["--recipe", recipe, plan["recipes"][recipe]["conversion_report"],
                     str(root / recipe / "shortlist")]
    for recipe, k, w, capacity in cells:
        assembly += ["--capacity", recipe, str(k), str(w), str(root / recipe / f"capacity-k{k}-w{w}")]
        if 1 not in capacity["eligible_concurrency"]:
            continue
        output = root / recipe / f"c1-k{k}-w{w}"
        screen = performance_cells(plan["route"], evidence[recipe], k, w, output, [1])
        assembly += ["--c1", recipe, str(k), str(w), str(output)]
        if screen["matched_speed_by_concurrency"]["1"]["pass"]:
            survivors.append((recipe, k, w, capacity["eligible_concurrency"]))
    licensed = any(k == 4 for _, k, _, _ in survivors)
    lines = [_stage(plan, root, "validate-plan")]
    for recipe, k, w, concurrency in survivors if licensed else []:
        concurrency = [c for c in concurrency if c != 1]
        if not concurrency:
            continue
        output = root / recipe / f"pareto-k{k}-w{w}"
        lines += [_matrix_shell(_common(plan, recipe, "dflash-pareto", output, k, w, concurrency), output)]
        assembly += ["--pareto", recipe, str(k), str(w), str(output)]
    assembly += ["--out", str(root / "selection.json")]
    lines += [shlex.join(assembly)]
    _write_script(root / "pareto-and-select.sh", lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("prepare")
    for name in ("selection", "out", "bench", "planner"):
        first.add_argument("--" + name, type=Path, required=True)
    first.add_argument("--conversion-python", type=Path, default=CONVERSION_PYTHON)
    first.add_argument("--conversion-pythonpath")
    first.add_argument("--source", type=Path, default=DFLASH_SOURCE)
    for name in ("validate-plan", "advance-capacity", "advance-c1", "advance-pareto"):
        sub.add_parser(name).add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.selection, args.out, bench=args.bench, planner=args.planner,
                    conversion_python=args.conversion_python, pythonpath=args.conversion_pythonpath,
                    source=args.source)
        else:
            {"validate-plan": _load_plan, "advance-capacity": advance_capacity,
             "advance-c1": advance_c1, "advance-pareto": advance_pareto}[args.command](args.plan)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
