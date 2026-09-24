#!/usr/bin/env python3
"""Fail-closed revalidation for the evaluation-only DFlash conversion package."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
PLAN = PACKAGE / "plan.json"
CONVERTER = REPO / "tools/convert/qwen3_8_27b_r9700/convert_dflash2_q4.py"
INVENTORY = REPO / "tools/convert/qwen3_8_27b_r9700/dflash2_q4_inventory.py"
RECIPES = REPO / "tools/convert/qwen3_8_27b_r9700/dflash2_matrix_recipes.py"


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"DFlash evaluation conversion preflight: FAIL: {message}")


def main() -> None:
    sys.path.insert(0, str(REPO))
    from tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 import (
        preflight, preflight_summary,
    )

    plan = json.loads(PLAN.read_text())
    if (plan.get("artifact_type") != "ninfer_r9700_dflash_eval_companion_conversion_plan" or
            plan.get("schema_version") != 1 or plan.get("status") != "prepared_not_executed" or
            plan.get("purpose") != "registered canonical-Q4 DFlash evaluation control only" or
            plan.get("production_selection_bypassed") is not False or
            plan.get("weight_recipe_selected") is not False):
        fail("plan status would bypass the production recipe decision")
    base = Path(plan["base"]["path"])
    source = Path(plan["dflash_source"]["path"])
    output = Path(plan["output"]["path"])
    for target in (output, Path(str(output) + ".conversion.pending.json"),
                   Path(str(output) + ".conversion.json")):
        if os.path.lexists(target):
            fail(f"fresh output target is occupied: {target}")
    checked = preflight(base, source)
    actual = preflight_summary(checked, output)
    if actual["base"] != {
        "path": str(base),
        "identity": {"model_id": plan["base"]["model_id"],
                     "weights_id": plan["base"]["weights_id"]},
        "bytes": plan["base"]["bytes"], "sha256": plan["base"]["sha256"],
        "authority": plan["base"]["authority"], "objects": plan["base"]["objects"],
        "payload_copy": "byte_exact",
    }:
        fail("base identity, bytes, hash, inventory, or N16 migration authority changed")
    actual_source = actual["dflash_source"]
    expected_source = plan["dflash_source"]
    if (actual_source.get("path") != str(source) or
            any(actual_source.get(key) != expected_source[key] for key in (
                "tensor_count", "safetensors_bytes", "safetensors_sha256",
                "config_sha256", "readme_sha256"))):
        fail("complete BF16 DFlash source identity changed")
    recipe = actual["dflash_plan"]["recipe"]
    if any(recipe.get(key) != value for key, value in plan["recipe"].items()):
        fail("canonical-Q4/BF16-state recipe changed")
    if (actual["combined_plan"]["objects"] != plan["combined_plan"]["objects"] or
            actual["combined_plan"]["sha256"] != plan["combined_plan"]["sha256"] or
            actual["projected_file_bytes"] != plan["output"]["projected_file_bytes"] or
            actual["projected_device_arena_bytes"] !=
            plan["output"]["projected_device_arena_bytes"] or
            actual["output_identity"] != {
                "model_id": plan["output"]["model_id"],
                "weights_id": plan["output"]["weights_id"],
            }):
        fail("combined object plan, capacity, or output identity changed")
    code = plan["converter"]
    if (sha(CONVERTER) != code["source_sha256"] or
            sha(INVENTORY) != code["inventory_sha256"] or
            sha(RECIPES) != code["recipe_source_sha256"]):
        fail("converter implementation identity changed")
    environment = plan["environment"]
    launcher = Path(environment["launcher"])
    pyvenv = Path(environment["pyvenv_cfg"])
    if sha(launcher) != environment["launcher_sha256"] or \
            sha(pyvenv) != environment["pyvenv_cfg_sha256"]:
        fail("conversion interpreter identity changed")
    probe = (
        "import json,sys,torch,safetensors; "
        "print(json.dumps({'executable':sys.executable,'python':sys.version.split()[0],"
        "'torch':torch.__version__,'torch_hip':torch.version.hip,"
        "'safetensors':safetensors.__version__}))"
    )
    process = subprocess.run(
        [str(launcher), "-c", probe], cwd=REPO, check=True, capture_output=True, text=True,
        env={**os.environ, "LD_LIBRARY_PATH": environment["LD_LIBRARY_PATH"]},
    )
    runtime = json.loads(process.stdout)
    if runtime != {key: environment[key] for key in (
            "executable", "python", "torch", "torch_hip", "safetensors")
            if key in environment} | {"executable": str(launcher)}:
        fail("conversion Python/ROCm environment changed")
    free = shutil.disk_usage(output.parent).free
    if free < plan["output"]["minimum_free_filesystem_bytes"]:
        fail(f"output filesystem has only {free} free bytes")
    print(json.dumps({
        "status": "passed", "base_sha256": plan["base"]["sha256"],
        "source_sha256": expected_source["safetensors_sha256"],
        "output_identity": actual["output_identity"],
        "projected_file_bytes": actual["projected_file_bytes"],
        "projected_device_arena_bytes": actual["projected_device_arena_bytes"],
        "free_filesystem_bytes": free, "production_selection_bypassed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
