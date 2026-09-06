#!/usr/bin/env python3
"""Prepare the selected schema-v7 winner's exact same-route token campaign."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import shlex
import tempfile
from pathlib import Path

from tools.bench.run_ninfer_bench_matrix import inspect_executable
from tools.ppl.pareto import load_payload, validate_terminal_production_authority
from tools.ppl.run import QUALITY_TIERS
from tools.reference.qwen3_8_27b_bf16.protocol import validate_checkpoint_files

REPO = Path(__file__).resolve().parents[2]
RESOLVER = REPO / "profiles/bench/post-terminal-focused-verification-20260905/resolve.py"
RUNNER = REPO / "tools/ppl/run.py"
VALIDATOR = REPO / "tools/ppl/validate_selected_exact_token.py"
BF16_SCORER = REPO / "tools/reference/qwen3_8_27b_bf16/ppl.py"
BF16_PROTOCOL = REPO / "tools/reference/qwen3_8_27b_bf16/protocol.py"
BF16_SOURCE = Path("/ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16")
BF16_RECEIPT = REPO / "profiles/ppl/r9700-bf16-source-checkpoint-preflight-20260905.json"
IDS = REPO / "tools/ppl/corpus.ids"
PYTHON = Path("/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python")
POWER = Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
LENGTHS = (8192, 32768)
AT_FDCWD = -100
RENAME_NOREPLACE = 1


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(
        AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(destination), RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def resolve_route(selection: Path) -> dict:
    spec = importlib.util.spec_from_file_location("ninfer_terminal_route", RESOLVER)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load terminal route resolver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve(selection)


def selected_quality(selection: Path, winner: str, group: int, route: dict) -> tuple[dict, dict]:
    value = load_payload(selection.read_text(encoding="utf-8"))
    validate_terminal_production_authority(value)
    matches = [row for row in value["source_provenance"] if row.get("candidate") == winner]
    if len(matches) != 1:
        raise ValueError("terminal winner lacks unique quality provenance")
    binding = matches[0].get("quality")
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise ValueError("terminal winner lacks a quality authority")
    path = Path(binding["path"]).resolve(strict=True)
    if sha(path) != binding.get("sha256"):
        raise ValueError("selected quality authority bytes changed")
    campaign = json.loads(path.read_text(encoding="utf-8"))
    profile = f"r9700-g{group}"
    cells = [
        row for row in campaign.get("cells", [])
        if isinstance(row, dict) and row.get("scheme") == profile
        and row.get("schedule") == "prefill" and row.get("prompt_tokens") in LENGTHS
    ]
    lengths = [row.get("prompt_tokens") for row in cells]
    tiers = {row.get("quality_tier") for row in cells}
    if (
        campaign.get("artifact_type") != "ninfer_r9700_ppl_campaign"
        or campaign.get("schema_version") != 6 or campaign.get("pass") is not True
        or campaign.get("prefill_chunk") != route["selected_prefill_chunk"]
        or campaign.get("xattention_profile") != route["execution_profile"]["xattention_profile"]
        or sorted(lengths) != list(LENGTHS) or len(set(lengths)) != len(LENGTHS)
        or tiers not in ({"accuracy"}, {"capacity-speed"})
        or any(row.get("quality_eligible") is not True or row.get("pass") is not True for row in cells)
    ):
        raise ValueError("selected quality authority lacks its exact admitted G16/G32 cells")
    artifact = campaign.get("candidate_artifact", {})
    selected_artifact = route["artifact"]
    if any(artifact.get(key) != selected_artifact.get(key) for key in ("weights_id", "sha256")):
        raise ValueError("selected quality authority binds a different artifact")
    return {"path": str(path), "sha256": sha(path)}, {"tier": tiers.pop(), "profile": profile}


def prepare(selection: Path, output: Path) -> dict:
    selection = selection.resolve(strict=True)
    output = (output if output.is_absolute() else Path.cwd() / output)
    output = output.parent.resolve(strict=True) / output.name
    if os.path.lexists(output):
        raise ValueError(f"output directory already exists: {output}")
    route = resolve_route(selection)
    group = route["cache_profile"]["value_group"]
    quality, selected = selected_quality(selection, route["winner"], group, route)
    artifact = Path(route["artifact"]["path"]).resolve(strict=True)
    scorer = Path(route["build_directory"]).resolve(strict=True) / "apps/ninfer-ppl"
    scorer_identity = inspect_executable(scorer)
    bf16_scorer_identity = inspect_executable(BF16_SCORER)
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise ValueError("selected ROCm Python interpreter is unavailable")
    pyvenv = PYTHON.parent.parent / "pyvenv.cfg"
    if not pyvenv.is_file():
        raise ValueError("selected ROCm Python launcher lacks pyvenv.cfg")
    receipt = json.loads(BF16_RECEIPT.read_text(encoding="utf-8"))
    source_validation = validate_checkpoint_files(BF16_SOURCE)
    if (
        receipt.get("artifact_type") != "ninfer_qwen3_8_27b_bf16_checkpoint_preflight"
        or receipt.get("status") != "passed" or receipt.get("source") != str(BF16_SOURCE)
        or receipt.get("shards", {}).get("expected_count") != 18
        or receipt.get("shards", {}).get("all_named_exactly") is not True
        or receipt.get("shards", {}).get("all_regular_nonempty") is not True
        or len(set(source_validation.values())) != 18
    ):
        raise ValueError("BF16 checkpoint receipt/current source validation failed")
    tier = selected["tier"]
    gate = QUALITY_TIERS[tier]["maximum_mean_nll_delta"]
    profile = selected["profile"]
    group_flag = "--g16" if group == 16 else "--g32"
    xattention = route["execution_profile"]["xattention_profile"]
    campaign = output / "campaign"
    admission = output / "admission.json"
    command = [
        str(PYTHON), str(RUNNER),
        "--bf16-reference-ppl-bin", str(BF16_SCORER),
        "--bf16-reference-weights", str(BF16_SOURCE),
        f"{group_flag}-ppl-bin", str(scorer), f"{group_flag}-weights", str(artifact),
        "--ids", str(IDS), "--profiles", f"bf16-reference,{profile}",
        "--quality-tier", tier, "--gate", f"{profile}={gate}",
        "--schedule", "decode", "--prefill-chunk", str(route["selected_prefill_chunk"]),
        "--device", "0",
        "--spec", "none", "--execution-parity-max-abs-nll", "0",
        "--no-position-extras",
        "--expected-q4-activation-bits", "8", "--expected-w8-activation-bits", "8",
        "--expected-fp8-qk-wmma", "1", "--expected-xattention-profile", xattention,
        "--out", str(campaign),
    ]
    if route["artifact"]["weights_id"] == "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
        command.insert(-2, "--require-fp8-hybrid")

    with tempfile.TemporaryDirectory(prefix=f".{output.name}.prepare-", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        staged.mkdir()
        plan = {
            "artifact_type": "ninfer_r9700_selected_exact_token_plan", "schema_version": 1,
            "status": "command_only_not_executed", "terminal_route": route,
            "terminal_selection": {"path": str(selection), "sha256": sha(selection)},
            "quality_authority": quality, "bf16_source_receipt": {
                "path": str(BF16_RECEIPT), "sha256": sha(BF16_RECEIPT)},
            "bf16_source": {"path": str(BF16_SOURCE), "shard_count": 18},
            "python": {"launcher_path": str(PYTHON), "sha256": sha(PYTHON),
                       "pyvenv_cfg": {"path": str(pyvenv), "sha256": sha(pyvenv)}},
            "candidate_scorer": scorer_identity,
            "bf16_scorer": bf16_scorer_identity,
            "corpus": {"path": str(IDS), "sha256": sha(IDS),
                       "manifest_path": str(IDS.with_name(f"{IDS.stem}.manifest.json")),
                       "manifest_sha256": sha(IDS.with_name(f"{IDS.stem}.manifest.json"))},
            "workload": {"concurrency": 1, "lengths": list(LENGTHS), "schedule": "decode",
                         "spec": "none", "draft_tokens": 0, "device": 0,
                         "execution_parity_max_abs_nll": 0.0},
            "quality_tier": tier, "quality_mean_nll_gate": gate,
            "candidate_profile": profile, "command": command,
            "outputs": {"campaign": str(campaign), "admission": str(admission)},
            "gate": "exact I32 greedy-token and zero-NLL-delta parity for selected-route ordinary graph/eager execution at 8K and 32K; BF16 argmax differences remain diagnostic and MTP comparisons are optional non-admission diagnostics",
        }
        plan_path = staged / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        commands = staged / "commands.sh"
        commands.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n\n"
            "set -o noclobber\n\n"
            f"readonly root={shlex.quote(str(output))}\n"
            f"readonly power={shlex.quote(str(POWER))}\n"
            f"cd {shlex.quote(str(REPO))}\n"
            'sha256sum --check --strict "$root/prepared.sha256"\n'
            'test "$(cat "$power")" = auto\n'
            'test ! -e "$root/campaign" && test ! -L "$root/campaign"\n'
            'test ! -e "$root/admission.json" && test ! -L "$root/admission.json"\n'
            f"{shlex.quote(str(PYTHON))} -c 'from pathlib import Path; from "
            "tools.reference.qwen3_8_27b_bf16.protocol import validate_checkpoint_files; "
            f"validate_checkpoint_files(Path(\"{BF16_SOURCE}\"))'\n"
            f"{shlex.join(command)}\n"
            'test "$(cat "$power")" = auto\n'
            f"{shlex.quote(str(PYTHON))} -m tools.ppl.validate_selected_exact_token "
            '--plan "$root/plan.json" '
            '--campaign "$root/campaign/results.json" --out "$root/admission.json"\n',
            encoding="utf-8",
        )
        closure = [Path(__file__).resolve(), VALIDATOR, RUNNER, RESOLVER, BF16_SCORER,
                   BF16_PROTOCOL, PYTHON, pyvenv,
                   BF16_RECEIPT, IDS, selection, Path(quality["path"]), artifact, scorer,
                   plan_path, commands]
        if route.get("hybrid_width_tool"):
            closure.append(Path(route["hybrid_width_tool"]["path"]))
        (staged / "prepared.sha256").write_text(
            "".join(
                f"{sha(path)}  {published.relative_to(REPO) if published.is_relative_to(REPO) else published}\n"
                for path in closure
                for published in [output / path.name if path in (plan_path, commands) else path]
            ), encoding="utf-8")
        rename_noreplace(staged, output)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        prepare(args.selection, args.out)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
