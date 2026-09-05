#!/usr/bin/env python3
"""Validate and atomically publish selected-winner exact execution-token evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

from tools.ppl.pareto import load_payload, validate_terminal_production_authority
from tools.ppl.run import cell_ok, file_sha256, load_argmax, load_nlls, sidecar_parity

HYBRID_WEIGHTS_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"


def _identity(path: Path) -> tuple[int, int]:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"exact-token publication path is not a regular file: {path}")
    return metadata.st_dev, metadata.st_ino


def _unlink_if_owned(path: Path, owner: tuple[int, int] | None) -> bool:
    if owner is None:
        return False
    try:
        if _identity(path) != owner:
            return False
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def validate(plan_path: Path, campaign_path: Path) -> dict:
    if plan_path.is_symlink() or campaign_path.is_symlink() or campaign_path.parent.is_symlink():
        raise ValueError("exact-token plan/campaign authority must not be symlinked")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    workload = plan.get("workload", {})
    profile = plan.get("candidate_profile")
    if (
        plan.get("artifact_type") != "ninfer_r9700_selected_exact_token_plan"
        or plan.get("schema_version") != 1 or plan.get("status") != "command_only_not_executed"
        or workload != {"concurrency": 1, "lengths": [8192, 32768], "schedule": "decode",
                           "spec": "mtp", "draft_tokens": 3,
                           "execution_parity_max_abs_nll": 0.0}
        or profile not in ("r9700-g16", "r9700-g32")
        or not isinstance(plan.get("command"), list)
        or plan["command"].count("--no-position-extras") != 1
    ):
        raise ValueError("malformed selected exact-token plan")
    for key in ("terminal_selection", "quality_authority", "bf16_source_receipt"):
        binding = plan.get(key, {})
        path = Path(binding.get("path", ""))
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            raise ValueError(f"{key} bytes changed")
    python = plan.get("python", {})
    pyvenv = python.get("pyvenv_cfg", {})
    if (
        not Path(python.get("launcher_path", "")).is_file()
        or file_sha256(Path(python["launcher_path"])) != python.get("sha256")
        or not Path(pyvenv.get("path", "")).is_file()
        or file_sha256(Path(pyvenv["path"])) != pyvenv.get("sha256")
    ):
        raise ValueError("selected Python environment bytes changed")
    selection = load_payload(Path(plan["terminal_selection"]["path"]).read_text(encoding="utf-8"))
    terminal, _ = validate_terminal_production_authority(selection)
    route = plan["terminal_route"]
    hybrid = route.get("artifact", {}).get("weights_id") == HYBRID_WEIGHTS_ID
    hybrid_tool = route.get("hybrid_width_tool")
    if (
        plan["command"].count("--require-fp8-hybrid") != int(hybrid)
        or (hybrid and not isinstance(hybrid_tool, dict))
        or (not hybrid and hybrid_tool is not None)
    ):
        raise ValueError("plan hybrid identity/planner contract differs from selected route")
    if hybrid:
        hybrid_path = Path(hybrid_tool.get("path", ""))
        if not hybrid_path.is_file() or file_sha256(hybrid_path) != hybrid_tool.get("sha256"):
            raise ValueError("selected hybrid planner bytes changed")
    if (
        terminal.get("winner") != route.get("winner")
        or any(terminal.get("winner_artifact", {}).get(key) != route["artifact"].get(key)
               for key in ("weights_id", "sha256"))
        or terminal.get("winner_cache_profile") != route.get("cache_profile")
        or terminal.get("winner_execution_profile") != route.get("execution_profile")
        or selection.get("selected_prefill_chunk") != route.get("selected_prefill_chunk")
    ):
        raise ValueError("plan route differs from current terminal selection")
    artifact_path = Path(route["artifact"]["path"])
    scorer_binding = plan.get("candidate_scorer", {})
    scorer_path = Path(scorer_binding.get("path", ""))
    bf16_scorer_binding = plan.get("bf16_scorer", {})
    bf16_scorer_path = Path(bf16_scorer_binding.get("path", ""))
    corpus_binding = plan.get("corpus", {})
    corpus_path = Path(corpus_binding.get("path", ""))
    corpus_manifest_path = Path(corpus_binding.get("manifest_path", ""))
    if (
        not artifact_path.is_file() or file_sha256(artifact_path) != route["artifact"].get("sha256")
        or not scorer_path.is_file() or file_sha256(scorer_path) != scorer_binding.get("sha256")
        or not bf16_scorer_path.is_file()
        or file_sha256(bf16_scorer_path) != bf16_scorer_binding.get("sha256")
        or not corpus_path.is_file() or file_sha256(corpus_path) != corpus_binding.get("sha256")
        or not corpus_manifest_path.is_file()
        or file_sha256(corpus_manifest_path) != corpus_binding.get("manifest_sha256")
    ):
        raise ValueError("selected artifact, scorer, or corpus bytes changed")
    if (
        campaign.get("artifact_type") != "ninfer_r9700_ppl_campaign"
        or campaign.get("schema_version") != 6 or campaign.get("pass") is not True
        or campaign.get("model_id") != "qwen3.8-27b"
        or campaign.get("lengths") != [8192, 32768]
        or campaign.get("schedules") != ["decode"]
        or campaign.get("prefill_chunk") != plan["terminal_route"]["selected_prefill_chunk"]
        or set(campaign.get("weights_inputs", {})) != {"bf16-reference", profile}
        or set(campaign.get("scorers", {})) != {"bf16-reference", profile}
        or campaign.get("quality_tier") != plan["quality_tier"]
        or campaign.get("execution_parity_max_abs_nll") != 0.0
        or campaign.get("position_extras_enabled") is not False
        or campaign.get("required_candidate_identity")
        != ("fp8-hybrid-selection-authority" if hybrid else None)
        or campaign.get("xattention_profile")
        != plan["terminal_route"]["execution_profile"]["xattention_profile"]
    ):
        raise ValueError("campaign differs from the selected exact-token contract")
    artifact = campaign.get("candidate_artifact", {})
    route_artifact = plan["terminal_route"]["artifact"]
    if any(artifact.get(key) != route_artifact.get(key) for key in ("weights_id", "sha256")):
        raise ValueError("campaign artifact differs from terminal winner")
    scorer = campaign["scorers"][profile]
    reference_scorer = campaign["scorers"]["bf16-reference"]
    if (
        scorer.get("path") != scorer_binding.get("path")
        or scorer.get("sha256") != scorer_binding.get("sha256")
        or scorer.get("bytes") != scorer_binding.get("file_size_bytes")
        or reference_scorer.get("path") != bf16_scorer_binding.get("path")
        or reference_scorer.get("sha256") != bf16_scorer_binding.get("sha256")
        or reference_scorer.get("bytes") != bf16_scorer_binding.get("file_size_bytes")
        or campaign.get("corpus", {}).get("path") != corpus_binding.get("path")
        or campaign.get("corpus", {}).get("ids_sha256") != corpus_binding.get("sha256")
        or campaign.get("corpus", {}).get("manifest_path") != corpus_binding.get("manifest_path")
        or campaign.get("corpus", {}).get("manifest_sha256")
        != corpus_binding.get("manifest_sha256")
    ):
        raise ValueError("campaign scorer or corpus differs from selected inputs")
    receipt = json.loads(Path(plan["bf16_source_receipt"]["path"]).read_text(encoding="utf-8"))
    reference = campaign.get("reference_source", {})
    if (
        campaign.get("weights_inputs", {}).get("bf16-reference") != plan["bf16_source"]["path"]
        or reference.get("config_sha256") != receipt.get("metadata", {}).get("config", {}).get("sha256")
        or reference.get("index_sha256") != receipt.get("metadata", {}).get("index", {}).get("sha256")
        or reference.get("tensor_count") != 1199 or reference.get("shard_count") != 18
        or len(reference.get("shards_sha256", {})) != 18
    ):
        raise ValueError("campaign BF16 source differs from validated checkpoint")
    required = {(8192, "device_graph_parity"), (8192, "spec_parity"),
                (8192, "draft_window_parity"), (32768, "device_graph_parity"),
                (32768, "spec_parity")}
    expected_files = {
        (8192, None): f"8192.decode.{profile}.json",
        (32768, None): f"32768.decode.{profile}.json",
        (8192, "device_graph_parity"): f"8192.decode.{profile}.eager.json",
        (32768, "device_graph_parity"): f"32768.decode.{profile}.eager.json",
        (8192, "spec_parity"): f"8192.decode.{profile}.ordinary.json",
        (32768, "spec_parity"): f"32768.decode.{profile}.ordinary.json",
        (8192, "draft_window_parity"): f"8192.decode.{profile}.mtp.json",
    }
    observed: dict[tuple[int, str], dict] = {}
    campaign_root = campaign_path.parent.resolve()
    primary: dict[int, Path] = {}
    candidate_cells: list[tuple[dict, Path]] = []
    cells = campaign.get("cells", [])
    if not isinstance(cells, list) or len(cells) != 9:
        raise ValueError("campaign must contain exactly two BF16 and seven candidate cells")
    baseline_tokens: set[int] = set()
    observed_files: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("scheme") not in {"bf16-reference", profile}:
            raise ValueError("campaign contains an unexpected cell profile")
        command = cell.get("command")
        if not isinstance(command, list) or command.count("--out-json") != 1:
            raise ValueError("candidate cell lacks one output identity")
        raw_path = Path(command[command.index("--out-json") + 1]).resolve()
        if raw_path.parent != campaign_root or not raw_path.is_file():
            raise ValueError("candidate raw cell escapes or is missing from campaign")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if any(cell.get(key) != value for key, value in raw.items()):
            raise ValueError("candidate raw cell differs from aggregate")
        for suffix, field in ((".nllf32", "nll_sha256"), (".argmaxi32", "argmax_sha256")):
            sidecar = raw_path.with_suffix(suffix)
            if not sidecar.is_file() or file_sha256(sidecar) != cell.get(field):
                raise ValueError("candidate sidecar bytes changed")
        if not cell_ok(cell, load_nlls(raw_path), load_argmax(raw_path)):
            raise ValueError("campaign sidecars are incomplete, nonfinite, or unaligned")
        if cell.get("scheme") == "bf16-reference":
            if command[:2] != [plan["python"]["launcher_path"], str(bf16_scorer_path)]:
                raise ValueError("BF16 raw command uses a different scorer environment")
            tokens = cell.get("prompt_tokens")
            if raw_path.name != f"{tokens}.decode.bf16-reference.json" or tokens in baseline_tokens:
                raise ValueError("campaign has unexpected or duplicate BF16 primary evidence")
            expected_reference = campaign["reference_source"]
            if (
                cell.get("source_config_sha256") != expected_reference["config_sha256"]
                or cell.get("source_index_sha256") != expected_reference["index_sha256"]
                or cell.get("source_shards_sha256") != expected_reference["shards_sha256"]
                or cell.get("source_tensor_count") != expected_reference["tensor_count"]
                or cell.get("source_shard_count") != expected_reference["shard_count"]
            ):
                raise ValueError("BF16 raw cell source identity differs from campaign authority")
            baseline_tokens.add(tokens)
            continue
        candidate_cells.append((cell, raw_path))
        if command[:1] != [str(scorer_path)]:
            raise ValueError("candidate raw command uses a different scorer")
        tokens = cell.get("prompt_tokens")
        parity_keys = [key for key in ("device_graph_parity", "spec_parity",
                                       "draft_window_parity") if key in cell]
        if len(parity_keys) > 1:
            raise ValueError("candidate variant carries multiple parity identities")
        identity = (tokens, parity_keys[0] if parity_keys else None)
        if expected_files.get(identity) != raw_path.name or raw_path.name in observed_files:
            raise ValueError("candidate cell differs from exact execution inventory")
        observed_files.add(raw_path.name)
        flags = command[command.index("--out-json") + 2:]
        expected_flags = {
            None: ["--spec", "mtp", "--draft-tokens", "3"],
            "device_graph_parity": ["--spec", "mtp", "--draft-tokens", "3",
                                      "--no-device-graph"],
            "spec_parity": [],
            "draft_window_parity": ["--spec", "mtp", "--draft-tokens", "4"],
        }[identity[1]]
        if flags != expected_flags:
            raise ValueError("candidate execution variant command differs from exact gate")
        if raw_path.name == f"{tokens}.decode.{profile}.json":
            if tokens in primary:
                raise ValueError("duplicate primary candidate cell")
            primary[tokens] = raw_path
    if (set(primary) != {8192, 32768} or baseline_tokens != {8192, 32768}
            or observed_files != set(expected_files.values())):
        raise ValueError("campaign lacks the exact BF16/candidate execution inventory")
    for cell, raw_path in candidate_cells:
        for parity_key in ("device_graph_parity", "spec_parity", "draft_window_parity"):
            if parity_key not in cell:
                continue
            key = (cell.get("prompt_tokens"), parity_key)
            if key in observed:
                raise ValueError("duplicate exact-token parity cell")
            parity = cell[parity_key]
            recomputed = sidecar_parity(primary[cell["prompt_tokens"]], raw_path, max_abs_nll=0.0)
            if (
                not isinstance(parity, dict) or parity.get("comparison_kind") != "same-route-execution"
                or parity.get("complete_finite_aligned") is not True
                or parity.get("argmax_identity_is_gate") is not True
                or parity.get("argmax_exact") is not True or parity.get("argmax_mismatches") != 0
                or parity.get("max_abs_nll_gate") != 0.0 or parity.get("max_abs_delta_nll") != 0.0
                or parity.get("pass") is not True or parity != recomputed
            ):
                raise ValueError("same-route exact-token/NLL parity failed")
            observed[key] = parity
    if set(observed) != required:
        raise ValueError("campaign lacks the exact required execution parity inventory")
    return {
        "artifact_type": "ninfer_r9700_selected_exact_token_admission", "schema_version": 1,
        "status": "passed", "terminal_selection": plan["terminal_selection"],
        "winner": plan["terminal_route"]["winner"],
        "artifact": route_artifact, "cache_profile": plan["terminal_route"]["cache_profile"],
        "execution_profile": plan["terminal_route"]["execution_profile"],
        "selected_prefill_chunk": plan["terminal_route"]["selected_prefill_chunk"],
        "bf16_source_receipt": plan["bf16_source_receipt"],
        "quality_authority": plan["quality_authority"],
        "campaign": {"path": str(campaign_path.resolve()), "sha256": file_sha256(campaign_path)},
        "concurrency": 1, "compared_parity_cells": len(required),
        "criterion": "exact same-route I32 greedy-token identity and zero maximum absolute NLL delta",
        "bf16_argmax_identity": "diagnostic_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    output = (args.out if args.out.is_absolute() else Path.cwd() / args.out)
    output = output.parent.resolve(strict=True) / output.name
    if os.path.lexists(output):
        raise SystemExit(f"refusing to overwrite {args.out}")
    pending = output.with_name(f".{output.name}.pending-{os.getpid()}")
    if os.path.lexists(pending):
        raise SystemExit(f"refusing occupied exact-token pending namespace: {pending}")
    pending_owner: tuple[int, int] | None = None
    output_owner: tuple[int, int] | None = None
    durable = False
    try:
        for authority, label in ((args.plan, "plan"), (args.campaign, "campaign")):
            if authority.is_symlink():
                raise ValueError(f"exact-token {label} authority must not be symlinked")
        if args.campaign.parent.is_symlink():
            raise ValueError("exact-token campaign directory must not be symlinked")
        plan = args.plan.resolve(strict=True)
        campaign = args.campaign.resolve(strict=True)
        value = validate(plan, campaign)
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            metadata = os.fstat(stream.fileno())
            pending_owner = (metadata.st_dev, metadata.st_ino)
        os.link(pending, output)
        # The link created by this process initially aliases the already captured pending
        # owner. Record that owner before any path lookup so a raced replacement is never
        # mistaken for publication-owned output during rollback.
        output_owner = pending_owner
        if _identity(output) != output_owner or _identity(pending) != pending_owner:
            raise ValueError("exact-token admission hard-link inode differs")
        if json.loads(output.read_text(encoding="utf-8")) != value:
            raise ValueError("published exact-token admission bytes differ")
        if validate(plan, campaign) != value:
            raise ValueError("published exact-token admission does not revalidate")
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        durable = True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    finally:
        _unlink_if_owned(pending, pending_owner)
        if not durable:
            _unlink_if_owned(output, output_owner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
