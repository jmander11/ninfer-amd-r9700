from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.bench import validate_final_cutover_admission as gate


def route(weights_id: str = "r9700-q4g64-n16k16-eval", attention: str = "b128-s16-tau900") -> dict:
    return {
        "winner": "winner", "weights_id": weights_id,
        "artifact": {"sha256": "a" * 64, "file_size_bytes": 11},
        "executable": {"sha256": "b" * 64, "file_size_bytes": 12},
        "cache_group": 16, "attention_profile": attention, "prefill_chunk": 2048,
    }


def test_dense_control_is_the_only_profile_executable_exception() -> None:
    selected = route()
    dense = {
        "selected_route": {
            "winner": "winner", "artifact": selected["artifact"],
            "executable": {"sha256": "c" * 64, "file_size_bytes": 13},
            "value_group": 16, "xattention_profile": "dense", "prefill_chunk": 2048,
        }
    }
    dense_route = {**selected, "attention_profile": "dense", "executable": dense["selected_route"]["executable"]}
    gate.require_route("low context", dense, dense_route, dense=True)
    dense["selected_route"]["executable"] = selected["executable"]
    with pytest.raises(ValueError, match="attention profile differs"):
        gate.require_route("ordinary gate", dense, selected)


@pytest.mark.parametrize(
    ("weights_id", "fp8"),
    [
        ("r9700-q4g64-n16k16-eval", []),
        (gate.MIXED, []),
        (gate.HYBRID, [{"authority": {}}, {"authority": {}}]),
    ],
)
def test_conditional_proofs_are_exact(weights_id: str, fp8: list) -> None:
    result = gate.validate_conditionals(
        route(weights_id), {"authorities": {
            "dispatch_reconciliation": {}, "trace": {}, "static_audit": {}},
            "loaded_fp8_proofs": fp8}
    )
    assert result["four_role_loaded_fp8"] == (weights_id == gate.HYBRID)


@pytest.mark.parametrize("obsolete", ["mtp_shortlist_head", "mixed_mtp_bulk_w8"])
def test_obsolete_mtp_optimization_proof_is_rejected(obsolete: str) -> None:
    hardware = {"authorities": {"dispatch_reconciliation": {}, "trace": {},
                                 "static_audit": {}, obsolete: {}},
                "loaded_fp8_proofs": []}
    with pytest.raises(ValueError, match="obsolete conditional proof"):
        gate.validate_conditionals(route(gate.MIXED), hardware)


def test_missing_dependency_cannot_publish_receipt() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan = root / "plan.json"
        output = root / "admission.json"
        plan.write_text(json.dumps({"inputs": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="prepared closure"):
            gate.publish(plan, output)
        assert not os.path.lexists(output)


def test_joined_gate_requires_selected_vision_completion_inputs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan = root / "plan.json"
        inputs = {
            name: "missing" for name in (
                "selection", "prefill_chunk", "quality_map", "exact_plan",
                "exact_campaign", "exact_admission", "low_manifest",
                "low_admission", "niah_plan", "niah_root", "niah_admission",
                "focused_closure", "focused", "hardware", "dflash",
                "converter_preflight",
            )
        }
        plan.write_text(json.dumps({"inputs": inputs}), encoding="utf-8")
        with patch.object(gate, "validate_prepared_closure", return_value={}), pytest.raises(
            ValueError, match="input inventory differs"
        ):
            gate.assemble(plan)


def test_converter_preflight_must_revalidate_against_exact_winner() -> None:
    selected = route()
    receipt = {
        "status": "passed_no_artifact_no_device",
        "selected_route": {
            "winner": selected["winner"],
            "weights_id": selected["weights_id"],
            "evaluation_artifact": selected["artifact"],
        },
    }
    with patch.object(gate, "revalidate_converter", return_value=receipt):
        assert gate.validate_converter_preflight(Path("receipt"), Path("selection"), selected) == receipt
        changed = {**selected, "weights_id": gate.MIXED}
        with pytest.raises(ValueError, match="does not bind"):
            gate.validate_converter_preflight(Path("receipt"), Path("selection"), changed)


def test_publication_rollback_preserves_replacement_inode() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan = root / "plan.json"
        output = root / "admission.json"
        plan.write_text("{}", encoding="utf-8")
        value = {"artifact_type": gate.ARTIFACT_TYPE, "status": "passed"}
        calls = 0

        def assemble(_plan: Path):
            nonlocal calls
            calls += 1
            if calls == 2:
                replacement = root / "replacement.json"
                replacement.write_text("replacement\n", encoding="utf-8")
                replacement.replace(output)
                return {**value, "status": "changed"}
            return value

        with patch.object(gate, "assemble", side_effect=assemble), pytest.raises(
            ValueError, match="does not revalidate"
        ):
            gate.publish(plan, output)
        assert output.read_text(encoding="utf-8") == "replacement\n"


def test_occupied_dangling_output_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "admission.json"
        output.symlink_to(root / "missing")
        with pytest.raises(ValueError, match="refusing to overwrite"):
            gate.publish(root / "unused-plan.json", output)
