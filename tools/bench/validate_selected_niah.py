#!/usr/bin/env python3
"""Validate the exact post-terminal 64K five-position NIAH admission result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tools.bench.prepare_selected_niah import POSITIONS, REPO, resolve_route, sha
from tools.bench.run_niah_check import (
    DEFAULT_NEEDLE,
    file_identity,
    matrix_cases,
    resolve_fixture,
)


def validate_closure(root: Path, plan_path: Path) -> dict[Path, str]:
    entries = {}
    for line in (root / "prepared.sha256").read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError("prepared closure is malformed")
        path = Path(parts[1])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("prepared closure has an invalid path")
        resolved = (REPO / path).resolve()
        if resolved in entries:
            raise ValueError("prepared closure has a duplicate path")
        entries[resolved] = parts[0]
    required = {plan_path.resolve(), Path(__file__).resolve()}
    if not required.issubset(entries):
        raise ValueError("prepared closure lacks NIAH authorities")
    for path, digest in entries.items():
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"prepared closure bytes changed: {path}")
    return entries


def validate(plan_path: Path, root: Path, *, route_resolver=resolve_route) -> dict:
    root = root.resolve(strict=True)
    plan_path = plan_path.resolve(strict=True)
    if plan_path != root / "plan.json":
        raise ValueError("NIAH plan must belong to the supplied campaign root")
    closure = validate_closure(root, plan_path)
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    workload = plan.get("workload", {})
    route = plan.get("terminal_route")
    if not isinstance(route, dict) or not isinstance(route.get("terminal_selection"), dict):
        raise ValueError("NIAH plan lacks a terminal route")
    try:
        recomputed_route = route_resolver(Path(route["terminal_selection"]["path"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("NIAH terminal route no longer validates") from error
    if recomputed_route != route:
        raise ValueError("NIAH terminal route differs from current schema-v7 authority")
    build = Path(route["build_directory"])
    selected_server = (build / "apps/ninfer-serve").resolve(strict=True)
    expected_server = {
        **file_identity(selected_server),
        "host": "127.0.0.1", "port": 18081,
        "max_context": 262144, "kv_capacity": 262144,
        "max_concurrency": 1, "prefix_reuse": False,
    }
    expected_outputs = {
        "server_log": str(root / "server.requests.jsonl"),
        "server_stdout": str(root / "server.stdout.log"),
        "server_stderr": str(root / "server.stderr.log"),
        "evidence": str(root / "niah.evidence.json"),
        "admission": str(root / "admission.json"),
    }
    if (
        plan.get("artifact_type") != "ninfer_r9700_selected_niah_plan"
        or plan.get("schema_version") != 1
        or plan.get("status") != "command_only_not_executed"
        or workload != {
            "model": "qwen3.8-27b", "length": "64k", "positions": list(POSITIONS),
            "runs_per_cell": 1, "max_tokens": 64, "thinking": False,
            "needle": DEFAULT_NEEDLE, "answer_match": "exact",
            "maximum_concurrency": 1,
        }
        or route.get("maximum_runtime_concurrency") != 4
        or plan.get("server") != expected_server
        or plan.get("outputs") != expected_outputs
    ):
        raise ValueError("NIAH plan contract differs")
    expected_cases = matrix_cases(["64k"], list(POSITIONS))
    expected_labels = [label for label, _ in expected_cases]
    fixtures = plan.get("fixtures")
    if (
        not isinstance(fixtures, list)
        or [(row.get("label"), row.get("ref")) for row in fixtures] != expected_cases
    ):
        raise ValueError("NIAH plan lacks the exact 64K ladder")
    for fixture, (_, ref) in zip(fixtures, expected_cases, strict=True):
        expected_path = resolve_fixture(ref).resolve(strict=True)
        if Path(fixture["path"]) != expected_path or file_identity(expected_path) != {
            key: fixture[key] for key in ("path", "bytes", "sha256")
        }:
            raise ValueError("NIAH fixture bytes changed")
        if expected_path not in closure:
            raise ValueError("prepared closure does not freeze every NIAH fixture")
    evidence_path = Path(plan["outputs"]["evidence"]).resolve(strict=True)
    evidence_raw = evidence_path.read_bytes()
    evidence = json.loads(evidence_raw)
    if (
        evidence.get("artifact_type") != "ninfer_niah_evidence"
        or evidence.get("schema_version") != 2 or evidence.get("pass") is not True
        or evidence.get("evidence_mode") != "provenance-bound"
        or evidence.get("model") != "qwen3.8-27b" or evidence.get("needle") != DEFAULT_NEEDLE
        or evidence.get("answer_match") != "exact"
        or evidence.get("max_tokens") != 64 or evidence.get("thinking") is not False
        or evidence.get("seed") is not None or evidence.get("runs") != 1
        or [row.get("label") for row in evidence.get("cases", [])] != expected_labels
        or evidence.get("fresh_full_prefill", {}).get("pass") is not True
        or evidence.get("fresh_full_prefill", {}).get("request_count") != 5
    ):
        raise ValueError("NIAH evidence does not pass the exact admission ladder")
    evidence_cases = evidence["cases"]
    fresh_requests = evidence["fresh_full_prefill"].get("requests")
    if not isinstance(fresh_requests, list) or len(fresh_requests) != len(evidence_cases):
        raise ValueError("NIAH evidence lacks exact fresh-prefill request detail")
    previous_request_id = 0
    for index, (case, fixture, fresh) in enumerate(
        zip(evidence_cases, fixtures, fresh_requests, strict=True)
    ):
        requests = case.get("requests")
        if (
            case.get("passed") is not True
            or case.get("retrieved") != 1
            or case.get("total") != 1
            or case.get("recall") != 1.0
            or case.get("fixture") != fixture["ref"]
            or case.get("fixture_identity") != {
                key: fixture[key] for key in ("path", "bytes", "sha256")
            }
            or case.get("fixture_unchanged") is not True
            or not isinstance(requests, list)
            or len(requests) != 1
            or requests[0].get("run") != 1
            or requests[0].get("status") != "pass"
            or requests[0].get("answer_bytes") != len(DEFAULT_NEEDLE.encode("utf-8"))
            or requests[0].get("answer_sha256")
            != hashlib.sha256(DEFAULT_NEEDLE.encode("utf-8")).hexdigest()
        ):
            raise ValueError(f"NIAH evidence case {index} differs from its frozen fixture/run")
        try:
            prompt_tokens = int(requests[0]["prompt_tokens"])
            completion_tokens = int(requests[0]["completion_tokens"])
            request_id = int(fresh["request_id"])
            fresh_prompt = int(fresh["prompt_tokens"])
            computed = int(fresh["computed_prefill_tokens"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"NIAH evidence case {index} lacks request metrics") from error
        if (
            prompt_tokens <= 0
            or completion_tokens < 0
            or request_id <= previous_request_id
            or fresh_prompt != prompt_tokens
            or computed != prompt_tokens
        ):
            raise ValueError(f"NIAH evidence case {index} lacks one ordered full-prefill request")
        previous_request_id = request_id
    provenance = evidence.get("provenance", {})
    if (
        provenance.get("static_profile_selection", {}).get("sha256")
        != route["terminal_selection"]["sha256"]
        or provenance.get("artifact", {}).get("sha256") != route["artifact"]["sha256"]
        or provenance.get("artifact", {}).get("weights_id") != route["artifact"]["weights_id"]
        or provenance.get("static_profile_selection", {}).get("kv_value_group")
        != route["cache_profile"]["value_group"]
        or provenance.get("static_profile_selection", {}).get("xattention_profile")
        != route["execution_profile"]["xattention_profile"]
        or provenance.get("static_profile_selection", {}).get("prefill_chunk")
        != route["selected_prefill_chunk"]
    ):
        raise ValueError("NIAH evidence differs from terminal route")
    serve = Path(provenance["server_executable"]["path"])
    if (
        provenance["server_executable"] != {
            key: plan["server"][key] for key in ("path", "bytes", "sha256")
        }
        or file_identity(serve) != provenance["server_executable"]
    ):
        raise ValueError("NIAH server executable bytes changed")
    server_log = Path(plan["outputs"]["server_log"]).resolve(strict=True)
    log_bytes = server_log.read_bytes()
    fresh = evidence["fresh_full_prefill"]
    if (
        len(log_bytes) != fresh.get("validated_log_bytes")
        or hashlib.sha256(log_bytes).hexdigest() != fresh.get("validated_log_sha256")
    ):
        raise ValueError("NIAH server log differs from validated fresh-prefill evidence")
    if sha(plan_path) != hashlib.sha256(plan_raw).hexdigest():
        raise ValueError("NIAH plan changed while validating")
    if sha(evidence_path) != hashlib.sha256(evidence_raw).hexdigest():
        raise ValueError("NIAH evidence changed while validating")
    return {
        "artifact_type": "ninfer_r9700_selected_niah_admission",
        "schema_version": 1, "status": "pass", "terminal_route": route,
        "campaign": {
            "plan": {"path": str(plan_path), "sha256": sha(plan_path)},
            "prepared_closure": {
                "path": str(root / "prepared.sha256"),
                "sha256": sha(root / "prepared.sha256"),
            },
        },
        "evidence": {"path": str(evidence_path), "sha256": sha(evidence_path)},
        "server_executable": provenance["server_executable"],
        "gate": "64k_start_q25_mid_q75_end_exact_format_fresh_prefill_once",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")
    published = False
    revalidated = False
    published_inode = None
    try:
        result = validate(args.plan, args.root)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    pending = args.out.with_name(f".{args.out.name}.pending-{os.getpid()}")
    try:
        with pending.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(pending, args.out)
        published = True
        current = os.stat(args.out, follow_symlinks=False)
        published_inode = (current.st_dev, current.st_ino)
        if validate(args.plan, args.root) != result:
            raise ValueError("published NIAH admission does not revalidate")
        directory = os.open(args.out.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        revalidated = True
    finally:
        pending.unlink(missing_ok=True)
        if published and not revalidated and published_inode is not None:
            try:
                current = os.stat(args.out, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == published_inode:
                    args.out.unlink()
            except FileNotFoundError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
