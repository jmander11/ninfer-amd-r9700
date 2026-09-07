#!/usr/bin/env python3
"""Validate the live R9700 ledger against its conciseness-rewrite invariants."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVARIANTS = ROOT / "plans/r9700-autonomous-todos-invariants.json"
TASK = re.compile(r"^- \[ \] `([^`]+)`([^\n]*(?:\n  [^\n]*)*)", re.MULTILINE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    contract = json.loads(INVARIANTS.read_text())
    ledger_path = ROOT / contract["ledger"]
    ledger = ledger_path.read_text()
    rows = TASK.findall(ledger)
    ids = [task_id for task_id, _ in rows]

    require(ids == contract["unchecked_ids"], "unchecked task IDs/order changed")
    require(len(ids) == contract["rewritten_unchecked_count"], "unchecked task count changed")

    dependencies: dict[str, list[str]] = {}
    conditional_ids: list[str] = []
    for task_id, body in rows:
        match = re.search(r"\[depends: ([^\]]+)\]", body)
        dependencies[task_id] = (
            [item.strip() for item in match.group(1).split(",")] if match else []
        )
        if "[if:" in body:
            conditional_ids.append(task_id)
    require(dependencies == contract["dependencies"], "task dependency graph changed")
    require(conditional_ids == contract["conditional_ids"], "conditional task set changed")

    known = set(ids) | set(contract["external_dependencies"])
    require(
        all(dep in known for values in dependencies.values() for dep in values),
        "task graph contains an unknown dependency",
    )
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(task_id: str) -> None:
        require(task_id not in visiting, f"task graph cycle at {task_id}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in dependencies[task_id]:
            if dependency in dependencies:
                visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in ids:
        visit(task_id)

    for literal in contract["required_literals"]:
        require(literal in ledger, f"required ledger text missing: {literal}")
    for owner in (
        contract["required_active_owner_paths"]
        + contract["required_historical_nonrunnable_owner_paths"]
    ):
        require(owner in ledger, f"required owner path missing: {owner}")
    require(ledger.count(contract["next_command"]) == 1, "next command must occur exactly once")
    require(
        ledger.count(contract["prepared_closure_sha256"]) == 1,
        "prepared closure SHA-256 must occur exactly once",
    )

    source = contract["rewrite_source"]
    relative_ledger = ledger_path.relative_to(ROOT).as_posix()
    baseline = subprocess.run(
        ["git", "show", f"{source['commit']}:{relative_ledger}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    require(
        hashlib.sha256(baseline).hexdigest() == source["ledger_sha256"],
        "rewrite-source ledger hash does not match its commit",
    )
    require(source["unchecked_count"] == len(contract["unchecked_ids"]),
            "rewrite-source task count does not match preserved IDs")
    print(f"validated {len(ids)} tasks, {len(conditional_ids)} conditionals, and an acyclic DAG")


if __name__ == "__main__":
    main()
