from __future__ import annotations

import json
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest import mock

from tools.bench import validate_selected_niah as validator_module
from tools.bench.prepare_selected_niah import POSITIONS, REPO, sha
from tools.bench.run_niah_check import DEFAULT_NEEDLE, file_identity, matrix_cases, resolve_fixture
from tools.bench.validate_selected_niah import validate


class SelectedNiahValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=REPO / "profiles/bench")
        self.root = Path(self.temp.name)
        self.server = self.root / "apps/ninfer-serve"
        self.server.parent.mkdir()
        self.server.write_bytes(b"server")
        self.selection = self.root / "selection.json"
        self.selection.write_text("{}\n", encoding="utf-8")
        self.log = self.root / "server.requests.jsonl"
        self.log.write_bytes(b'{"event":"server_start"}\n')
        cases = matrix_cases(["64k"], list(POSITIONS))
        labels = [label for label, _ in cases]
        fixtures = []
        for label, ref in cases:
            path = resolve_fixture(ref).resolve(strict=True)
            fixtures.append({"label": label, "ref": ref, **file_identity(path)})
        selection_sha = "1" * 64
        artifact_sha = "2" * 64
        evidence = {
            "artifact_type": "ninfer_niah_evidence", "schema_version": 2, "pass": True,
            "evidence_mode": "provenance-bound", "model": "qwen3.8-27b",
            "needle": DEFAULT_NEEDLE, "answer_match": "exact",
            "max_tokens": 64, "thinking": False,
            "seed": None, "runs": 1,
            "cases": [
                {
                    "label": label, "fixture": fixture["ref"],
                    "fixture_identity": {
                        key: fixture[key] for key in ("path", "bytes", "sha256")
                    },
                    "fixture_unchanged": True, "passed": True,
                    "retrieved": 1, "total": 1, "recall": 1.0,
                    "requests": [{
                        "run": 1, "status": "pass", "prompt_tokens": 1024 + index,
                        "completion_tokens": 3,
                        "answer_bytes": len(DEFAULT_NEEDLE.encode("utf-8")),
                        "answer_sha256": sha256(DEFAULT_NEEDLE.encode("utf-8")).hexdigest(),
                    }],
                }
                for index, (label, fixture) in enumerate(zip(labels, fixtures, strict=True))
            ],
            "fresh_full_prefill": {
                "pass": True, "request_count": 5,
                "requests": [
                    {
                        "request_id": index + 1, "prompt_tokens": 1024 + index,
                        "computed_prefill_tokens": 1024 + index,
                    }
                    for index in range(5)
                ],
                "validated_log_bytes": self.log.stat().st_size,
                "validated_log_sha256": sha(self.log),
            },
            "provenance": {
                "static_profile_selection": {
                    "sha256": selection_sha, "kv_value_group": 16,
                    "xattention_profile": "dense", "prefill_chunk": 4096,
                },
                "artifact": {"sha256": artifact_sha, "weights_id": "weights"},
                "server_executable": file_identity(self.server),
            },
        }
        self.evidence_path = self.root / "niah.evidence.json"
        self.evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
        self.plan = {
            "artifact_type": "ninfer_r9700_selected_niah_plan", "schema_version": 1,
            "terminal_route": {
                "maximum_runtime_concurrency": 4,
                "terminal_selection": {"path": str(self.selection), "sha256": selection_sha},
                "artifact": {"sha256": artifact_sha, "weights_id": "weights"},
                "cache_profile": {"value_group": 16},
                "execution_profile": {"xattention_profile": "dense"},
                "selected_prefill_chunk": 4096,
                "build_directory": str(self.root),
            },
            "status": "command_only_not_executed",
            "workload": {
                "model": "qwen3.8-27b", "length": "64k", "positions": list(POSITIONS),
                "runs_per_cell": 1, "max_tokens": 64, "thinking": False,
                "needle": DEFAULT_NEEDLE, "answer_match": "exact",
                "maximum_concurrency": 1,
            },
            "fixtures": fixtures,
            "server": {
                **file_identity(self.server), "host": "127.0.0.1", "port": 18081,
                "max_context": 262144, "kv_capacity": 262144,
                "max_concurrency": 1, "prefix_reuse": False,
            },
            "outputs": {
                "evidence": str(self.evidence_path), "server_log": str(self.log),
                "server_stdout": str(self.root / "server.stdout.log"),
                "server_stderr": str(self.root / "server.stderr.log"),
                "admission": str(self.root / "admission.json"),
            },
        }
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan) + "\n", encoding="utf-8")
        validator = REPO / "tools/bench/validate_selected_niah.py"
        closure = [self.plan_path, validator, *(Path(row["path"]) for row in fixtures)]
        (self.root / "prepared.sha256").write_text("".join(
            f"{sha(path)}  {path.relative_to(REPO)}\n" for path in closure
        ), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_accepts_exact_admission_and_binds_server_log(self) -> None:
        result = validate(
            self.plan_path, self.root,
            route_resolver=lambda _path: self.plan["terminal_route"],
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["evidence"]["sha256"], sha(self.evidence_path))
        self.assertEqual(result["campaign"]["plan"]["sha256"], sha(self.plan_path))

    def test_rejects_server_log_changed_after_checker(self) -> None:
        with self.log.open("ab") as output:
            output.write(b"{}\n")
        with self.assertRaisesRegex(ValueError, "server log differs"):
            validate(
                self.plan_path, self.root,
                route_resolver=lambda _path: self.plan["terminal_route"],
            )

    def test_rejects_route_that_no_longer_recomputes(self) -> None:
        changed = dict(self.plan["terminal_route"])
        changed["selected_prefill_chunk"] = 2048
        with self.assertRaisesRegex(ValueError, "differs from current schema-v7"):
            validate(self.plan_path, self.root, route_resolver=lambda _path: changed)

    def test_rejects_stale_fixture_or_request_detail(self) -> None:
        evidence = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        evidence["cases"][0]["fixture_identity"]["sha256"] = "0" * 64
        self.evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen fixture/run"):
            validate(
                self.plan_path, self.root,
                route_resolver=lambda _path: self.plan["terminal_route"],
            )

    def test_rejects_nonexact_answer_evidence(self) -> None:
        evidence = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        evidence["answer_match"] = "contains"
        self.evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exact admission ladder"):
            validate(
                self.plan_path, self.root,
                route_resolver=lambda _path: self.plan["terminal_route"],
            )

    def test_rejects_noncanonical_server_or_output_contract(self) -> None:
        for section, key, value in (
            ("server", "prefix_reuse", True),
            ("server", "max_concurrency", 2),
            ("outputs", "evidence", str(self.root / "stale.json")),
        ):
            original = self.plan[section][key]
            self.plan[section][key] = value
            self.plan_path.write_text(json.dumps(self.plan) + "\n", encoding="utf-8")
            validator = REPO / "tools/bench/validate_selected_niah.py"
            closure = [
                self.plan_path, validator,
                *(Path(row["path"]) for row in self.plan["fixtures"]),
            ]
            (self.root / "prepared.sha256").write_text("".join(
                f"{sha(path)}  {path.relative_to(REPO)}\n" for path in closure
            ), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "plan contract differs"):
                validate(
                    self.plan_path, self.root,
                    route_resolver=lambda _path: self.plan["terminal_route"],
                )
            self.plan[section][key] = original

    def test_main_revalidates_after_exclusive_publication(self) -> None:
        output = self.root / "admission.json"
        result = {"status": "pass"}
        argv = [
            "validate_selected_niah.py", "--plan", str(self.plan_path),
            "--root", str(self.root), "--out", str(output),
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            validator_module, "validate", side_effect=[result, result]
        ) as validate_call:
            self.assertEqual(validator_module.main(), 0)
        self.assertEqual(validate_call.call_count, 2)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

    def test_main_preserves_dangling_output_namespace(self) -> None:
        output = self.root / "admission.json"
        output.symlink_to(self.root / "missing.json")
        argv = [
            "validate_selected_niah.py", "--plan", str(self.plan_path),
            "--root", str(self.root), "--out", str(output),
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            validator_module, "validate"
        ) as validate_call:
            with self.assertRaises(SystemExit):
                validator_module.main()
        validate_call.assert_not_called()
        self.assertTrue(output.is_symlink())

    def test_main_does_not_remove_replacement_after_revalidation_failure(self) -> None:
        output = self.root / "admission.json"
        result = {"status": "pass"}
        argv = [
            "validate_selected_niah.py", "--plan", str(self.plan_path),
            "--root", str(self.root), "--out", str(output),
        ]

        def validate_then_replace(*_args):
            if not output.exists():
                return result
            output.unlink()
            output.write_text('{"foreign": true}\n', encoding="utf-8")
            raise ValueError("revalidation failed")

        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            validator_module, "validate", side_effect=validate_then_replace
        ), self.assertRaisesRegex(ValueError, "revalidation failed"):
            validator_module.main()
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"foreign": True})


if __name__ == "__main__":
    unittest.main()
