#!/usr/bin/env python3
"""Focused fail-closed tests for the post-PPL hybrid greedy diagnostic."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.ppl import run
from tools.ppl import validate_fp8_hybrid_greedy as validator


class HybridGreedyDiagnosticTest(unittest.TestCase):
    def test_binding_accepts_same_resolved_path_and_rejects_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            path.write_text("{}\n", encoding="utf-8")
            expected = validator._binding(path)
            validator._require_binding(
                {"path": str(path.parent / "." / path.name), "sha256": expected["sha256"]},
                expected,
                "fixture",
            )
            with self.assertRaisesRegex(SystemExit, "differs from the required authority"):
                validator._require_binding(
                    {"path": str(path), "sha256": "0" * 64}, expected, "fixture"
                )

    def test_candidate_sidecars_are_hash_and_domain_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "8192.prefill.r9700-g16.json"
            raw = {field: None for field in run.CANDIDATE_SCORER_REPORT_FIELDS}
            raw.update(
                {
                    "scheme": "r9700-g16",
                    "prompt_tokens": 4,
                    "skip_tokens": 2,
                    "tokens_scored": 1,
                    "argmax_tokens": 1,
                    "non_finite": 0,
                    "mean_nll": 1.0,
                }
            )
            path.write_text(json.dumps(raw), encoding="utf-8")
            path.with_suffix(".nllf32").write_bytes(struct.pack("<f", 1.0))
            path.with_suffix(".argmaxi32").write_bytes(struct.pack("<i", 7))
            cell = {
                **raw,
                "command": ["scorer", "--out-json", str(path)],
                "nll_sha256": run.file_sha256(path.with_suffix(".nllf32")),
                "argmax_sha256": run.file_sha256(path.with_suffix(".argmaxi32")),
                "complete_finite_aligned": True,
            }
            _, nll, argmax = validator._candidate_sidecars(cell)
            self.assertEqual(nll, [1.0])
            self.assertEqual(argmax, [7])

            path.with_suffix(".argmaxi32").write_bytes(struct.pack("<i", 8))
            with self.assertRaisesRegex(SystemExit, "sidecar hashes differ"):
                validator._candidate_sidecars(cell)
            cell["argmax_sha256"] = run.file_sha256(path.with_suffix(".argmaxi32"))
            path.with_suffix(".argmaxi32").write_bytes(
                struct.pack("<i", run.TOKEN_DOMAIN)
            )
            cell["argmax_sha256"] = run.file_sha256(path.with_suffix(".argmaxi32"))
            with self.assertRaisesRegex(SystemExit, "in-domain"):
                validator._candidate_sidecars(cell)

    def test_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.json"
            validator._write_new(path, {"valid": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"valid": True})
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                validator._write_new(path, {"valid": False})


if __name__ == "__main__":
    unittest.main()
