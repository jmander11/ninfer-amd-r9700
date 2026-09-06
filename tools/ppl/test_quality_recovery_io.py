#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.ppl.quality_recovery_io import (
    EXPECTED_AUTHORITIES,
    publish,
    require_preflight,
    validate_authority_map,
)


class QualityRecoveryIoTest(unittest.TestCase):
    def make_authority_map(self, root: Path) -> tuple[Path, dict]:
        selected = root / "selection.json"
        selected.write_text("selection\n", encoding="utf-8")
        authorities = {}
        for name, (weights_id, profile) in EXPECTED_AUTHORITIES.items():
            campaign = root / f"{name}.json"
            campaign.write_text(json.dumps({"profile": profile}), encoding="utf-8")
            receipt = {"recipe_id": weights_id, "authority": f"receipt-{weights_id}"}
            authorities[name] = {
                "path": str(campaign.resolve()),
                "sha256": hashlib.sha256(campaign.read_bytes()).hexdigest(),
                "artifact": {
                    "weights_id": weights_id,
                    "sha256": hashlib.sha256(weights_id.encode()).hexdigest(),
                    "file_size_bytes": len(weights_id),
                    "conversion_receipt": receipt,
                },
            }
        value = {
            "artifact_type": "ninfer_r9700_terminal_quality_authority_map",
            "schema_version": 2,
            "selected_prefill_chunk": 4096,
            "selected_prefill_chunk_authority": {
                "path": str(selected.resolve()),
                "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
            },
            "concurrency": 1,
            "authorities": authorities,
        }
        path = root / "authority.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path, value

    @staticmethod
    def campaign_source(campaign, weights_id, group, chunk):
        return {}, {
            "weights_id": weights_id,
            "sha256": hashlib.sha256(weights_id.encode()).hexdigest(),
            "file_size_bytes": len(weights_id),
            "conversion_receipt": {
                "recipe_id": weights_id, "authority": f"receipt-{weights_id}",
            },
            "representation": {"xattention_profile": campaign["profile"]},
            "cells": {"group": group, "chunk": chunk},
        }

    def test_authority_map_reopens_exact_n16_artifacts_and_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, value = self.make_authority_map(Path(directory))
            selected = value["selected_prefill_chunk_authority"]
            summary = {**selected, "selected_prefill_chunk": 4096}
            with (
                patch(
                    "tools.bench.prefill_chunk_authority.validate_prefill_chunk_authority",
                    return_value=(summary, {}),
                ),
                patch(
                    "tools.ppl.assemble_pareto._campaign_quality_candidate",
                    side_effect=self.campaign_source,
                ),
            ):
                self.assertEqual(validate_authority_map(path), value)

    def test_authority_map_rejects_n16_identity_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, original = self.make_authority_map(root)
            selected = original["selected_prefill_chunk_authority"]
            summary = {**selected, "selected_prefill_chunk": 4096}
            mutations = {
                "weights_id": "wrong-n16-identity",
                "sha256": "0" * 64,
                "file_size_bytes": 1,
                "conversion_receipt": {"recipe_id": "wrong"},
            }
            for field, replacement in mutations.items():
                with self.subTest(field=field):
                    value = copy.deepcopy(original)
                    value["authorities"]["ALL_Q4_DENSE_QUALITY"]["artifact"][field] = replacement
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with (
                        patch(
                            "tools.bench.prefill_chunk_authority.validate_prefill_chunk_authority",
                            return_value=(summary, {}),
                        ),
                        patch(
                            "tools.ppl.assemble_pareto._campaign_quality_candidate",
                            side_effect=self.campaign_source,
                        ),
                        self.assertRaisesRegex(ValueError, "N16 artifact binding changed"),
                    ):
                        validate_authority_map(path)

    def test_preflight_rejects_dangling_output_and_validates_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"; checkpoint.mkdir()
            output = root / "output"; output.symlink_to(root / "missing")
            with patch("tools.ppl.quality_recovery_io.validate_checkpoint_files") as validate:
                with self.assertRaisesRegex(ValueError, "occupied"):
                    require_preflight(checkpoint, [output])
            validate.assert_called_once_with(checkpoint.resolve())

    def test_publish_is_exclusive_and_removes_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending = root / "pending.json"; pending.write_bytes(b"authority")
            final = root / "final.json"
            with patch("tools.ppl.quality_recovery_io.validate_authority_map"):
                publish(pending, final)
            self.assertEqual(final.read_bytes(), b"authority")
            self.assertFalse(pending.exists())
            new_pending = root / "pending.json"; new_pending.write_bytes(b"replacement")
            with patch("tools.ppl.quality_recovery_io.validate_authority_map"):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    publish(new_pending, final)
            self.assertEqual(final.read_bytes(), b"authority")

    def test_publish_rejects_symlink_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"; source.write_bytes(b"authority")
            pending = root / "pending"; pending.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                publish(pending, root / "final")

    def test_publish_rejects_malformed_authority_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending = root / "pending.json"; pending.write_text("{}", encoding="utf-8")
            final = root / "final.json"
            with self.assertRaisesRegex(ValueError, "schema is invalid"):
                publish(pending, final)
            self.assertFalse(final.exists())


if __name__ == "__main__":
    unittest.main()
