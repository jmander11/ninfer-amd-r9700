#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from tools.ppl.compare_xattention import compare_campaigns


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CompareXAttentionTest(unittest.TestCase):
    @staticmethod
    def _reference_execution() -> dict:
        return {"profile": "deterministic-fixture", "implementation": "a" * 64}

    def _cell(
        self, root: Path, campaign: str, scheme: str, tokens: int,
        nll: list[float], argmax: list[int], *, sparse: bool, seconds: float,
    ) -> dict:
        path = root / campaign / f"{tokens}.prefill.{scheme}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        group = None if scheme == "bf16-reference" else (16 if scheme.endswith("g16") else 32)
        expected_count = tokens - tokens // 2 - 1
        nll = [*nll, *([1.0] * (expected_count - len(nll)))]
        argmax = [*argmax, *([10] * (expected_count - len(argmax)))]
        raw = {
            "scheme": scheme,
            "model_id": "qwen3.8-27b",
            "weights_id": "bf16-source" if group is None else "r9700-q4g64-n16k16-eval",
            "kv_format": "bf16-reference" if group is None else "fp8-k-int4-v",
            "schedule": "prefill",
            "spec": "none",
            "draft_tokens": 0,
            "device_graph": False if group is not None else None,
            "prefill_chunk": 4096,
            "skip_tokens": tokens // 2,
            "prompt_tokens": tokens,
            "tokens_scored": len(nll),
            "argmax_tokens": len(argmax),
            "terrible_nll": 10,
            "non_finite": 0,
            "mean_nll": sum(nll) / len(nll),
            "score_seconds": seconds,
        }
        if group is not None:
            raw.update({
                "kv_value_group": group,
                "kv_plane_layouts": {"key": "k", "value": "v", "value_scale": "s"},
                "q4_activation_bits": 8,
                "w8_activation_bits": 8,
                "fp8_qk_wmma_enabled": True,
                "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                "fp8_qk_wmma_t1_min_context": 64,
                "fp8_qk_wmma_t2_min_context": 320,
                "xattention_qualification": sparse,
            })
            if sparse:
                raw.update({
                    "xattention_profile": "b128-s16-tau900",
                    "xattention_find_block": 128,
                    "xattention_stride": 16,
                    "xattention_tau_permille": 900,
                })
        else:
            raw.update({
                "source_config_sha256": "c" * 64,
                "source_index_sha256": "i" * 64,
                "source_shards_sha256": {"model-00001": "s" * 64},
                "source_tensor_count": 4,
                "source_text_tensor_count": 3,
                "source_shard_count": 1,
                "execution_provenance": self._reference_execution(),
            })
        path.write_text(json.dumps(raw), encoding="utf-8")
        path.with_suffix(".nllf32").write_bytes(struct.pack("<" + "f" * len(nll), *nll))
        path.with_suffix(".argmaxi32").write_bytes(
            struct.pack("<" + "i" * len(argmax), *argmax)
        )
        return {
            **raw,
            "command": ["scorer", "--out-json", str(path)],
            "nll_sha256": sha256(path.with_suffix(".nllf32")),
            "argmax_sha256": sha256(path.with_suffix(".argmaxi32")),
            "complete_finite_aligned": True,
        }

    def _campaign(self, root: Path, name: str, sparse: bool) -> Path:
        cells = []
        for tokens in (8192, 32768):
            cells.append(self._cell(
                root, name, "bf16-reference", tokens, [1.0, 2.0, 3.0], [10, 11, 12],
                sparse=sparse, seconds=3.0,
            ))
            for scheme in ("r9700-g16", "r9700-g32"):
                nll = [1.5, 9.0, 13.0] if sparse else [1.0, 11.0, 3.0]
                argmax = [10, 99, 12] if sparse else [10, 11, 12]
                cells.append(self._cell(
                    root, name, scheme, tokens, nll, argmax, sparse=sparse,
                    seconds=5.0 if sparse else 10.0,
                ))
        profile = "b128-s16-tau900" if sparse else "dense"
        path = root / name / "results.json"
        path.write_text(json.dumps({
            "artifact_type": "ninfer_r9700_ppl_campaign",
            "schema_version": 6,
            "xattention_profile": profile,
            "model_id": "qwen3.8-27b",
            "reference_weights_id": "bf16-source",
            "reference_source": {
                "config_sha256": "c" * 64,
                "index_sha256": "i" * 64,
                "shards_sha256": {"model-00001": "s" * 64},
                "tensor_count": 4,
                "text_tensor_count": 3,
                "shard_count": 1,
            },
            "reference_execution": self._reference_execution(),
            "candidate_artifact": {
                "model_id": "qwen3.8-27b", "weights_id": "r9700-q4g64-n16k16-eval",
                "sha256": "a" * 64, "file_size_bytes": 123,
            },
            "weights_inputs": {
                "bf16-reference": "/bf16", "r9700-g16": "/model.ninfer",
                "r9700-g32": "/model.ninfer",
            },
            "corpus": {"ids_sha256": "b" * 64, "tokens": 32768},
            "lengths": [8192, 32768],
            "skip": "half",
            "prefill_chunk": 4096,
            "schedules": ["prefill"],
            "spec": "none",
            "draft_tokens": 0,
            "q4_activation_bits": 8,
            "w8_activation_bits": 8,
            "candidate_kv_plane_layouts": {"key": "k", "value": "v", "value_scale": "s"},
            "fp8_qk_wmma_enabled": True,
            "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
            "fp8_qk_wmma_t1_min_context": 64,
            "fp8_qk_wmma_t2_min_context": 320,
            "baseline": "bf16-reference",
            "gates": {
                "r9700-g16": math.log(1.05), "r9700-g32": math.log(1.05),
            },
            "quality_tier": "capacity-speed",
            "quality_gate_contract": {
                "tier": "capacity-speed",
                "tier_maximum_mean_nll_delta": math.log(1.05),
                "maximum_mean_nll_delta_by_profile": {
                    "r9700-g16": math.log(1.05), "r9700-g32": math.log(1.05),
                },
            },
            "schedule_parity_max_abs_nll": None,
            "execution_parity_max_abs_nll": 0.0,
            "terrible_nll": 10.0,
            "scorers": {
                "bf16-reference": {"path": "/bf16.py", "sha256": "d" * 64, "bytes": 1},
                "r9700-g16": {"path": f"/{name}-g16", "sha256": ("e" if sparse else "f") * 64, "bytes": 2},
                "r9700-g32": {"path": f"/{name}-g32", "sha256": ("1" if sparse else "2") * 64, "bytes": 2},
            },
            "cells": cells,
            "pass": True,
        }), encoding="utf-8")
        return path

    def test_assembles_direct_quality_severe_flip_and_timing_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = self._campaign(root, "dense", False)
            sparse = self._campaign(root, "sparse", True)
            result = compare_campaigns(dense, sparse)
            self.assertEqual(result["schema_version"], 1)
            self.assertEqual(len(result["comparisons"]), 4)
            row = result["comparisons"][0]
            self.assertAlmostEqual(
                row["route_quality"]["mean_delta_nll"],
                8.5 / (8192 - 8192 // 2 - 1),
            )
            self.assertEqual(row["route_quality"]["new_xattention_severe_positions"], 1)
            self.assertEqual(row["route_quality"]["repaired_dense_severe_positions"], 1)
            self.assertEqual(row["route_quality"]["argmax_mismatches"], 1)
            self.assertEqual(row["timing"]["dense_over_xattention_speedup"], 2.0)

    def test_rejects_artifact_or_workload_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = self._campaign(root, "dense", False)
            sparse = self._campaign(root, "sparse", True)
            payload = json.loads(sparse.read_text(encoding="utf-8"))
            payload["candidate_artifact"]["sha256"] = "9" * 64
            sparse.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate_artifact"):
                compare_campaigns(dense, sparse)

    def test_rejects_reference_execution_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = self._campaign(root, "dense", False)
            sparse = self._campaign(root, "sparse", True)
            payload = json.loads(sparse.read_text(encoding="utf-8"))
            payload["reference_execution"]["implementation"] = "9" * 64
            sparse.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reference_execution"):
                compare_campaigns(dense, sparse)

    def test_rejects_matching_but_wrong_accuracy_tier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = self._campaign(root, "dense", False)
            sparse = self._campaign(root, "sparse", True)
            for path in (dense, sparse):
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["quality_tier"] = "accuracy"
                payload["gates"] = {"r9700-g16": 0.02, "r9700-g32": 0.02}
                payload["quality_gate_contract"] = {
                    "tier": "accuracy",
                    "tier_maximum_mean_nll_delta": 0.02,
                    "maximum_mean_nll_delta_by_profile": payload["gates"],
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "capacity-speed gate"):
                compare_campaigns(dense, sparse)

    def test_rejects_cells_that_disagree_with_top_level_workload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = self._campaign(root, "dense", False)
            sparse = self._campaign(root, "sparse", True)
            for campaign in (dense, sparse):
                payload = json.loads(campaign.read_text(encoding="utf-8"))
                for cell in payload["cells"]:
                    if cell["scheme"] == "bf16-reference":
                        continue
                    cell["schedule"] = "decode"
                    cell_path = Path(cell["command"][2])
                    raw = json.loads(cell_path.read_text(encoding="utf-8"))
                    raw["schedule"] = "decode"
                    cell_path.write_text(json.dumps(raw), encoding="utf-8")
                campaign.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "campaign contract at schedule"):
                compare_campaigns(dense, sparse)

    def test_rejects_changed_sidecar_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = self._campaign(root, "dense", False)
            sparse = self._campaign(root, "sparse", True)
            payload = json.loads(sparse.read_text(encoding="utf-8"))
            candidate = next(cell for cell in payload["cells"] if cell["scheme"] == "r9700-g16")
            sidecar = Path(candidate["command"][2]).with_suffix(".nllf32")
            sidecar.write_bytes(sidecar.read_bytes() + struct.pack("<f", 4.0))
            with self.assertRaisesRegex(ValueError, "hash changed"):
                compare_campaigns(dense, sparse)

    def test_rejects_reference_sidecar_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = self._campaign(root, "dense", False)
            sparse = self._campaign(root, "sparse", True)
            payload = json.loads(sparse.read_text(encoding="utf-8"))
            reference = next(cell for cell in payload["cells"] if cell["scheme"] == "bf16-reference")
            path = Path(reference["command"][2])
            sidecar = path.with_suffix(".nllf32")
            count = len(sidecar.read_bytes()) // 4
            changed = list(struct.unpack("<" + "f" * count, sidecar.read_bytes()))
            changed[2] = 4.0
            sidecar.write_bytes(struct.pack("<" + "f" * count, *changed))
            reference["nll_sha256"] = sha256(path.with_suffix(".nllf32"))
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["mean_nll"] = sum(changed) / len(changed)
            path.write_text(json.dumps(raw), encoding="utf-8")
            reference["mean_nll"] = raw["mean_nll"]
            sparse.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "BF16 sidecars differ"):
                compare_campaigns(dense, sparse)


if __name__ == "__main__":
    unittest.main()
