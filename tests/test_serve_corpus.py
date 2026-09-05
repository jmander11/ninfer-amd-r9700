from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.bench.run_serve_corpus import (
    CampaignError,
    Fixture,
    RUN_ARTIFACT_TYPE,
    RUN_SCHEMA_VERSION,
    RunSpec,
    load_existing_records,
    parse_args as parse_corpus_args,
    require_compiled_profile,
    require_server_log_identity,
    server_command,
    summary_row,
    validate_server_start,
)
from tools.bench.run_serve_concurrency import parse_args as parse_concurrency_args


class ServeCorpusTest(unittest.TestCase):
    def test_corpus_campaign_requires_selected_prefill_chunk(self) -> None:
        common = [
            "--artifact", "/tmp/model.ninfer", "--output", "/tmp/out",
            "--expected-kv-value-group", "16", "--expected-xattention-profile", "dense",
        ]
        with self.assertRaises(SystemExit):
            parse_corpus_args(common)
        parsed = parse_corpus_args([*common, "--prefill-chunk", "1024"])
        self.assertEqual(parsed.prefill_chunk, 1024)

    def test_concurrency_campaign_requires_selected_prefill_chunk(self) -> None:
        common = [
            "--artifact", "/tmp/model.ninfer", "--suite", "corpus-makespan",
            "--concurrency", "1", "--output", "/tmp/out",
            "--expected-kv-value-group", "16", "--expected-xattention-profile", "dense",
        ]
        with self.assertRaises(SystemExit):
            parse_concurrency_args(common)
        parsed = parse_concurrency_args([*common, "--prefill-chunk", "2048"])
        self.assertEqual(parsed.prefill_chunk, 2048)

    def test_selected_prefill_chunk_is_launched_and_binds_resume(self) -> None:
        fixture = Fixture("fixture", [], False, 1, "suite")
        artifact = Path("/tmp/selected.ninfer").resolve()
        spec = RunSpec(
            "qwen3_8_27b_r9700", "qwen3.8-27b", artifact, "mtp0", "none", 0, 0,
            "greedy", fixture, 7,
        )
        command = server_command(
            Path("/tmp/ninfer-serve"), spec, Path("/tmp/server.jsonl"), 8080, 0, 8192
        )
        self.assertEqual(command[command.index("--prefill-chunk") + 1], "8192")
        self.assertEqual(command[command.index("--max-concurrency") + 1], "1")

        start = {
            "artifact_type": "ninfer_serve_request_log", "schema_version": 20,
            "event": "server_start", "server_instance_id": "server",
            "engine": {
                "device": 0, "max_concurrency": 1,
                "max_context": 262144, "kv_capacity": 262144,
                "prefill_chunk": 8192, "kv_cache_format": "fp8-k-int4-v",
                "device_graph": True, "prefix_reuse": False,
                "speculative_backend": "none", "speculative_draft_window": 0,
                "proposal_head": "full", "kv_value_group": 16,
                "xattention_qualification": False,
            },
            "sampling_defaults": {"greedy": True},
            "artifact": {"target": spec.target, "weights_id": "weights"},
            "server": {"public_model_id": spec.model_id},
        }
        self.assertEqual(
            validate_server_start(start, spec, 0, 8192, 16, "dense"),
            ("server", "weights"),
        )
        with self.assertRaisesRegex(CampaignError, "Engine configuration mismatch"):
            validate_server_start(start, spec, 0, 4096, 16, "dense")
        start["engine"]["max_concurrency"] = 5
        with self.assertRaisesRegex(CampaignError, "Engine configuration mismatch"):
            validate_server_start(start, spec, 0, 8192, 16, "dense")

        record = {
            "artifact_type": RUN_ARTIFACT_TYPE, "schema_version": RUN_SCHEMA_VERSION,
            "target": spec.target, "speculative_mode": spec.speculative_mode,
            "sampling_mode": spec.sampling_mode, "fixture": fixture.name, "seed": spec.seed,
            "artifact_path": str(artifact), "prefill_chunk": 8192,
            "kv_value_group": 16, "xattention_profile": "dense",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(
                load_existing_records(path, {spec.key: spec}, 8192, 16, "dense"),
                {spec.key: record},
            )
            with self.assertRaisesRegex(CampaignError, "prefill chunk differs"):
                load_existing_records(path, {spec.key: spec}, 4096, 16, "dense")
            record["prefill_chunk"] = True
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CampaignError, "prefill chunk differs"):
                load_existing_records(path, {spec.key: spec}, 1, 16, "dense")

    def test_request_log_v20_identity_is_accepted(self) -> None:
        current = {
            "artifact_type": "ninfer_serve_request_log",
            "schema_version": 20,
            "event": "server_start",
        }
        require_server_log_identity(current, "server_start")

        stale = dict(current, schema_version=9)
        with self.assertRaises(CampaignError):
            require_server_log_identity(stale, "server_start")

    def test_schema_v20_requires_selected_compiled_profile(self) -> None:
        require_compiled_profile({"kv_value_group": 16, "xattention_qualification": False}, 16,
                                 "dense")
        require_compiled_profile({
            "kv_value_group": 32, "xattention_qualification": True,
            "xattention_profile": "b128-s16-tau900", "xattention_find_block": 128,
            "xattention_stride": 16, "xattention_tau_permille": 900,
        }, 32, "b128-s16-tau900")
        with self.assertRaises(CampaignError):
            require_compiled_profile({"kv_value_group": 32, "xattention_qualification": False},
                                     16, "dense")

    def test_summary_retains_one_canonical_weights_id(self) -> None:
        records = [{"weights_id": "fixture-weights", "metrics": {}}]
        row = summary_row(
            "context_profile",
            "qwen3_8_27b_r9700",
            "fixture",
            "fixture",
            "mtp0",
            "greedy",
            records,
        )
        self.assertEqual(row["weights_id"], "fixture-weights")

        with self.assertRaises(CampaignError):
            summary_row(
                "context_profile",
                "qwen3_8_27b_r9700",
                "fixture",
                "fixture",
                "mtp0",
                "greedy",
                [*records, {"weights_id": "different-fixture-weights", "metrics": {}}],
            )

    def test_serve_corpus_rejects_stale_dense_profile_identity(self) -> None:
        require_compiled_profile({"kv_value_group": 16, "xattention_qualification": False}, 16,
                                 "dense")
        with self.assertRaisesRegex(CampaignError, "selected dense"):
            require_compiled_profile({"kv_value_group": 16, "xattention_qualification": True},
                                     16, "dense")
        with self.assertRaisesRegex(CampaignError, "selected dense"):
            require_compiled_profile({"kv_value_group": 16, "xattention_qualification": 0},
                                     16, "dense")
        with self.assertRaisesRegex(CampaignError, "retains XAttention"):
            require_compiled_profile({
                "kv_value_group": 16,
                "xattention_qualification": False,
                "xattention_profile": "b128-s16-tau900",
            }, 16, "dense")


if __name__ == "__main__":
    unittest.main()
