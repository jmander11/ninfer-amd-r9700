from __future__ import annotations

import json

from tools.bench.run_ninfer_bench_matrix import (
    REPORT_SCHEMA_VERSION,
    R9700_KV_PLANE_LAYOUTS,
    BenchCase,
    report_rows,
)


def test_current_report_is_flattened_for_matrix_summary(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "artifact_type": "ninfer_bench_report",
                "tool": "ninfer_bench",
                "artifact": {"path": "model.ninfer"},
                "environment": {
                    "gpu_name": "AMD Radeon AI PRO R9700",
                    "architecture_name": "gfx1201",
                    "hip_runtime_version": "7.15.26333",
                    "hip_driver_version": "7.15.26333",
                    "device_id": 0,
                },
                "load": {
                    "target": "qwen3_8_27b_r9700",
                    "weights_id": "r9700-integer",
                    "load_seconds": 2.5,
                    "upload_seconds": 2.0,
                    "artifact_bytes_read": 17_500_000_000,
                    "host_to_device_bytes": 17_400_000_000,
                    "peak_staging_bytes": 134_217_728,
                },
                "memory": {
                    "kv_capacity_mode": "explicit",
                    "kv_capacity": 8192,
                    "kv_payload_bytes": 123_456,
                    "weights": {"capacity_bytes": 17_400_000_000},
                    "sequence": {"capacity_bytes": 2_000_000_000},
                    "workspace": {"capacity_bytes": 100_000_000},
                    "request_transient": {"capacity_bytes": 50_000_000},
                    "device_graph_allowance_bytes": 150_000_000,
                },
                "config": {
                    "max_context": 4096,
                    "prefill_chunk": 1024,
                    "concurrency": 4,
                    "pending_timeout_ms": 0xFFFFFFFF,
                    "pending_deadline": "unbounded",
                    "kv_cache_format": "fp8-k-int4-v",
                    "kv_value_group": 16,
                    "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
                    "q4_activation_bits": 8,
                    "q4_prefill_cta_profile": "m64n128-pingpong-production",
                    "w8_activation_bits": 16,
                    "fp8_qk_wmma_enabled": True,
                    "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                    "fp8_qk_wmma_t1_min_context": 64,
                    "fp8_qk_wmma_t2_min_context": 320,
                    "xattention_qualification": False,
                    "draft_tokens": 5,
                    "spec": "mtp",
                    "speculative_execution": True,
                    "dflash_verify_width_requested": 0,
                    "dflash_verify_width": 0,
                    "proposal_head": "optimized",
                    "decode_path": "device-graph",
                    "use_device_graph": True,
                    "retain_token_ids": False,
                    "decode_graph_prime": {"primed": True, "output_tokens": 13},
                    "repetitions": 2,
                    "warmup": 1,
                },
                "tests": [
                    {
                        "label": "tg3",
                        "kind": "tg",
                        "n_prompt": 0,
                        "n_gen": 3,
                        "requested_output_tokens": 4,
                        "workspace_peak_bytes": 1_048_576,
                        "workspace_allocator_peak_bytes": 524_288,
                        "decode_output_tok_s_mean": 4.5,
                        "decode_engine_tok_s_mean": 7.5,
                        "decode_seconds_mean": 0.75,
                        "total_seconds_mean": 0.875,
                        "speculative": {
                            "enabled": True,
                            "draft_window": 5,
                            "acceptance_rate": 1.0,
                            "acceptance_length": 6.0,
                            "rounds": 1,
                            "drafted_tokens": 5,
                            "accepted_tokens": 5,
                            "fallback_steps": 3,
                            "accepted_per_position": [1, 1, 1, 1, 1],
                        },
                        "reps": [{}, {}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = report_rows(
        report_path,
        BenchCase(
            "pure_decode",
            "tg3_k5_graph",
            ("-n", "3", "--spec", "mtp", "--draft-tokens", "5", "--lm-head-draft"),
            repetitions=2,
            warmup=1,
        ),
    )

    assert len(rows) == 1
    row = rows[0]
    assert (row["suite"], row["case"], row["label"], row["kind"]) == (
        "pure_decode",
        "tg3_k5_graph",
        "tg3",
        "tg",
    )
    assert (row["target"], row["weights_id"], row["artifact_path"], row["gpu_name"]) == (
        "qwen3_8_27b_r9700",
        "r9700-integer",
        "model.ninfer",
        "AMD Radeon AI PRO R9700",
    )
    assert (row["decode_path"], row["decode_graph_primed"]) == (
        "device-graph",
        True,
    )
    assert row["decode_graph_prime_output_tokens"] == 13
    assert row["kv_capacity"] == 8192
    assert row["concurrency"] == 4
    assert row["kv_value_group"] == 16
    assert row["kv_plane_layouts"] == R9700_KV_PLANE_LAYOUTS
    assert row["q4_activation_bits"] == 8
    assert row["q4_prefill_cta_profile"] == "m64n128-pingpong-production"
    assert row["w8_activation_bits"] == 16
    assert row["fp8_qk_wmma_enabled"] is True
    assert row["fp8_qk_wmma_t1_min_context"] == 64
    assert row["fp8_qk_wmma_t2_min_context"] == 320
    assert row["xattention_qualification"] is False
    assert row["xattention_profile"] is None
    assert row["host_to_device_bytes"] == 17_400_000_000
    assert row["workspace_capacity_bytes"] == 100_000_000
    assert row["request_transient_capacity_bytes"] == 50_000_000
    assert row["device_graph_allowance_bytes"] == 150_000_000
    assert row["workspace_peak_bytes"] == 1_048_576
    assert row["workspace_allocator_peak_bytes"] == 524_288
    assert row["decode_output_tok_s_mean"] == 4.5
    assert row["decode_engine_tok_s_mean"] == 7.5
    assert row["spec_fallback_steps"] == 3
    assert row["spec_accepted_per_position"] == "[1,1,1,1,1]"
    assert (
        row["architecture_name"],
        row["hip_runtime_version"],
        row["hip_driver_version"],
        row["device_id"],
    ) == ("gfx1201", "7.15.26333", "7.15.26333", 0)
