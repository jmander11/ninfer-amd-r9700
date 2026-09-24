"""Entry point for the independent checkpoint-direct BF16 PPL authority."""

from __future__ import annotations

import sys

from .protocol import (
    ScoreProvenance,
    cache_value_group,
    clear_result,
    enable_strict_torch_determinism,
    establish_deterministic_environment,
    execution_provenance,
    expected_text_tensors,
    file_sha256,
    ids_sha256,
    parse_options,
    read_ids,
    resolve_score_begin,
    source_shard_sha256,
    validate_checkpoint_files,
    write_result,
)


def main(argv: list[str] | None = None) -> int:
    try:
        options = parse_options(argv)
        # These process-wide choices must be fixed before importing PyTorch, FLA, or the
        # backend module: importing any of them may construct or cache a BLAS handle.
        establish_deterministic_environment()
        clear_result(options)
        ids = read_ids(options.ids, options.tokens)
        weight_map = validate_checkpoint_files(options.weights)
        source_provenance = {
            "config_sha256": file_sha256(options.weights / "config.json"),
            "index_sha256": file_sha256(options.weights / "model.safetensors.index.json"),
            "shard_sha256": source_shard_sha256(options.weights, weight_map),
            "corpus_ids_sha256": ids_sha256(ids),
            "source_tensor_count": len(weight_map),
            "source_text_tensor_count": len(expected_text_tensors()),
            "source_shard_count": len(set(weight_map.values())),
        }

        # Heavy dependencies remain after exact source preflight. Deterministic enforcement
        # precedes the backend import as well as construction of any scorer/backend object.
        torch = __import__("torch")
        enable_strict_torch_determinism(torch)
        from .backend import LayerMajorTextScorer, SourceCheckpoint

        checkpoint = SourceCheckpoint(options.weights, weight_map)
        checkpoint.validate_metadata()
        provenance = ScoreProvenance(
            **source_provenance,
            execution=execution_provenance(
                torch, options.device, stage_trace_enabled=options.trace_json is not None
            ),
        )
        trace = None
        if options.trace_json is not None:
            from .stage_trace import StageTrace

            trace = StageTrace(
                torch,
                options.trace_json,
                prompt_tokens=len(ids),
                skip_tokens=resolve_score_begin(len(ids), options.schedule, options.skip),
                prefill_chunk=options.prefill_chunk,
                device_index=options.device,
                provenance={
                    "config_sha256": provenance.config_sha256,
                    "index_sha256": provenance.index_sha256,
                    "shards_sha256": dict(provenance.shard_sha256),
                    "corpus_ids_sha256": provenance.corpus_ids_sha256,
                },
            )
        scorer = LayerMajorTextScorer(
            checkpoint,
            device_index=options.device,
            prefill_chunk=options.prefill_chunk,
            schedule=options.schedule,
            skip_text=options.skip,
            kv_value_group=cache_value_group(options.scheme),
            trace=trace,
        )
        vectors = scorer.score(ids)
        write_result(options, vectors, provenance)
        if trace is not None:
            trace.write(vectors)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ninfer-ppl-bf16-source: {error}", file=sys.stderr)
        return 1


__all__ = ["main"]
