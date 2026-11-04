#!/usr/bin/env python3
"""Prepare or replay private Chat qualification inputs; never execute proposed tools.

Input JSONL records: {"id": "private identity", "request": <Chat request>}.
Requires explicitly selected input, output, endpoints and optional --execute.
All derived payloads and responses remain outside repositories. Schema validation
uses the existing jsonschema package when scoring a returned tool call.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def private_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    for ancestor in (path, *path.parents):
        if (ancestor / ".git").exists():
            raise ValueError(f"private evaluation path is inside a repository: {path}")
    return path


def intervention(response: dict, arm: str, budget: int | None) -> dict:
    stats = response.get("usage", {}).get("ninfer", {}).get("reasoning_limit")
    if budget is None:
        return {
            "acknowledged": stats is None or stats.get("budget_tokens") is None,
            "intervened": False,
        }
    acknowledged = isinstance(stats, dict) and stats.get("budget_tokens") == budget
    reached = acknowledged and stats.get("limit_reached") is True
    forced = stats.get("forced_closes", 0) if acknowledged else 0
    retries = stats.get("budget_retries", 0) if acknowledged else 0
    correct = (forced > 0 and retries == 0) if arm == "close" else (retries == 1 and forced == 0)
    return {
        "acknowledged": acknowledged,
        "intervened": bool(reached and correct),
        "strategy_matches": bool(not reached or correct),
        "stats": stats,
    }


def reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant: {value}")


def repeats_latest_tool_batch(calls: list, request: dict) -> bool:
    """Semantic batch repetition is a diagnostic, not proof of an invalid action."""
    previous = next(
        (
            message.get("tool_calls", [])
            for message in reversed(request.get("messages", []))
            if message.get("role") == "assistant"
        ),
        [],
    )

    def signature(batch: list) -> list:
        def value_key(value):
            if isinstance(value, dict):
                return (
                    "object",
                    tuple(sorted((key, value_key(item)) for key, item in value.items())),
                )
            if isinstance(value, list):
                return ("array", tuple(value_key(item) for item in value))
            if isinstance(value, bool):
                return ("boolean", value)
            if isinstance(value, (int, float)):
                return ("number", value)
            return (type(value).__name__, value)

        return [
            (
                call["function"]["name"],
                value_key(
                    json.loads(call["function"]["arguments"], parse_constant=reject_nonfinite)
                ),
            )
            for call in batch
        ]

    try:
        return bool(previous) and signature(calls) == signature(previous)
    except (KeyError, TypeError, ValueError):
        return False


def score(response: dict, request: dict) -> dict:
    choices = response.get("choices", [])
    if len(choices) != 1:
        return {"complete_action": False, "reason": "missing unique completed choice"}
    choice = choices[0]
    message = choice.get("message", {})
    calls = message.get("tool_calls", [])
    if choice.get("finish_reason") == "tool_calls" and calls:
        try:
            import jsonschema
        except ImportError:
            return {
                "complete_action": False,
                "reason": "independent jsonschema validator unavailable",
                "tool_candidate_unvalidated": True,
            }
        declared = {
            tool["function"]["name"]: tool["function"]
            for tool in request.get("tools", [])
            if tool.get("type") == "function"
        }
        try:
            for call in calls:
                function = call["function"]
                if call.get("type") != "function" or not call.get("id"):
                    raise ValueError("unfinished tool identity")
                schema = declared[function["name"]]
                arguments = json.loads(function["arguments"], parse_constant=reject_nonfinite)
                jsonschema.validate(arguments, schema.get("parameters", {"type": "object"}))
        except (
            KeyError,
            TypeError,
            ValueError,
            jsonschema.ValidationError,
            jsonschema.SchemaError,
        ) as error:
            return {"complete_action": False, "reason": str(error)}
        return {
            "complete_action": True,
            "repeats_latest_tool_batch": repeats_latest_tool_batch(calls, request),
            "quality": "requires private human context grading",
            "task_success": "unverified; proposed tools were not executed",
        }
    text = message.get("content")
    return {
        "complete_action": False,
        "completed_answer_candidate": choice.get("finish_reason") == "stop"
        and isinstance(text, str)
        and bool(text.strip()),
        "quality": "requires private human grading; nonempty prose is not success",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--endpoint",
        required=True,
        help="explicit isolated Chat completions endpoint for this sequential run",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["unlimited", "close:4096", "close:8192", "close:16384"],
        help="unlimited or close:N or retry:N; one diagnostic strategy per run",
    )
    parser.add_argument("--model", help="explicit endpoint model identity override; recorded")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send requests only within an explicitly authorized test window",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    source = private_path(args.input_jsonl)
    output = private_path(args.output_dir)
    arms = []
    strategies = set()
    for value in args.arms:
        if value == "unlimited":
            arms.append(("unlimited", None))
            continue
        strategy, separator, cap = value.partition(":")
        if (
            not separator
            or strategy not in {"close", "retry"}
            or cap not in {"4096", "8192", "16384"}
        ):
            raise ValueError(
                "arms must be unlimited or close/retry with an accepted 4096/8192/16384 cap"
            )
        strategies.add(strategy)
        arms.append((strategy, int(cap)))
    if len(strategies) > 1 or len(set(arms)) != len(arms):
        raise ValueError("select unique arms and one startup strategy per sequential run")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    output.chmod(0o700)
    manifest = output / "manifest.jsonl"
    if manifest.exists():
        raise ValueError("choose a fresh output directory; existing evidence is never overwritten")
    with source.open() as inputs, manifest.open("x") as records:
        for index, line in enumerate(inputs):
            case = json.loads(line)
            original = case["request"]
            if "preserve_thinking" not in original:
                raise ValueError("each controlled input must explicitly choose preserve_thinking")
            if original.get("max_completion_tokens", original.get("max_tokens")) != 32768:
                raise ValueError("this comparison requires the accepted original 32768 ceiling")
            for arm, budget in arms:
                payload = dict(original)
                payload.update(stream=False, seed=args.seed)
                if args.model is not None:
                    payload["model"] = args.model
                payload.pop("stream_options", None)
                # Null explicitly disables a configured server default for this arm.
                payload["thinking_token_budget"] = budget
                stem = f"{index:04d}-{arm}-{budget or 0}-seed{args.seed}"
                (output / f"{stem}.request.json").write_text(
                    json.dumps(payload, ensure_ascii=False)
                )
                endpoint = args.endpoint
                record = {
                    "id": case["id"],
                    "arm": arm,
                    "budget": budget,
                    "seed": args.seed,
                    "payload": f"{stem}.request.json",
                    "endpoint": endpoint,
                    "executed": args.execute,
                    "original_model": original.get("model"),
                    "resolved_model": payload.get("model"),
                    "history_policy": case.get("history_policy"),
                    "preserve_thinking": original["preserve_thinking"],
                    "source_route": case.get("source_route"),
                    "failure_kind": case.get("failure_kind"),
                }
                if args.execute:
                    started = time.monotonic()
                    request = urllib.request.Request(
                        endpoint,
                        data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    try:
                        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as reply:
                            response_bytes = reply.read()
                        (output / f"{stem}.response.json").write_bytes(response_bytes)
                        response = json.loads(response_bytes)
                        record["policy"] = intervention(response, arm, budget)
                        record["grading"] = score(response, payload)
                        if not record["policy"]["acknowledged"] or not record["policy"].get(
                            "strategy_matches", True
                        ):
                            record["failed"] = True
                            record["error"] = {
                                "kind": "policy_acknowledgment",
                                "reason": "cap or strategy not acknowledged",
                            }
                    except urllib.error.HTTPError as error:
                        error_name = f"{stem}.error-response.bin"
                        record["failed"] = True
                        record["error"] = {
                            "kind": "http",
                            "status": error.code,
                            "reason": str(error.reason),
                        }
                        try:
                            error_bytes = error.read()
                            (output / error_name).write_bytes(error_bytes)
                            record["error"]["response"] = error_name
                            try:
                                error_body = json.loads(error_bytes)
                                ledger = (
                                    error_body.get("error", {})
                                    .get("ninfer", {})
                                    .get("reasoning_limit")
                                )
                                if ledger is not None:
                                    record["policy_error_stats"] = ledger
                            except (ValueError, TypeError, AttributeError):
                                pass
                        except (TimeoutError, OSError) as read_error:
                            record["error"]["response_read_error"] = str(read_error)
                    except (urllib.error.URLError, TimeoutError, OSError) as error:
                        record["failed"] = True
                        record["error"] = {"kind": "transport", "reason": str(error)}
                    except (ValueError, TypeError, KeyError) as error:
                        record["failed"] = True
                        record["error"] = {
                            "kind": "invalid_response",
                            "reason": str(error),
                            "response": f"{stem}.response.json",
                        }
                    finally:
                        record["seconds"] = time.monotonic() - started
                records.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.flush()


if __name__ == "__main__":
    main()
