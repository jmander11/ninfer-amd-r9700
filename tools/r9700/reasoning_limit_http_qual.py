#!/usr/bin/env python3
"""Focused real-HTTP reasoning-limit protocol checks against an isolated local server.

Never starts a server or executes proposed tools. Explicit --execute and a private
fresh evidence directory are required. The declared endpoint must use local port
18080; production gateways and port8080 are outside this qualification route.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request

from reasoning_limit_qual import private_path

LONG = (
    "Work through 987654321 times 123456789 in detail, checking each carry. "
    "Then enumerate the first one hundred squares and explain the grouping."
)
ANSWER = "Compute 6 times 7. Your final answer must contain only the integer."
TOOL_SCHEMA = {
    "type": "object",
    "properties": {"number": {"type": "integer"}},
    "required": ["number"],
    "additionalProperties": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def body(
    protocol: str,
    model: str,
    prompt: str,
    cap: int | None,
    stream: bool = False,
    maximum: int = 256,
    tool: bool = False,
) -> dict:
    value = {"model": model, "stream": stream, "temperature": 0, "seed": 17}
    if protocol != "chat":
        value.pop("seed")  # These protocol surfaces do not expose a seed; greedy is deterministic.
    if protocol == "responses":
        value["store"] = False
        value.update(
            input=prompt,
            max_output_tokens=maximum,
            reasoning={"effort": "medium"},
            preserve_thinking=True,
        )
        value["ninfer"] = {"thinking_token_budget": cap}
        if tool:
            value["tools"] = [
                {"type": "function", "name": "record_number", "parameters": TOOL_SCHEMA}
            ]
            value["tool_choice"] = {"type": "function", "name": "record_number"}
    else:
        value["messages"] = [{"role": "user", "content": prompt}]
        value["max_tokens"] = maximum
        value["preserve_thinking"] = True
        if protocol == "chat":
            value.update(enable_thinking=True, reasoning_effort="medium")
            value["thinking_token_budget"] = cap
            if tool:
                value["tools"] = [
                    {
                        "type": "function",
                        "function": {"name": "record_number", "parameters": TOOL_SCHEMA},
                    }
                ]
                value["tool_choice"] = {"type": "function", "function": {"name": "record_number"}}
        else:
            value["thinking"] = {"type": "enabled"}
            value["thinking"]["budget_tokens"] = cap
            if tool:
                value["tools"] = [{"name": "record_number", "input_schema": TOOL_SCHEMA}]
                value["tool_choice"] = {"type": "tool", "name": "record_number"}
    return value


def sse(response):
    event, parts = "", []
    for wire in response:
        line = wire.decode("utf-8").rstrip("\r\n")
        if not line:
            if parts:
                data = "\n".join(parts)
                yield event, None if data == "[DONE]" else json.loads(data)
            event, parts = "", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            parts.append(line[5:].lstrip())


def normalized(protocol: str, payload: dict) -> tuple[str, str, list]:
    if protocol == "chat":
        message = payload["choices"][0]["message"]
        return (
            message.get("content") or "",
            message.get("reasoning_content") or "",
            message.get("tool_calls", []),
        )
    if protocol == "responses":
        text, reasoning, calls = [], [], []
        for item in payload.get("output", []):
            if item.get("type") == "function_call":
                calls.append(item)
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text.append(part["text"])
                elif part.get("type") == "reasoning_text":
                    reasoning.append(part["text"])
        return "".join(text), "".join(reasoning), calls
    blocks = payload.get("content", [])
    return (
        "".join(p.get("text", "") for p in blocks),
        "".join(p.get("thinking", "") for p in blocks),
        [p for p in blocks if p.get("type") == "tool_use"],
    )


def ledger(usage: dict, cap: int, forced: bool, raw_logprob: bool = False) -> dict:
    stats = usage.get("ninfer", {}).get("reasoning_limit")
    require(
        isinstance(stats, dict) and stats.get("budget_tokens") == cap,
        "resolved cap missing from final usage",
    )
    require(0 <= stats["used_tokens"] <= cap, "reasoning cap overshot")
    if forced:
        require(
            stats["limit_reached"]
            and stats["used_tokens"] == cap
            and stats["forced_closes"] == 1
            and stats["budget_retries"] == 0,
            "native close intervention not demonstrated",
        )
        records = stats.get("forced_tokens", [])
        require(len(records) == 1, "forced framing provenance missing")
        token = records[0]
        require(
            token["policy_forced"] is True
            and isinstance(token["token"], int)
            and isinstance(token["generated_token_index"], int),
            "forced framing provenance incorrect",
        )
        probability = token["target_logprob"]
        require(
            (isinstance(probability, (int, float)) and math.isfinite(probability))
            if raw_logprob
            else probability is None,
            "forced framing raw logprob availability incorrect",
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    endpoint = urllib.parse.urlparse(args.base_url)
    require(
        endpoint.scheme == "http"
        and endpoint.hostname in {"127.0.0.1", "localhost"}
        and endpoint.port == 18080,
        "use the isolated localhost18080 server",
    )
    require(args.execute, "explicit --execute is required; this script never starts a server")
    output = private_path(args.output_dir)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    paths = {
        "chat": "/v1/chat/completions",
        "responses": "/v1/responses",
        "anthropic": "/v1/messages",
    }
    results = []

    def request(name: str, protocol: str, payload: dict, cancel: bool = False) -> tuple[dict, dict]:
        (output / f"{name}.request.json").write_text(json.dumps(payload))
        began = time.monotonic()
        req = urllib.request.Request(
            args.base_url.rstrip("/") + paths[protocol],
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        )
        try:
            response = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as error:
            (output / f"{name}.error-response.bin").write_bytes(error.read())
            raise
        with response as reply:
            if not payload["stream"]:
                data = json.load(reply)
                (output / f"{name}.response.json").write_text(json.dumps(data))
                return data, data.get("usage", {})
            events, final, usage = [], {}, {}
            cancel_observed, terminal_seen = False, False
            try:
                for event, data in sse(reply):
                    require(time.monotonic() - began < 120, "stream exceeded request deadline")
                    events.append({"event": event, "data": data})
                    if data is None:
                        terminal_seen = protocol == "chat"
                        continue
                    require(
                        event not in {"error", "response.failed"} and "error" not in data,
                        "stream returned an error",
                    )
                    if cancel and (
                        (
                            protocol == "chat"
                            and any(
                                c.get("delta", {}).get("reasoning_content")
                                or c.get("delta", {}).get("content")
                                for c in data.get("choices", [])
                            )
                        )
                        or event
                        in {
                            "response.reasoning_text.delta",
                            "response.output_text.delta",
                            "content_block_delta",
                        }
                    ):
                        cancel_observed = True
                        break
                    terminal_seen = terminal_seen or event in {
                        "response.completed",
                        "response.incomplete",
                        "message_stop",
                    }
                    if protocol == "responses" and event in {
                        "response.completed",
                        "response.incomplete",
                    }:
                        final = data["response"]
                        usage = final["usage"]
                    elif data.get("usage"):
                        usage = data["usage"]
            finally:
                (output / f"{name}.events.json").write_text(json.dumps(events))
            require(time.monotonic() - began < 120, "stream exceeded bounded request time")
            require(bool(events), "stream produced no events")
            require(
                cancel_observed if cancel else terminal_seen,
                "disconnect lacked an output delta"
                if cancel
                else "stream ended without its terminal event",
            )
            return final, usage

    def check(protocol: str, phase: str, category: str, operation):
        # Each phase owns its failure: an invalid healthy answer must not prevent a
        # forced-close, streaming, tool, schema, caller-stop or cancellation request.
        try:
            value = operation()
            results.append(
                {"protocol": protocol, "phase": phase, "category": category, "passed": True}
            )
        except Exception as error:
            value = None
            results.append(
                {
                    "protocol": protocol,
                    "phase": phase,
                    "category": category,
                    "passed": False,
                    "error": str(error),
                }
            )
        (output / "summary.json").write_text(json.dumps(results, indent=2))
        return value

    def exact_answer(protocol: str, response, expected: str) -> None:
        require(response is not None, "answer unavailable because its request failed")
        actual = normalized(protocol, response[0])[0].strip()
        # Do not remove punctuation or spurious markup. Arithmetic and instruction
        # adherence remain quality gates independent of cap accounting/invariance.
        require(
            actual == expected,
            f"known-answer quality: expected exactly {expected!r}, received {actual!r}",
        )

    def nonbinding_invariance(protocol: str, baseline, capped) -> None:
        require(
            baseline is not None and capped is not None,
            "invariance unavailable because a control request failed",
        )
        require(
            normalized(protocol, baseline[0]) == normalized(protocol, capped[0]),
            "nonbinding cap changed fixed-seed output",
        )

    def nonbinding_ledger(capped) -> None:
        require(capped is not None, "nonbinding ledger unavailable because its request failed")
        stats = ledger(capped[1], 4096, False)
        require(
            stats["forced_closes"] == 0
            and stats["budget_retries"] == 0
            and not stats["limit_reached"],
            "nonbinding cap intervened",
        )

    def forced_check(protocol: str, stream: bool) -> None:
        forced_body = body(protocol, args.model, LONG, 8, stream)
        if protocol == "chat":
            forced_body.update(logprobs=True, top_logprobs=0)
        name = f"{protocol}-stream-forced" if stream else f"{protocol}-forced"
        _, usage = request(name, protocol, forced_body)
        ledger(usage, 8, True, raw_logprob=protocol == "chat")

    def tool_check(protocol: str) -> None:
        tool_reply, usage = request(
            f"{protocol}-tool",
            protocol,
            body(
                protocol,
                args.model,
                "Call record_number with integer 907. Do not execute any external action.",
                8,
                tool=True,
            ),
        )
        ledger(usage, 8, False)
        if protocol == "chat":
            require(
                tool_reply["choices"][0]["finish_reason"] == "tool_calls",
                "tool completion unfinished",
            )
        elif protocol == "responses":
            require(tool_reply["status"] == "completed", "tool response unfinished")
        else:
            require(tool_reply["stop_reason"] == "tool_use", "tool completion unfinished")
        _, _, calls = normalized(protocol, tool_reply)
        require(len(calls) == 1, "healthy tool control did not complete exactly one call")
        call = calls[0].get("function", calls[0])
        require(call.get("name") == "record_number", "healthy tool selected the wrong function")
        arguments = (
            calls[0].get("input") if protocol == "anthropic" else json.loads(call["arguments"])
        )
        import jsonschema

        jsonschema.validate(arguments, TOOL_SCHEMA)
        require(arguments == {"number": 907}, "healthy tool arguments changed")

    for protocol in paths:
        baseline = check(
            protocol,
            "unlimited-response",
            "transport",
            lambda protocol=protocol: request(
                f"{protocol}-unlimited", protocol, body(protocol, args.model, ANSWER, None)
            ),
        )
        capped = check(
            protocol,
            "nonbinding-response",
            "transport",
            lambda protocol=protocol: request(
                f"{protocol}-nonbinding", protocol, body(protocol, args.model, ANSWER, 4096)
            ),
        )
        check(
            protocol,
            "nonbinding-invariance",
            "limiter",
            lambda protocol=protocol, baseline=baseline, capped=capped: nonbinding_invariance(
                protocol, baseline, capped
            ),
        )
        check(
            protocol,
            "nonbinding-ledger",
            "limiter",
            lambda capped=capped: nonbinding_ledger(capped),
        )
        check(
            protocol,
            "unlimited-answer-quality",
            "quality",
            lambda protocol=protocol, baseline=baseline: exact_answer(protocol, baseline, "42"),
        )
        check(
            protocol,
            "nonbinding-answer-quality",
            "quality",
            lambda protocol=protocol, capped=capped: exact_answer(protocol, capped, "42"),
        )
        check(
            protocol,
            "forced-close",
            "limiter",
            lambda protocol=protocol: forced_check(protocol, False),
        )
        check(
            protocol,
            "stream-forced-close",
            "limiter",
            lambda protocol=protocol: forced_check(protocol, True),
        )
        check(protocol, "tool-control", "tool", lambda protocol=protocol: tool_check(protocol))

    def short_check():
        result = request(
            "chat-short",
            "chat",
            body("chat", args.model, "Your final answer must be the single word ready.", 8),
        )
        ledger(result[1], 8, False)
        return result

    short = check("chat", "short-control", "limiter", short_check)
    check("chat", "short-answer-quality", "quality", lambda: exact_answer("chat", short, "ready"))

    def schema_check() -> None:
        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer", "enum": [42]}},
            "required": ["value"],
            "additionalProperties": False,
        }
        constrained = body("chat", args.model, "Return an object with integer value 42.", 8)
        constrained["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": schema},
        }
        result, usage = request("chat-schema", "chat", constrained)
        ledger(usage, 8, False)
        import jsonschema

        jsonschema.validate(json.loads(normalized("chat", result)[0]), schema)

    check("chat", "schema-control", "schema", schema_check)

    # A bounded disconnect exercises cancellation at the transport boundary; a following
    # complete request checks that the single resident worker remains usable. Its exact
    # answer quality is independent of worker liveness and of the caller-stop case.
    check(
        "chat",
        "cancel",
        "transport",
        lambda: request(
            "chat-cancel", "chat", body("chat", args.model, LONG, 8, True, 64), cancel=True
        ),
    )

    def after_cancel_check():
        result = request(
            "chat-after-cancel", "chat", body("chat", args.model, ANSWER, 8, maximum=64)
        )
        ledger(result[1], 8, False)
        require(
            result[0]["choices"][0]["finish_reason"] == "stop",
            "post-disconnect request did not complete normally",
        )
        return result

    following = check("chat", "after-cancel", "transport", after_cancel_check)
    check(
        "chat",
        "after-cancel-answer-quality",
        "quality",
        lambda: exact_answer("chat", following, "42"),
    )

    def caller_stop_check() -> None:
        stopped_body = body("chat", args.model, ANSWER, 8, maximum=64)
        stopped_body["stop"] = ["42"]
        stopped, usage = request("chat-caller-stop", "chat", stopped_body)
        ledger(usage, 8, False)
        require(
            stopped["choices"][0]["finish_reason"] == "stop"
            and "42" not in normalized("chat", stopped)[0],
            "caller stop did not retain precedence",
        )

    check("chat", "caller-stop", "limiter", caller_stop_check)
    categories = sorted({result["category"] for result in results})
    print(
        json.dumps(
            {
                "checks": len(results),
                "passed": sum(r["passed"] for r in results),
                "categories": {
                    category: {
                        "checks": sum(r["category"] == category for r in results),
                        "passed": sum(r["category"] == category and r["passed"] for r in results),
                    }
                    for category in categories
                },
            }
        )
    )
    require(
        all(r["passed"] for r in results), "protocol qualification failed; inspect private summary"
    )


if __name__ == "__main__":
    main()
