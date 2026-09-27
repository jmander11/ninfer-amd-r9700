"""Real-engine checks for structured output, forced tools and candidate scoring.

Run against a server owned by the caller under the shared GPU lock. This client
never starts/stops a server or selects an artifact. Repeat for each admitted
ordinary/MTP/DFlash and concurrency configuration; report that configuration
alongside its output. Restart-persistence uses --save-response and --restore-response.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .serve_contract import (
    ContractError, json_response, parse_openai_stream, parse_responses_stream,
    request, response_text, wait_for_health,
)

EXPECTED = {"literal": "<tool_call><function=f>", "ok": True}
SCHEMA = {"type": "object", "properties": {
    "literal": {"const": EXPECTED["literal"]}, "ok": {"const": True}},
    "required": ["literal", "ok"], "additionalProperties": False}
FORMAT = {"type": "json_schema", "name": "result", "schema": SCHEMA, "strict": True}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def structured(base: str, model: str, index: int) -> dict:
    messages = [{"role": "user", "content": "Return the JSON value described by the schema. "
                 "Use at most one short sentence of reasoning, then emit the JSON. No explanation."}]
    common = {"model": model, "temperature": 0 if index < 2 else 1.0,
              "enable_thinking": index == 3}
    if index % 2 == 0:
        payload = {**common, "messages": messages, "max_completion_tokens": 512,
                   "response_format": {"type": "json_schema", "json_schema": {
                       key: value for key, value in FORMAT.items() if key != "type"}}}
        if index == 2:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            content, _, finish, _ = parse_openai_stream(request(base, "POST", "/v1/chat/completions", payload))
        else:
            reply = json_response(base, "POST", "/v1/chat/completions", payload)
            choice = reply["choices"][0]
            content, finish = choice["message"]["content"], choice["finish_reason"]
        require(finish == "stop", f"Chat JSON did not complete: {finish}")
    else:
        payload = {"model": model, "temperature": common["temperature"], "input": messages,
                   "reasoning": {"effort": "low" if index == 3 else "none"},
                   "max_output_tokens": 2048 if index == 3 else 512,
                   "text": {"format": FORMAT}, "store": False}
        if index == 3:
            payload["stream"] = True
            content, _, reply = parse_responses_stream(request(base, "POST", "/v1/responses", payload))
        else:
            reply = json_response(base, "POST", "/v1/responses", payload)
            content, _ = response_text(reply)
        if index == 3 and reply["status"] == "incomplete":
            require(reply.get("incomplete_details", {}).get("reason") == "max_output_tokens",
                    "thinking structured response has an unexpected incomplete reason")
            return {"case": index, "json": False, "qualification_status": "incomplete",
                    "reason": "thinking exhausted the output-token budget",
                    "usage": reply.get("usage"), "incomplete_details": reply["incomplete_details"]}
        require(reply["status"] == "completed", "Responses JSON did not complete")
        require(reply["text"]["format"] == FORMAT, "Responses did not echo the format")
    require(json.loads(content) == EXPECTED, f"wrong constrained value: {content!r}")
    return {"case": index, "json": True}


def forced_tools(base: str, model: str) -> None:
    function = {"type": "function", "name": "weather", "parameters": {
        "type": "object", "properties": {"city": {"const": "Regina"}},
        "required": ["city"], "additionalProperties": False}}
    other = {**function, "name": "other"}
    for choice in ("required", {"type": "function", "name": "weather"}):
        reply = json_response(base, "POST", "/v1/responses", {
            "model": model, "input": "Use weather for Regina, then stop.",
            "tools": [function] if choice == "required" else [function, other],
            "tool_choice": choice, "reasoning": {"effort": "none"}, "temperature": 0,
            "max_output_tokens": 128, "store": False})
        calls = [item for item in reply["output"] if item["type"] == "function_call"]
        require(bool(calls), "required/named request produced no complete call")
        require(all(call["name"] == "weather" and json.loads(call["arguments"]) == {"city": "Regina"}
                    for call in calls), "wrong forced function/arguments")


def score(base: str, model: str) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": "The capital of France is"}],
               "candidates": ["Paris.", "Tokyo."], "enable_thinking": False}
    batch = json_response(base, "POST", "/v1/score", payload)
    for index, candidate in enumerate(payload["candidates"]):
        single = json_response(base, "POST", "/v1/score", {**payload, "candidates": [candidate]})["data"][0]
        row = batch["data"][index]
        require(row["scored_tokens"] > 0 and row["scored_tokens"] == single["scored_tokens"],
                "candidate score token boundary changed")
        require(math.isclose(row["sum_nll"], single["sum_nll"], rel_tol=1e-6, abs_tol=1e-6),
                "batch and standalone continuation NLL differ")
        require(math.isclose(row["mean_nll"] * row["scored_tokens"], row["sum_nll"],
                             rel_tol=1e-6, abs_tol=1e-6), "mean/sum NLL disagree")
        require(row["log_probability"] == -row["sum_nll"], "log probability sign incorrect")
    return batch


def score_busy(base: str, model: str) -> None:
    payload = {"model": model, "messages": [{"role": "user", "content":
        "Count from 1 to 1000, writing every number. Do not summarize or explain."}],
        "enable_thinking": False, "temperature": 0, "max_completion_tokens": 2048,
        "stream": True}
    # Force a long remaining output after the first streamed content. An unconstrained
    # counting instruction may terminate early, making successful scoring a valid race.
    payload["messages"] = [{"role": "user", "content": "Return the schema's exact JSON value."}]
    payload["response_format"] = {"type": "json_schema", "json_schema": {
        "name": "busy", "strict": True, "schema": {
            "type": "object", "properties": {"text": {"const": "busy_" * 2048}},
            "required": ["text"], "additionalProperties": False}}}
    pending = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(pending, timeout=120) as stream:
        for line in stream:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices", [])
            if not choices or not choices[0].get("delta", {}).get("content"):
                continue
            scoring = urllib.request.Request(base + "/v1/score", data=json.dumps({
                "model": model, "messages": [{"role": "user", "content": "Say OK"}],
                "candidates": ["OK"], "enable_thinking": False}).encode(),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(scoring, timeout=120):
                    raise ContractError("score admitted while generation was active")
            except urllib.error.HTTPError as error:
                require(error.code == 429, f"busy scoring returned {error.code}")
            break
        else:
            raise ContractError("busy-score fixture produced no content")
    # Give the disconnected stream's next round boundary time to publish cancellation.
    # Later scoring retries only the expected busy status; no mathematical result is retried.
    deadline = time.monotonic() + 30
    while True:
        try:
            with urllib.request.urlopen(scoring, timeout=120) as response:
                require(len(json.load(response)["data"]) == 1, "score failed after cancellation")
            return
        except urllib.error.HTTPError as error:
            if error.code != 429 or time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def save_response(base: str, model: str, path: Path) -> None:
    reply = json_response(base, "POST", "/v1/responses", {
        "model": model, "input": "Remember the word violet. Reply OK.",
        "reasoning": {"effort": "none"}, "temperature": 0, "max_output_tokens": 32})
    require(reply["status"] == "completed", "stored response did not complete")
    path.write_text(json.dumps(reply, indent=2) + "\n")


def restore_response(base: str, model: str, path: Path) -> None:
    saved = json.loads(path.read_text())
    restored = json_response(base, "GET", "/v1/responses/" + saved["id"])
    require(restored == saved, "restart changed stored protocol object")
    branch = json_response(base, "POST", "/v1/responses", {
        "model": model, "previous_response_id": saved["id"], "input": "Reply OK again.",
        "reasoning": {"effort": "none"}, "temperature": 0, "max_output_tokens": 32})
    require(branch["status"] == "completed", "restored continuation failed")
    deleted = json_response(base, "DELETE", "/v1/responses/" + saved["id"])
    require(deleted["deleted"], "stored parent deletion failed")
    descendant = json_response(base, "POST", "/v1/responses", {
        "model": model, "previous_response_id": branch["id"], "input": "Reply OK.",
        "reasoning": {"effort": "none"}, "temperature": 0, "max_output_tokens": 32})
    require(descendant["status"] == "completed", "deleting parent broke descendant")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--save-response", type=Path)
    parser.add_argument("--restore-response", type=Path)
    args = parser.parse_args()
    wait_for_health(args.base_url, 300)
    if args.restore_response:
        restore_response(args.base_url, args.model, args.restore_response)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        values = list(pool.map(lambda i: structured(args.base_url, args.model, i), range(4)))
    forced_tools(args.base_url, args.model)
    scores = score(args.base_url, args.model)
    score_busy(args.base_url, args.model)
    structured(args.base_url, args.model, 0)  # Generation after scoring must still work.
    if args.save_response:
        save_response(args.base_url, args.model, args.save_response)
    complete = all(value["json"] for value in values)
    print(json.dumps({"qualification_status": "passed" if complete else "incomplete",
                      "structured": values, "forced_tools": True, "scores": scores,
                      "generation_after_score": True, "score_busy_admission": True}, indent=2))
    if not complete:
        sys.exit(3)  # Inconclusive output-budget exhaustion, distinct from an assertion failure.


if __name__ == "__main__":
    main()
