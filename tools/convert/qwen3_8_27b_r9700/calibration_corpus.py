"""Build the fixed-length token-id calibration set for activation-aware FP8LUT4 conversion.

The input is a JSON list of groups `{"label", "kind", "sequences", "files"[, "max_file_tokens"]}`.
`kind` "text" tokenizes each file with the checkpoint tokenizer (no special tokens), optionally
truncated to its first `max_file_tokens` tokens, and joins the files in order with a blank line
into one stream. `kind` "chat" renders each file (an OpenAI-style request with `messages` and
optional `tools`, or a JSONL of `{"body": request}` records, of which the last is used) through the
checkpoint chat template, one stream per file. Each stream offers its non-overlapping
`--length`-token windows; a group takes its `sequences` windows round-robin over its streams,
evenly spaced within each stream. A window is rejected when any of its 32-token spans occurs in an
`--exclude` id file (the PPL evaluation corpora) or in an already selected window. The output holds
one window per line; the manifest records every input file's SHA-256 and the per-group counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SPAN = 32


def _spans(ids: list[int]) -> set[int]:
    return {hash(tuple(ids[i:i + SPAN])) for i in range(0, max(0, len(ids) - SPAN + 1))}


def _chat_request(path: Path) -> dict:
    if path.suffix == ".jsonl":
        return json.loads(path.read_text().strip().splitlines()[-1])["body"]
    return json.loads(path.read_text())


def _messages(messages: list[dict]) -> list[dict]:
    """Tool-call arguments as mappings, as the chat template requires."""
    result = []
    for message in messages:
        message = json.loads(json.dumps(message))
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            if isinstance(function.get("arguments"), str):
                try:
                    function["arguments"] = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    function["arguments"] = {"raw": function["arguments"]}
        result.append(message)
    return result


def _windows(stream: list[int], length: int, wanted: int) -> list[list[int]]:
    """The stream's non-overlapping windows, evenly spaced first, then the rest."""
    count = len(stream) // length
    step = max(1, count // max(1, wanted))
    order = [i for phase in range(step) for i in range(phase, count, step)]
    return [stream[i * length:(i + 1) * length] for i in order]


def build(sources: list[dict], model: Path, length: int,
          exclude: list[Path]) -> tuple[list[list[int]], list[dict]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model))

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    rejected: set[int] = set()
    for path in exclude:
        rejected |= _spans([int(token) for token in path.read_text().split()])
    sequences: list[list[int]] = []
    manifest: list[dict] = []
    for group in sources:
        streams: list[list[int]] = []
        used: list[dict] = []
        if group["kind"] == "text":
            stream: list[int] = []
            for name in group["files"]:
                if len(stream) >= 2 * group["sequences"] * length:
                    break
                data = Path(name).read_bytes()
                if stream:
                    stream.extend(encode("\n\n"))
                stream.extend(encode(data.decode("utf-8", errors="replace"))[:group.get("max_file_tokens")])
                used.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest()))
            streams.append(stream)
            orders = [[stream[i:i + length] for i in range(0, len(stream) - length + 1, length)]]
        elif group["kind"] == "chat":
            for name in group["files"]:
                request = _chat_request(Path(name))
                streams.append(encode(tokenizer.apply_chat_template(
                    _messages(request["messages"]), tools=request.get("tools"), tokenize=False)))
                used.append(dict(path=name, sha256=hashlib.sha256(Path(name).read_bytes()).hexdigest()))
            per_stream = -(-group["sequences"] // len(streams))
            orders = [_windows(stream, length, per_stream) for stream in streams]
        else:
            raise ValueError(f"unknown calibration group kind {group['kind']}")
        taken, skipped = 0, 0
        while taken < group["sequences"] and any(orders):
            for order in orders:
                if not order or taken == group["sequences"]:
                    continue
                window = order.pop(0)
                spans = _spans(window)
                if spans & rejected:
                    skipped += 1
                    continue
                rejected |= spans
                sequences.append(window)
                taken += 1
        if taken < group["sequences"]:
            raise ValueError(f"group {group['label']} provides {taken} of {group['sequences']} windows")
        manifest.append(dict(label=group["label"], kind=group["kind"], sequences=taken,
                             rejected_windows=skipped, files=used))
    return sequences, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True,
                        help="checkpoint directory (tokenizer and chat template)")
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="token-id file whose 32-token spans no window may contain")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = Path(str(args.out) + ".manifest.json")
    if args.out.exists() or manifest_path.exists():
        raise FileExistsError(args.out)
    sequences, groups = build(json.loads(args.sources.read_text()), args.model, args.length, args.exclude)
    text = "".join(" ".join(map(str, sequence)) + "\n" for sequence in sequences)
    args.out.write_text(text)
    manifest = dict(length=args.length, sequences=len(sequences),
                    ids_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    excluded=[str(path) for path in args.exclude], groups=groups)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("length", "sequences", "ids_sha256")}),
          json.dumps([(g["label"], g["sequences"], g["rejected_windows"]) for g in groups]))


if __name__ == "__main__":
    main()
