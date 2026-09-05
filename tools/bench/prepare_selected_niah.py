#!/usr/bin/env python3
"""Prepare the exact post-terminal 64K NIAH admission campaign."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import shlex
import tempfile
from pathlib import Path

from tools.bench.run_niah_check import DEFAULT_NEEDLE, file_identity, matrix_cases, resolve_fixture

REPO = Path(__file__).resolve().parents[2]
ROUTE_RESOLVER = (
    REPO / "profiles/bench/post-terminal-focused-verification-20260905/resolve.py"
)
RUN_NIAH = REPO / "tools/bench/run_niah_check.py"
VALIDATOR = REPO / "tools/bench/validate_selected_niah.py"
POSITIONS = ("start", "q25", "mid", "q75", "end")
AT_FDCWD = -100
RENAME_NOREPLACE = 1


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(
        AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(destination), RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def resolve_route(selection: Path) -> dict:
    spec = importlib.util.spec_from_file_location("ninfer_terminal_route", ROUTE_RESOLVER)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load terminal route resolver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve(selection)


def prepare(selection: Path, output: Path) -> dict:
    selection = selection.resolve(strict=True)
    output = output.parent.resolve(strict=True) / output.name
    if os.path.lexists(output):
        raise ValueError(f"output directory already exists: {output}")
    route = resolve_route(selection)
    artifact = Path(route["artifact"]["path"]).resolve(strict=True)
    bench = Path(route["benchmark"]["path"]).resolve(strict=True)
    build = Path(route["build_directory"]).resolve(strict=True)
    serve = (build / "apps/ninfer-serve").resolve(strict=True)
    if bench != build / "bench/ninfer_bench":
        raise ValueError("terminal benchmark does not belong to selected build")
    if serve != build / "apps/ninfer-serve" or not os.access(serve, os.X_OK):
        raise ValueError("selected build lacks its executable serving binary")
    fixtures = []
    for label, ref in matrix_cases(["64k"], list(POSITIONS)):
        path = resolve_fixture(ref).resolve(strict=True)
        fixtures.append({"label": label, "ref": ref, **file_identity(path)})

    with tempfile.TemporaryDirectory(prefix=f".{output.name}.prepare-", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        staged.mkdir()
        plan = {
            "artifact_type": "ninfer_r9700_selected_niah_plan",
            "schema_version": 1,
            "status": "command_only_not_executed",
            "terminal_route": route,
            "workload": {
                "model": "qwen3.8-27b", "length": "64k",
                "positions": list(POSITIONS), "runs_per_cell": 1,
                "max_tokens": 64, "thinking": False, "needle": DEFAULT_NEEDLE,
                "answer_match": "exact",
                "maximum_concurrency": 1,
            },
            "fixtures": fixtures,
            "server": {
                **file_identity(serve),
                "host": "127.0.0.1", "port": 18081,
                "max_context": 262144, "kv_capacity": 262144,
                "max_concurrency": 1, "prefix_reuse": False,
            },
            "outputs": {
                "server_log": str(output / "server.requests.jsonl"),
                "server_stdout": str(output / "server.stdout.log"),
                "server_stderr": str(output / "server.stderr.log"),
                "evidence": str(output / "niah.evidence.json"),
                "admission": str(output / "admission.json"),
            },
        }
        plan_path = staged / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        commands_path = staged / "commands.sh"
        commands_path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\nset -o noclobber\n\n"
            f"readonly root={shlex.quote(str(output))}\n"
            f"readonly serve={shlex.quote(str(serve))}\n"
            f"readonly artifact={shlex.quote(str(artifact))}\n"
            'cd "' + str(REPO) + '"\n'
            'sha256sum --check --strict "$root/prepared.sha256"\n'
            'for path in "$root/server.requests.jsonl" "$root/server.stdout.log" '
            '"$root/server.stderr.log" "$root/niah.evidence.json" "$root/admission.json"; do\n'
            '  test ! -e "$path" && test ! -L "$path"\n'
            'done\n'
            'test -x "$serve"\n'
            '"$serve" "$artifact" --host 127.0.0.1 --port 18081 '
            '--model-id qwen3.8-27b --max-context 262144 --kv-capacity 262144 '
            '--max-concurrency 1 --prefill-chunk ' + str(route["selected_prefill_chunk"]) + ' '
            '--no-prefix-reuse --no-thinking --request-log-jsonl "$root/server.requests.jsonl" '
            '>"$root/server.stdout.log" 2>"$root/server.stderr.log" &\n'
            'readonly server_pid=$!\n'
            'cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }\n'
            'trap cleanup EXIT\n'
            "NINFER_SERVER_PID=\"$server_pid\" python3 - <<'PY'\n"
            "import json, os, time, urllib.request\n"
            "pid = int(os.environ['NINFER_SERVER_PID'])\n"
            "deadline = time.monotonic() + 300\n"
            "last = None\n"
            "while time.monotonic() < deadline:\n"
            "    try:\n"
            "        with urllib.request.urlopen('http://127.0.0.1:18081/health', timeout=2) as r:\n"
            "            if json.loads(r.read()) == {'status': 'ok'}:\n"
            "                break\n"
            "    except Exception as error:\n"
            "        last = error\n"
            "    try:\n"
            "        os.kill(pid, 0)\n"
            "    except OSError:\n"
            "        raise SystemExit('selected server exited before becoming healthy')\n"
            "    time.sleep(1)\n"
            "else:\n"
            "    raise SystemExit(f'server health timeout: {last}')\n"
            "PY\n"
            "python3 -m tools.bench.run_niah_check --base http://127.0.0.1:18081 "
            "--model qwen3.8-27b --key local-niah-gate --lengths 64k "
            "--positions start,q25,mid,q75,end --runs 1 --max-tokens 64 --exact-answer "
            '--server-log "$root/server.requests.jsonl" --artifact "$artifact" '
            f"--serve-bin \"$serve\" --selection {shlex.quote(str(selection))} "
            '--out "$root/niah.evidence.json"\n'
            'cleanup\ntrap - EXIT\n'
            "python3 -m tools.bench.validate_selected_niah --plan \"$root/plan.json\" "
            '--root "$root" --out "$root/admission.json"\n',
            encoding="utf-8",
        )
        closure = [
            (Path(__file__).resolve(), Path(__file__).resolve()),
            (VALIDATOR, VALIDATOR), (RUN_NIAH, RUN_NIAH),
            (ROUTE_RESOLVER, ROUTE_RESOLVER), (selection, selection),
            (artifact, artifact), (bench, bench), (serve, serve),
            (plan_path, output / "plan.json"),
            (commands_path, output / "commands.sh"),
            *((Path(row["path"]), Path(row["path"])) for row in fixtures),
            *((Path(row["path"]), Path(row["path"]))
              for row in route["source_matrices"].values()),
            *([(Path(route["hybrid_width_tool"]["path"]),
                Path(route["hybrid_width_tool"]["path"]))]
              if route["hybrid_width_tool"] else []),
        ]
        (staged / "prepared.sha256").write_text(
            "".join(
                f"{sha(source)}  {published.relative_to(REPO)}\n"
                for source, published in closure
            ),
            encoding="utf-8",
        )
        rename_noreplace(staged, output)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        prepare(args.selection, args.out)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
