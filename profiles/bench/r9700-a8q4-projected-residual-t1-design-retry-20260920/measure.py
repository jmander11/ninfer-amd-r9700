#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from preflight import verify
from prepare import (ATTEMPT, PACKAGE, PLAN, ROOT, compile_commands, digest,
                     identity, open_exclusive, write_json)


def retained_step(name: str, command: list[str], stdout_path: Path | None = None) -> dict:
    stdout_path = stdout_path or ATTEMPT / f"{name}.stdout"
    stderr_path = ATTEMPT / f"{name}.stderr"
    record = {"name": name, "argv": command, "started_unix_ns": time.time_ns(),
              "exit_code": None}
    process = None
    try:
        with open_exclusive(stdout_path) as stdout, open_exclusive(stderr_path) as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            record["pid"] = process.pid
            record["exit_code"] = process.wait()
        if record["exit_code"] != 0:
            raise RuntimeError(f"{name} exited {record['exit_code']}")
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        if process is not None and process.poll() is None:
            # Stop the entire compiler/qualifier process group before sealing evidence.
            import os
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if process is not None:
            record["exit_code"] = process.returncode
        raise
    finally:
        record["finished_unix_ns"] = time.time_ns()
        record["stdout"] = identity(stdout_path) if stdout_path.exists() else None
        record["stderr"] = identity(stderr_path) if stderr_path.exists() else None
        write_json(ATTEMPT / f"{name}.process.json", record)
    return record


def interrupted(signum, _frame):
    raise InterruptedError(f"measurement interrupted by signal {signum}")


def main() -> int:
    verify()
    started = time.time_ns()
    # The exclusive directory is the retry boundary; never touch an existing attempt.
    ATTEMPT.mkdir(mode=0o755)
    status = "failed"
    error_text = None
    previous_handlers = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        host, device, checker = compile_commands(ATTEMPT)
        static_receipt = ATTEMPT / "static-receipt.txt"
        compile_receipt = ATTEMPT / "compile-receipt.json"
        report = ATTEMPT / "qualification.json"
        qualifier = [str(ATTEMPT / "qual"), "--out-json", str(report), "--assembly",
                     str(ATTEMPT / "qual.s"), "--static-receipt", str(static_receipt),
                     "--compile-receipt", str(compile_receipt), "--package", str(PACKAGE)]
        steps = [retained_step("host-compile", host), retained_step("device-compile", device),
                 retained_step("static-check", checker, static_receipt)]
        write_json(compile_receipt, {
            "schema": "ninfer.r9700.immutable-compile-process-receipt.v1",
            "attempt": str(ATTEMPT), "create_only": True, "plan": identity(PLAN), "steps": steps,
            "outputs": {"executable": identity(ATTEMPT / "qual"),
                        "assembly": identity(ATTEMPT / "qual.s"),
                        "static_receipt": identity(static_receipt)},
            "measurement_process": {"argv": qualifier},
        })
        retained_step("qualifier", qualifier)
        qualification = json.loads(report.read_text(encoding="utf-8"))
        if qualification.get("status") != "qualified_for_whole_ab_only":
            raise RuntimeError("qualification report did not admit whole A/B")
        status = "passed"
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        print(f"projected_residual_retry_measure: FAIL {error_text}", file=sys.stderr)
    finally:
        # Normal errors and catchable interrupts close the attempt, including partial
        # compiler outputs. SIGKILL/power loss cannot be closed by any running harness.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        artifacts = [identity(path) for path in sorted(ATTEMPT.iterdir()) if path.is_file()]
        write_json(ATTEMPT / "closure.json", {
            "schema": "ninfer.r9700.projected-residual-attempt-closure.v1",
            "status": status, "error": error_text, "started_unix_ns": started,
            "finished_unix_ns": time.time_ns(), "plan": identity(PLAN),
            "create_only": True, "artifacts": artifacts,
        })
        with open_exclusive(ATTEMPT / "result.sha256") as stream:
            for path in sorted(ATTEMPT.iterdir()):
                if path.is_file() and path.name != "result.sha256":
                    stream.write(f"{digest(path)}  {path.relative_to(ROOT)}\n")
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if status != "passed":
        return 1
    print("projected_residual_retry_measure: PASS; whole C1 A/B preparation only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
