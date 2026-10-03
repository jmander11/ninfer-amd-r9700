#!/usr/bin/env python3
"""Run ROCm's clang-tidy over NInfer's C++ and HIP translation units.

The analyzer is the clang-tidy shipped with the selected ROCm toolchain (/opt/rocm/lib/llvm), the
same LLVM that compiles the gfx1201 HIP code, so HIP compile commands from the CMake tree's
compile_commands.json are analyzed as they are; host C++ commands come from GCC and only gain
-Wno-unknown-warning-option. Diagnostics from headers are reported once, not once per including
TU. Configuration: .clang-tidy; policy: docs/maintainer/code-quality.md.

Usage:
  ./scripts/run-clang-tidy.py                    # every project TU
  ./scripts/run-clang-tidy.py --changed          # diagnostics on lines changed vs. origin/HEAD
  ./scripts/run-clang-tidy.py --changed HEAD~3   # diagnostics on lines changed since a ref
  ./scripts/run-clang-tidy.py src/serve/foo.cpp  # named TUs (headers select their includers)

--changed analyzes every TU that is or includes a changed file and reports only diagnostics on
changed lines. Exit status is 1 when any diagnostic is reported.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLANG_TIDY = Path(os.environ.get("NINFER_CLANG_TIDY", "/opt/rocm/lib/llvm/bin/clang-tidy"))
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".hip"}
HEADER_SUFFIXES = {".h", ".hpp"}
# Third-party sources, including the patched copies CMake generates inside build trees.
EXCLUDED_PREFIXES = ("third_party/", "build/", "build-")
MAX_JOBS = 8
DIAGNOSTIC = re.compile(
    r"^(?P<path>[^\s:][^:]*):(?P<line>\d+):\d+: (?:warning|error): .*\[[\w.,-]+\]$"
)
HUNK = re.compile(r"^@@ -\S+ \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")
# Device index arithmetic is 32-bit by design and bounded by each kernel's documented shape
# contract, so the widening checks apply to host translation units only (see .clang-tidy).
HIP_DISABLED_CHECKS = (
    "-bugprone-implicit-widening-of-multiplication-result,-bugprone-misplaced-widening-cast"
)
GIT = ["git", "-C", str(ROOT)]


def clang_tidy() -> str:
    if not CLANG_TIDY.is_file():
        sys.exit(
            f"{CLANG_TIDY} not found; install the selected ROCm toolchain or set NINFER_CLANG_TIDY"
        )
    return str(CLANG_TIDY)


def project_relative(path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None
    if relative.startswith(EXCLUDED_PREFIXES):
        return None
    return relative


def write_database(build: Path, output: Path) -> dict[str, dict]:
    """Write the project subset of the compile database and return its entries by path."""
    entries = json.loads((build / "compile_commands.json").read_text())
    selected: dict[str, dict] = {}
    for entry in entries:
        source = Path(entry["directory"], entry["file"])
        relative = project_relative(source)
        if relative is None or source.suffix not in SOURCE_SUFFIXES:
            continue
        arguments = entry.get("arguments") or shlex.split(entry["command"])
        # GCC-only host warning flags are not errors for the clang front end.
        arguments = [*arguments, "-Wno-unknown-warning-option"]
        selected[relative] = {
            "directory": entry["directory"],
            "file": str(source),
            "arguments": arguments,
        }
    output.mkdir(parents=True, exist_ok=True)
    # Write then rename, so concurrent runs never read a partially written database.
    staging = output / f"compile_commands.json.{os.getpid()}"
    staging.write_text(json.dumps(list(selected.values()), indent=1))
    staging.replace(output / "compile_commands.json")
    return selected


def header_includers(build: Path, headers: set[str], sources: set[str]) -> set[str]:
    """Map changed headers to the TUs that include them, from Ninja's recorded dependencies."""
    if not headers:
        return set()
    deps = subprocess.run(
        ["ninja", "-C", str(build), "-t", "deps"], capture_output=True, text=True, check=False
    )
    if deps.returncode != 0:
        sys.exit("header changes need a built tree; run the build first (ninja -t deps failed)")
    selected: set[str] = set()
    current: list[str] = []
    for line in [*deps.stdout.splitlines(), ""]:
        if line and not line.startswith(" "):
            current = []
            continue
        if line.strip():
            current.append(line.strip())
            continue
        if not current:
            continue
        paths = {project_relative(Path(build, dep)) for dep in current}
        if paths & headers:
            selected |= paths & sources
        current = []
    return selected


def changed_lines(base: str | None) -> dict[str, set[int] | None]:
    """Changed project files mapped to their added or modified lines (None: the whole file)."""
    merge_base = subprocess.run(
        [*GIT, "merge-base", "HEAD", base or "origin/HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if merge_base.returncode != 0:
        sys.exit(
            f"cannot find a merge base with {base or 'origin/HEAD'}; pass --changed REF "
            "(or run `git remote set-head origin --auto`)"
        )
    diff = subprocess.run(
        [*GIT, "diff", "-U0", "--no-color", "--diff-filter=d", merge_base.stdout.strip()],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    changed: dict[str, set[int] | None] = {}
    current: set[int] | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            name = line[len("+++ b/") :]
            current = (
                None if name.startswith(EXCLUDED_PREFIXES) else changed.setdefault(name, set())
            )
        elif current is not None and (hunk := HUNK.match(line)):
            start = int(hunk["start"])
            current.update(range(start, start + int(hunk["count"] or 1)))
    untracked = subprocess.run(
        [*GIT, "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    for name in untracked:
        if not name.startswith(EXCLUDED_PREFIXES):
            changed[name] = None
    return changed


def run_one(tidy: str, database: Path, source: str, fix: bool) -> str:
    command = [tidy, "--quiet", f"-p={database}", str(ROOT / source)]
    if source.endswith(".hip"):
        command.append(f"--checks={HIP_DISABLED_CHECKS}")
    if fix:
        command.append("--fix")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout


def in_scope(path: str, line: int, lines: dict[str, set[int] | None] | None) -> bool:
    """Project files only (not system or third-party headers), and changed lines if given."""
    relative = project_relative(Path(path))
    if relative is None:
        return False
    if lines is None:
        return True
    if relative not in lines:
        return False
    changed = lines[relative]
    return changed is None or line in changed


def split_diagnostics(
    output: str, lines: dict[str, set[int] | None] | None
) -> list[tuple[str, str]]:
    """Split clang-tidy output into diagnostic blocks, keeping only changed lines if given."""
    blocks: list[tuple[str, list[str]]] = []
    keep = False
    for line in output.splitlines():
        if diagnostic := DIAGNOSTIC.match(line):
            keep = in_scope(diagnostic["path"], int(diagnostic["line"]), lines)
            if keep:
                blocks.append((line, [line]))
        elif keep and blocks:
            blocks[-1][1].append(line)
    return [(head, "\n".join(body)) for head, body in blocks]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-p",
        "--build-dir",
        type=Path,
        default=ROOT / "build-r9700",
        help="CMake tree with compile_commands.json (default: build-r9700)",
    )
    parser.add_argument("-j", "--jobs", type=int, default=MAX_JOBS)
    parser.add_argument(
        "--changed",
        nargs="?",
        const="",
        metavar="REF",
        help="report only lines changed since REF (default: origin/HEAD)",
    )
    parser.add_argument("--fix", action="store_true", help="apply clang-tidy fix-its")
    parser.add_argument("files", nargs="*", help="source or header paths to check")
    args = parser.parse_args()
    if not 1 <= args.jobs <= MAX_JOBS:
        sys.exit(f"--jobs must be 1..{MAX_JOBS} on the shared host")
    if args.fix:
        args.jobs = 1  # concurrent fix-its would edit a shared header more than once

    tidy = clang_tidy()
    build = args.build_dir.resolve()
    if not (build / "compile_commands.json").is_file():
        sys.exit(f"{build}/compile_commands.json not found; configure the CMake tree first")
    database = build / "clang-tidy"
    sources = set(write_database(build, database))

    lines: dict[str, set[int] | None] | None = None
    if args.changed is not None:
        lines = changed_lines(args.changed or None)
        requested = set(lines)
    elif args.files:
        requested = {project_relative(Path(name)) or name for name in args.files}
    else:
        requested = sources
    headers = {name for name in requested if Path(name).suffix in HEADER_SUFFIXES}
    selected = sorted((requested & sources) | header_includers(build, headers, sources))
    if not selected:
        print("clang-tidy: no project translation units selected")
        return 0

    version = subprocess.run([tidy, "--version"], capture_output=True, text=True, check=True)
    release = re.search(r"version (\S+)", version.stdout)
    print(
        f"clang-tidy {release[1] if release else '?'}: {len(selected)} translation units",
        flush=True,
    )
    seen: set[str] = set()
    reported = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, tidy, database, s, args.fix) for s in selected]
        for future in concurrent.futures.as_completed(futures):
            for head, block in split_diagnostics(future.result(), lines):
                if head in seen:
                    continue
                seen.add(head)
                reported += 1
                print(block, flush=True)
    print(f"clang-tidy: {reported} diagnostics")
    return 1 if reported else 0


if __name__ == "__main__":
    sys.exit(main())
