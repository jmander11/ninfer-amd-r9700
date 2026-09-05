#!/usr/bin/env python3
"""Prepare one post-terminal selected-artifact Vision BF16 diagnostic."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shlex
import stat
import subprocess
import tempfile
from pathlib import Path

from tools.bench.prepare_selected_niah import resolve_route
from tools.reference.qwen3_8_27b_bf16.protocol import validate_checkpoint_files


REPO = Path(__file__).resolve().parents[2]
PREPARE_INPUT = REPO / "tools/parity/qwen3_8_27b/prepare_vision_input.py"
VISION = REPO / "tools/parity/qwen3_8_27b/vision.py"
VALIDATOR = REPO / "tools/bench/validate_selected_vision_diagnostic.py"
PROTOCOL = REPO / "tools/reference/qwen3_8_27b_bf16/protocol.py"
SOURCE = Path("/ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16")
SOURCE_RECEIPT = REPO / "profiles/ppl/r9700-bf16-source-checkpoint-preflight-20260905.json"
MESSAGES = REPO / "examples/cli/messages/image_chart.json"
MEDIA = REPO / "examples/cli/media/visual_chart.png"
FRONTEND_PYTHON = Path("/ssdpool2nvme/local_llm/ninfer-dylan2/eval/.venv/bin/python")
GPU_PYTHON = Path("/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python")
POWER = Path("/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level")
REFERENCE_AUTHORITIES = tuple(REPO / path for path in (
    "tools/reference/qwen3_8_27b/bindings.py",
    "tools/reference/qwen3_8_27b/weights.py",
    "tools/reference/qwen3_8_27b/vision.py",
    "tools/reference/qwen3/common/frontend.py",
    "tools/reference/qwen3/common/multimodal.py",
    "tools/convert/qwen3_8_27b_r9700/inventory.py",
    "tools/convert/qwen3_8_27b_r9700/q4_inventory.py",
    "tools/convert/qwen3_8_27b_r9700/q4_w8_mse_inventory.py",
    "tools/convert/qwen3_8_27b_r9700/fp8_hybrid_inventory.py",
))
AT_FDCWD = -100
RENAME_NOREPLACE = 1


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"required file is not regular: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def inspect_python(path: Path, modules: tuple[str, ...]) -> dict[str, object]:
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Python launcher is unavailable: {path}")
    script = (
        "import importlib.metadata,json,sys; "
        f"names={list(modules)!r}; "
        "[__import__(name) for name in names]; "
        "print(json.dumps({'version':'.'.join(map(str,sys.version_info[:3])),"
        "'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
        "'packages':{name:importlib.metadata.version('pillow' if name=='PIL' else name) "
        "for name in names}}))"
    )
    completed = subprocess.run(
        [str(path), "-c", script], check=True, capture_output=True, text=True,
        env={**os.environ, "LD_LIBRARY_PATH": "/opt/rocm/lib:/opt/rocm/core-10.0/lib"},
    )
    details = json.loads(completed.stdout)
    pyvenv = Path(details["prefix"]) / "pyvenv.cfg"
    return {
        "launcher": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)},
        **details,
        "pyvenv_cfg": file_identity(pyvenv) if pyvenv.is_file() else None,
    }


def validate_source_receipt() -> dict[str, object]:
    if SOURCE_RECEIPT.is_symlink():
        raise ValueError("BF16 source receipt must not be a symlink")
    receipt_path = SOURCE_RECEIPT.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    weight_map = validate_checkpoint_files(SOURCE)
    shards = sorted(set(weight_map.values()))
    if any(
        (SOURCE / name).is_symlink()
        or not stat.S_ISREG(os.stat(SOURCE / name, follow_symlinks=False).st_mode)
        for name in shards
    ):
        raise ValueError("BF16 source shards must be regular non-symlink files")
    expected_files = [
        {"name": name, "bytes": (SOURCE / name).stat().st_size} for name in shards
    ]
    if (
        receipt.get("artifact_type") != "ninfer_qwen3_8_27b_bf16_checkpoint_preflight"
        or receipt.get("status") != "passed"
        or Path(receipt.get("source", "")).resolve(strict=True) != SOURCE.resolve(strict=True)
        or receipt.get("validator", {}).get("path") != str(PROTOCOL.resolve(strict=True))
        or receipt.get("validator", {}).get("sha256") != sha(PROTOCOL)
        or receipt.get("validator", {}).get("call") != "validate_checkpoint_files"
        or receipt.get("metadata", {}).get("indexed_tensor_count") != len(weight_map)
        or receipt.get("shards", {}).get("expected_count") != 18
        or receipt.get("shards", {}).get("files") != expected_files
        or receipt.get("shards", {}).get("all_named_exactly") is not True
        or receipt.get("shards", {}).get("all_regular_nonempty") is not True
    ):
        raise ValueError("BF16 source receipt differs from the current checkpoint protocol")
    return {"receipt": file_identity(receipt_path), "source": str(SOURCE.resolve())}


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.renameat2(
        AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(destination), RENAME_NOREPLACE
    ) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def prepare(selection: Path, output: Path) -> dict[str, object]:
    selection = selection.resolve(strict=True)
    output = output.parent.resolve(strict=True) / output.name
    if os.path.lexists(output):
        raise ValueError(f"output directory already exists: {output}")
    route = resolve_route(selection)
    source = validate_source_receipt()
    frontend_python = inspect_python(
        FRONTEND_PYTHON, ("torch", "safetensors", "transformers", "torchvision", "PIL")
    )
    gpu_python = inspect_python(GPU_PYTHON, ("torch", "safetensors", "transformers"))
    messages = file_identity(MESSAGES)
    media = file_identity(MEDIA)
    artifact = Path(route["artifact"]["path"])
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.prepare-", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        staged.mkdir()
        staged_input = staged / "prepared-input.safetensors"
        completed = subprocess.run(
            [str(FRONTEND_PYTHON), "-m", "tools.parity.qwen3_8_27b.prepare_vision_input",
             "--weights", str(artifact),
             "--messages", str(MESSAGES), "--out", str(staged_input)],
            cwd=REPO, check=True, capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        input_contract = json.loads(completed.stdout.splitlines()[-1])
        final_input = output / staged_input.name
        prepared_input = {
            "path": str(final_input), "bytes": staged_input.stat().st_size,
            "sha256": sha(staged_input), "contract": input_contract,
        }
        plan = {
            "artifact_type": "ninfer_r9700_selected_vision_diagnostic_plan",
            "schema_version": 1,
            "status": "command_only_not_executed",
            "terminal_route": route,
            "source_checkpoint": source,
            "frontend_python": frontend_python,
            "gpu_python": gpu_python,
            "fixture": {"messages": messages, "media": media},
            "prepared_input": prepared_input,
            "workload": {
                "maximum_concurrency": 1, "thinking": False, "prefix_reuse": False,
                "speculative_decode": False, "images": 1, "videos": 0,
                "capture_names": ["block_00", "block_13", "block_26", "merger"],
                "gate": "diagnostic completion with finite exact-shape comparisons; no numeric threshold",
            },
            "outputs": {
                "raw": str(output / "vision.raw.json"),
                "admission": str(output / "admission.json"),
            },
        }
        plan_path = staged / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        commands_path = staged / "commands.sh"
        commands_path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\nset -o noclobber\n"
            f"readonly root={shlex.quote(str(output))}\n"
            f"readonly power={shlex.quote(str(POWER))}\n"
            "cd " + shlex.quote(str(REPO)) + "\n"
            'sha256sum --check --strict "$root/prepared.sha256"\n'
            'test "$(cat "$power")" = auto\n'
            'for path in "$root/vision.raw.json" "$root/admission.json"; do '
            'test ! -e "$path" && test ! -L "$path"; done\n'
            "LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib "
            f"{shlex.quote(str(GPU_PYTHON))} -m tools.parity.qwen3_8_27b.vision "
            f"--weights {shlex.quote(str(artifact))} --model-dir {shlex.quote(str(SOURCE))} "
            f"--messages {shlex.quote(str(MESSAGES))} --prepared-input \"$root/prepared-input.safetensors\" "
            '--device cuda:0 --no-thinking --output "$root/vision.raw.json"\n'
            'test "$(cat "$power")" = auto\n'
            f"python3 -m tools.bench.validate_selected_vision_diagnostic --plan \"$root/plan.json\" "
            '--root "$root" --out "$root/admission.json"\n',
            encoding="utf-8",
        )
        os.chmod(commands_path, 0o755)
        closure_sources = [
            Path(__file__).resolve(), VALIDATOR, PREPARE_INPUT, VISION, PROTOCOL,
            selection, artifact, MESSAGES, MEDIA, SOURCE_RECEIPT,
            Path(route["source_matrices"]["pareto-capacity"]["path"]),
            Path(route["source_matrices"]["pareto-whole"]["path"]),
            Path(route["build_identity"]["cmake_cache"]["path"]),
            Path(route["build_identity"]["ctest_root"]["path"]),
            *REFERENCE_AUTHORITIES,
        ]
        if route["hybrid_width_tool"]:
            closure_sources.append(Path(route["hybrid_width_tool"]["path"]))
        closure = [
            *( (path, path) for path in closure_sources ),
            (staged_input, final_input), (plan_path, output / "plan.json"),
            (commands_path, output / "commands.sh"),
        ]
        (staged / "prepared.sha256").write_text(
            "".join(f"{sha(source_path)}  {published.relative_to(REPO)}\n"
                    for source_path, published in closure),
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
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError,
            json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
