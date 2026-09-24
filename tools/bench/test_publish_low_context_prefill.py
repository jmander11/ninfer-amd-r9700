from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.bench import publish_low_context_prefill as publication


REPO = Path(__file__).resolve().parents[2]


def _arguments(root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    paths = tuple(root / name for name in (
        "pending.json", "output.json", "manifest.json", "bench", "artifact", "selection.json"
    ))
    for path in paths[2:]:
        path.write_text("source\n", encoding="utf-8")
    paths[0].write_text(json.dumps({"passes_p2048_gate": True}), encoding="utf-8")
    return paths


def test_publishes_exact_inode_and_removes_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending, output, manifest, executable, artifact, selection = _arguments(tmp_path)
    expected = {"passes_p2048_gate": True}
    monkeypatch.setattr(publication, "validate_ladder", lambda *args: expected)
    publication.publish(pending, output, manifest, executable, artifact, selection, 2000.0)
    assert json.loads(output.read_text()) == expected
    assert not pending.exists()


def test_preserves_occupied_or_dangling_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending, output, manifest, executable, artifact, selection = _arguments(tmp_path)
    output.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(
        publication, "validate_ladder", lambda *args: {"passes_p2048_gate": True}
    )
    with pytest.raises(ValueError, match="occupied"):
        publication.publish(pending, output, manifest, executable, artifact, selection, 2000.0)
    assert output.is_symlink()
    assert pending.exists()


def test_post_publish_validation_failure_rolls_back_owned_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending, output, manifest, executable, artifact, selection = _arguments(tmp_path)
    calls = 0

    def validate(*args):
        nonlocal calls
        calls += 1
        return ({"passes_p2048_gate": True} if calls == 1
                else {"passes_p2048_gate": False})

    monkeypatch.setattr(publication, "validate_ladder", validate)
    with pytest.raises(ValueError, match="published"):
        publication.publish(pending, output, manifest, executable, artifact, selection, 2000.0)
    assert not output.exists()
    assert not pending.exists()


def test_package_uses_importable_module_publisher() -> None:
    command = subprocess.run(
        [sys.executable, "-m", "tools.bench.publish_low_context_prefill", "--help"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert command.returncode == 0, command.stderr
    script = (
        REPO / "profiles/bench/low-context-selected-ladder-20260905/run-and-publish.sh"
    ).read_text(encoding="utf-8")
    assert "python3 -m tools.bench.publish_low_context_prefill" in script
    assert 'python3 "$publisher"' not in script
