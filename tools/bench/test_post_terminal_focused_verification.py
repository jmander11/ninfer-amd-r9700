import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest


RESOLVER = (
    Path(__file__).resolve().parents[2]
    / "profiles/bench/post-terminal-focused-verification-20260905/resolve.py"
)
SPEC = importlib.util.spec_from_file_location("post_terminal_focused_resolve", RESOLVER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_interpreter_identity_preserves_explicit_launcher() -> None:
    launcher = Path(sys.executable)

    identity = MODULE.inspect_test_python(launcher)

    assert identity["launcher_path"] == str(launcher)
    assert identity["python_version"] == ".".join(map(str, sys.version_info[:3]))
    assert set(identity["packages"]) == {"pytest", "torch", "safetensors"}
    assert identity["prefix"] == sys.prefix


def test_resolver_rejects_dangling_output_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "route.json"
        output.symlink_to(root / "missing.json")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "resolve.py",
                "--selection",
                str(root / "not-opened.json"),
                "--test-python",
                sys.executable,
                "--out",
                str(output),
            ],
        )
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            MODULE.main()
        assert output.is_symlink()
