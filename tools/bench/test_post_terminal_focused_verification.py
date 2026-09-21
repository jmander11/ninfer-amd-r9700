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


def test_hybrid_identity_is_the_registered_n16_artifact() -> None:
    assert MODULE.is_hybrid_weights("r9700-q4g64-f8e4m3-four-role-n16k16-eval")
    assert not MODULE.is_hybrid_weights("r9700-q4g64-f8e4m3-four-role-eval")


def test_selected_dense_build_ignores_inactive_sparse_parameters(tmp_path: Path) -> None:
    cache = tmp_path / "CMakeCache.txt"
    settings = {
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_GENERATOR": "Ninja",
        "NINFER_R9700_KV_VALUE_GROUP": "16",
        "NINFER_R9700_Q4_ACTIVATION_BITS": "8",
        "NINFER_R9700_W8_ACTIVATION_BITS": "8",
        "NINFER_R9700_FP8_QK_WMMA": "1",
        "NINFER_R9700_XATTENTION_QUALIFICATION": "OFF",
        "NINFER_R9700_XATTENTION_STRIDE": "16",
        "NINFER_R9700_XATTENTION_TAU_PERMILLE": "1000",
    }

    def write():
        cache.write_text("".join(f"{key}:STRING={value}\n" for key, value in settings.items()))

    write()
    MODULE.validate_build_profile(cache, 16, False)
    settings["NINFER_R9700_XATTENTION_QUALIFICATION"] = "ON"
    write()
    with pytest.raises(ValueError, match="TAU_PERMILLE"):
        MODULE.validate_build_profile(cache, 16, True)
    settings["NINFER_R9700_XATTENTION_TAU_PERMILLE"] = "900"
    write()
    MODULE.validate_build_profile(cache, 16, True)
    settings["NINFER_R9700_XATTENTION_STRIDE"] = "8"
    write()
    with pytest.raises(ValueError, match="STRIDE"):
        MODULE.validate_build_profile(cache, 16, True)


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
