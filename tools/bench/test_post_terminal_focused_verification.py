import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

RESOLVER = (
    Path(__file__).resolve().parents[2]
    / "profiles/bench/post-terminal-focused-verification-20260905/resolve.py"
)
SPEC = importlib.util.spec_from_file_location("post_terminal_focused_resolve", RESOLVER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PostTerminalFocusedVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

    def test_hybrid_identity_is_the_registered_n16_artifact(self) -> None:
        self.assertTrue(MODULE.is_hybrid_weights("r9700-q4g64-f8e4m3-four-role-n16k16-eval"))
        self.assertFalse(MODULE.is_hybrid_weights("r9700-q4g64-f8e4m3-four-role-eval"))

    def test_selected_dense_build_ignores_inactive_sparse_parameters(self) -> None:
        cache = self.tmp_path / "CMakeCache.txt"
        settings = {
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_GENERATOR": "Ninja",
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
        with self.assertRaisesRegex(ValueError, "TAU_PERMILLE"):
            MODULE.validate_build_profile(cache, 16, True)
        settings["NINFER_R9700_XATTENTION_TAU_PERMILLE"] = "900"
        write()
        MODULE.validate_build_profile(cache, 16, True)
        settings["NINFER_R9700_XATTENTION_STRIDE"] = "8"
        write()
        with self.assertRaisesRegex(ValueError, "STRIDE"):
            MODULE.validate_build_profile(cache, 16, True)

    @unittest.skipUnless(
        importlib.util.find_spec("pytest"),
        "the retained campaign's TEST_PYTHON contract needs pytest, Torch, and safetensors",
    )
    def test_interpreter_identity_preserves_explicit_launcher(self) -> None:
        launcher = Path(sys.executable)

        identity = MODULE.inspect_test_python(launcher)

        self.assertEqual(identity["launcher_path"], str(launcher))
        self.assertEqual(identity["python_version"], ".".join(map(str, sys.version_info[:3])))
        self.assertEqual(set(identity["packages"]), {"pytest", "torch", "safetensors"})
        self.assertEqual(identity["prefix"], sys.prefix)

    def test_resolver_rejects_dangling_output_namespace(self) -> None:
        root = self.tmp_path
        output = root / "route.json"
        output.symlink_to(root / "missing.json")
        argv = [
            "resolve.py",
            "--selection",
            str(root / "not-opened.json"),
            "--test-python",
            sys.executable,
            "--out",
            str(output),
        ]
        with (
            patch.object(sys, "argv", argv),
            self.assertRaisesRegex(SystemExit, "refusing to overwrite"),
        ):
            MODULE.main()
        self.assertTrue(output.is_symlink())


if __name__ == "__main__":
    unittest.main()
