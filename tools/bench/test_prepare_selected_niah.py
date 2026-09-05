import errno
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from tools.bench import prepare_selected_niah as preparer
from tools.bench.prepare_selected_niah import prepare, rename_noreplace


REPO = Path(__file__).resolve().parents[2]


def test_prepare_directory_rename_is_exclusive() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        staged = root / "staged"
        staged.mkdir()
        output = root / "output"
        output.symlink_to(root / "missing")
        with pytest.raises(OSError) as raised:
            rename_noreplace(staged, output)
        assert raised.value.errno == errno.EEXIST
        assert staged.is_dir()
        assert output.is_symlink()


def test_prepare_directory_rename_publishes_complete_tree() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        staged = root / "staged"
        staged.mkdir()
        (staged / "plan.json").write_text("{}\n", encoding="utf-8")
        output = root / "output"
        rename_noreplace(staged, output)
        assert not staged.exists()
        assert (output / "plan.json").read_text(encoding="utf-8") == "{}\n"


def test_prepare_rejects_dangling_output_before_route_resolution() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selection = root / "selection.json"
        selection.write_text("{}\n", encoding="utf-8")
        output = root / "campaign"
        output.symlink_to(root / "missing")
        with pytest.raises(ValueError, match="already exists"):
            prepare(selection, output)
        assert output.is_symlink()


def test_generated_commands_use_importable_repo_modules() -> None:
    with tempfile.TemporaryDirectory(dir=REPO / "profiles/bench") as directory:
        root = Path(directory)
        selection = root / "selection.json"; selection.write_text("{}\n")
        artifact = root / "artifact.ninfer"; artifact.write_bytes(b"artifact")
        build = root / "build"
        bench = build / "bench/ninfer_bench"; bench.parent.mkdir(parents=True)
        bench.write_bytes(b"bench")
        serve = build / "apps/ninfer-serve"; serve.parent.mkdir(parents=True)
        serve.write_bytes(b"serve"); serve.chmod(0o755)
        matrix = root / "matrix.json"; matrix.write_text("{}\n")
        fixture = root / "fixture.json"; fixture.write_text("{}\n")
        route = {
            "artifact": {"path": str(artifact)}, "benchmark": {"path": str(bench)},
            "build_directory": str(build), "selected_prefill_chunk": 2048,
            "source_matrices": {"whole": {"path": str(matrix)}},
            "hybrid_width_tool": None,
        }
        output = root / "campaign"
        with mock.patch.object(preparer, "resolve_route", return_value=route), \
             mock.patch.object(preparer, "matrix_cases", return_value=[("64k-start", "ref")]), \
             mock.patch.object(preparer, "resolve_fixture", return_value=fixture):
            prepare(selection, output)
        commands = (output / "commands.sh").read_text(encoding="utf-8")
        assert "python3 -m tools.bench.run_niah_check" in commands
        assert "python3 -m tools.bench.validate_selected_niah" in commands
        assert "python3 /" not in commands


def test_all_niah_module_entry_points_import_from_repo() -> None:
    for module in (
        "tools.bench.prepare_selected_niah",
        "tools.bench.run_niah_check",
        "tools.bench.validate_selected_niah",
    ):
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"], cwd=REPO,
            text=True, capture_output=True, check=False,
        )
        assert result.returncode == 0, (module, result.stderr)
