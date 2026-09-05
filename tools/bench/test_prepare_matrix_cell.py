from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from tools.bench.prepare_matrix_cell import prepare_or_validate


PRODUCER = r'''from pathlib import Path
import json, sys
args=sys.argv[1:]
out=Path(args[args.index("--output-dir")+1])
out.mkdir()
(out/"json"/"suite"/"c1").mkdir(parents=True)
(out/"logs").mkdir()
(out/"diagnostics").mkdir()
(out/"commands.sh").write_text(f"run --output-file {out}/json/suite/c1/result.json\n")
(out/"manifest.json").write_text(json.dumps({
    "artifact_type":"ninfer_bench_matrix_run", "schema_version":14,
    "created_at_utc":str(out), "prepare_only":True,
    "commands":[{"report":str(out/"json/suite/c1/result.json")}],
})+"\n")
'''


class PrepareMatrixCellTest(unittest.TestCase):
    def command(self, producer: Path, output: Path) -> list[str]:
        return [sys.executable, str(producer), "--output-dir", str(output)]

    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        producer = root / "producer.py"
        producer.write_text(PRODUCER, encoding="utf-8")
        return temporary, root, producer

    def test_absent_cell_is_created(self) -> None:
        temporary, root, producer = self.fixture()
        with temporary:
            output = root / "cell"
            self.assertEqual(prepare_or_validate(output, self.command(producer, output)), "created")
            self.assertTrue((output / "manifest.json").is_file())

    def test_existing_exact_cell_is_validated_without_replacement(self) -> None:
        temporary, root, producer = self.fixture()
        with temporary:
            output = root / "cell"
            prepare_or_validate(output, self.command(producer, output))
            before = (output / "manifest.json").read_bytes()
            self.assertEqual(
                prepare_or_validate(output, self.command(producer, output)), "validated"
            )
            self.assertEqual((output / "manifest.json").read_bytes(), before)

    def test_changed_manifest_is_rejected(self) -> None:
        temporary, root, producer = self.fixture()
        with temporary:
            output = root / "cell"
            prepare_or_validate(output, self.command(producer, output))
            manifest = json.loads((output / "manifest.json").read_text())
            manifest["prepare_only"] = False
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                prepare_or_validate(output, self.command(producer, output))

    def test_partial_or_executed_cell_is_rejected(self) -> None:
        temporary, root, producer = self.fixture()
        with temporary:
            output = root / "cell"
            prepare_or_validate(output, self.command(producer, output))
            (output / "json/suite/c1/result.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "partial or unexpected"):
                prepare_or_validate(output, self.command(producer, output))

    def test_symlink_cell_is_rejected(self) -> None:
        temporary, root, producer = self.fixture()
        with temporary:
            real = root / "real"
            real.mkdir()
            output = root / "cell"
            output.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real directory"):
                prepare_or_validate(output, self.command(producer, output))


if __name__ == "__main__":
    unittest.main()
