#!/usr/bin/env python3
import copy
import tempfile
import unittest
from pathlib import Path

import analyze
import run


class AnalyzerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid=analyze.load(analyze.PACKAGE/"preliminary/result.json")

    def test_preliminary_exact_result_passes(self):
        analyze.validate_result(self.valid)

    def test_gate_up_mismatch_mutation_rejected(self):
        value=copy.deepcopy(self.valid); value["cells"][0]["bf16_output_prefix_mismatch_count"]=290
        with self.assertRaisesRegex(RuntimeError,"result contract"):
            analyze.validate_result(value)

    def test_query_key_exact_mutation_rejected(self):
        value=copy.deepcopy(self.valid); value["cells"][1]["bf16_output_prefix_exact"]=False
        with self.assertRaisesRegex(RuntimeError,"result contract"):
            analyze.validate_result(value)

    def test_algorithm_fingerprint_mutation_rejected(self):
        value=copy.deepcopy(self.valid); value["cells"][0]["profiles"][0]["algorithm_fingerprint"]="00"*16
        with self.assertRaisesRegex(RuntimeError,"result contract"):
            analyze.validate_result(value)

    def test_artifact_mutation_rejected(self):
        value=copy.deepcopy(self.valid); value["artifact"]["sha256"]="00"*32
        with self.assertRaisesRegex(RuntimeError,"result contract"):
            analyze.validate_result(value)

    def test_create_only_writer_rejects_existing_and_dangling_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            existing=root/"existing"
            run.write_exclusive(existing,b"first")
            with self.assertRaises(FileExistsError):
                run.write_exclusive(existing,b"second")
            dangling=root/"dangling"
            dangling.symlink_to(root/"missing")
            with self.assertRaises(FileExistsError):
                run.write_exclusive(dangling,b"payload")


if __name__ == "__main__": unittest.main()
