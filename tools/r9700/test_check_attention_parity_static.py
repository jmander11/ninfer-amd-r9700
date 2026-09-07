#!/usr/bin/env python3

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECK = ROOT / "check_attention_parity_static.py"


class AttentionParityStaticTest(unittest.TestCase):
    def test_current_gfx1201_assembly(self) -> None:
        assembly = ROOT / "build" / "kv_op_qual.s"
        if not assembly.exists():
            self.skipTest("build/kv_op_qual.s has not been generated")
        subprocess.run(["python3", str(CHECK), str(assembly)], check=True)

    def test_rejects_resource_mutation(self) -> None:
        assembly = ROOT / "build" / "kv_op_qual.s"
        if not assembly.exists():
            self.skipTest("build/kv_op_qual.s has not been generated")
        text = assembly.read_text(encoding="utf-8")
        marker = ".vgpr_count:     24"
        symbol = "qk_wmma_batched_w5_kernelILb1EE"
        metadata = text.find(".amdgpu_metadata")
        symbol_at = text.find(symbol, metadata)
        start = text.rfind(".name:", metadata, symbol_at)
        location = text.find(marker, start)
        self.assertGreaterEqual(location, 0)
        with tempfile.TemporaryDirectory() as temporary:
            mutated = Path(temporary) / "mutated.s"
            mutated.write_text(text[:location] + ".vgpr_count:     25" +
                               text[location + len(marker):],
                               encoding="utf-8")
            result = subprocess.run(["python3", str(CHECK), str(mutated)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
