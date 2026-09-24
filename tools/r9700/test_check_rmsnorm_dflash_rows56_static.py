from pathlib import Path
import unittest
from unittest.mock import patch

from tools.r9700.check_rmsnorm_dflash_rows56_static import EXPECTED, exact


class ExactStaticTest(unittest.TestCase):
    def test_accepts_exact_resource_record(self):
        with patch("tools.r9700.check_rmsnorm_dflash_rows56_static.check",
                   return_value={"symbol": "kernel", **EXPECTED}):
            self.assertEqual(exact(Path("unused"))["vgprs"], 17)

    def test_rejects_any_resource_drift(self):
        for key in EXPECTED:
            with self.subTest(key=key):
                changed = dict(EXPECTED); changed[key] += 1
                with patch("tools.r9700.check_rmsnorm_dflash_rows56_static.check",
                           return_value={"symbol": "kernel", **changed}):
                    with self.assertRaisesRegex(ValueError, key): exact(Path("unused"))


if __name__ == "__main__": unittest.main()
