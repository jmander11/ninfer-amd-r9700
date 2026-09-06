import unittest
from unittest import mock
from tools.r9700 import run_a8q4_n16k16_scalar_base_repeat1 as repeat

class RepeatBindingTest(unittest.TestCase):
 def test_real_first_report_authorizes_exactly_one_repeat(self):
  self.assertEqual(repeat.validate_first()["decision"]["classification"],"inconclusive")
 def test_wrong_first_hash_rejected(self):
  with mock.patch.object(repeat,"FIRST_SHA","0"*64),self.assertRaises(RuntimeError):repeat.validate_first()
 def test_stable_first_report_rejected(self):
  with mock.patch.object(repeat.base,"validate",return_value=None),mock.patch.object(repeat.json,"loads",return_value={"decision":{"classification":"terminal_reject","stability_pass":True}}),self.assertRaises(RuntimeError):repeat.validate_first()

if __name__=="__main__":unittest.main()
