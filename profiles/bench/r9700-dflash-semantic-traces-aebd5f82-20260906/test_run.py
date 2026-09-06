#!/usr/bin/env python3
from __future__ import annotations
import os,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parent)); import run  # noqa:E402

class RunTest(unittest.TestCase):
    def test_broad_instrumentation_detection(self):
        with mock.patch.dict(os.environ,{"ROCPROF_FUTURE_FLAG":"1"},clear=True): self.assertEqual(run.injected(),["ROCPROF_FUTURE_FLAG"])
        with mock.patch.dict(os.environ,{"LD_PRELOAD":"x"},clear=True): self.assertEqual(run.injected(),["LD_PRELOAD"])
    def test_trace_variables_are_rejected_from_parent(self):
        with mock.patch.dict(os.environ,{"NINFER_DFLASH_DECISION_TRACE_OUT":"x"},clear=True): self.assertEqual(run.injected(),["NINFER_DFLASH_DECISION_TRACE_OUT"])
    def test_clean_environment(self):
        with mock.patch.dict(os.environ,{"PATH":"/bin"},clear=True): self.assertEqual(run.injected(),[])
    def test_dangling_results_symlink_is_not_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"results"; path.symlink_to(Path(directory)/"missing")
            with self.assertRaisesRegex(RuntimeError,"already exists"): run.require_fresh_results(path)

if __name__=="__main__": unittest.main()
