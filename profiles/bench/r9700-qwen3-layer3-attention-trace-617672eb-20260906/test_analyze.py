#!/usr/bin/env python3
import json, os, tempfile, unittest
from pathlib import Path
import analyze, run

class RepairedContract(unittest.TestCase):
    def test_exact_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp)/"report.json"
            ordinary=analyze.command("target-ordinary",report)
            dflash=analyze.command("target-dflash",report)
            self.assertIn("--no-device-graph",ordinary)
            self.assertEqual(ordinary[-2:],["--draft-tokens","0"])
            self.assertEqual(dflash[-7:],["--spec","dflash","--draft-tokens","4","--dflash-verify-width","5","--lm-head-draft"])
            self.assertEqual(ordinary[ordinary.index("--concurrency")+1],"1")
            self.assertEqual(ordinary[ordinary.index("--whole-pg")+1],"129,1")
    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"out"
            run.write_new(path,"one")
            with self.assertRaises(FileExistsError): run.write_new(path,"two")
            link=Path(tmp)/"link"; link.symlink_to(Path(tmp)/"missing")
            with self.assertRaises(FileExistsError): run.write_new(link,"bad")
    def test_plan_scope(self):
        self.assertEqual(analyze.PLAN["expected_generated_tokens"],[96558,96917])
        self.assertEqual(analyze.PLAN["trace"]["combined_device_staging_bytes"],1674508)
        self.assertEqual(set(analyze.ROLES),{"target-ordinary","target-dflash"})
    def test_injection_detection(self):
        old=os.environ.get("LD_AUDIT")
        try:
            os.environ["LD_AUDIT"]="x"
            self.assertIn("LD_AUDIT",run.injected())
        finally:
            if old is None: os.environ.pop("LD_AUDIT",None)
            else: os.environ["LD_AUDIT"]=old
        old=os.environ.get("NINFER_QWEN3_GDN_STATE_TRACE_TEXT_LAYER")
        try:
            os.environ["NINFER_QWEN3_GDN_STATE_TRACE_TEXT_LAYER"]="1"
            self.assertIn("NINFER_QWEN3_GDN_STATE_TRACE_TEXT_LAYER",run.injected())
        finally:
            if old is None: os.environ.pop("NINFER_QWEN3_GDN_STATE_TRACE_TEXT_LAYER",None)
            else: os.environ["NINFER_QWEN3_GDN_STATE_TRACE_TEXT_LAYER"]=old
    def test_speculative_contract_and_mutations(self):
        valid={"enabled":True,"draft_window":4,"rounds":1,"drafted_tokens":4,"accepted_tokens":2,"fallback_steps":0,"acceptance_rate":0.5,"acceptance_length":3.0,"accepted_per_position":[1,1,0,0]}
        analyze.validate_speculative(valid,True)
        for key,value in (("draft_window",5),("accepted_tokens",5),("acceptance_rate",0.6),("acceptance_length",2.0),("accepted_per_position",[2,2,0,0])):
            bad=dict(valid); bad[key]=value
            with self.assertRaises(RuntimeError): analyze.validate_speculative(bad,True)
        zero={"enabled":True,"draft_window":4,"rounds":0,"drafted_tokens":0,"accepted_tokens":0,"fallback_steps":1,"acceptance_rate":None,"acceptance_length":None,"accepted_per_position":[0,0,0,0]}
        analyze.validate_speculative(zero,True)
    def test_full_ordinary_report_and_material_mutation(self):
        source=Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700/profiles/bench/r9700-qwen3-layer3-attention-trace-72cebe82-20260906/results-failed-attempt2/target-ordinary.json")
        with tempfile.TemporaryDirectory() as tmp:
            prior=analyze.RESULTS; analyze.RESULTS=Path(tmp)
            try:
                report=json.loads(source.read_text())
                path=analyze.RESULTS/"target-ordinary.json"
                report["command"]=" ".join(analyze.command("target-ordinary",path))
                path.write_text(json.dumps(report))
                analyze.validate_report("target-ordinary")
                report["config"]["q4_prefill_cta_profile"]="mutated"
                path.write_text(json.dumps(report))
                with self.assertRaises(RuntimeError): analyze.validate_report("target-ordinary")
            finally: analyze.RESULTS=prior
    def test_process_header_mutations(self):
        stem="target-ordinary"
        valid={"command":analyze.command(stem,analyze.RESULTS/f"{stem}.json"),"trace_environment":analyze.trace_environment(stem),"exit_code":0,"started_unix_ns":1,"finished_unix_ns":2,"power_before":"auto","power_after":"auto","injection_environment":[],"stdout":{},"stderr":{}}
        analyze.validate_process_header(stem,valid)
        mutations=(("extra",1),("trace_environment",{}),("started_unix_ns",True),("finished_unix_ns",1))
        for key,value in mutations:
            bad=dict(valid); bad[key]=value
            with self.assertRaises(RuntimeError): analyze.validate_process_header(stem,bad)
    def test_exact_layer_boundary_tuple_and_wrong_width(self):
        self.assertEqual(analyze.layer_expected("target-ordinary"),("target-ordinary-frontier130",1,0,130,129))
        self.assertEqual(analyze.layer_expected("target-dflash"),("target-dflash-frontier130-column0",5,0,130,129))
        ordinary=analyze.RESULTS/"target-ordinary.trace.json"
        analyze.LAYER.load(ordinary,analyze.layer_expected("target-ordinary"))
        with self.assertRaisesRegex(RuntimeError,"width"):
            analyze.LAYER.load(ordinary,("target-ordinary-frontier130",5,0,130,129))

if __name__=="__main__": unittest.main()
