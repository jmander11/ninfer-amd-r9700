#!/usr/bin/env python3
from __future__ import annotations
import copy, json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent)); import analyze  # noqa:E402

def ranked(): return {"top1_token":7,"top1_bf16_bits":0x4000,"top1_logit":2.0,"top2_token":9,"top2_bf16_bits":0x3f80,"top2_logit":1.0,"margin":1.0}

def dflash_fixture():
    outputs=[100]; events=[]; frontier=129; context_frontier=129
    rounds=drafted=accepted=0; accepted_positions=[0,0,0,0]
    while len(outputs)<28:
        extent=min(4,27-(len(outputs)-1)-1); count=min(2,extent)
        columns=[]
        for col in range(extent+1):
            verify=outputs[-1] if col==0 else 1000+len(outputs)*10+col
            columns.append({"column":col,"logits_stage":"pre_accept_target_verify","verify_token":verify,"input_cache_position":frontier+col,"absolute_frontier":frontier+col+1,"target_argmax":7,"target_argmax_matches_top1":True,"decision":ranked()})
        licensed=[columns[i+1]["verify_token"] for i in range(count)]+[7]
        events.append({"kind":"dflash","lane":0,"token_domain":248077,"verify_width":5,"tree_verify":False,"anchor":outputs[-1],"base_frontier":frontier,"context_frontier":context_frontier,"proposal_extent":extent,"accepted_drafts":count,"accepted_column":count,"licensed_tokens":licensed,"next_frontier":frontier+len(licensed),"columns":columns})
        rounds+=1; drafted+=extent; accepted+=count
        for i in range(count): accepted_positions[i]+=1
        outputs.extend(licensed); context_frontier=frontier; frontier+=len(licensed)
    spec={"enabled":True,"draft_window":4,"rounds":rounds,"drafted_tokens":drafted,"accepted_tokens":accepted,"fallback_steps":0,"acceptance_rate":accepted/drafted,"acceptance_length":1+accepted/rounds,"accepted_per_position":accepted_positions}
    return outputs,{"artifact_type":"ninfer_dflash_target_decision_trace","schema_version":1,"diagnostic_only":True,"timing_eligible":False,"production_routing_authorized":False,"events":events},spec

class AnalyzeTest(unittest.TestCase):
    def test_exact_plan_contract(self):
        analyze.validate_plan()

    def test_ordinary_trace_covers_index27(self):
        outputs=list(range(28)); events=[]
        for i in range(1,28): events.append({"kind":"ordinary","logits_stage":"pre_sample","lane":0,"token_domain":248077,"input_cache_position":128+i,"absolute_frontier":129+i,"sampled_token":outputs[i],"sampled_matches_top1":True,"next_frontier":129+i,"decision":ranked()})
        value={"artifact_type":"ninfer_dflash_target_decision_trace","schema_version":1,"diagnostic_only":True,"timing_eligible":False,"production_routing_authorized":False,"events":events}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"trace.json"; p.write_text(json.dumps(value)); self.assertEqual(analyze.validate_decision_trace(p,"ordinary",outputs,{})["path"],str(p))
            bad=copy.deepcopy(value); bad["events"][0]["sampled_token"]=True; p.write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError,"sampled token"): analyze.validate_decision_trace(p,"ordinary",outputs,{})

    def test_dflash_licensed_chain_covers_index27(self):
        outputs,value,spec=dflash_fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"trace.json"; p.write_text(json.dumps(value)); analyze.validate_decision_trace(p,"dflash",outputs,spec)
            bad=copy.deepcopy(value); bad["events"][0]["columns"][0]["target_argmax"]=8; p.write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError,"argmax"): analyze.validate_decision_trace(p,"dflash",outputs,spec)

    def test_dflash_rejects_semantic_mutations(self):
        outputs,value,spec=dflash_fixture()
        mutations=(("position",lambda v:v["events"][0]["columns"][0].__setitem__("input_cache_position",999)),
                   ("anchor",lambda v:v["events"][0].__setitem__("anchor",999)),
                   ("accepted draft",lambda v:v["events"][0]["licensed_tokens"].__setitem__(0,999)),
                   ("terminal",lambda v:v["events"][0]["licensed_tokens"].__setitem__(2,999)),
                   ("extent",lambda v:v["events"][0].__setitem__("proposal_extent",3)),
                   ("contract",lambda v:v["events"][0].__setitem__("accepted_drafts",True)),
                   ("bool token",lambda v:v["events"][0]["columns"][3].__setitem__("verify_token",True)),
                   ("domain token",lambda v:v["events"][0]["columns"][3].__setitem__("verify_token",248077)))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"trace.json"
            for label,mutate in mutations:
                bad=copy.deepcopy(value); mutate(bad); p.write_text(json.dumps(bad))
                with self.subTest(label=label),self.assertRaises(RuntimeError): analyze.validate_decision_trace(p,"dflash",outputs,spec)
            p.write_text(json.dumps(value)); bad_spec=copy.deepcopy(spec); bad_spec["accepted_tokens"]+=1
            with self.assertRaisesRegex(RuntimeError,"acceptance"): analyze.validate_decision_trace(p,"dflash",outputs,bad_spec)

    def test_rejects_duplicate_decision_key(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.json"; p.write_text('{"a":1,"a":2}')
            with self.assertRaisesRegex(RuntimeError,"duplicate"): analyze.load(p)

    def test_exact_commands(self):
        p=analyze.RESULTS/"x.json"
        ordinary=analyze.command("decision-ordinary",p); dflash=analyze.command("decision-dflash",p)
        self.assertIn("129,27",ordinary); self.assertNotIn("--spec",ordinary)
        self.assertEqual(dflash[dflash.index("--draft-tokens")+1],"4")
        append=analyze.command("tail-append",p); self.assertIn("--isolate-prompt-decode",append); self.assertIn("128,1",append)

    def test_exact_normal_benchmark_stderr(self):
        for stem in ("decision-ordinary","decision-dflash","tail-fresh","tail-append"):
            analyze.validate_benchmark_stderr(stem,analyze.RESULTS/f"{stem}.stderr")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"stderr"; path.write_text((analyze.RESULTS/"tail-fresh.stderr").read_text()+"extra\n")
            with self.assertRaisesRegex(RuntimeError,"stderr differs"):
                analyze.validate_benchmark_stderr("tail-fresh",path)

if __name__=="__main__": unittest.main()
