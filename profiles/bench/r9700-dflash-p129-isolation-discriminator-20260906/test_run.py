#!/usr/bin/env python3

import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("p129_run", HERE / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUN)


class DiscriminatorTests(unittest.TestCase):
    def setUp(self):
        self.plan = RUN.validate_plan()
        self.inputs = {
            "control": {"path": "/exact/control"},
            "artifact": {"path": "/exact/artifact", "bytes": 100},
            "source_corpus": {"path": "/exact/source.ids"},
            "history_corpus": {"path": "/exact/history.ids"},
        }

    def test_exact_twelve_single_test_commands(self):
        commands = []
        for phase in ("eager", "graph"):
            for arm in ("ordinary_seed_p128", "dflash_seed_p128",
                        "ordinary_isolated", "ordinary_fresh_p129",
                        "dflash_isolated", "dflash_fresh_p129"):
                command = RUN.command_for(self.plan, self.inputs, phase, arm,
                                          Path(f"/out/{phase}-{arm}.json"))
                commands.append(tuple(command))
                self.assertEqual(command.count("-pg") + command.count("--whole-pg"), 1)
                isolated = arm.endswith("_isolated")
                seed = arm.endswith("_seed_p128")
                self.assertEqual("--isolate-prompt-decode" in command, isolated)
                self.assertEqual("--no-device-graph" in command, phase == "eager")
                self.assertEqual("--spec" in command, arm.startswith("dflash_"))
                corpus = command[command.index("--corpus") + 1]
                self.assertEqual(corpus, "/exact/source.ids" if isolated or seed
                                 else "/exact/history.ids")
        self.assertEqual(len(set(commands)), 12)

    def test_history_fixture_is_exact_source_prefix_plus_seed(self):
        authority = self.plan["history_authority"]
        source = [int(value) for value in Path(authority["source_corpus"]).read_text().split()]
        history = [int(value) for value in (HERE / authority["history_corpus"]).read_text().split()]
        self.assertEqual(history, source[:128] + [authority["seed_token"]])
        self.assertEqual(len(history), 129)
        self.assertEqual(RUN.sha(HERE / authority["history_corpus"]),
                         authority["history_corpus_sha256"])

    def test_classifies_general_prefix_reuse_first(self):
        values = self._values()
        values[("eager", "ordinary_isolated")]["tokens"] = (9, 2)
        result = RUN.classify(values)
        self.assertEqual(result["phase_results"]["eager"]["diagnosis"],
                         "ordinary_append_vs_fresh_execution_path_dependence")

    def test_classifies_dflash_prefix_state(self):
        values = self._values()
        values[("eager", "dflash_isolated")]["tokens"] = (1, 9)
        result = RUN.classify(values)
        self.assertEqual(result["phase_results"]["eager"]["diagnosis"],
                         "dflash_append_vs_fresh_execution_path_dependence")

    def test_classifies_verify_semantics(self):
        values = self._values()
        for phase in ("eager", "graph"):
            values[(phase, "dflash_isolated")]["tokens"] = (3, 4)
            values[(phase, "dflash_fresh_p129")]["tokens"] = (3, 4)
        result = RUN.classify(values)
        self.assertEqual(result["phase_results"]["eager"]["diagnosis"],
                         "dflash_vs_ordinary_execution_path_dependence")

    def test_seed_authority_commands_are_same_mode_fresh_p128(self):
        for phase in ("eager", "graph"):
            for arm in ("ordinary_seed_p128", "dflash_seed_p128"):
                command = RUN.command_for(self.plan, self.inputs, phase, arm, Path("/out.json"))
                self.assertEqual(command[command.index("--whole-pg") + 1], "128,1")
                self.assertNotIn("--isolate-prompt-decode", command)
        self.assertEqual(RUN.expected_corpus_token_count(False, True), 65536)
        self.assertEqual(RUN.expected_corpus_token_count(True, False), 65536)
        self.assertEqual(RUN.expected_corpus_token_count(False, False), 129)

    def test_profiler_environment_policy_covers_loader_and_broad_families(self):
        for key in ("LD_PRELOAD", "LD_AUDIT", "ROCPROFILER_TOOL_LIBRARIES",
                    "ROCP_TOOL_LIBRARIES", "ROCTRACER_DOMAIN", "HSA_TOOLS_LIB",
                    "HIP_TRACE_API", "AQLPROFILE_READ_API", "ATT_PROFILE_PATH"):
            self.assertTrue(RUN.instrumentation_key(key), key)
        self.assertFalse(RUN.instrumentation_key("ROCM_PATH"))

    def test_detects_graph_specific_difference(self):
        values = self._values()
        values[("graph", "dflash_fresh_p129")]["tokens"] = (1, 9)
        result = RUN.classify(values)
        self.assertTrue(result["graph_specific_difference"])

    def test_report_recomputes_exact_spec_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.json"
            command = RUN.command_for(self.plan, self.inputs, "eager", "dflash_isolated", path)
            payload = self._report(command, dflash=True)
            path.write_text(json.dumps(payload))
            result = RUN.validate_report(self.plan, self.inputs, "eager", "dflash_isolated",
                                         path, command)
            self.assertEqual(result["speculative"][:4], (16, 64, 48, 0))

    def test_report_rejects_nonfinite_and_mutated_engine_work(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.json"
            command = RUN.command_for(self.plan, self.inputs, "eager", "dflash_isolated", path)
            payload = self._report(command, dflash=True)
            changed = copy.deepcopy(payload)
            changed["tests"][0]["reps"][0]["timings"]["decode_seconds"] = float("nan")
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, "finite"):
                RUN.validate_report(self.plan, self.inputs, "eager", "dflash_isolated",
                                    path, command)
            changed = copy.deepcopy(payload)
            changed["tests"][0]["reps"][0]["decode_engine_tokens"] = 65
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, "decode_engine_tokens"):
                RUN.validate_report(self.plan, self.inputs, "eager", "dflash_isolated",
                                    path, command)

    def test_report_rejects_mutated_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.json"
            command = RUN.command_for(self.plan, self.inputs, "eager", "dflash_isolated", path)
            payload = self._report(command, dflash=True)
            payload["tests"][0]["speculative"]["accepted_per_position"] = [15, 16, 16, 1]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "aggregate"):
                RUN.validate_report(self.plan, self.inputs, "eager", "dflash_isolated",
                                    path, command)

    @staticmethod
    def _values():
        return {(phase, arm): {"tokens": (1, 2)}
                for phase in ("eager", "graph")
                for arm in ("ordinary_isolated", "ordinary_fresh_p129",
                            "dflash_isolated", "dflash_fresh_p129")}

    def _report(self, command, *, dflash):
        receipt = json.loads(Path(self.plan["build"]["receipt"]).read_text())
        config = dict(receipt["expected_benchmark_profile"])
        config.update({
            "max_context": 203 if dflash else 193, "concurrency": 1,
            "prefill_chunk": 4096, "spec": "dflash" if dflash else "none",
            "draft_tokens": 4 if dflash else 0, "speculative_execution": dflash,
            "dflash_verify_width_requested": 5 if dflash else 0,
            "dflash_verify_width": 5 if dflash else 0,
            "proposal_head": "optimized" if dflash else "full",
            "use_device_graph": False, "retain_token_ids": True,
            "isolate_prompt_decode": True,
            "decode_path": "dflash_eager" if dflash else "eager",
            "decode_graph_prime": {"primed": False, "output_tokens": 0},
            "repetitions": 1, "warmup": 0,
            "corpus_path": self.inputs["source_corpus"]["path"],
            "corpus_tokens": 65536, "dflash_small_t_candidate": False,
        })
        spec = ({"enabled": True, "draft_window": 4, "rounds": 16,
                 "drafted_tokens": 64, "accepted_tokens": 48, "fallback_steps": 0,
                 "acceptance_rate": 0.75, "acceptance_length": 4.0,
                 "accepted_per_position": [16, 16, 16, 0]} if dflash else
                {"enabled": False, "draft_window": 0, "rounds": 0,
                 "drafted_tokens": 0, "accepted_tokens": 0, "fallback_steps": 0,
                 "acceptance_rate": None, "acceptance_length": None,
                 "accepted_per_position": []})
        rep = {"generated_output_tokens": 65, "decode_output_tokens": 64,
               "decode_engine_tokens": 64,
               "generated_token_ids_by_lane": [list(range(65))],
               "timings": {"prepare_seconds": 0.1, "prefill_seconds": 0.0,
                           "decode_seconds": 1.0, "total_seconds": 1.1},
               "speculative": copy.deepcopy(spec)}
        return {"schema_version": 20, "artifact_type": "ninfer_bench_report",
                "tool": "ninfer_bench", "command": shlex_join(command),
                "environment": {"gpu_name": "AMD Radeon AI PRO R9700",
                                "architecture_name": "gfx1201", "device_id": 0},
                "artifact": {"path": self.inputs["artifact"]["path"],
                             "file_size_bytes": self.inputs["artifact"].get("bytes", 100)},
                "load": {"target": "qwen3_8_27b_r9700",
                         "weights_id": self.plan["artifact"]["weights_id"]},
                "config": config,
                "tests": [{"label": "pp128+tg64", "kind": "pp+tg",
                           "n_prompt": 128, "n_gen": 64, "requested_output_tokens": 65,
                           "speculative": copy.deepcopy(spec), "reps": [rep]}]}


def shlex_join(command):
    import shlex
    return shlex.join(command)


if __name__ == "__main__":
    unittest.main()
