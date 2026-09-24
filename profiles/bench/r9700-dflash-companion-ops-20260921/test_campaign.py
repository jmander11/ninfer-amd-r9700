import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("dflash_companion_campaign", Path(__file__).with_name("campaign.py"))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def linear_output():
    lines = []
    for n, k in campaign.SHAPES:
        for t in (*campaign.WIDTHS, *((2048,) if (n, k) in campaign.SHAPES[:2] else ())):
            profile = "A8W8" if (n, k) == (5120, 17408) else "BF16xW8"
            lines.append(f"dflash-companion-linear: PASS N={n} K={k} T={t} public_profile={profile} "
                f"fp64_samples={9*(t if t<=24 else 7)} full_K={k} graph_replays=2 workspace_bytes=1024 "
                "max_abs=0.0001 max_normalized=0.1 worst_token=0 worst_row=0")
    lines.append("dflash-companion-linear: ALL PASS represented-BF16-input FP64 sampled oracle")
    return "\n".join(lines)


class CampaignTest(unittest.TestCase):
    def test_exact_public_op_callsite_frontier(self):
        commands = list(campaign.commands())
        self.assertEqual(len(commands), 17)
        for k in (4, 5):
            for c in (1, 2, 3, 4):
                by_name = dict(commands)
                self.assertEqual(by_name[f"conv-w{k+1}-c{c}"][-2:], [str(k+1), str(c)])
                self.assertEqual(by_name[f"selector-k{k}-c{c}"][-2:], [str(k), str(c)])

    def test_linear_requires_complete_fp64_and_graph_evidence(self):
        output = linear_output()
        self.assertEqual(campaign.validate_output("linear", output), 42)
        for changed in ("\n".join(output.splitlines()[1:]),
                        output.replace("graph_replays=2", "graph_replays=1", 1),
                        output.replace("max_normalized=0.1", "max_normalized=nan", 1),
                        output.replace("max_normalized=0.1", "max_normalized=1.01", 1),
                        output.replace("public_profile=A8W8", "public_profile=BF16xW8", 1)):
            with self.subTest(changed=changed[:80]), self.assertRaises(ValueError):
                campaign.validate_output("linear", changed)

    def test_selector_preserves_real_optimized_head_and_full_bf16_codebooks(self):
        lines = [f"dflash-companion-selector: PASS K=5 C=4 mapped={mapped} graph_replays=2 "
                 "full_FP64_selector_oracle logits_rows=131072 bf16_codebook_rows=248320 workspace_bytes=256"
                 for mapped in (0, 1)]
        text = "\n".join(lines)
        self.assertEqual(campaign.validate_output("selector-k5-c4", text), 2)
        for bad in (text.replace("248320", "131072"), text.replace("mapped=1", "mapped=0"),
                    text.replace("K=5", "K=4")):
            with self.assertRaises(ValueError):
                campaign.validate_output("selector-k5-c4", bad)

    def test_conv_requires_finite_complete_oracle(self):
        text = ("dflash-companion-conv: PASS W=6 C=4 graph_replays=2 "
                "full_FP64_projection_and_conv_oracle max_abs=.001 relative_l2=.001 workspace_bytes=256")
        self.assertEqual(campaign.validate_output("conv-w6-c4", text), 1)
        for bad in (text.replace("max_abs=.001", "max_abs=nan"), text.replace("relative_l2=.001", "relative_l2=.03")):
            with self.assertRaises(ValueError):
                campaign.validate_output("conv-w6-c4", bad)

    def test_existing_failure_is_never_overwritten_or_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retained = root / "failed.stdout.txt"
            retained.write_text("original failure")
            with patch.object(campaign, "OUTPUT", root), patch.object(campaign, "preflight") as preflight, \
                 patch.object(campaign, "idle") as idle, self.assertRaises(ValueError):
                campaign.run()
            preflight.assert_not_called()
            idle.assert_not_called()
            self.assertEqual(retained.read_text(), "original failure")

    def test_preflight_is_read_only_and_rejects_stale_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            (build / "src").mkdir(parents=True)
            for target in campaign.TARGETS.values():
                path = build / "src" / target
                path.write_text("fixture executable")
                path.chmod(0o755)
            (build / "CMakeCache.txt").write_text("\n".join(f"{k}:STRING={v}" for k, v in {
                "CMAKE_BUILD_TYPE":"Release", "CMAKE_HIP_ARCHITECTURES":"gfx1201",
                "NINFER_R9700_Q4_ACTIVATION_BITS":"8", "NINFER_R9700_W8_ACTIVATION_BITS":"8",
                "NINFER_R9700_KV_VALUE_GROUP":"16", "NINFER_R9700_XATTENTION_QUALIFICATION":"OFF"}.items()))
            (root / "campaign.py").write_text("fixture")
            (root / "commands.sh").write_text("fixture")
            original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with patch.object(campaign, "BUILD", build), patch.object(campaign, "PACKAGE", root), \
                 patch.object(campaign, "SOURCES", ()), patch.object(campaign.subprocess, "run",
                     return_value=subprocess.CompletedProcess([], 0, "ninja: no work to do.\n", "")) as process:
                value = campaign.preflight()
                self.assertEqual(json.loads(json.dumps(value)), value)
                self.assertEqual(process.call_args.args[0][-1], "-n")
                process.return_value.stdout = "[1/2] Build stale object\n"
                with self.assertRaisesRegex(ValueError, "stale"):
                    campaign.preflight()
            self.assertEqual(original, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
