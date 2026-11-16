"""Historical precision comparisons must not inherit the production cache default."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from tools.ppl import compare_nvfp4 as comparison


def test_historical_amd_commands_pin_fp8_int4():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "manifest.json").write_text(json.dumps({"nvidia_model": "/nvidia-model"}))
        power = root / "power"
        power.write_text("auto")
        commands = []

        def retain(command, cell):
            commands.append(command)
            cell.mkdir()
            (cell / "report.json").write_text(
                json.dumps(
                    {
                        "non_finite": 0,
                        "tokens_scored": 2047 if "decode" not in command else 128,
                        "q4_activation_bits": 8,
                        "w8_activation_bits": 8,
                        "ppl": 2.0,
                    }
                )
            )

        with (
            patch.object(comparison, "CONFIGS", [("fixture", 8)]),
            patch.object(comparison, "ARTIFACTS", {"fixture": Path("/model.ninfer")}),
            patch.object(comparison, "SAMPLES", ["sample"]),
            patch.object(comparison, "run", side_effect=retain),
            patch.object(comparison, "power_profile_path", return_value=power),
        ):
            comparison.quality(root, "amd")
            comparison.quality(root, "amd", "decode")
            comparison.speed(root)
            assert len(commands) == 3
            for command in commands:
                assert command[command.index("--kv-cache") + 1] == "fp8-int4"
            comparison.quality(root, "nvidia")
            assert "--kv-cache" not in commands[-1]
            assert commands[-1][commands[-1].index("--kv-dtype") + 1] == "nvfp4"
