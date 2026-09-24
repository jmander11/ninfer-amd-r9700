"""Focused CPU tests for the evaluation-only rowwise-E4M3 converter."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from safetensors.torch import save_file
import torch

from tools.artifact.container import Artifact
from tools.convert.qwen3.common import conversion as family_conversion
from tools.convert.qwen3.common.inventory import ResourceSpec, TensorSpec

from . import codec, convert_e4m3, e4m3_inventory, source_recipe


class E4M3ConverterTest(unittest.TestCase):
    def _synthetic_preflight(self, root: Path):
        tensor_name = "fixture/weight"
        shard_name = "model-00001-of-00001.safetensors"
        tensor = torch.tensor(
            [[448.0, 1.0625, -0.0], [-448.0, 1.1875, 0.0]],
            dtype=torch.bfloat16,
        )
        save_file({tensor_name: tensor}, root / shard_name)
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {tensor_name: shard_name}}),
            encoding="utf-8",
        )
        ranking = root / "ranking.i64"
        ranking.write_bytes(b"synthetic-ranking")
        resource = family_conversion.ResourcePayload(
            "frontend/tokenizer.json", b'{"synthetic":true}'
        )
        object_specs = (
            ResourceSpec(resource.name),
            TensorSpec(
                tensor_name,
                (2, 3),
                e4m3_inventory.F8E4M3_ROW_F32S,
                e4m3_inventory.ROW_SCALED_LAYOUT,
            ),
        )
        plan = family_conversion.build_object_plan(
            object_specs, {resource.name: resource.data}
        )
        ranking_provenance = SimpleNamespace(
            ranking_path=ranking,
            ranking_sha256="1" * 64,
            sidecar_path=root / "ranking.i64.provenance.json",
            sidecar_sha256="2" * 64,
            total_tokens=2,
            distinct_token_ids=2,
        )
        preflight = convert_e4m3.E4M3ConversionPreflight(
            model_dir=root,
            config_summary={"synthetic": True},
            source=source_recipe.SourcePreflight(1, 1, 1, {"BF16": 1}),
            resources=(resource,),
            draft=SimpleNamespace(ranking=ranking, n=1),
            draft_ranking=ranking_provenance,
            object_plan=plan,
        )
        return tensor, object_specs, preflight

    def test_synthetic_shard_and_resource_write_exact_evaluation_artifact(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor, object_specs, preflight = self._synthetic_preflight(root)
            output = root / "candidate.ninfer"

            with (
                mock.patch.object(
                    convert_e4m3, "preflight_conversion", return_value=preflight
                ),
                mock.patch.object(
                    e4m3_inventory, "OBJECT_SPECS", object_specs
                ),
                mock.patch.object(
                    convert_e4m3.source,
                    "materialize_tensor",
                    side_effect=lambda spec, reader, draft: reader.get(spec.name),
                ),
                redirect_stdout(io.StringIO()),
            ):
                report_path = convert_e4m3.convert(
                    root, output, draft_ranking=preflight.draft.ranking
                )

            represented = tensor.float().reshape(-1).tolist()
            expected = codec.encode_e4m3_rowwise_reference(represented, 2, 3)
            with Artifact.open(output) as artifact:
                self.assertEqual(
                    artifact.identity.model_id, e4m3_inventory.MODEL_ID
                )
                self.assertEqual(
                    artifact.identity.weights_id, e4m3_inventory.WEIGHTS_ID
                )
                self.assertEqual(len(artifact.objects), 2)
                self.assertEqual(
                    bytes(artifact.payload("frontend/tokenizer.json")),
                    b'{"synthetic":true}',
                )
                tensor_object = artifact.find("fixture/weight")
                self.assertEqual(tensor_object.shape, (2, 3))
                self.assertEqual(tensor_object.format, "F8E4M3_ROW_F32S")
                self.assertEqual(tensor_object.layout, "row-scaled-k128-v1")
                self.assertEqual(bytes(artifact.payload(tensor_object)), expected)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["identity"],
                {
                    "model_id": e4m3_inventory.MODEL_ID,
                    "weights_id": e4m3_inventory.WEIGHTS_ID,
                },
            )
            self.assertEqual(report["candidate"]["status"], "registered-evaluation-only")
            self.assertFalse(report["candidate"]["weight_recipe_selected"])
            self.assertEqual(report["arguments"]["device"], "cpu")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                convert_e4m3.convert(
                    root, output, draft_ranking=preflight.draft.ranking
                )

    def test_preflight_only_writes_no_artifact_and_reports_evaluation_identity(self) -> None:
        summary = {
            "identity": {
                "model_id": e4m3_inventory.MODEL_ID,
                "weights_id": e4m3_inventory.WEIGHTS_ID,
            }
        }
        stdout = io.StringIO()
        with (
            mock.patch.object(convert_e4m3, "preflight_conversion", return_value=object()),
            mock.patch.object(convert_e4m3, "preflight_summary", return_value=summary),
            redirect_stdout(stdout),
        ):
            convert_e4m3.main(
                [
                    "--model",
                    "synthetic-model",
                    "--draft-ranking",
                    "synthetic-ranking.i64",
                    "--preflight-only",
                ]
            )
        self.assertEqual(json.loads(stdout.getvalue()), summary)

    def test_preflight_only_rejects_output_and_conversion_requires_output(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                convert_e4m3.main(
                    [
                        "--model",
                        "synthetic-model",
                        "--draft-ranking",
                        "synthetic-ranking.i64",
                        "--out",
                        "forbidden.ninfer",
                        "--preflight-only",
                    ]
                )
            with self.assertRaises(SystemExit):
                convert_e4m3.main(
                    [
                        "--model",
                        "synthetic-model",
                        "--draft-ranking",
                        "synthetic-ranking.i64",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
