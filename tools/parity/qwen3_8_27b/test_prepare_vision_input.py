import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock

import torch

from tools.parity.qwen3_8_27b.prepare_vision_input import prepare
from tools.parity.qwen3_8_27b.vision import load_prepared_batch
from tools.reference.qwen3.common.multimodal import MultimodalBatch


class PrepareVisionInputTest(unittest.TestCase):
    @staticmethod
    def batch() -> MultimodalBatch:
        return MultimodalBatch(
            input_ids=torch.tensor([1, 2]),
            mm_token_type_ids=torch.tensor([1, 0]),
            position_ids=torch.tensor([[0, 1], [0, 1], [0, 1]]),
            rope_delta=0,
            pixel_values=torch.zeros((4, 1536)),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            pixel_values_videos=None,
            video_grid_thw=None,
        )

    def test_freezes_and_reopens_exact_single_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.ninfer"
            messages = root / "messages.json"
            output = root / "input.safetensors"
            weights.write_bytes(b"weights")
            messages.write_text('[{"role":"user","content":"x"}]')
            binding = mock.MagicMock()
            binding.__enter__.return_value = binding
            with mock.patch(
                "tools.parity.qwen3_8_27b.prepare_vision_input.VisionArtifactBinding.open",
                return_value=binding,
            ), mock.patch(
                "tools.parity.qwen3_8_27b.prepare_vision_input.Frontend"
            ) as frontend:
                frontend.return_value.process.return_value = self.batch()
                contract = prepare(weights, messages, output)
            reopened, reopened_contract = load_prepared_batch(output)
            self.assertEqual(contract, reopened_contract)
            self.assertEqual(reopened.image_grid_thw.tolist(), [[1, 2, 2]])
            self.assertEqual(reopened.pixel_values.shape, (4, 1536))

    def test_rejects_dangling_output_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.ninfer"
            messages = root / "messages.json"
            weights.write_bytes(b"weights")
            messages.write_text('[{"role":"user","content":"x"}]')
            output = root / "input.safetensors"
            output.symlink_to(root / "missing")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                prepare(weights, messages, output)
            self.assertTrue(output.is_symlink())

    def test_pending_replacement_is_not_published_or_removed_as_owned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.ninfer"
            messages = root / "messages.json"
            weights.write_bytes(b"weights")
            messages.write_text('[{"role":"user","content":"x"}]')
            output = root / "input.safetensors"
            binding = mock.MagicMock()
            binding.__enter__.return_value = binding
            real_link = os.link

            def replace_then_link(source, destination):
                pending = Path(source)
                foreign = pending.with_name(f"{pending.name}.foreign")
                foreign.write_bytes(b"foreign")
                os.replace(foreign, pending)
                real_link(pending, destination)

            with mock.patch(
                "tools.parity.qwen3_8_27b.prepare_vision_input.VisionArtifactBinding.open",
                return_value=binding,
            ), mock.patch(
                "tools.parity.qwen3_8_27b.prepare_vision_input.Frontend"
            ) as frontend, mock.patch(
                "tools.parity.qwen3_8_27b.prepare_vision_input.os.link",
                side_effect=replace_then_link,
            ):
                frontend.return_value.process.return_value = self.batch()
                with self.assertRaisesRegex(ValueError, "pending inode"):
                    prepare(weights, messages, output)
            self.assertEqual(output.read_bytes(), b"foreign")


if __name__ == "__main__":
    unittest.main()
