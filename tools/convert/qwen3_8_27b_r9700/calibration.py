"""Layer-major input second moments of the Qwen3.8-27B Text projections for FP8LUT4 GPTQ.

The calibration sequences are evaluated through the checkpoint-direct BF16 reference
(`tools.reference.qwen3_8_27b_bf16.backend`) one source layer at a time, with every sequence's
hidden state resident on the device. While layer L runs, each projection input is accumulated
as H = sum_t x_t x_t^T in FP32; `layer(L)` returns those matrices keyed by the artifact object
suffix they calibrate. Layers must be requested in ascending order; later layers see the BF16
(not the quantized) outputs of earlier ones. After the last layer, `final()` returns the second
moment of the output-head input.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from tools.reference.qwen3_8_27b_bf16 import backend
from tools.reference.qwen3_8_27b_bf16.protocol import LAYERS

# Source projection whose input calibrates each artifact object (all consumers of one input
# share its matrix).
_INPUT_OF = {
    "self_attn.q_proj.weight": ("attention/query_key", "attention/gate_value"),
    "self_attn.o_proj.weight": ("attention/output",),
    "linear_attn.in_proj_qkv.weight": ("gdn/query_key", "gdn/value_z"),
    "linear_attn.out_proj.weight": ("gdn/output",),
    "mlp.gate_proj.weight": ("mlp/gate_up",),
    "mlp.down_proj.weight": ("mlp/down",),
}


def read_sequences(path: Path) -> list[list[int]]:
    sequences = [[int(token) for token in line.split()] for line in path.read_text().splitlines()
                 if line.strip()]
    if not sequences or len({len(sequence) for sequence in sequences}) != 1:
        raise ValueError("calibration ids must be nonempty equal-length sequences, one per line")
    return sequences


class InputMoments:
    def __init__(self, model: Path, ids: Path, device: torch.device):
        weight_map = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        self.checkpoint = backend.SourceCheckpoint(model, weight_map)
        self.scorer = backend.LayerMajorTextScorer(
            self.checkpoint, device_index=device.index or 0, prefill_chunk=2048,
            schedule="prefill", skip_text="", kv_value_group=None)
        self.device = self.scorer.device
        sequences = read_sequences(ids)
        self.tokens = sum(len(sequence) for sequence in sequences)
        self.hidden = self.checkpoint.embedding_rows(
            [token for sequence in sequences for token in sequence], self.device
        ).reshape(len(sequences), len(sequences[0]), -1)
        self.next_layer = 0

    def layer(self, layer: int) -> dict[str, torch.Tensor]:
        if layer != self.next_layer:
            raise ValueError(f"calibration layers are layer-major: expected {self.next_layer}")
        prefix = f"model.language_model.layers.{layer}."
        weights = self.checkpoint.load_many(backend._layer_names(layer), self.device)
        watched = {id(weights[prefix + name]): name for name in _INPUT_OF if prefix + name in weights}
        moments: dict[str, torch.Tensor] = {}
        linear = backend._linear

        def recording(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            name = watched.get(id(weight))
            if name is not None:
                rows = x.reshape(-1, x.shape[-1]).float()
                moment = moments.get(name)
                if moment is None:
                    moments[name] = rows.t() @ rows
                else:
                    moment.addmm_(rows.t(), rows)
            return linear(x, weight)

        backend._linear = recording
        try:
            with torch.inference_mode():
                for hidden in self.hidden:
                    if layer in backend.FULL_ATTENTION_LAYERS:
                        self.scorer._full_attention_layer(hidden, layer, weights, 0)
                    else:
                        self.scorer._gdn_layer(hidden, layer, weights, 0)
                    for begin, end in self.scorer._spans(hidden.shape[0], 0):
                        self.scorer._mlp_span(hidden, begin, end, layer, weights)
        finally:
            backend._linear = linear
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        self.next_layer += 1
        return {suffix: moments[name] for name in moments for suffix in _INPUT_OF[name]}

    def final(self) -> torch.Tensor:
        """Second moment of the output-head input (the final RMSNorm of the last layer's output)."""
        if self.next_layer != LAYERS:
            raise ValueError("output-head moments require every Text layer first")
        norm = self.checkpoint.load("model.language_model.norm.weight", self.device)
        moment = None
        with torch.inference_mode():
            for hidden in self.hidden:
                rows = backend._rmsnorm(hidden, norm).float()
                moment = rows.t() @ rows if moment is None else moment.addmm_(rows.t(), rows)
        return moment


__all__ = ["InputMoments", "read_sequences"]
