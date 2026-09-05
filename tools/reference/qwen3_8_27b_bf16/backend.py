"""Layer-major checkpoint-native Qwen3.8-27B BF16 text evaluation.

Only one original BF16 layer is resident on the accelerator at a time. The
causal graph is evaluated in layer-major topological order over one hidden
sequence. Prefill/decode are comparison labels with independently selected
score starts; product candidates, not this mathematical oracle, exercise T=1.
"""

from __future__ import annotations

from collections import defaultdict
import gc
from pathlib import Path
import time
from typing import Iterable

try:
    import torch
    import torch.nn.functional as F
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    from safetensors import safe_open
except (ImportError, OSError) as exc:  # pragma: no cover - exercised only with model dependencies
    raise RuntimeError(
        "BF16 source scoring requires PyTorch with ROCm, safetensors, and "
        "flash-linear-attention; "
        "install the package requirements explicitly"
    ) from exc

from .protocol import (
    ATTENTION_PV_SOURCE_ROW_CHUNK,
    FULL_ATTENTION_LAYERS,
    GDN_QKV_RANGES,
    HIDDEN_SIZE,
    ScoreVectors,
    TOKEN_DOMAIN,
    expected_text_tensors,
    layer_spans,
    resolve_score_begin,
    validate_scored_targets,
)
from .kv_tensor_codec import QuantizedFullAttentionCache, validate_codec_device


HEAD_DIM = 256
Q_HEADS = 24
KV_HEADS = 4
Q_PER_KV = Q_HEADS // KV_HEADS
ROTARY_DIM = 64
ROPE_THETA = 1.0e7
RMS_EPS = 1.0e-6
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_KEY_DIM = 128
GDN_VALUE_DIM = 128
GDN_SCALE = GDN_KEY_DIM**-0.5
ATTENTION_SCALE = HEAD_DIM**-0.5
CONV_CHANNELS = 10240
CONV_WIDTH = 4
INTERMEDIATE = 17408
LOGIT_ROWS = 64
ATTENTION_QUERY_ROWS = 32


class SourceCheckpoint:
    def __init__(self, root: Path, weight_map: dict[str, str]):
        self.root = root
        self.weight_map = weight_map

    def validate_metadata(self) -> None:
        """Prove the 851 text tensors are original BF16 values at exact shapes."""
        by_shard: dict[str, list] = defaultdict(list)
        for requirement in expected_text_tensors():
            by_shard[self.weight_map[requirement.name]].append(requirement)
        for shard, requirements in sorted(by_shard.items()):
            with safe_open(str(self.root / shard), framework="pt", device="cpu") as handle:
                available = frozenset(handle.keys())
                for requirement in requirements:
                    if requirement.name not in available:
                        raise ValueError(
                            f"{shard} does not contain indexed tensor {requirement.name}"
                        )
                    tensor_slice = handle.get_slice(requirement.name)
                    shape = tuple(tensor_slice.get_shape())
                    dtype = str(tensor_slice.get_dtype())
                    if shape != requirement.shape or dtype != requirement.dtype:
                        raise ValueError(
                            f"{requirement.name} is {dtype}{shape}; expected "
                            f"{requirement.dtype}{requirement.shape}"
                        )

    def load(self, name: str, device: torch.device) -> torch.Tensor:
        shard = self.weight_map[name]
        with safe_open(str(self.root / shard), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"{name} is not source BF16")
            return tensor.to(device=device, dtype=torch.bfloat16)

    def load_many(self, names: Iterable[str], device: torch.device) -> dict[str, torch.Tensor]:
        requested = tuple(names)
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name in requested:
            by_shard[self.weight_map[name]].append(name)
        result: dict[str, torch.Tensor] = {}
        for shard, shard_names in by_shard.items():
            with safe_open(str(self.root / shard), framework="pt", device="cpu") as handle:
                for name in shard_names:
                    tensor = handle.get_tensor(name)
                    if tensor.dtype != torch.bfloat16:
                        raise ValueError(f"{name} is not source BF16")
                    result[name] = tensor.to(device=device, dtype=torch.bfloat16)
        return result

    def embedding_rows(self, ids: list[int], device: torch.device) -> torch.Tensor:
        name = "model.language_model.embed_tokens.weight"
        shard = self.weight_map[name]
        with safe_open(str(self.root / shard), framework="pt", device="cpu") as handle:
            embedding = handle.get_tensor(name)
            if embedding.dtype != torch.bfloat16:
                raise ValueError("source embedding is not BF16")
            index = torch.tensor(ids, dtype=torch.long)
            selected = embedding.index_select(0, index)
        return selected.to(device=device, dtype=torch.bfloat16)


def _bf16(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16)


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _bf16(x.to(torch.bfloat16) @ weight.to(torch.bfloat16).t())


def _rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    unit_offset: bool = True,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + RMS_EPS)
    scale = weight.float() + (1.0 if unit_offset else 0.0)
    out = xf * inv * scale
    if z is not None:
        out = out * F.silu(z.float())
    return _bf16(out)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return _bf16(xf * torch.rsqrt(torch.sum(xf * xf, dim=-1, keepdim=True) + RMS_EPS))


def _residual(x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    return _bf16(x.float() + update.float())


def _rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    pair = torch.arange(ROTARY_DIM // 2, device=x.device, dtype=torch.float32)
    frequency = torch.pow(
        torch.tensor(ROPE_THETA, device=x.device, dtype=torch.float32),
        -2.0 * pair / ROTARY_DIM,
    )
    angle = positions.float()[:, None] * frequency[None, :]
    cosine = torch.cos(angle)[:, None, :]
    sine = torch.sin(angle)[:, None, :]
    half = ROTARY_DIM // 2
    first = x[:, :, :half].float()
    second = x[:, :, half:ROTARY_DIM].float()
    output = x.clone()
    output[:, :, :half] = _bf16(first * cosine - second * sine)
    output[:, :, half:ROTARY_DIM] = _bf16(second * cosine + first * sine)
    return output


def _attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    query_begin: int,
    detail_trace=None,
) -> torch.Tensor:
    """Independent FP32 softmax and PV over BF16 GQA operands."""
    rows = query.shape[0]
    output = torch.empty_like(query, dtype=torch.bfloat16)
    all_key_positions = torch.arange(keys.shape[0], device=query.device)
    for begin in range(0, rows, ATTENTION_QUERY_ROWS):
        end = min(rows, begin + ATTENTION_QUERY_ROWS)
        absolute_query = torch.arange(
            query_begin + begin, query_begin + end, device=query.device
        )
        allowed = all_key_positions.unsqueeze(0) <= absolute_query.unsqueeze(1)
        for kv_head in range(KV_HEADS):
            q0 = kv_head * Q_PER_KV
            q1 = q0 + Q_PER_KV
            q = query[begin:end, q0:q1].float()
            k = keys[:, kv_head].float()
            v = values[:, kv_head].float()
            scores = torch.einsum("thd,sd->ths", q, k) * ATTENTION_SCALE
            scores = scores.masked_fill(~allowed[:, None, :], -torch.inf)
            if detail_trace:
                detail_trace.capture_attention_kernel_detail(
                    "masked-qk-fp32", kv_head, scores,
                    query_begin + begin, query_begin + end,
                )
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
            if detail_trace:
                detail_trace.capture_attention_kernel_detail(
                    "softmax-fp32", kv_head, probabilities,
                    query_begin + begin, query_begin + end,
                )
            attended = _attention_pv(probabilities, v)
            if detail_trace:
                detail_trace.capture_attention_kernel_detail(
                    "pv-fp32", kv_head, attended,
                    query_begin + begin, query_begin + end,
                )
            output[begin:end, q0:q1] = _bf16(attended)
    return output


def _attention_pv(probabilities: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Deterministic FP32 PV in fixed ascending absolute source-row chunks."""
    probabilities_fp32 = probabilities.to(dtype=torch.float32)
    values_fp32 = values.to(dtype=torch.float32)
    source_rows = values_fp32.shape[0]
    if probabilities_fp32.shape[-1] != source_rows:
        raise ValueError("attention probability/value source-row counts differ")

    feature_rows = values_fp32.shape[1]
    attended = torch.zeros(
        (*probabilities_fp32.shape[:-1], feature_rows),
        device=probabilities_fp32.device,
        dtype=torch.float32,
    )
    attended_flat = attended.view(-1, feature_rows)
    for source_begin in range(0, source_rows, ATTENTION_PV_SOURCE_ROW_CHUNK):
        source_end = min(source_rows, source_begin + ATTENTION_PV_SOURCE_ROW_CHUNK)
        probability_chunk = probabilities_fp32[..., source_begin:source_end].reshape(
            -1, source_end - source_begin
        )
        attended_flat.add_(torch.mm(probability_chunk, values_fp32[source_begin:source_end]))
    return attended


def _gdn_recurrence_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit FP32 state-transition authority, used by every T=1 decode span."""
    rows = q.shape[0]
    output = torch.empty_like(value, dtype=torch.bfloat16)
    head_map = torch.arange(GDN_VALUE_HEADS, device=q.device) // (
        GDN_VALUE_HEADS // GDN_KEY_HEADS
    )
    current = state
    for row in range(rows):
        key = k[row].float().index_select(0, head_map)
        query = q[row].float().index_select(0, head_map)
        current.mul_(torch.exp(decay[row].float()).view(GDN_VALUE_HEADS, 1, 1))
        prediction = torch.einsum("hkv,hk->hv", current, key)
        delta = beta[row].float().unsqueeze(-1) * (value[row].float() - prediction)
        current.add_(key.unsqueeze(-1) * delta.unsqueeze(-2))
        result = torch.einsum("hkv,hk->hv", current, query) * GDN_SCALE
        output[row] = _bf16(result)
    return output, current


def _gdn_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.shape[0] == 1:
        return _gdn_recurrence_naive(q, k, value, decay, beta, state)
    output, final = fused_recurrent_gated_delta_rule(
        q.unsqueeze(0).clone(),
        k.unsqueeze(0).clone(),
        value.unsqueeze(0).clone(),
        g=decay.unsqueeze(0).clone(),
        beta=beta.unsqueeze(0).clone(),
        scale=GDN_SCALE,
        initial_state=state.unsqueeze(0),
        output_final_state=True,
    )
    return _bf16(output.squeeze(0)), final.squeeze(0).float()


def _layer_names(layer: int) -> tuple[str, ...]:
    prefix = f"model.language_model.layers.{layer}."
    common = (
        prefix + "input_layernorm.weight",
        prefix + "post_attention_layernorm.weight",
        prefix + "mlp.gate_proj.weight",
        prefix + "mlp.up_proj.weight",
        prefix + "mlp.down_proj.weight",
    )
    if layer in FULL_ATTENTION_LAYERS:
        return common + (
            prefix + "self_attn.q_proj.weight",
            prefix + "self_attn.k_proj.weight",
            prefix + "self_attn.v_proj.weight",
            prefix + "self_attn.q_norm.weight",
            prefix + "self_attn.k_norm.weight",
            prefix + "self_attn.o_proj.weight",
        )
    return common + (
        prefix + "linear_attn.A_log",
        prefix + "linear_attn.dt_bias",
        prefix + "linear_attn.conv1d.weight",
        prefix + "linear_attn.in_proj_a.weight",
        prefix + "linear_attn.in_proj_b.weight",
        prefix + "linear_attn.in_proj_qkv.weight",
        prefix + "linear_attn.in_proj_z.weight",
        prefix + "linear_attn.norm.weight",
        prefix + "linear_attn.out_proj.weight",
    )


class LayerMajorTextScorer:
    def __init__(
        self,
        checkpoint: SourceCheckpoint,
        *,
        device_index: int,
        prefill_chunk: int,
        schedule: str,
        skip_text: str,
        kv_value_group: int | None,
        trace=None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("BF16 source scoring requires a ROCm-enabled PyTorch accelerator")
        self.device = torch.device("cuda", device_index)
        try:
            torch.cuda.get_device_properties(self.device)
        except Exception as exc:
            raise RuntimeError(f"ROCm device {device_index} is unavailable") from exc
        self.checkpoint = checkpoint
        self.prefill_chunk = prefill_chunk
        self.schedule = schedule
        self.skip_text = skip_text
        if kv_value_group not in (None, 16, 32):
            raise ValueError("BF16-source KV diagnostic group must be G16 or G32")
        self.kv_value_group = kv_value_group
        self.trace = trace
        if self.kv_value_group is not None:
            validate_codec_device(self.device, self.kv_value_group)

    def _spans(self, length: int, prefix: int) -> Iterable[tuple[int, int]]:
        return layer_spans(self.schedule, length, prefix, self.prefill_chunk)

    def _full_attention_layer(
        self,
        hidden: torch.Tensor,
        layer: int,
        weights: dict[str, torch.Tensor],
        prefix: int,
    ) -> None:
        root = f"model.language_model.layers.{layer}."
        length = hidden.shape[0]
        if self.kv_value_group is None:
            key_cache = torch.empty(
                (length, KV_HEADS, HEAD_DIM), device=self.device, dtype=torch.bfloat16
            )
            value_cache = torch.empty_like(key_cache)
            quantized_cache = None
        else:
            key_cache = None
            value_cache = None
            quantized_cache = QuantizedFullAttentionCache(
                length,
                KV_HEADS,
                HEAD_DIM,
                self.kv_value_group,
                self.device,
            )
        for begin, end in self._spans(length, prefix):
            x = hidden[begin:end]
            normalized = _rmsnorm(x, weights[root + "input_layernorm.weight"])
            if self.trace:
                self.trace.capture_attention_detail(
                    layer, "normalized-input", normalized, begin, end
                )
            projected = _linear(normalized, weights[root + "self_attn.q_proj.weight"])
            projected = projected.reshape(end - begin, Q_HEADS, 2 * HEAD_DIM)
            if self.trace:
                self.trace.capture_attention_detail(
                    layer, "q-gate-projection", projected, begin, end
                )
            query = projected[:, :, :HEAD_DIM]
            gate = projected[:, :, HEAD_DIM:]
            key = _linear(normalized, weights[root + "self_attn.k_proj.weight"]).reshape(
                end - begin, KV_HEADS, HEAD_DIM
            )
            if self.trace:
                self.trace.capture_attention_detail(layer, "k-projection", key, begin, end)
            value = _linear(normalized, weights[root + "self_attn.v_proj.weight"]).reshape(
                end - begin, KV_HEADS, HEAD_DIM
            )
            if self.trace:
                self.trace.capture_attention_detail(layer, "v-projection", value, begin, end)
            positions = torch.arange(begin, end, device=self.device, dtype=torch.int32)
            query = _rope(
                _rmsnorm(query, weights[root + "self_attn.q_norm.weight"]), positions
            )
            key = _rope(_rmsnorm(key, weights[root + "self_attn.k_norm.weight"]), positions)
            if self.trace:
                self.trace.capture_attention_detail(layer, "rope-q", query, begin, end)
                self.trace.capture_attention_detail(layer, "rope-k", key, begin, end)
            if quantized_cache is None:
                assert key_cache is not None and value_cache is not None
                key_cache[begin:end] = key
                value_cache[begin:end] = value
                represented_keys = key_cache[:end]
                represented_values = value_cache[:end]
            else:
                # The cache-only diagnostic changes exactly this semantic boundary: append the
                # represented post-RoPE BF16 K and projected BF16 V into the fixed codecs, then
                # decode their stored planes only when attention consumes the growing prefix.
                quantized_cache.append(begin, key, value)
                represented_keys, represented_values = quantized_cache.read(end)
            attended = _attention(
                query, represented_keys, represented_values, begin,
                self.trace if layer == 3 else None,
            )
            if self.trace:
                self.trace.capture_attention_detail(
                    layer, "attended-output", attended, begin, end
                )
            gated = _bf16(torch.sigmoid(gate.float()) * attended.float()).reshape(
                end - begin, Q_HEADS * HEAD_DIM
            )
            if self.trace:
                self.trace.capture_attention_detail(
                    layer, "gated-flatten", gated, begin, end
                )
            update = _linear(gated, weights[root + "self_attn.o_proj.weight"])
            if self.trace:
                self.trace.capture_attention_detail(
                    layer, "output-projection-update", update, begin, end
                )
            residual = _residual(x, update)
            if self.trace:
                self.trace.capture_attention_detail(
                    layer, "post-residual", residual, begin, end
                )
            hidden[begin:end] = residual

    def _gdn_layer(
        self,
        hidden: torch.Tensor,
        layer: int,
        weights: dict[str, torch.Tensor],
        prefix: int,
    ) -> None:
        root = f"model.language_model.layers.{layer}."
        conv_state = torch.zeros(
            (CONV_CHANNELS, CONV_WIDTH - 1), device=self.device, dtype=torch.float32
        )
        state = torch.zeros(
            (GDN_VALUE_HEADS, GDN_KEY_DIM, GDN_VALUE_DIM),
            device=self.device,
            dtype=torch.float32,
        )
        a_log = weights[root + "linear_attn.A_log"].float()
        dt_bias = weights[root + "linear_attn.dt_bias"].float()
        conv_weight = weights[root + "linear_attn.conv1d.weight"].float()
        for begin, end in self._spans(hidden.shape[0], prefix):
            x = hidden[begin:end]
            normalized = _rmsnorm(x, weights[root + "input_layernorm.weight"])
            qkv = _linear(normalized, weights[root + "linear_attn.in_proj_qkv.weight"])
            a = _linear(normalized, weights[root + "linear_attn.in_proj_a.weight"]).float()
            b = _linear(normalized, weights[root + "linear_attn.in_proj_b.weight"]).float()
            z = _linear(normalized, weights[root + "linear_attn.in_proj_z.weight"])
            sequence = torch.cat((conv_state.t(), qkv.float()), dim=0)
            convolved = F.conv1d(
                sequence.t().unsqueeze(0), conv_weight, groups=CONV_CHANNELS
            ).squeeze(0).t()
            conv_state = sequence[-(CONV_WIDTH - 1) :].t().contiguous()
            qkv = _bf16(F.silu(convolved))
            (q0, q1), (k0, k1), (v0, v1) = GDN_QKV_RANGES
            query = _l2norm(qkv[:, q0:q1].reshape(-1, GDN_KEY_HEADS, GDN_KEY_DIM))
            key = _l2norm(qkv[:, k0:k1].reshape(-1, GDN_KEY_HEADS, GDN_KEY_DIM))
            value = qkv[:, v0:v1].reshape(-1, GDN_VALUE_HEADS, GDN_VALUE_DIM)
            softplus_input = a + dt_bias
            softplus = torch.where(
                softplus_input > 20.0, softplus_input, F.softplus(softplus_input)
            )
            decay = -torch.exp(a_log) * softplus
            beta = torch.sigmoid(b)
            recurrent, state = _gdn_recurrence(
                query, key, value, decay, beta, state
            )
            recurrent = _rmsnorm(
                recurrent,
                weights[root + "linear_attn.norm.weight"],
                unit_offset=False,
                z=z.reshape(-1, GDN_VALUE_HEADS, GDN_VALUE_DIM),
            )
            update = _linear(
                recurrent.reshape(-1, GDN_VALUE_HEADS * GDN_VALUE_DIM),
                weights[root + "linear_attn.out_proj.weight"],
            )
            hidden[begin:end] = _residual(x, update)

    @staticmethod
    def _mlp_span(
        hidden: torch.Tensor,
        begin: int,
        end: int,
        layer: int,
        weights: dict[str, torch.Tensor],
    ) -> None:
        root = f"model.language_model.layers.{layer}."
        x = hidden[begin:end]
        normalized = _rmsnorm(x, weights[root + "post_attention_layernorm.weight"])
        gate = _linear(normalized, weights[root + "mlp.gate_proj.weight"])
        up = _linear(normalized, weights[root + "mlp.up_proj.weight"])
        activated = _bf16(F.silu(gate.float()) * up.float())
        update = _linear(activated, weights[root + "mlp.down_proj.weight"])
        hidden[begin:end] = _residual(x, update)

    def _score_output(self, hidden: torch.Tensor, ids: list[int], skip: int) -> tuple[list[float], list[int]]:
        norm = self.checkpoint.load("model.language_model.norm.weight", self.device)
        for begin in range(0, hidden.shape[0], self.prefill_chunk):
            end = min(hidden.shape[0], begin + self.prefill_chunk)
            hidden[begin:end] = _rmsnorm(hidden[begin:end], norm)
        del norm
        if self.trace: self.trace.capture_hidden("final-norm", hidden)
        head = self.checkpoint.load("lm_head.weight", self.device)
        nlls: list[float] = []
        argmax_ids: list[int] = []
        for begin in range(skip, hidden.shape[0], LOGIT_ROWS):
            end = min(hidden.shape[0], begin + LOGIT_ROWS)
            logits = _linear(hidden[begin:end], head)[:, :TOKEN_DOMAIN]
            if self.trace: self.trace.capture_logits(begin, end, logits)
            logits_fp32 = logits.float()
            targets = torch.tensor(ids[begin + 1 : end + 1], device=self.device, dtype=torch.long)
            nll = torch.logsumexp(logits_fp32, dim=-1) - logits_fp32.gather(
                1, targets[:, None]
            ).squeeze(1)
            argmax = torch.argmax(logits_fp32, dim=-1)
            nlls.extend(float(value) for value in nll.cpu().tolist())
            argmax_ids.extend(int(value) for value in argmax.cpu().tolist())
        return nlls, argmax_ids

    def score(self, ids: list[int]) -> ScoreVectors:
        prompt_tokens = len(ids)
        scored_begin = resolve_score_begin(prompt_tokens, self.schedule, self.skip_text)
        prefix = scored_begin if self.schedule == "decode" else 0
        validate_scored_targets(ids, scored_begin)
        # The last prompt token has no teacher-forced target and cannot affect an earlier causal
        # output. Excluding it saves one hidden row while preserving every represented score.
        hidden = self.checkpoint.embedding_rows(ids[:-1], self.device)
        if self.trace: self.trace.capture_hidden("embedding", hidden)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            for layer in range(64):
                weights = self.checkpoint.load_many(_layer_names(layer), self.device)
                if layer in FULL_ATTENTION_LAYERS:
                    self._full_attention_layer(hidden, layer, weights, prefix)
                else:
                    self._gdn_layer(hidden, layer, weights, prefix)
                if self.trace: self.trace.capture_hidden(f"layer-{layer:02d}.{'attention' if layer in FULL_ATTENTION_LAYERS else 'gdn'}", hidden)
                for begin, end in self._spans(hidden.shape[0], prefix):
                    self._mlp_span(hidden, begin, end, layer, weights)
                if self.trace: self.trace.capture_hidden(f"layer-{layer:02d}.mlp", hidden)
                del weights
                gc.collect()
                torch.cuda.empty_cache()
            nlls, argmax = self._score_output(hidden, ids, scored_begin)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        expected = prompt_tokens - scored_begin - 1
        if len(nlls) != expected or len(argmax) != expected:
            raise RuntimeError("BF16 source score lost teacher-forced alignment")
        return ScoreVectors(
            prompt_tokens=prompt_tokens,
            skip_tokens=scored_begin,
            nlls=tuple(nlls),
            argmax=tuple(argmax),
            score_seconds=elapsed,
        )


__all__ = ["LayerMajorTextScorer", "SourceCheckpoint"]
