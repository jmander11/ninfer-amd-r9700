"""Qwen3.8 diagnostic-model cache and recurrent state ownership."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import CFG


class KVCache:
    """PyTorch implementation profile of the FP8-K/INT4-V-G16 cache codec.

    Tensor casts and rounding here deliberately exercise the library path; they
    are not the independent scalar oracle for the production codec.
    """

    VALUE_GROUP = 16
    VALUE_CODE_MAX = 7
    KEY_MAX_FINITE = 448.0

    def __init__(self, layers: int, capacity: int, device: torch.device):
        if capacity <= 0:
            raise ValueError("KV capacity must be positive")
        if CFG.head_dim % self.VALUE_GROUP != 0 or CFG.head_dim % 2 != 0:
            raise ValueError("KV head dimension is incompatible with INT4-V-G16")
        self.layers = layers
        self.capacity = capacity
        self.device = device
        self.length = 0
        self._k: dict[int, torch.Tensor] = {}
        self._v: dict[int, torch.Tensor] = {}
        self._vs: dict[int, torch.Tensor] = {}

    def _allocate(self, layer: int) -> None:
        if layer in self._k:
            return
        if layer < 0 or layer >= self.layers:
            raise ValueError("KV layer is outside the cache")
        self._k[layer] = torch.empty(
            (self.capacity, CFG.kv_heads, CFG.head_dim),
            device=self.device,
            dtype=torch.float8_e4m3fn,
        )
        self._v[layer] = torch.empty(
            (self.capacity, CFG.kv_heads, CFG.head_dim // 2),
            device=self.device,
            dtype=torch.uint8,
        )
        self._vs[layer] = torch.empty(
            (self.capacity, CFG.kv_heads, CFG.head_dim // self.VALUE_GROUP),
            device=self.device,
            dtype=torch.float16,
        )

    @classmethod
    def _quantize_values(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(torch.isfinite(x).all()):
            raise ValueError("KV values must be finite")
        groups = x.float().reshape(
            *x.shape[:-1], CFG.head_dim // cls.VALUE_GROUP, cls.VALUE_GROUP
        )
        scale = (groups.abs().amax(dim=-1) / cls.VALUE_CODE_MAX).to(torch.float16)
        if not bool(torch.isfinite(scale).all()):
            raise ValueError("KV value scale overflows FP16")
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale).float()
        code = (
            torch.round(groups / safe_scale.unsqueeze(-1))
            .clamp(-cls.VALUE_CODE_MAX, cls.VALUE_CODE_MAX)
            .to(torch.int8)
        )
        code = torch.where((scale == 0).unsqueeze(-1), 0, code).to(torch.int8)
        low = code[..., 0::2].to(torch.int16) & 0x0F
        high = code[..., 1::2].to(torch.int16) & 0x0F
        packed = (low | (high << 4)).to(torch.uint8)
        return packed.reshape(*x.shape[:-1], CFG.head_dim // 2), scale

    @classmethod
    def _dequantize_values(
        cls, packed: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        packed_i16 = packed.to(torch.int16)
        low = packed_i16 & 0x0F
        high = (packed_i16 >> 4) & 0x0F
        codes = torch.stack((low, high), dim=-1).reshape(
            *packed.shape[:-1], CFG.head_dim
        )
        codes = torch.where(codes >= 8, codes - 16, codes).float()
        groups = codes.reshape(
            *codes.shape[:-1], CFG.head_dim // cls.VALUE_GROUP, cls.VALUE_GROUP
        )
        return (
            groups * scale.float().unsqueeze(-1)
        ).reshape(*codes.shape).to(torch.bfloat16)

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        end = start + k.shape[0]
        if start < 0 or end > self.capacity or v.shape != k.shape:
            raise ValueError("KV write range or shape mismatch")
        if tuple(k.shape[1:]) != (CFG.kv_heads, CFG.head_dim):
            raise ValueError("KV write tensor shape mismatch")
        self._allocate(layer)
        if not bool(torch.isfinite(k).all()):
            raise ValueError("KV keys must be finite")
        key_codes = k.float().clamp(-self.KEY_MAX_FINITE, self.KEY_MAX_FINITE).to(
            torch.float8_e4m3fn
        )
        value_codes, value_scales = self._quantize_values(v)
        self._k[layer][start:end].copy_(key_codes)
        self._v[layer][start:end].copy_(value_codes)
        self._vs[layer][start:end].copy_(value_scales)

    def read(self, layer: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        if end < 0 or end > self.capacity or layer not in self._k:
            raise ValueError("KV read range or layer mismatch")
        return (
            self._k[layer][:end].to(torch.bfloat16),
            self._dequantize_values(self._v[layer][:end], self._vs[layer][:end]),
        )

    def rewind(self, position: int) -> None:
        if position < 0 or position > self.length:
            raise ValueError("KV rewind cannot move forward")
        self.length = position


@dataclass
class ModelState:
    device: torch.device
    capacity: int
    kv: KVCache = field(init=False)
    mtp_kv: KVCache = field(init=False)
    conv: list[torch.Tensor] = field(init=False)
    ssm: list[torch.Tensor] = field(init=False)
    position: int = 0
    rope_delta: int = 0
    mrope: bool = False

    def __post_init__(self) -> None:
        self.kv = KVCache(CFG.full_layers, self.capacity, self.device)
        self.mtp_kv = KVCache(1, self.capacity, self.device)
        self.conv = [
            torch.zeros(CFG.conv_dim, CFG.conv_width - 1, device=self.device, dtype=torch.float32)
            for _ in range(CFG.gdn_layers)
        ]
        self.ssm = [
            torch.zeros(
                1,
                CFG.gdn_v_heads,
                CFG.gdn_k_dim,
                CFG.gdn_v_dim,
                device=self.device,
                dtype=torch.float32,
            )
            for _ in range(CFG.gdn_layers)
        ]

    def snapshot(self) -> "StateSnapshot":
        return StateSnapshot(
            position=self.position,
            rope_delta=self.rope_delta,
            mrope=self.mrope,
            kv_length=self.kv.length,
            mtp_kv_length=self.mtp_kv.length,
            conv=[tensor.clone() for tensor in self.conv],
            ssm=[tensor.clone() for tensor in self.ssm],
        )

    def restore(self, snapshot: "StateSnapshot") -> None:
        if snapshot.position < 0 or snapshot.position > self.capacity:
            raise ValueError("snapshot position is outside state capacity")
        self.position = snapshot.position
        self.rope_delta = snapshot.rope_delta
        self.mrope = snapshot.mrope
        self.kv.length = snapshot.kv_length
        self.mtp_kv.length = snapshot.mtp_kv_length
        for target, source in zip(self.conv, snapshot.conv, strict=True):
            target.copy_(source)
        for target, source in zip(self.ssm, snapshot.ssm, strict=True):
            target.copy_(source)


@dataclass
class StateSnapshot:
    position: int
    rope_delta: int
    mrope: bool
    kv_length: int
    mtp_kv_length: int
    conv: list[torch.Tensor]
    ssm: list[torch.Tensor]
