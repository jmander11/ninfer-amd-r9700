"""Evaluation-only round-trip cache codecs for the checkpoint-direct BF16 scorer.

These codecs compare candidate growing-cache representations on the independent BF16 model.
Each append encodes represented post-RoPE BF16 K and projected BF16 V into a codec and
immediately decodes it; attention consumes the decoded FP32 values. The stored planes are
therefore implicit, while every reconstructed value equals the decode of its codes.

NVFP4 (product semantics): per 16 contiguous features, scale = E4M3FN_satfinite(amax / 6) and
code = E2M1_rne(x / scale) saturated at +/-6; a zero scale yields zero codes. The "4/6" encoder
also tries scale = E4M3FN_satfinite(amax / 4) and keeps whichever reconstruction has the lower
squared error (6 on ties). Its decode and storage are identical to NVFP4.

TurboQuant (MSE variant with norm correction): y = H (s * x) with the orthonormal 256-point
Sylvester-Hadamard matrix H and fixed signs s; u = y * sqrt(256) / ||x||; each coordinate of u
takes the nearest Gaussian Lloyd-Max level of its bit width; the stored FP16 scale is
||x|| / ||c|| (norm correction) so the decode s * H^T (scale * c) has the source norm up to
FP16 rounding. A zero vector decodes to zero. "3.5-bit" gives the first 128 rotated
coordinates 4 bits and the remaining 128 3 bits.
"""

from __future__ import annotations

import math

import torch

HEAD_DIM = 256
E4M3FN_MAX = 448.0
FP16_MAX = 65504.0
NVFP4_GROUP = 16
# Gaussian Lloyd-Max reconstruction magnitudes (Max, 1960), positive half of a symmetric set.
LLOYD_MAX_LEVELS = {
    3: (0.2451, 0.7560, 1.3439, 2.1520),
    4: (0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326),
}

KEY_VALUE_CODECS = {
    # name: (key codec, value codec)
    "nvfp4": ("nvfp4", "nvfp4"),
    "nvfp4-46": ("nvfp4-46", "nvfp4-46"),
    "k-nvfp4-46-v-tq3": ("nvfp4-46", "tq3"),
    "k-tq4-v-tq3": ("tq4", "tq3"),
    "k-tq4-v-tq3p5": ("tq4", "tq3p5"),
    "k-tq4-v-tq4": ("tq4", "tq4"),
    # The product tq-k4-v3 codec (ops/r9700/kv/tq_k4v3_codec.h): binary16-high-byte levels.
    "tq-k4-v3": ("ptq4", "ptq3"),
}
# Stored bytes per 256-feature head vector, including scales or the FP16 norm.
CODEC_BYTES = {
    "nvfp4": 128 + 16,
    "nvfp4-46": 128 + 16,
    "tq4": 128 + 2,
    "tq3": 96 + 2,
    "tq3p5": 112 + 2,
    "ptq4": 128 + 2,
    "ptq3": 96 + 2,
}
# Product levels (positive half) and level-domain scale alpha: index = number of midpoints that
# |y| * 16 * alpha / ||x|| exceeds.
PRODUCT_LEVELS = {
    "ptq4": ((0.09375, 0.3125, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0), 0.75),
    "ptq3": ((0.25, 0.75, 1.25, 2.0), 0.96),
}
TQ_BITS = {
    "tq4": (4,) * HEAD_DIM,
    "tq3": (3,) * HEAD_DIM,
    "tq3p5": (4,) * (HEAD_DIM // 2) + (3,) * (HEAD_DIM // 2),
}


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def rotation_signs() -> tuple[int, ...]:
    """Fixed, library-independent +/-1 diagonal of the randomized Hadamard rotation."""
    return tuple(1 if _splitmix64(index) & 1 else -1 for index in range(HEAD_DIM))


def hadamard_matrix(device: torch.device) -> torch.Tensor:
    """Orthonormal Sylvester-Hadamard matrix of order 256 (entries +/-1/16, exact in FP32)."""
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < HEAD_DIM:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)), dim=0
        )
    return (matrix / math.sqrt(HEAD_DIM)).to(device)


def _e4m3_satfinite(values: torch.Tensor) -> torch.Tensor:
    return values.clamp(-E4M3FN_MAX, E4M3FN_MAX).to(torch.float8_e4m3fn).float()


def _e2m1_rne_saturated(values: torch.Tensor) -> torch.Tensor:
    """Round to the signed E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to even, saturating."""
    magnitude = values.abs()
    code = torch.full_like(magnitude, 6.0)
    # Assign from the top down so each tie lands on the even-mantissa neighbor.
    code = torch.where(magnitude <= 5.0, 4.0, code)
    code = torch.where(magnitude < 3.5, 3.0, code)
    code = torch.where(magnitude <= 2.5, 2.0, code)
    code = torch.where(magnitude < 1.75, 1.5, code)
    code = torch.where(magnitude <= 1.25, 1.0, code)
    code = torch.where(magnitude < 0.75, 0.5, code)
    code = torch.where(magnitude <= 0.25, 0.0, code)
    return torch.copysign(code, values)


def _nvfp4_groups(groups: torch.Tensor, divisor: float) -> torch.Tensor:
    amax = groups.abs().amax(dim=-1, keepdim=True)
    scale = _e4m3_satfinite(amax / divisor)
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = _e2m1_rne_saturated(groups / safe)
    return torch.where(scale == 0, torch.zeros_like(groups), codes * scale)


def round_trip_nvfp4(source: torch.Tensor, *, four_over_six: bool) -> torch.Tensor:
    if source.shape[-1] % NVFP4_GROUP:
        raise ValueError("NVFP4 requires a final dimension divisible by 16")
    values = source.float()
    groups = values.reshape(*values.shape[:-1], values.shape[-1] // NVFP4_GROUP, NVFP4_GROUP)
    decoded = _nvfp4_groups(groups, 6.0)
    if four_over_six:
        alternate = _nvfp4_groups(groups, 4.0)
        error = (decoded - groups).square().sum(dim=-1, keepdim=True)
        alternate_error = (alternate - groups).square().sum(dim=-1, keepdim=True)
        decoded = torch.where(alternate_error < error, alternate, decoded)
    return decoded.reshape(values.shape)


class TurboQuantRoundTrip:
    """Vectorized TurboQuant MSE encode/decode for 256-feature head vectors."""

    def __init__(self, name: str, device: torch.device) -> None:
        self.device = device
        self.hadamard = hadamard_matrix(device)
        self.signs = torch.tensor(rotation_signs(), device=device, dtype=torch.float32)
        self.masks = []
        if name in PRODUCT_LEVELS:
            positive, alpha = PRODUCT_LEVELS[name]
            levels = torch.tensor(
                tuple(-level for level in reversed(positive)) + positive,
                device=device,
                dtype=torch.float32,
            )
            # Midpoints in the alpha-scaled level domain mapped back to unit-variance units.
            thresholds = (levels[1:] + levels[:-1]) / 2 / alpha
            self.masks.append(
                (torch.ones(HEAD_DIM, dtype=torch.bool, device=device), levels, thresholds)
            )
            return
        bits = torch.tensor(TQ_BITS[name], device=device)
        for width, positive in LLOYD_MAX_LEVELS.items():
            mask = bits == width
            if bool(mask.any()):
                levels = torch.tensor(
                    tuple(-level for level in reversed(positive)) + positive,
                    device=device,
                    dtype=torch.float32,
                )
                thresholds = (levels[1:] + levels[:-1]) / 2
                self.masks.append((mask, levels, thresholds))

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if source.shape[-1] != HEAD_DIM:
            raise ValueError("TurboQuant evaluation codec requires 256-feature head vectors")
        values = source.float()
        norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
        rotated = (values * self.signs) @ self.hadamard.T
        safe_norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        unit = rotated * (math.sqrt(HEAD_DIM) / safe_norm)
        codes = torch.empty_like(unit)
        for mask, levels, thresholds in self.masks:
            index = torch.bucketize(unit[..., mask].contiguous(), thresholds)
            codes[..., mask] = levels[index]
        code_norm = torch.linalg.vector_norm(codes, dim=-1, keepdim=True)
        scale = norm / code_norm
        if bool((scale > FP16_MAX).any()):
            raise OverflowError("TurboQuant norm-corrected scale exceeds FP16")
        scale = scale.to(torch.float16).float()
        scale = torch.where(norm == 0, torch.zeros_like(scale), scale)
        return ((codes * scale) @ self.hadamard) * self.signs


class RoundTripFullAttentionCache:
    """Append-owned cache holding the decoded values of an evaluation codec pair."""

    def __init__(self, capacity: int, heads: int, name: str, device: torch.device) -> None:
        if name not in KEY_VALUE_CODECS:
            raise ValueError(f"unknown evaluation cache codec {name!r}")
        self.capacity = capacity
        self.heads = heads
        self.length = 0
        self.codecs = tuple(self._codec(item, device) for item in KEY_VALUE_CODECS[name])
        # Head-major storage: each KV head's prefix is a contiguous attention operand.
        self.keys = torch.empty((heads, capacity, HEAD_DIM), device=device, dtype=torch.float32)
        self.values = torch.empty_like(self.keys)

    @staticmethod
    def _codec(name: str, device: torch.device):
        if name == "nvfp4":
            return lambda x: round_trip_nvfp4(x, four_over_six=False)
        if name == "nvfp4-46":
            return lambda x: round_trip_nvfp4(x, four_over_six=True)
        return TurboQuantRoundTrip(name, device)

    def append(self, begin: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        if begin != self.length or keys.shape != values.shape:
            raise ValueError("KV cache append must be contiguous with matching K/V shapes")
        if keys.ndim != 3 or tuple(keys.shape[1:]) != (self.heads, HEAD_DIM):
            raise ValueError("KV cache append has the wrong head geometry")
        end = begin + keys.shape[0]
        if end > self.capacity:
            raise ValueError("KV cache append exceeds capacity")
        if not bool(torch.isfinite(keys).all()) or not bool(torch.isfinite(values).all()):
            raise ValueError("KV cache append requires finite K/V")
        key_codec, value_codec = self.codecs
        self.keys[:, begin:end] = key_codec(keys).transpose(0, 1)
        self.values[:, begin:end] = value_codec(values).transpose(0, 1)
        self.length = end

    def read(self, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= end <= self.length:
            raise ValueError("KV cache read exceeds the represented prefix")
        return self.keys[:, :end].transpose(0, 1), self.values[:, :end].transpose(0, 1)


def head_token_bytes(name: str) -> int:
    key, value = KEY_VALUE_CODECS[name]
    return CODEC_BYTES[key] + CODEC_BYTES[value]


__all__ = [
    "KEY_VALUE_CODECS",
    "RoundTripFullAttentionCache",
    "TurboQuantRoundTrip",
    "hadamard_matrix",
    "head_token_bytes",
    "rotation_signs",
    "round_trip_nvfp4",
]
