"""Bounded, opt-in stage hashes for localizing BF16 scorer nondeterminism."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import struct
import sys
import uuid
from pathlib import Path

from .gdn_determinism_probe import distribution_record, file_sha256, python_tree_sha256
from .protocol import (
    ATTENTION_PV_EXECUTION,
    DETERMINISTIC_EXECUTION_PROFILE,
    EXECUTION_ENVIRONMENT_KEYS,
    GDN_RECURRENCE_EXECUTION,
    resolved_matmul_reduction,
    resolved_preferred_blas_library,
    resolved_triton_codegen,
)


TYPE = "ninfer_qwen3_8_bf16_stage_trace"
VERSION = 8
TRACE_ROWS = 64
LOGIT_ROWS = 64
ATTENTION_DETAIL_LAYER = 3
ATTENTION_DETAIL_KV_HEADS = 4
ATTENTION_DETAIL_Q_PER_KV = 6
ATTENTION_DETAIL_HEAD_DIM = 256
ATTENTION_DETAIL_STAGE_SHAPES = (
    ("normalized-input", (5120,)),
    ("q-gate-projection", (24, 512)),
    ("k-projection", (4, 256)),
    ("v-projection", (4, 256)),
    ("rope-q", (24, 256)),
    ("rope-k", (4, 256)),
    ("attended-output", (24, 256)),
    ("gated-flatten", (6144,)),
    ("output-projection-update", (5120,)),
    ("post-residual", (5120,)),
)
_ATTENTION_DETAIL_STAGES = dict(ATTENTION_DETAIL_STAGE_SHAPES)
ATTENTION_KERNEL_DETAIL_STAGES = (
    "masked-qk-fp32",
    "softmax-fp32",
    "pv-fp32",
)
TRACE_ENVIRONMENT_KEYS = EXECUTION_ENVIRONMENT_KEYS


class StageTrace:
    def __init__(self, torch, path: Path, *, prompt_tokens: int, skip_tokens: int,
                 prefill_chunk: int, device_index: int, provenance):
        self.torch = torch
        self.path = path
        self.prompt_tokens = prompt_tokens
        self.skip = skip_tokens
        self.chunk = prefill_chunk
        self.device_index = device_index
        self.source = provenance
        self.items = []

    def capture_hidden(self, name, value):
        rows = value.shape[0]
        width = min(TRACE_ROWS, rows)
        self.items += [
            (name + ".first", 0, width, value[:width].clone()),
            (name + ".last", rows - width, rows, value[rows - width:].clone()),
        ]

    def capture_attention_detail(self, layer, stage, value, begin, end):
        if layer != ATTENTION_DETAIL_LAYER:
            return
        expected_tail = _ATTENTION_DETAIL_STAGES.get(stage)
        if expected_tail is None or tuple(value.shape[1:]) != expected_tail:
            raise RuntimeError(f"invalid layer-{layer:02d} attention trace stage {stage}")
        name = f"layer-{layer:02d}.attention-detail.{stage}.last"
        self._capture_tail(name, value, begin, end)

    def capture_attention_kernel_detail(self, stage, kv_head, value, begin, end):
        if stage not in ATTENTION_KERNEL_DETAIL_STAGES:
            raise RuntimeError(f"invalid layer-03 attention kernel trace stage {stage}")
        if not 0 <= kv_head < ATTENTION_DETAIL_KV_HEADS:
            raise RuntimeError("invalid layer-03 attention trace KV head")
        hidden_rows = self.prompt_tokens - 1
        # Earlier prefill spans legitimately have a shorter visible K/V prefix. They cannot
        # contribute to the global final-row trace, so reject malformed shapes only after the
        # span intersects that retained tail.
        tail_begin = hidden_rows - min(TRACE_ROWS, hidden_rows)
        if end <= tail_begin or begin >= hidden_rows:
            return
        expected_tail = (
            (ATTENTION_DETAIL_Q_PER_KV, hidden_rows)
            if stage != "pv-fp32"
            else (ATTENTION_DETAIL_Q_PER_KV, ATTENTION_DETAIL_HEAD_DIM)
        )
        if tuple(value.shape[1:]) != expected_tail:
            raise RuntimeError(f"invalid layer-03 attention kernel trace shape for {stage}")
        name = f"layer-03.attention-detail.kv-{kv_head:02d}.{stage}.last"
        self._capture_tail(name, value, begin, end)

    def _capture_tail(self, name, value, begin, end):
        hidden_rows = self.prompt_tokens - 1
        width = min(TRACE_ROWS, hidden_rows)
        tail_begin = hidden_rows - width
        capture_begin = max(begin, tail_begin)
        capture_end = min(end, hidden_rows)
        if capture_begin >= capture_end:
            return
        holder = getattr(self, "_attention_detail_holders", None)
        if holder is None:
            holder = self._attention_detail_holders = {}
        parts = holder.get(name)
        if parts is None:
            parts = holder[name] = []
            self.items.append((name, tail_begin, hidden_rows, parts))
        parts.append(value[capture_begin - begin:capture_end - begin].clone())

    def capture_logits(self, begin, end, value):
        if begin == self.skip or end == self.prompt_tokens - 1:
            name = "logits.first" if begin == self.skip else "logits.last"
            self.items.append((name, begin, end, value.clone()))

    def write(self, vectors):
        torch = self.torch
        rows = []
        for name, begin, end, value in self.items:
            if isinstance(value, list):
                value = torch.cat(value, dim=0)
            host = value.detach().contiguous().cpu()
            raw = host.view(torch.uint8).numpy().tobytes()
            rows.append({
                "name": name,
                "begin": begin,
                "end": end,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "finite": bool(torch.isfinite(host).all()),
                "nan": bool(torch.isnan(host).any()),
                "positive_infinity": bool(torch.isposinf(host).any()),
                "negative_infinity": bool(torch.isneginf(host).any()),
            })

        root = Path(__file__).resolve().parent
        executable = Path(sys.executable).resolve()
        properties = torch.cuda.get_device_properties(self.device_index)
        fla = importlib.import_module("fla")
        triton = importlib.import_module("triton")
        fla_root = Path(fla.__file__).resolve().parent
        payload = {
            "artifact_type": TYPE,
            "schema_version": VERSION,
            "quality_evidence": False,
            "prompt_tokens": self.prompt_tokens,
            "skip_tokens": self.skip,
            "prefill_chunk": self.chunk,
            "trace_rows": TRACE_ROWS,
            "logit_rows": LOGIT_ROWS,
            "run": {"id": str(uuid.uuid4()), "process_id": os.getpid()},
            "source": self.source,
            "checkpoints": rows,
            "scores": {
                "nll_sha256": hashlib.sha256(
                    struct.pack("<" + "f" * len(vectors.nlls), *vectors.nlls)
                ).hexdigest(),
                "argmax_sha256": hashlib.sha256(
                    struct.pack("<" + "i" * len(vectors.argmax), *vectors.argmax)
                ).hexdigest(),
            },
            "provenance": {
                "execution_profile": DETERMINISTIC_EXECUTION_PROFILE,
                "attention_pv": dict(ATTENTION_PV_EXECUTION),
                "gdn_recurrence": dict(GDN_RECURRENCE_EXECUTION),
                "tunable_op": {"enabled": torch.cuda.tunable.is_enabled()},
                "triton_codegen": resolved_triton_codegen(triton),
                "python": sys.version,
                "python_executable": str(executable),
                "python_executable_sha256": file_sha256(executable),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_git": torch.version.git_version,
                "hip": torch.version.hip,
                "device_index": self.device_index,
                "device_name": properties.name,
                "device_arch": getattr(properties, "gcnArchName", None),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "preferred_blas_library": resolved_preferred_blas_library(torch),
                "deterministic_algorithms": {
                    "enabled": torch.are_deterministic_algorithms_enabled(),
                    "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                },
                "matmul_reduction": resolved_matmul_reduction(torch),
                "environment": {
                    key: os.environ.get(key)
                    for key in TRACE_ENVIRONMENT_KEYS
                },
                "scorer_python_tree_sha256": python_tree_sha256(root),
                "fla_python_tree_sha256": python_tree_sha256(fla_root),
                "distributions": {
                    key: distribution_record(key)
                    for key in ("torch", "triton", "flash-linear-attention", "safetensors")
                },
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + f".tmp-{os.getpid()}")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
