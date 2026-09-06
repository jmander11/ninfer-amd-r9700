"""Candidate matrix recipes for BF16-source R9700 DFlash2 conversion.

This module deliberately owns quantization only.  It does not assign artifact identities,
select a production recipe, or permit a candidate to bypass the base-model selection gates.
Every recipe changes exactly the 32 DFlash matrix objects; selector codebooks and all other
DFlash state retain their represented BF16 bytes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from tools.artifact.layouts import encoded_size
from tools.convert.qwen3.common.inventory import BF16, Q4, W8, TensorSpec, tensor_spec


CANONICAL_Q4G64 = "canonical-q4g64"
SOURCE_MSE_Q4G64 = "source-mse-q4g64"
SOURCE_MSE_W8G32 = "source-mse-w8g32"


@dataclass(frozen=True, slots=True)
class MatrixRecipe:
    key: str
    recipe_id: str
    matrix_format: str
    scale_objective: str


RECIPES = (
    MatrixRecipe(
        CANONICAL_Q4G64,
        "r9700-dflash2-all-q4g64-n16k16-bf16-codebook-eval-v1",
        Q4,
        "canonical-fp16-absmax-div7",
    ),
    MatrixRecipe(
        SOURCE_MSE_Q4G64,
        "r9700-dflash2-source-mse-q4g64-n16k16-bf16-codebook-eval-v1",
        Q4,
        "source-bf16-eight-step-least-squares-fp16-scale",
    ),
    MatrixRecipe(
        SOURCE_MSE_W8G32,
        "r9700-dflash2-source-mse-w8g32-bf16-codebook-eval-v1",
        W8,
        "source-bf16-eight-step-least-squares-fp16-scale",
    ),
)
_BY_KEY = {recipe.key: recipe for recipe in RECIPES}


def get_recipe(key: str) -> MatrixRecipe:
    try:
        return _BY_KEY[key]
    except KeyError as error:
        raise ValueError(
            f"unknown DFlash2 matrix recipe {key!r}; expected one of {tuple(_BY_KEY)}"
        ) from error


def _resolve(recipe: MatrixRecipe | str) -> MatrixRecipe:
    selected = get_recipe(recipe) if isinstance(recipe, str) else recipe
    if get_recipe(selected.key) != selected:
        raise ValueError("DFlash2 matrix recipe is not an owned recipe")
    return selected


def tensor_specs(
    canonical_specs: Sequence[TensorSpec], recipe: MatrixRecipe | str
) -> tuple[TensorSpec, ...]:
    """Apply one matrix recipe without changing names, shapes, order, or BF16 objects."""

    selected = _resolve(recipe)
    matrix_specs = tuple(spec for spec in canonical_specs if spec.format != BF16)
    if len(matrix_specs) != 32 or any(len(spec.shape) != 2 for spec in matrix_specs):
        raise ValueError("canonical DFlash2 topology must contain exactly 32 matrices")
    if len(canonical_specs) - len(matrix_specs) != 34:
        raise ValueError("canonical DFlash2 topology must contain exactly 34 BF16 objects")
    result = tuple(
        spec if spec.format == BF16 else tensor_spec(spec.name, spec.shape, selected.matrix_format)
        for spec in canonical_specs
    )
    validate_specs(result, selected)
    return result


def validate_specs(specs: Sequence[TensorSpec], recipe: MatrixRecipe | str) -> None:
    selected = _resolve(recipe)
    counts = Counter(spec.format for spec in specs)
    if counts != Counter({BF16: 34, selected.matrix_format: 32}):
        raise ValueError(f"DFlash2 recipe format counts differ: {dict(counts)}")
    codebooks = tuple(spec for spec in specs if spec.name.endswith("codebook"))
    if len(codebooks) != 2 or any(
        spec.format != BF16 or spec.layout != "contiguous-le-v1" for spec in codebooks
    ):
        raise ValueError("both DFlash2 selector codebooks must remain contiguous BF16")
    if any(spec.format != BF16 for spec in specs if spec.name.endswith("base_kernel")):
        raise ValueError("DFlash2 private convolution state must remain BF16")


def summary(
    canonical_specs: Sequence[TensorSpec], recipe: MatrixRecipe | str
) -> dict[str, object]:
    selected = _resolve(recipe)
    specs = tensor_specs(canonical_specs, selected)
    counts = dict(sorted(Counter(spec.format for spec in specs).items()))
    by_format = {
        numeric_format: sum(
            encoded_size(spec.layout, spec.format, spec.shape)
            for spec in specs
            if spec.format == numeric_format
        )
        for numeric_format in counts
    }
    return {
        "key": selected.key,
        "recipe_id": selected.recipe_id,
        "matrix_format": selected.matrix_format,
        "scale_objective": selected.scale_objective,
        "represented_source": "BF16",
        "matrix_objects": 32,
        "preserved_bf16_objects": 34,
        "selector_codebook_format": BF16,
        "format_counts": counts,
        "format_encoded_bytes": by_format,
        "tensor_encoded_bytes": sum(by_format.values()),
        "selection_status": (
            "registered-evaluation-control-with-bound-artifact-identities"
            if selected.key == CANONICAL_Q4G64
            else "candidate-only-no-artifact-identity"
        ),
        "materialization": (
            "existing-canonical-converter-route"
            if selected.key == CANONICAL_Q4G64
            else "disabled-until-base-selection"
        ),
        "calibration_inputs": "none",
        "preserved_bf16": "both-selector-codebooks-and-all-private-nonmatrix-state",
        "runtime_repack": False,
    }


def encode_matrix_payload(tensor, spec: TensorSpec, recipe: MatrixRecipe | str, device) -> bytes:
    """Encode one matrix directly from represented BF16 source values."""

    selected = _resolve(recipe)
    if spec.format == BF16 or len(spec.shape) != 2:
        raise ValueError(f"{spec.name}: DFlash2 matrix encoder requires a matrix object")
    if spec.format != selected.matrix_format:
        raise ValueError(
            f"{spec.name}: {selected.key} requires {selected.matrix_format}, got {spec.format}"
        )
    if str(getattr(tensor, "dtype", None)) != "torch.bfloat16":
        raise TypeError(f"{spec.name}: DFlash2 recipe source must be represented BF16")
    if tuple(getattr(tensor, "shape", ())) != spec.shape:
        raise ValueError(
            f"{spec.name}: DFlash2 source shape {tuple(getattr(tensor, 'shape', ()))} "
            f"differs from {spec.shape}"
        )
    # Torch is intentionally duck-typed here so inventory and recipe inspection stay
    # dependency-light. The production encoders perform the exact dtype/shape validation.
    if selected.key == CANONICAL_Q4G64:
        from tools.convert.qwen3.common import conversion as family_conversion

        return family_conversion.encode_tensor_payload(tensor, spec, device)
    from . import mse_quantize

    if selected.key == SOURCE_MSE_Q4G64:
        return mse_quantize.quantize_and_encode_q4g64_mse(tensor, device=device)
    if selected.key == SOURCE_MSE_W8G32:
        return mse_quantize.quantize_and_encode_mse(tensor, device=device)
    raise AssertionError("unreachable DFlash2 recipe dispatch")


def encode_matrix_reference(
    values: Sequence[float], rows: int, columns: int, recipe: MatrixRecipe | str
) -> bytes:
    """Dependency-light independent scalar oracle for synthetic recipe tests."""

    selected = _resolve(recipe)
    from . import codec

    if selected.key == CANONICAL_Q4G64:
        return codec.encode_q4g64_n16k16_reference(values, rows, columns)
    if selected.key == SOURCE_MSE_Q4G64:
        return codec.encode_q4g64_n16k16_reference(
            values, rows, columns, refined=True
        )
    if selected.key == SOURCE_MSE_W8G32:
        return codec.encode_w8g32_mse_reference(values, rows, columns)
    raise AssertionError("unreachable DFlash2 recipe dispatch")


__all__ = [
    "CANONICAL_Q4G64",
    "MatrixRecipe",
    "RECIPES",
    "SOURCE_MSE_Q4G64",
    "SOURCE_MSE_W8G32",
    "encode_matrix_payload",
    "encode_matrix_reference",
    "get_recipe",
    "summary",
    "tensor_specs",
    "validate_specs",
]
